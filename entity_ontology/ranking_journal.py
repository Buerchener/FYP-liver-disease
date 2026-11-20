"""Durable complete requests/responses; successful responses are reused on resume."""
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from functools import wraps
import json
from threading import RLock, local

from .model import ProviderError
from .storage import atomic_json, digest, read_json


class BudgetExceeded(RuntimeError):
    pass


class ContractError(ValueError):
    pass


def strict_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON number')
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, TypeError) as error:
        raise ContractError('Invalid JSON response: ' + str(error)) from error


@dataclass(frozen=True, slots=True)
class _Entry:
    key: str | None
    correction_key: str | None
    status: str | None
    finish_reason: str | None


def _comparable_json(value):
    # Correction replay uses Python's numeric equality, including 1 == 1.0.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _comparable_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_comparable_json(item) for item in value]
    return value


def _context_key(request, payload):
    values = [request.get("stage"), request.get("prompt_sha256"), request.get("parameters"), payload]
    return digest(_comparable_json(values))


def _locked(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return function(self, *args, **kwargs)
    return wrapped


class LLMJournal:
    def __init__(self, directory, provider, prompts, stage_settings, maximum_calls):
        self.directory, self.provider = directory, provider
        directory.mkdir(parents=True, exist_ok=True)
        self.prompts, self.settings, self.maximum_calls = prompts, stage_settings, maximum_calls
        self._lock = RLock()
        self._local = local()
        self._inflight = {}
        self.last_ref = None
        self._stamp = None
        self._sequence = 0
        self._entries = {}
        self._exact = defaultdict(set)
        self._corrections = defaultdict(set)

    @property
    def last_ref(self):
        """The last request made by this thread, including cache hits."""
        return getattr(self._local, "last_ref", None)

    @last_ref.setter
    def last_ref(self, value):
        self._local.last_ref = value

    def _index(self, path, record):
        request = record.get("request") or {}
        response = record.get("response") or {}
        exact_key = record.get("request_sha256")
        correction_key = None
        if isinstance(request.get("input"), dict):
            payload = dict(request["input"])
            if payload.pop("output_format_correction", None) is not None:
                correction_key = _context_key(request, payload)
        old = self._entries.get(path)
        if old is not None:
            if old.key is not None:
                self._exact[old.key].discard(path)
            if old.correction_key is not None:
                self._corrections[old.correction_key].discard(path)
        self._entries[path] = _Entry(exact_key, correction_key, record.get("status"),
                                    response.get("finish_reason"))
        if exact_key is not None:
            self._exact[exact_key].add(path)
        if correction_key is not None:
            self._corrections[correction_key].add(path)

    def _sync(self):
        # The log is owned by the current run and is updated via atomic replacement.
        # Notice other atomic writes, but avoid scanning files for every request.
        stamp = self.directory.stat().st_mtime_ns
        if stamp != self._stamp:
            self._entries.clear()
            self._exact.clear()
            self._corrections.clear()
            for path in self.directory.glob("*.json"):
                self._index(path, read_json(path))
            self._sequence = max((int(path.name.split('-', 1)[0]) for path in self._entries
                                  if path.name.split('-', 1)[0].isdigit()), default=0)
            self._stamp = stamp

    def _store(self, path, record):
        atomic_json(path, record)
        self._index(path, record)
        self._stamp = self.directory.stat().st_mtime_ns

    def _paths(self, key, status):
        return [path for path in self._exact.get(key, ()) if self._entries[path].status == status]

    def request_record(self, stage, payload):
        return {'stage': stage, 'prompt': self.prompts[stage],
                'prompt_sha256': digest(self.prompts[stage]),
                'parameters': self.settings[stage], 'input': payload}

    @_locked
    def invalid_reason(self, stage, payload):
        """Return the validation error from the latest rejected exact request."""
        self._sync()
        key = digest(self.request_record(stage, payload))
        path = max(self._paths(key, "response_invalid"), default=None)
        return read_json(path)["validation_error"] if path is not None else None

    @_locked
    def mark_invalid(self, reason):
        """Keep a failed response for review, but allow a fresh request on resume."""
        self._sync()
        path = self.directory / self.last_ref.removeprefix("llm/")
        record = read_json(path)
        record.update(status="response_invalid", validation_error=reason)
        self._store(path, record)

    @_locked
    def revalidate_invalid(self, stage, payload, validate):
        """Try the current contract on complete saved output; never change its content."""
        self._sync()
        request = self.request_record(stage, payload)
        key = digest(request)
        if self._paths(key, 'response_received'):
            return None
        candidates = set(self._exact.get(key, ()))
        if 'output_format_correction' not in payload:
            candidates.update(self._corrections.get(_context_key(request, payload), ()))
        for path in sorted(candidates):
            entry = self._entries[path]
            if entry.status not in ('response_invalid', 'response_received') or entry.finish_reason != 'stop':
                continue
            record = read_json(path)
            saved = record.get('request', {})
            saved_input = dict(saved['input']) if isinstance(saved.get('input'), dict) else {}
            correction = saved_input.pop('output_format_correction', None)
            same_correction_context = (correction is not None and 'output_format_correction' not in payload
                and saved_input == payload and saved.get('stage') == stage
                and saved.get('prompt_sha256') == request['prompt_sha256']
                and saved.get('parameters') == request['parameters'])
            if record.get('request_sha256') != key and not same_correction_context:
                continue
            try:
                result = validate(strict_json(record['response']['content']))
            except ContractError:
                continue
            if record['status'] == 'response_invalid':
                record.setdefault('validation_history', []).append({
                    'previous_status': 'response_invalid',
                    'previous_error': record.pop('validation_error', None),
                    'action': 'revalidated_saved_output_with_current_contract',
                    'raw_response_changed': False,
                })
                record['status'] = 'response_received'
                self._store(path, record)
            self.last_ref = 'llm/' + path.name
            return result, self.last_ref
        return None

    def call(self, stage, payload):
        request = self.request_record(stage, payload)
        key = digest(request)
        owner = False
        waiting = None
        with self._lock:
            self._sync()
            path = max(self._paths(key, 'response_received'), default=None)
            if path is not None:
                cached = read_json(path)
            elif key in self._inflight:
                waiting = self._inflight[key]
            else:
                if len(self._entries) >= self.maximum_calls:
                    raise BudgetExceeded('maximum_llm_calls')
                self._sequence += 1
                path = self.directory / ('%06d-%s.json' % (self._sequence, stage))
                cached = {'request_sha256': key, 'request': request, 'status': 'request_started',
                          'billing_note': 'If interrupted before a durable response, resume may issue another billed request.'}
                self._store(path, cached)
                waiting = self._inflight[key] = Future()
                owner = True
        if owner:
            self.last_ref = 'llm/' + path.name
            try:
                cached['response'] = self.provider.generate(stage, self.prompts[stage], payload, self.settings[stage])
                cached['status'] = 'response_received'
                with self._lock:
                    self._store(path, cached)
                    self._inflight.pop(key)
                    waiting.set_result((path, cached))
            except BaseException as error:
                with self._lock:
                    cached['status'] = 'provider_error'
                    cached['error'] = str(error) if isinstance(error, ProviderError) else type(error).__name__
                    self._store(path, cached)
                    self._inflight.pop(key, None)
                    waiting.set_exception(error)
                raise
        elif waiting is not None:
            # Network calls occur outside the metadata lock. Identical concurrent
            # requests wait for that one durable response instead of being billed twice.
            path, cached = waiting.result()
        self.last_ref = 'llm/' + path.name
        response = cached['response']
        if response.get('finish_reason') != 'stop':
            raise ContractError('Incomplete or non-stop LLM response: ' + str(response.get('finish_reason')))
        return strict_json(response['content'])
