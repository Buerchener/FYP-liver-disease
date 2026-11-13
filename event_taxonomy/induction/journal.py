"""Durable complete requests/responses; successful responses are reused on resume."""
import json
from .contracts import BudgetExceeded, ContractError, ProviderError
from .graph import digest
from .storage import atomic_json, read_json


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


class LLMJournal:
    def __init__(self, directory, provider, prompts, stage_settings, maximum_calls):
        self.directory, self.provider = directory, provider
        directory.mkdir(parents=True, exist_ok=True)
        self.prompts, self.settings, self.maximum_calls = prompts, stage_settings, maximum_calls
        self.last_ref = None

    def call(self, stage, payload):
        request = {'stage': stage, 'prompt': self.prompts[stage],
                   'prompt_sha256': digest(self.prompts[stage]),
                   'parameters': self.settings[stage], 'input': payload}
        key = digest(request)
        records = sorted(self.directory.glob('*.json'))
        cached = None
        for path in records:
            row = read_json(path)
            if row.get('request_sha256') == key and row.get('status') == 'response_received':
                cached = row; self.last_ref = 'llm/' + path.name
        if cached is None:
            if len(records) >= self.maximum_calls:
                raise BudgetExceeded('maximum_llm_calls')
            path = self.directory / ('%06d-%s.json' % (len(records) + 1, stage))
            self.last_ref = 'llm/' + path.name
            cached = {'request_sha256': key, 'request': request, 'status': 'request_started',
                      'billing_note': 'If interrupted before a durable response, resume may issue another billed request.'}
            atomic_json(path, cached)
            try:
                cached['response'] = self.provider.generate(stage, self.prompts[stage], payload, self.settings[stage])
                cached['status'] = 'response_received'
            except ProviderError as error:
                cached['status'] = 'provider_error'; cached['error'] = str(error)
                atomic_json(path, cached)
                raise
            atomic_json(path, cached)
        response = cached['response']
        if response.get('finish_reason') != 'stop':
            raise ContractError('Incomplete or non-stop LLM response: ' + str(response.get('finish_reason')))
        return strict_json(response['content'])
