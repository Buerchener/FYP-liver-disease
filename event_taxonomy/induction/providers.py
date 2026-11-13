"""OpenAI-compatible HTTP transport and deterministic, explicitly synthetic mock."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from .contracts import ProviderError


class OpenAICompatibleProvider:
    def __init__(self, config):
        self.config = config

    def generate(self, stage, prompt, payload, parameters):
        key = os.environ.get(self.config['api_key_env'])
        if not key:
            raise ProviderError('API key environment variable is not set')
        url = self.config['base_url'].rstrip('/') + '/chat/completions'
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ProviderError('Invalid API base URL')
        body = {'model': parameters['model'], 'temperature': parameters['temperature'],
            'max_tokens': parameters['max_tokens'], 'response_format': {'type': 'json_object'},
            'messages': [{'role': 'system', 'content': prompt},
                         {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, allow_nan=False)}]}
        for name, value in self.config.get('extra_body', {}).items():
            if name in body or name in {'api_key', 'authorization', 'headers', 'stream'}:
                raise ProviderError('extra_body cannot override required request fields')
            body[name] = value
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=self.config['timeout_seconds']) as response:
                raw = response.read().decode('utf-8').replace(key, '[REDACTED]')
            envelope = json.loads(raw)
            choice = envelope['choices'][0]
            content = choice['message']['content']
            if not isinstance(content, str):
                raise ValueError('Non-text message')
            return {'content': content, 'model': envelope.get('model', parameters['model']),
                    'usage': envelope.get('usage'), 'finish_reason': choice.get('finish_reason'),
                    'provider_response': envelope}
        except urllib.error.HTTPError as error:
            # Error bodies can echo credentials; record only safe status information.
            raise ProviderError('HTTP status %d; request may have reached provider' % error.code) from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError):
            raise ProviderError('Transport or response-envelope error; request may have reached provider') from None


class MockProvider:
    """A fixture replay, not a medical reasoning model. Rules have no mutable cursor."""
    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def generate(self, stage, prompt, payload, parameters):
        self.calls.append({'stage': stage, 'cursor': payload['cursor']})
        context = payload.get('context', payload.get('before_context'))
        cursor = payload['cursor']
        for rule in self.plan['rules']:
            if rule['stage'] != stage:
                continue
            when = rule.get('when', {})
            if any(cursor.get(k) != v for k, v in when.items() if k != 'child_labels'):
                continue
            if 'child_labels' in when and set(when['child_labels']) != {n['concept_label'] for n in context['children']}:
                continue
            response = self._expand(rule['response'], payload, context)
            break
        else:
            if stage != 'diagnosis':
                raise ProviderError('Synthetic mock plan has no matching ' + stage + ' rule')
            response = self.plan['default_diagnosis']
        return {'content': json.dumps(response, ensure_ascii=False), 'model': 'synthetic-rule-mock-v1',
                'usage': None, 'finish_reason': 'stop', 'synthetic': True}

    def _expand(self, value, payload, context):
        if isinstance(value, list):
            return [self._expand(v, payload, context) for v in value]
        if isinstance(value, dict):
            if set(value) == {'$children'}:
                result = []
                lookup_context = payload.get('after_context', context) if payload.get('macro_after_context') else context
                for label in value['$children']:
                    matches = [n['concept_id'] for n in lookup_context['children'] if n['concept_label'] == label]
                    if len(matches) != 1:
                        raise ProviderError('Ambiguous or missing synthetic child label: ' + label)
                    result.append(matches[0])
                return result
            if set(value) == {'$group'}:
                return next(g['child_ids'] for g in payload['groups'] if g['group_id'] == value['$group'])
            if set(value) == {'$all_edge_checks'}:
                return [{'parent_id': e['parent_id'], 'child_id': e['child_id'],
                         'strict_subsumption': value['$all_edge_checks'],
                         'reasoning': 'Synthetic fixture assertion; not a real semantic judgment.'}
                        for e in payload['proposed_changes']['added_edges']]
            return {k: self._expand(v, payload, context) for k, v in value.items()}
        return value
