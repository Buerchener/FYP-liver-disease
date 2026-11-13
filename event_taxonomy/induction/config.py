from copy import deepcopy
from pathlib import Path
import urllib.parse
from .storage import read_json


def validate_config(config):
    if config.get('schema_version') != 'event-taxonomy-induction/1':
        raise ValueError('Unknown induction configuration schema')
    if config['provider']['kind'] not in ('mock', 'openai_compatible'):
        raise ValueError('Unknown provider')
    provider = config['provider']
    if any(key in provider for key in ('api_key', 'authorization', 'headers', 'password')):
        raise ValueError('Credentials must be supplied through an environment variable, never configuration')
    if provider['kind'] == 'openai_compatible':
        url = urllib.parse.urlsplit(provider['base_url'])
        if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError('Invalid credential-free API base URL')
        if not isinstance(provider['api_key_env'], str) or not provider['api_key_env'].isidentifier():
            raise ValueError('Invalid API key environment variable name')
        if type(provider['timeout_seconds']) not in (int, float) or not 0 < provider['timeout_seconds'] <= 600:
            raise ValueError('Invalid provider timeout')
    for key in ('iterations_per_parent', 'candidate_attempts', 'context_visits_per_node', 'llm_calls'):
        value = config['limits'][key]
        if type(value) is not int or value < 1:
            raise ValueError('Invalid limit: ' + key)
    for key in ('max_depth', 'new_nodes'):
        if type(config['limits'][key]) is not int or config['limits'][key] < 0:
            raise ValueError('Invalid limit: ' + key)
    for value in config['thresholds'].values():
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError('Confidence thresholds must be in [0,1]')
    for key in ('evidence_per_concept', 'max_ancestor_paths', 'max_characters'):
        if type(config['context'][key]) is not int or config['context'][key] < 1:
            raise ValueError('Invalid context limit')
    for stage in ('diagnosis', 'induction', 'validation'):
        params = config['stages'][stage]
        if not isinstance(params['model'], str) or not params['model'] or type(params['temperature']) not in (int, float) or not 0 <= params['temperature'] <= 2:
            raise ValueError('Invalid model/temperature setting')
        if type(params['max_tokens']) is not int or params['max_tokens'] < 1:
            raise ValueError('Invalid model token budget')
    if type(config['require_semantic_input_frozen']) is not bool:
        raise ValueError('require_semantic_input_frozen must be boolean')
    if config['clustering']['method'] not in ('llm_guided', 'embedding'):
        raise ValueError('Unknown clustering method')
    return config


def read_config(path):
    path = Path(path).resolve()
    config = validate_config(read_json(path))
    for key in ('dataset', 'output_root'):
        config[key] = str((path.parent / config[key]).resolve())
    for stage, prompt_path in config['prompts'].items():
        config['prompts'][stage] = str((path.parent / prompt_path).resolve())
    if config['provider']['kind'] == 'mock':
        config['provider']['plan_file'] = str((path.parent / config['provider']['plan_file']).resolve())
    if config['clustering'].get('vectors_file'):
        config['clustering']['vectors_file'] = str((path.parent / config['clustering']['vectors_file']).resolve())
    return config
