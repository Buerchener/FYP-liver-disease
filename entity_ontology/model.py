"""Read model settings and call an OpenAI-compatible chat endpoint."""
from http.client import HTTPException
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import urlsplit

DEFAULT_CONFIG = Path("entity-ontology.local.json")


class ProviderError(RuntimeError):
    pass


def endpoint(connection):
    return connection.get("base_url") or os.environ.get(connection.get("base_url_env", ""), "")


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
        extra_body = {**self.config.get('extra_body', {}), **parameters.get('extra_body', {})}
        for name, value in extra_body.items():
            if name in body or name in {'api_key', 'authorization', 'headers', 'stream'}:
                raise ProviderError('extra_body cannot override required request fields')
            body[name] = value
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=self.config['timeout_seconds']) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            # Error bodies can echo credentials; record only safe status information.
            raise ProviderError('HTTP status %d; request may have reached provider' % error.code) from None
        except (OSError, HTTPException, ValueError, TypeError):
            raise ProviderError('Transport or response-envelope error; request may have reached provider') from None

        envelope = None
        try:
            raw = raw.decode('utf-8').replace(key, '[REDACTED]')
            envelope = json.loads(raw)
            choice = envelope['choices'][0]
            content = choice['message']['content']
            if not isinstance(content, str):
                raise ValueError('Non-text message')
            model, usage = envelope.get('model', parameters['model']), envelope.get('usage')
            result = {'content': content, 'model': model, 'usage': usage, 'finish_reason': choice.get('finish_reason'),
                      'provider_response': envelope}
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError('Transport or response-envelope error; request may have reached provider') from None
        return result


def read_settings(path, stage):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if "llm" in config:
        config = config["llm"]
        default_stage = "audit_edges"
    else:
        default_stage = "validation"
    connection = dict(config["provider"])
    if connection.get("kind") != "openai_compatible":
        raise ValueError("Configure an OpenAI-compatible provider")
    if any(field in connection for field in ("api_key", "authorization", "headers", "password")):
        raise ValueError("Set credentials through the provider's api_key_env")
    stage = stage or default_stage
    if stage not in config["stages"]:
        raise ValueError(f"No model settings for stage {stage!r}")
    parameters = config["stages"][stage]
    if not isinstance(parameters.get("model"), str) or not parameters["model"].strip():
        raise ValueError("Missing LLM model ID")
    if type(parameters.get("temperature")) not in (int, float) or not 0 <= parameters["temperature"] <= 2:
        raise ValueError("Invalid LLM temperature")
    if type(parameters.get("max_tokens")) is not int or parameters["max_tokens"] < 1:
        raise ValueError("Invalid LLM max_tokens")
    return connection, parameters


def live_provider(connection, parameters):
    key_env = connection["api_key_env"]
    if not isinstance(key_env, str) or not key_env.isidentifier():
        raise ValueError("Invalid LLM credential environment variable name")
    connection["base_url"] = endpoint(connection)
    base = urlsplit(connection["base_url"])
    if (
        base.scheme not in ("http", "https") or not base.hostname
        or base.username or base.password or base.query or base.fragment
        or base.hostname.upper().startswith("YOUR_") or parameters["model"].startswith("YOUR_")
    ):
        raise ValueError("Configure a credential-free LLM endpoint and a real model ID")
    timeout = connection.get("timeout_seconds")
    if type(timeout) not in (int, float) or not 0 < timeout <= 600:
        raise ValueError("Invalid LLM provider timeout")
    if not os.environ.get(key_env):
        binding = connection.get("macos_keychain")
        if binding is None:
            raise ValueError(f"Set {key_env} for the configured LLM provider")
        os.environ[key_env] = _read_keychain(binding, connection["base_url"])
    return OpenAICompatibleProvider(connection)


def _read_keychain(binding, base_url):
    if (
        urlsplit(base_url).scheme != "https"
        or base_url.rstrip("/") != binding["base_url"].rstrip("/")
    ):
        raise ValueError("The macOS Keychain item is bound to a different HTTPS endpoint")
    try:
        result = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-s", binding["service"],
             "-a", binding["account"], "-w"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired):
        raise ValueError("Could not read the configured item from macOS Keychain") from None
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("Could not read the configured item from macOS Keychain")
    return result.stdout.strip()
