"""Qwen native tool calls over the project's OpenAI-compatible HTTP interface."""

import json
import math
import os
import subprocess
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ModelError(RuntimeError):
    """A safe model/transport diagnostic suitable for a saved run."""


class ModelOutputError(ModelError):
    """The provider answered, but its decision cannot be used by the graph."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _get_key(provider, base_url):
    name = provider.get("api_key_env")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("Configure llm.provider.api_key_env with an environment variable name")
    if key := os.environ.get(name):
        return key
    binding = provider.get("macos_keychain")
    if not binding:
        raise ValueError(f"Set {name} for the configured model endpoint")
    if (not isinstance(binding, dict) or not base_url.startswith("https://")
            or str(binding.get("base_url", "")).rstrip("/") != base_url
            or not all(isinstance(binding.get(k), str) and binding[k] for k in ("service", "account"))):
        raise ValueError("Keychain credential must be bound to the configured HTTPS endpoint")
    try:
        result = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-s", binding["service"],
             "-a", binding["account"], "-w"],
            capture_output=True, check=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        raise ValueError("Could not read the configured macOS Keychain credential") from None
    if not result.stdout.strip():
        raise ValueError("The configured macOS Keychain credential is empty")
    return result.stdout.strip()


def _assistant(envelope, allow_tools):
    choice = envelope["choices"][0]
    if choice.get("finish_reason") not in ("stop", "tool_calls"):
        raise ModelOutputError("Model completion was interrupted or truncated")
    message = choice["message"]
    content = message.get("content")
    if message.get("role") != "assistant" or not (content is None or isinstance(content, str)):
        raise ModelOutputError("Invalid model assistant message")
    assistant = {"role": "assistant", "content": content}
    calls = message.get("tool_calls") or []
    if calls:
        if not allow_tools or not isinstance(calls, list):
            raise ModelOutputError("Model returned unexpected tool calls")
        parsed = []
        ids = set()
        for call in calls:
            function, call_id = call["function"], call["id"]
            if (call["type"] != "function" or not isinstance(call_id, str) or not call_id
                    or call_id in ids or not isinstance(function["name"], str)
                    or not isinstance(function["arguments"], str)):
                raise ModelOutputError("Invalid model tool call")
            ids.add(call_id)
            parsed.append({"type": "function", "id": call_id, "function": {
                "name": function["name"], "arguments": function["arguments"],
            }})
        assistant["tool_calls"] = parsed
    elif not isinstance(content, str) or not content.strip():
        raise ModelOutputError("Model returned no text or tool calls")
    elif not allow_tools and not isinstance(json.loads(content), dict):
        raise ModelOutputError("Final model answer must be a JSON object")
    return assistant


class QwenModel:
    """Read-only model client; one HTTP request per graph decision."""

    def __init__(self, *, base_url, model, api_key, timeout_seconds=60,
                 temperature=0, max_tokens=2048, extra_body=None):
        if not isinstance(base_url, str):
            raise ValueError("Missing model base URL")
        url = urlsplit(base_url)
        local_http = url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
        if ((url.scheme != "https" and not local_http) or not url.hostname or url.username is not None
                or url.password is not None or url.fragment or url.query or any(c.isspace() for c in base_url)):
            raise ValueError("Configure a credential-free HTTPS model endpoint or HTTP loopback endpoint")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Missing model ID")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Missing configured model credential")
        if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 600:
            raise ValueError("Model timeout must be within (0, 600] seconds")
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("Invalid model temperature")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Invalid model token limit")
        extra = {} if extra_body is None else extra_body
        if not isinstance(extra, dict) or extra.keys() - {"enable_thinking"} or any(type(v) is not bool for v in extra.values()):
            raise ValueError("Model extra_body supports the boolean enable_thinking option")
        self.base_url = base_url.rstrip("/")
        self.model_name = model.strip()
        self.timeout_seconds = timeout_seconds
        self._key = api_key.strip()
        self._options = {"temperature": temperature, "max_tokens": max_tokens, **extra}

    @classmethod
    def from_config(cls, path, *, stage=None, timeout_seconds=None):
        config = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        llm = config.get("llm", config)
        provider = llm["provider"]
        if provider.get("kind", "openai_compatible") != "openai_compatible":
            raise ValueError("Select an OpenAI-compatible model provider")
        if {"api_key", "authorization", "password", "headers"} & provider.keys():
            raise ValueError("Supply model credentials through api_key_env or an endpoint-bound Keychain item")
        stages = llm.get("stages", {})
        if stage is None and stages:
            stage = "audit_edges" if "audit_edges" in stages else "validation"
        if stage is not None and stage not in stages:
            raise ValueError("Selected model configuration stage is missing")
        params = stages[stage] if stage else llm.get("parameters", {})
        base_url = provider.get("base_url") or os.environ.get(provider.get("base_url_env", ""), "")
        return cls(base_url=base_url, model=params.get("model", provider.get("model")),
                   api_key=_get_key(provider, base_url.rstrip("/")),
                   timeout_seconds=(provider.get("timeout_seconds", 60) if timeout_seconds is None else timeout_seconds),
                   temperature=params.get("temperature", 0), max_tokens=params.get("max_tokens", 2048),
                   extra_body={**provider.get("extra_body", {}), **params.get("extra_body", {})})

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.model_name, "messages": messages, **self._options}
        if tools:
            body.update(tools=tools, parallel_tool_calls=False)
        else:
            body["response_format"] = {"type": "json_object"}
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with build_opener(_NoRedirect()).open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(2_000_001)
        except HTTPError as exc:
            exc.close()
            raise ModelError(f"Model HTTP {exc.code}") from None
        except (OSError, URLError, HTTPException, ValueError):
            raise ModelError("Model transport failed") from None
        if len(raw) > 2_000_000:
            raise ModelOutputError("Model response exceeded the size limit")
        try:
            text = raw.decode("utf-8")
            if self._key in text:
                raise ModelOutputError("Model response contained credential material")
            return _assistant(json.loads(text), bool(tools))
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            raise ModelOutputError("Invalid model response envelope or final JSON") from None
