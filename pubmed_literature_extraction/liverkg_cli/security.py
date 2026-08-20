from __future__ import annotations

import os
from collections.abc import Mapping


SECRET_ENV_KEYS = {
    "gemini_api_key": ("GEMINI_API_KEY", "LLM_API_KEY"),
    "neo4j_password": ("NEO4J_PASSWORD",),
    "deepseek_api_key": ("DEEPSEEK_API_KEY", "SECOND_LLM_API_KEY"),
    "qwen_api_key": ("QWEN_API_KEY", "DASHSCOPE_API_KEY", "ALIYUN_API_KEY", "ALIYUN_MAAS_API_KEY"),
}

KEYRING_SERVICE = "LiverKG"


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def redact_mapping(data: Mapping[str, object]) -> dict[str, object]:
    redacted: dict[str, object] = {}
    for key, value in data.items():
        lower = key.lower()
        if any(part in lower for part in ("key", "secret", "password", "token")):
            redacted[key] = mask_secret(str(value)) if value else ""
        elif isinstance(value, Mapping):
            redacted[key] = redact_mapping(value)
        else:
            redacted[key] = value
    return redacted


def get_secret(name: str) -> str:
    for env_key in SECRET_ENV_KEYS.get(name, ()):
        value = os.environ.get(env_key, "")
        if value:
            return value
    try:
        from .env import load_project_env

        local_env = load_project_env()
        for env_key in SECRET_ENV_KEYS.get(name, ()):
            value = local_env.get(env_key, "")
            if value:
                return value
    except Exception:
        pass
    try:
        import keyring
    except Exception:
        return ""
    try:
        return keyring.get_password(KEYRING_SERVICE, name) or ""
    except Exception:
        return ""


def set_secret(name: str, value: str) -> None:
    try:
        import keyring
    except Exception as exc:
        raise RuntimeError("keyring is not available; export the value as an environment variable instead") from exc
    keyring.set_password(KEYRING_SERVICE, name, value)
