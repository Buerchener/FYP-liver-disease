#!/usr/bin/env python3
"""Dual-model structured calling for Agent v3.

DeepSeek is the primary adjudicator/rule inducer and Qwen is the independent
critic.  Credentials are environment-only and never included in serialized
configuration, cache keys or reports.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class AuxModelSpec:
    role: str
    provider: str
    model_id: str
    api_base: str
    api_key: str = field(repr=False, default="")
    timeout_s: float = 90.0
    thinking_enabled: bool = False

    @property
    def configured(self) -> bool:
        return bool(self.model_id and self.api_base and self.api_key)

    def public_dict(self) -> dict[str, Any]:
        return {
            "role": self.role, "provider": self.provider,
            "model_id": self.model_id, "api_base": self.api_base,
            "configured": self.configured, "timeout_s": self.timeout_s,
            "thinking_enabled": self.thinking_enabled,
        }


@dataclass
class StructuredModelResult:
    role: str
    model_id: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 0
    invalid_json_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class AuxModelRegistry:
    VALID_ROLES = frozenset({"primary", "critic"})

    def __init__(
        self, specs: list[AuxModelSpec], *,
        generate: dict[str, Callable[[str], dict | str]] | None = None,
    ):
        self.specs = {spec.role: spec for spec in specs}
        if set(self.specs) - self.VALID_ROLES:
            raise ValueError("unsupported auxiliary model role")
        self.generate = generate or {}
        self.usage: dict[str, dict[str, float | int]] = {
            role: {
                "attempted": 0, "successful": 0, "failed": 0,
                "invalid_json": 0, "latency_s": 0.0,
                "prompt_tokens": 0, "output_tokens": 0,
            } for role in self.specs
        }

    @classmethod
    def from_environment(
        cls, *, primary_model: str = "deepseek-v4-flash",
        critic_model: str = "qwen-flash", timeout_s: float = 90.0,
    ) -> "AuxModelRegistry":
        primary = AuxModelSpec(
            role="primary", provider="openai", model_id=primary_model,
            api_base=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""), timeout_s=timeout_s,
        )
        critic = AuxModelSpec(
            role="critic", provider="openai", model_id=critic_model,
            api_base=(os.environ.get("ALIYUN_MAAS_API_BASE", "")
                      or os.environ.get("DASHSCOPE_API_BASE", "")),
            api_key=(os.environ.get("ALIYUN_MAAS_API_KEY", "")
                     or os.environ.get("DASHSCOPE_API_KEY", "")),
            timeout_s=timeout_s,
        )
        return cls([primary, critic])

    def configured(self, role: str) -> bool:
        spec = self.specs.get(role)
        return bool(spec and (spec.configured or role in self.generate))

    def public_config(self) -> dict[str, Any]:
        return {role: spec.public_dict() for role, spec in self.specs.items()}

    def call_json(
        self, role: str, *, system_prompt: str, user_prompt: str,
        schema_hint: dict[str, Any] | None = None,
    ) -> StructuredModelResult:
        if role not in self.VALID_ROLES or role not in self.specs:
            return StructuredModelResult(role, "", "UNCONFIGURED", error="unknown role")
        spec = self.specs[role]
        if not self.configured(role):
            return StructuredModelResult(role, spec.model_id, "UNCONFIGURED", error="model credentials unavailable")
        started = time.perf_counter()
        usage = self.usage.setdefault(role, {})
        usage["attempted"] = int(usage.get("attempted", 0)) + 1
        try:
            if role in self.generate:
                raw = self.generate[role](user_prompt)
                payload = raw if isinstance(raw, dict) else json.loads(raw)
                prompt_tokens = output_tokens = 0
            else:
                payload, prompt_tokens, output_tokens = self._openai_call(
                    spec, system_prompt=system_prompt,
                    user_prompt=user_prompt, schema_hint=schema_hint,
                )
            if not isinstance(payload, dict):
                raise ValueError("structured response must be a JSON object")
            latency = time.perf_counter() - started
            usage["successful"] = int(usage.get("successful", 0)) + 1
            usage["latency_s"] = float(usage.get("latency_s", 0.0)) + latency
            usage["prompt_tokens"] = int(usage.get("prompt_tokens", 0)) + prompt_tokens
            usage["output_tokens"] = int(usage.get("output_tokens", 0)) + output_tokens
            return StructuredModelResult(
                role, spec.model_id, "OK", payload=payload,
                latency_s=round(latency, 4), prompt_tokens=prompt_tokens,
                output_tokens=output_tokens, attempts=1,
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            message = str(exc)
            if spec.api_key:
                message = message.replace(spec.api_key, "[REDACTED]")
            invalid = int("json" in message.casefold() or "decode" in message.casefold())
            usage["failed"] = int(usage.get("failed", 0)) + 1
            usage["invalid_json"] = int(usage.get("invalid_json", 0)) + invalid
            usage["latency_s"] = float(usage.get("latency_s", 0.0)) + latency
            return StructuredModelResult(
                role, spec.model_id, "FALLBACK", error=message[:500],
                latency_s=round(latency, 4), attempts=1,
                invalid_json_attempts=invalid,
            )

    @staticmethod
    def _openai_call(
        spec: AuxModelSpec, *, system_prompt: str, user_prompt: str,
        schema_hint: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], int, int]:
        from openai import OpenAI

        client = OpenAI(api_key=spec.api_key, base_url=spec.api_base, timeout=spec.timeout_s)
        request: dict[str, Any] = {
            "model": spec.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        if "aliyuncs.com" in spec.api_base.casefold():
            request["extra_body"] = {"enable_thinking": spec.thinking_enabled}
        else:
            request["extra_body"] = {
                "thinking": {"type": "enabled" if spec.thinking_enabled else "disabled"}
            }
        if schema_hint:
            request["messages"][0]["content"] += "\nRequired JSON shape: " + json.dumps(schema_hint, ensure_ascii=False)
        response = client.chat.completions.create(**request)
        content = str(response.choices[0].message.content or "")
        payload = json.loads(content)
        api_usage = getattr(response, "usage", None)
        return (
            payload,
            int(getattr(api_usage, "prompt_tokens", 0) or 0),
            int(getattr(api_usage, "completion_tokens", 0) or 0),
        )

    def audit(self) -> dict[str, Any]:
        return {
            "models": self.public_config(),
            "usage": {
                role: {**values, "latency_s": round(float(values.get("latency_s", 0.0)), 4)}
                for role, values in self.usage.items()
            },
        }
