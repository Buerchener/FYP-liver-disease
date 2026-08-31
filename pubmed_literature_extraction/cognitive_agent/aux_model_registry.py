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
from contextlib import contextmanager
from contextvars import ContextVar
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
    # Retries are intentionally provider-call scoped.  They never alter an
    # adjudication result, only recover transient transport/provider failures.
    max_retries: int = 4
    retry_base_delay_s: float = 2.0
    retry_max_delay_s: float = 45.0

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
    provider_cache_read_tokens: int = 0
    provider_cache_write_tokens: int = 0
    provider_cache_miss_tokens: int = 0
    provider_cache_hit_rate: float = 0.0
    uncached_input_tokens: int = 0
    estimated_cached_cost: float = 0.0
    estimated_uncached_cost: float = 0.0
    local_result_hit: bool = False
    provider_prompt_hit: bool = False
    singleflight_shared: bool = False
    attempts: int = 0
    invalid_json_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class AuxModelRegistry:
    VALID_ROLES = frozenset({"primary", "critic", "judge", "recovery"})

    def __init__(
        self, specs: list[AuxModelSpec], *,
        generate: dict[str, Callable[[str], dict | str]] | None = None,
    ):
        self.specs = {spec.role: spec for spec in specs}
        if set(self.specs) - self.VALID_ROLES:
            raise ValueError("unsupported auxiliary model role")
        self.generate = generate or {}
        # Context-local interception is safe for concurrent article workers.
        # It lets the article controller cache/audit bounded judge/recovery
        # calls without changing model credentials or global provider state.
        self._call_interceptor: ContextVar[Callable[..., StructuredModelResult] | None] = (
            ContextVar("aux_model_call_interceptor", default=None)
        )
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
        critic_model: str = "qwen3.6-flash", timeout_s: float = 90.0,
        judge_model: str = "deepseek-v4-flash",
        judge_api_base: str = "", judge_api_key: str = "",
        recovery_model: str = "deepseek-v4-flash",
        recovery_api_base: str = "", recovery_api_key: str = "",
    ) -> "AuxModelRegistry":
        max_retries = max(0, int(os.environ.get("AUX_MODEL_MAX_RETRIES", "4")))
        retry_base_delay_s = max(0.0, float(os.environ.get("AUX_MODEL_RETRY_BASE_DELAY_S", "2")))
        retry_max_delay_s = max(
            retry_base_delay_s,
            float(os.environ.get("AUX_MODEL_RETRY_MAX_DELAY_S", "45")),
        )

        def retry_options() -> dict[str, float | int]:
            return {
                "max_retries": max_retries,
                "retry_base_delay_s": retry_base_delay_s,
                "retry_max_delay_s": retry_max_delay_s,
            }

        def endpoint_for(
            model_id: str, *, explicit_base: str = "", explicit_key: str = "",
        ) -> tuple[str, str]:
            """Select provider credentials from the model family, not role name."""
            if explicit_base or explicit_key:
                return (
                    explicit_base or os.environ.get("DEEPSEEK_API_BASE", "")
                    or "https://api.deepseek.com",
                    explicit_key or os.environ.get("DEEPSEEK_API_KEY", ""),
                )
            if "qwen" in str(model_id or "").casefold():
                return (
                    os.environ.get("ALIYUN_MAAS_API_BASE", "")
                    or os.environ.get("DASHSCOPE_API_BASE", ""),
                    os.environ.get("ALIYUN_MAAS_API_KEY", "")
                    or os.environ.get("DASHSCOPE_API_KEY", ""),
                )
            return (
                os.environ.get("DEEPSEEK_API_BASE", "") or "https://api.deepseek.com",
                os.environ.get("DEEPSEEK_API_KEY", ""),
            )

        primary = AuxModelSpec(
            role="primary", provider="openai", model_id=primary_model,
            api_base=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""), timeout_s=timeout_s,
            **retry_options(),
        )
        critic = AuxModelSpec(
            role="critic", provider="openai", model_id=critic_model,
            api_base=(os.environ.get("ALIYUN_MAAS_API_BASE", "")
                      or os.environ.get("DASHSCOPE_API_BASE", "")),
            api_key=(os.environ.get("ALIYUN_MAAS_API_KEY", "")
                     or os.environ.get("DASHSCOPE_API_KEY", "")),
            timeout_s=timeout_s, **retry_options(),
        )
        judge_base, judge_key = endpoint_for(
            judge_model, explicit_base=judge_api_base, explicit_key=judge_api_key,
        )
        judge = AuxModelSpec(
            role="judge", provider="openai", model_id=judge_model,
            api_base=judge_base, api_key=judge_key,
            timeout_s=timeout_s, **retry_options(),
        )
        recovery_base, recovery_key = endpoint_for(
            recovery_model, explicit_base=recovery_api_base, explicit_key=recovery_api_key,
        )
        recovery = AuxModelSpec(
            role="recovery", provider="openai", model_id=recovery_model,
            api_base=recovery_base, api_key=recovery_key,
            timeout_s=timeout_s, **retry_options(),
        )
        return cls([primary, critic, judge, recovery])

    def configured(self, role: str) -> bool:
        spec = self.specs.get(role)
        return bool(spec and (spec.configured or role in self.generate))

    def public_config(self) -> dict[str, Any]:
        return {role: spec.public_dict() for role, spec in self.specs.items()}

    @contextmanager
    def intercept_calls(self, interceptor: Callable[..., StructuredModelResult]):
        token = self._call_interceptor.set(interceptor)
        try:
            yield
        finally:
            self._call_interceptor.reset(token)

    def call_json(
        self, role: str, *, system_prompt: str, user_prompt: str,
        schema_hint: dict[str, Any] | None = None,
    ) -> StructuredModelResult:
        interceptor = self._call_interceptor.get()
        if interceptor is not None:
            return interceptor(
                role=role,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_hint=schema_hint,
                invoke=self._call_json_direct,
            )
        return self._call_json_direct(
            role, system_prompt=system_prompt, user_prompt=user_prompt,
            schema_hint=schema_hint,
        )

    def _call_json_direct(
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
        max_attempts = 1 if role in self.generate else 1 + max(0, int(spec.max_retries))
        invalid_json_attempts = 0
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            try:
                if role in self.generate:
                    raw = self.generate[role](user_prompt)
                    payload = raw if isinstance(raw, dict) else json.loads(raw)
                    prompt_tokens = output_tokens = 0
                    usage_details = self._parse_usage(None)
                else:
                    payload, usage_details = self._openai_call(
                        spec, system_prompt=system_prompt,
                        user_prompt=user_prompt, schema_hint=schema_hint,
                    )
                    prompt_tokens = int(usage_details.get("prompt_tokens", 0))
                    output_tokens = int(usage_details.get("output_tokens", 0))
                if not isinstance(payload, dict):
                    raise ValueError("structured response must be a JSON object")
                latency = time.perf_counter() - started
                usage["attempted"] = int(usage.get("attempted", 0)) + attempt
                usage["successful"] = int(usage.get("successful", 0)) + 1
                usage["latency_s"] = float(usage.get("latency_s", 0.0)) + latency
                usage["prompt_tokens"] = int(usage.get("prompt_tokens", 0)) + prompt_tokens
                usage["output_tokens"] = int(usage.get("output_tokens", 0)) + output_tokens
                for key in (
                    "provider_cache_read_tokens", "provider_cache_write_tokens",
                    "provider_cache_miss_tokens", "uncached_input_tokens",
                ):
                    usage[key] = int(usage.get(key, 0)) + int(usage_details.get(key, 0))
                return StructuredModelResult(
                    role, spec.model_id, "OK", payload=payload,
                    latency_s=round(latency, 4), prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    provider_cache_read_tokens=int(usage_details.get("provider_cache_read_tokens", 0)),
                    provider_cache_write_tokens=int(usage_details.get("provider_cache_write_tokens", 0)),
                    provider_cache_miss_tokens=int(usage_details.get("provider_cache_miss_tokens", 0)),
                    provider_cache_hit_rate=float(usage_details.get("provider_cache_hit_rate", 0.0)),
                    uncached_input_tokens=int(usage_details.get("uncached_input_tokens", 0)),
                    estimated_cached_cost=float(usage_details.get("estimated_cached_cost", 0.0)),
                    estimated_uncached_cost=float(usage_details.get("estimated_uncached_cost", 0.0)),
                    provider_prompt_hit=bool(usage_details.get("provider_prompt_hit", False)),
                    attempts=attempt,
                    invalid_json_attempts=invalid_json_attempts,
                )
            except Exception as exc:
                last_error = self._redact_error(spec, exc)
                invalid_json_attempts += int(self._is_invalid_json_error(last_error))
                if attempt >= max_attempts or not self._is_retryable_error(exc, last_error):
                    break
                time.sleep(self._retry_delay_s(spec, attempt, exc))

        latency = time.perf_counter() - started
        usage["attempted"] = int(usage.get("attempted", 0)) + attempt
        usage["failed"] = int(usage.get("failed", 0)) + 1
        usage["invalid_json"] = int(usage.get("invalid_json", 0)) + invalid_json_attempts
        usage["latency_s"] = float(usage.get("latency_s", 0.0)) + latency
        return StructuredModelResult(
            role, spec.model_id, "FALLBACK", error=last_error[:500],
            latency_s=round(latency, 4), attempts=attempt,
            invalid_json_attempts=invalid_json_attempts,
        )

    @staticmethod
    def _redact_error(spec: AuxModelSpec, exc: Exception) -> str:
        message = str(exc)
        return message.replace(spec.api_key, "[REDACTED]") if spec.api_key else message

    @staticmethod
    def _is_invalid_json_error(message: str) -> bool:
        text = message.casefold()
        return "json" in text or "decode" in text

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        value = getattr(exc, "status_code", None)
        if value is None:
            response = getattr(exc, "response", None)
            value = getattr(response, "status_code", None)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _is_retryable_error(cls, exc: Exception, message: str) -> bool:
        status = cls._status_code(exc)
        if status in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        text = message.casefold()
        return any(token in text for token in (
            "timeout", "timed out", "connection", "temporar", "rate limit",
            "too many requests", "bad gateway", "service unavailable",
            "expecting value", "json decode", "empty structured response",
        ))

    @staticmethod
    def _decode_json_object(content: str) -> dict[str, Any]:
        """Decode common OpenAI-compatible JSON wrappers without guessing fields."""
        text = str(content or "").strip()
        if not text:
            raise ValueError("empty structured response")
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().casefold() in {"```", "```json"}:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"json decode error: {exc}") from exc
            try:
                payload = json.loads(text[start:end + 1])
            except json.JSONDecodeError as nested:
                raise ValueError(f"json decode error: {nested}") from nested
        if not isinstance(payload, dict):
            raise ValueError("structured response must be a JSON object")
        return payload

    @classmethod
    def _retry_delay_s(cls, spec: AuxModelSpec, attempt: int, exc: Exception) -> float:
        retry_after = None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
        try:
            if retry_after is not None:
                return min(float(spec.retry_max_delay_s), max(0.0, float(retry_after)))
        except (TypeError, ValueError):
            pass
        return min(
            float(spec.retry_max_delay_s),
            float(spec.retry_base_delay_s) * (2 ** max(0, attempt - 1)),
        )

    @staticmethod
    def _openai_call(
        spec: AuxModelSpec, *, system_prompt: str, user_prompt: str,
        schema_hint: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from openai import OpenAI

        client = OpenAI(api_key=spec.api_key, base_url=spec.api_base, timeout=spec.timeout_s)
        system_message: dict[str, Any] = {"role": "system", "content": system_prompt}
        if (
            "aliyuncs.com" in spec.api_base.casefold()
            and os.environ.get("QWEN_EXPLICIT_CACHE_CONTROL", "").casefold()
            in {"1", "true", "yes", "on"}
        ):
            system_message["cache_control"] = {"type": "ephemeral"}
        request: dict[str, Any] = {
            "model": spec.model_id,
            "messages": [
                system_message,
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
        payload = AuxModelRegistry._decode_json_object(content)
        api_usage = getattr(response, "usage", None)
        return payload, AuxModelRegistry._parse_usage(api_usage)

    @staticmethod
    def _usage_dict(api_usage: Any) -> dict[str, Any]:
        if api_usage is None:
            return {}
        if isinstance(api_usage, dict):
            return api_usage
        for method in ("model_dump", "dict"):
            if hasattr(api_usage, method):
                try:
                    value = getattr(api_usage, method)()
                    if isinstance(value, dict):
                        return value
                except Exception:
                    pass
        return {
            key: getattr(api_usage, key)
            for key in dir(api_usage)
            if not key.startswith("_") and not callable(getattr(api_usage, key))
        }

    @staticmethod
    def _parse_usage(api_usage: Any) -> dict[str, Any]:
        raw = AuxModelRegistry._usage_dict(api_usage)
        details = raw.get("prompt_tokens_details") or raw.get("prompt_token_details") or {}
        if not isinstance(details, dict):
            details = AuxModelRegistry._usage_dict(details)
        prompt_tokens = int(raw.get("prompt_tokens", 0) or raw.get("input_tokens", 0) or 0)
        output_tokens = int(raw.get("completion_tokens", 0) or raw.get("output_tokens", 0) or 0)
        deepseek_hit = int(raw.get("prompt_cache_hit_tokens", 0) or 0)
        deepseek_miss = int(raw.get("prompt_cache_miss_tokens", 0) or 0)
        qwen_cached = int(details.get("cached_tokens", 0) or raw.get("cached_tokens", 0) or 0)
        cache_read = deepseek_hit or qwen_cached
        cache_write = int(
            raw.get("cache_creation_input_tokens", 0)
            or details.get("cache_creation_input_tokens", 0)
            or 0
        )
        cache_miss = deepseek_miss or max(0, prompt_tokens - cache_read)
        denominator = cache_read + cache_miss
        hit_rate = round(cache_read / denominator, 6) if denominator else 0.0
        return {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "provider_cache_read_tokens": cache_read,
            "provider_cache_write_tokens": cache_write,
            "provider_cache_miss_tokens": cache_miss,
            "provider_cache_hit_rate": hit_rate,
            "uncached_input_tokens": cache_miss,
            "estimated_cached_cost": 0.0,
            "estimated_uncached_cost": 0.0,
            "provider_prompt_hit": cache_read > 0,
        }

    def audit(self) -> dict[str, Any]:
        return {
            "models": self.public_config(),
            "usage": {
                role: {**values, "latency_s": round(float(values.get("latency_s", 0.0)), 4)}
                for role, values in self.usage.items()
            },
        }
