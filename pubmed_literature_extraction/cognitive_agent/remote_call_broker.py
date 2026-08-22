#!/usr/bin/env python3
"""Article-scoped cache/budget boundary for bounded auxiliary model calls.

The existing second-model/critic wrappers retain their specialised merge
logic.  Their requests are already cache-first and Controller-recorded.  This
broker covers Judge, Recovery and the registry-based evidence-entailment
calls, so each of those request families has one budget/audit boundary.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable

from cognitive_agent.aux_model_registry import StructuredModelResult
from cognitive_agent.central_agent_v2 import ArticleAgentState, CentralAgentV2
from cognitive_agent.extraction_cache import LightweightExtractionCache


class ArticleRemoteCallBroker:
    """Cache-first, budgeted registry-backed model calls for one article."""

    INTERCEPTED_ROLES = frozenset({"primary", "critic", "judge", "recovery"})
    ROLE_TO_TOOL = {
        "primary": "evidence_entailment",
        "critic": "evidence_entailment",
        "judge": "pairwise_judge",
        "recovery": "entity_recovery",
    }

    def __init__(
        self,
        *,
        cache: LightweightExtractionCache,
        controller: CentralAgentV2 | None = None,
        state: ArticleAgentState | None = None,
    ):
        self.cache = cache
        self.controller = controller
        self.state = state
        self._scope = ""
        self._pending: dict[str, list[tuple[Any, bool]]] = defaultdict(list)

    @contextmanager
    def scope(self, tool: str):
        previous = self._scope
        self._scope = str(tool or "")
        try:
            yield
        finally:
            self._scope = previous

    def intercept(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        schema_hint: dict[str, Any] | None,
        invoke: Callable[..., StructuredModelResult],
    ) -> StructuredModelResult:
        if role not in self.INTERCEPTED_ROLES:
            return invoke(
                role, system_prompt=system_prompt, user_prompt=user_prompt,
                schema_hint=schema_hint,
            )
        tool = self._scope or self.ROLE_TO_TOOL[role]
        if self.controller is not None and self.state is not None:
            allowed, reason = self.controller.can_call_remote(self.state, tool)
            if not allowed:
                model_id = ""
                return StructuredModelResult(
                    role=role, model_id=model_id, status="BUDGET_BLOCKED", error=reason,
                )
        else:
            reason = "legacy_cache_first"

        key = CentralAgentV2.tool_cache_key(tool, {
            "tool_version": "article-remote-broker-v1",
            "role": role,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema_hint": schema_hint or {},
        })

        def factory() -> tuple[dict[str, Any], float]:
            result = invoke(
                role, system_prompt=system_prompt, user_prompt=user_prompt,
                schema_hint=schema_hint,
            )
            return result.to_dict(), float(result.latency_s or 0.0)

        payload, cache_status = self.cache.get_or_compute(
            key,
            factory,
            cacheable=lambda value: value.get("status") == "OK",
        )
        fields = StructuredModelResult.__dataclass_fields__
        result = StructuredModelResult(**{
            key: value for key, value in payload.items() if key in fields
        })
        cache_hit = cache_status in {
            "memory_hit", "persistent_hit", "singleflight_shared",
        }
        result.local_result_hit = cache_status in {"memory_hit", "persistent_hit"}
        result.singleflight_shared = cache_status == "singleflight_shared"
        if self.controller is not None and self.state is not None:
            before = self.state.fingerprint()
            trace = self.controller.record_action(
                self.state,
                tool=tool,
                decision="CACHE_HIT" if cache_hit else "CALL",
                reason=reason,
                before=before,
                after=before,
                latency_s=result.latency_s,
                cache_status=cache_status,
                remote=True,
                result_status=result.status,
                prompt_tokens=result.prompt_tokens,
                output_tokens=result.output_tokens,
                attempt_count=max(0, int(result.attempts or 0)),
                retry_count=max(0, int(result.attempts or 0) - 1),
                details={
                    "role": role,
                    "defer_state_change": True,
                    "local_result_hit": result.local_result_hit,
                    "singleflight_shared": result.singleflight_shared,
                    "provider_prompt_hit": result.provider_prompt_hit,
                    "provider_cache_read_tokens": result.provider_cache_read_tokens,
                    "provider_cache_miss_tokens": result.provider_cache_miss_tokens,
                    "provider_cache_hit_rate": result.provider_cache_hit_rate,
                },
            )
            self._pending[tool].append((trace, not cache_hit))
        return result

    def mark_effect(self, tool: str, *, state_changed: bool) -> None:
        """Commit delayed state-change accounting after a phase validates output."""
        if self.controller is None or self.state is None:
            return
        pending = self._pending.pop(tool, [])
        if not pending:
            return
        remote = [trace for trace, was_remote in pending if was_remote]
        if state_changed:
            # A batch is one semantic tool action even if it needed several
            # requests.  Attribute the committed change to its final request.
            if remote:
                trace = remote[-1]
                trace.state_changed = True
                trace.state_fingerprint_after = self.state.fingerprint()
                usage = self.state.remote_usage.get(tool)
                if usage is not None:
                    usage.state_changes += 1
                self.state.consecutive_remote_no_change = 0
            return
        usage = self.state.remote_usage.get(tool)
        if usage is not None and remote:
            usage.zero_change_calls += len(remote)
            self.state.consecutive_remote_no_change += len(remote)
