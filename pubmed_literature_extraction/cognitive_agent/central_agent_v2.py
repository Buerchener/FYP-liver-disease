#!/usr/bin/env python3
"""Quality-first deterministic controller for the article-level Agent v2.

The controller owns budgets, action selection, audit trails and termination.
It deliberately does not own biomedical extraction or write permission: tools
produce candidates and the deterministic verifier/decision engine remain the
only authority for import readiness and persistence.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from cognitive_agent.relation_contract import SEMANTIC_REJECT_FLAGS


ROUTE_ORDER = {"FAST": 0, "STANDARD": 1, "DEEP": 2}
ROUTE_BUDGETS = {
    "FAST": {"max_actions": 12, "max_aux_remote_calls": 2, "max_neo4j_calls": 2, "soft_timeout_s": 30.0},
    "STANDARD": {"max_actions": 20, "max_aux_remote_calls": 4, "max_neo4j_calls": 4, "soft_timeout_s": 60.0},
    "DEEP": {"max_actions": 28, "max_aux_remote_calls": 6, "max_neo4j_calls": 6, "soft_timeout_s": 120.0},
}

HARD_RELATION_FLAGS = SEMANTIC_REJECT_FLAGS
REVIEWABLE_FLAGS = frozenset({
    "trigger_missing", "trigger_not_linking_endpoints", "trigger_direction_mismatch",
    "weak_evidence", "uncertain", "pair_low_confidence", "pair_ambiguous_predicate",
    "judge_uncertain", "judge_verifier_conflict", "agent_evidence_repaired", "manual_review",
})


@dataclass
class AgentBudget:
    route: str
    profile: str = "quality"
    max_actions: int = 12
    max_aux_remote_calls: int = 2
    max_neo4j_calls: int = 2
    soft_timeout_s: float = 30.0
    hard_timeout_s: float = 180.0
    hard_max_actions: int = 40
    hard_max_aux_remote_calls: int = 8
    hard_max_neo4j_calls: int = 8

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class AgentActionTrace:
    sequence: int
    tool: str
    decision: str
    reason: str
    started_at_s: float
    latency_s: float = 0.0
    cache_status: str = "not_applicable"
    state_fingerprint_before: str = ""
    state_fingerprint_after: str = ""
    state_changed: bool = False
    remote: bool = False
    neo4j: bool = False
    prompt_tokens: int = 0
    output_tokens: int = 0
    result_status: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RemoteToolUsage:
    attempted: int = 0
    successful: int = 0
    retried: int = 0
    cached: int = 0
    timeouts: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    state_changes: int = 0
    zero_change_calls: int = 0

    def to_dict(self) -> dict:
        return {
            **self.__dict__,
            "latency_s": round(self.latency_s, 4),
        }


@dataclass
class ArticleAgentState:
    pmid: str
    execution_mode: str
    route: str
    budget: AgentBudget
    started_at: float = field(default_factory=time.monotonic)
    initial_budget: dict = field(default_factory=dict)
    budget_escalations: list[dict] = field(default_factory=list)
    action_trace: list[AgentActionTrace] = field(default_factory=list)
    remote_usage: dict[str, RemoteToolUsage] = field(default_factory=dict)
    cache: dict[str, int] = field(default_factory=lambda: {
        "memory_hits": 0, "persistent_hits": 0, "misses": 0,
        "singleflight_shared": 0, "remote_calls_avoided": 0,
    })
    entities: list[dict] = field(default_factory=list)
    candidate_pairs: list[dict] = field(default_factory=list)
    verified_relations: list[dict] = field(default_factory=list)
    semantic_relations: list[dict] = field(default_factory=list)
    accepted_relations: list[dict] = field(default_factory=list)
    rejected_relations: list[dict] = field(default_factory=list)
    review_relations: list[dict] = field(default_factory=list)
    disputed_relations: list[dict] = field(default_factory=list)
    hypothesis_relations: list[dict] = field(default_factory=list)
    unresolved_issues: list[str] = field(default_factory=list)
    linking_ambiguity: bool = False
    model_disagreement: bool = False
    consecutive_remote_no_change: int = 0
    termination_reason: str = ""
    terminal_status: str = "running"
    fingerprints_seen: set[str] = field(default_factory=set, repr=False)

    @property
    def elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    @property
    def aux_remote_calls(self) -> int:
        return sum(item.attempted for item in self.remote_usage.values())

    @property
    def neo4j_calls(self) -> int:
        return sum(1 for item in self.action_trace if item.neo4j and item.decision == "CALL")

    def fingerprint(self) -> str:
        payload = {
            "entities": self.entities,
            "candidate_pairs": self.candidate_pairs,
            "verified_relations": self.verified_relations,
            "hypothesis_relations": self.hypothesis_relations,
            "issues": self.unresolved_issues,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "execution_mode": self.execution_mode,
            "route": self.route,
            "budget": self.budget.to_dict(),
            "initial_budget": self.initial_budget,
            "budget_escalations": self.budget_escalations,
            "action_trace": [item.to_dict() for item in self.action_trace],
            "remote_usage": {key: value.to_dict() for key, value in self.remote_usage.items()},
            "cache": self.cache,
            "state_summary": {
                "entity_count": len(self.entities),
                "candidate_pair_count": len(self.candidate_pairs),
                "verified_relation_count": len(self.verified_relations),
                "semantic_relation_count": len(self.semantic_relations),
                "accepted_relation_count": len(self.accepted_relations),
                "rejected_relation_count": len(self.rejected_relations),
                "review_relation_count": len(self.review_relations),
                "disputed_relation_count": len(self.disputed_relations),
                "hypothesis_relation_count": len(self.hypothesis_relations),
                "unresolved_issues": list(self.unresolved_issues),
                "linking_ambiguity": self.linking_ambiguity,
                "model_disagreement": self.model_disagreement,
            },
            "termination": {
                "status": self.terminal_status,
                "reason": self.termination_reason,
                "elapsed_s": round(self.elapsed_s, 4),
                "action_count": len(self.action_trace),
                "aux_remote_calls": self.aux_remote_calls,
                "neo4j_calls": self.neo4j_calls,
            },
        }


class CentralAgentV2:
    """Deterministic, verifier-guided controller with auditable budgets."""

    VALID_MODES = frozenset({"legacy", "agent-v2-shadow", "agent-v2"})
    VALID_PROFILES = frozenset({"quality", "balanced", "speed"})

    def __init__(
        self, *, execution_mode: str = "legacy", budget_profile: str = "quality",
        max_actions: int = 0, max_aux_remote_calls: int = 0,
        max_neo4j_calls: int = 0, soft_timeout_s: float = 0.0,
        hard_timeout_s: float = 180.0,
    ):
        if execution_mode not in self.VALID_MODES:
            raise ValueError(f"execution_mode must be one of {sorted(self.VALID_MODES)}")
        if budget_profile not in self.VALID_PROFILES:
            raise ValueError(f"agent budget profile must be one of {sorted(self.VALID_PROFILES)}")
        self.execution_mode = execution_mode
        self.budget_profile = budget_profile
        self.overrides = {
            "max_actions": max(0, int(max_actions)),
            "max_aux_remote_calls": max(0, int(max_aux_remote_calls)),
            "max_neo4j_calls": max(0, int(max_neo4j_calls)),
            "soft_timeout_s": max(0.0, float(soft_timeout_s)),
            "hard_timeout_s": max(1.0, float(hard_timeout_s)),
        }

    @property
    def enabled(self) -> bool:
        return self.execution_mode != "legacy"

    @property
    def active(self) -> bool:
        return self.execution_mode == "agent-v2"

    def _budget(self, route: str) -> AgentBudget:
        route = route if route in ROUTE_BUDGETS else "STANDARD"
        values = dict(ROUTE_BUDGETS[route])
        if self.budget_profile == "balanced":
            values["max_aux_remote_calls"] = max(1, values["max_aux_remote_calls"] - 1)
        elif self.budget_profile == "speed":
            values["max_actions"] = max(8, values["max_actions"] // 2)
            values["max_aux_remote_calls"] = 1
            values["max_neo4j_calls"] = min(2, values["max_neo4j_calls"])
            values["soft_timeout_s"] = min(30.0, values["soft_timeout_s"])
        for key in ("max_actions", "max_aux_remote_calls", "max_neo4j_calls", "soft_timeout_s"):
            if self.overrides[key] > 0:
                values[key] = self.overrides[key]
        return AgentBudget(
            route=route,
            profile=self.budget_profile,
            hard_timeout_s=self.overrides["hard_timeout_s"],
            **values,
        )

    def start(self, pmid: str, route: str) -> ArticleAgentState:
        if route == "RECOVERY":
            route = "DEEP"
        budget = self._budget(route)
        state = ArticleAgentState(
            pmid=str(pmid), execution_mode=self.execution_mode,
            route=budget.route, budget=budget,
        )
        state.initial_budget = budget.to_dict()
        return state

    @staticmethod
    def _relation_partition(
        relations: list[dict],
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        accepted, rejected, review, semantic = [], [], [], []
        for relation in relations:
            flags = set(relation.get("quality_flags", []) or [])
            semantic_status = str(relation.get("semantic_status", "") or "")
            if semantic_status == "REJECTED" or flags & HARD_RELATION_FLAGS:
                rejected.append(relation)
            elif relation.get("import_ready"):
                accepted.append(relation)
                semantic.append(relation)
            elif semantic_status == "ACCEPTED":
                semantic.append(relation)
            else:
                review.append(relation)
        return accepted, rejected, review, semantic

    def observe(
        self, state: ArticleAgentState, *, entities: list[dict],
        candidate_pairs: list[dict], verification: dict,
        extraction_error: str = "", recovery_candidate_count: int = 0,
    ) -> None:
        state.entities = list(entities or [])
        state.candidate_pairs = list(candidate_pairs or [])
        state.verified_relations = list(verification.get("relations", []) or [])
        accepted, rejected, review, semantic = self._relation_partition(
            state.verified_relations
        )
        state.accepted_relations = accepted
        state.rejected_relations = rejected
        state.review_relations = review
        state.semantic_relations = semantic
        state.linking_ambiguity = any(
            item.get("ambiguity_reason") or len(item.get("candidates", []) or []) > 1
            for item in verification.get("entities", []) or []
        )
        state.model_disagreement = any(
            "manual_review" in set(item.get("quality_flags", []) or [])
            for item in state.verified_relations
        )
        issues: list[str] = []
        if extraction_error:
            issues.append("primary_extraction_error")
        if recovery_candidate_count:
            issues.append("relation_gap_candidates")
        if state.linking_ambiguity:
            issues.append("linking_ambiguity")
        if state.review_relations:
            issues.append("reviewable_relations")
        if state.model_disagreement:
            issues.append("model_disagreement")
        state.unresolved_issues = list(dict.fromkeys(issues))

    def can_call_remote(self, state: ArticleAgentState, tool: str) -> tuple[bool, str]:
        if state.elapsed_s >= state.budget.hard_timeout_s:
            return False, "hard_timeout_reached"
        if state.elapsed_s >= state.budget.soft_timeout_s:
            return False, "soft_timeout_remote_gate"
        if len(state.action_trace) >= min(state.budget.max_actions, state.budget.hard_max_actions):
            return False, "action_budget_exhausted"
        if state.aux_remote_calls >= min(
            state.budget.max_aux_remote_calls, state.budget.hard_max_aux_remote_calls
        ):
            return False, "aux_remote_budget_exhausted"
        if state.consecutive_remote_no_change >= 2:
            return False, "two_remote_calls_without_state_change"
        usage = state.remote_usage.get(tool, RemoteToolUsage())
        per_tool_limit = {
            "article_profiler": 1,
            "entity_recovery": 1,
            "pairwise_judge": 4,
            "evidence_entailment": 2,
            "second_llm_refiner": 4,
            "debug_reviewer": 1,
            "qwen_edit_critic": 1,
        }.get(tool, 4)
        if usage.attempted >= per_tool_limit:
            return False, "per_tool_remote_budget_exhausted"
        return True, "within_remote_budget"

    def can_call_neo4j(self, state: ArticleAgentState) -> tuple[bool, str]:
        if state.elapsed_s >= state.budget.hard_timeout_s:
            return False, "hard_timeout_reached"
        if state.elapsed_s >= state.budget.soft_timeout_s:
            return False, "soft_timeout_neo4j_gate"
        if len(state.action_trace) >= min(state.budget.max_actions, state.budget.hard_max_actions):
            return False, "action_budget_exhausted"
        if state.neo4j_calls >= min(state.budget.max_neo4j_calls, state.budget.hard_max_neo4j_calls):
            return False, "neo4j_budget_exhausted"
        return True, "within_neo4j_budget"

    def maybe_escalate(self, state: ArticleAgentState, reason: str) -> bool:
        current_rank = ROUTE_ORDER.get(state.route, 1)
        if current_rank >= ROUTE_ORDER["DEEP"]:
            return False
        if not state.unresolved_issues:
            return False
        next_route = ("STANDARD", "DEEP")[current_rank]
        old = state.budget.to_dict()
        new_budget = self._budget(next_route)
        state.route = next_route
        state.budget = new_budget
        state.budget_escalations.append({
            "from": old["route"], "to": next_route, "reason": reason,
            "elapsed_s": round(state.elapsed_s, 4),
        })
        return True

    @staticmethod
    def tool_cache_key(tool: str, payload: dict) -> str:
        def remove_secrets(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: remove_secrets(item)
                    for key, item in value.items()
                    if "key" not in str(key).casefold()
                }
            if isinstance(value, (list, tuple)):
                return [remove_secrets(item) for item in value]
            return value

        safe_payload = remove_secrets(payload)
        encoded = json.dumps(
            {"namespace": "central-agent-v2-tool-cache-v1", "tool": tool, "payload": safe_payload},
            sort_keys=True, ensure_ascii=False, default=str,
        )
        return "agent-v2:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def record_action(
        self, state: ArticleAgentState, *, tool: str, decision: str,
        reason: str, before: str = "", after: str = "", latency_s: float = 0.0,
        cache_status: str = "not_applicable", remote: bool = False,
        neo4j: bool = False, result_status: str = "", prompt_tokens: int = 0,
        output_tokens: int = 0, attempt_count: int = 1, retry_count: int = 0,
        details: dict | None = None,
    ) -> AgentActionTrace:
        changed = bool(before and after and before != after)
        trace = AgentActionTrace(
            sequence=len(state.action_trace) + 1, tool=tool, decision=decision,
            reason=reason, started_at_s=round(state.elapsed_s, 4),
            latency_s=round(float(latency_s or 0.0), 4), cache_status=cache_status,
            state_fingerprint_before=before, state_fingerprint_after=after,
            state_changed=changed, remote=remote, neo4j=neo4j,
            prompt_tokens=int(prompt_tokens or 0), output_tokens=int(output_tokens or 0),
            result_status=result_status, details=details or {},
        )
        state.action_trace.append(trace)
        cache_counter = {
            "memory_hit": "memory_hits",
            "persistent_hit": "persistent_hits",
            "miss": "misses",
            "singleflight_shared": "singleflight_shared",
        }.get(cache_status)
        if cache_counter:
            state.cache[cache_counter] += 1
        if cache_status in {"memory_hit", "persistent_hit", "singleflight_shared"}:
            state.cache["remote_calls_avoided"] += 1
        if remote:
            usage = state.remote_usage.setdefault(tool, RemoteToolUsage())
            cached = cache_status in {"memory_hit", "persistent_hit", "singleflight_shared"}
            defer_state_change = bool((details or {}).get("defer_state_change"))
            if cached:
                usage.cached += 1
            else:
                usage.attempted += max(0, int(attempt_count))
                usage.retried += max(0, int(retry_count))
                usage.successful += int(result_status == "OK")
                usage.timeouts += int("timeout" in result_status.casefold())
                usage.prompt_tokens += int(prompt_tokens or 0)
                usage.output_tokens += int(output_tokens or 0)
                usage.latency_s += float(latency_s or 0.0)
                if defer_state_change:
                    # Some batched tools update article state only after all
                    # model responses have been validated.  Their broker will
                    # resolve state-change accounting once that phase commits.
                    pass
                elif changed:
                    usage.state_changes += 1
                    state.consecutive_remote_no_change = 0
                else:
                    usage.zero_change_calls += 1
                    state.consecutive_remote_no_change += 1
        return trace

    def should_use_rag(self, state: ArticleAgentState, *, enabled: bool, memory_available: bool) -> tuple[bool, str]:
        if not enabled:
            return False, "neo4j_rag_disabled"
        if not memory_available:
            return False, "neo4j_unavailable"
        if not (state.linking_ambiguity or state.review_relations or state.disputed_relations):
            return False, "no_linking_or_conflict_uncertainty"
        return self.can_call_neo4j(state)

    def should_adjudicate(self, state: ArticleAgentState, *, enabled: bool) -> tuple[bool, str]:
        if not enabled:
            return False, "second_llm_disabled"
        if not (state.review_relations or "relation_gap_candidates" in state.unresolved_issues):
            return False, "no_reviewable_or_recovery_candidate"
        return self.can_call_remote(state, "second_llm_refiner")

    def should_review(self, state: ArticleAgentState, *, enabled: bool) -> tuple[bool, str]:
        if not enabled:
            return False, "debug_reviewer_disabled"
        if not (state.model_disagreement or state.review_relations):
            return False, "no_unresolved_high_value_review"
        return self.can_call_remote(state, "debug_reviewer")

    def should_run_causal(self, state: ArticleAgentState) -> tuple[bool, str]:
        if len(state.accepted_relations) < 2:
            return False, "fewer_than_two_import_ready_relations"
        return True, "compose_only_verified_import_ready_relations"

    def should_run_conflict(self, state: ArticleAgentState) -> tuple[bool, str]:
        graph_statuses = {"KNOWN", "INVERTED", "CONTRADICTING"}
        if not any(item.get("neo4j_status") in graph_statuses for item in state.verified_relations):
            return False, "no_existing_graph_relation_to_compare"
        return True, "verified_relation_requires_graph_conflict_policy"

    def finalize(self, state: ArticleAgentState, reason: str = "all_candidates_terminal") -> None:
        if state.elapsed_s >= state.budget.hard_timeout_s:
            state.terminal_status = "human_review"
            state.termination_reason = "hard_timeout_reached"
        elif state.unresolved_issues:
            state.terminal_status = "human_review"
            state.termination_reason = reason if reason != "all_candidates_terminal" else "unresolved_issues_remain"
        else:
            state.terminal_status = "accepted"
            state.termination_reason = reason
