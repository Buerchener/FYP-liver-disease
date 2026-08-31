"""Deterministic cache-aware execution ledger for auxiliary model stages."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable


CACHE_NAMESPACE = "remote-execution-v2"


def stable_request_id(
    *, tool: str, stage: str, round_index: int, batch: str | int,
    model: str, schema_version: str, input_hash: str,
) -> str:
    payload = {
        "namespace": CACHE_NAMESPACE, "tool": tool, "stage": stage,
        "round": int(round_index), "batch": str(batch), "model": model,
        "schema_version": schema_version, "input_hash": input_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "req-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


@dataclass
class RemoteExecutionRecord:
    request_id: str
    tool: str
    stage: str
    round: int
    batch: str
    logical_step: int
    physical_attempts: int = 0
    cache_replayed: bool = False
    blocked_kind: str = ""
    result_status: str = ""
    decision_at: float = 0.0
    completed_at: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    duplicate_of: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class RemoteExecutionLedger:
    max_logical_steps: int
    max_physical_attempts: int
    started_at: float = field(default_factory=time.monotonic, repr=False)
    records: list[RemoteExecutionRecord] = field(default_factory=list)
    completed_request_ids: set[str] = field(default_factory=set, repr=False)

    @property
    def logical_aux_steps(self) -> int:
        return sum(not bool(item.duplicate_of) for item in self.records)

    @property
    def physical_remote_attempts(self) -> int:
        return sum(item.physical_attempts for item in self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CACHE_NAMESPACE,
            "logical_aux_steps": self.logical_aux_steps,
            "physical_remote_attempts": self.physical_remote_attempts,
            "successful_results": sum(item.result_status == "OK" for item in self.records),
            "cached_results": sum(item.cache_replayed for item in self.records),
            "records": [item.to_dict() for item in self.records],
        }


class RemoteExecutionGateway:
    """Own logical routing and physical budgets independently of cache state.

    ``cache_lookup`` returns ``(payload, cache_status)``.  It must perform the
    lookup before invoking its supplied factory so a cache hit remains legal
    after a soft SLO violation or after the physical budget is exhausted.
    """

    def __init__(self, ledger: RemoteExecutionLedger, *, soft_timeout_s: float = 0.0,
                 hard_timeout_s: float = 180.0):
        self.ledger = ledger
        self.soft_timeout_s = max(0.0, float(soft_timeout_s))
        self.hard_timeout_s = max(1.0, float(hard_timeout_s))
        self._results: dict[str, tuple[Any, str]] = {}

    def execute(
        self, *, request_id: str, tool: str, stage: str, round_index: int,
        batch: str | int, cache_lookup: Callable[[Callable[[], Any]], tuple[Any, str]],
        invoke: Callable[[], Any], attempts_of: Callable[[Any], int] | None = None,
        status_of: Callable[[Any], str] | None = None,
    ) -> tuple[Any | None, str, RemoteExecutionRecord]:
        now = time.monotonic()
        if request_id in self._results:
            result, cache_status = self._results[request_id]
            record = RemoteExecutionRecord(
                request_id=request_id, tool=tool, stage=stage, round=int(round_index),
                batch=str(batch), logical_step=self.ledger.logical_aux_steps,
                cache_replayed=True, result_status=(status_of(result) if status_of else "OK"),
                decision_at=now, completed_at=now, duplicate_of=request_id,
            )
            self.ledger.records.append(record)
            return result, cache_status, record
        if self.ledger.logical_aux_steps >= self.ledger.max_logical_steps:
            return self._blocked(request_id, tool, stage, round_index, batch, "logical_budget")
        if now - self.ledger.started_at >= self.hard_timeout_s:
            return self._blocked(request_id, tool, stage, round_index, batch, "hard_timeout")

        logical_step = self.ledger.logical_aux_steps + 1
        invoked = False
        invoke_attempts = 0

        def guarded_invoke() -> Any:
            nonlocal invoked, invoke_attempts
            if self.ledger.physical_remote_attempts >= self.ledger.max_physical_attempts:
                raise PhysicalBudgetExhausted("physical_remote_budget")
            invoked = True
            result = invoke()
            invoke_attempts = max(1, int(attempts_of(result) if attempts_of else 1))
            if self.ledger.physical_remote_attempts + invoke_attempts > self.ledger.max_physical_attempts:
                raise PhysicalBudgetExhausted("physical_remote_budget")
            return result

        try:
            result, cache_status = cache_lookup(guarded_invoke)
        except PhysicalBudgetExhausted:
            return self._blocked(request_id, tool, stage, round_index, batch, "physical_budget")
        completed = time.monotonic()
        cached = cache_status in {"memory_hit", "persistent_hit", "singleflight_shared"}
        record = RemoteExecutionRecord(
            request_id=request_id, tool=tool, stage=stage, round=int(round_index),
            batch=str(batch), logical_step=logical_step,
            physical_attempts=0 if cached else invoke_attempts,
            cache_replayed=cached, result_status=(status_of(result) if status_of else "OK"),
            decision_at=now, completed_at=completed, latency_s=completed - now,
        )
        self.ledger.records.append(record)
        self.ledger.completed_request_ids.add(request_id)
        self._results[request_id] = (result, cache_status)
        return result, cache_status, record

    def _blocked(self, request_id: str, tool: str, stage: str, round_index: int,
                 batch: str | int, kind: str):
        now = time.monotonic()
        record = RemoteExecutionRecord(
            request_id=request_id, tool=tool, stage=stage, round=int(round_index),
            batch=str(batch), logical_step=self.ledger.logical_aux_steps + 1,
            blocked_kind=kind, result_status="INCOMPLETE_TIMEOUT" if kind == "hard_timeout" else "BUDGET_BLOCKED",
            decision_at=now, completed_at=now,
        )
        self.ledger.records.append(record)
        return None, "blocked", record


class PhysicalBudgetExhausted(RuntimeError):
    pass
