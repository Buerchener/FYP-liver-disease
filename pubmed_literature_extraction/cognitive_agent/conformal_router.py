#!/usr/bin/env python3
"""Non-parametric conformal risk routing for Agent v3.

The router can only add review or abstention.  It never overrides deterministic
Verifier/Safe Write failures and requires a frozen calibration artifact.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_agent.relation_contract import TIERED_V2_VERIFICATION_POLICY


ACCEPT_LOCAL = "ACCEPT_LOCAL"
CALL_DEEPSEEK = "CALL_DEEPSEEK"
CALL_QWEN_CRITIC = "CALL_QWEN_CRITIC"
ABSTAIN = "ABSTAIN"
HUMAN_REVIEW = "HUMAN_REVIEW"
ROUTE_DECISIONS = frozenset({ACCEPT_LOCAL, CALL_DEEPSEEK, CALL_QWEN_CRITIC, ABSTAIN, HUMAN_REVIEW})
HARD_VERIFIER_FLAGS = frozenset({
    *TIERED_V2_VERIFICATION_POLICY.factual_reject_flags,
    # Historical feature aliases, mapped only to the same four factual gates.
    "invalid_schema", "missing_endpoint", "evidence_not_in_source", "hard_negation",
})


@dataclass(frozen=True)
class RiskFeatures:
    candidate_id: str
    relation_score: float
    evidence_score: float
    verifier_passed: bool
    verifier_flags: list[str] = field(default_factory=list)
    local_label: str = ""
    deepseek_label: str = ""
    qwen_label: str = ""
    rule_support: int = 0
    rule_conflict: bool = False
    linking_ambiguity: float = 0.0
    study_type: str = "unknown"
    predicate: str = "unknown"
    section: str = "ABSTRACT"
    high_value_direction_change: bool = False
    semantic_only: bool = False

    @property
    def mondrian_group(self) -> str:
        return f"{self.predicate}|{self.section}|{self.study_type}"

    def risk_score(self) -> float:
        relation = max(0.0, min(1.0, float(self.relation_score)))
        evidence = max(0.0, min(1.0, float(self.evidence_score)))
        ambiguity = max(0.0, min(1.0, float(self.linking_ambiguity)))
        risk = 0.38 * (1.0 - relation) + 0.34 * (1.0 - evidence) + 0.12 * ambiguity
        risk += min(0.12, 0.03 * len(set(self.verifier_flags)))
        risk += 0.12 if self.rule_conflict else 0.0
        risk -= min(0.08, 0.02 * max(0, self.rule_support))
        labels = [item for item in (self.local_label, self.deepseek_label, self.qwen_label) if item]
        if len(labels) >= 2:
            risk += 0.16 if len(set(labels)) > 1 else -0.08
        if self.high_value_direction_change:
            risk += 0.10
        return round(max(0.0, min(1.0, risk)), 6)


@dataclass(frozen=True)
class CalibrationExample:
    candidate_id: str
    risk_score: float
    error: int
    mondrian_group: str = "global"
    risk_target: str = "semantic"  # semantic | write

    @classmethod
    def from_features(
        cls, features: RiskFeatures, *, error: bool, risk_target: str = "semantic",
    ) -> "CalibrationExample":
        if risk_target not in {"semantic", "write"}:
            raise ValueError("risk_target must be semantic or write")
        return cls(
            features.candidate_id, features.risk_score(), int(bool(error)),
            features.mondrian_group, risk_target,
        )


@dataclass
class ConformalCalibration:
    examples: list[CalibrationExample] = field(default_factory=list)
    version: str = "agent-v3-conformal-v1"
    source_manifest_hash: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConformalCalibration":
        return cls(
            examples=[CalibrationExample(**item) for item in payload.get("examples", [])],
            version=str(payload.get("version", "agent-v3-conformal-v1")),
            source_manifest_hash=str(payload.get("source_manifest_hash", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_manifest_hash": self.source_manifest_hash,
            "examples": [asdict(item) for item in self.examples],
        }

    def pool(
        self, group: str, min_group_size: int, risk_target: str = "semantic",
    ) -> tuple[list[CalibrationExample], str]:
        target_examples = [
            item for item in self.examples if item.risk_target == risk_target
        ]
        grouped = [item for item in target_examples if item.mondrian_group == group]
        if len(grouped) >= min_group_size:
            return grouped, group
        return target_examples, "global_fallback"

    def is_viable(self, risk_target: str, minimum: int = 20) -> bool:
        examples = [item for item in self.examples if item.risk_target == risk_target]
        outcomes = {int(item.error) for item in examples}
        return len(examples) >= minimum and outcomes == {0, 1}

    @staticmethod
    def _finite_sample_quantile(values: list[float], probability: float) -> float:
        if not values:
            return 1.0
        ordered = sorted(values)
        rank = min(len(ordered), math.ceil((len(ordered) + 1) * probability))
        return ordered[max(0, rank - 1)]

    def upper_error_risk(
        self, *, predicted_risk: float, group: str, alpha: float,
        min_group_size: int, risk_target: str = "semantic",
    ) -> tuple[float, str, int, float]:
        pool, group_used = self.pool(group, min_group_size, risk_target)
        residuals = [float(item.error) - float(item.risk_score) for item in pool]
        qhat = self._finite_sample_quantile(residuals, 1.0 - alpha)
        upper = max(0.0, min(1.0, float(predicted_risk) + qhat))
        return round(upper, 6), group_used, len(pool), round(qhat, 6)


@dataclass(frozen=True)
class RiskRoute:
    candidate_id: str
    decision: str
    predicted_risk: float
    conformal_upper_risk: float
    alpha: float
    calibration_group: str
    calibration_size: int
    qhat: float
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConformalRiskRouter:
    def __init__(
        self, calibration: ConformalCalibration | None = None, *,
        alpha_import_ready: float = 0.05, alpha_semantic: float = 0.10,
        min_group_size: int = 20,
    ):
        self.calibration = calibration or ConformalCalibration()
        self.alpha_import_ready = alpha_import_ready
        self.alpha_semantic = alpha_semantic
        self.min_group_size = min_group_size

    @classmethod
    def from_path(cls, path: str | Path = "", **kwargs) -> "ConformalRiskRouter":
        if not path:
            return cls(**kwargs)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(ConformalCalibration.from_dict(payload), **kwargs)

    def route(self, features: RiskFeatures) -> RiskRoute:
        alpha = self.alpha_semantic if features.semantic_only else self.alpha_import_ready
        risk_target = "semantic" if features.semantic_only else "write"
        predicted = features.risk_score()
        upper, group, size, qhat = self.calibration.upper_error_risk(
            predicted_risk=predicted, group=features.mondrian_group,
            alpha=alpha, min_group_size=self.min_group_size,
            risk_target=risk_target,
        )
        flags = set(features.verifier_flags)
        hard_flags = sorted(flags & HARD_VERIFIER_FLAGS)
        reasons: list[str] = []
        if not features.verifier_passed or hard_flags:
            decision = ABSTAIN
            reasons.extend(["verifier_not_passed", *[f"hard_flag:{item}" for item in hard_flags]])
        elif features.rule_conflict:
            decision = HUMAN_REVIEW
            reasons.append("rule_conflict")
        elif upper <= alpha and not features.high_value_direction_change:
            decision = ACCEPT_LOCAL
            reasons.append("conformal_error_target_met")
        elif not features.deepseek_label:
            decision = CALL_DEEPSEEK
            reasons.append("uncertainty_requires_primary_adjudicator")
        elif features.qwen_label and len({features.local_label, features.deepseek_label, features.qwen_label}) > 1:
            decision = HUMAN_REVIEW
            reasons.append("dual_model_disagreement")
        elif features.deepseek_label != features.local_label or features.high_value_direction_change:
            decision = CALL_QWEN_CRITIC
            reasons.append("deepseek_disagreement_or_high_value_direction")
        else:
            decision = ABSTAIN if not features.semantic_only else HUMAN_REVIEW
            reasons.append("conformal_risk_exceeds_target")
        return RiskRoute(
            features.candidate_id, decision, predicted, upper, alpha,
            group, size, qhat, reasons,
        )

    @staticmethod
    def enforce_no_upgrade(route: RiskRoute, *, verifier_passed: bool) -> RiskRoute:
        if verifier_passed or route.decision != ACCEPT_LOCAL:
            return route
        return RiskRoute(
            **{**route.to_dict(), "decision": ABSTAIN,
               "reason_codes": [*route.reason_codes, "accept_downgraded_by_verifier"]}
        )
