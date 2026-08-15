#!/usr/bin/env python3
"""Offline rule induction, critique and promotion for Agent v3."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognitive_agent.aux_model_registry import AuxModelRegistry, StructuredModelResult
from cognitive_agent.rule_memory import (
    ErrorCard, RuleBundle, RulePromotionGate, RuleValidationError,
    RuleValidator, SoftRule, merge_candidate_rules, render_rule_context,
)


RULE_INDUCTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rules"],
    "properties": {
        "rules": {
            "type": "array", "maxItems": 24,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": [
                    "kind", "conditions", "action", "value", "guidance",
                    "rationale", "support_error_ids",
                ],
                "properties": {
                    "kind": {"type": "string", "enum": [
                        "prompt_guidance", "pair_prior", "routing", "downgrade",
                    ]},
                    "conditions": {"type": "object"},
                    "action": {"type": "string", "enum": [
                        "ADD_GUIDANCE", "ADJUST_PAIR_SCORE", "CALL_DEEPSEEK",
                        "CALL_QWEN_CRITIC", "ABSTAIN", "REVIEW", "REJECT",
                    ]},
                    "value": {}, "guidance": {"type": "string", "maxLength": 400},
                    "rationale": {"type": "string", "maxLength": 800},
                    "support_error_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

ACTION_ALIASES = {
    "add_guidance": "ADD_GUIDANCE",
    "adjust_pair_score": "ADJUST_PAIR_SCORE",
    "call_deepseek": "CALL_DEEPSEEK",
    "call_qwen_critic": "CALL_QWEN_CRITIC",
    "abstain": "ABSTAIN", "review": "REVIEW", "reject": "REJECT",
}
CONDITION_ALIASES = {
    "predicate": "predicates", "subject_type": "subject_types",
    "object_type": "object_types", "study_type": "study_types",
    "section": "sections", "direction": "directions",
    "lexical_cue": "lexical_cues", "forbidden_cue": "forbidden_cues",
    "quality_flag": "quality_flags",
}


@dataclass
class RuleLearningRun:
    status: str = "NOT_STARTED"
    candidates: list[SoftRule] = field(default_factory=list)
    promoted: list[SoftRule] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    primary_result: dict[str, Any] = field(default_factory=dict)
    critic_results: list[dict[str, Any]] = field(default_factory=list)
    bundle_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "candidates": [item.to_dict() for item in self.candidates],
            "promoted": [item.to_dict() for item in self.promoted],
            "rejected": self.rejected,
            "primary_result": self.primary_result,
            "critic_results": self.critic_results,
            "bundle_hash": self.bundle_hash,
        }


class RuleLearner:
    def __init__(self, registry: AuxModelRegistry):
        self.registry = registry

    @staticmethod
    def _prompt(cards: list[ErrorCard]) -> str:
        safe_cards = [{
            "error_id": card.error_id,
            "category": card.category,
            "expected": card.expected,
            "observed": card.observed,
            "reason_codes": card.reason_codes,
            "split": card.split,
        } for card in cards]
        return (
            "Induce general, closed-DSL soft rules from these biomedical extraction errors. "
            "Never include PMID, exact article text, entity-specific memorization, executable code, "
            "schema changes, or permission to bypass verification. Rules may only add prompt guidance, "
            "adjust pair score within [-0.20,0.20], route to a model, or downgrade/reject. "
            "Use only these exact action enums: ADD_GUIDANCE, ADJUST_PAIR_SCORE, "
            "CALL_DEEPSEEK, CALL_QWEN_CRITIC, ABSTAIN, REVIEW, REJECT. "
            "Condition keys are plural: predicates, subject_types, object_types, "
            "study_types, sections, lexical_cues, forbidden_cues, directions, "
            "quality_flags; numeric confidence bounds and requires_both_endpoints are also allowed. "
            "Return strict JSON matching the supplied schema.\nERROR_CARDS="
            + json.dumps(safe_cards, ensure_ascii=False)
        )

    @staticmethod
    def _normalize_protocol(payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize only documented spelling aliases; unknown DSL stays invalid."""
        normalized = dict(payload)
        raw_action = str(normalized.get("action", "") or "")
        normalized["action"] = ACTION_ALIASES.get(raw_action, raw_action)
        conditions: dict[str, Any] = {}
        for raw_key, raw_value in dict(normalized.get("conditions", {}) or {}).items():
            key = CONDITION_ALIASES.get(str(raw_key), str(raw_key))
            value = raw_value
            if key not in {
                "requires_both_endpoints", "min_evidence_confidence",
                "max_evidence_confidence",
            } and not isinstance(value, list):
                value = [value]
            conditions[key] = value
        normalized["conditions"] = conditions
        return normalized

    @staticmethod
    def _rule_from_payload(payload: dict[str, Any], cards: list[ErrorCard], model: str) -> SoftRule:
        payload = RuleLearner._normalize_protocol(payload)
        support_ids = [str(item) for item in payload.get("support_error_ids", [])]
        by_id = {card.error_id: card for card in cards}
        support_pmids = sorted({by_id[item].pmid for item in support_ids if item in by_id})
        core = {
            "kind": str(payload.get("kind", "")),
            "conditions": dict(payload.get("conditions", {}) or {}),
            "action": str(payload.get("action", "")),
            "value": payload.get("value"),
            "guidance": str(payload.get("guidance", "") or ""),
        }
        rule = SoftRule(
            rule_id=RuleValidator.content_id(core), version=1,
            kind=core["kind"], status="candidate",
            conditions=core["conditions"], action=core["action"],
            value=core["value"], guidance=core["guidance"],
            rationale=str(payload.get("rationale", "") or ""),
            support_error_ids=support_ids, support_pmids=support_pmids,
            learner_model=model,
        )
        RuleValidator.validate(rule)
        return rule

    def induce(self, cards: list[ErrorCard]) -> tuple[list[SoftRule], StructuredModelResult, list[dict[str, Any]]]:
        result = self.registry.call_json(
            "primary",
            system_prompt=(
                "You are a biomedical extraction rule inducer. Produce only general soft rules "
                "in the supplied closed DSL. Hard verifier and Safe Write constraints are immutable."
            ),
            user_prompt=self._prompt(cards), schema_hint=RULE_INDUCTION_SCHEMA,
        )
        rejected: list[dict[str, Any]] = []
        if result.status != "OK":
            return [], result, rejected
        rules: list[SoftRule] = []
        raw_rules = result.payload.get("rules", []) or []
        for index, payload in enumerate(raw_rules):
            try:
                rules.append(self._rule_from_payload(payload, cards, result.model_id))
            except Exception as exc:
                rejected.append({"index": index, "reason": f"induction_validation:{exc}"})
        if raw_rules and not rules:
            result.status = "PROTOCOL_INVALID"
            result.error = "all induced rules violated the closed DSL"
        return merge_candidate_rules(rules), result, rejected

    def critique(self, rule: SoftRule) -> StructuredModelResult:
        public_rule = rule.to_dict()
        public_rule["support_pmids"] = []
        return self.registry.call_json(
            "critic",
            system_prompt=(
                "You are an independent safety critic for biomedical extraction soft rules. "
                "Reject memorization, over-broad conditions, schema changes, evidence fabrication, "
                "or any verifier/Safe Write bypass. Return JSON only."
            ),
            user_prompt=json.dumps({"rule": public_rule}, ensure_ascii=False),
            schema_hint={"approved": True, "safety_objections": [], "suggested_revision": {}},
        )

    def validate_and_promote(
        self, rules: list[SoftRule], *, replay_metrics: dict[str, dict[str, dict[str, float]]],
        shadow_completed: bool,
    ) -> tuple[list[SoftRule], list[dict[str, Any]], list[dict[str, Any]]]:
        promoted: list[SoftRule] = []
        rejected: list[dict[str, Any]] = []
        critic_audit: list[dict[str, Any]] = []
        for rule in rules:
            critique = self.critique(rule)
            critic_audit.append(critique.to_dict())
            if critique.status != "OK":
                rejected.append({"rule": rule.to_dict(), "reasons": ["critic_unavailable"]})
                continue
            objections = critique.payload.get("safety_objections", []) or []
            rule.critic_model = critique.model_id
            rule.critic_approved = bool(critique.payload.get("approved")) and not objections
            metrics = replay_metrics.get(rule.rule_id, {})
            validation = metrics.get("validation", {})
            calibration = metrics.get("calibration", {})
            allowed, reasons = RulePromotionGate.decide(
                rule, validation=validation, calibration=calibration,
                shadow_completed=shadow_completed,
            )
            rule.metrics = {
                **{f"validation_{key}": value for key, value in validation.items()},
                **{f"calibration_{key}": value for key, value in calibration.items()},
            }
            if allowed:
                rule.status = "active"
                promoted.append(rule)
            else:
                rule.status = "rejected"
                rejected.append({"rule": rule.to_dict(), "reasons": reasons or objections})
        return promoted, rejected, critic_audit

    @staticmethod
    def build_bundle(existing: RuleBundle, promoted: list[SoftRule]) -> RuleBundle:
        active_by_id = {
            rule.rule_id: rule for rule in existing.rules if rule.status == "active"
        }
        active_by_id.update({rule.rule_id: rule for rule in promoted})
        return RuleBundle(
            revision=existing.revision + 1,
            status="active",
            rules=[active_by_id[key] for key in sorted(active_by_id)],
            previous_bundle_hash=existing.bundle_hash,
            metadata={"promotion_count": len(promoted)},
        )

    def run(
        self, cards: list[ErrorCard], *, existing: RuleBundle | None = None,
        replay_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
        shadow_completed: bool = False,
    ) -> tuple[RuleBundle, RuleLearningRun]:
        existing = existing or RuleBundle()
        audit = RuleLearningRun(status="INDUCING")
        candidates, primary, induction_rejections = self.induce(cards)
        audit.primary_result = primary.to_dict()
        audit.candidates = candidates
        audit.rejected.extend(induction_rejections)
        promoted, rejected, critic_results = self.validate_and_promote(
            candidates, replay_metrics=replay_metrics or {},
            shadow_completed=shadow_completed,
        )
        audit.promoted = promoted
        audit.rejected.extend(rejected)
        audit.critic_results = critic_results
        bundle = self.build_bundle(existing, promoted)
        audit.bundle_hash = bundle.bundle_hash
        audit.status = "PROMOTED" if promoted else "NO_PROMOTION"
        return bundle, audit


def write_rule_artifacts(
    output_dir: Path, bundle: RuleBundle, audit: RuleLearningRun,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "active_rules.json").write_text(
        json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "RULE_CONTEXT.md").write_text(render_rule_context(bundle), encoding="utf-8")
    (output_dir / "rule_learning_audit.json").write_text(
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
