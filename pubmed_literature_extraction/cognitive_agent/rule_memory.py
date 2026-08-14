#!/usr/bin/env python3
"""Versioned, verifier-safe soft-rule memory for Agent v3.

Rules are data, never executable code.  They may alter prompt guidance, bounded
pair priors or routing decisions, but cannot weaken deterministic safety gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RULE_MEMORY_VERSION = "agent-v3-rule-memory-v1"
RULE_STATUSES = frozenset({"candidate", "validated", "shadow", "active", "rejected", "retired"})
RULE_KINDS = frozenset({"prompt_guidance", "pair_prior", "routing", "downgrade"})
ALLOWED_ACTIONS = frozenset({
    "ADD_GUIDANCE", "ADJUST_PAIR_SCORE", "CALL_DEEPSEEK", "CALL_QWEN_CRITIC",
    "ABSTAIN", "REVIEW", "REJECT",
})
ALLOWED_CONDITION_KEYS = frozenset({
    "predicates", "subject_types", "object_types", "study_types", "sections",
    "lexical_cues", "forbidden_cues", "directions", "requires_both_endpoints",
    "min_evidence_confidence", "max_evidence_confidence", "quality_flags",
})
FORBIDDEN_TEXT = re.compile(
    r"(?:\bPMID\b|\b\d{7,9}\b|import_ready\s*=\s*true|bypass|override\s+verifier|"
    r"schema\s+(?:change|extension)|safe\s*write\s+(?:bypass|override)|```|__import__|eval\(|exec\()",
    re.IGNORECASE,
)
SAFE_TOKEN_RE = re.compile(r"^[\w\s\-+./():,%]+$", re.UNICODE)


class RuleValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ErrorCard:
    error_id: str
    pmid: str
    category: str
    expected: dict[str, Any] = field(default_factory=dict)
    observed: dict[str, Any] = field(default_factory=dict)
    evidence: str = ""
    reason_codes: list[str] = field(default_factory=list)
    split: str = "induction"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SoftRule:
    rule_id: str
    version: int
    kind: str
    status: str
    conditions: dict[str, Any]
    action: str
    value: float | str | None = None
    guidance: str = ""
    rationale: str = ""
    support_error_ids: list[str] = field(default_factory=list)
    support_pmids: list[str] = field(default_factory=list)
    counterexample_error_ids: list[str] = field(default_factory=list)
    parent_rule_ids: list[str] = field(default_factory=list)
    learner_model: str = ""
    critic_model: str = ""
    critic_approved: bool = False
    metrics: dict[str, float | int] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SoftRule":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in payload.items() if key in fields})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuleMatch:
    rule_id: str
    kind: str
    action: str
    value: float | str | None
    guidance: str
    specificity: int
    reason_codes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleBundle:
    bundle_version: str = RULE_MEMORY_VERSION
    revision: int = 0
    status: str = "active"
    rules: list[SoftRule] = field(default_factory=list)
    previous_bundle_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RuleBundle":
        return cls(
            bundle_version=str(payload.get("bundle_version", RULE_MEMORY_VERSION)),
            revision=int(payload.get("revision", 0)),
            status=str(payload.get("status", "active")),
            rules=[SoftRule.from_dict(item) for item in payload.get("rules", [])],
            previous_bundle_hash=str(payload.get("previous_bundle_hash", "")),
            created_at=str(payload.get("created_at", "")) or datetime.now(timezone.utc).isoformat(),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_version": self.bundle_version,
            "revision": self.revision,
            "status": self.status,
            "rules": [rule.to_dict() for rule in self.rules],
            "previous_bundle_hash": self.previous_bundle_hash,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @property
    def bundle_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class RuleValidator:
    """Validate the closed rule DSL and immutable safety boundary."""

    @classmethod
    def validate(cls, rule: SoftRule) -> None:
        if not re.fullmatch(r"rule-[a-f0-9]{12}", rule.rule_id):
            raise RuleValidationError("rule_id must be content-derived: rule-<12 hex>")
        if rule.version < 1:
            raise RuleValidationError("rule version must be positive")
        if rule.kind not in RULE_KINDS:
            raise RuleValidationError("unsupported rule kind")
        if rule.status not in RULE_STATUSES:
            raise RuleValidationError("unsupported rule status")
        if rule.action not in ALLOWED_ACTIONS:
            raise RuleValidationError("unsupported rule action")
        unknown = set(rule.conditions) - ALLOWED_CONDITION_KEYS
        if unknown:
            raise RuleValidationError(f"unsupported condition keys: {sorted(unknown)}")
        inspectable = json.dumps({
            "conditions": rule.conditions,
            "value": rule.value,
            "guidance": rule.guidance,
            "rationale": rule.rationale,
        }, ensure_ascii=False)
        if FORBIDDEN_TEXT.search(inspectable):
            raise RuleValidationError("rule contains an identifier, executable text, or safety override")
        for key, value in rule.conditions.items():
            if key in {"requires_both_endpoints"}:
                if not isinstance(value, bool):
                    raise RuleValidationError(f"{key} must be boolean")
            elif key in {"min_evidence_confidence", "max_evidence_confidence"}:
                if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                    raise RuleValidationError(f"{key} must be within [0,1]")
            elif not isinstance(value, list) or any(
                not isinstance(item, str) or not item.strip() or len(item) > 80
                for item in value
            ):
                raise RuleValidationError(f"{key} must be a bounded list of strings")
        for key in ("lexical_cues", "forbidden_cues"):
            cues = rule.conditions.get(key, []) or []
            if not isinstance(cues, list) or len(cues) > 12:
                raise RuleValidationError(f"{key} must be a list with at most 12 items")
            for cue in cues:
                cue = str(cue).strip()
                if not cue or len(cue) > 80 or not SAFE_TOKEN_RE.fullmatch(cue):
                    raise RuleValidationError(f"unsafe lexical cue in {key}")
        if rule.kind == "pair_prior":
            if rule.action != "ADJUST_PAIR_SCORE" or not isinstance(rule.value, (int, float)):
                raise RuleValidationError("pair_prior requires numeric ADJUST_PAIR_SCORE")
            if not -0.20 <= float(rule.value) <= 0.20:
                raise RuleValidationError("pair score prior must be within [-0.20, 0.20]")
        if rule.kind == "downgrade" and rule.action not in {"ABSTAIN", "REVIEW", "REJECT"}:
            raise RuleValidationError("downgrade rules may only abstain/review/reject")
        if rule.kind == "routing" and rule.action not in {"CALL_DEEPSEEK", "CALL_QWEN_CRITIC", "ABSTAIN", "REVIEW"}:
            raise RuleValidationError("routing rule has invalid action")
        if rule.kind == "prompt_guidance":
            if rule.action != "ADD_GUIDANCE" or not rule.guidance.strip():
                raise RuleValidationError("prompt guidance rule requires bounded guidance")
            if len(rule.guidance) > 400 or FORBIDDEN_TEXT.search(rule.guidance):
                raise RuleValidationError("prompt guidance is unsafe or too long")
        if any(not str(item).isdigit() for item in rule.support_pmids):
            raise RuleValidationError("support_pmids must contain numeric PMID strings")

    @staticmethod
    def content_id(payload: dict[str, Any]) -> str:
        core = {
            "kind": payload.get("kind"), "conditions": payload.get("conditions", {}),
            "action": payload.get("action"), "value": payload.get("value"),
            "guidance": payload.get("guidance", ""),
        }
        raw = json.dumps(core, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return "rule-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


class RulePromotionGate:
    """Automatic promotion gate.  Safety metrics are non-negotiable."""

    @staticmethod
    def decide(rule: SoftRule, *, validation: dict[str, float], calibration: dict[str, float], shadow_completed: bool) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        try:
            RuleValidator.validate(rule)
        except RuleValidationError as exc:
            return False, [f"dsl_invalid:{exc}"]
        if not rule.critic_approved:
            reasons.append("critic_not_approved")
        if len(set(rule.support_pmids)) < 3:
            reasons.append("fewer_than_three_support_pmids")
        if int(validation.get("errors_fixed", 0)) < 2:
            reasons.append("fewer_than_two_errors_fixed")
        if int(validation.get("new_regressions", 0)) > 1:
            reasons.append("too_many_new_regressions")
        if float(validation.get("dangerous_writes", math.inf)) != 0:
            reasons.append("dangerous_write_detected")
        if float(validation.get("strict_precision_delta", -1.0)) < 0:
            reasons.append("strict_precision_regression")
        if float(calibration.get("bootstrap_non_negative_probability", 0.0)) < 0.90:
            reasons.append("calibration_bootstrap_gate_failed")
        if not shadow_completed:
            reasons.append("shadow_replay_required")
        return not reasons, reasons


class RuleMemory:
    """Load, retrieve and audit an immutable-per-run rule bundle."""

    def __init__(self, *, mode: str = "off", bundle_path: str | Path = "", max_rules: int = 8, max_context_tokens: int = 1200):
        if mode not in {"off", "shadow", "active"}:
            raise ValueError("rule memory mode must be off, shadow, or active")
        self.mode = mode
        self.bundle_path = Path(bundle_path) if bundle_path else None
        self.max_rules = max(1, min(32, int(max_rules)))
        self.max_context_tokens = max(100, min(4000, int(max_context_tokens)))
        self.bundle = RuleBundle(status=mode)
        self.load_error = ""
        if self.bundle_path and self.bundle_path.exists():
            try:
                self.bundle = RuleBundle.from_dict(json.loads(self.bundle_path.read_text(encoding="utf-8")))
                for rule in self.bundle.rules:
                    RuleValidator.validate(rule)
            except Exception as exc:
                self.load_error = str(exc)
                self.bundle = RuleBundle(status="rejected", metadata={"load_error": self.load_error})
                self.mode = "off"
        self.frozen_hash = self.bundle.bundle_hash

    @staticmethod
    def _list_match(expected: Iterable[str], actual: str) -> bool:
        values = {str(item).casefold() for item in expected}
        return not values or str(actual).casefold() in values

    @classmethod
    def _match(cls, rule: SoftRule, context: dict[str, Any]) -> tuple[bool, list[str]]:
        conditions = rule.conditions
        reasons: list[str] = []
        mapping = {
            "predicates": "predicate", "subject_types": "subject_type",
            "object_types": "object_type", "study_types": "study_type",
            "sections": "section", "directions": "direction",
        }
        for key, context_key in mapping.items():
            values = conditions.get(key, []) or []
            if values and not cls._list_match(values, str(context.get(context_key, ""))):
                return False, []
            if values:
                reasons.append(f"{key}_matched")
        evidence = str(context.get("evidence", ""))
        low_evidence = evidence.casefold()
        cues = [str(item).casefold() for item in conditions.get("lexical_cues", []) or []]
        if cues and not all(cue in low_evidence for cue in cues):
            return False, []
        if cues:
            reasons.append("lexical_cues_matched")
        forbidden = [str(item).casefold() for item in conditions.get("forbidden_cues", []) or []]
        if any(cue in low_evidence for cue in forbidden):
            return False, []
        flags = set(context.get("quality_flags", []) or [])
        required_flags = set(conditions.get("quality_flags", []) or [])
        if required_flags and not required_flags.issubset(flags):
            return False, []
        confidence = float(context.get("evidence_confidence", 0.0) or 0.0)
        if confidence < float(conditions.get("min_evidence_confidence", 0.0) or 0.0):
            return False, []
        if confidence > float(conditions.get("max_evidence_confidence", 1.0) or 1.0):
            return False, []
        if conditions.get("requires_both_endpoints") and not context.get("both_endpoints_in_evidence"):
            return False, []
        return True, reasons

    def retrieve(self, context: dict[str, Any]) -> list[RuleMatch]:
        if self.mode == "off" or self.load_error:
            return []
        matches: list[RuleMatch] = []
        allowed_status = {"active"} if self.mode == "active" else {"active", "shadow"}
        for rule in self.bundle.rules:
            if rule.status not in allowed_status:
                continue
            matched, reasons = self._match(rule, context)
            if matched:
                specificity = sum(bool(value) for value in rule.conditions.values())
                matches.append(RuleMatch(
                    rule_id=rule.rule_id, kind=rule.kind, action=rule.action,
                    value=rule.value, guidance=rule.guidance,
                    specificity=specificity, reason_codes=reasons,
                ))
        matches.sort(key=lambda item: (-item.specificity, item.rule_id))
        return matches[:self.max_rules]

    def prompt_context(self, context: dict[str, Any]) -> tuple[str, list[RuleMatch]]:
        matches = [item for item in self.retrieve(context) if item.kind == "prompt_guidance"]
        lines: list[str] = []
        token_estimate = 0
        retained: list[RuleMatch] = []
        for item in matches:
            line = f"- [{item.rule_id}] {item.guidance.strip()}"
            cost = max(1, len(line) // 4)
            if token_estimate + cost > self.max_context_tokens:
                break
            lines.append(line)
            retained.append(item)
            token_estimate += cost
        if not lines:
            return "", retained
        return "Validated soft-rule guidance (cannot override verifier safety):\n" + "\n".join(lines), retained

    def phase_payload(self, matches: list[RuleMatch], *, applied: bool) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "bundle_version": self.bundle.bundle_version,
            "bundle_revision": self.bundle.revision,
            "bundle_hash": self.frozen_hash,
            "bundle_frozen_for_run": True,
            "load_error": self.load_error,
            "retrieved_rule_count": len(matches),
            "matches": [item.to_dict() for item in matches],
            "applied": bool(applied and self.mode == "active"),
        }


def merge_candidate_rules(rules: Iterable[SoftRule]) -> list[SoftRule]:
    """Deduplicate exact semantic rules and retain the most supported version."""
    grouped: dict[str, SoftRule] = {}
    for rule in rules:
        RuleValidator.validate(rule)
        key = RuleValidator.content_id(rule.to_dict())
        incumbent = grouped.get(key)
        if incumbent is None or len(set(rule.support_pmids)) > len(set(incumbent.support_pmids)):
            rule.rule_id = key
            grouped[key] = rule
    return [grouped[key] for key in sorted(grouped)]


def render_rule_context(bundle: RuleBundle) -> str:
    lines = [
        "# Validated Agent v3 rule context", "",
        f"- Bundle revision: `{bundle.revision}`",
        f"- Bundle hash: `{bundle.bundle_hash}`",
        f"- Active rules: {sum(rule.status == 'active' for rule in bundle.rules)}", "",
        "These rules are soft guidance. They cannot override deterministic verifier or Safe Write gates.", "",
    ]
    for rule in sorted(bundle.rules, key=lambda item: (item.status, item.rule_id)):
        lines.extend([
            f"## {rule.rule_id} — {rule.status}", "",
            f"- Kind/action: `{rule.kind}` / `{rule.action}`",
            f"- Conditions: `{json.dumps(rule.conditions, ensure_ascii=False, sort_keys=True)}`",
            f"- Guidance/value: {rule.guidance or rule.value}",
            f"- Support documents: {len(set(rule.support_pmids))}", "",
        ])
    return "\n".join(lines).rstrip() + "\n"
