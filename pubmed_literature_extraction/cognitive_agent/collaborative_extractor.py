#!/usr/bin/env python3
"""Bounded second-model adjudication for Phase B.

The second model is an edit-only judge.  It cannot invent endpoints, entities,
or relations.  It receives a compact list of grounded candidate triples and
may keep, reject, or edit a predicate/direction/evidence selection.  Every edit
is re-verified by the deterministic Phase-A verifier.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from cognitive_agent.evidence_units import ArticleEvidenceReader, EvidenceUnit
from cognitive_agent.extraction_quality import normalize_surface
from cognitive_agent.relation_contract import (
    ACTIVITY_CHANGES,
    ASSOCIATION_SIGNS,
    EXPRESSION_CHANGES,
    RELATION_DIRECTIONS,
    normalize_relation_semantics,
    stable_candidate_id,
    verification_policy as get_verification_policy,
)
from cognitive_agent.schema.entity_classes import ENTITY_CLASSES
from cognitive_agent.schema.predicate_cards import relation_description
from cognitive_agent.schema.relation_signatures import RELATION_SIGNATURES


ENTITY_TYPES = frozenset(item["label"] for item in ENTITY_CLASSES.values())
PREDICATES = frozenset(RELATION_SIGNATURES)
MAX_REVIEW_CANDIDATES = 20
MAX_EVIDENCE_UNITS = 40

DECISION_ACTIONS = frozenset({
    "KEEP", "REJECT", "CHANGE_PREDICATE", "CHANGE_DIRECTION", "CHANGE_EVIDENCE",
    "ADD_RELATION",
})
REASON_CODES = frozenset({
    "EXPLICIT_DIRECT_RELATION", "CO_OCCURRENCE_ONLY", "BACKGROUND_ONLY",
    "METHOD_OR_PREDICTION_ONLY", "TYPE_SIGNATURE_MISMATCH", "WRONG_PREDICATE",
    "WRONG_DIRECTION", "EVIDENCE_MISMATCH", "UNCERTAIN_OR_HEDGED",
    "INSUFFICIENT_SUPPORT",
})

# These flags indicate that a candidate cannot become a current-article fact by
# semantic reinterpretation.  Keeping them in the final relation list inflated
# candidate metrics and repeatedly triggered the expensive reviewer.
# Only claims that are not valid current-article semantics are pruned.  Study
# scope, species and linking ambiguity are Safe Write concerns and must remain
# available as semantic-only/review relations.
ADJUDICATION_VERDICTS = frozenset({
    "SUPPORTED", "AMBIGUOUS", "UNSUPPORTED", "CONTRADICTED",
})

# Same-type associations and interactions are semantic unordered pairs.  This
# affects candidate consolidation only; it does not alter the frozen main-KG
# write contract or directional predicates such as PROGRESSES_TO.
UNDIRECTED_PREDICATES = frozenset({"ASSOCIATED_WITH", "INTERACTS_WITH"})

# Recovery has only one-model support (the first extractor omitted the edge),
# so its action space is narrower than ordinary adjudication.  Broad
# association predicates produced almost all false-positive gains in the
# 50-document counterfactual audit; keep recovery for predicates with a
# specific lexical/semantic trigger and leave broad associations to the
# two-model candidate-review path.
RECOVERABLE_PREDICATES = frozenset({
    "ENCODES", "PARTICIPATES_IN", "INTERACTS_WITH", "EXPRESSED_IN",
    "PROGNOSTIC_IN", "PROGRESSES_TO",
})

COLLABORATION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["review_decisions", "review_reason"],
    "properties": {
        "review_decisions": {
            "type": "array",
            "maxItems": MAX_REVIEW_CANDIDATES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "candidate_id", "action", "new_predicate", "new_direction",
                    "evidence_unit_id", "swap_endpoints", "reason_code", "reason", "confidence",
                ],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "candidate_version": {"type": "integer", "minimum": 1},
                    "verdict": {"type": "string", "enum": sorted(ADJUDICATION_VERDICTS)},
                    "action": {"type": "string", "enum": sorted(DECISION_ACTIONS)},
                    "new_predicate": {"type": "string"},
                    "new_direction": {"type": "string"},
                    "relation_direction": {"type": "string", "enum": sorted(RELATION_DIRECTIONS)},
                    "association_sign": {"type": "string", "enum": sorted(ASSOCIATION_SIGNS)},
                    "expression_change": {"type": "string", "enum": sorted(EXPRESSION_CHANGES)},
                    "activity_change": {"type": "string", "enum": sorted(ACTIVITY_CHANGES)},
                    "claim_role": {"type": "string"},
                    "supporting_span_ids": {"type": "array", "items": {"type": "string"}},
                    "evidence_unit_id": {"type": "string"},
                    "swap_endpoints": {"type": "boolean"},
                    "reason_code": {"type": "string", "enum": sorted(REASON_CODES)},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "review_reason": {"type": "string"},
    },
}


@dataclass(frozen=True)
class CollaborativeConfig:
    enabled: bool = False
    provider: str = "openai"
    api_key: str = ""
    api_base: str = "https://api.deepseek.com"
    model_id: str = "deepseek-v4-flash"
    mode: str = "conditional"
    timeout: float = 45.0
    max_output_tokens: int | None = None
    thinking_enabled: bool = False
    max_review_candidates: int = MAX_REVIEW_CANDIDATES
    generic_rate_threshold: float = 0.20
    duplicate_rate_threshold: float = 0.20
    evidence_failure_threshold: float = 0.40
    # Round-4: the second model is an INDEPENDENT VERIFIER of already-positive,
    # semantically-uncertain relations (CONFIRM / REJECT / REVIEW).  It must
    # NOT add new relations on its own; recovery-candidate ADD_RELATION is
    # disabled unless explicitly re-enabled.
    add_relations: bool = False
    verification_policy: str = "legacy"


@dataclass
class CollaborationResult:
    status: str = "DISABLED"
    triggered: bool = False
    trigger_reasons: list[str] = field(default_factory=list)
    review_decisions: list[dict] = field(default_factory=list)
    review_candidates: list[dict] = field(default_factory=list)
    recovery_candidates: list[dict] = field(default_factory=list)
    review_reason: str = ""
    parse_warnings: list[str] = field(default_factory=list)
    error: str = ""
    model_id: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    invalid_json_attempts: int = 0
    critic_audit: dict = field(default_factory=dict)
    # Deprecated compatibility fields.  They intentionally remain empty.
    corrected_entities: list[dict] = field(default_factory=list)
    corrected_relations: list[dict] = field(default_factory=list)
    rejected_candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {key: value for key, value in self.__dict__.items()}


@dataclass
class CollaborationMerge:
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    entity_additions: int = 0
    relation_additions: int = 0
    relation_edits: int = 0
    relation_rejections: int = 0
    deterministic_rejections: list[dict] = field(default_factory=list)
    recovered_relations: list[dict] = field(default_factory=list)
    recovery_rejections: list[dict] = field(default_factory=list)
    agreements: list[dict] = field(default_factory=list)
    manual_review: list[dict] = field(default_factory=list)
    second_model_rejections: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entity_additions": 0,
            "relation_additions": self.relation_additions,
            "relation_edits": self.relation_edits,
            "relation_rejections": self.relation_rejections,
            "deterministic_rejection_count": len(self.deterministic_rejections),
            "deterministic_rejections": self.deterministic_rejections,
            "recovered_relations": self.recovered_relations,
            "recovery_rejections": self.recovery_rejections,
            "agreements": self.agreements,
            "manual_review": self.manual_review,
            "second_model_rejections": self.second_model_rejections,
        }


@dataclass
class _ModelResponse:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""


class CollaborativeExtractor:
    """Compact relation adjudicator with deterministic fallback."""

    def __init__(
        self,
        config: CollaborativeConfig | None = None,
        generate: Optional[Callable[[str], Any]] = None,
    ):
        self.config = config or CollaborativeConfig()
        self.policy = get_verification_policy(self.config.verification_policy)
        self._generate = generate
        self.reader = ArticleEvidenceReader()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and (self._generate or (self.config.api_key and self.config.model_id)))

    @staticmethod
    def _score(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _relation_key(item: dict) -> tuple:
        return (
            normalize_surface(item.get("subject", "")),
            str(item.get("subject_type", "")),
            str(item.get("predicate", "")).upper(),
            normalize_surface(item.get("object", "")),
            str(item.get("object_type", "")),
            str(item.get("evidence", "") or ""),
        )

    @staticmethod
    def _allowed_predicates(subject_type: str, object_type: str, current: str) -> list[str]:
        allowed = [
            predicate for predicate, signatures in RELATION_SIGNATURES.items()
            if (subject_type, object_type) in signatures
        ]
        current = str(current or "").upper()
        if current in allowed:
            allowed.remove(current)
            allowed.insert(0, current)
        return allowed[:4]

    def build_review_candidates(self, verification: dict) -> list[dict]:
        candidates: list[dict] = []
        for raw_index, relation in enumerate(verification.get("relations", []) or []):
            flags = set(relation.get("quality_flags", []) or [])
            if str(relation.get("factual_status", "VALID")).upper() == "REJECTED":
                continue
            if not relation.get("schema_valid", True):
                continue
            allowed = self._allowed_predicates(
                str(relation.get("subject_type", "")),
                str(relation.get("object_type", "")),
                str(relation.get("predicate", "")),
            )
            if not allowed:
                continue
            predicate = str(relation.get("predicate", "") or "").upper()
            evidence = str(relation.get("evidence", "") or "")
            fixable_flags = {
                "trigger_missing", "trigger_not_linking_endpoints",
                "trigger_direction_mismatch", "weak_evidence", "uncertain",
                "agent_evidence_repaired", "pair_low_confidence",
                "pair_ambiguous_predicate", "judge_uncertain",
                "predicate_card_type_only",
            }
            high_risk_predicate = predicate in {
                "PROGNOSTIC_IN", "INTERACTS_WITH", "EXPRESSED_IN", "PROGRESSES_TO",
            }
            hedged_or_proxy = bool(re.search(
                r"\b(?:potential|candidate|implicated|may|might|could|suggest\w*|"
                r"target for|diagnos\w*|treatment)\b",
                evidence,
                re.IGNORECASE,
            ))
            # Semantic uncertainty is precisely what the bounded second model
            # exists for: hard blockers stay local, everything uncertain may
            # be adjudicated.  The old pair-core gate starved the adjudicator
            # and cost more precision than the calls it saved.
            non_current = str(
                relation.get("claim_role", "CURRENT_FINDING") or "CURRENT_FINDING"
            ).upper() != "CURRENT_FINDING"
            non_role_fixable = flags & fixable_flags
            if non_current and not non_role_fixable:
                continue
            if not (non_role_fixable or high_risk_predicate or hedged_or_proxy):
                continue
            candidate_id = str(relation.get("candidate_id", "") or stable_candidate_id(relation))
            candidates.append({
                "candidate_id": candidate_id,
                "candidate_version": max(1, int(relation.get("candidate_version", 1) or 1)),
                "pair_candidate_id": relation.get("pair_candidate_id", ""),
                "raw_index": raw_index,
                "subject": relation.get("subject", ""),
                "subject_type": relation.get("subject_type", ""),
                "predicate": predicate,
                "object": relation.get("object", ""),
                "object_type": relation.get("object_type", ""),
                "direction": relation.get("direction", "unknown"),
                "relation_direction": relation.get("relation_direction", "UNKNOWN"),
                "association_sign": relation.get("association_sign", "UNKNOWN"),
                "expression_change": relation.get("expression_change", "UNKNOWN"),
                "activity_change": relation.get("activity_change", "UNKNOWN"),
                "claim_role": relation.get("claim_role", "CURRENT_FINDING"),
                "evidence": relation.get("evidence", ""),
                "evidence_pack": relation.get("evidence_pack", {}),
                "import_ready": bool(relation.get("import_ready")),
                "quality_flags": sorted(flags),
                "classifier_confidence": relation.get("classifier_confidence"),
                "relation_probability": relation.get("relation_probability"),
                "no_relation_probability": relation.get("no_relation_probability"),
                "classifier_margin": relation.get("classifier_margin"),
                "classifier_source": relation.get("classifier_source", ""),
                "evidence_unit_id": relation.get("evidence_unit_id", ""),
                "allowed_predicates": [
                    {
                        "predicate": pred,
                        "meaning": relation_description(
                            pred,
                            str(relation.get("subject_type", "")),
                            str(relation.get("object_type", "")),
                        ),
                    }
                    for pred in allowed
                ],
            })
        candidates.sort(key=lambda item: (
            not item["import_ready"],
            len(item["quality_flags"]),
            item["raw_index"],
        ))
        return candidates[: max(1, int(self.config.max_review_candidates))]

    def trigger_reasons(
        self,
        extraction: dict,
        verification: dict,
        context: dict | None = None,
    ) -> list[str]:
        reasons: list[str] = []
        if extraction.get("error"):
            reasons.append("first_extractor_error_quarantine")
        if any("parse" in str(item).casefold() for item in extraction.get("warnings", []) or []):
            reasons.append("relation_or_entity_parse_failure")
        if self.build_review_candidates(verification):
            reasons.append("grounded_relation_semantic_adjudication")
        return reasons

    def collaborate(
        self,
        text: str,
        extraction: dict,
        verification: dict,
        pmid: str = "",
        context: dict | None = None,
        rag_context: dict | None = None,
        router_reasons: list[str] | None = None,
        recovery_candidates: list[dict] | None = None,
        pair_review_candidates: list[dict] | None = None,
    ) -> CollaborationResult:
        reasons = list(dict.fromkeys([
            *self.trigger_reasons(extraction, verification, context),
            *(router_reasons or []),
        ]))
        review_candidates = self.build_review_candidates(verification)
        # The explicit list is an audit/defence-in-depth channel.  In active
        # mode these candidates should already be present in verification; do
        # not duplicate or allow an unverified shadow candidate into merging.
        explicit_pair_ids = {
            str(item.get("candidate_id", "")) for item in (pair_review_candidates or [])
            if item.get("candidate_id")
        }
        if explicit_pair_ids:
            review_candidates = [
                item for item in review_candidates
                if str(item.get("candidate_id", "")) in explicit_pair_ids
                or "pair_low_confidence" in set(item.get("quality_flags", []))
            ]
        bounded_recovery: list[dict] = []
        for item in (recovery_candidates or [])[:MAX_REVIEW_CANDIDATES]:
            candidate = copy.deepcopy(item)
            candidate["candidate_kind"] = "recovery"
            candidate["allowed_predicates"] = [
                {
                    "predicate": str(value.get("predicate", "") if isinstance(value, dict) else value).upper(),
                    "meaning": relation_description(
                        str(value.get("predicate", "") if isinstance(value, dict) else value).upper(),
                        str(candidate.get("subject_type", "")),
                        str(candidate.get("object_type", "")),
                    ),
                }
                for value in candidate.get("allowed_predicates", [])
                if str(value.get("predicate", "") if isinstance(value, dict) else value).upper()
                in RECOVERABLE_PREDICATES
            ]
            if candidate["allowed_predicates"]:
                bounded_recovery.append(candidate)
        # Reserve capacity for both tasks so a noisy first-pass relation list
        # cannot starve the entity-first recovery lattice (or vice versa).
        review_budget = 12
        recovery_budget = MAX_REVIEW_CANDIDATES - review_budget
        review_candidates = review_candidates[:review_budget]
        bounded_recovery = bounded_recovery[:recovery_budget]
        candidates = [*review_candidates, *bounded_recovery]
        if not self.enabled:
            return CollaborationResult(status="DISABLED", trigger_reasons=reasons, model_id=self.config.model_id)
        if extraction.get("error") and not candidates:
            return CollaborationResult(
                status="RECOVERY_QUARANTINED", triggered=False, trigger_reasons=reasons,
                review_reason="Primary extraction failed; open generation is disabled and the article is quarantined.",
                model_id=self.config.model_id,
            )
        if not candidates:
            return CollaborationResult(
                status="NOT_TRIGGERED", triggered=False, trigger_reasons=reasons,
                review_reason="No grounded, schema-compatible relation requires semantic adjudication.",
                model_id=self.config.model_id,
            )

        units = self.reader.read(text)
        prompt = self._build_prompt(text, candidates, units, reasons, pmid, rag_context or {})
        started = time.perf_counter()
        try:
            response = self._call_model(prompt)
            payload = self._parse_json(response.text)
            decisions, warnings = self._normalize_decisions(payload, candidates, units)
            return CollaborationResult(
                status="OK", triggered=True, trigger_reasons=reasons,
                review_decisions=decisions, review_candidates=candidates,
                recovery_candidates=bounded_recovery,
                review_reason=str(payload.get("review_reason", "") or "")[:1500],
                parse_warnings=warnings, model_id=self.config.model_id,
                latency_s=round(time.perf_counter() - started, 4),
                prompt_tokens=response.prompt_tokens, output_tokens=response.output_tokens,
                total_tokens=response.total_tokens, finish_reason=response.finish_reason,
            )
        except Exception as exc:
            error = str(exc)
            if self.config.api_key:
                error = error.replace(self.config.api_key, "[REDACTED]")
            return CollaborationResult(
                status="FALLBACK", triggered=True, trigger_reasons=reasons,
                review_candidates=candidates, recovery_candidates=bounded_recovery,
                review_reason="Second-model adjudication failed; keep deterministic Phase-A results.",
                error=error[:500], model_id=self.config.model_id,
                latency_s=round(time.perf_counter() - started, 4),
                invalid_json_attempts=int("json" in error.casefold()),
            )

    def _call_model(self, prompt: str) -> _ModelResponse:
        if self._generate is not None:
            value = self._generate(prompt)
            if isinstance(value, _ModelResponse):
                return value
            return _ModelResponse(text=value if isinstance(value, str) else json.dumps(value))

        if self.config.provider == "openai":
            from openai import OpenAI

            client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base or None,
                timeout=self.config.timeout,
            )
            request: dict[str, Any] = {
                "model": self.config.model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a precision-first biomedical relation adjudicator. "
                            "Return one strict JSON object only. Never create an entity, endpoint, "
                            "evidence span, or open-vocabulary predicate. ADD_RELATION is allowed "
                            "only for a supplied recovery candidate. The JSON must satisfy this schema: "
                            + json.dumps(COLLABORATION_JSON_SCHEMA, ensure_ascii=False)
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            api_host = str(self.config.api_base or "").casefold()
            if "aliyuncs.com" in api_host:
                # Alibaba Model Studio's OpenAI-compatible endpoint uses the
                # Qwen `enable_thinking` switch rather than DeepSeek's nested
                # `thinking.type` object.
                request["extra_body"] = {
                    "enable_thinking": bool(self.config.thinking_enabled)
                }
            else:
                request["extra_body"] = {
                    "thinking": {
                        "type": "enabled" if self.config.thinking_enabled else "disabled"
                    }
                }
            if self.config.max_output_tokens is not None:
                request["max_tokens"] = self.config.max_output_tokens
            response = client.chat.completions.create(**request)
            usage = getattr(response, "usage", None)
            choice = response.choices[0]
            return _ModelResponse(
                text=str(choice.message.content or ""),
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            )

        if self.config.provider != "gemini":
            raise ValueError(f"unsupported second LLM provider: {self.config.provider}")

        from google import genai
        from google.genai import types

        http_options: dict[str, Any] = {"timeout": int(self.config.timeout * 1000)}
        if self.config.api_base:
            http_options["base_url"] = self.config.api_base
        client = genai.Client(api_key=self.config.api_key, http_options=http_options)
        generation_config: dict[str, Any] = {
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_json_schema": COLLABORATION_JSON_SCHEMA,
        }
        if self.config.max_output_tokens is not None:
            generation_config["max_output_tokens"] = self.config.max_output_tokens
        response = client.models.generate_content(
            model=self.config.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(**generation_config),
        )
        usage = getattr(response, "usage_metadata", None)
        return _ModelResponse(
            text=str(getattr(response, "text", response) or ""),
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            total_tokens=int(getattr(usage, "total_token_count", 0) or 0),
        )

    @staticmethod
    def _parse_json(raw: Any) -> dict:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            raise ValueError("second model returned a non-text response")
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("second model JSON must be an object")
        return payload

    def _normalize_decisions(
        self,
        payload: dict,
        candidates: list[dict],
        units: list[EvidenceUnit],
    ) -> tuple[list[dict], list[str]]:
        warnings: list[str] = []
        candidate_map = {item["candidate_id"]: item for item in candidates}
        unit_map = {item.unit_id: item for item in units}
        unit_ids = set(unit_map)
        decisions: list[dict] = []
        seen: set[str] = set()
        value = payload.get("review_decisions", [])
        if not isinstance(value, list):
            return [], ["review_decisions_not_array"]
        for item in value[:MAX_REVIEW_CANDIDATES]:
            if not isinstance(item, dict):
                warnings.append("invalid_review_decision_ignored")
                continue
            candidate_id = str(item.get("candidate_id", "") or "")
            action = str(item.get("action", "") or "").upper()
            candidate = candidate_map.get(candidate_id)
            verdict = str(item.get("verdict", "") or "").upper()
            if verdict not in ADJUDICATION_VERDICTS:
                verdict = (
                    "UNSUPPORTED" if action == "REJECT"
                    else "SUPPORTED" if action in {"KEEP", "CHANGE_PREDICATE", "CHANGE_DIRECTION", "CHANGE_EVIDENCE"}
                    else "AMBIGUOUS"
                )
            if not candidate or candidate_id in seen or action not in DECISION_ACTIONS:
                warnings.append("invalid_review_decision_ignored")
                continue
            new_predicate = str(item.get("new_predicate", "") or "").upper()
            allowed = {entry["predicate"] for entry in candidate["allowed_predicates"]}
            candidate_kind = str(candidate.get("candidate_kind", "review"))
            if candidate_kind == "recovery" and action not in {"ADD_RELATION", "REJECT"}:
                warnings.append("invalid_recovery_action_ignored")
                continue
            if action == "ADD_RELATION" and not self.config.add_relations:
                # Round-4: the second model is an independent verifier; it
                # cannot add new relations.  Recovery ADD_RELATION is disabled.
                warnings.append("second_llm_relation_addition_disabled")
                continue
            if candidate_kind != "recovery" and action == "ADD_RELATION":
                warnings.append("open_relation_addition_ignored")
                continue
            if action in {"CHANGE_PREDICATE", "ADD_RELATION"} and new_predicate not in allowed:
                warnings.append("predicate_outside_shortlist_ignored")
                continue
            if action == "ADD_RELATION" and new_predicate not in RECOVERABLE_PREDICATES:
                warnings.append("broad_recovery_predicate_ignored")
                continue
            new_direction = str(item.get("new_direction", "") or "").lower()
            if action == "CHANGE_DIRECTION" and new_direction not in {
                "positive", "negative", "increase", "decrease", "none", "unknown",
            }:
                warnings.append("invalid_direction_edit_ignored")
                continue
            evidence_unit_id = str(item.get("evidence_unit_id", "") or "")
            if action == "CHANGE_EVIDENCE" and evidence_unit_id not in unit_ids:
                warnings.append("invalid_evidence_unit_ignored")
                continue
            swap_endpoints = bool(item.get("swap_endpoints", False)) if candidate_kind == "recovery" else False
            if (
                action == "ADD_RELATION"
                and swap_endpoints
                and (
                    str(candidate.get("object_type", "")),
                    str(candidate.get("subject_type", "")),
                ) not in RELATION_SIGNATURES.get(new_predicate, set())
            ):
                warnings.append("invalid_endpoint_swap_ignored")
                continue
            reason_code = str(item.get("reason_code", "") or "").upper()
            if reason_code not in REASON_CODES:
                reason_code = "INSUFFICIENT_SUPPORT"
            decisions.append({
                "candidate_id": candidate_id,
                "candidate_version": max(1, int(candidate.get("candidate_version", 1) or 1)),
                "raw_index": candidate["raw_index"],
                "pair_candidate_id": candidate.get("pair_candidate_id", ""),
                "candidate_kind": candidate_kind,
                "action": action,
                "verdict": verdict,
                "new_predicate": new_predicate,
                "new_direction": new_direction,
                "relation_direction": str(item.get("relation_direction", "") or candidate.get("relation_direction", "UNKNOWN")).upper(),
                "association_sign": str(item.get("association_sign", "") or candidate.get("association_sign", "UNKNOWN")).upper(),
                "expression_change": str(item.get("expression_change", "") or candidate.get("expression_change", "UNKNOWN")).upper(),
                "activity_change": str(item.get("activity_change", "") or candidate.get("activity_change", "UNKNOWN")).upper(),
                "claim_role": str(item.get("claim_role", "") or candidate.get("claim_role", "CURRENT_FINDING")).upper(),
                "supporting_span_ids": list(item.get("supporting_span_ids", []) or []),
                "evidence_unit_id": evidence_unit_id,
                "evidence_text": (
                    unit_map[evidence_unit_id].text
                    if action == "CHANGE_EVIDENCE" and evidence_unit_id in unit_map else ""
                ),
                "swap_endpoints": swap_endpoints,
                "reason_code": reason_code,
                "reason": str(item.get("reason", "") or "")[:800],
                "confidence": self._score(item.get("confidence", 0.0)),
            })
            seen.add(candidate_id)
        missing = set(candidate_map) - seen
        if missing:
            warnings.append(f"missing_decisions_kept:{len(missing)}")
        return decisions, warnings

    def merge(
        self,
        raw_entities: list[dict],
        raw_relations: list[dict],
        initial_verification: dict,
        collaboration: CollaborationResult,
    ) -> CollaborationMerge:
        merged = CollaborationMerge(entities=copy.deepcopy(raw_entities))
        verified_relations = initial_verification.get("relations", []) or []
        verified_by_lineage = {
            (
                str(item.get("candidate_id", "") or ""),
                max(1, int(item.get("candidate_version", 1) or 1)),
            ): item
            for item in verified_relations
            if str(item.get("candidate_id", "") or "")
        }
        decision_map = {
            (
                str(item.get("candidate_id", "") or ""),
                max(1, int(item.get("candidate_version", 1) or 1)),
            ): item
            for item in collaboration.review_decisions
            if str(item.get("candidate_id", "") or "")
        }
        legacy_decision_by_index = {
            int(item.get("raw_index", -1)): item
            for item in collaboration.review_decisions
            if int(item.get("raw_index", -1)) >= 0
            and not any(
                key[0] == str(item.get("candidate_id", "") or "")
                for key in verified_by_lineage
            )
        }
        for index, raw_relation in enumerate(raw_relations):
            relation = normalize_relation_semantics(copy.deepcopy(raw_relation))
            candidate_id = str(relation.get("candidate_id", "") or stable_candidate_id(
                relation, lane=str(relation.get("candidate_lane", "extracted_hint") or "extracted_hint"),
            ))
            candidate_version = max(1, int(relation.get("candidate_version", 1) or 1))
            relation.update({
                "candidate_id": candidate_id,
                "candidate_version": candidate_version,
                "parent_version": max(0, int(relation.get("parent_version", 0) or 0)),
            })
            verified = verified_by_lineage.get(
                (candidate_id, candidate_version),
                verified_relations[index] if index < len(verified_relations) else {},
            )
            flags = set(verified.get("quality_flags", []) or [])
            if str(verified.get("factual_status", "VALID")).upper() == "REJECTED":
                merged.deterministic_rejections.append({
                    "raw_index": index,
                    "candidate_id": candidate_id,
                    "candidate_version": candidate_version,
                    "subject": raw_relation.get("subject", ""),
                    "predicate": raw_relation.get("predicate", ""),
                    "object": raw_relation.get("object", ""),
                    "reason_codes": list(verified.get("semantic_reasons", []) or sorted(
                        self.policy.hard_reject_reasons(flags)
                    )),
                })
                continue

            decision = decision_map.get((candidate_id, candidate_version))
            if decision is None:
                # Read-only compatibility for historical cached adjudications.
                # New outputs always use stable candidate_id/version lineage.
                decision = legacy_decision_by_index.get(index)
            if not decision:
                merged.relations.append(relation)
                continue
            action = decision["action"]
            if action == "REJECT":
                rejection = {
                    **decision,
                    "subject": verified.get("subject", relation.get("subject", "")),
                    "predicate": verified.get("predicate", relation.get("predicate", "")),
                    "object": verified.get("object", relation.get("object", "")),
                }
                merged.relation_rejections += 1
                merged.second_model_rejections.append(rejection)
                merged.manual_review.append(rejection)
                relation.setdefault("quality_flags", []).append("second_llm_rejected")
                audit_decision = copy.deepcopy(decision)
                audit_decision["model_id"] = collaboration.model_id
                relation["adjudication"] = audit_decision
                relation["adjudication_model_id"] = collaboration.model_id
                relation["adjudication_verdict"] = decision.get("verdict", "UNSUPPORTED")
                merged.relations.append(relation)
                continue
            changed = action in {
                "CHANGE_PREDICATE", "CHANGE_DIRECTION", "CHANGE_EVIDENCE",
            }
            if changed:
                relation["parent_candidate_id"] = candidate_id
                relation["parent_version"] = candidate_version
                relation["candidate_version"] = candidate_version + 1
                relation["edit_reason_code"] = str(
                    decision.get("reason_code", "") or action
                )
            if action == "CHANGE_PREDICATE":
                relation["predicate"] = decision["new_predicate"]
                relation.setdefault("quality_flags", []).append("second_llm_edited")
                relation["collaboration_reason"] = decision["reason"]
                merged.relation_edits += 1
            elif action == "CHANGE_DIRECTION":
                relation["direction"] = decision["new_direction"]
                relation.setdefault("quality_flags", []).append("second_llm_edited")
                relation["collaboration_reason"] = decision["reason"]
                merged.relation_edits += 1
            elif action == "CHANGE_EVIDENCE":
                relation["evidence"] = decision.get("evidence_text", relation.get("evidence", ""))
                relation.setdefault("quality_flags", []).append("second_llm_evidence_review")
                relation["collaboration_reason"] = decision["reason"]
                merged.relation_edits += 1
            else:
                # A bounded CoRE adjudication may clear classifier uncertainty,
                # but never deterministic evidence/schema blockers.  Low model
                # confidence keeps the manual-review barrier in place.
                if (
                    "pair_low_confidence" in set(relation.get("quality_flags", []) or [])
                    and float(decision.get("confidence", 0.0) or 0.0) >= 0.75
                    and decision.get("reason_code") == "EXPLICIT_DIRECT_RELATION"
                ):
                    relation["quality_flags"] = [
                        flag for flag in relation.get("quality_flags", [])
                        if flag not in {
                            "pair_low_confidence", "pair_ambiguous_predicate", "manual_review",
                        }
                    ]
                    relation.setdefault("quality_flags", []).append("second_llm_confirmed")
                    relation["collaboration_reason"] = decision.get("reason", "")
                merged.agreements.append({
                    "candidate_type": "relation",
                    "subject": relation.get("subject", ""),
                    "predicate": relation.get("predicate", ""),
                    "object": relation.get("object", ""),
                })
            audit_decision = copy.deepcopy(decision)
            audit_decision["model_id"] = collaboration.model_id
            relation.update({
                "relation_direction": decision.get("relation_direction", relation.get("relation_direction", "UNKNOWN")),
                "association_sign": decision.get("association_sign", relation.get("association_sign", "UNKNOWN")),
                "expression_change": decision.get("expression_change", relation.get("expression_change", "UNKNOWN")),
                "activity_change": decision.get("activity_change", relation.get("activity_change", "UNKNOWN")),
                "claim_role": decision.get("claim_role", relation.get("claim_role", "CURRENT_FINDING")),
                "adjudication_verdict": decision.get("verdict", "SUPPORTED"),
                "supporting_span_ids": list(decision.get("supporting_span_ids", []) or []),
                "adjudication": audit_decision,
                "adjudication_reason_code": decision.get("reason_code", ""),
                "adjudication_confidence": float(decision.get("confidence", 0.0) or 0.0),
                "adjudication_model_id": collaboration.model_id,
            })
            if decision.get("verdict") == "AMBIGUOUS":
                relation.setdefault("quality_flags", []).extend(["manual_review", "adjudicator_ambiguous"])
            elif decision.get("verdict") == "SUPPORTED":
                relation.setdefault("quality_flags", []).append("adjudicator_entailed")
            relation = normalize_relation_semantics(relation)
            merged.relations.append(relation)

        existing_keys = {self._relation_key(item) for item in merged.relations}
        recovery_map = {
            item.get("candidate_id"): item for item in collaboration.recovery_candidates
        }
        for decision in collaboration.review_decisions:
            if decision.get("candidate_kind") != "recovery":
                continue
            candidate = recovery_map.get(decision.get("candidate_id"))
            if not candidate:
                continue
            if decision.get("action") == "REJECT":
                merged.recovery_rejections.append({
                    **decision,
                    "subject": candidate.get("subject", ""),
                    "predicate": "NONE",
                    "object": candidate.get("object", ""),
                })
                continue
            if decision.get("action") != "ADD_RELATION":
                continue
            relation = {
                "subject": (
                    candidate.get("object", "")
                    if decision.get("swap_endpoints") else candidate.get("subject", "")
                ),
                "subject_type": (
                    candidate.get("object_type", "")
                    if decision.get("swap_endpoints") else candidate.get("subject_type", "")
                ),
                "predicate": decision.get("new_predicate", ""),
                "object": (
                    candidate.get("subject", "")
                    if decision.get("swap_endpoints") else candidate.get("object", "")
                ),
                "object_type": (
                    candidate.get("subject_type", "")
                    if decision.get("swap_endpoints") else candidate.get("object_type", "")
                ),
                "direction": decision.get("new_direction") or candidate.get("direction", "unknown"),
                "negated": False,
                "uncertain": False,
                "evidence": candidate.get("evidence", ""),
                "confidence": decision.get("confidence", 0.7),
                "grounded": True,
                "quality_flags": ["agent_recovered_relation"],
                "candidate_lane": "recovery",
                "candidate_version": 1,
                "parent_version": 0,
                "collaboration_reason": decision.get("reason", ""),
            }
            relation["candidate_id"] = str(
                candidate.get("candidate_id", "") or stable_candidate_id(relation, lane="recovery")
            )
            relation.update({
                "relation_direction": decision.get("relation_direction", "UNKNOWN"),
                "association_sign": decision.get("association_sign", "UNKNOWN"),
                "expression_change": decision.get("expression_change", "UNKNOWN"),
                "activity_change": decision.get("activity_change", "UNKNOWN"),
                "claim_role": decision.get("claim_role", candidate.get("claim_role", "CURRENT_FINDING")),
                "adjudication_verdict": decision.get("verdict", "SUPPORTED"),
                "supporting_span_ids": list(decision.get("supporting_span_ids", []) or []),
                "adjudication_reason_code": decision.get("reason_code", ""),
                "adjudication_confidence": float(decision.get("confidence", 0.0) or 0.0),
                "adjudication_model_id": collaboration.model_id,
                "adjudication": {
                    **copy.deepcopy(decision),
                    "model_id": collaboration.model_id,
                },
            })
            relation = normalize_relation_semantics(relation)
            key = self._relation_key(relation)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            merged.relations.append(relation)
            merged.relation_additions += 1
            merged.recovered_relations.append({
                **relation,
                "candidate_id": decision.get("candidate_id", ""),
                "reason_code": decision.get("reason_code", ""),
            })
        return merged

    def finalize_after_reverification(
        self,
        raw_relations: list[dict],
        verification: dict,
        source_text: str = "",
    ) -> tuple[list[dict], dict]:
        """Pure normalization/provenance aggregation; never re-judge a claim."""
        verified = verification.get("relations", []) or []
        checked_by_lineage = {
            (
                str(item.get("candidate_id", "") or ""),
                max(1, int(item.get("candidate_version", 1) or 1)),
            ): item
            for item in verified
            if str(item.get("candidate_id", "") or "")
        }
        mention_to_canonical = (
            verification.get("review", {}).get("mention_to_canonical", {}) or {}
        )
        verified_entities = verification.get("entities", []) or []
        canonical_aliases: dict[str, list[str]] = {}
        for mention, canonical_name in mention_to_canonical.items():
            canonical_aliases.setdefault(normalize_surface(canonical_name), []).append(str(mention))
        entity_type_rank: dict[tuple[str, str], tuple[int, float]] = {}
        for entity in verified_entities:
            key = (normalize_surface(entity.get("mention", "")), str(entity.get("type", "")))
            status = str(entity.get("neo4j_status", "") or "").upper()
            identifier = str(entity.get("neo4j_node_id", "") or entity.get("normalized_id", "") or "")
            rank = 0 if status == "EXACT_MATCH" else (1 if identifier else 2)
            value = (rank, -float(entity.get("confidence", 0.0) or 0.0))
            if key not in entity_type_rank or value < entity_type_rank[key]:
                entity_type_rank[key] = value

        def checked_for(index: int, relation: dict) -> dict:
            key = (
                str(relation.get("candidate_id", "") or ""),
                max(1, int(relation.get("candidate_version", 1) or 1)),
            )
            if key[0]:
                checked = checked_by_lineage.get(key)
                if checked is not None:
                    return checked
                return {
                    **relation,
                    "factual_status": "REVIEW",
                    "semantic_status": "REVIEW",
                    "write_status": "HUMAN_REVIEW",
                    "quality_flags": sorted(set([
                        *(relation.get("quality_flags", []) or []),
                        "lineage_binding_missing",
                    ])),
                    "semantic_reasons": ["lineage_binding_missing"],
                }
            # Historical rows in which neither side has an ID retain read-only
            # positional compatibility.  A partially identified live row does not.
            if index < len(verified) and not str(verified[index].get("candidate_id", "") or ""):
                return verified[index]
            return {}

        def canonical(value: str) -> str:
            return str(mention_to_canonical.get(value, value) or value)

        def explicit_type_rank(mention: str, entity_type: str) -> int:
            if not source_text or entity_type not in {"Gene", "Protein"}:
                return 1
            aliases = [mention, *canonical_aliases.get(normalize_surface(mention), [])]
            type_word = "protein" if entity_type == "Protein" else "gene"
            for alias in dict.fromkeys(str(item).strip() for item in aliases if str(item).strip()):
                escaped = re.escape(alias)
                if any(re.search(pattern, source_text) for pattern in (
                    rf"(?i)(?:{escaped})\s+{type_word}\b",
                    rf"(?i)\b{type_word}\s+(?:named\s+|called\s+)?(?:{escaped})\b",
                    rf"(?i)\b{type_word}\b[^.\n]{{0,60}}\(\s*{escaped}\s*\)",
                )):
                    return 0
            return 1

        def variant_rank(item: tuple[int, dict, dict]) -> tuple:
            index, relation, checked = item
            subject = canonical(str(checked.get("subject", relation.get("subject", "")) or ""))
            obj = canonical(str(checked.get("object", relation.get("object", "")) or ""))
            subject_type = str(checked.get("subject_type", relation.get("subject_type", "")) or "")
            object_type = str(checked.get("object_type", relation.get("object_type", "")) or "")
            return (
                0 if str(relation.get("candidate_lane", "")) == "extracted_hint" else 1,
                explicit_type_rank(subject, subject_type),
                entity_type_rank.get((normalize_surface(subject), subject_type), (3, 0.0)),
                explicit_type_rank(obj, object_type),
                entity_type_rank.get((normalize_surface(obj), object_type), (3, 0.0)),
                -int(relation.get("candidate_version", 1) or 1),
                str(relation.get("candidate_id", "") or ""),
                index,
            )

        def first_mention_position(evidence: str, canonical_name: str, raw_name: str) -> int:
            aliases = [raw_name, canonical_name, *canonical_aliases.get(normalize_surface(canonical_name), [])]
            positions = []
            for alias in dict.fromkeys(str(item).strip() for item in aliases if str(item).strip()):
                match = re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", evidence, re.IGNORECASE)
                if match:
                    positions.append(match.start())
            return min(positions, default=10**9)

        def schema_symmetric(predicate: str, subject_type: str, object_type: str) -> bool:
            signatures = RELATION_SIGNATURES.get(predicate, set())
            return (
                (subject_type, object_type) in signatures
                and (object_type, subject_type) in signatures
            )

        groups: dict[tuple, list[tuple[int, dict, dict]]] = {}
        factual_discards: list[dict] = []
        superseded_version_rows: list[dict] = []
        latest_version_by_lineage: dict[str, int] = {}
        for index, raw in enumerate(raw_relations):
            relation = normalize_relation_semantics(raw)
            checked = checked_for(index, relation)
            if str(checked.get("factual_status", "VALID")).upper() == "REJECTED":
                continue
            candidate_id = str(relation.get("candidate_id", "") or "")
            if candidate_id:
                latest_version_by_lineage[candidate_id] = max(
                    latest_version_by_lineage.get(candidate_id, 0),
                    int(relation.get("candidate_version", 1) or 1),
                )
        type_variants_by_untyped_edge: dict[tuple, set[tuple[str, str]]] = {}
        for index, raw in enumerate(raw_relations):
            relation = normalize_relation_semantics(raw)
            checked = checked_for(index, relation)
            if str(checked.get("factual_status", "VALID")).upper() == "REJECTED":
                continue
            subject = canonical(str(checked.get("subject", relation.get("subject", "")) or ""))
            obj = canonical(str(checked.get("object", relation.get("object", "")) or ""))
            predicate = str(checked.get("predicate", relation.get("predicate", "")) or "").upper()
            relation_direction = str(
                checked.get("relation_direction", relation.get("relation_direction", "UNKNOWN"))
                or "UNKNOWN"
            ).upper()
            subject_type = str(checked.get("subject_type", relation.get("subject_type", "")) or "")
            object_type = str(checked.get("object_type", relation.get("object_type", "")) or "")
            surfaces = (
                tuple(sorted((normalize_surface(subject), normalize_surface(obj))))
                if relation_direction in {"NON_DIRECTIONAL", "BIDIRECTIONAL"}
                else (normalize_surface(subject), normalize_surface(obj))
            )
            type_variants_by_untyped_edge.setdefault(
                (predicate, relation_direction, *surfaces), set()
            ).add((subject_type, object_type))

        superseded_versions = 0
        for index, raw in enumerate(raw_relations):
            relation = normalize_relation_semantics(raw)
            checked = checked_for(index, relation)
            if str(checked.get("factual_status", "VALID")).upper() == "REJECTED":
                factual_discards.append({
                    "candidate_id": relation.get("candidate_id", ""),
                    "candidate_version": relation.get("candidate_version", 1),
                    "reason_codes": list(checked.get("semantic_reasons", []) or []),
                })
                continue
            candidate_id = str(relation.get("candidate_id", "") or "")
            if (
                candidate_id
                and int(relation.get("candidate_version", 1) or 1)
                < latest_version_by_lineage.get(candidate_id, 1)
            ):
                superseded_versions += 1
                superseded_version_rows.append({
                    "candidate_id": candidate_id,
                    "candidate_version": int(relation.get("candidate_version", 1) or 1),
                    "representative_candidate_id": candidate_id,
                    "representative_candidate_version": latest_version_by_lineage[candidate_id],
                })
                continue
            subject = canonical(str(checked.get("subject", relation.get("subject", "")) or ""))
            obj = canonical(str(checked.get("object", relation.get("object", "")) or ""))
            predicate = str(checked.get("predicate", relation.get("predicate", "")) or "").upper()
            relation_direction = str(
                checked.get("relation_direction", relation.get("relation_direction", "UNKNOWN"))
                or "UNKNOWN"
            ).upper()
            subject_type = str(checked.get("subject_type", relation.get("subject_type", "")) or "")
            object_type = str(checked.get("object_type", relation.get("object_type", "")) or "")
            undirected = (
                relation_direction in {"NON_DIRECTIONAL", "BIDIRECTIONAL"}
                and schema_symmetric(predicate, subject_type, object_type)
            )
            untyped_surfaces = (
                tuple(sorted((normalize_surface(subject), normalize_surface(obj))))
                if relation_direction in {"NON_DIRECTIONAL", "BIDIRECTIONAL"}
                else (normalize_surface(subject), normalize_surface(obj))
            )
            untyped_key = (predicate, relation_direction, *untyped_surfaces)
            if len(type_variants_by_untyped_edge.get(untyped_key, set())) > 1:
                digest = hashlib.sha1("|".join(map(str, untyped_key)).encode("utf-8")).hexdigest()[:12]
                relation["type_conflict_group_id"] = f"tc-{digest}"
                relation.setdefault("quality_flags", []).extend([
                    "endpoint_type_conflict", "manual_review",
                ])
            endpoint_key = (
                tuple(sorted((
                    (normalize_surface(subject), subject_type),
                    (normalize_surface(obj), object_type),
                )))
                if undirected else (
                    (normalize_surface(subject), subject_type),
                    (normalize_surface(obj), object_type),
                )
            )
            groups.setdefault((predicate, relation_direction, *endpoint_key), []).append(
                (index, relation, checked)
            )

        finalized: list[tuple[int, dict]] = []
        duplicate_count = 0
        orientation_changes = 0
        for items in groups.values():
            items.sort(key=variant_rank)
            index, selected, checked = items[0]
            relation = copy.deepcopy(selected)
            relation["subject"] = canonical(str(checked.get("subject", relation.get("subject", "")) or ""))
            relation["object"] = canonical(str(checked.get("object", relation.get("object", "")) or ""))
            relation["subject_type"] = str(checked.get("subject_type", relation.get("subject_type", "")) or "")
            relation["object_type"] = str(checked.get("object_type", relation.get("object_type", "")) or "")
            relation["predicate"] = str(checked.get("predicate", relation.get("predicate", "")) or "").upper()

            if (
                str(relation.get("relation_direction", "UNKNOWN")).upper()
                in {"NON_DIRECTIONAL", "BIDIRECTIONAL"}
                and schema_symmetric(
                    relation["predicate"], relation["subject_type"], relation["object_type"]
                )
            ):
                evidence = str(relation.get("evidence", "") or "")
                subject_position = first_mention_position(
                    evidence, relation["subject"], str(selected.get("subject", "") or "")
                )
                object_position = first_mention_position(
                    evidence, relation["object"], str(selected.get("object", "") or "")
                )
                if object_position < subject_position:
                    relation["subject"], relation["object"] = relation["object"], relation["subject"]
                    relation["subject_type"], relation["object_type"] = (
                        relation["object_type"], relation["subject_type"]
                    )
                    orientation_changes += 1

            evidence_candidates = []
            evidence_spans = []
            claim_instances = []
            provenance = []
            merged_ids = []
            source_lanes = []
            for _, variant, variant_checked in items:
                merged_ids.extend([
                    str(variant.get("candidate_id", "") or ""),
                    *(str(item) for item in (variant.get("merged_candidate_ids", []) or [])),
                    *(str(item) for item in (variant.get("source_candidate_ids", []) or [])),
                ])
                evidence_candidates.extend([
                    str(variant.get("evidence", "") or ""),
                    *(variant.get("evidence_candidates", []) or []),
                ])
                pack = variant_checked.get("evidence_pack", variant.get("evidence_pack", {})) or {}
                evidence_spans.extend(pack.get("spans", []) or variant_checked.get("evidence_spans", []) or [])
                variant_instances = variant.get("claim_instances", []) or [{
                        "candidate_id": variant.get("candidate_id", ""),
                        "candidate_version": variant.get("candidate_version", 1),
                        "parent_candidate_id": variant.get("parent_candidate_id", ""),
                        "parent_version": variant.get("parent_version", 0),
                        "candidate_lane": variant.get("candidate_lane", "extracted_hint"),
                        "subject": variant_checked.get("subject", variant.get("subject", "")),
                        "subject_type": variant_checked.get("subject_type", variant.get("subject_type", "")),
                        "predicate": variant_checked.get("predicate", variant.get("predicate", "")),
                        "object": variant_checked.get("object", variant.get("object", "")),
                        "object_type": variant_checked.get("object_type", variant.get("object_type", "")),
                        "pair_candidate_id": variant.get("pair_candidate_id", ""),
                        "source_candidate_ids": list(
                            variant.get("source_candidate_ids", []) or []
                        ),
                        "claim_role": variant_checked.get("claim_role", variant.get("claim_role", "CURRENT_FINDING")),
                        "semantic_status": variant_checked.get("semantic_status", "UNVERIFIED"),
                        "factual_status": variant_checked.get("factual_status", "UNVERIFIED"),
                        "write_status": variant_checked.get("write_status", "UNASSESSED"),
                        "evidence": variant.get("evidence", ""),
                        "evidence_pack": copy.deepcopy(pack),
                        "support_mode": pack.get("support_mode", "UNRESOLVED"),
                        "minimal_support_span_ids": list(
                            pack.get("minimal_support_span_ids", []) or []
                        ),
                    }]
                for raw_instance in variant_instances:
                    instance = copy.deepcopy(raw_instance)
                    instance.setdefault(
                        "candidate_lane",
                        variant.get("candidate_lane", "extracted_hint"),
                    )
                    instance.setdefault("subject", variant_checked.get(
                        "subject", variant.get("subject", "")
                    ))
                    instance.setdefault("subject_type", variant_checked.get(
                        "subject_type", variant.get("subject_type", "")
                    ))
                    instance.setdefault("predicate", variant_checked.get(
                        "predicate", variant.get("predicate", "")
                    ))
                    instance.setdefault("object", variant_checked.get(
                        "object", variant.get("object", "")
                    ))
                    instance.setdefault("object_type", variant_checked.get(
                        "object_type", variant.get("object_type", "")
                    ))
                    instance.setdefault("claim_role", variant_checked.get(
                        "claim_role", variant.get("claim_role", "CURRENT_FINDING")
                    ))
                    instance.setdefault("factual_status", variant_checked.get(
                        "factual_status", "UNVERIFIED"
                    ))
                    instance.setdefault("semantic_status", variant_checked.get(
                        "semantic_status", "UNVERIFIED"
                    ))
                    instance.setdefault("write_status", variant_checked.get(
                        "write_status", "UNASSESSED"
                    ))
                    instance.setdefault("evidence_pack", copy.deepcopy(pack))
                    claim_instances.append(instance)
                source_lanes.extend([
                    str(variant.get("candidate_lane", "extracted_hint") or "extracted_hint"),
                    *(str(item) for item in variant.get("source_lanes", []) or []),
                ])
                provenance.extend(variant.get("provenance", []) or [])
            relation["evidence_candidates"] = list(dict.fromkeys(
                item.strip() for item in evidence_candidates if str(item).strip()
            ))[:3]
            if relation["evidence_candidates"]:
                relation["evidence"] = relation["evidence_candidates"][0]
            span_by_id = {
                str(item.get("span_id", "") or f"{item.get('char_start', item.get('start', -1))}:{item.get('char_end', item.get('end', -1))}"): item
                for item in evidence_spans if isinstance(item, dict)
            }
            relation["evidence_spans"] = list(span_by_id.values())[:3]
            relation["claim_instances"] = claim_instances
            relation["source_lanes"] = sorted(set(item for item in source_lanes if item))
            relation["provenance"] = sorted(set(item for item in provenance if item))
            relation["merged_candidate_ids"] = sorted(set(item for item in merged_ids if item))
            finalized.append((index, relation))
            duplicate_count += max(0, len(items) - 1)

        finalized.sort(key=lambda item: item[0])
        return [item[1] for item in finalized], {
            "rolled_back_count": 0,
            "rolled_back": [],
            "soft_flag_hard_reject_count": 0,
            "factual_discard_count": len(factual_discards),
            "factual_discards": factual_discards,
            "duplicate_relations_removed": duplicate_count,
            "symmetric_orientation_changes": orientation_changes,
            "status_mutations": 0,
            "superseded_versions_removed": superseded_versions,
            "superseded_versions": superseded_version_rows,
        }

    @staticmethod
    def _build_prompt(
        text: str,
        candidates: list[dict],
        units: list[EvidenceUnit],
        reasons: list[str],
        pmid: str,
        rag_context: dict,
    ) -> str:
        relevant_units: list[EvidenceUnit] = []
        seen: set[str] = set()
        for candidate in candidates:
            unit = ArticleEvidenceReader.containing_unit(candidate.get("evidence", ""), units)
            if unit and unit.unit_id not in seen:
                relevant_units.append(unit)
                seen.add(unit.unit_id)
        if not relevant_units:
            relevant_units = units[:MAX_EVIDENCE_UNITS]
        unit_payload = [item.to_dict() for item in relevant_units[:MAX_EVIDENCE_UNITS]]
        rag_payload = {
            "usage_policy": rag_context.get("usage_policy", {}),
            "entity_contexts": (rag_context.get("entity_contexts", []) or [])[:6],
        }
        return f"""你是医学关系候选的结构化语义裁判，不是自由抽取器。

你只能审核下面已有的 candidate_id。禁止自由生成实体、端点、证据或谓词。必须为每个候选返回一个决定：
- KEEP：原文支持当前关系；背景、方法或预测仍可 KEEP，但必须用 claim_role 标明，后续写入门会独立阻断；
- REJECT：端点间没有该关系，或原文在目标谓词作用域内明确反驳该关系；
- CHANGE_PREDICATE：只能从该候选 allowed_predicates 中选择；
- CHANGE_DIRECTION：只修正明确的方向；
- CHANGE_EVIDENCE：只能选择给定 evidence_unit_id。
- ADD_RELATION：仅限 candidate_kind=recovery；实体对和 evidence 已固定，new_predicate 必须从 allowed_predicates 选择；仅当原文方向相反时设置 swap_endpoints=true。

verdict 必须为 SUPPORTED、AMBIGUOUS、UNSUPPORTED 或 CONTRADICTED。关系真假、语义把握和主库写入资格彼此独立；你无权决定写入。

关键语义边界：
1. "potential target for diagnosis/treatment" 不等于 PROGNOSTIC_IN；
2. 两个分子在同一句中表达改变，不等于 INTERACTS_WITH；
3. 共同出现、同一列表或同一研究背景，不等于 ASSOCIATED_WITH；
4. 数据库筛选、富集、docking、预测关系应标为 PREDICTION/METHOD，不得冒充本文实验发现，但若文本确实陈述该关系可判 SUPPORTED；
5. 综述背景知识应标为 BACKGROUND，不得冒充本文新发现，但 claim_role 不决定关系真假。

recovery 候选只是“同一证据单元内的类型合法实体对”，共同出现本身不构成关系；
只有原文明确触发 allowed_predicates 中某个关系时才 ADD_RELATION，否则 REJECT。
如果文本存在支持但谓词或指代仍有歧义，返回 AMBIGUOUS，而不是删除候选。每个决定同时返回 candidate_id、candidate_version、四类方向字段、claim_role 和 supporting_span_ids。不要补充候选。输出严格 JSON。

PMID: {pmid}
触发原因: {json.dumps(reasons, ensure_ascii=False)}
候选关系: {json.dumps(candidates, ensure_ascii=False, default=str)}
原文证据单元（均为连续原文 span）: {json.dumps(unit_payload, ensure_ascii=False)}
查询级 Neo4j 上下文（只能做别名、类型和冲突提示，绝不能作为当前文章证据）:
{json.dumps(rag_payload, ensure_ascii=False, default=str)}
"""
