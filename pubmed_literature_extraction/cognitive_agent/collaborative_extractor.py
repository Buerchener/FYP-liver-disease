#!/usr/bin/env python3
"""Bounded second-model adjudication for Phase B.

The second model is an edit-only judge.  It cannot invent endpoints, entities,
or relations.  It receives a compact list of grounded candidate triples and
may keep, reject, or edit a predicate/direction/evidence selection.  Every edit
is re-verified by the deterministic Phase-A verifier.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from cognitive_agent.evidence_units import ArticleEvidenceReader, EvidenceUnit
from cognitive_agent.extraction_quality import normalize_surface
from cognitive_agent.relation_contract import SEMANTIC_REJECT_FLAGS
from cognitive_agent.schema.entity_classes import ENTITY_CLASSES
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
HARD_RELATION_BLOCKERS = SEMANTIC_REJECT_FLAGS

# A relation changed by an agent action has a higher burden than an untouched
# first-pass candidate.  It must survive the verifier with direct endpoint-
# linking evidence; otherwise the controller rolls the action back.
POST_ACTION_BLOCKERS = frozenset({
    *HARD_RELATION_BLOCKERS,
    "trigger_missing", "trigger_not_linking_endpoints",
    "trigger_direction_mismatch", "weak_evidence", "uncertain",
})

# Only predicates whose inverse has exactly the same meaning are consolidated
# as an unordered pair.  ASSOCIATED_WITH is intentionally excluded because the
# project evaluates and imports its argument roles directionally.
UNDIRECTED_PREDICATES = frozenset({"INTERACTS_WITH"})

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

RELATION_DESCRIPTIONS = {
    "ASSOCIATED_WITH": "The text explicitly states an association, correlation, or linked change.",
    "ENCODES": "A gene explicitly encodes a protein product.",
    "PARTICIPATES_IN": "A gene or protein explicitly participates in or regulates a named pathway.",
    "INTERACTS_WITH": "The text explicitly reports molecular binding or interaction.",
    "EXPRESSED_IN": "The text explicitly locates gene/protein expression in a tissue or cell type.",
    "PROGNOSTIC_IN": "A marker explicitly predicts prognosis, survival, recurrence, or clinical outcome.",
    "PROGRESSES_TO": "One disease or stage explicitly progresses to another.",
    "ASSOCIATED_WITH_METABOLITE": "The text explicitly associates an entity with a metabolite.",
}


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
                    "action": {"type": "string", "enum": sorted(DECISION_ACTIONS)},
                    "new_predicate": {"type": "string"},
                    "new_direction": {"type": "string"},
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
            if flags & HARD_RELATION_BLOCKERS:
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
            if not (flags & fixable_flags or high_risk_predicate or hedged_or_proxy):
                continue
            candidates.append({
                "candidate_id": f"r{raw_index:03d}",
                "pair_candidate_id": relation.get("candidate_id", ""),
                "raw_index": raw_index,
                "subject": relation.get("subject", ""),
                "subject_type": relation.get("subject_type", ""),
                "predicate": predicate,
                "object": relation.get("object", ""),
                "object_type": relation.get("object_type", ""),
                "direction": relation.get("direction", "unknown"),
                "evidence": relation.get("evidence", ""),
                "import_ready": bool(relation.get("import_ready")),
                "quality_flags": sorted(flags),
                "classifier_confidence": relation.get("classifier_confidence"),
                "relation_probability": relation.get("relation_probability"),
                "no_relation_probability": relation.get("no_relation_probability"),
                "classifier_margin": relation.get("classifier_margin"),
                "classifier_source": relation.get("classifier_source", ""),
                "evidence_unit_id": relation.get("evidence_unit_id", ""),
                "allowed_predicates": [
                    {"predicate": pred, "meaning": RELATION_DESCRIPTIONS.get(pred, pred)}
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
                    "meaning": RELATION_DESCRIPTIONS.get(
                        str(value.get("predicate", "") if isinstance(value, dict) else value).upper(),
                        str(value),
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
                "raw_index": candidate["raw_index"],
                "pair_candidate_id": candidate.get("pair_candidate_id", ""),
                "candidate_kind": candidate_kind,
                "action": action,
                "new_predicate": new_predicate,
                "new_direction": new_direction,
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
        decision_map = {
            int(item.get("raw_index", -1)): item for item in collaboration.review_decisions
            if int(item.get("raw_index", -1)) >= 0
        }
        for index, raw_relation in enumerate(raw_relations):
            verified = verified_relations[index] if index < len(verified_relations) else {}
            flags = set(verified.get("quality_flags", []) or [])
            if flags & HARD_RELATION_BLOCKERS:
                merged.deterministic_rejections.append({
                    "raw_index": index,
                    "subject": raw_relation.get("subject", ""),
                    "predicate": raw_relation.get("predicate", ""),
                    "object": raw_relation.get("object", ""),
                    "reason_codes": sorted(flags & HARD_RELATION_BLOCKERS),
                })
                continue

            relation = copy.deepcopy(raw_relation)
            decision = decision_map.get(index)
            if "agent_evidence_repaired" in flags and not decision:
                merged.deterministic_rejections.append({
                    "raw_index": index,
                    "subject": raw_relation.get("subject", ""),
                    "predicate": raw_relation.get("predicate", ""),
                    "object": raw_relation.get("object", ""),
                    "reason_codes": ["unreviewed_agent_evidence_repair"],
                })
                continue
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
                continue
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
                "collaboration_reason": decision.get("reason", ""),
            }
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
        """Rollback failed actions and keep one strongest canonical edge.

        Re-verification may normalize an abbreviation after the first merge
        (for example PBC -> Primary Biliary Cholangitis) or expose the same
        molecular mention as both Gene and Protein.  Consolidation therefore
        uses the verifier's canonical endpoints, not the pre-verification raw
        strings.  Truly undirected predicates also share one unordered key.
        """
        verified = verification.get("relations", []) or []
        verified_entities = verification.get("entities", []) or []
        review = verification.get("review", {}) or {}
        mention_to_canonical = review.get("mention_to_canonical", {}) or {}

        canonical_aliases: dict[str, list[str]] = {}
        for mention, canonical in mention_to_canonical.items():
            canonical_aliases.setdefault(normalize_surface(canonical), []).append(str(mention))
        for canonical in mention_to_canonical.values():
            canonical_aliases.setdefault(normalize_surface(canonical), []).append(str(canonical))

        # Prefer a type backed by an exact graph/identifier match, then by the
        # extractor confidence.  This resolves duplicate Gene/Protein views of
        # one article-local molecular mention without inventing a new type.
        entity_type_rank: dict[tuple[str, str], tuple[int, float]] = {}
        for entity in verified_entities:
            key = (
                normalize_surface(entity.get("mention", "")),
                str(entity.get("type", "")),
            )
            status = str(entity.get("neo4j_status", "") or "").upper()
            identifier = str(
                entity.get("neo4j_node_id", "") or entity.get("normalized_id", "") or ""
            )
            support_rank = 0 if status == "EXACT_MATCH" else (1 if identifier else 2)
            try:
                confidence = float(entity.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            value = (support_rank, -confidence)
            if key not in entity_type_rank or value < entity_type_rank[key]:
                entity_type_rank[key] = value

        def canonical_endpoint(checked: dict, relation: dict, role: str) -> str:
            return str(checked.get(role, "") or relation.get(role, "") or "")

        def explicit_type_rank(mention: str, entity_type: str) -> int:
            if not source_text or entity_type not in {"Gene", "Protein"}:
                return 1
            aliases = [
                mention,
                *canonical_aliases.get(normalize_surface(mention), []),
            ]
            type_word = "protein" if entity_type == "Protein" else "gene"
            for alias in dict.fromkeys(str(item).strip() for item in aliases if str(item).strip()):
                escaped = re.escape(alias)
                patterns = (
                    rf"(?i)(?:{escaped})\s+{type_word}\b",
                    rf"(?i)\b{type_word}\s+(?:named\s+|called\s+)?(?:{escaped})\b",
                    rf"(?i)\b{type_word}\b[^.\n]{{0,60}}\(\s*{escaped}\s*\)",
                )
                if any(re.search(pattern, source_text) for pattern in patterns):
                    return 0
            return 1

        def endpoint_type_rank(
            checked: dict, relation: dict, role: str
        ) -> tuple[int, int, float]:
            mention = canonical_endpoint(checked, relation, role)
            entity_type = str(
                checked.get(f"{role}_type", "") or relation.get(f"{role}_type", "") or ""
            )
            support = entity_type_rank.get((normalize_surface(mention), entity_type), (3, 0.0))
            return (explicit_type_rank(mention, entity_type), *support)

        def first_alias_position(evidence: str, canonical: str, raw: str) -> int:
            aliases = [
                raw,
                canonical,
                *canonical_aliases.get(normalize_surface(canonical), []),
            ]
            positions: list[int] = []
            for alias in dict.fromkeys(str(item).strip() for item in aliases if str(item).strip()):
                pattern = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE)
                match = pattern.search(evidence)
                if match:
                    positions.append(match.start())
            return min(positions, default=10**9)
        survivors: list[tuple[int, dict, dict]] = []
        rolled_back: list[dict] = []
        for index, relation in enumerate(raw_relations):
            checked = verified[index] if index < len(verified) else {}
            flags = set(checked.get("quality_flags", []) or [])
            action_changed = bool(flags & {
                "agent_evidence_repaired", "agent_recovered_relation",
                "second_llm_edited", "second_llm_evidence_review",
                "second_llm_confirmed",
            })
            if action_changed and flags & POST_ACTION_BLOCKERS:
                rolled_back.append({
                    "raw_index": index,
                    "subject": relation.get("subject", ""),
                    "predicate": relation.get("predicate", ""),
                    "object": relation.get("object", ""),
                    "reason_codes": sorted(flags & POST_ACTION_BLOCKERS),
                })
                continue
            survivors.append((index, relation, checked))

        groups: dict[tuple, list[tuple[int, dict, dict]]] = {}
        for item in survivors:
            relation, checked = item[1], item[2]
            predicate = str(checked.get("predicate", "") or relation.get("predicate", "")).upper()
            subject = normalize_surface(canonical_endpoint(checked, relation, "subject"))
            object_name = normalize_surface(canonical_endpoint(checked, relation, "object"))
            endpoints = tuple(sorted((subject, object_name))) if predicate in UNDIRECTED_PREDICATES else (
                subject, object_name
            )
            # Endpoint types are deliberately absent: identical canonical
            # mentions with competing Gene/Protein labels are one edge and the
            # best-supported typed view is selected below.
            key = (predicate, *endpoints)
            groups.setdefault(key, []).append(item)

        kept: list[tuple[int, dict, dict]] = []
        duplicate_count = 0
        orientation_changes = 0
        for items in groups.values():
            items.sort(key=lambda item: (
                not bool(item[2].get("import_ready")),
                int(item[2].get("evidence_level", 3) or 3),
                endpoint_type_rank(item[2], item[1], "subject"),
                endpoint_type_rank(item[2], item[1], "object"),
                "agent_recovered_relation" in set(item[2].get("quality_flags", []) or []),
                len(item[2].get("quality_flags", []) or []),
                len(str(item[1].get("evidence", "") or "")),
                item[0],
            ))
            selected = items[0]
            relation = copy.deepcopy(selected[1])
            checked = selected[2]
            predicate = str(checked.get("predicate", "") or relation.get("predicate", "")).upper()
            if predicate in UNDIRECTED_PREDICATES:
                evidence = str(relation.get("evidence", "") or "")
                subject_position = first_alias_position(
                    evidence,
                    canonical_endpoint(checked, relation, "subject"),
                    str(relation.get("subject", "") or ""),
                )
                object_position = first_alias_position(
                    evidence,
                    canonical_endpoint(checked, relation, "object"),
                    str(relation.get("object", "") or ""),
                )
                if object_position < subject_position:
                    relation["subject"], relation["object"] = (
                        relation.get("object", ""), relation.get("subject", "")
                    )
                    relation["subject_type"], relation["object_type"] = (
                        relation.get("object_type", ""), relation.get("subject_type", "")
                    )
                    orientation_changes += 1
            kept.append((selected[0], relation, checked))
            duplicate_count += max(0, len(items) - 1)
        kept.sort(key=lambda item: item[0])
        return [copy.deepcopy(item[1]) for item in kept], {
            "rolled_back_count": len(rolled_back),
            "rolled_back": rolled_back,
            "duplicate_relations_removed": duplicate_count,
            "symmetric_orientation_changes": orientation_changes,
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
        return f"""你是医学知识图谱的精确优先关系裁判，不是抽取器。

你只能审核下面已有的 candidate_id。禁止自由生成实体、端点、证据或谓词。必须为每个候选返回一个决定：
- KEEP：原文直接支持当前谓词；
- REJECT：只是共同出现、背景陈述、方法/预测、目标陈述、过度推断或证据不足；
- CHANGE_PREDICATE：只能从该候选 allowed_predicates 中选择；
- CHANGE_DIRECTION：只修正明确的方向；
- CHANGE_EVIDENCE：只能选择给定 evidence_unit_id。
- ADD_RELATION：仅限 candidate_kind=recovery；实体对和 evidence 已固定，new_predicate 必须从 allowed_predicates 选择；仅当原文方向相反时设置 swap_endpoints=true。

关键硬负例：
1. "potential target for diagnosis/treatment" 不等于 PROGNOSTIC_IN；
2. 两个分子在同一句中表达改变，不等于 INTERACTS_WITH；
3. 共同出现、同一列表或同一研究背景，不等于 ASSOCIATED_WITH；
4. 数据库筛选、富集、docking、预测关系不是当前文章实验事实；
5. 综述中的背景知识不能冒充本文新发现。

recovery 候选只是“同一证据单元内的类型合法实体对”，共同出现本身不构成关系；
只有原文明确触发 allowed_predicates 中某个关系时才 ADD_RELATION，否则 REJECT。
如果不确定，REJECT。不要补充候选。输出严格 JSON。

PMID: {pmid}
触发原因: {json.dumps(reasons, ensure_ascii=False)}
候选关系: {json.dumps(candidates, ensure_ascii=False, default=str)}
原文证据单元（均为连续原文 span）: {json.dumps(unit_payload, ensure_ascii=False)}
查询级 Neo4j 上下文（只能做别名、类型和冲突提示，绝不能作为当前文章证据）:
{json.dumps(rag_payload, ensure_ascii=False, default=str)}
"""
