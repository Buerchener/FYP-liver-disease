#!/usr/bin/env python3
"""Canonical relation contract shared by extraction, verification and scoring.

The primary extractor may emit relations both as top-level records and inside
entity attributes.  This module projects both representations into one closed,
auditable candidate form.  It deliberately does not decide whether a relation
may be written: semantic validity and write eligibility are separate states.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from cognitive_agent.abbreviation_detector import AbbreviationDetector
from cognitive_agent.extraction_quality import locate_contiguous, normalize_surface
from cognitive_agent.schema.predicate_cards import relation_card
from cognitive_agent.candidate_lineage import normalize_lineage


ATTRIBUTE_PREDICATES = {
    "associated_with": "ASSOCIATED_WITH",
    "associated_with_metabolite": "ASSOCIATED_WITH_METABOLITE",
    "interacts_with": "INTERACTS_WITH",
    "expressed_in": "EXPRESSED_IN",
    "participates_in": "PARTICIPATES_IN",
    "prognostic_in": "PROGNOSTIC_IN",
    "progresses_to": "PROGRESSES_TO",
    "encodes": "ENCODES",
}

ENTITY_TYPE_ALIASES = {
    "gene": "Gene", "protein": "Protein", "disease": "Disease",
    "pathway": "Pathway", "metabolite": "Metabolite", "tissue": "Tissue",
    "celltype": "CellType", "cell_type": "CellType", "cell type": "CellType",
}

# These invalidate a claim as a current-article semantic relation.
SEMANTIC_REJECT_FLAGS = frozenset({
    "schema_mismatch", "negated", "contradiction",
    "subject_endpoint_missing", "object_endpoint_missing", "empty_evidence",
    "evidence_not_contiguous", "subject_not_grounded", "object_not_grounded",
    "endpoint_not_in_evidence", "method_only", "prediction_only",
    "filtered_endpoint", "unresolved_endpoint", "background_only",
    "objective_only", "method_section_only",
})

# These preserve the semantic candidate but require abstention/review.
SEMANTIC_REVIEW_FLAGS = frozenset({
    "uncertain", "weak_evidence", "trigger_missing",
    "trigger_not_linking_endpoints", "trigger_direction_mismatch",
    "ambiguous_endpoint", "manual_review", "pair_low_confidence",
    "pair_ambiguous_predicate", "evidence_not_entailed", "title_only",
    "review_article",
    # A single-model judge-ENTAILED semantic proposal conflicts with the
    # deterministic semantic flags: the conflict is a REVIEW-worthy flag that
    # routes the relation to the independent (second-model) adjudicator.
    "judge_verifier_conflict",
})

# Trigger/evidence heuristics that ONLY the independent second-model
# endorsement (flag `adjudicator_entailed`, set by a DeepSeek KEEP during
# bounded adjudication) may clear for SEMANTIC acceptance.  A single-model
# judge-ENTAILED proposal alone never overrides these; it only produces a
# `judge_verifier_conflict` REVIEW.  The verifier's factual checks (schema,
# endpoints, negation, method/background sections, evidence continuity)
# always stay in force, and the write gate (WRITE_BLOCK_FLAGS) is untouched:
# these flags still block import_ready.
JUDGE_SEMANTIC_OVERRIDABLE_FLAGS = frozenset({
    "trigger_missing", "trigger_not_linking_endpoints",
    "trigger_direction_mismatch", "weak_evidence",
})

JUDGE_BACKEND_NAMES = frozenset({"pairwise_judge_v1"})

# Write policy is deliberately stricter than semantic acceptance.
WRITE_BLOCK_FLAGS = frozenset({
    *SEMANTIC_REJECT_FLAGS, *SEMANTIC_REVIEW_FLAGS,
    "invalid_direction", "non_human", "non_human_article",
    "article_out_of_scope", "second_llm_rejected", "review_article",
    # An asserted prior/background claim can remain a semantic relation, but
    # it is not this article's new importable evidence.
    "non_current_finding_role",
    # Judge-sourced relations may only be auto-written with very high judge
    # confidence; the bounded adjudicator's KEEP clears this flag.
    "judge_no_write_endorsement",
})

# Tiered-v2 separates factual legality, semantic confidence, and write
# eligibility.  A factual rejection means the candidate may be discarded;
# consequently this set contains only the four provenance invariants.  Empty,
# partial, fuzzy or cross-sentence evidence is repair/review material until the
# source resolver explicitly records ``evidence_untraceable``.
FACTUAL_REJECT_FLAGS = frozenset({
    "schema_mismatch",
    "subject_endpoint_missing", "object_endpoint_missing",
    "unresolved_endpoint",
    "evidence_untraceable",
    "scoped_negation", "evidence_contradicted", "source_refutation",
})

FACTUAL_REVIEW_FLAGS = frozenset({
    "contradiction", "opposite_direction", "negated", "empty_evidence",
    "evidence_not_contiguous", "subject_not_grounded",
    "object_not_grounded", "endpoint_not_in_evidence",
    "ambiguous_endpoint", "subject_ambiguous", "object_ambiguous",
    "judge_quote_not_in_source", "endpoint_type_ambiguous",
    "evidence_fuzzy_aligned", "evidence_partial_aligned",
    "multi_span_support", "coreference_only_support", "composite_endpoint", "schema_gap",
    "filtered_endpoint", "subject_filtered", "object_filtered",
    "endpoint_type_conflict",
})

TIERED_SEMANTIC_REJECT_FLAGS = frozenset({
    "second_llm_rejected", "dual_model_unsupported",
})

TIERED_SEMANTIC_REVIEW_FLAGS = frozenset({
    "uncertain", "weak_evidence", "trigger_missing",
    "trigger_not_linking_endpoints", "trigger_direction_mismatch",
    "pair_low_confidence", "pair_ambiguous_predicate", "pair_no_relation_dissent",
    "evidence_not_entailed", "judge_verifier_conflict",
    "judge_entailment_without_trigger_support",
    "adjudicator_ambiguous", "adjudication_span_mismatch",
    "adjudication_span_support_insufficient",
    "manual_review_unclassified", "relation_direction_mismatch",
    "predicate_card_conflict", "predicate_card_type_only",
    "prediction_only", "method_only", "background_only",
    "objective_only", "method_section_only", "title_only", "review_article",
    "non_current_finding_role", "claim_role_conflict", "endpoint_type_conflict",
    "filtered_endpoint", "subject_filtered", "object_filtered",
    "endpoint_type_ambiguous", "subject_type_ambiguous", "object_type_ambiguous",
    "composite_endpoint", "schema_gap", "lineage_binding_missing",
})

# Audit-only observations must never decide whether the asserted relation is
# semantically true.  They remain available to reporting and may still be used
# by the independent write gate where appropriate.
TIERED_DIAGNOSTIC_FLAGS = frozenset({
    "subject_novel", "object_novel", "cross_sentence",
    "legacy_direction_ambiguous", "direction_migration_review",
    "association_sign_not_applicable",
})

# A bounded, source-grounded adjudication may resolve only these semantic
# uncertainties.  Direction disagreement and claim-role/scope flags are
# deliberately excluded: they require a versioned edit/reverification or stay
# in REVIEW.
ADJUDICATION_OVERRIDABLE_FLAGS = frozenset({
    "trigger_missing", "trigger_not_linking_endpoints", "weak_evidence",
    "judge_entailment_without_trigger_support", "pair_no_relation_dissent",
    "pair_low_confidence", "judge_verifier_conflict", "predicate_card_type_only",
})

WRITE_REVIEW_FLAGS = frozenset({
    *FACTUAL_REVIEW_FLAGS, *TIERED_SEMANTIC_REVIEW_FLAGS,
    "invalid_direction", "non_human", "non_human_article",
    "article_out_of_scope", "second_llm_rejected", "review_article",
    "non_current_finding_role", "judge_no_write_endorsement",
    # ``cross_sentence`` alone is diagnostic.  The support-specific factual
    # flags above keep genuinely multi-span/coreference claims curator-gated.
    "manual_review",
})

# Kept as a compatibility export for callers/tests that import the old name.
# Models are not permitted to override write policy in tiered-v2.
MODEL_OVERRIDABLE_WRITE_FLAGS = frozenset()


VERIFICATION_POLICY_CONTRACT_VERSION = "tiered-v2-monotonic-v5"


@dataclass(frozen=True)
class VerificationPolicy:
    """One policy source for verifier, controller and reconciliation layers."""

    name: str
    contract_version: str
    factual_reject_flags: frozenset[str]
    factual_review_flags: frozenset[str]
    semantic_reject_flags: frozenset[str]
    semantic_review_flags: frozenset[str]
    semantic_model_overridable_flags: frozenset[str]
    write_review_flags: frozenset[str]
    diagnostic_flags: frozenset[str] = frozenset()
    adjudication_overridable_flags: frozenset[str] = frozenset()

    def hard_reject_reasons(self, flags: set[str] | frozenset[str]) -> set[str]:
        return set(flags) & self.factual_reject_flags

    def factual_review_reasons(self, flags: set[str] | frozenset[str]) -> set[str]:
        return set(flags) & self.factual_review_flags

    def semantic_review_reasons(self, flags: set[str] | frozenset[str]) -> set[str]:
        return set(flags) & self.semantic_review_flags

    def write_reasons(self, flags: set[str] | frozenset[str]) -> set[str]:
        return set(flags) & self.write_review_flags


LEGACY_VERIFICATION_POLICY = VerificationPolicy(
    name="legacy",
    contract_version="legacy-v1",
    factual_reject_flags=SEMANTIC_REJECT_FLAGS,
    factual_review_flags=frozenset(),
    semantic_reject_flags=SEMANTIC_REJECT_FLAGS,
    semantic_review_flags=SEMANTIC_REVIEW_FLAGS,
    semantic_model_overridable_flags=JUDGE_SEMANTIC_OVERRIDABLE_FLAGS,
    write_review_flags=WRITE_BLOCK_FLAGS,
    diagnostic_flags=frozenset(),
    adjudication_overridable_flags=JUDGE_SEMANTIC_OVERRIDABLE_FLAGS,
)

TIERED_V2_VERIFICATION_POLICY = VerificationPolicy(
    name="tiered-v2",
    contract_version=VERIFICATION_POLICY_CONTRACT_VERSION,
    factual_reject_flags=FACTUAL_REJECT_FLAGS,
    factual_review_flags=FACTUAL_REVIEW_FLAGS,
    semantic_reject_flags=TIERED_SEMANTIC_REJECT_FLAGS,
    semantic_review_flags=TIERED_SEMANTIC_REVIEW_FLAGS,
    semantic_model_overridable_flags=JUDGE_SEMANTIC_OVERRIDABLE_FLAGS,
    write_review_flags=WRITE_REVIEW_FLAGS,
    diagnostic_flags=TIERED_DIAGNOSTIC_FLAGS,
    adjudication_overridable_flags=ADJUDICATION_OVERRIDABLE_FLAGS,
)


def verification_policy(name: str) -> VerificationPolicy:
    if name == "legacy":
        return LEGACY_VERIFICATION_POLICY
    if name == "tiered-v2":
        return TIERED_V2_VERIFICATION_POLICY
    raise ValueError("verification_policy must be legacy or tiered-v2")


RELATION_DIRECTIONS = frozenset({
    "SUBJECT_TO_OBJECT", "OBJECT_TO_SUBJECT", "BIDIRECTIONAL",
    "NON_DIRECTIONAL", "UNKNOWN",
})
ASSOCIATION_SIGNS = frozenset({"POSITIVE", "NEGATIVE", "NONE", "UNKNOWN"})
EXPRESSION_CHANGES = frozenset({"UP", "DOWN", "NO_CHANGE", "MIXED", "UNKNOWN"})
ACTIVITY_CHANGES = frozenset({
    "ACTIVATED", "INHIBITED", "NO_CHANGE", "MIXED", "UNKNOWN",
})
NON_DIRECTIONAL_PREDICATES = frozenset({
    "ASSOCIATED_WITH", "ASSOCIATED_WITH_METABOLITE", "INTERACTS_WITH",
})
SUBJECT_TO_OBJECT_PREDICATES = frozenset({
    "ENCODES", "PARTICIPATES_IN", "EXPRESSED_IN", "PROGNOSTIC_IN",
    "PROGRESSES_TO",
})

EXPRESSION_CUE_RE = re.compile(
    r"\b(?:express(?:ed|ion|ing)?|up-?regulat\w*|down-?regulat\w*|level[s]?|abundan\w*)\b",
    re.IGNORECASE,
)
ACTIVITY_CUE_RE = re.compile(
    r"\b(?:activat\w*|inhibit\w*|suppress\w*|block\w*|signal(?:ing)?|phosphorylat\w*)\b",
    re.IGNORECASE,
)
POSITIVE_ASSOCIATION_CUE_RE = re.compile(
    r"\b(?:positive(?:ly)? (?:association|associated|correlation|causal effect)|"
    r"increased? risk|higher risk|risk factor|promot\w*|contribut\w*)\b",
    re.IGNORECASE,
)
NEGATIVE_ASSOCIATION_CUE_RE = re.compile(
    r"\b(?:inverse(?:ly)?|negative(?:ly)? (?:association|associated|correlation)|"
    r"reduced? risk|lower risk|protect\w* against)\b",
    re.IGNORECASE,
)


def normalize_relation_semantics(relation: dict[str, Any]) -> dict[str, Any]:
    """Return predicate-specific direction fields plus a legacy adapter.

    The old ``direction`` value is interpreted only when its meaning is
    unambiguous for the predicate/evidence.  Ambiguous increase/decrease values
    are retained in ``legacy_direction`` and routed to review.
    """
    output = copy.deepcopy(relation)
    predicate = str(output.get("predicate", "") or "").upper()
    evidence = str(output.get("evidence", "") or "")
    legacy = str(output.get("legacy_direction", output.get("direction", "unknown")) or "unknown").lower()
    flags = set(output.get("quality_flags", []) or [])

    card = relation_card(predicate)
    default_direction = card.relation_direction if card else (
        "NON_DIRECTIONAL" if predicate in NON_DIRECTIONAL_PREDICATES
        else "SUBJECT_TO_OBJECT" if predicate in SUBJECT_TO_OBJECT_PREDICATES
        else "UNKNOWN"
    )
    relation_direction = str(output.get("relation_direction", "") or default_direction).upper()
    association_sign = str(output.get("association_sign", "") or "UNKNOWN").upper()
    expression_change = str(output.get("expression_change", "") or "UNKNOWN").upper()
    activity_change = str(output.get("activity_change", "") or "UNKNOWN").upper()

    if predicate in {"ASSOCIATED_WITH", "ASSOCIATED_WITH_METABOLITE"}:
        if association_sign == "UNKNOWN" and POSITIVE_ASSOCIATION_CUE_RE.search(evidence):
            association_sign = "POSITIVE"
        elif association_sign == "UNKNOWN" and NEGATIVE_ASSOCIATION_CUE_RE.search(evidence):
            association_sign = "NEGATIVE"

    if legacy in {"positive", "negative"} and association_sign == "UNKNOWN":
        if predicate in {"ASSOCIATED_WITH", "ASSOCIATED_WITH_METABOLITE"}:
            association_sign = legacy.upper()
        else:
            flags.update({"legacy_direction_ambiguous", "direction_migration_review"})
    elif legacy in {"increase", "decrease"}:
        if EXPRESSION_CUE_RE.search(evidence) and expression_change == "UNKNOWN":
            expression_change = "UP" if legacy == "increase" else "DOWN"
        elif ACTIVITY_CUE_RE.search(evidence) and activity_change == "UNKNOWN":
            activity_change = "ACTIVATED" if legacy == "increase" else "INHIBITED"
        else:
            flags.add("legacy_direction_ambiguous")
    elif legacy == "none" and relation_direction == "UNKNOWN":
        relation_direction = "NON_DIRECTIONAL"

    invalid = []
    if relation_direction not in RELATION_DIRECTIONS:
        invalid.append("relation_direction")
        relation_direction = "UNKNOWN"
    if association_sign not in ASSOCIATION_SIGNS:
        invalid.append("association_sign")
        association_sign = "UNKNOWN"
    if expression_change not in EXPRESSION_CHANGES:
        invalid.append("expression_change")
        expression_change = "UNKNOWN"
    if activity_change not in ACTIVITY_CHANGES:
        invalid.append("activity_change")
        activity_change = "UNKNOWN"
    if invalid:
        flags.update({"schema_mismatch", "invalid_relation_semantics"})

    if (
        predicate in NON_DIRECTIONAL_PREDICATES
        and relation_direction not in {"NON_DIRECTIONAL", "UNKNOWN"}
    ):
        flags.update({"relation_direction_mismatch", "manual_review"})
    if (
        predicate in SUBJECT_TO_OBJECT_PREDICATES
        and relation_direction not in {"SUBJECT_TO_OBJECT", "UNKNOWN"}
    ):
        flags.update({"relation_direction_mismatch", "manual_review"})
    if (
        predicate not in {"ASSOCIATED_WITH", "ASSOCIATED_WITH_METABOLITE"}
        and association_sign in {"POSITIVE", "NEGATIVE"}
    ):
        flags.update({"association_sign_not_applicable", "direction_migration_review"})
        # Preserve the observation as audit metadata, but do not leak an
        # association polarity into predicates whose semantics are expressed by
        # relation_direction/expression_change/activity_change.
        output["legacy_association_sign"] = association_sign
        association_sign = "UNKNOWN"

    if association_sign in {"POSITIVE", "NEGATIVE"}:
        compatible_direction = association_sign.lower()
    elif expression_change in {"UP", "DOWN"}:
        compatible_direction = "increase" if expression_change == "UP" else "decrease"
    elif activity_change in {"ACTIVATED", "INHIBITED"}:
        compatible_direction = "increase" if activity_change == "ACTIVATED" else "decrease"
    elif relation_direction == "NON_DIRECTIONAL":
        compatible_direction = "none"
    else:
        compatible_direction = "unknown"

    output.update({
        "predicate": predicate,
        "legacy_direction": legacy,
        "direction": compatible_direction,
        "relation_direction": relation_direction,
        "association_sign": association_sign,
        "expression_change": expression_change,
        "activity_change": activity_change,
        "quality_flags": sorted(flags),
    })
    return output


def stable_candidate_id(relation: dict[str, Any], *, lane: str = "extracted_hint") -> str:
    """Stable lineage ID independent of evidence text and list ordering."""
    raw = "|".join((
        str(lane or "extracted_hint"),
        normalize_surface(relation.get("subject")),
        normalize_entity_type(relation.get("subject_type")),
        str(relation.get("predicate", "") or "").upper(),
        normalize_surface(relation.get("object")),
        normalize_entity_type(relation.get("object_type")),
    ))
    return "c-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def normalize_entity_type(value: Any) -> str:
    raw = str(value or "").strip()
    return ENTITY_TYPE_ALIASES.get(raw.casefold(), raw)


@dataclass
class EntityRegistryEntry:
    mention: str
    entity_type: str
    family: str
    aliases: list[str] = field(default_factory=list)


class ArticleEntityRegistry:
    """Article-local alias registry without collapsing Gene and Protein types."""

    def __init__(self, entities: list[dict], text: str = ""):
        abbreviation_map = AbbreviationDetector().detect(text) if text else None
        self.entries: list[EntityRegistryEntry] = []
        self.by_surface_type: dict[tuple[str, str], EntityRegistryEntry] = {}
        self.by_surface: dict[str, list[EntityRegistryEntry]] = {}
        for entity in entities:
            mention = str(entity.get("mention", "") or "").strip()
            entity_type = normalize_entity_type(
                entity.get("type", entity.get("entity_type", ""))
            )
            if not mention or not entity_type:
                continue
            attrs = entity.get("attributes", {}) or {}
            aliases = [mention, *(entity.get("canonical_mentions", []) or [])]
            if abbreviation_map:
                aliases.extend([
                    abbreviation_map.resolve_to_long(mention),
                    abbreviation_map.resolve_to_short(mention),
                ])
            aliases = list(dict.fromkeys(
                str(item).strip() for item in aliases if str(item or "").strip()
            ))
            family_source = (
                attrs.get("normalized_id")
                or attrs.get("gene_symbol")
                or attrs.get("protein_name")
                or (abbreviation_map.canonical_name(mention) if abbreviation_map else "")
                or mention
            )
            entry = EntityRegistryEntry(
                mention=mention,
                entity_type=entity_type,
                family=normalize_surface(family_source),
                aliases=aliases,
            )
            self.entries.append(entry)
            for alias in aliases:
                surface = normalize_surface(alias)
                self.by_surface_type.setdefault((surface, entity_type), entry)
                self.by_surface.setdefault(surface, []).append(entry)

    def resolve(self, mention: Any, entity_type: Any = "") -> EntityRegistryEntry | None:
        surface = normalize_surface(mention)
        resolved_type = normalize_entity_type(entity_type)
        if not surface:
            return None
        if resolved_type:
            exact = self.by_surface_type.get((surface, resolved_type))
            if exact:
                return exact
        candidates = self.by_surface.get(surface, [])
        if len(candidates) == 1:
            return candidates[0]
        return None


@dataclass
class ProjectionResult:
    relations: list[dict] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)


class RelationCandidateProjector:
    """Project every supported extractor representation into one candidate ledger."""

    @staticmethod
    def _candidate_id(relation: dict) -> str:
        return stable_candidate_id(
            relation, lane=str(relation.get("candidate_lane", "extracted_hint") or "extracted_hint")
        )

    @staticmethod
    def _triple_key(relation: dict) -> tuple[str, str, str, str, str]:
        return (
            normalize_surface(relation.get("subject_family") or relation.get("subject")),
            normalize_entity_type(relation.get("subject_type")),
            str(relation.get("predicate", "")).upper(),
            normalize_surface(relation.get("object_family") or relation.get("object")),
            normalize_entity_type(relation.get("object_type")),
        )

    @staticmethod
    def _evidence_rank(relation: dict, text: str) -> tuple[int, int, int]:
        evidence = str(relation.get("evidence", "") or "")
        contiguous, _, _ = locate_contiguous(evidence, text)
        endpoints = int(
            bool(evidence)
            and normalize_surface(relation.get("subject")) in normalize_surface(evidence)
            and normalize_surface(relation.get("object")) in normalize_surface(evidence)
        )
        return (int(contiguous), endpoints, -len(evidence))

    def _normalize_relation(
        self, relation: dict, registry: ArticleEntityRegistry, text: str,
        provenance: str,
    ) -> dict:
        output = normalize_relation_semantics(relation)
        output["predicate"] = str(output.get("predicate", "") or "").upper()
        for role in ("subject", "object"):
            entity_type = normalize_entity_type(output.get(f"{role}_type", ""))
            entry = registry.resolve(output.get(role, ""), entity_type)
            output[f"{role}_type"] = entry.entity_type if entry else entity_type
            output[f"{role}_family"] = (
                entry.family if entry else normalize_surface(output.get(role, ""))
            )
            if entry:
                output[role] = entry.mention
        evidence = str(output.get("evidence", "") or "").strip()
        grounded, start, end = locate_contiguous(evidence, text)
        output.update({
            "evidence": evidence,
            "evidence_char_start": start,
            "evidence_char_end": end,
            "evidence_contiguous": grounded,
            "provenance": list(dict.fromkeys([
                *(output.get("provenance", []) or []), provenance,
            ])),
            "semantic_status": str(output.get("semantic_status", "UNVERIFIED")),
            "write_status": str(output.get("write_status", "UNASSESSED")),
            "candidate_lane": str(output.get("candidate_lane", "extracted_hint") or "extracted_hint"),
            "candidate_version": max(1, int(output.get("candidate_version", 1) or 1)),
            "parent_version": max(0, int(output.get("parent_version", 0) or 0)),
        })
        output["candidate_id"] = str(output.get("candidate_id", "")) or self._candidate_id(output)
        return normalize_lineage(output, lane="extracted_hint")

    def project(
        self, entities: list[dict], top_level_relations: list[dict], text: str = "",
    ) -> ProjectionResult:
        registry = ArticleEntityRegistry(entities, text)
        projected: list[dict] = [
            self._normalize_relation(item, registry, text, "top_level")
            for item in top_level_relations
        ]
        attribute_count = 0
        unresolved_attribute_endpoints = 0
        for entity in entities:
            subject = str(entity.get("mention", "") or "").strip()
            subject_type = normalize_entity_type(
                entity.get("type", entity.get("entity_type", ""))
            )
            for attribute, predicate in ATTRIBUTE_PREDICATES.items():
                values = (entity.get("attributes", {}) or {}).get(attribute, []) or []
                if not isinstance(values, list):
                    continue
                for value in values:
                    if not isinstance(value, dict):
                        continue
                    target = str(value.get("target_entity", "") or "").strip()
                    target_type = normalize_entity_type(value.get("target_type", ""))
                    if not subject or not target:
                        unresolved_attribute_endpoints += 1
                        continue
                    attribute_count += 1
                    relation = {
                        "subject": subject,
                        "subject_type": subject_type,
                        "predicate": predicate,
                        "object": target,
                        "object_type": target_type,
                        "direction": value.get("direction", "unknown"),
                        "negated": bool(value.get("negated", False)),
                        "uncertain": bool(value.get("uncertain", False)),
                        "evidence": value.get("evidence", ""),
                        "species": value.get("species", (entity.get("attributes", {}) or {}).get("species", "")),
                        "confidence": value.get("confidence", entity.get("confidence", 0.7)),
                        "quality_flags": ["projected_from_entity_attribute"],
                    }
                    projected.append(self._normalize_relation(
                        relation, registry, text, f"entity_attribute:{attribute}"
                    ))

        # One typed triple is one semantic candidate.  Choose a display quote,
        # but retain every distinct claim/evidence instance for reconciliation.
        grouped: dict[tuple[str, str, str, str, str], list[dict]] = {}
        for relation in projected:
            key = self._triple_key(relation)
            if not all((key[0], key[1], key[2], key[3], key[4])):
                continue
            grouped.setdefault(key, []).append(relation)
        consolidated: list[dict] = []
        duplicate_count = 0
        for key in sorted(grouped):
            variants = grouped[key]
            variants.sort(
                key=lambda item: self._evidence_rank(item, text), reverse=True
            )
            winner = copy.deepcopy(variants[0])
            winner["provenance"] = sorted({
                source for item in variants for source in item.get("provenance", []) or []
            })
            winner["candidate_id"] = str(
                winner.get("candidate_id", "") or self._candidate_id(winner)
            )
            winner["source_candidate_ids"] = sorted({
                str(source_id)
                for item in variants
                for source_id in (
                    item.get("source_candidate_ids", [])
                    or [item.get("candidate_id", "")]
                )
                if str(source_id)
            })
            winner["merged_candidate_ids"] = sorted({
                str(item.get("candidate_id", "") or "")
                for item in variants if str(item.get("candidate_id", "") or "")
            })
            winner["evidence_candidates"] = list(dict.fromkeys(
                str(item.get("evidence", "") or "").strip()
                for item in variants if str(item.get("evidence", "") or "").strip()
            ))[:3]
            winner["claim_instances"] = [
                {
                    "candidate_id": str(item.get("candidate_id", "") or ""),
                    "candidate_version": max(1, int(item.get("candidate_version", 1) or 1)),
                    "parent_candidate_id": str(item.get("parent_candidate_id", "") or ""),
                    "parent_version": max(0, int(item.get("parent_version", 0) or 0)),
                    "evidence": str(item.get("evidence", "") or ""),
                    "provenance": list(item.get("provenance", []) or []),
                    "claim_role": str(item.get("claim_role", "") or "CURRENT_FINDING"),
                    "legacy_direction": str(item.get("legacy_direction", item.get("direction", "unknown")) or "unknown"),
                }
                for item in variants
            ]
            consolidated.append(winner)
            duplicate_count += len(variants) - 1
        return ProjectionResult(relations=consolidated, audit={
            "top_level_relation_count": len(top_level_relations),
            "attribute_relation_count": attribute_count,
            "projected_before_dedup": len(projected),
            "projected_relation_count": len(consolidated),
            "duplicates_merged": duplicate_count,
            "unresolved_attribute_endpoints": unresolved_attribute_endpoints,
            "entity_registry_size": len(registry.entries),
            "semantic_write_contract": "separate_v1",
        })

    def consolidate(
        self, relations: list[dict], text: str = "", entities: list[dict] | None = None,
    ) -> list[dict]:
        """Deduplicate already projected/pair candidates without entity rewriting."""
        if entities:
            registry = ArticleEntityRegistry(entities, text)
            relations = [
                self._normalize_relation(item, registry, text, "relation_core")
                for item in relations
            ]
        grouped: dict[tuple[str, str, str, str, str], list[dict]] = {}
        for relation in relations:
            grouped.setdefault(self._triple_key(relation), []).append(copy.deepcopy(relation))
        output: list[dict] = []
        for key in sorted(grouped):
            if not all(key):
                continue
            variants = grouped[key]
            variants.sort(key=lambda item: self._evidence_rank(item, text), reverse=True)
            winner = variants[0]
            winner["provenance"] = sorted({
                source for item in variants
                for source in (item.get("provenance", []) or [item.get("classifier_source", "")])
                if source
            })
            winner["candidate_id"] = str(winner.get("candidate_id", "")) or self._candidate_id(winner)
            winner["source_candidate_ids"] = sorted({
                str(source_id)
                for item in variants
                for source_id in (
                    item.get("source_candidate_ids", [])
                    or [item.get("candidate_id", "")]
                )
                if str(source_id)
            })
            winner["merged_candidate_ids"] = sorted({
                str(source_id)
                for item in variants
                for source_id in (
                    item.get("merged_candidate_ids", [])
                    or [item.get("candidate_id", "")]
                )
                if str(source_id)
            })
            winner["evidence_candidates"] = list(dict.fromkeys(
                str(item.get("evidence", "") or "").strip()
                for item in variants if str(item.get("evidence", "") or "").strip()
            ))[:3]
            winner["claim_instances"] = [
                copy.deepcopy(instance)
                for item in variants
                for instance in (
                    item.get("claim_instances", [])
                    or [{
                        "candidate_id": str(item.get("candidate_id", "") or ""),
                        "candidate_version": max(1, int(item.get("candidate_version", 1) or 1)),
                        "parent_candidate_id": str(item.get("parent_candidate_id", "") or ""),
                        "parent_version": max(0, int(item.get("parent_version", 0) or 0)),
                        "evidence": str(item.get("evidence", "") or ""),
                        "provenance": list(item.get("provenance", []) or []),
                        "claim_role": str(item.get("claim_role", "") or "CURRENT_FINDING"),
                    }]
                )
            ]
            output.append(winner)
        return output
