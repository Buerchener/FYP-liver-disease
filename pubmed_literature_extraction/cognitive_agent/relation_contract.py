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
from dataclasses import dataclass, field
from typing import Any

from cognitive_agent.abbreviation_detector import AbbreviationDetector
from cognitive_agent.extraction_quality import locate_contiguous, normalize_surface


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
})

# Write policy is deliberately stricter than semantic acceptance.
WRITE_BLOCK_FLAGS = frozenset({
    *SEMANTIC_REJECT_FLAGS, *SEMANTIC_REVIEW_FLAGS,
    "invalid_direction", "non_human", "non_human_article",
    "article_out_of_scope", "second_llm_rejected", "review_article",
})


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
        raw = "|".join((
            normalize_surface(relation.get("subject")),
            str(relation.get("subject_type", "")),
            str(relation.get("predicate", "")).upper(),
            normalize_surface(relation.get("object")),
            str(relation.get("object_type", "")),
            normalize_surface(relation.get("evidence", "")),
        ))
        return "c-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

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
        output = copy.deepcopy(relation)
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
        })
        output["candidate_id"] = str(output.get("candidate_id", "")) or self._candidate_id(output)
        return output

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

        # One typed triple is one semantic candidate.  Keep its best grounded,
        # endpoint-covering span and union all provenance.
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
            winner["candidate_id"] = self._candidate_id(winner)
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
            output.append(winner)
        return output
