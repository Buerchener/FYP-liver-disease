#!/usr/bin/env python3
"""Bounded, read-only Neo4j context retrieval for Phase C."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from cognitive_agent.extraction_quality import normalize_surface
from cognitive_agent.memory.kg_memory import ALLOWED_ENTITY_TYPES, KGMemory


RAG_USAGE_POLICY = {
    "read_only": True,
    "memory_only": True,
    "allowed_uses": ["entity_linking", "type_disambiguation", "synonym_hint", "conflict_hint"],
    "current_article_evidence": False,
    "may_auto_accept_relation": False,
    "may_write_neo4j": False,
}


@dataclass(frozen=True)
class RAGConfig:
    enabled: bool = False
    max_entities: int = 12
    max_candidates_per_entity: int = 3
    max_total_candidates: int = 20
    max_neighbors_per_candidate: int = 4
    max_synonyms_per_candidate: int = 8
    max_evidence_chars: int = 500
    max_total_chars: int = 6000


@dataclass
class RAGContext:
    status: str = "DISABLED"  # DISABLED | OFFLINE | EMPTY | OK | ERROR
    entity_contexts: list[dict] = field(default_factory=list)
    candidate_count: int = 0
    total_chars: int = 0
    truncated: bool = False
    error: str = ""
    limits: dict = field(default_factory=dict)
    usage_policy: dict = field(default_factory=lambda: dict(RAG_USAGE_POLICY))
    schema_profile: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "entity_contexts": self.entity_contexts,
            "candidate_count": self.candidate_count,
            "total_chars": self.total_chars,
            "truncated": self.truncated,
            "error": self.error,
            "limits": self.limits,
            "usage_policy": self.usage_policy,
            "schema_profile": self.schema_profile,
        }


class ControlledNeo4jRAG:
    """Build small context cards without treating graph memory as evidence."""

    def __init__(self, kg_memory: KGMemory, config: RAGConfig | None = None):
        self.kg_memory = kg_memory
        self.config = config or RAGConfig()

    def build(
        self,
        entities: list[dict],
        known_entities: dict[str, dict] | None = None,
        focus_relations: list[dict] | None = None,
    ) -> RAGContext:
        context = RAGContext(limits=self._limits())
        if not self.config.enabled:
            return context
        if not self.kg_memory.is_connected:
            context.status = "OFFLINE"
            return context
        schema_reader = getattr(self.kg_memory, "get_schema_profile", None)
        context.schema_profile = schema_reader() if callable(schema_reader) else {}

        inputs = self._input_entities(entities, known_entities or {})
        if focus_relations:
            focus_keys = self._focus_entity_keys(focus_relations)
            inputs = [
                item for item in inputs
                if (normalize_surface(item["mention"]), item["entity_type"]) in focus_keys
            ]
        if not inputs:
            context.status = "EMPTY"
            return context

        try:
            total_candidates = 0
            total_chars = 0
            for item in inputs[: self.config.max_entities]:
                if total_candidates >= self.config.max_total_candidates:
                    context.truncated = True
                    break
                entity_context = {
                    "mention": item["mention"],
                    "entity_type": item["entity_type"],
                    "normalized_id": item.get("normalized_id", ""),
                    "candidates": [],
                }
                candidates = self._retrieve_candidates(item)
                remaining = self.config.max_total_candidates - total_candidates
                for candidate in candidates[: min(self.config.max_candidates_per_entity, remaining)]:
                    card = self._hydrate_candidate(candidate)
                    projected = len(json.dumps(card, ensure_ascii=False, default=str))
                    if total_chars + projected > self.config.max_total_chars:
                        context.truncated = True
                        continue
                    entity_context["candidates"].append(card)
                    total_candidates += 1
                    total_chars += projected
                if entity_context["candidates"]:
                    context.entity_contexts.append(entity_context)

            # Count the complete serialized context, including wrappers.  If
            # wrappers push it over the cap, remove candidates from the tail.
            serialized_chars = len(json.dumps(
                {
                    "usage_policy": context.usage_policy,
                    "entity_contexts": context.entity_contexts,
                },
                ensure_ascii=False,
                default=str,
            ))
            while context.entity_contexts and serialized_chars > self.config.max_total_chars:
                tail = context.entity_contexts[-1]
                if tail["candidates"]:
                    tail["candidates"].pop()
                    total_candidates -= 1
                    context.truncated = True
                if not tail["candidates"]:
                    context.entity_contexts.pop()
                serialized_chars = len(json.dumps(
                    {
                        "usage_policy": context.usage_policy,
                        "entity_contexts": context.entity_contexts,
                    },
                    ensure_ascii=False,
                    default=str,
                ))
            context.candidate_count = total_candidates
            context.total_chars = serialized_chars if total_candidates else 0
            context.status = "OK" if total_candidates else "EMPTY"
            return context
        except Exception as exc:
            # Query errors never block extraction or collaboration and never
            # leak partial graph context into the model input.
            context.status = "ERROR"
            context.entity_contexts = []
            context.candidate_count = 0
            context.total_chars = 0
            context.error = str(exc)[:500]
            return context

    @staticmethod
    def _focus_entity_keys(relations: list[dict]) -> set[tuple[str, str]]:
        """Select endpoints only from the most decision-relevant relation queries."""
        hard = {
            "schema_mismatch", "evidence_not_contiguous", "empty_evidence",
            "subject_endpoint_missing", "object_endpoint_missing", "filtered_endpoint",
            "unresolved_endpoint", "method_only", "prediction_only",
        }
        ranked = sorted(
            relations,
            key=lambda item: (
                not bool(item.get("import_ready")),
                len(set(item.get("quality_flags", []) or []) & hard),
            ),
        )
        keys: set[tuple[str, str]] = set()
        for relation in ranked[:6]:
            flags = set(relation.get("quality_flags", []) or [])
            if flags & hard:
                continue
            for side in ("subject", "object"):
                mention = str(relation.get(side, "") or "").strip()
                entity_type = str(relation.get(f"{side}_type", "") or "")
                if mention and entity_type in ALLOWED_ENTITY_TYPES:
                    keys.add((normalize_surface(mention), entity_type))
        return keys

    def _limits(self) -> dict:
        return {
            key: int(getattr(self.config, key))
            for key in (
                "max_entities", "max_candidates_per_entity", "max_total_candidates",
                "max_neighbors_per_candidate", "max_synonyms_per_candidate",
                "max_evidence_chars", "max_total_chars",
            )
        }

    def _input_entities(
        self,
        entities: list[dict],
        known_entities: dict[str, dict],
    ) -> list[dict]:
        selected: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for raw in entities:
            mention = str(raw.get("mention", "") or "").strip()
            entity_type = str(raw.get("type", raw.get("entity_type", "")) or "")
            if not mention or entity_type not in ALLOWED_ENTITY_TYPES:
                continue
            key = (normalize_surface(mention), entity_type)
            if key in seen:
                continue
            seen.add(key)
            attrs = raw.get("attributes", {}) or {}
            selected.append({
                "mention": mention,
                "entity_type": entity_type,
                "normalized_id": str(
                    raw.get("normalized_id", attrs.get("normalized_id", "")) or ""
                ),
            })
        for mention, raw in known_entities.items():
            labels = raw.get("labels", []) or []
            entity_type = str(labels[0] if labels else raw.get("entity_type", ""))
            key = (normalize_surface(mention), entity_type)
            if not mention or entity_type not in ALLOWED_ENTITY_TYPES or key in seen:
                continue
            seen.add(key)
            selected.append({
                "mention": mention,
                "entity_type": entity_type,
                "normalized_id": str(raw.get("normalized_id", "") or ""),
            })
        return selected

    def _retrieve_candidates(self, item: dict) -> list[dict]:
        mention = item["mention"]
        entity_type = item["entity_type"]
        normalized_id = item.get("normalized_id", "")
        candidates: list[dict] = []
        seen: set[str] = set()
        candidate_limit = max(0, self.config.max_candidates_per_entity)
        if not candidate_limit:
            return candidates

        def add(candidate: dict | None, method: str, default_score: float) -> None:
            if not candidate:
                return
            candidate_type = str(
                candidate.get("entity_type")
                or (candidate.get("labels", []) or [entity_type])[0]
            )
            key = str(
                candidate.get("element_id")
                or candidate.get("node_id")
                or f"{candidate_type}:{normalize_surface(candidate.get('name', ''))}"
            )
            if not key or key in seen:
                return
            seen.add(key)
            candidates.append({
                **candidate,
                "entity_type": candidate_type,
                "match_method": method,
                "score": float(candidate.get("score", default_score) or default_score),
            })

        # Required retrieval order: exact name → normalized ID → same type → fuzzy.
        add(
            self.kg_memory.find_entity(mention, entity_type=entity_type, normalized_id=""),
            "exact_name",
            1.0,
        )
        if normalized_id and len(candidates) < candidate_limit:
            add(
                self.kg_memory.find_entity("", entity_type=entity_type, normalized_id=normalized_id),
                "normalized_id",
                1.0,
            )
        if len(candidates) < candidate_limit:
            same_type = self.kg_memory.find_entity_fuzzy(mention, entity_type=entity_type) or {}
            for candidate in same_type.get("fuzzy_matches", []):
                add(candidate, "same_type_candidate", 0.75)
                if len(candidates) >= candidate_limit:
                    break
        if len(candidates) < candidate_limit:
            fuzzy = self.kg_memory.find_entity_fuzzy(mention, entity_type="") or {}
            for candidate in fuzzy.get("fuzzy_matches", []):
                add(candidate, "fuzzy_candidate", 0.60)
                if len(candidates) >= candidate_limit:
                    break
        return candidates

    def _hydrate_candidate(self, candidate: dict) -> dict:
        node_context = self.kg_memory.get_rag_entity_context(
            candidate.get("element_id", ""),
            relation_limit=self.config.max_neighbors_per_candidate,
            evidence_limit=self.config.max_evidence_chars,
        )
        synonyms: list[str] = []
        for value in node_context.get("synonyms", []) or []:
            text = str(value or "").strip()
            if text and text not in synonyms:
                synonyms.append(text[:200])
        node_id = str(candidate.get("node_id", "") or "")
        if candidate.get("entity_type") == "Disease" and node_id:
            for synonym, normalized_id in KGMemory.DISEASE_SYNONYMS.items():
                if normalized_id == node_id and synonym not in synonyms:
                    synonyms.append(synonym)
        relations = []
        for relation in (node_context.get("relations", []) or [])[
            : self.config.max_neighbors_per_candidate
        ]:
            relations.append({
                "predicate": str(relation.get("predicate", "") or "")[:100],
                "target_name": str(relation.get("target_name", "") or "")[:200],
                "target_type": str(relation.get("target_type", "") or "")[:50],
                "edge_orientation": str(relation.get("edge_orientation", "") or "")[:20],
                "direction": str(relation.get("direction", "") or "")[:50],
                "source_pmid": str(relation.get("source_pmid", "") or "")[:50],
                "source_evidence": str(relation.get("source_evidence", "") or "")[
                    : self.config.max_evidence_chars
                ],
                "memory_only": True,
            })
        return {
            "entity_name": str(candidate.get("name", "") or "")[:200],
            "entity_type": candidate.get("entity_type", ""),
            "normalized_id": node_id[:200],
            "synonyms": synonyms[: self.config.max_synonyms_per_candidate],
            "candidate_relations": relations,
            "match_method": candidate.get("match_method", ""),
            "score": round(float(candidate.get("score", 0.0) or 0.0), 4),
        }
