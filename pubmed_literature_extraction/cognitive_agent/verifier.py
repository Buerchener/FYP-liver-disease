#!/usr/bin/env python3
"""
cognitive_agent/verifier.py — Phase 3: 图谱溯源验证

将 LangExtract 提取结果与 Neo4j 已有知识交叉验证：
- 实体溯源 (EXACT_MATCH / FUZZY_MATCH / NOVEL)
- 关系验证 (KNOWN / INVERTED / CONTRADICTING / NOVEL)
- Schema 合规检查
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from cognitive_agent.memory.kg_memory import KGMemory
from cognitive_agent.schema.relation_signatures import (
    LITERATURE_CANDIDATE_SIGNATURES,
    ALLOWED_DIRECTIONS,
)
from cognitive_agent.schema.write_contract import SchemaAdapter
from cognitive_agent.extraction_quality import (
    article_quality_flags,
    evaluate_relation_evidence,
    locate_contiguous,
    prepare_extraction,
    ratio_metric,
)
from cognitive_agent.relation_contract import (
    FACTUAL_REJECT_FLAGS,
    FACTUAL_REVIEW_FLAGS,
    JUDGE_BACKEND_NAMES,
    JUDGE_SEMANTIC_OVERRIDABLE_FLAGS,
    MODEL_OVERRIDABLE_WRITE_FLAGS,
    SEMANTIC_REJECT_FLAGS,
    SEMANTIC_REVIEW_FLAGS,
    TIERED_SEMANTIC_REJECT_FLAGS,
    TIERED_SEMANTIC_REVIEW_FLAGS,
    WRITE_REVIEW_FLAGS,
    WRITE_BLOCK_FLAGS,
)


@dataclass
class VerifiedEntity:
    mention: str
    entity_type: str
    neo4j_status: str  # EXACT_MATCH | FUZZY_MATCH | NOVEL
    neo4j_element_id: str = ""
    neo4j_node_id: str = ""
    confidence: float = 0.0
    attributes: dict = field(default_factory=dict)  # ← v2: 保留原始提取属性
    candidates: list[dict] = field(default_factory=list)
    ambiguity_reason: str = ""
    source_span: str = ""
    char_start: int = -1
    char_end: int = -1
    grounded: bool = False
    normalized_id: str = ""
    canonical_key: str = ""
    canonical_mentions: list[str] = field(default_factory=list)
    filter_status: str = "retained"
    filter_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "mention": self.mention,
            "type": self.entity_type,
            "neo4j_status": self.neo4j_status,
            "neo4j_node_id": self.neo4j_node_id,
            "confidence": self.confidence,
            "attributes": self.attributes,
            "candidates": self.candidates,
            "ambiguity_reason": self.ambiguity_reason,
            "source_span": self.source_span,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "grounded": self.grounded,
            "normalized_id": self.normalized_id,
            "canonical_key": self.canonical_key,
            "canonical_mentions": self.canonical_mentions,
            "filter_status": self.filter_status,
            "filter_reason": self.filter_reason,
        }


@dataclass
class VerifiedRelation:
    subject: str
    predicate: str
    object: str
    subject_type: str = ""
    object_type: str = ""
    neo4j_status: str = ""  # KNOWN | INVERTED | CONTRADICTING | NOVEL
    existing_rel_id: str = ""
    existing_confidence: float = 0.0
    existing_direction: str = ""
    # ``schema_valid`` is retained for existing semantic consumers.  It means
    # valid in the broad literature candidate schema, never Neo4j-writable.
    schema_valid: bool = False
    candidate_schema_valid: bool = False
    write_contract_valid: bool = False
    schema_gap_reasons: list[str] = field(default_factory=list)
    write_contract_version: str = ""
    import_ready: bool = False
    evidence: str = ""
    direction: str = ""
    negated: bool = False
    uncertain: bool = False
    quality_flags: list[str] = field(default_factory=list)
    evidence_char_start: int = -1
    evidence_char_end: int = -1
    evidence_contiguous: bool = False
    evidence_level: int = 3
    subject_grounded_in_evidence: bool = False
    object_grounded_in_evidence: bool = False
    trigger_present: bool = False
    direction_trigger_consistent: bool = False
    original_subject: str = ""
    original_object: str = ""
    endpoint_remapped: bool = False
    candidate_id: str = ""
    classifier_source: str = ""
    classifier_confidence: float = 0.0
    relation_probability: float = 0.0
    no_relation_probability: float = 0.0
    classifier_margin: float = 0.0
    predicate_candidates: dict[str, float] = field(default_factory=dict)
    evidence_unit_id: str = ""
    evidence_role: str = ""
    evidence_confidence: float = 0.0
    evidence_entailment: str = ""
    rule_score_delta: float = 0.0
    rule_matches: list[dict] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    subject_family: str = ""
    object_family: str = ""
    factual_status: str = "VALID"
    semantic_status: str = "UNVERIFIED"
    write_status: str = "UNASSESSED"
    claim_role: str = "CURRENT_FINDING"
    evidence_spans: list[dict] = field(default_factory=list)
    semantic_reasons: list[str] = field(default_factory=list)
    write_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class VerifiedExtraction:
    """Phase 3 验证后的提取结果"""
    pmid: str = ""
    entities: list[VerifiedEntity] = field(default_factory=list)
    relations: list[VerifiedRelation] = field(default_factory=list)
    causal_chains: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    raw_entities: list[dict] = field(default_factory=list)
    raw_relations: list[dict] = field(default_factory=list)
    filtered_entities: list[dict] = field(default_factory=list)
    merged_entities: list[dict] = field(default_factory=list)
    mention_to_canonical: dict[str, str] = field(default_factory=dict)
    canonical_key_to_entity: dict[str, dict] = field(default_factory=dict)
    remapped_relations: list[dict] = field(default_factory=list)
    unresolved_relations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "causal_chain_count": len(self.causal_chains),
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "causal_chains": self.causal_chains,
            "summary": self.summary,
            "review": {
                "raw_entities": self.raw_entities,
                "raw_relations": self.raw_relations,
                "filtered_entities": self.filtered_entities,
                "merged_entities": self.merged_entities,
                "mention_to_canonical": self.mention_to_canonical,
                "canonical_key_to_entity": self.canonical_key_to_entity,
                "remapped_relations": self.remapped_relations,
                "unresolved_relations": self.unresolved_relations,
            },
        }


class KGVerifier:
    """Phase 3: 图谱溯源验证器"""

    def __init__(self, kg_memory: KGMemory, verification_policy: str = "legacy"):
        if verification_policy not in {"legacy", "tiered-v2"}:
            raise ValueError("verification_policy must be legacy or tiered-v2")
        self.kg_memory = kg_memory
        self.verification_policy = verification_policy
        self.schema_adapter = SchemaAdapter()

    def verify(
        self,
        raw_entities: list[dict],
        raw_relations: list[dict],
        pmid: str = "",
        text: str = "",
    ) -> VerifiedExtraction:
        """
        验证提取结果，将实体和关系与 Neo4j 交叉比对。

        Args:
            raw_entities: Phase 2 提取的原始实体列表
            raw_relations: Phase 2 提取的原始关系列表
            pmid: PubMed ID
            text: Current article source text (title + abstract)

        Returns:
            VerifiedExtraction: 带 Neo4j 验证状态的提取结果
        """
        prepared = prepare_extraction(raw_entities, raw_relations, text=text)
        result = VerifiedExtraction(
            pmid=pmid,
            raw_entities=prepared.raw_entities,
            raw_relations=prepared.raw_relations,
            filtered_entities=prepared.filtered_entities,
            merged_entities=prepared.merged_entities,
            mention_to_canonical=prepared.mention_to_canonical,
            canonical_key_to_entity=prepared.canonical_key_to_entity,
            remapped_relations=prepared.remapped_relations,
            unresolved_relations=prepared.unresolved_relations,
        )

        # 实体验证
        verified_entities = {}
        for ent in prepared.entities:
            ve = self._verify_entity(ent)
            key = f"{ve.mention}|{ve.entity_type}"
            verified_entities[key] = ve
            result.entities.append(ve)

        # 关系验证
        standardized_entities = {
            (entity.mention, entity.entity_type): entity
            for entity in result.entities
        }
        article_flags = article_quality_flags(text)
        for rel in prepared.relations:
            rel.setdefault("quality_flags", [])
            rel["quality_flags"] = sorted(set(rel["quality_flags"]) | article_flags)
            vr = self._verify_relation(
                rel,
                verified_entities,
                text=text,
                aliases_by_canonical=prepared.aliases_by_canonical,
            )
            self.schema_adapter.apply(
                vr,
                standardized_entities=standardized_entities,
            )
            result.relations.append(vr)

        result.summary = self._build_summary(result)

        return result

    @staticmethod
    def _metric_value(metric: dict, empty_default: float = 1.0) -> float:
        value = metric.get("value")
        return empty_default if value is None else float(value)

    def _build_summary(self, result: VerifiedExtraction) -> dict:
        """Build separated structural, semantic, evidence, and linking metrics."""
        total_entities = len(result.raw_entities)
        retained_entities = len(result.entities)
        total_relations = len(result.relations)
        schema_valid = sum(1 for rel in result.relations if rel.schema_valid)
        write_contract_valid = sum(
            1 for rel in result.relations if rel.write_contract_valid
        )
        import_ready = sum(1 for rel in result.relations if rel.import_ready)
        semantic_accepted = sum(
            1 for rel in result.relations if rel.semantic_status == "ACCEPTED"
        )
        semantic_review = sum(
            1 for rel in result.relations if rel.semantic_status == "REVIEW"
        )
        semantic_rejected = sum(
            1 for rel in result.relations if rel.semantic_status == "REJECTED"
        )
        semantic_only = sum(
            1 for rel in result.relations if rel.write_status == "SEMANTIC_ONLY"
        )
        human_review = sum(
            1 for rel in result.relations if rel.write_status == "HUMAN_REVIEW"
        )
        blocked_relations = sum(
            1 for rel in result.relations if rel.write_status == "BLOCKED"
        )
        filtered_generic = sum(
            1 for entity in result.filtered_entities
            if entity.get("filter_reason") in {
                "generic_or_context_term", "method_or_database_term", "statistical_term",
                "bare_process_without_direct_evidence",
            }
        )
        type_pass = sum(
            1 for entity in result.raw_entities
            if not any(
                item.get("mention") == entity.get("mention")
                and item.get("filter_reason") in {
                    "invalid_entity_type", "tissue_context_not_anatomy",
                    "generic_tissue_term", "non_specific_cell_type",
                    "cell_type_not_specific", "generic_pathway_term",
                    "pathological_process_not_disease",
                }
                for item in result.filtered_entities
            )
        )
        evidence_exact = sum(1 for rel in result.relations if rel.evidence_contiguous)
        endpoint_covered = sum(
            1 for rel in result.relations
            if rel.subject_grounded_in_evidence and rel.object_grounded_in_evidence
        )
        direction_consistent = sum(
            1 for rel in result.relations
            if rel.trigger_present and rel.direction_trigger_consistent
        )
        strong_evidence = sum(1 for rel in result.relations if rel.evidence_level in {1, 2})
        filtered_endpoints = sum(1 for rel in result.relations if "filtered_endpoint" in rel.quality_flags)
        unresolved_endpoints = sum(1 for rel in result.relations if "unresolved_endpoint" in rel.quality_flags)
        weak_evidence = sum(1 for rel in result.relations if "weak_evidence" in rel.quality_flags)
        negated = sum(1 for rel in result.relations if rel.negated)
        uncertain = sum(1 for rel in result.relations if rel.uncertain)
        required_fields = sum(
            1 for rel in result.relations
            if rel.subject and rel.object and rel.predicate and rel.subject_type and rel.object_type
        )

        metrics = {
            "valid_entity_ratio": ratio_metric(retained_entities, total_entities),
            "generic_entity_rate": ratio_metric(filtered_generic, total_entities),
            "duplicate_entity_rate": ratio_metric(len(result.merged_entities), total_entities),
            "type_constraint_pass_rate": ratio_metric(type_pass, total_entities),
            "relation_evidence_exactness": ratio_metric(evidence_exact, total_relations),
            "relation_endpoint_coverage": ratio_metric(endpoint_covered, total_relations),
            "direction_trigger_consistency": ratio_metric(direction_consistent, total_relations),
            "filtered_endpoint_rate": ratio_metric(filtered_endpoints, total_relations),
            "unresolved_endpoint_rate": ratio_metric(unresolved_endpoints, total_relations),
            "weak_evidence_rate": ratio_metric(weak_evidence, total_relations),
            "negated_relation_count": ratio_metric(negated, total_relations),
            "uncertain_relation_count": ratio_metric(uncertain, total_relations),
            "import_ready_relation_rate": ratio_metric(import_ready, total_relations),
            "main_kg_write_contract_rate": ratio_metric(
                write_contract_valid, total_relations
            ),
        }
        schema_metric = ratio_metric(schema_valid, total_relations)
        required_metric = ratio_metric(required_fields, total_relations)
        structural_score = round(
            0.6 * self._metric_value(schema_metric)
            + 0.4 * self._metric_value(required_metric),
            4,
        )
        semantic_score = round(
            0.35 * self._metric_value(metrics["valid_entity_ratio"])
            + 0.25 * self._metric_value(metrics["type_constraint_pass_rate"])
            + 0.2 * self._metric_value(metrics["relation_endpoint_coverage"])
            + 0.2 * self._metric_value(metrics["direction_trigger_consistency"]),
            4,
        )
        evidence_score = round(
            0.4 * self._metric_value(metrics["relation_evidence_exactness"])
            + 0.35 * self._metric_value(metrics["relation_endpoint_coverage"])
            + 0.25 * self._metric_value(ratio_metric(strong_evidence, total_relations)),
            4,
        )
        linking_stats = {
            "total": retained_entities,
            "exact_matches": sum(1 for e in result.entities if e.neo4j_status == "EXACT_MATCH"),
            "fuzzy_matches": sum(1 for e in result.entities if e.neo4j_status == "FUZZY_MATCH"),
            "ambiguous": sum(1 for e in result.entities if e.neo4j_status == "AMBIGUOUS"),
            "novel": sum(1 for e in result.entities if e.neo4j_status == "NOVEL"),
        }
        return {
            "total_entities": len(result.entities),
            "raw_entity_count": total_entities,
            "filtered_entity_count": len(result.filtered_entities),
            "merged_entity_count": len(result.merged_entities),
            "exact_matches": sum(1 for e in result.entities if e.neo4j_status == "EXACT_MATCH"),
            "fuzzy_matches": sum(1 for e in result.entities if e.neo4j_status == "FUZZY_MATCH"),
            "ambiguous_entities": sum(1 for e in result.entities if e.neo4j_status == "AMBIGUOUS"),
            "novel_entities": sum(1 for e in result.entities if e.neo4j_status == "NOVEL"),
            "total_relations": len(result.relations),
            "known_relations": sum(1 for r in result.relations if r.neo4j_status == "KNOWN"),
            "novel_relations": sum(1 for r in result.relations if r.neo4j_status == "NOVEL"),
            "contradicting_relations": sum(1 for r in result.relations if r.neo4j_status == "CONTRADICTING"),
            "schema_valid": schema_valid,
            "candidate_schema_valid": schema_valid,
            "write_contract_valid": write_contract_valid,
            "import_ready": import_ready,
            "semantic_accepted": semantic_accepted,
            "semantic_review": semantic_review,
            "semantic_rejected": semantic_rejected,
            "semantic_only": semantic_only,
            "human_review": human_review,
            "blocked_relations": blocked_relations,
            "structural_score": structural_score,
            "semantic_score": semantic_score,
            "evidence_score": evidence_score,
            "quality_metrics": metrics,
            "linking_stats": linking_stats,
        }

    @staticmethod
    def _normalize_confidence(value) -> float:
        """Normalize model confidence while tolerating malformed output."""
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.7
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _normalize_claim_role(value: str, flags: set[str]) -> str:
        role = str(value or "").strip().upper()
        mapping = {
            "DIRECT_FINDING": "CURRENT_FINDING",
            "CURRENT": "CURRENT_FINDING",
            "CURRENT_FINDING": "CURRENT_FINDING",
            "PRIOR_WORK": "PRIOR_WORK",
            "BACKGROUND": "BACKGROUND",
            "BACKGROUND_ONLY": "BACKGROUND",
            "METHOD": "METHOD",
            "METHOD_ONLY": "METHOD",
            "PREDICTION": "PREDICTION",
            "PREDICTION_ONLY": "PREDICTION",
            "OBJECTIVE": "BACKGROUND",
            "OBJECTIVE_ONLY": "BACKGROUND",
        }
        if role in mapping:
            return mapping[role]
        if flags & {"background_only", "objective_only"}:
            return "BACKGROUND"
        if flags & {"method_only", "method_section_only"}:
            return "METHOD"
        if "prediction_only" in flags:
            return "PREDICTION"
        if "non_current_finding_role" in flags:
            return "BACKGROUND"
        return "CURRENT_FINDING"

    @staticmethod
    def _model_dual_endorsed(flags: set[str]) -> bool:
        """Return True only for independent DeepSeek + Qwen style agreement."""
        return bool(
            "adjudicator_entailed" in flags
            and (
                "critic_approved" in flags
                or "qwen_critic_approved" in flags
                or "qwen_approved" in flags
                or "dual_model_entailed" in flags
            )
        )

    @staticmethod
    def _evidence_spans(evidence: str, text: str) -> list[dict]:
        """Locate up to three quoted evidence fragments for audit."""
        if not evidence or not text:
            return []
        fragments = [
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+|\n+", evidence)
            if item.strip()
        ]
        if len(fragments) <= 1:
            fragments = [evidence.strip()]
        spans: list[dict] = []
        sentence_boundaries = [
            (match.start(), match.end())
            for match in re.finditer(r"[^.!?\n]+[.!?]?", text)
        ]
        for fragment in fragments[:3]:
            grounded, start, end = locate_contiguous(fragment, text)
            if not grounded:
                continue
            sentence_index = -1
            for index, (s_start, s_end) in enumerate(sentence_boundaries):
                if s_start <= start < s_end:
                    sentence_index = index
                    break
            spans.append({
                "text": fragment,
                "start": start,
                "end": end,
                "sentence_index": sentence_index,
            })
        return spans

    def _verify_entity(self, entity: dict) -> VerifiedEntity:
        """验证单个实体"""
        mention = entity.get("mention", "")
        etype = entity.get("type", "")
        raw_attrs = entity.get("attributes", {}) or {}
        normalized_id = raw_attrs.get("normalized_id", entity.get("normalized_id", ""))

        ve = VerifiedEntity(
            mention=mention,
            entity_type=etype,
            neo4j_status="NOVEL",
            attributes=raw_attrs,
            confidence=self._normalize_confidence(entity.get("confidence", 0.7)),
            source_span=entity.get("source_span", ""),
            char_start=entity.get("char_start", -1),
            char_end=entity.get("char_end", -1),
            grounded=bool(entity.get("grounded", False)),
            normalized_id=normalized_id,
            canonical_key=entity.get("canonical_key", ""),
            canonical_mentions=list(entity.get("canonical_mentions", [mention])),
            filter_status=entity.get("filter_status", "retained"),
            filter_reason=entity.get("filter_reason", ""),
        )

        if not self.kg_memory.is_connected:
            return ve

        # 精确匹配
        match = self.kg_memory.find_entity(mention, entity_type=etype, normalized_id=normalized_id)
        if match:
            ve.neo4j_status = "EXACT_MATCH"
            ve.neo4j_element_id = match["element_id"]
            ve.neo4j_node_id = match["node_id"]
            return ve

        # 模糊匹配
        fuzzy = self.kg_memory.find_entity_fuzzy(mention, entity_type=etype)
        if fuzzy:
            candidates = fuzzy.get("fuzzy_matches", [])
            ambiguous = fuzzy.get("ambiguous", False)
            if ambiguous:
                ve.neo4j_status = "AMBIGUOUS"
                ve.ambiguity_reason = "multiple close fuzzy candidates"
            else:
                ve.neo4j_status = "FUZZY_MATCH"
                ve.neo4j_element_id = fuzzy.get("element_id", "")
                ve.neo4j_node_id = fuzzy.get("node_id", "")
            ve.candidates = [
                {k: c.get(k, "") for k in ("name", "node_id", "score", "match_kind")}
                for c in candidates
            ]

        return ve

    def _check_relation(self, subject: str, predicate: str, object_name: str, **kwargs) -> dict | None:
        """Call old fake KG implementations as well as the richer KG API."""
        try:
            return self.kg_memory.check_relation_exists(
                subject, predicate, object_name, **kwargs
            )
        except TypeError:
            return self.kg_memory.check_relation_exists(subject, predicate, object_name)

    @staticmethod
    def _directions_conflict(new_direction: str, old_direction: str) -> bool:
        opposites = {
            ("increase", "decrease"), ("decrease", "increase"),
            ("positive", "negative"), ("negative", "positive"),
        }
        return (new_direction, old_direction) in opposites

    def _endpoint_resolves(self, mention: str, entity_type: str) -> bool:
        """Allow a relation endpoint backed by an existing KG node."""
        if not self.kg_memory.is_connected or not mention or not entity_type:
            return False
        try:
            return bool(self.kg_memory.find_entity(mention, entity_type=entity_type))
        except Exception:
            return False

    def _verify_relation(
        self,
        relation: dict,
        verified_entities: dict[str, VerifiedEntity],
        text: str = "",
        aliases_by_canonical: dict[str, list[str]] | None = None,
    ) -> VerifiedRelation:
        """验证单条关系"""
        vr = VerifiedRelation(
            subject=relation.get("subject", ""),
            predicate=relation.get("predicate", ""),
            object=relation.get("object", ""),
            subject_type=relation.get("subject_type", ""),
            object_type=relation.get("object_type", ""),
            neo4j_status="NOVEL",
            evidence=relation.get("evidence", ""),
            direction=relation.get("direction", "unknown"),
            negated=relation.get("negated", False),
            uncertain=relation.get("uncertain", False),
            quality_flags=list(relation.get("quality_flags", [])),
            original_subject=relation.get("original_subject", relation.get("subject", "")),
            original_object=relation.get("original_object", relation.get("object", "")),
            endpoint_remapped=bool(relation.get("endpoint_remapped", False)),
            candidate_id=str(relation.get("candidate_id", "") or ""),
            classifier_source=str(relation.get("classifier_source", "") or ""),
            classifier_confidence=self._normalize_confidence(
                relation.get("classifier_confidence", 0.0)
            ),
            relation_probability=self._normalize_confidence(
                relation.get("relation_probability", 0.0)
            ),
            no_relation_probability=self._normalize_confidence(
                relation.get("no_relation_probability", 0.0)
            ),
            classifier_margin=self._normalize_confidence(
                relation.get("classifier_margin", 0.0)
            ),
            predicate_candidates=dict(relation.get("predicate_candidates", {}) or {}),
            evidence_unit_id=str(relation.get("evidence_unit_id", "") or ""),
            evidence_role=str(relation.get("evidence_role", "") or ""),
            evidence_confidence=self._normalize_confidence(
                relation.get("evidence_confidence", 0.0)
            ),
            evidence_entailment=str(relation.get("evidence_entailment", "") or ""),
            rule_score_delta=float(relation.get("rule_score_delta", 0.0) or 0.0),
            rule_matches=list(relation.get("rule_matches", []) or []),
            provenance=list(relation.get("provenance", []) or []),
            subject_family=str(relation.get("subject_family", "") or ""),
            object_family=str(relation.get("object_family", "") or ""),
            claim_role=str(relation.get("claim_role", "") or "CURRENT_FINDING"),
        )

        # ── 1. Schema 合规检查 ──
        allowed_pairs = LITERATURE_CANDIDATE_SIGNATURES.get(vr.predicate, set())
        pair = (vr.subject_type, vr.object_type)
        vr.candidate_schema_valid = pair in allowed_pairs
        vr.schema_valid = vr.candidate_schema_valid

        if not vr.candidate_schema_valid:
            vr.quality_flags.append("schema_mismatch")

        # ── 2. 方向检查 ──
        if vr.direction not in ALLOWED_DIRECTIONS:
            vr.quality_flags.append("invalid_direction")

        # ── 3. 否定/不确定标记 ──
        if vr.negated:
            vr.quality_flags.append("negated")
        if vr.uncertain:
            vr.quality_flags.append("uncertain")

        # ── 4. 物种检查 ──
        species = relation.get("species", "")
        if species and species.lower() in {"mus musculus", "mouse", "mice", "rat", "rattus norvegicus"}:
            vr.quality_flags.append("non_human")

        # ── 5. Neo4j 关系验证 ──
        if self.kg_memory.is_connected and vr.candidate_schema_valid:
            existing = self._check_relation(
                vr.subject, vr.predicate, vr.object,
                subject_type=vr.subject_type, object_type=vr.object_type,
            )
            if existing:
                vr.existing_rel_id = existing.get("rel_element_id", "")
                vr.existing_confidence = self._normalize_confidence(
                    existing.get("confidence", 0.7)
                )
                vr.existing_direction = existing.get("direction", "")
                if self._directions_conflict(vr.direction, vr.existing_direction):
                    vr.neo4j_status = "CONTRADICTING"
                    vr.quality_flags.extend(["contradiction", "opposite_direction"])
                else:
                    vr.neo4j_status = "KNOWN"
                    vr.quality_flags.append("duplicate_evidence")
            else:
                inverse = self._check_relation(
                    vr.object, vr.predicate, vr.subject,
                    subject_type=vr.object_type, object_type=vr.subject_type,
                )
                if inverse:
                    vr.neo4j_status = "INVERTED"
                    vr.existing_rel_id = inverse.get("rel_element_id", "")
                    vr.existing_confidence = self._normalize_confidence(
                        inverse.get("confidence", 0.7)
                    )
                    vr.existing_direction = inverse.get("direction", "")
                    vr.quality_flags.append("inverted_existing_relation")

        # ── 6. 实体链接状态 ──
        subj_key = f"{vr.subject}|{vr.subject_type}"
        obj_key = f"{vr.object}|{vr.object_type}"
        subj_entity = verified_entities.get(subj_key)
        obj_entity = verified_entities.get(obj_key)

        # Current-article endpoints are mandatory.  Existing KG nodes are linking
        # context only and cannot replace an entity grounded in this article.
        if not subj_entity:
            vr.quality_flags.extend(["subject_endpoint_missing", "unresolved_endpoint"])
        if not obj_entity:
            vr.quality_flags.extend(["object_endpoint_missing", "unresolved_endpoint"])

        if subj_entity and subj_entity.neo4j_status == "NOVEL":
            vr.quality_flags.append("subject_novel")
        if obj_entity and obj_entity.neo4j_status == "NOVEL":
            vr.quality_flags.append("object_novel")
        if subj_entity and subj_entity.neo4j_status == "AMBIGUOUS":
            vr.quality_flags.extend(["subject_ambiguous", "ambiguous_endpoint"])
        if obj_entity and obj_entity.neo4j_status == "AMBIGUOUS":
            vr.quality_flags.extend(["object_ambiguous", "ambiguous_endpoint"])

        # ── 7. Current-article evidence verification ──
        evidence_result = evaluate_relation_evidence(
            {**relation, **vr.to_dict(), "quality_flags": vr.quality_flags},
            text=text,
            aliases_by_canonical=aliases_by_canonical or {},
        )
        for key, value in evidence_result.items():
            if hasattr(vr, key):
                setattr(vr, key, value)
        vr.quality_flags = sorted(set(evidence_result["quality_flags"]))

        vr.evidence_spans = self._evidence_spans(vr.evidence, text)

        # ── 8. Separate semantic validity from Safe Write eligibility ──
        # Animal/cell/case-report evidence can be a correct semantic relation
        # while remaining ineligible for automatic graph import.
        flags = set(vr.quality_flags)
        vr.claim_role = self._normalize_claim_role(vr.claim_role, flags)
        if vr.claim_role != "CURRENT_FINDING":
            flags.add("non_current_finding_role")
            vr.quality_flags = sorted(flags)
        if self.verification_policy == "tiered-v2":
            self._apply_tiered_v2_status(vr, flags)
            return vr
        self._apply_legacy_status(vr, flags)
        return vr

    def _apply_legacy_status(self, vr: VerifiedRelation, flags: set[str]) -> None:
        semantic_reject = sorted(flags & SEMANTIC_REJECT_FLAGS)
        semantic_review = sorted(flags & SEMANTIC_REVIEW_FLAGS)
        # A single-model judge-ENTAILED proposal is a semantic PROPOSAL, not
        # an override: it never clears the deterministic trigger/weak-evidence
        # flags by itself.  Only the independent second-model endorsement
        # (`adjudicator_entailed`, set by a DeepSeek KEEP during bounded
        # adjudication) may clear them.  When the judge's proposal conflicts
        # with the deterministic flags the conflict is recorded as
        # `judge_verifier_conflict` and the relation goes to REVIEW, which
        # routes it to the independent adjudicator.  Hard blockers
        # (SEMANTIC_REJECT_FLAGS) are never touched and the Safe Write gate
        # is unchanged: every overridable flag still blocks import_ready.
        if (
            vr.classifier_source in JUDGE_BACKEND_NAMES
            and vr.evidence_entailment == "ENTAILED"
            and "adjudicator_entailed" in flags
        ):
            semantic_review = sorted(
                set(semantic_review) - JUDGE_SEMANTIC_OVERRIDABLE_FLAGS
            )
        elif (
            vr.classifier_source in JUDGE_BACKEND_NAMES
            and vr.evidence_entailment == "ENTAILED"
            and set(semantic_review) & JUDGE_SEMANTIC_OVERRIDABLE_FLAGS
        ):
            vr.quality_flags.append("judge_verifier_conflict")
            vr.quality_flags = sorted(set(vr.quality_flags))
            semantic_review = sorted(
                set(semantic_review) | {"judge_verifier_conflict"}
            )
        vr.factual_status = "REJECTED" if semantic_reject else "VALID"
        if semantic_reject:
            vr.semantic_status = "REJECTED"
            vr.semantic_reasons = semantic_reject
        elif semantic_review:
            vr.semantic_status = "REVIEW"
            vr.semantic_reasons = semantic_review
        else:
            vr.semantic_status = "ACCEPTED"
            vr.semantic_reasons = []
        vr.write_reasons = sorted(flags & WRITE_BLOCK_FLAGS)
        vr.import_ready = (
            vr.semantic_status == "ACCEPTED"
            and vr.candidate_schema_valid
            and bool(vr.subject and vr.object)
            and not vr.negated
            and not vr.uncertain
            and vr.neo4j_status != "AMBIGUOUS"
            and vr.evidence_level in {1, 2}
            and not vr.write_reasons
        )
        if vr.import_ready:
            vr.write_status = "IMPORT_READY"
        elif vr.semantic_status == "REJECTED":
            vr.write_status = "BLOCKED"
        else:
            vr.write_status = "SEMANTIC_ONLY"

    def _apply_tiered_v2_status(self, vr: VerifiedRelation, flags: set[str]) -> None:
        hard_reject = set(flags & FACTUAL_REJECT_FLAGS)
        factual_review = set(flags & FACTUAL_REVIEW_FLAGS)
        semantic_reject = set(flags & TIERED_SEMANTIC_REJECT_FLAGS)
        semantic_review = set(flags & TIERED_SEMANTIC_REVIEW_FLAGS)
        dual_endorsed = self._model_dual_endorsed(flags)

        # Independent model endorsement may clear semantic uncertainty and
        # write-scope flags, but never the four non-overridable factual gates.
        if dual_endorsed or (
            vr.classifier_source in JUDGE_BACKEND_NAMES
            and vr.evidence_entailment == "ENTAILED"
            and "adjudicator_entailed" in flags
        ):
            semantic_review -= JUDGE_SEMANTIC_OVERRIDABLE_FLAGS
            if dual_endorsed:
                semantic_review -= MODEL_OVERRIDABLE_WRITE_FLAGS

        if hard_reject:
            vr.factual_status = "REJECTED"
        elif factual_review:
            vr.factual_status = "REVIEW"
        else:
            vr.factual_status = "VALID"

        if vr.factual_status == "REJECTED" or semantic_reject:
            vr.semantic_status = "REJECTED"
            vr.semantic_reasons = sorted(hard_reject | semantic_reject)
        elif semantic_review:
            vr.semantic_status = "REVIEW"
            vr.semantic_reasons = sorted(semantic_review)
        else:
            vr.semantic_status = "ACCEPTED"
            vr.semantic_reasons = []

        write_reasons = set(flags & WRITE_REVIEW_FLAGS)
        if dual_endorsed:
            write_reasons -= MODEL_OVERRIDABLE_WRITE_FLAGS
        if vr.factual_status == "REJECTED":
            write_reasons |= hard_reject
        vr.write_reasons = sorted(write_reasons)

        fast_path = (
            vr.factual_status == "VALID"
            and vr.semantic_status == "ACCEPTED"
            and vr.candidate_schema_valid
            and bool(vr.subject and vr.object)
            and not vr.negated
            and vr.evidence_contiguous
            and vr.subject_grounded_in_evidence
            and vr.object_grounded_in_evidence
            and vr.evidence_level in {1, 2}
            and not vr.write_reasons
        )
        scoped_import = (
            dual_endorsed
            and vr.factual_status == "VALID"
            and vr.semantic_status == "ACCEPTED"
            and vr.candidate_schema_valid
            and bool(vr.subject and vr.object)
            and vr.evidence_contiguous
            and not (FACTUAL_REJECT_FLAGS & flags)
        )
        vr.import_ready = bool(fast_path or scoped_import)
        if vr.import_ready:
            vr.write_status = "IMPORT_READY"
        elif vr.factual_status == "REJECTED" or vr.semantic_status == "REJECTED":
            vr.write_status = "BLOCKED"
        elif vr.factual_status == "REVIEW" or vr.semantic_status == "REVIEW":
            vr.write_status = "HUMAN_REVIEW"
        else:
            vr.write_status = "SEMANTIC_ONLY"
