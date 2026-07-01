#!/usr/bin/env python3
"""
cognitive_agent/verifier.py — Phase 3: 图谱溯源验证

将 LangExtract 提取结果与 Neo4j 已有知识交叉验证：
- 实体溯源 (EXACT_MATCH / FUZZY_MATCH / NOVEL)
- 关系验证 (KNOWN / INVERTED / CONTRADICTING / NOVEL)
- Schema 合规检查
"""

from __future__ import annotations

from dataclasses import dataclass, field
from cognitive_agent.memory.kg_memory import KGMemory
from cognitive_agent.schema.relation_signatures import RELATION_SIGNATURES, ALLOWED_DIRECTIONS


@dataclass
class VerifiedEntity:
    mention: str
    entity_type: str
    neo4j_status: str  # EXACT_MATCH | FUZZY_MATCH | NOVEL
    neo4j_element_id: str = ""
    neo4j_node_id: str = ""
    confidence: float = 0.0
    attributes: dict = field(default_factory=dict)  # ← v2: 保留原始提取属性

    def to_dict(self) -> dict:
        return {
            "mention": self.mention,
            "type": self.entity_type,
            "neo4j_status": self.neo4j_status,
            "neo4j_node_id": self.neo4j_node_id,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }


@dataclass
class VerifiedRelation:
    subject: str
    predicate: str
    object: str
    subject_type: str = ""
    object_type: str = ""
    neo4j_status: str = ""  # KNOWN | CONTRADICTING | INVERTED | NOVEL
    existing_rel_id: str = ""
    existing_confidence: float = 0.0
    schema_valid: bool = False
    import_ready: bool = False
    evidence: str = ""
    direction: str = ""
    negated: bool = False
    uncertain: bool = False
    quality_flags: list[str] = field(default_factory=list)

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
        }


class KGVerifier:
    """Phase 3: 图谱溯源验证器"""

    def __init__(self, kg_memory: KGMemory):
        self.kg_memory = kg_memory

    def verify(
        self,
        raw_entities: list[dict],
        raw_relations: list[dict],
        pmid: str = "",
    ) -> VerifiedExtraction:
        """
        验证提取结果，将实体和关系与 Neo4j 交叉比对。

        Args:
            raw_entities: Phase 2 提取的原始实体列表
            raw_relations: Phase 2 提取的原始关系列表
            pmid: PubMed ID

        Returns:
            VerifiedExtraction: 带 Neo4j 验证状态的提取结果
        """
        result = VerifiedExtraction(pmid=pmid)

        # 实体验证
        verified_entities = {}
        for ent in raw_entities:
            ve = self._verify_entity(ent)
            key = f"{ve.mention}|{ve.entity_type}"
            verified_entities[key] = ve
            result.entities.append(ve)

        # 关系验证
        for rel in raw_relations:
            vr = self._verify_relation(rel, verified_entities)
            result.relations.append(vr)

        # 汇总统计
        result.summary = {
            "total_entities": len(result.entities),
            "exact_matches": sum(1 for e in result.entities if e.neo4j_status == "EXACT_MATCH"),
            "fuzzy_matches": sum(1 for e in result.entities if e.neo4j_status == "FUZZY_MATCH"),
            "novel_entities": sum(1 for e in result.entities if e.neo4j_status == "NOVEL"),
            "total_relations": len(result.relations),
            "known_relations": sum(1 for r in result.relations if r.neo4j_status == "KNOWN"),
            "novel_relations": sum(1 for r in result.relations if r.neo4j_status == "NOVEL"),
            "contradicting_relations": sum(1 for r in result.relations if r.neo4j_status == "CONTRADICTING"),
            "schema_valid": sum(1 for r in result.relations if r.schema_valid),
            "import_ready": sum(1 for r in result.relations if r.import_ready),
        }

        return result

    def _verify_entity(self, entity: dict) -> VerifiedEntity:
        """验证单个实体"""
        mention = entity.get("mention", "")
        etype = entity.get("type", "")
        raw_attrs = entity.get("attributes", {})
        normalized_id = raw_attrs.get("normalized_id", "")

        ve = VerifiedEntity(
            mention=mention,
            entity_type=etype,
            neo4j_status="NOVEL",
            attributes=raw_attrs,  # ← v2: 保留原始属性
            confidence=getattr(entity, "confidence", 0.7) if hasattr(entity, "confidence") else 0.7,
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
            ve.neo4j_status = "FUZZY_MATCH"
            ve.neo4j_element_id = fuzzy["element_id"]
            ve.neo4j_node_id = fuzzy["node_id"]
            return ve

        return ve

    def _verify_relation(
        self,
        relation: dict,
        verified_entities: dict[str, VerifiedEntity],
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
            quality_flags=[],
        )

        # ── 1. Schema 合规检查 ──
        allowed_pairs = RELATION_SIGNATURES.get(vr.predicate, set())
        pair = (vr.subject_type, vr.object_type)
        vr.schema_valid = pair in allowed_pairs

        if not vr.schema_valid:
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
        if self.kg_memory.is_connected and vr.schema_valid:
            existing = self.kg_memory.check_relation_exists(vr.subject, vr.predicate, vr.object)

            if existing:
                vr.existing_rel_id = existing.get("rel_element_id", "")
                vr.neo4j_status = "KNOWN"
                vr.quality_flags.append("duplicate_evidence")

        # ── 6. 实体链接状态 ──
        subj_key = f"{vr.subject}|{vr.subject_type}"
        obj_key = f"{vr.object}|{vr.object_type}"
        subj_entity = verified_entities.get(subj_key)
        obj_entity = verified_entities.get(obj_key)

        if subj_entity and subj_entity.neo4j_status == "NOVEL":
            vr.quality_flags.append("subject_novel")
        if obj_entity and obj_entity.neo4j_status == "NOVEL":
            vr.quality_flags.append("object_novel")

        # ── 7. Import-ready 判定 ──
        vr.import_ready = (
            vr.schema_valid
            and not vr.negated
            and not vr.uncertain
            and "non_human" not in vr.quality_flags
            and "contradiction" not in vr.quality_flags
            and "invalid_direction" not in vr.quality_flags
        )

        return vr
