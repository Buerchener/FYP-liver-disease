#!/usr/bin/env python3
"""
cognitive_agent/decision_engine.py — Phase 5: 决策执行引擎

根据验证结果，做出6种决策：
- CREATE_ENTITY: 创建新实体节点
- CREATE_RELATION: 创建新关系
- UPDATE_RELATION: 更新已有关系
- MARK_DISPUTED: 标记学术争议
- PROPOSE_HYPOTHESIS: 提出因果推断假设
- DISCARD: 丢弃（质量不合格）
- NO_ACTION: 暂不处理
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional
from cognitive_agent.memory.kg_memory import KGMemory
from cognitive_agent.schema.entity_classes import ENTITY_CREATION_POLICY
from cognitive_agent.tools.ncbi_validator import NCBIGeneValidator
from cognitive_agent.abbreviation_detector import (
    AbbreviationMap, score_entity_name_quality,
)


@dataclass
class Action:
    """单条决策动作"""
    type: str  # CREATE_ENTITY | CREATE_RELATION | UPDATE_RELATION | MARK_DISPUTED | DISCARD | NO_ACTION
    entity: Optional[dict] = None
    relation: Optional[dict] = None
    reason: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "entity_mention": self.entity.get("mention", "") if self.entity else "",
            "relation": f"{self.relation.get('subject','')} -{self.relation.get('predicate','')}-> {self.relation.get('object','')}"
            if self.relation else "",
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class ExecutionLog:
    """Phase 5 执行日志"""
    pmid: str = ""
    actions: list[Action] = field(default_factory=list)
    entities_created: int = 0
    relations_created: int = 0
    relations_updated: int = 0
    disputed: int = 0
    discarded: int = 0

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "total_actions": len(self.actions),
            "entities_created": self.entities_created,
            "relations_created": self.relations_created,
            "relations_updated": self.relations_updated,
            "disputed": self.disputed,
            "discarded": self.discarded,
            "actions": [a.to_dict() for a in self.actions],
        }


class DecisionEngine:
    """Phase 5: 决策执行引擎

    从"被动判断 import/not" 升级为 "主动决策 create/update/dispute/discard"
    """

    def __init__(self, kg_memory: KGMemory, skip_neo4j_write: bool = True):
        self.kg_memory = kg_memory
        self.skip_neo4j_write = skip_neo4j_write
        self._ncbi_validator: Optional[NCBIGeneValidator] = None  # lazy init

    def decide(
        self,
        verified_entities: list["VerifiedEntity"],
        verified_relations: list["VerifiedRelation"],
        pmid: str = "",
        strategy: Optional[dict] = None,
        conflict_resolution: Any = None,
        abbr_map: Optional[AbbreviationMap] = None,
        restrict_entities_to_import_ready_endpoints: bool = False,
    ) -> ExecutionLog:
        """
        基于验证结果做决策。

        决策逻辑:
        1. 实体: NOVEL + 高置信 + 满足创建策略 → CREATE_ENTITY
        2. 关系: NOVEL + schema_valid + import_ready → CREATE_RELATION
        3. 关系: CONTRADICTING + 新证据更可靠 → UPDATE_RELATION
        4. 关系: CONTRADICTING + 双方高置信 → MARK_DISPUTED
        5. 关系: 质量差 → DISCARD
        6. 其他 → NO_ACTION

        新增（v3）:
        - 缩写消歧：通过 AbbreviationMap 将 "T2DM"→"type 2 diabetes mellitus"
        - 方法论噪声过滤：is_methodology_noise() 拒绝实验技术术语
        - 实体名称质量评分：score_entity_name_quality() 惩罚泛化词
        """
        log = ExecutionLog(pmid=pmid)
        strategy = strategy or {}
        resolution_lookup = self._build_resolution_lookup(conflict_resolution)

        # ── v3: 缩写消歧 + 同批去重 ──
        # 将同批内所有实体按 canonical name 分组，每组只保留一个
        canonical_entities = self._deduplicate_entities(verified_entities, abbr_map)
        if restrict_entities_to_import_ready_endpoints:
            eligible_endpoints = {
                (relation.subject, relation.subject_type)
                for relation in verified_relations
                if relation.import_ready
            } | {
                (relation.object, relation.object_type)
                for relation in verified_relations
                if relation.import_ready
            }
            canonical_entities = [
                entity for entity in canonical_entities
                if (entity.mention, entity.entity_type) in eligible_endpoints
            ]

        # ── 实体决策 ──
        for entity in canonical_entities:
            action = self._decide_entity(entity, pmid, strategy=strategy, abbr_map=abbr_map)
            log.actions.append(action)
            if action.type == "CREATE_ENTITY":
                log.entities_created += 1

        # ── 关系决策 ──
        # Keep all verified candidates in the review report, but make at most
        # one write decision for a canonical typed triple.
        decision_relations: list["VerifiedRelation"] = []
        relation_index: dict[tuple[str, str, str, str, str], int] = {}
        for relation in verified_relations:
            key = (
                relation.subject.strip().casefold(), relation.subject_type,
                relation.predicate.strip().upper(),
                relation.object.strip().casefold(), relation.object_type,
            )
            current_index = relation_index.get(key)
            if current_index is None:
                relation_index[key] = len(decision_relations)
                decision_relations.append(relation)
                continue
            current = decision_relations[current_index]
            current_rank = (
                bool(current.import_ready), -int(current.evidence_level),
                len(current.evidence or ""),
            )
            candidate_rank = (
                bool(relation.import_ready), -int(relation.evidence_level),
                len(relation.evidence or ""),
            )
            if candidate_rank > current_rank:
                decision_relations[current_index] = relation

        for relation in decision_relations:
            resolution_item = resolution_lookup.get(self._relation_key(
                relation.subject, relation.predicate, relation.object
            ))
            action = self._decide_relation(
                relation,
                pmid,
                strategy=strategy,
                resolution_item=resolution_item,
                abbr_map=abbr_map,
            )
            log.actions.append(action)
            if action.type == "CREATE_RELATION":
                log.relations_created += 1
            elif action.type == "UPDATE_RELATION":
                log.relations_updated += 1
            elif action.type == "MARK_DISPUTED":
                log.disputed += 1
            elif action.type == "DISCARD":
                log.discarded += 1

        if restrict_entities_to_import_ready_endpoints:
            write_endpoints = {
                (action.relation.get("subject", ""), action.relation.get("subject_type", ""))
                for action in log.actions
                if action.type == "CREATE_RELATION" and action.relation
            } | {
                (action.relation.get("object", ""), action.relation.get("object_type", ""))
                for action in log.actions
                if action.type == "CREATE_RELATION" and action.relation
            }
            for action in log.actions:
                if action.type != "CREATE_ENTITY" or not action.entity:
                    continue
                key = (action.entity.get("mention", ""), action.entity.get("type", ""))
                if key not in write_endpoints:
                    action.type = "NO_ACTION"
                    action.reason += " (no relation selected for write)"
            log.entities_created = sum(
                action.type == "CREATE_ENTITY" for action in log.actions
            )

        return log

    def _deduplicate_entities(
        self,
        entities: list["VerifiedEntity"],
        abbr_map: Optional[AbbreviationMap],
    ) -> list["VerifiedEntity"]:
        """将同批次实体按 canonical name 去重。

        优先保留：长形式 > 缩写，有 normalized_id > 无，先出现 > 后出现。
        """
        # Phase-A canonicalization now runs at the KGVerifier entrance, before
        # linking and relation verification.  Re-merging here would make entity
        # and relation views diverge, especially for curated-but-not-in-text aliases.
        return list(entities)

    def _decide_entity(
        self,
        entity: "VerifiedEntity",
        pmid: str,
        strategy: Optional[dict] = None,
        abbr_map: Optional[AbbreviationMap] = None,
    ) -> Action:
        """实体级别的决策。

        集成 NCBI E-utilities 进行基因外部验证。
        v3: 方法论噪声过滤 + 实体名称质量评分。
        """
        mention = entity.mention

        if entity.attributes.get("collaboration_status") == "manual_review":
            return Action(
                type="NO_ACTION",
                entity={"mention": mention, "type": entity.entity_type},
                reason=(
                    "Second model rejected a Phase-A-retained entity; manual review required: "
                    + str(entity.attributes.get("collaboration_reason", ""))[:300]
                ),
                confidence=entity.confidence,
            )

        if getattr(entity, "filter_status", "retained") != "retained":
            return Action(
                type="DISCARD",
                entity={"mention": mention, "type": entity.entity_type},
                reason=getattr(entity, "filter_reason", "entity quality gate rejected candidate"),
                confidence=0.0,
            )

        # ── v3: 实体名称质量评分 ──
        name_quality, quality_reasons = score_entity_name_quality(mention)
        if name_quality < 0.3:
            return Action(
                type="DISCARD",
                entity={"mention": mention, "type": entity.entity_type},
                reason=f"Low name quality ({name_quality:.2f}): {'; '.join(quality_reasons[:3])}",
                confidence=name_quality,
            )

        if entity.neo4j_status == "AMBIGUOUS":
            return Action(
                type="NO_ACTION",
                entity={"mention": mention, "type": entity.entity_type},
                reason="Ambiguous entity match — manual disambiguation required",
                confidence=entity.confidence,
            )

        # 只有 NOVEL 实体才考虑创建
        if entity.neo4j_status != "NOVEL":
            return Action(type="NO_ACTION", entity={
                "mention": mention, "type": entity.entity_type
            }, reason=f"Already exists in KG ({entity.neo4j_status})")

        # 检查创建策略
        policy = ENTITY_CREATION_POLICY.get(entity.entity_type, {})
        min_conf = self._effective_entity_threshold(entity.entity_type, policy, strategy or {})
        require_external = policy.get("require_external_db", False)

        if entity.confidence + 1e-9 < min_conf:
            return Action(
                type="NO_ACTION",
                entity={"mention": entity.mention, "type": entity.entity_type},
                reason=f"Confidence {entity.confidence:.2f} < {min_conf}",
                confidence=entity.confidence,
            )

        # ── 外部数据库验证 ──
        validated_props = {}
        if require_external:
            if entity.entity_type == "Gene":
                validated_props = self._validate_gene(entity)
            elif entity.entity_type == "Protein":
                # Future: UniProt validation
                validated_props = {}
            elif entity.entity_type == "Metabolite":
                # Future: HMDB validation
                validated_props = {}

        # ── 组装实体属性（传递给 create_entity） ──
        # v2: 保留原始提取属性 + 外部验证属性
        entity_props = {
            "mention": entity.mention,
            "type": entity.entity_type,
            **entity.attributes,      # ← 原始提取属性 (disease_name, gene_symbol, ...)
            **validated_props,        # ← NCBI 等外部验证结果
        }

        if require_external and not validated_props:
            # 需要外部验证但验证失败 → 仍创建但用 PROJECT 前缀，降低置信度
            return Action(
                type="CREATE_ENTITY",
                entity=entity_props,
                reason=f"Novel {entity.entity_type} — external validation failed, "
                       f"creating with PROJECT prefix (conf={entity.confidence:.2f})",
                confidence=min(entity.confidence, 0.5),
            )
        elif require_external:
            return Action(
                type="CREATE_ENTITY",
                entity=entity_props,
                reason=f"Novel {entity.entity_type} — externally validated "
                       f"({validated_props.get('ncbi_gene_id', '')})",
                confidence=entity.confidence,
            )
        else:
            return Action(
                type="CREATE_ENTITY",
                entity=entity_props,
                reason=f"Novel {entity.entity_type} (no external validation required, "
                       f"conf={entity.confidence:.2f})",
                confidence=entity.confidence,
            )

    def _validate_gene(self, entity: "VerifiedEntity") -> dict:
        """使用 NCBI E-utilities 验证基因符号。返回 validated properties 字典。"""
        if self._ncbi_validator is None:
            self._ncbi_validator = NCBIGeneValidator()

        result = self._ncbi_validator.validate(entity.mention)
        if result.found:
            return {
                "ncbi_gene_id": result.ncbi_gene_id,
                "gene_symbol": result.official_symbol,
                "normalized_id": f"NCBIGene:{result.ncbi_gene_id}",
                "official_name": result.official_name,
                "aliases": result.aliases,
            }
        return {}

    def _decide_relation(
        self,
        relation: "VerifiedRelation",
        pmid: str,
        strategy: Optional[dict] = None,
        resolution_item: Any = None,
        abbr_map: Optional[AbbreviationMap] = None,
    ) -> Action:
        """关系级别的决策（v3: 缩写消歧）"""
        strategy = strategy or {}
        # Endpoints were canonicalized before verification.  Do not mutate them
        # after conflict resolution and evidence checks.
        rel_info = {
            "subject": relation.subject,
            "predicate": relation.predicate,
            "object": relation.object,
            "subject_type": relation.subject_type,
            "object_type": relation.object_type,
            "direction": relation.direction,
            "evidence": relation.evidence,
            "existing_rel_id": relation.existing_rel_id,
            "existing_confidence": relation.existing_confidence,
        }

        if relation.neo4j_status == "INVERTED":
            return Action(
                type="NO_ACTION",
                relation=rel_info,
                reason="Inverse relation exists — do not auto-create reverse edge",
                confidence=relation.existing_confidence,
            )

        if "manual_review" in relation.quality_flags:
            return Action(
                type="NO_ACTION",
                relation=rel_info,
                reason="Second-model disagreement; relation preserved for manual review",
                confidence=0.0,
            )

        # 1. 质量过滤
        if not relation.schema_valid:
            return Action(
                type="DISCARD",
                relation=rel_info,
                reason=f"Schema mismatch: ({relation.subject_type})-[:{relation.predicate}]->({relation.object_type})",
                confidence=0.0,
            )

        if relation.negated:
            return Action(
                type="DISCARD",
                relation=rel_info,
                reason="Negated relation",
                confidence=0.0,
            )

        if not relation.import_ready:
            return Action(
                type="DISCARD",
                relation=rel_info,
                reason=(
                    "Deterministic quality gate blocked relation: "
                    + ", ".join(relation.quality_flags[:8])
                ),
                confidence=0.0,
            )

        if "non_human" in relation.quality_flags:
            return Action(
                type="NO_ACTION",
                relation=rel_info,
                reason="Non-human species — flag for review",
                confidence=0.5,
            )

        # 2. 冲突解决表优先驱动最终动作
        resolution_action = self._action_from_resolution(
            relation=relation,
            rel_info=rel_info,
            resolution_item=resolution_item,
            strategy=strategy,
        )
        if resolution_action:
            return resolution_action

        # 3. 旧逻辑 fallback：没有 resolution item 时仍可独立决策
        if relation.neo4j_status == "CONTRADICTING":
            new_conf = 0.8 if relation.uncertain else 0.9  # 从提取质量估计
            old_conf = relation.existing_confidence
            mode = strategy.get("conflict_resolution_mode", "balanced")

            if new_conf > 0.8 and old_conf < 0.6:
                # 新证据压倒旧知识 → 更新
                if mode == "conservative":
                    return Action(
                        type="NO_ACTION",
                        relation=rel_info,
                        reason="Conservative mode: contradiction requires review before update",
                        confidence=new_conf,
                    )
                return Action(
                    type="UPDATE_RELATION",
                    relation=rel_info,
                    reason=f"New evidence (conf={new_conf:.1f}) overrides old (conf={old_conf:.1f})",
                    confidence=new_conf,
                )
            elif new_conf > 0.8 and old_conf > 0.8:
                # 双方都高置信 → 标记争议
                return Action(
                    type="MARK_DISPUTED",
                    relation=rel_info,
                    reason=f"High-confidence contradiction: new={new_conf:.1f} vs old={old_conf:.1f}",
                    confidence=0.5,
                )
            else:
                return Action(
                    type="NO_ACTION",
                    relation=rel_info,
                    reason=f"New evidence (conf={new_conf:.1f}) insufficient to override old (conf={old_conf:.1f})",
                    confidence=new_conf,
                )

        # 3. 已知关系 → 补充证据
        if relation.neo4j_status == "KNOWN":
            return Action(
                type="NO_ACTION",
                relation=rel_info,
                reason="Relation already exists — evidence can be appended",
                confidence=relation.existing_confidence,
            )

        # 4. 新关系 → 创建
        if relation.neo4j_status == "NOVEL" and relation.import_ready:
            confidence = 0.7 if relation.uncertain else 0.85
            min_conf = self._effective_relation_threshold(strategy)
            if confidence + 1e-9 < min_conf:
                return Action(
                    type="NO_ACTION",
                    relation=rel_info,
                    reason=f"Relation confidence {confidence:.2f} < active threshold {min_conf:.2f}",
                    confidence=confidence,
                )
            return Action(
                type="CREATE_RELATION",
                relation=rel_info,
                reason="Novel relation with high-quality evidence",
                confidence=confidence,
            )

        # 5. 默认 → 不操作
        return Action(
            type="NO_ACTION",
            relation=rel_info,
            reason=f"Does not meet criteria (status={relation.neo4j_status}, import_ready={relation.import_ready})",
            confidence=0.0,
        )

    def _effective_entity_threshold(
        self,
        entity_type: str,
        policy: dict,
        strategy: dict,
    ) -> float:
        """Apply the run-level strategy threshold while preserving type policy shape."""
        policy_min = policy.get("min_confidence", 0.7)
        strategy_min = strategy.get("entity_confidence_threshold", 0.7)
        mode = strategy.get("extraction_mode", "balanced")

        # Treat 0.7 as the neutral baseline and apply strategy deltas per type.
        threshold = policy_min + (strategy_min - 0.7)
        if mode == "exploratory":
            threshold = min(threshold, policy_min - 0.1)
        elif mode == "focused":
            threshold = max(threshold, policy_min)

        return round(max(0.3, min(0.95, threshold)), 6)

    def _effective_relation_threshold(self, strategy: dict) -> float:
        """Return active relation creation threshold."""
        threshold = strategy.get("relation_confidence_threshold", 0.7)
        if strategy.get("extraction_mode") == "focused":
            threshold = max(threshold, 0.75)
        return max(0.3, min(0.95, threshold))

    def _action_from_resolution(
        self,
        relation: "VerifiedRelation",
        rel_info: dict,
        resolution_item: Any,
        strategy: dict,
    ) -> Optional[Action]:
        """Translate ConflictResolver output into the final decision action."""
        if not resolution_item:
            return None

        decision = self._item_value(resolution_item, "decision", "NO_ACTION")
        reasoning = self._item_value(resolution_item, "reasoning_trace", "")
        adjusted_conf = float(self._item_value(
            resolution_item, "adjusted_confidence", 0.0
        ) or 0.0)
        confidence = adjusted_conf or (0.7 if relation.uncertain else 0.85)
        mode = strategy.get("conflict_resolution_mode", "balanced")
        min_rel_conf = self._effective_relation_threshold(strategy)

        if decision == "DISCARD":
            return Action(
                type="DISCARD",
                relation=rel_info,
                reason=reasoning or "Conflict resolver discarded relation",
                confidence=confidence,
            )

        if decision == "KEEP_OLD":
            return Action(
                type="NO_ACTION",
                relation=rel_info,
                reason=reasoning or "Conflict resolver kept existing KG relation",
                confidence=confidence,
            )

        if decision == "DISPUTE":
            return Action(
                type="MARK_DISPUTED",
                relation=rel_info,
                reason=reasoning or "Conflict resolver marked relation as disputed",
                confidence=confidence,
            )

        if decision == "UPDATE":
            if mode == "conservative":
                return Action(
                    type="NO_ACTION",
                    relation=rel_info,
                    reason=(reasoning or "Conflict resolver proposed update")
                    + " (conservative mode: review required)",
                    confidence=confidence,
                )
            return Action(
                type="UPDATE_RELATION",
                relation=rel_info,
                reason=reasoning or "Conflict resolver proposed relation update",
                confidence=confidence,
            )

        if decision == "CREATE":
            if not relation.import_ready:
                return Action(
                    type="NO_ACTION",
                    relation=rel_info,
                    reason=(reasoning or "Conflict resolver proposed create")
                    + " (blocked by import_ready=false)",
                    confidence=confidence,
                )
            if confidence < min_rel_conf - 1e-9:
                return Action(
                    type="NO_ACTION",
                    relation=rel_info,
                    reason=f"Relation confidence {confidence:.2f} < active threshold {min_rel_conf:.2f}",
                    confidence=confidence,
                )
            return Action(
                type="CREATE_RELATION",
                relation=rel_info,
                reason=reasoning or "Conflict resolver approved novel relation",
                confidence=confidence,
            )

        if decision == "CREATE_WITH_FLAG":
            if not relation.import_ready:
                return Action(
                    type="NO_ACTION",
                    relation=rel_info,
                    reason=(reasoning or "Moderate-confidence relation")
                    + " (blocked by deterministic import_ready gate)",
                    confidence=confidence,
                )
            if mode != "aggressive":
                return Action(
                    type="NO_ACTION",
                    relation=rel_info,
                    reason=(reasoning or "Moderate-confidence relation")
                    + " (flagged for review; aggressive mode required to create)",
                    confidence=confidence,
                )
            if not relation.schema_valid or relation.negated or "non_human" in relation.quality_flags:
                return Action(
                    type="NO_ACTION",
                    relation=rel_info,
                    reason=(reasoning or "Moderate-confidence relation")
                    + " (blocked by safety flags)",
                    confidence=confidence,
                )
            return Action(
                type="CREATE_RELATION",
                relation=rel_info,
                reason=(reasoning or "Moderate-confidence relation")
                + " (created in aggressive mode)",
                confidence=confidence,
            )

        return None

    def _build_resolution_lookup(self, conflict_resolution: Any) -> dict[tuple[str, str, str], Any]:
        """Build a relation-triple lookup from a ResolutionResult or dict."""
        if not conflict_resolution:
            return {}
        items = (
            conflict_resolution.get("items", [])
            if isinstance(conflict_resolution, dict)
            else getattr(conflict_resolution, "items", [])
        )
        lookup = {}
        for item in items:
            key = self._relation_key(
                self._item_value(item, "subject", ""),
                self._item_value(item, "predicate", ""),
                self._item_value(item, "object", ""),
            )
            lookup[key] = item
        return lookup

    @staticmethod
    def _relation_key(subject: str, predicate: str, obj: str) -> tuple[str, str, str]:
        return (
            (subject or "").strip().lower(),
            (predicate or "").strip().upper(),
            (obj or "").strip().lower(),
        )

    @staticmethod
    def _item_value(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _execute_batch(self, log: ExecutionLog) -> ExecutionLog:
        """批量执行 CREATE actions，减少 Neo4j 往返；更新/争议沿用现有 API。"""
        entity_actions = [a for a in log.actions if a.type == "CREATE_ENTITY" and a.entity]
        entity_rows = []
        for action in entity_actions:
            row = dict(action.entity)
            row["confidence"] = action.confidence
            entity_rows.append(row)
        cache = self.kg_memory.create_entities_batch(entity_rows, pmid=log.pmid) if entity_rows else {}
        for action in entity_actions:
            key = (action.entity.get("type", ""), action.entity.get("mention", "").casefold())
            element_id = cache.get(key)
            if element_id:
                action.entity["created_element_id"] = element_id
            else:
                action.type = "NO_ACTION"
                action.reason += " (batch entity write failed)"

        relation_rows = []
        relation_actions = [a for a in log.actions if a.type == "CREATE_RELATION" and a.relation]
        for action in relation_actions:
            rel = action.relation
            subj_key = (rel.get("subject_type", ""), rel.get("subject", "").casefold())
            obj_key = (rel.get("object_type", ""), rel.get("object", "").casefold())
            subj_id = cache.get(subj_key)
            obj_id = cache.get(obj_key)
            if not subj_id:
                match = self.kg_memory.find_entity(rel.get("subject", ""), rel.get("subject_type", ""))
                subj_id = match.get("element_id") if match else None
            if not obj_id:
                match = self.kg_memory.find_entity(rel.get("object", ""), rel.get("object_type", ""))
                obj_id = match.get("element_id") if match else None
            if not subj_id or not obj_id:
                action.type = "NO_ACTION"
                action.reason += " (batch relation endpoint not found)"
                continue
            relation_rows.append({
                **{k: v for k, v in rel.items() if not isinstance(v, (list, dict))},
                "subject_element_id": subj_id,
                "object_element_id": obj_id,
                "confidence": action.confidence,
            })
        written = self.kg_memory.create_relations_batch(relation_rows, pmid=log.pmid) if relation_rows else []
        written_set = set(written)
        relation_actions_for_rows = [
            a for a in relation_actions if a.type == "CREATE_RELATION"
        ]
        for action, row in zip(relation_actions_for_rows, relation_rows):
            relation_identity = (
                f"{row['predicate']}|{row['subject_element_id']}|"
                f"{row['object_element_id']}"
            )
            digest = hashlib.sha256(relation_identity.encode()).hexdigest()[:20]
            relation_id = f"LLM_{row['predicate']}:{digest}"
            if relation_id in written_set:
                action.relation["created_rel_id"] = relation_id
            else:
                action.type = "NO_ACTION"
                action.reason += " (batch relation write failed)"

        log.entities_created = sum(a.type == "CREATE_ENTITY" for a in log.actions)
        log.relations_created = sum(a.type == "CREATE_RELATION" for a in log.actions)
        log.relations_updated = sum(a.type == "UPDATE_RELATION" for a in log.actions)
        log.disputed = sum(a.type == "MARK_DISPUTED" for a in log.actions)
        log.discarded = sum(a.type == "DISCARD" for a in log.actions)
        return log

    def execute(self, log: ExecutionLog) -> ExecutionLog:
        """执行决策（写入 Neo4j），标记是否实际写入了。

        Phase 1: 创建实体；Phase 2: 更新/争议；Phase 3: 创建关系。
        """
        if self.skip_neo4j_write:
            return log

        if hasattr(self.kg_memory, "create_entities_batch") and any(
            action.type in ("CREATE_ENTITY", "CREATE_RELATION") for action in log.actions
        ) and not any(action.type in ("UPDATE_RELATION", "MARK_DISPUTED") for action in log.actions):
            # Batch CREATE actions; UPDATE/DISPUTE continue through the compatible path below.
            return self._execute_batch(log)

        # ── Phase 1: 实体写入 ──
        for action in log.actions:
            if action.type == "CREATE_ENTITY" and action.entity:
                entity_props = {
                    k: v for k, v in action.entity.items()
                    if k not in ("mention", "type")
                }
                element_id = self.kg_memory.create_entity(
                    entity_type=action.entity["type"],
                    name=action.entity["mention"],
                    properties=entity_props,
                    evidence="",
                    pmid=log.pmid,
                    confidence=action.confidence,
                )
                if element_id:
                    action.entity["created_element_id"] = element_id
                else:
                    action.type = "NO_ACTION"
                    action.reason += " (write failed)"

        # ── Phase 2: 已有关系更新/争议标记 ──
        for action in log.actions:
            if action.type not in ("UPDATE_RELATION", "MARK_DISPUTED") or not action.relation:
                continue

            rel = action.relation
            rel_element_id = self._resolve_existing_relation_id(rel)
            if not rel_element_id:
                action.type = "NO_ACTION"
                action.reason += " (existing relation not found for update/dispute)"
                continue

            if action.type == "UPDATE_RELATION":
                ok = self.kg_memory.update_relation(
                    rel_element_id=rel_element_id,
                    new_evidence=rel.get("evidence", ""),
                    new_confidence=action.confidence,
                    pmid=log.pmid,
                )
                if not ok:
                    action.type = "NO_ACTION"
                    action.reason += " (relation update failed)"

            elif action.type == "MARK_DISPUTED":
                ok = self.kg_memory.mark_disputed(
                    rel_element_id=rel_element_id,
                    dispute_reason=action.reason,
                    conflicting_evidence=rel.get("evidence", ""),
                    pmid=log.pmid,
                )
                if not ok:
                    action.type = "NO_ACTION"
                    action.reason += " (mark disputed failed)"

        # ── Phase 3: 新关系写入 ──
        for action in log.actions:
            if action.type != "CREATE_RELATION" or not action.relation:
                continue

            rel = action.relation
            subj_name = rel.get("subject", "")
            obj_name = rel.get("object", "")
            subj_type = rel.get("subject_type", "")
            obj_type = rel.get("object_type", "")
            predicate = rel.get("predicate", "")

            # 查找两端实体
            subj_match = self.kg_memory.find_entity(subj_name, subj_type) if subj_name else None
            obj_match = self.kg_memory.find_entity(obj_name, obj_type) if obj_name else None

            if not subj_match:
                action.type = "NO_ACTION"
                action.reason += f" (subject '{subj_name}' not found in KG)"
                continue
            if not obj_match:
                action.type = "NO_ACTION"
                action.reason += f" (object '{obj_name}' not found in KG)"
                continue

            # 提取关系属性（仅保留标量值）
            rel_props = {}
            for k, v in rel.items():
                if k in (
                    "subject", "object", "subject_type", "object_type", "predicate",
                    "existing_rel_id", "existing_confidence", "created_rel_id",
                ):
                    continue
                if isinstance(v, (list, dict)):
                    continue
                rel_props[k] = v

            evidence = rel.get("evidence", "")

            rel_element_id = self.kg_memory.create_relation(
                subject_element_id=subj_match["element_id"],
                predicate=predicate,
                object_element_id=obj_match["element_id"],
                properties=rel_props,
                evidence=evidence,
                pmid=log.pmid,
                confidence=action.confidence,
            )
            if rel_element_id:
                rel["created_rel_id"] = rel_element_id
            else:
                action.type = "NO_ACTION"
                action.reason += " (relation write failed)"

        # 重新统计实际写入数量（执行过程中可能 fail → NO_ACTION）
        log.entities_created = sum(1 for a in log.actions if a.type == "CREATE_ENTITY")
        log.relations_created = sum(1 for a in log.actions if a.type == "CREATE_RELATION")
        log.relations_updated = sum(1 for a in log.actions if a.type == "UPDATE_RELATION")
        log.disputed = sum(1 for a in log.actions if a.type == "MARK_DISPUTED")
        log.discarded = sum(1 for a in log.actions if a.type == "DISCARD")

        return log

    def _resolve_existing_relation_id(self, rel: dict) -> str:
        """Find an existing relation id from action metadata or KG lookup."""
        rel_element_id = rel.get("existing_rel_id", "")
        if rel_element_id:
            return rel_element_id

        existing = self.kg_memory.check_relation_exists(
            subject_name=rel.get("subject", ""),
            predicate=rel.get("predicate", ""),
            object_name=rel.get("object", ""),
        )
        return existing.get("rel_element_id", "") if existing else ""
