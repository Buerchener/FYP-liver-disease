#!/usr/bin/env python3
"""
cognitive_agent/context_activator.py — Phase 1: 先验知识激活

提取前主动查询 Neo4j，激活相关背景知识：
- 识别摘要中的核心实体提及
- 查询 Neo4j 获取已知关系
- 识别知识缺口
- 生成提取目标
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from cognitive_agent.memory.kg_memory import KGMemory


# 常见基因符号模式 (大写字母+数字组合)
GENE_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]{1,8})\b')
# 肝病相关疾病名称（小写匹配）
LIVER_DISEASE_KEYWORDS = [
    "hepatocellular carcinoma", "hcc", "liver cancer",
    "nafld", "nash", "masld", "mash",
    "liver fibrosis", "hepatic fibrosis", "cirrhosis",
    "alcoholic liver disease", "ald",
    "non-alcoholic fatty liver disease",
    "metabolic dysfunction-associated steatotic liver disease",
    "liver injury", "hepatic injury",
    "cholangiocarcinoma", "liver cirrhosis",
    "hepatitis b", "hepatitis c", "hbv", "hcv",
    "primary biliary cholangitis", "primary sclerosing cholangitis",
    "autoimmune hepatitis", "wilson disease", "hemochromatosis",
]


@dataclass
class ContextCard:
    """Phase 1 输出的上下文卡片"""
    pmid: str = ""
    known_entities: dict[str, dict] = field(default_factory=dict)
    existing_relations: list[dict] = field(default_factory=list)
    knowledge_gaps: list[dict] = field(default_factory=list)
    extraction_goals: list[str] = field(default_factory=list)
    has_conflicts: bool = False
    coverage_score: float = 0.0  # KG 覆盖度 (0-1)

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "known_entity_count": len(self.known_entities),
            "existing_relation_count": len(self.existing_relations),
            "knowledge_gap_count": len(self.knowledge_gaps),
            "extraction_goals": self.extraction_goals,
            "has_conflicts": self.has_conflicts,
            "coverage_score": self.coverage_score,
            "known_entities": {
                k: {"name": v.get("name", ""), "labels": v.get("labels", [])}
                for k, v in self.known_entities.items()
            },
        }


class ContextActivator:
    """Phase 1: 从 Neo4j 激活先验知识"""

    def __init__(self, kg_memory: KGMemory):
        self.kg_memory = kg_memory

    def activate(self, text: str, pmid: str = "") -> ContextCard:
        """
        扫描文本，查询 Neo4j，构建上下文卡片。

        Args:
            text: PubMed 摘要全文 (TITLE + ABSTRACT)
            pmid: PubMed ID

        Returns:
            ContextCard: 包含已知实体、已有关系、知识缺口的上下文
        """
        card = ContextCard(pmid=pmid)

        if not self.kg_memory.is_connected:
            card.extraction_goals = ["full_extraction"]
            return card

        # 1. 快速扫描实体提及
        surface_mentions = self._scan_mentions(text)
        if not surface_mentions:
            card.extraction_goals = ["full_extraction"]
            return card

        # 2. 查询 Neo4j 获取已知实体
        known = {}
        for mention, etype in surface_mentions.items():
            entity = self.kg_memory.find_entity(mention, entity_type=etype)
            if entity:
                known[mention] = entity

                # 获取已有关系
                relations = self.kg_memory.get_relations(entity["element_id"])
                for rel in relations:
                    card.existing_relations.append({
                        "subject": mention,
                        **rel,
                    })

        card.known_entities = known

        # 3. 识别知识缺口
        novel_mentions = set(surface_mentions.keys()) - set(known.keys())
        for mention in novel_mentions:
            etype = surface_mentions.get(mention, "")
            # 尝试模糊匹配
            fuzzy = self.kg_memory.find_entity_fuzzy(mention, entity_type=etype)
            if fuzzy:
                card.knowledge_gaps.append({
                    "mention": mention,
                    "type": "fuzzy_match",
                    "candidates": fuzzy.get("fuzzy_matches", []),
                })
            else:
                card.knowledge_gaps.append({
                    "mention": mention,
                    "type": "novel_entity",
                })

        # 4. 计算覆盖度
        total = len(surface_mentions)
        exact_matches = len(known)
        fuzzy_matches = sum(1 for g in card.knowledge_gaps if g["type"] == "fuzzy_match")
        card.coverage_score = (exact_matches + 0.5 * fuzzy_matches) / max(total, 1)

        # 5. 生成提取目标
        card.extraction_goals = self._generate_goals(card)

        return card

    def _scan_mentions(self, text: str) -> dict[str, str]:
        """快速扫描文本中的实体提及 (规则 + 关键词，不调 LLM)"""
        mentions: dict[str, str] = {}

        # 基因符号
        for match in GENE_PATTERN.finditer(text):
            symbol = match.group(1)
            # 过滤常见英文单词和缩写噪声
            if symbol in {"THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT",
                          "WAS", "ARE", "HAS", "HAD", "NOT", "BUT", "ALL",
                          "ITS", "CAN", "MAY", "NEW", "ONE", "TWO", "VIA",
                          "HR", "CI", "OR", "SD", "SE", "CT", "MRI", "DNA",
                          "RNA", "PCR", "ELISA", "WB", "ALT", "AST", "BMI",
                          "GGT", "ALP", "AFP", "HBV", "HCV", "HCC", "NAFLD",
                          "NASH", "MASLD", "MASH", "ALD", "HSC", "HPC",
                          "IL", "TNF", "TGF", "EGF", "FGF", "VEGF", "PDGF",
                          "MAPK", "JNK", "ERK", "JAK", "STAT", "NFKB",
                          "ABSTRACT", "METHODS", "RESULTS", "CONCLUSION",
                          "BACKGROUND", "OBJECTIVE", "AIMS"}:
                continue
            # 检查是否存在于文本中（排除部分边界情况）
            if len(symbol) >= 2:
                mentions[symbol] = "Gene"

        # 肝病名称
        text_lower = text.lower()
        for disease_kw in LIVER_DISEASE_KEYWORDS:
            if disease_kw in text_lower:
                # 找到原始提及形式
                idx = text_lower.find(disease_kw)
                original = text[idx:idx + len(disease_kw)]
                mentions[original] = "Disease"

        # 常见通路关键词
        pathway_keywords = [
            "ferroptosis", "apoptosis", "autophagy", "pyroptosis",
            "oxidative stress", "inflammation", "nf-kb", "tnf signaling",
            "wnt signaling", "hedgehog", "notch", "tgf-beta",
            "mapk signaling", "pi3k-akt", "mtor", "ampk",
        ]
        for pw in pathway_keywords:
            if pw in text_lower:
                idx = text_lower.find(pw)
                original = text[idx:idx + len(pw)]
                mentions[original] = "Pathway"

        # 常见细胞类型
        cell_keywords = [
            "hepatocyte", "hepatocytes", "kupffer cell", "kupffer cells",
            "stellate cell", "stellate cells", "hsc", "t cell", "t cells",
            "macrophage", "macrophages", "neutrophil", "neutrophils",
        ]
        for ct in cell_keywords:
            if ct in text_lower:
                idx = text_lower.find(ct)
                original = text[idx:idx + len(ct)]
                mentions[original] = "CellType"

        return mentions

    def _generate_goals(self, card: ContextCard) -> list[str]:
        """根据知识缺口生成提取目标"""
        goals = []

        novel_count = sum(1 for g in card.knowledge_gaps if g["type"] == "novel_entity")
        fuzzy_count = sum(1 for g in card.knowledge_gaps if g["type"] == "fuzzy_match")

        if card.coverage_score < 0.2:
            goals.append("exploratory_extraction")  # 大量新实体，全面提取
        elif card.coverage_score < 0.5:
            goals.append("focused_extraction")      # 部分已知，聚焦新实体

        if novel_count > 3:
            goals.append("novel_entity_discovery")
        if fuzzy_count > 0:
            goals.append("entity_disambiguation")
        if len(card.existing_relations) > 5:
            goals.append("relation_completion")     # 已知实体多，补全关系
        if not card.existing_relations:
            goals.append("baseline_characterization")  # 无已知关系，建立基线

        if not goals:
            goals.append("full_extraction")

        return goals
