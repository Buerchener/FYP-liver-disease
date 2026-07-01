#!/usr/bin/env python3
"""
multi_stage_extraction_pipeline.py
===================================
Multi-Stage LangExtract Agent — 从 PubMed 文献摘要中提取结构化知识。

架构（5 阶段）:
  Stage 0 — 文本分类: 临床记录 / 影像报告 / 文献摘要
  Stage 1 — 分块初提取: 实体识别 + 关系候选（Few-shot 提示）
  Stage 2 — 校验补全: 证据句自检、否定/不确定检测、物种归一化
  Stage 3 — 精提取与标准化: 实体→UMLS/HGNC 映射，关系归一化
  Stage 4 — 融合与冲突检测: 对比已有图谱，标记冲突/重复
  Stage 5 — Neo4j 写入: 仅按目标 KG schema 写入已有节点之间的候选关系 + 溯源报告

用法:
    export DEEPSEEK_API_KEY="sk-xxx"
    export NEO4J_PASSWORD="xxx"
    python multi_stage_extraction_pipeline.py \
        --input workstreams/literature_hmdb_kegg/data/staging/literature/pubmed_demo_2026-06-14/literature_records.jsonl \
        --limit 10 \
        --run-id demo_001
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
import re
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 配置 —— 全部从环境变量读取
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

NEO4J_URL = os.environ.get("NEO4J_URL", "bolt://100.104.181.96:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "liver-kg-core-v02")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "extraction_output"

ALLOWED_ENTITY_TYPES = {
    "Gene",
    "Disease",
    "Protein",
    "Pathway",
    "Metabolite",
    "Tissue",
    "CellType",
}

RELATION_SIGNATURES = {
    "ASSOCIATED_WITH": {("Gene", "Disease"), ("Metabolite", "Disease")},
    "PROGNOSTIC_IN": {("Gene", "Disease")},
    "PROGRESSES_TO": {("Disease", "Disease")},
    "ENCODES": {("Gene", "Protein")},
    "INTERACTS_WITH": {("Protein", "Protein")},
    "PARTICIPATES_IN": {("Gene", "Pathway")},
    "EXPRESSED_IN": {("Gene", "Tissue"), ("Gene", "CellType")},
    "ASSOCIATED_WITH_METABOLITE": {("Gene", "Metabolite")},
}

NEO4J_IMPORTABLE_PREDICATES = {
    "ASSOCIATED_WITH",
    "PROGNOSTIC_IN",
    "INTERACTS_WITH",
    "PARTICIPATES_IN",
    "EXPRESSED_IN",
    "ASSOCIATED_WITH_METABOLITE",
}

NEO4J_RELATION_ID_PROPERTY = {
    "ASSOCIATED_WITH": "relation_id",
    "PROGNOSTIC_IN": "relationship_id",
    "INTERACTS_WITH": "interaction_id",
    "PARTICIPATES_IN": "relationship_id",
    "EXPRESSED_IN": "relationship_id",
    "ASSOCIATED_WITH_METABOLITE": "relationship_id",
}

ALLOWED_DIRECTIONS = {"positive", "negative", "increase", "decrease", "none", "unknown"}
NON_HUMAN_SPECIES = {"mus musculus", "mouse", "mice", "rat", "rattus norvegicus"}
GENERIC_PROGRESSION_CUES = (
    "potential to progress",
    "can progress",
    "may progress",
    "progressive liver disease",
    "towards cirrhosis",
    "toward cirrhosis",
    "and even hepatocellular carcinoma",
)
SCHEMA_FORCED_ENTITY_CUES = {
    "Metabolite": (
        "adjuvant",
        "therapy",
        "therapies",
        "treatment",
        "drug",
        "inhibitor",
        "extract",
        "formula",
        "decoction",
    ),
    "Protein": ("inhibitor", "therapy", "treatment", "drug"),
    "Pathway": ("therapy", "treatment", "drug", "extract", "formula", "decoction"),
    "CellType": ("therapy", "treatment"),
}


# ============================================================
# Stage 0: 文本分类器
# ============================================================
TEXT_TYPE_RULES = {
    "clinical_note": {
        "keywords": [
            "patient", "diagnosis", "admission", "discharge",
            "prescribed", "vital signs", "history of present illness",
            "入院", "出院", "诊断", "患者", "主诉", "查体",
        ],
        "weight": 1.0,
    },
    "imaging_report": {
        "keywords": [
            "MRI", "CT scan", "ultrasound", "biopsy", "histology",
            "lesion", "nodule", "enhancement", "contrast",
            "影像", "超声", "穿刺", "病理", "结节", "增强",
        ],
        "weight": 1.0,
    },
    "literature_abstract": {
        "keywords": [
            "abstract", "methods", "results", "conclusion",
            "objective", "background", "p <", "cohort",
            "背景", "方法", "结果", "结论", "目的",
        ],
        "weight": 1.0,
    },
}


def classify_text(text: str) -> dict[str, float]:
    """基于关键词规则 + 启发式的文本分类器。返回各类置信度。"""
    text_lower = text.lower()
    scores: dict[str, float] = {}
    for category, rule in TEXT_TYPE_RULES.items():
        score = 0.0
        for kw in rule["keywords"]:
            if kw.lower() in text_lower:
                score += rule["weight"]
        # 归一化到关键词命中比例
        max_hits = len(rule["keywords"])
        scores[category] = min(score / max(max_hits * 0.5, 1), 1.0)
    # 如果是 PubMed 文献记录（有 pmid 字段），优先文献
    return scores


def classify_record(record: dict[str, Any]) -> str:
    """对单条记录分类，返回最佳类别。"""
    # 如果已有 source 字段指向 PubMed，直接分类
    source = record.get("source", "")
    if source == "PubMed":
        return "literature_abstract"

    # 合并标题和摘要进行判断
    text = f"{record.get('title', '')} {record.get('abstract', '')} {record.get('raw_subject', '')}"
    scores = classify_text(text)
    if not scores:
        return "literature_abstract"  # 默认归为文献
    return max(scores, key=scores.get)


# ============================================================
# Stage 1: 分块初提取 —— Few-shot 提示词
# ============================================================
FEWSHOT_EXAMPLES_LITERATURE = [
    {
        "abstract_snippet": "TP53 mutations are strongly associated with hepatocellular carcinoma progression.",
        "extraction": {
            "entities": [
                {"mention": "TP53", "type": "Gene", "normalized_id": "HGNC:11998"},
                {"mention": "hepatocellular carcinoma", "type": "Disease", "normalized_id": "UMLS:C2239176"},
            ],
            "relations": [
                {
                    "subject": "TP53",
                    "predicate": "ASSOCIATED_WITH",
                    "object": "hepatocellular carcinoma",
                    "evidence": "TP53 mutations are strongly associated with hepatocellular carcinoma progression.",
                    "direction": "positive",
                    "negated": False,
                    "uncertain": False,
                    "species": "Homo sapiens",
                    "disease_stage": "HCC",
                }
            ],
        },
    },
    {
        "abstract_snippet": "SLC7A11 deficiency accelerated MASLD progression via ferroptosis in mice.",
        "extraction": {
            "entities": [
                {"mention": "SLC7A11", "type": "Gene", "normalized_id": "HGNC:10916"},
                {"mention": "MASLD", "type": "Disease", "normalized_id": "UMLS:C0400966"},
                {"mention": "ferroptosis", "type": "Pathway", "normalized_id": "WP:WP4313"},
            ],
            "relations": [
                {
                    "subject": "SLC7A11",
                    "predicate": "ASSOCIATED_WITH",
                    "object": "MASLD",
                    "evidence": "SLC7A11 deficiency accelerated MASLD progression via ferroptosis.",
                    "direction": "increase",
                    "negated": False,
                    "uncertain": False,
                    "species": "Mus musculus",
                    "disease_stage": "progression",
                }
            ],
        },
    },
]

# 临床记录 Few-shot 示例
FEWSHOT_EXAMPLES_CLINICAL = [
    {
        "abstract_snippet": "患者男，55岁，因腹胀入院。CT示肝右叶占位，AFP>400ng/mL。诊断：原发性肝癌。",
        "extraction": {
            "entities": [
                {"mention": "原发性肝癌", "type": "Disease", "normalized_id": "UMLS:C2239176"},
                {"mention": "AFP", "type": "Protein", "normalized_id": "HGNC:317"},
            ],
            "relations": [
                {
                    "subject": "AFP",
                    "predicate": "BIOMARKER_OF",
                    "object": "原发性肝癌",
                    "evidence": "AFP>400ng/mL。诊断：原发性肝癌。",
                    "direction": "increase",
                    "negated": False,
                    "uncertain": False,
                    "species": "Homo sapiens",
                    "disease_stage": "HCC",
                }
            ],
        },
    },
]

# 影像报告 Few-shot 示例
FEWSHOT_EXAMPLES_IMAGING = [
    {
        "abstract_snippet": "腹部超声：肝脏回声增强，符合脂肪肝表现。肝右叶见一低回声结节，大小约2.3cm×1.8cm。",
        "extraction": {
            "entities": [
                {"mention": "脂肪肝", "type": "Disease", "normalized_id": "UMLS:C0400966"},
                {"mention": "低回声结节", "type": "Finding", "normalized_id": ""},
            ],
            "relations": [
                {
                    "subject": "脂肪肝",
                    "predicate": "HAS_FINDING",
                    "object": "低回声结节",
                    "evidence": "肝右叶见一低回声结节，大小约2.3cm×1.8cm。",
                    "direction": "none",
                    "negated": False,
                    "uncertain": False,
                    "species": "Homo sapiens",
                    "disease_stage": "NAFLD",
                }
            ],
        },
    },
]


def build_stage1_prompt(record: dict[str, Any], text_type: str) -> str:
    """构建第一阶段的 Few-shot 提取提示词。"""

    # 选择对应的 Few-shot 示例
    if text_type == "literature_abstract":
        examples = FEWSHOT_EXAMPLES_LITERATURE
    elif text_type == "clinical_note":
        examples = FEWSHOT_EXAMPLES_CLINICAL
    else:
        examples = FEWSHOT_EXAMPLES_IMAGING

    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"""
Example {i}:
Input: {ex['abstract_snippet']}
Output: {json.dumps(ex['extraction'], indent=2, ensure_ascii=False)}
"""

    # 定义允许的关系签名（与目标 Neo4j KG Schema 对齐）
    allowed_signatures = """
Allowed entity types: Gene, Disease, Protein, Pathway, Metabolite, Tissue, CellType
Allowed relation types:
  - Gene -> Disease : ASSOCIATED_WITH, PROGNOSTIC_IN
  - Disease -> Disease : PROGRESSES_TO
  - Gene -> Protein : ENCODES
  - Protein -> Protein : INTERACTS_WITH
  - Gene -> Pathway : PARTICIPATES_IN
  - Gene -> Tissue/CellType : EXPRESSED_IN
  - Gene -> Metabolite : ASSOCIATED_WITH_METABOLITE
  - Metabolite -> Disease : ASSOCIATED_WITH
"""

    prompt = f"""You are a biomedical knowledge extraction expert. Extract entities and relations from the text below.

{allowed_signatures}

Few-shot examples:
{examples_text}

Now extract from:
Title: {record.get('title', '')}
Abstract: {record.get('abstract', '')}
PMID: {record.get('pmid', '')}

Instructions:
1. Extract entities (Gene, Disease, Protein, Pathway, Metabolite, Tissue, CellType).
2. For each entity, provide the exact mention text from the source.
3. Extract at most 8 high-confidence relations that match EXACTLY one allowed signature.
4. For every relation, include subject_type and object_type, and make subject/object exactly match entity mention text.
5. Provide the evidence EXACTLY as it appears in the source text.
6. Do not extract a relation if the evidence only says the relation is unclear, speculative, or a generic background statement.
7. Do not use PROGRESSES_TO for generic background disease-stage chains unless this article directly studies or supports that progression claim.
8. Do not use ENCODES unless the text explicitly states a gene encodes a protein.
9. Do not force unsupported drug, biomarker, protein-disease, pathway-disease, or treatment claims into Gene->Disease ASSOCIATED_WITH.
10. Do not relabel drugs, therapies, extracts, adjuvants, immune checkpoint inhibitors, CAR-T, herbal formulas, or broad chemical classes as Metabolite/Protein/Pathway just to fit the schema; omit unsupported claims.
11. Mark negated=true only for real negation of the relation; "not only ... but also" is not negation.
12. Mark uncertain=true if the text uses hedged language (may, might, suggests, unclear, potential).
13. Set species based on context (Homo sapiens, Mus musculus, etc.).
14. Set disease_stage if mentioned (NAFLD, NASH, MASH, Fibrosis, Cirrhosis, HCC).
15. Do NOT invent identifiers — leave normalized_id empty if no normalized ID is provided by the text.
16. Output ONLY a single JSON object, not Markdown.

Return format:
{{"entities": [{{"mention": "...", "type": "...", "normalized_id": ""}}],
  "relations": [{{"subject": "...", "subject_type": "...", "predicate": "...",
                  "object": "...", "object_type": "...", "evidence": "...",
                  "direction": "positive|negative|increase|decrease|none|unknown",
                  "negated": false, "uncertain": false,
                  "species": "...", "disease_stage": "..."}}]}}
"""
    return prompt


def call_deepseek_api(prompt: str, system_prompt: str = "") -> dict[str, Any]:
    """调用 DeepSeek API 进行提取。"""
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

    import urllib.request
    import urllib.error

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt or "You are a biomedical knowledge extraction expert. Output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API HTTP error {e.code}: {e.read().decode()}")

    content = data["choices"][0]["message"]["content"]
    # Strip code fences
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:]) if not lines[-1].startswith("```") else "\n".join(lines[1:-1])
    parsed = json.loads(content)
    return coerce_extraction(parsed)


def coerce_extraction(value: Any) -> dict[str, Any]:
    """Normalize LLM output into the pipeline's extraction object shape."""
    if isinstance(value, dict):
        entities = value.get("entities", [])
        relations = value.get("relations", [])
        return {
            **value,
            "entities": entities if isinstance(entities, list) else [],
            "relations": relations if isinstance(relations, list) else [],
        }
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], dict) and (
            "entities" in value[0] or "relations" in value[0]
        ):
            return coerce_extraction(value[0])
        return {"entities": [], "relations": [item for item in value if isinstance(item, dict)]}
    return {"entities": [], "relations": []}


# ============================================================
# Stage 2: 校验补全
# ============================================================
def validate_extraction(extraction: dict, source_record: dict) -> dict:
    """
    校验提取结果:
    - 检查证据句是否在原文中
    - 检查 subject/object 是否能对齐实体表
    - 检查关系类型是否符合 schema 签名
    - 标记 import_ready，避免低质量候选直接写库
    """
    extraction = coerce_extraction(extraction)
    abstract = source_record.get("abstract", "")
    title = source_record.get("title", "")
    full_text = f"{title}\n{abstract}"

    entities = _clean_entities(extraction.get("entities", []), full_text)
    relations = [r for r in extraction.get("relations", []) if isinstance(r, dict)]

    # 如果关系里出现了实体表漏掉的 mention，且类型合法、原文可定位，则补入实体。
    for rel in relations:
        for side in ("subject", "object"):
            mention = _clean_text(rel.get(side, ""))
            etype = _normalize_entity_type(rel.get(f"{side}_type", ""))
            if mention and etype and not _find_entity(entities, mention, etype):
                if _mention_present(mention, full_text):
                    entities.append({"mention": mention, "type": etype, "normalized_id": ""})
    extraction["entities"] = _dedupe_entities(entities)

    validated_relations = []
    for raw_rel in relations:
        rel = dict(raw_rel)
        flags: list[str] = []

        subject = _clean_text(rel.get("subject", rel.get("subject_mention", "")))
        obj = _clean_text(rel.get("object", rel.get("object_mention", "")))
        predicate = _normalize_predicate(rel.get("predicate", rel.get("relation_type_candidate", "")))
        evidence = _clean_text(rel.get("evidence", rel.get("evidence_sentence", "")))

        rel["subject"] = subject
        rel["object"] = obj
        rel["predicate"] = predicate
        rel["evidence"] = evidence

        grounded, start, end = _locate_evidence(evidence, full_text)
        rel["evidence_grounded"] = grounded
        rel["evidence_char_start"] = start
        rel["evidence_char_end"] = end
        if not grounded:
            flags.append("ungrounded_evidence")

        subject_type = _normalize_entity_type(rel.get("subject_type", ""))
        object_type = _normalize_entity_type(rel.get("object_type", ""))
        subject_entity = _find_entity(extraction["entities"], subject, subject_type)
        object_entity = _find_entity(extraction["entities"], obj, object_type)

        if not subject_entity:
            subject_entity = _find_entity(extraction["entities"], subject, "")
        if not object_entity:
            object_entity = _find_entity(extraction["entities"], obj, "")
        if subject_entity and not subject_type:
            subject_type = subject_entity.get("type", "")
        if object_entity and not object_type:
            object_type = object_entity.get("type", "")

        rel["subject_type"] = subject_type
        rel["object_type"] = object_type

        if not subject:
            flags.append("missing_subject")
        elif not subject_entity:
            flags.append("subject_not_in_entities")
        if not obj:
            flags.append("missing_object")
        elif not object_entity:
            flags.append("object_not_in_entities")
        if not predicate:
            flags.append("missing_predicate")
        elif predicate not in RELATION_SIGNATURES:
            flags.append("invalid_predicate")

        schema_valid = (
            bool(predicate)
            and predicate in RELATION_SIGNATURES
            and (subject_type, object_type) in RELATION_SIGNATURES.get(predicate, set())
        )
        rel["schema_valid"] = schema_valid
        if predicate in RELATION_SIGNATURES and not schema_valid:
            flags.append(f"schema_mismatch:{subject_type or '?'}->{object_type or '?'}")

        rel["negated"] = _detect_negation(evidence)
        rel["uncertain"] = bool(rel.get("uncertain")) or _detect_uncertainty(evidence)
        if rel["negated"]:
            flags.append("negated_relation")
        if rel["uncertain"]:
            flags.append("uncertain_relation")

        rel["direction"] = _normalize_direction(rel.get("direction", "unknown"))
        rel["species"] = _normalize_species(rel.get("species", ""), evidence)
        rel["disease_stage"] = _clean_text(rel.get("disease_stage", rel.get("disease_stage_candidate", "")))
        if _is_non_human_or_mixed(rel["species"], evidence):
            flags.append("non_human_or_mixed_species")

        if predicate == "PROGRESSES_TO" and _looks_like_generic_progression(evidence):
            flags.append("generic_background_progression")
        if predicate in RELATION_SIGNATURES and predicate not in NEO4J_IMPORTABLE_PREDICATES:
            flags.append("not_importable_policy")
        if _looks_like_forced_entity_class(subject, subject_type) or _looks_like_forced_entity_class(obj, object_type):
            flags.append("unsupported_entity_class")

        rel["quality_flags"] = sorted(set(flags))
        rel["requires_review"] = bool(rel["quality_flags"])
        rel["confidence_score"] = _score_relation(rel)
        rel["import_ready"] = _is_import_ready(rel)
        validated_relations.append(rel)

    extraction["relations"] = _dedupe_relations(validated_relations)
    _attach_relation_entity_ids(extraction)
    extraction["validation_summary"] = _summarize_extraction(extraction)
    return extraction


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_entity_type(value: Any) -> str:
    aliases = {
        "gene": "Gene",
        "genes": "Gene",
        "disease": "Disease",
        "disease_or_stage": "Disease",
        "protein": "Protein",
        "proteins": "Protein",
        "pathway": "Pathway",
        "pathways": "Pathway",
        "metabolite": "Metabolite",
        "chemical": "Metabolite",
        "compound": "Metabolite",
        "tissue": "Tissue",
        "cell": "CellType",
        "celltype": "CellType",
        "cell_type": "CellType",
        "cell type": "CellType",
    }
    text = _clean_text(value)
    if text in ALLOWED_ENTITY_TYPES:
        return text
    return aliases.get(text.lower(), "")


def _normalize_predicate(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()
    aliases = {
        "ASSOCIATED_WITH_DISEASE": "ASSOCIATED_WITH",
        "ASSOCIATED_WITH_METABOLIC_CHANGE": "ASSOCIATED_WITH",
        "PREDICTS": "PROGNOSTIC_IN",
        "PROGNOSTIC_FOR": "PROGNOSTIC_IN",
        "PART_OF": "PARTICIPATES_IN",
        "INVOLVED_IN": "PARTICIPATES_IN",
    }
    return aliases.get(normalized, normalized)


def _normalize_direction(value: Any) -> str:
    normalized = _clean_text(value).lower().replace(" ", "_")
    aliases = {
        "up": "increase",
        "upregulated": "increase",
        "increased": "increase",
        "promotes": "increase",
        "exacerbates": "increase",
        "down": "decrease",
        "downregulated": "decrease",
        "decreased": "decrease",
        "suppresses": "decrease",
        "reduces": "decrease",
        "positive_association": "positive",
        "negative_association": "negative",
        "no_change": "none",
        "": "unknown",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in ALLOWED_DIRECTIONS else "unknown"


def _normalize_species(value: Any, evidence: str) -> str:
    supplied = _clean_text(value)
    lower = supplied.lower()
    if lower in {"homo sapiens", "human", "humans", "patients"}:
        return "Homo sapiens"
    if lower in {"mus musculus", "mouse", "mice"}:
        return "Mus musculus"
    if lower in {"rat", "rats", "rattus norvegicus"}:
        return "Rattus norvegicus"
    evidence_lower = evidence.lower()
    has_human = any(term in evidence_lower for term in ("human", "patients", "patient"))
    has_mouse = any(term in evidence_lower for term in ("mouse", "mice", "murine"))
    if has_human and has_mouse:
        return "mixed: Homo sapiens; Mus musculus"
    if has_mouse:
        return "Mus musculus"
    if has_human:
        return "Homo sapiens"
    return supplied or "Homo sapiens"


def _detect_negation(evidence: str) -> bool:
    text = evidence.lower()
    text = re.sub(r"\bnot\s+only\b", "notonly", text)
    patterns = [
        r"\bnot\b",
        r"\bno\b",
        r"\bneither\b",
        r"\bwithout\b",
        r"\babsence of\b",
        r"\bfailed to\b",
        r"\black of\b",
        r"不含",
        r"未检测到",
        r"无",
        r"排除",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _detect_uncertainty(evidence: str) -> bool:
    text = evidence.lower()
    patterns = [
        r"\bmay\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bsuggests?\b",
        r"\bsuggested\b",
        r"\bpotential(ly)?\b",
        r"\bpossibly\b",
        r"\blikely\b",
        r"\bunclear\b",
        r"\bindicates?\b",
        r"可能",
        r"提示",
        r"推测",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _locate_evidence(evidence: str, full_text: str) -> tuple[bool, int, int]:
    if not evidence:
        return False, -1, -1
    start = full_text.find(evidence)
    if start >= 0:
        return True, start, start + len(evidence)
    pattern = r"\s+".join(re.escape(part) for part in evidence.split())
    match = re.search(pattern, full_text)
    if match:
        return True, match.start(), match.end()
    return False, -1, -1


def _mention_present(mention: str, full_text: str) -> bool:
    if not mention:
        return False
    return mention in full_text or mention.lower() in full_text.lower()


def _clean_entities(raw_entities: list, full_text: str) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for raw in raw_entities:
        if not isinstance(raw, dict):
            continue
        mention = _clean_text(raw.get("mention", raw.get("name", "")))
        etype = _normalize_entity_type(raw.get("type", raw.get("entity_type", "")))
        if not mention or not etype:
            continue
        entity = dict(raw)
        entity["mention"] = mention
        entity["type"] = etype
        entity.setdefault("normalized_id", "")
        entity["mention_grounded"] = _mention_present(mention, full_text)
        entities.append(entity)
    return _dedupe_entities(entities)


def _dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for entity in entities:
        key = (
            entity.get("type", ""),
            entity.get("normalized_id", "") or entity.get("mention", "").lower(),
            entity.get("mention", "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(entity)
    return unique


def _entity_key(mention: str) -> str:
    return re.sub(r"\s+", " ", mention.strip()).lower()


def _find_entity(
    entities: list[dict[str, Any]],
    mention: str,
    entity_type: str = "",
) -> dict[str, Any] | None:
    key = _entity_key(mention)
    if not key:
        return None
    for entity in entities:
        if entity_type and entity.get("type") != entity_type:
            continue
        if _entity_key(entity.get("mention", "")) == key:
            return entity
    return None


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str, str, str, str], int] = {}
    unique: list[dict[str, Any]] = []
    for rel in relations:
        key = (
            rel.get("subject", "").lower(),
            rel.get("subject_type", ""),
            rel.get("predicate", ""),
            rel.get("object", "").lower(),
            rel.get("object_type", ""),
        )
        if key in seen:
            existing = unique[seen[key]]
            evidence = rel.get("evidence", "")
            if evidence and evidence != existing.get("evidence", ""):
                existing.setdefault("duplicate_evidence", []).append(evidence)
            existing["duplicate_count"] = existing.get("duplicate_count", 1) + 1
            if rel.get("confidence_score", 0) > existing.get("confidence_score", 0):
                duplicate_evidence = existing.get("duplicate_evidence", [])
                duplicate_count = existing.get("duplicate_count", 1)
                rel["duplicate_evidence"] = duplicate_evidence
                rel["duplicate_count"] = duplicate_count
                unique[seen[key]] = rel
            continue
        seen[key] = len(unique)
        unique.append(rel)
    return unique


def _attach_relation_entity_ids(extraction: dict) -> None:
    entities = extraction.get("entities", [])
    for rel in extraction.get("relations", []):
        flags = set(rel.get("quality_flags", []))
        subject_entity = _find_entity(entities, rel.get("subject", ""), rel.get("subject_type", ""))
        object_entity = _find_entity(entities, rel.get("object", ""), rel.get("object_type", ""))
        if subject_entity:
            rel["subject_id"] = subject_entity.get("normalized_id") or _fallback_entity_id(subject_entity)
        else:
            flags.add("subject_not_in_entities")
        if object_entity:
            rel["object_id"] = object_entity.get("normalized_id") or _fallback_entity_id(object_entity)
        else:
            flags.add("object_not_in_entities")
        rel["quality_flags"] = sorted(flags)
        rel["requires_review"] = bool(rel["quality_flags"])
        rel["import_ready"] = _is_import_ready(rel)


def _fallback_entity_id(entity: dict[str, Any]) -> str:
    return f"MENTION:{entity.get('type','Entity')}_{entity.get('mention','').replace(' ', '_')}"


def _looks_like_generic_progression(evidence: str) -> bool:
    lower = evidence.lower()
    return any(cue in lower for cue in GENERIC_PROGRESSION_CUES)


def _looks_like_forced_entity_class(mention: str, entity_type: str) -> bool:
    lower = _clean_text(mention).lower()
    return any(cue in lower for cue in SCHEMA_FORCED_ENTITY_CUES.get(entity_type, ()))


def _is_non_human_or_mixed(species: str, evidence: str) -> bool:
    lower = f"{species} {evidence}".lower()
    return "mixed:" in lower or any(term in lower for term in NON_HUMAN_SPECIES)


def _score_relation(rel: dict[str, Any]) -> float:
    score = 0.9
    flags = set(rel.get("quality_flags", []))
    if "ungrounded_evidence" in flags:
        score -= 0.5
    if not rel.get("schema_valid"):
        score -= 0.35
    if rel.get("negated"):
        score -= 0.35
    if rel.get("uncertain"):
        score -= 0.15
    if "non_human_or_mixed_species" in flags:
        score -= 0.1
    if "generic_background_progression" in flags:
        score -= 0.25
    if "not_importable_policy" in flags:
        score -= 0.25
    if "unsupported_entity_class" in flags:
        score -= 0.3
    if "subject_not_in_entities" in flags or "object_not_in_entities" in flags:
        score -= 0.25
    return round(max(0.05, min(score, 0.95)), 3)


def _is_import_ready(rel: dict[str, Any]) -> bool:
    blocking_flags = {
        "ungrounded_evidence",
        "missing_subject",
        "missing_object",
        "missing_predicate",
        "invalid_predicate",
        "subject_not_in_entities",
        "object_not_in_entities",
        "negated_relation",
        "uncertain_relation",
        "generic_background_progression",
        "non_human_or_mixed_species",
        "not_importable_policy",
        "unsupported_entity_class",
    }
    flags = set(rel.get("quality_flags", []))
    return bool(
        rel.get("evidence_grounded")
        and rel.get("schema_valid")
        and rel.get("predicate") in NEO4J_IMPORTABLE_PREDICATES
        and not rel.get("negated")
        and not rel.get("uncertain")
        and not (flags & blocking_flags)
    )


def _summarize_extraction(extraction: dict) -> dict[str, Any]:
    relations = extraction.get("relations", [])
    flag_counts: dict[str, int] = {}
    for rel in relations:
        for flag in rel.get("quality_flags", []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    return {
        "total_entities": len(extraction.get("entities", [])),
        "total_relations": len(relations),
        "evidence_grounded": sum(1 for r in relations if r.get("evidence_grounded", False)),
        "schema_valid": sum(1 for r in relations if r.get("schema_valid", False)),
        "import_ready": sum(1 for r in relations if r.get("import_ready", False)),
        "requires_review": sum(1 for r in relations if r.get("requires_review", False)),
        "negated": sum(1 for r in relations if r.get("negated", False)),
        "uncertain": sum(1 for r in relations if r.get("uncertain", False)),
        "quality_flags": flag_counts,
    }


# ============================================================
# Stage 3: 实体标准化（基于规则 + 精确匹配现有图谱）
# ============================================================
KNOWN_GENE_IDS = {
    "TP53": "HGNC:11998",
    "SLC7A11": "HGNC:10916",
    "CDKN1B": "HGNC:1785",
    "TFAM": "HGNC:11741",
    "PNPLA3": "HGNC:18547",
    "HNF4A": "HGNC:5024",
    "TRIB3": "HGNC:16247",
    "FOXA3": "HGNC:5025",
    "DRAK2": "HGNC:24542",
    "MERTK": "HGNC:7035",
    "PPARA": "HGNC:9228",
    "ATGL": "HGNC:18335",
    "LONP1": "HGNC:8086",
    "BAZ2B": "HGNC:17694",
    "TM7SF3": "HGNC:20961",
    "ADAMTSL2": "HGNC:14632",
    "AKR1B10": "HGNC:379",
    "CFHR4": "HGNC:16963",
    "TREM2": "HGNC:17758",
    "MARCH3": "HGNC:17382",
    "GPX4": "HGNC:4558",
    "IL6": "HGNC:6018",
    "SREBP1C": "HGNC:11289",
    "TEAD1": "HGNC:11727",
    "EHBP1": "HGNC:24559",
}

KNOWN_DISEASE_IDS = {
    "NAFLD": "UMLS:C0400966",
    "MASLD": "UMLS:C0400966",
    "NASH": "UMLS:C3241937",
    "MASH": "UMLS:C3241937",
    "fibrosis": "UMLS:C0239946",
    "liver fibrosis": "UMLS:C0239946",
    "cirrhosis": "UMLS:C0023890",
    "liver cirrhosis": "UMLS:C0023890",
    "HCC": "UMLS:C2239176",
    "hepatocellular carcinoma": "UMLS:C2239176",
    "liver cancer": "UMLS:C2239176",
    "non-alcoholic fatty liver disease": "UMLS:C0400966",
    "metabolic dysfunction-associated steatohepatitis": "UMLS:C3241937",
    "metabolic dysfunction-associated steatotic liver disease": "UMLS:C0400966",
    "steatohepatitis": "UMLS:C3241937",
    "steatotic liver disease": "UMLS:C0400966",
}


def normalize_entity(entity: dict) -> dict:
    """使用规则 + 已知映射表进行实体标准化。"""
    mention = entity.get("mention", "").strip()
    etype = entity.get("type", "")

    if etype == "Gene":
        # 尝试精确匹配
        gene_upper = mention.upper()
        if gene_upper in KNOWN_GENE_IDS:
            entity["normalized_id"] = KNOWN_GENE_IDS[gene_upper]
        # 尝试从 mention 中提取第一个词
        elif len(mention.split()) > 1:
            first_word = mention.split()[0].upper().rstrip(",")
            if first_word in KNOWN_GENE_IDS:
                entity["normalized_id"] = KNOWN_GENE_IDS[first_word]
        else:
            entity["normalized_id"] = f"MENTION:Gene_{mention.replace(' ', '_')}"

    elif etype == "Disease":
        mention_lower = mention.lower().strip()
        # 精确匹配
        matched = False
        for key, umls_id in KNOWN_DISEASE_IDS.items():
            if key.lower() == mention_lower or key.lower() in mention_lower:
                entity["normalized_id"] = umls_id
                matched = True
                break
        if not matched:
            entity["normalized_id"] = f"MENTION:Disease_{mention.replace(' ', '_')}"

    elif etype == "Pathway":
        entity["normalized_id"] = f"MENTION:Pathway_{mention.replace(' ', '_')}"

    elif etype == "Protein":
        # 尝试映射 gene symbol → protein
        mention_upper = mention.upper()
        if mention_upper in KNOWN_GENE_IDS:
            entity["normalized_id"] = f"UNIPROT:derived_from_{KNOWN_GENE_IDS[mention_upper]}"
        else:
            entity["normalized_id"] = f"MENTION:Protein_{mention.replace(' ', '_')}"

    elif etype == "Metabolite":
        entity["normalized_id"] = f"MENTION:Metabolite_{mention.replace(' ', '_')}"

    else:
        entity["normalized_id"] = f"MENTION:{etype}_{mention.replace(' ', '_')}"

    return entity


# ============================================================
# Stage 4: 融合与冲突检测
# ============================================================
def check_conflicts(new_relations: list[dict], existing_kg_summary: dict) -> list[dict]:
    """
    对比新提取的关系与已有图谱摘要，标记潜在冲突。
    当前使用基于规则的轻量检测（不查询 Neo4j 全量）。
    """
    for rel in new_relations:
        conflicts = []
        subj = rel.get("subject", "")
        pred = rel.get("predicate", "")
        obj = rel.get("object", "")
        negated = rel.get("negated", False)
        direction = rel.get("direction", "")

        # 规则 1: 如果已有关系说 X promotes Y，新提取说 X inhibits Y → 标记冲突
        # （需要查询 Neo4j 实际数据，这里给出框架）
        if negated:
            conflicts.append({"type": "negated_claim", "severity": "low"})

        # 规则 2: 低置信度标记
        if rel.get("confidence_score", 1.0) < 0.5:
            conflicts.append({"type": "low_confidence", "severity": "warning"})

        rel["conflicts"] = conflicts

    return new_relations


# ============================================================
# Stage 5: Neo4j 写入模块
# ============================================================
def write_to_neo4j(extraction: dict, record: dict) -> dict:
    """
    将 LLM 提取结果按目标 Neo4j KG schema 写入已有核心节点之间。

    注意：LLM 抽取结果默认不是 curated fact，因此只写入带有
    validation_status='candidate' 的目标 schema 关系；不会创建 Article、
    LLMEntity 或任何新的核心实体节点。端点匹配不到现有节点时跳过。
    返回写入统计。
    """
    if not NEO4J_PASSWORD:
        return {"status": "skipped", "reason": "NEO4J_PASSWORD not set"}

    try:
        from neo4j import GraphDatabase
    except ImportError:
        return {"status": "skipped", "reason": "neo4j driver not installed"}

    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    stats = {
        "status": "ok",
        "relations_imported": 0,
        "relations_skipped_quality": 0,
        "relations_skipped_schema": 0,
        "relations_skipped_non_importable_predicate": 0,
        "relations_skipped_unmatched_endpoint": 0,
        "errors": [],
    }

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            pmid = str(record.get("pmid", ""))
            for rel in extraction.get("relations", []):
                if not rel.get("import_ready"):
                    stats["relations_skipped_quality"] += 1
                    continue

                pred = rel.get("predicate", "")
                source_type = rel.get("subject_type", "")
                target_type = rel.get("object_type", "")
                if (
                    pred not in RELATION_SIGNATURES
                    or (source_type, target_type) not in RELATION_SIGNATURES.get(pred, set())
                ):
                    stats["relations_skipped_schema"] += 1
                    continue
                if pred not in NEO4J_IMPORTABLE_PREDICATES:
                    stats["relations_skipped_non_importable_predicate"] += 1
                    continue

                source_node = _match_neo4j_node(
                    session=session,
                    entity_type=source_type,
                    mention=rel.get("subject", ""),
                    normalized_id=rel.get("subject_id")
                    or _lookup_entity_id(extraction, rel.get("subject", ""), source_type),
                )
                target_node = _match_neo4j_node(
                    session=session,
                    entity_type=target_type,
                    mention=rel.get("object", ""),
                    normalized_id=rel.get("object_id")
                    or _lookup_entity_id(extraction, rel.get("object", ""), target_type),
                )

                if not source_node or not target_node:
                    stats["relations_skipped_unmatched_endpoint"] += 1
                    continue

                evidence = rel.get("evidence", rel.get("evidence_sentence", ""))
                id_property = NEO4J_RELATION_ID_PROPERTY[pred]
                rel_id = _neo4j_candidate_relationship_id(
                    predicate=pred,
                    pmid=pmid,
                    source_node_id=source_node["node_id"],
                    target_node_id=target_node["node_id"],
                    evidence=evidence,
                )
                cypher = f"""
                MATCH (a) WHERE elementId(a) = $source_element_id
                MATCH (b) WHERE elementId(b) = $target_element_id
                MERGE (a)-[r:{pred} {{{id_property}: $relationship_id}}]->(b)
                ON CREATE SET r.created_at = $now
                SET r.source = 'LLM_extraction',
                    r.source_record_id = $source_record_id,
                    r.publication_id = $publication_id,
                    r.evidence_level = 'llm_extracted_candidate',
                    r.evidence_sentence = $evidence,
                    r.evidence_text = $evidence,
                    r.evidence_char_start = $evidence_char_start,
                    r.evidence_char_end = $evidence_char_end,
                    r.confidence_score = $confidence,
                    r.negated = $negated,
                    r.uncertain = $uncertain,
                    r.species = $species,
                    r.direction = $direction,
                    r.disease_stage = $disease_stage,
                    r.validation_status = 'candidate',
                    r.quality_flags_json = $quality_flags_json,
                    r.updated_at = $now
                """
                try:
                    session.run(cypher, {
                        "source_element_id": source_node["element_id"],
                        "target_element_id": target_node["element_id"],
                        "relationship_id": rel_id,
                        "source_record_id": record.get("pmid", ""),
                        "publication_id": f"PMID:{pmid}" if pmid else "",
                        "evidence": evidence,
                        "evidence_char_start": rel.get("evidence_char_start", -1),
                        "evidence_char_end": rel.get("evidence_char_end", -1),
                        "confidence": rel.get("confidence_score", 0.8),
                        "negated": rel.get("negated", False),
                        "uncertain": rel.get("uncertain", False),
                        "species": rel.get("species", ""),
                        "direction": rel.get("direction", ""),
                        "disease_stage": rel.get("disease_stage", ""),
                        "quality_flags_json": json.dumps(rel.get("quality_flags", []), ensure_ascii=False),
                        "now": datetime.now(timezone.utc).isoformat(),
                    })
                    stats["relations_imported"] += 1
                except Exception as e:
                    stats["errors"].append(f"Relation merge error: {e}")

    finally:
        driver.close()

    return stats


def _match_neo4j_node(session: Any, entity_type: str, mention: str, normalized_id: str | None = None) -> dict[str, str] | None:
    """在目标 KG 的核心节点中查找实体；不创建新节点。"""
    mention = _clean_text(mention)
    normalized_id = _clean_text(normalized_id)
    normalized_without_prefix = re.sub(r"^MENTION:[^_]+_", "", normalized_id).replace("_", " ")

    queries = {
        "Gene": """
            MATCH (n:Gene)
            WHERE n.gene_id = $normalized_id
               OR toLower(n.gene_symbol) = toLower($mention)
               OR toLower(n.name) = toLower($mention)
               OR toLower(n.gene_symbol) = toLower($normalized_without_prefix)
            RETURN elementId(n) AS element_id, n.gene_id AS node_id
            LIMIT 1
        """,
        "Disease": """
            MATCH (n:Disease)
            WHERE n.disease_id = $normalized_id
               OR toLower(n.name) = toLower($mention)
               OR toLower(n.disease_name) = toLower($mention)
               OR toLower(n.name) = toLower($normalized_without_prefix)
            RETURN elementId(n) AS element_id, n.disease_id AS node_id
            LIMIT 1
        """,
        "Protein": """
            MATCH (n:Protein)
            WHERE n.protein_id = $normalized_id
               OR n.string_protein_id = $normalized_id
               OR toLower(n.name) = toLower($mention)
               OR toLower(n.name) = toLower($normalized_without_prefix)
            RETURN elementId(n) AS element_id, coalesce(n.string_protein_id, n.protein_id) AS node_id
            LIMIT 1
        """,
        "Pathway": """
            MATCH (n:Pathway)
            WHERE n.pathway_id = $normalized_id
               OR toLower(n.name) = toLower($mention)
               OR toLower(n.name) = toLower($normalized_without_prefix)
            RETURN elementId(n) AS element_id, n.pathway_id AS node_id
            LIMIT 1
        """,
        "Metabolite": """
            MATCH (n:Metabolite)
            WHERE n.metabolite_id = $normalized_id
               OR toLower(n.name) = toLower($mention)
               OR toLower(n.name) = toLower($normalized_without_prefix)
            RETURN elementId(n) AS element_id, n.metabolite_id AS node_id
            LIMIT 1
        """,
        "Tissue": """
            MATCH (n:Tissue)
            WHERE n.tissue_id = $normalized_id
               OR toLower(n.name) = toLower($mention)
               OR toLower(n.tissue_name) = toLower($mention)
               OR toLower(n.name) = toLower($normalized_without_prefix)
            RETURN elementId(n) AS element_id, n.tissue_id AS node_id
            LIMIT 1
        """,
        "CellType": """
            MATCH (n:CellType)
            WHERE n.cell_type_id = $normalized_id
               OR toLower(n.name) = toLower($mention)
               OR toLower(n.cell_type_name) = toLower($mention)
               OR toLower(n.name) = toLower($normalized_without_prefix)
            RETURN elementId(n) AS element_id, n.cell_type_id AS node_id
            LIMIT 1
        """,
    }
    cypher = queries.get(entity_type)
    if not cypher or not mention:
        return None

    record = session.run(
        cypher,
        {
            "mention": mention,
            "normalized_id": normalized_id,
            "normalized_without_prefix": normalized_without_prefix,
        },
    ).single()
    if not record:
        return None
    return {"element_id": record["element_id"], "node_id": record["node_id"] or ""}


def _neo4j_candidate_relationship_id(
    predicate: str,
    pmid: str,
    source_node_id: str,
    target_node_id: str,
    evidence: str,
) -> str:
    payload = "|".join([predicate, pmid, source_node_id, target_node_id, evidence])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"LLM_{predicate}:{digest}"


def _lookup_entity_id(extraction: dict, mention: str, entity_type: str = "") -> str | None:
    """从提取结果中查找实体 mention 对应的 normalized_id。"""
    for entity in extraction.get("entities", []):
        if entity_type and entity.get("type") != entity_type:
            continue
        if _entity_key(entity.get("mention", "")) == _entity_key(mention):
            return entity.get("normalized_id", "")
    return None


# ============================================================
# 生成溯源报告
# ============================================================
def generate_provenance_report(
    all_results: list[dict],
    run_id: str,
    output_dir: Path,
) -> str:
    """生成一个简单的交互式 HTML 溯源报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"provenance_report_{run_id}.html"

    rows_html = ""
    for i, result in enumerate(all_results):
        record = result.get("record", {})
        extraction = result.get("extraction", {})
        classification = result.get("classification", "")
        stats = result.get("neo4j_stats", {})

        relations_html = ""
        for rel in extraction.get("relations", []):
            if rel.get("import_ready"):
                color = "#d4edda"
            elif rel.get("schema_valid") and rel.get("evidence_grounded"):
                color = "#fff3cd"
            else:
                color = "#f8d7da"
            flags = ", ".join(rel.get("quality_flags", []))
            relations_html += f"""
            <tr style="background:{color}">
                <td>{html.escape(rel.get('subject',''))}</td>
                <td>{html.escape(rel.get('predicate',''))}</td>
                <td>{html.escape(rel.get('object',''))}</td>
                <td>{html.escape(rel.get('direction',''))}</td>
                <td>{'yes' if rel.get('import_ready') else 'review'}</td>
                <td style="font-size:0.85em">{html.escape(rel.get('evidence','')[:160])}...</td>
                <td>{'yes' if rel.get('evidence_grounded') else 'no'}</td>
                <td style="font-size:0.8em">{html.escape(flags)}</td>
            </tr>"""

        entities_html = ""
        for ent in extraction.get("entities", []):
            entities_html += f"""
            <tr>
                <td>{html.escape(ent.get('mention',''))}</td>
                <td>{html.escape(ent.get('type',''))}</td>
                <td style="font-family:monospace;font-size:0.8em">{html.escape(ent.get('normalized_id',''))}</td>
            </tr>"""

        rows_html += f"""
        <div class="record">
            <h3>#{i+1} — PMID:{html.escape(str(record.get('pmid','?')))} | Class: {html.escape(str(classification))}</h3>
            <p><strong>Title:</strong> {html.escape(record.get('title',''))}</p>
            <details>
                <summary>Abstract (click to expand)</summary>
                <p class="abstract">{html.escape(record.get('abstract','')[:800])}...</p>
            </details>
            <h4>Entities ({len(extraction.get('entities',[]))})</h4>
            <table><tr><th>Mention</th><th>Type</th><th>Normalized ID</th></tr>{entities_html}</table>
            <h4>Relations ({len(extraction.get('relations',[]))})</h4>
            <table><tr><th>Subject</th><th>Predicate</th><th>Object</th><th>Direction</th><th>Import</th><th>Evidence</th><th>Grounded</th><th>Flags</th></tr>{relations_html}</table>
            <p><small>Validation: {html.escape(json.dumps(extraction.get('validation_summary',{}), ensure_ascii=False))}</small></p>
            <p><small>Neo4j write: {html.escape(json.dumps(stats, ensure_ascii=False))}</small></p>
        </div><hr>"""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Multi-Stage Extraction Provenance Report — {run_id}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width:1200px; margin:0 auto; padding:20px; background:#fafafa; }}
.record {{ background:white; border-radius:8px; padding:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,.1); }}
.abstract {{ color:#555; font-size:0.9em; line-height:1.5; }}
table {{ border-collapse:collapse; width:100%; margin:8px 0; }}
th, td {{ border:1px solid #ddd; padding:6px 10px; text-align:left; font-size:0.9em; }}
th {{ background:#f5f5f5; }}
h3 {{ color:#2c3e50; margin-bottom:4px; }}
h4 {{ color:#34495e; margin:12px 0 4px; }}
details {{ margin:8px 0; }}
summary {{ cursor:pointer; color:#3498db; }}
</style>
</head>
<body>
<h1>🔬 Multi-Stage LangExtract Provenance Report</h1>
<p>Run ID: <strong>{run_id}</strong> | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p>Total records processed: {len(all_results)}</p>
{rows_html}
</body>
</html>"""

    html_path.write_text(html_doc, encoding="utf-8")
    return str(html_path)


def generate_quality_report(
    all_results: list[dict],
    run_id: str,
    output_dir: Path,
) -> str:
    """生成机器可读的抽取质量摘要。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"quality_report_{run_id}.json"
    summary = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": len(all_results),
        "entities": 0,
        "relations": 0,
        "schema_valid_relations": 0,
        "import_ready_relations": 0,
        "review_relations": 0,
        "flag_counts": {},
        "records_detail": [],
    }

    for result in all_results:
        record = result.get("record", {})
        extraction = result.get("extraction", {})
        validation = extraction.get("validation_summary", {})
        summary["entities"] += len(extraction.get("entities", []))
        summary["relations"] += len(extraction.get("relations", []))
        summary["schema_valid_relations"] += validation.get("schema_valid", 0)
        summary["import_ready_relations"] += validation.get("import_ready", 0)
        summary["review_relations"] += validation.get("requires_review", 0)
        for flag, count in validation.get("quality_flags", {}).items():
            summary["flag_counts"][flag] = summary["flag_counts"].get(flag, 0) + count
        summary["records_detail"].append(
            {
                "pmid": record.get("pmid", ""),
                "title": record.get("title", ""),
                "relations": validation.get("total_relations", 0),
                "import_ready": validation.get("import_ready", 0),
                "requires_review": validation.get("requires_review", 0),
                "quality_flags": validation.get("quality_flags", {}),
            }
        )

    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(report_path)


# ============================================================
# 单篇文章处理（供并发调用）
# ============================================================
def _process_single_record(
    idx: int,
    total: int,
    record: dict,
    skip_neo4j: bool = True,
    progress_lock: threading.Lock | None = None,
    progress_counter: list | None = None,
) -> dict:
    """处理单条文献记录（五阶段 Pipeline），线程安全。"""
    pmid = record.get("pmid", "?")

    result = {"record": record, "stage_outputs": {}, "_batch_index": idx}

    # --- Stage 0: 文本分类 ---
    text_type = classify_record(record)
    result["classification"] = text_type

    # --- Stage 1: 分块初提取 ---
    try:
        if DEEPSEEK_API_KEY:
            prompt = build_stage1_prompt(record, text_type)
            extraction = call_deepseek_api(prompt)
        else:
            extraction = _mock_extraction(record)
        result["stage_outputs"]["stage1_raw"] = copy.deepcopy(extraction)
    except Exception as e:
        extraction = {"entities": [], "relations": [], "error": str(e)}

    # --- Stage 2: 校验补全 ---
    extraction = validate_extraction(extraction, record)
    result["stage_outputs"]["stage2_validated"] = {
        "validation_summary": extraction.get("validation_summary", {}),
    }

    # --- Stage 3: 实体标准化 ---
    for ent in extraction.get("entities", []):
        normalize_entity(ent)
    extraction["entities"] = _dedupe_entities(extraction.get("entities", []))
    _attach_relation_entity_ids(extraction)
    extraction["validation_summary"] = _summarize_extraction(extraction)
    result["stage_outputs"]["stage3_normalized"] = {
        "entity_count": len(extraction.get("entities", [])),
        "import_ready_relations": extraction.get("validation_summary", {}).get("import_ready", 0),
    }

    # --- Stage 4: 冲突检测 ---
    extraction["relations"] = check_conflicts(
        extraction.get("relations", []), {}
    )
    conflict_count = sum(
        1 for r in extraction.get("relations", []) if r.get("conflicts")
    )
    result["stage_outputs"]["stage4_conflicts"] = {
        "relations_with_conflicts": conflict_count,
    }

    # --- Stage 5: Neo4j 写入 ---
    if skip_neo4j:
        neo4j_stats = {"status": "skipped", "reason": "--skip-neo4j flag"}
    else:
        neo4j_stats = write_to_neo4j(extraction, record)
    result["neo4j_stats"] = neo4j_stats

    result["extraction"] = extraction

    # 线程安全进度输出
    if progress_lock and progress_counter is not None:
        with progress_lock:
            progress_counter[0] += 1
            done = progress_counter[0]

        n_entities = len(extraction.get("entities", []))
        n_relations = len(extraction.get("relations", []))
        grounded = extraction.get("validation_summary", {}).get("evidence_grounded", 0)
        import_ready = extraction.get("validation_summary", {}).get("import_ready", 0)
        err = extraction.get("error", "")
        status = "❌" if err else "✓"
        print(f"[{done:>3}/{total}] PMID:{pmid} → {n_entities}E/{n_relations}R "
              f"({grounded} grounded, {import_ready} IR) {status}")

    return result


# ============================================================
# 主 Pipeline
# ============================================================
def run_pipeline(
    input_path: Path,
    limit: int = 10,
    run_id: str = "default",
    skip_neo4j: bool = True,
    max_workers: int = 5,
) -> list[dict]:
    """
    完整的五阶段提取 Pipeline（并发模式）。

    Args:
        input_path: 文献记录 JSONL 文件路径
        limit: 处理条数上限
        run_id: 运行标识
        skip_neo4j: 是否跳过 Neo4j 写入（默认跳过）
        max_workers: 并发 worker 数（默认 5）
    """
    if not DEEPSEEK_API_KEY:
        print("[WARN] DEEPSEEK_API_KEY 未设置。将使用模拟提取（不含 LLM 调用）。")

    # 读取输入
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records = records[:limit]
    total = len(records)
    print(f"[INFO] 加载了 {total} 条文献记录。")
    print(f"[INFO] 并发 worker 数: {max_workers}")

    all_results = []
    progress_lock = threading.Lock()
    progress_counter = [0]

    # ── 并发处理 ──
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, record in enumerate(records):
            future = executor.submit(
                _process_single_record,
                idx=idx,
                total=total,
                record=record,
                skip_neo4j=skip_neo4j,
                progress_lock=progress_lock,
                progress_counter=progress_counter,
            )
            futures[future] = idx

        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                idx = futures[future]
                pmid = records[idx].get("pmid", "?")
                print(f"[ERR] PMID:{pmid} — unhandled error: {e}")
                all_results.append({
                    "record": records[idx],
                    "extraction": {"entities": [], "relations": [], "error": str(e)},
                    "_batch_index": idx,
                })

    # 按原始顺序排列
    all_results.sort(key=lambda r: r.get("_batch_index", 0))

    return all_results


def _mock_extraction(record: dict) -> dict:
    """模拟提取（用于测试，不调用 LLM）。"""
    abstract = record.get("abstract", "")
    title = record.get("title", "")

    # 基于已知基因列表简单匹配
    entities = []
    for gene_symbol, hgnc_id in KNOWN_GENE_IDS.items():
        if gene_symbol.lower() in abstract.lower() or gene_symbol.lower() in title.lower():
            entities.append({
                "mention": gene_symbol,
                "type": "Gene",
                "normalized_id": hgnc_id,
            })

    for disease_name, umls_id in KNOWN_DISEASE_IDS.items():
        if disease_name.lower() in abstract.lower() or disease_name.lower() in title.lower():
            if not any(e["mention"] == disease_name for e in entities):
                entities.append({
                    "mention": disease_name,
                    "type": "Disease",
                    "normalized_id": umls_id,
                })

    # 去重（按 mention 去重）
    seen = set()
    unique_entities = []
    for e in entities:
        if e["mention"] not in seen:
            seen.add(e["mention"])
            unique_entities.append(e)

    return {"entities": unique_entities, "relations": []}


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Stage LangExtract Agent for Liver Disease KG"
    )
    parser.add_argument(
        "--input", "-i",
        default=str(SCRIPT_DIR / "workstreams/literature_hmdb_kegg/data/staging/literature/pubmed_demo_2026-06-14/literature_records.jsonl"),
        help="输入 JSONL 文件路径",
    )
    parser.add_argument("--limit", "-n", type=int, default=5, help="处理条数上限")
    parser.add_argument("--run-id", default="demo_001", help="运行标识")
    parser.add_argument("--skip-neo4j", action="store_true", default=True,
                        help="跳过 Neo4j 写入（默认开启）")
    parser.add_argument("--write-neo4j", action="store_true",
                        help="执行 Neo4j 写入")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="输出目录")
    parser.add_argument("--max-workers", type=int, default=5,
                        help="并发 worker 数 (默认 5，设 1 为串行)")
    args = parser.parse_args()

    if args.write_neo4j:
        args.skip_neo4j = False

    print("=" * 70)
    print("  Multi-Stage LangExtract Agent — Liver Disease KG")
    print(f"  Input: {args.input}")
    print(f"  Limit: {args.limit}  |  Run ID: {args.run_id}")
    print(f"  Workers: {args.max_workers} (concurrent)")
    print(f"  Neo4j: {'ENABLED' if not args.skip_neo4j else 'SKIPPED'}")
    print("=" * 70)

    # 环境变量检查
    if not DEEPSEEK_API_KEY:
        print("[WARN] DEEPSEEK_API_KEY 未设置 → 使用模拟提取")
    if not args.skip_neo4j and not NEO4J_PASSWORD:
        print("[ERROR] NEO4J_PASSWORD 未设置，无法写入 Neo4j")
        sys.exit(1)

    results = run_pipeline(
        input_path=Path(args.input),
        limit=args.limit,
        run_id=args.run_id,
        skip_neo4j=args.skip_neo4j,
        max_workers=args.max_workers,
    )

    # 保存中间结果
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"extraction_results_{args.run_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] 提取结果已保存: {json_path}")

    # 生成溯源报告
    report_path = generate_provenance_report(results, args.run_id, out_dir)
    print(f"[OK] 溯源报告已生成: {report_path}")
    quality_report_path = generate_quality_report(results, args.run_id, out_dir)
    print(f"[OK] 质量报告已生成: {quality_report_path}")

    # 汇总统计
    total_entities = sum(len(r["extraction"].get("entities", [])) for r in results)
    total_relations = sum(len(r["extraction"].get("relations", [])) for r in results)
    import_ready_relations = sum(
        r["extraction"].get("validation_summary", {}).get("import_ready", 0)
        for r in results
    )
    print(f"\n{'='*70}")
    print(f"  Pipeline 汇总")
    print(f"  Records: {len(results)}")
    print(f"  Total entities extracted: {total_entities}")
    print(f"  Total relations extracted: {total_relations}")
    print(f"  Target-schema import-ready relations: {import_ready_relations}")
    print(f"  Output dir: {out_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
