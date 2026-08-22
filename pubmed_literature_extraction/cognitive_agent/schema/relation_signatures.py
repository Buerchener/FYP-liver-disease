#!/usr/bin/env python3
"""
cognitive_agent/schema/relation_signatures.py — 统一关系签名

v1 baseline、v2/v3 Cognitive Agent 和 entity-linking preflight 共用本文件。
文献候选签名只描述语义上可表达的类型组合；主库写入契约由
``write_contract.py`` 单独定义。两层绝不能混用。
"""

# ── 关系签名 ──
# key: 关系谓词
# value: 允许的 (subject Neo4j label, object Neo4j label) 集合
LITERATURE_CANDIDATE_SIGNATURES: dict[str, set[tuple[str, str]]] = {
    # ASSOCIATED_WITH 是通用关联谓词，覆盖分子生物学中常见的二元关系。
    # v2 扩展：从 4 → 18 种签名，消除 56% 的误杀率。
    "ASSOCIATED_WITH": {
        # ── 基因/蛋白/代谢物/通路 → 疾病 (经典分子-疾病关联) ──
        ("Gene", "Disease"),
        ("Protein", "Disease"),
        ("Metabolite", "Disease"),
        ("Pathway", "Disease"),
        # ── 疾病 → 组织/通路/疾病 (疾病影响范围) ──
        ("Disease", "Tissue"),        # HCC → tumor microenvironment
        ("Disease", "Pathway"),       # MASLD → ferroptosis
        ("Disease", "Disease"),       # NAFLD → HCC (comorbidity)
        # ── 通路/组织/细胞 → 疾病 (逆向关联) ──
        ("Pathway", "Tissue"),        # Ferroptosis → liver
        ("Tissue", "Disease"),        # Fibrotic liver → cirrhosis
        ("CellType", "Disease"),      # T cells → hepatitis
        # ── 细胞/基因/蛋白 → 通路/组织 (分子定位) ──
        ("CellType", "Pathway"),      # T cells → immune response
        ("CellType", "Tissue"),       # Kupffer cells → liver
        ("Gene", "Tissue"),           # CYP2E1 → liver (表达位置)
        ("Gene", "Pathway"),          # NFE2L2 → antioxidant response
        ("Protein", "Pathway"),       # p53 → apoptosis
        ("Protein", "Tissue"),        # Albumin → blood
        ("Metabolite", "Pathway"),    # Glucose → glycolysis
        # ── 基因/蛋白间功能关联 ──
        ("Gene", "Gene"),             # co-expression / functional linkage
        ("Gene", "Protein"),          # regulatory relationship
        ("Protein", "Protein"),       # functional association
        # ── 通路/组织互作 ──
        ("Pathway", "Pathway"),       # pathway cross-talk
        ("Tissue", "Pathway"),        # liver → drug metabolism
    },
    "PROGNOSTIC_IN": {
        ("Gene", "Disease"),
        ("Protein", "Disease"),       # 蛋白质作为预后标志物
    },
    "PROGRESSES_TO": {
        ("Disease", "Disease"),
    },
    "ENCODES": {
        ("Gene", "Protein"),
    },
    "INTERACTS_WITH": {
        ("Protein", "Protein"),
        ("Gene", "Protein"),
        ("Protein", "Gene"),
        ("Gene", "Gene"),             # 基因-基因调控/共表达
        ("CellType", "CellType"),     # T cell ↔ B cell
        ("Metabolite", "Protein"),    # 代谢物-蛋白质结合
    },
    "PARTICIPATES_IN": {
        ("Gene", "Pathway"),
        ("Protein", "Pathway"),
        ("CellType", "Pathway"),      # T cells → immune response
        ("Metabolite", "Pathway"),    # Glucose → glycolysis
    },
    "EXPRESSED_IN": {
        ("Gene", "Tissue"),
        ("Gene", "CellType"),
        ("Protein", "Tissue"),        # 蛋白质表达定位
        ("Protein", "CellType"),      # 蛋白质在特定细胞类型表达
    },
    "ASSOCIATED_WITH_METABOLITE": {
        ("Gene", "Metabolite"),
        ("Protein", "Metabolite"),    # 蛋白质-代谢物互作
    },
}

# Backwards-compatible name for extractors and semantic tooling.  It must not
# be used to authorise a Neo4j write; use MAIN_KG_WRITE_CONTRACT instead.
RELATION_SIGNATURES = LITERATURE_CANDIDATE_SIGNATURES

# Deprecated compatibility exports for older pipeline modules.  New Neo4j
# write code imports these directly from ``write_contract``.
from .write_contract import NEO4J_IMPORTABLE_PREDICATES, RELATION_ID_PROPERTY  # noqa: E402

# ── 方向类型 ──
ALLOWED_DIRECTIONS: set[str] = {"positive", "negative", "increase", "decrease", "none", "unknown"}
