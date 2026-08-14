#!/usr/bin/env python3
"""
cognitive_agent/schema/examples.py — Few-shot 示例库 (v2 — 优化版)

改进：
- 更清晰的提取指令，减少 schema 错误
- 覆盖更多摘要类型（实验研究、综述、生物信息学、临床）
- 提供只有实体没有关系的示例
- 关系属性格式明确，减少 "null" 字符串
"""

import langextract as lx

# ── 提取提示词 (v2) ──

KG_EXTRACTION_PROMPT = """\
Extract biomedical entities from the PubMed abstract. Output in the specified schema.

Entity classes (choose exactly one):
- gene: gene symbols like TP53, SLC7A11 (uppercase, no spaces)
- disease: diseases like "hepatocellular carcinoma", "MASLD", "fibrosis"
- protein: proteins like "p53", "collagen", "ALT"
- pathway: pathways/processes like "ferroptosis", "apoptosis"
- metabolite: small molecules like "glucose", "triglycerides"
- tissue: anatomical tissues/organs only, like "liver", "hepatic tissue", "kidney"
- cell_type: specific cell types, like "hepatocyte", "Kupffer cell", "T cell"

Do NOT extract methods, statistics, databases, experimental steps, generic category
words, contextual microenvironment phrases, or unspecified treatment/drug/immune
classes as entities.  A Tissue must be anatomical.  A CellType must name a
specific cell type.  A Pathway must be a specific pathway, signaling axis, or
explicitly discussed mechanism.  Disease must not be guessed from a symptom,
injury, phenotype, or generic pathological process.

Conditional processes such as inflammation, oxidative stress, angiogenesis,
metastasis, fibrosis, apoptosis, proliferation, and immune activation may be
entities only when the source explicitly discusses the process as a mechanism
and gives it a grounded span.  Do not globally discard a specific supported
mechanism such as ferroptosis, TNF signaling pathway, or AREG-EGFR signaling.

For each entity, include these attributes:
  gene: gene_symbol, species, normalized_id
  disease: disease_name, disease_stage, normalized_id
  protein: protein_name, species
  pathway: pathway_name
  metabolite: metabolite_name
  tissue: tissue_name
  cell_type: cell_type_name

Relationship attributes (attach to the SUBJECT entity):
  associated_with: for gene-disease, protein-disease, etc.
  encodes: for gene-protein
  participates_in: for gene/protein-pathway
  interacts_with: for protein-protein
  expressed_in: for gene/protein-tissue/cell_type
  prognostic_in: for gene/protein-disease (prognostic marker)
  progresses_to: for disease-disease

Each relationship entry MUST be a dict with these fields:
  {"target_entity": "...", "target_type": "...", "direction": "increase/decrease/unknown", "negated": false, "uncertain": false, "evidence": "quote from text"}

Important rules:
1. Every entity mention and relationship evidence must be an EXACT continuous
   span from the source. Do not paraphrase evidence.
2. Assign Gene versus Protein only from the source wording. Never convert one
   into the other from background knowledge or name similarity.
3. Only extract what the current article directly supports. Do not infer a fact
   from title co-occurrence, background, research aims, database screening,
   molecular docking, network pharmacology, enrichment, predictions, or hedging.
4. If no relationships are found, omit the relationship attributes — DO NOT output empty lists or "null"
5. For review articles without experimental data, still extract entity mentions
6. If the text describes findings in non-human species, mark species accordingly
7. Output one canonical entity for an explicitly defined long form (ABBR),
   preferring the long form. Relationship endpoints must use that canonical form.
8. Do not split a phrase such as AREG-EGFR signaling into extra entities or
   relations unless the source explicitly states those separate facts.
9. A relation must be supported by evidence containing both endpoints (or their
   explicitly defined article-local abbreviations) and a predicate/direction cue.
"""

# ═══════════════════════════════════════════════════
# 示例 1: 基因-疾病关联 (含关系)
# ═══════════════════════════════════════════════════

EXAMPLE_GENE_DISEASE = lx.data.ExampleData(
    text="TP53 mutations are strongly associated with hepatocellular carcinoma progression. "
         "The TP53 gene encodes the p53 tumor suppressor protein.",
    extractions=[
        lx.data.Extraction(
            extraction_class="gene",
            extraction_text="TP53",
            attributes={
                "gene_symbol": "TP53",
                "species": "Homo sapiens",
                "normalized_id": "HGNC:11998",
                "associated_with": [
                    {
                        "target_entity": "hepatocellular carcinoma",
                        "target_type": "disease",
                        "direction": "increase",
                        "negated": False,
                        "uncertain": False,
                        "disease_stage": "HCC",
                        "evidence": "TP53 mutations are strongly associated with hepatocellular carcinoma progression.",
                    }
                ],
                "encodes": [
                    {
                        "target_entity": "p53", "target_type": "protein",
                        "direction": "none", "negated": False, "uncertain": False,
                        "evidence": "The TP53 gene encodes the p53 tumor suppressor protein.",
                    }
                ],
            },
        ),
        lx.data.Extraction(
            extraction_class="disease",
            extraction_text="hepatocellular carcinoma",
            attributes={
                "disease_name": "hepatocellular carcinoma",
                "disease_stage": "HCC",
                "normalized_id": "UMLS:C2239176",
            },
        ),
        lx.data.Extraction(
            extraction_class="protein",
            extraction_text="p53",
            attributes={
                "protein_name": "p53",
                "species": "Homo sapiens",
            },
        ),
    ],
)

# ═══════════════════════════════════════════════════
# 示例 2: 代谢通路研究 (鼠模型, 含多种关系类型)
# ═══════════════════════════════════════════════════

EXAMPLE_METABOLIC_PATHWAY = lx.data.ExampleData(
    text="SLC7A11 deficiency accelerated MASLD progression via ferroptosis in mice. "
         "Nrf2 activation suppressed this effect.",
    extractions=[
        lx.data.Extraction(
            extraction_class="gene",
            extraction_text="SLC7A11",
            attributes={
                "gene_symbol": "SLC7A11",
                "species": "Mus musculus",
                "normalized_id": "HGNC:10916",
                "associated_with": [
                    {
                        "target_entity": "MASLD",
                        "target_type": "disease",
                        "direction": "increase",
                        "negated": False,
                        "uncertain": False,
                        "disease_stage": "progression",
                        "evidence": "SLC7A11 deficiency accelerated MASLD progression via ferroptosis in mice.",
                    }
                ],
                "participates_in": [
                    {
                        "target_entity": "ferroptosis", "target_type": "pathway",
                        "direction": "none", "negated": False, "uncertain": False,
                        "evidence": "SLC7A11 deficiency accelerated MASLD progression via ferroptosis in mice.",
                    }
                ],
            },
        ),
        lx.data.Extraction(
            extraction_class="disease",
            extraction_text="MASLD",
            attributes={
                "disease_name": "MASLD",
                "disease_stage": "MASLD",
            },
        ),
        lx.data.Extraction(
            extraction_class="pathway",
            extraction_text="ferroptosis",
            attributes={
                "pathway_name": "ferroptosis",
            },
        ),
        lx.data.Extraction(
            extraction_class="gene",
            extraction_text="Nrf2",
            attributes={
                "gene_symbol": "Nrf2",
                "species": "Mus musculus",
            },
        ),
    ],
)

# ═══════════════════════════════════════════════════
# 示例 3: 表达定位 (基因-组织/细胞, 无需疾病)
# ═══════════════════════════════════════════════════

EXAMPLE_EXPRESSION = lx.data.ExampleData(
    text="CYP2E1 is highly expressed in hepatocytes and contributes to oxidative stress "
         "in alcoholic liver disease.",
    extractions=[
        lx.data.Extraction(
            extraction_class="gene",
            extraction_text="CYP2E1",
            attributes={
                "gene_symbol": "CYP2E1",
                "species": "Homo sapiens",
                "expressed_in": [
                    {
                        "target_entity": "hepatocytes", "target_type": "cell_type",
                        "direction": "increase", "negated": False, "uncertain": False,
                        "evidence": "CYP2E1 is highly expressed in hepatocytes and contributes to oxidative stress in alcoholic liver disease.",
                    }
                ],
                "associated_with": [
                    {
                        "target_entity": "alcoholic liver disease",
                        "target_type": "disease",
                        "direction": "increase",
                        "negated": False,
                        "uncertain": False,
                        "disease_stage": "ALD",
                        "evidence": "CYP2E1 is highly expressed in hepatocytes and contributes to oxidative stress in alcoholic liver disease.",
                    }
                ],
            },
        ),
        lx.data.Extraction(
            extraction_class="cell_type",
            extraction_text="hepatocytes",
            attributes={
                "cell_type_name": "hepatocytes",
            },
        ),
        lx.data.Extraction(
            extraction_class="disease",
            extraction_text="alcoholic liver disease",
            attributes={
                "disease_name": "alcoholic liver disease",
                "disease_stage": "ALD",
            },
        ),
    ],
)

# ═══════════════════════════════════════════════════
# 示例 4: 蛋白质-蛋白质互作
# ═══════════════════════════════════════════════════

EXAMPLE_PROTEIN_INTERACTION = lx.data.ExampleData(
    text="p38 MAPK interacts with NF-kB to promote inflammatory cytokine release in NASH.",
    extractions=[
        lx.data.Extraction(
            extraction_class="protein",
            extraction_text="p38 MAPK",
            attributes={
                "protein_name": "p38 MAPK",
                "species": "Homo sapiens",
                "interacts_with": [
                    {
                        "target_entity": "NF-kB", "target_type": "protein",
                        "direction": "none", "negated": False, "uncertain": False,
                        "evidence": "p38 MAPK interacts with NF-kB to promote inflammatory cytokine release in NASH.",
                    }
                ],
            },
        ),
        lx.data.Extraction(
            extraction_class="protein",
            extraction_text="NF-kB",
            attributes={
                "protein_name": "NF-kB",
                "species": "Homo sapiens",
            },
        ),
        lx.data.Extraction(
            extraction_class="disease",
            extraction_text="NASH",
            attributes={
                "disease_name": "NASH",
                "disease_stage": "NASH",
            },
        ),
    ],
)

# ═══════════════════════════════════════════════════
# 示例 5: 综述/观察类 — 无明确关系，仅实体
# ═══════════════════════════════════════════════════

EXAMPLE_REVIEW_NO_RELS = lx.data.ExampleData(
    text="Non-alcoholic fatty liver disease (NAFLD) is a major cause of chronic liver disease "
         "worldwide. The gut-liver axis plays a critical role in NAFLD pathogenesis through "
         "microbial metabolites and bile acid signaling.",
    extractions=[
        lx.data.Extraction(
            extraction_class="disease",
            extraction_text="Non-alcoholic fatty liver disease",
            attributes={
                "disease_name": "Non-alcoholic fatty liver disease",
                "disease_stage": "NAFLD",
            },
        ),
        lx.data.Extraction(
            extraction_class="disease",
            extraction_text="chronic liver disease",
            attributes={
                "disease_name": "chronic liver disease",
            },
        ),
        lx.data.Extraction(
            extraction_class="pathway",
            extraction_text="gut-liver axis",
            attributes={
                "pathway_name": "gut-liver axis",
            },
        ),
        lx.data.Extraction(
            extraction_class="metabolite",
            extraction_text="microbial metabolites",
            attributes={
                "metabolite_name": "microbial metabolites",
            },
        ),
        lx.data.Extraction(
            extraction_class="metabolite",
            extraction_text="bile acid",
            attributes={
                "metabolite_name": "bile acid",
            },
        ),
    ],
)

# ═══════════════════════════════════════════════════
# 示例 6: 生物信息学/网络药理学 (多基因、多靶点)
# ═══════════════════════════════════════════════════

EXAMPLE_BIOINFORMATICS = lx.data.ExampleData(
    text="Network pharmacology identified AKT1, TP53, and TNF as key hub genes in "
         "hepatitis B virus-related HCC. Molecular docking confirmed binding of quercetin "
         "to AKT1 and TNF.",
    extractions=[
        lx.data.Extraction(
            extraction_class="gene",
            extraction_text="AKT1",
            attributes={
                "gene_symbol": "AKT1",
                "species": "Homo sapiens",
            },
        ),
        lx.data.Extraction(
            extraction_class="gene",
            extraction_text="TP53",
            attributes={
                "gene_symbol": "TP53",
                "species": "Homo sapiens",
                "normalized_id": "HGNC:11998",
            },
        ),
        lx.data.Extraction(
            extraction_class="gene",
            extraction_text="TNF",
            attributes={
                "gene_symbol": "TNF",
                "species": "Homo sapiens",
            },
        ),
        lx.data.Extraction(
            extraction_class="disease",
            extraction_text="hepatitis B virus-related HCC",
            attributes={
                "disease_name": "hepatitis B virus-related HCC",
                "disease_stage": "HCC",
            },
        ),
    ],
)

# ═══════════════════════════════════════════════════
# Gold 示例 7: 人体证据中的互作 + 细胞定位
# 来源于项目 50 篇人工金标；只保留一个可独立判定的原文证据单元。
# ═══════════════════════════════════════════════════

EXAMPLE_GOLD_OTUD5 = lx.data.ExampleData(
    text=(
        "This study focused on the overexpression of OTUD5 and its interaction "
        "with MAVS within macrophage subset 11 in patients with primary biliary "
        "cholangitis (PBC)."
    ),
    extractions=[
        lx.data.Extraction(
            extraction_class="gene",
            extraction_text="OTUD5",
            attributes={
                "gene_symbol": "OTUD5",
                "species": "Homo sapiens",
                "interacts_with": [{
                    "target_entity": "MAVS", "target_type": "protein",
                    "direction": "none", "negated": False, "uncertain": False,
                    "evidence": (
                        "This study focused on the overexpression of OTUD5 and its "
                        "interaction with MAVS within macrophage subset 11 in patients "
                        "with primary biliary cholangitis (PBC)."
                    ),
                }],
                "expressed_in": [{
                    "target_entity": "macrophage subset 11", "target_type": "cell_type",
                    "direction": "increase", "negated": False, "uncertain": False,
                    "evidence": (
                        "This study focused on the overexpression of OTUD5 and its "
                        "interaction with MAVS within macrophage subset 11 in patients "
                        "with primary biliary cholangitis (PBC)."
                    ),
                }],
            },
        ),
        lx.data.Extraction(
            extraction_class="protein", extraction_text="MAVS",
            attributes={
                "protein_name": "MAVS", "species": "Homo sapiens",
                "expressed_in": [{
                    "target_entity": "macrophage subset 11", "target_type": "cell_type",
                    "direction": "increase", "negated": False, "uncertain": False,
                    "evidence": (
                        "This study focused on the overexpression of OTUD5 and its "
                        "interaction with MAVS within macrophage subset 11 in patients "
                        "with primary biliary cholangitis (PBC)."
                    ),
                }],
            },
        ),
        lx.data.Extraction(
            extraction_class="cell_type", extraction_text="macrophage subset 11",
            attributes={"cell_type_name": "macrophage subset 11"},
        ),
        lx.data.Extraction(
            extraction_class="disease", extraction_text="primary biliary cholangitis",
            attributes={"disease_name": "primary biliary cholangitis"},
        ),
    ],
)

# ═══════════════════════════════════════════════════
# Gold 示例 8: 临床同一证据中的一对多疾病关联
# ═══════════════════════════════════════════════════

EXAMPLE_GOLD_CLINICAL_MULTI = lx.data.ExampleData(
    text="PN is associated with an increased risk of NAFLD, liver fibrosis, and cirrhosis.",
    extractions=[
        lx.data.Extraction(
            extraction_class="disease", extraction_text="PN",
            attributes={
                "disease_name": "PN",
                "associated_with": [
                    {
                        "target_entity": target, "target_type": "disease",
                        "direction": "increase", "negated": False, "uncertain": False,
                        "evidence": (
                            "PN is associated with an increased risk of NAFLD, liver "
                            "fibrosis, and cirrhosis."
                        ),
                    }
                    for target in ("NAFLD", "liver fibrosis", "cirrhosis")
                ],
            },
        ),
        lx.data.Extraction(
            extraction_class="disease", extraction_text="NAFLD",
            attributes={"disease_name": "NAFLD"},
        ),
        lx.data.Extraction(
            extraction_class="disease", extraction_text="liver fibrosis",
            attributes={"disease_name": "liver fibrosis"},
        ),
        lx.data.Extraction(
            extraction_class="disease", extraction_text="cirrhosis",
            attributes={"disease_name": "cirrhosis"},
        ),
    ],
)

# ── 所有示例 ──

ALL_EXAMPLES = [
    EXAMPLE_GENE_DISEASE,
    EXAMPLE_METABOLIC_PATHWAY,
    EXAMPLE_EXPRESSION,
    EXAMPLE_PROTEIN_INTERACTION,
    EXAMPLE_REVIEW_NO_RELS,
    EXAMPLE_BIOINFORMATICS,
    EXAMPLE_GOLD_OTUD5,
    EXAMPLE_GOLD_CLINICAL_MULTI,
]

# ── 默认示例集 (v2 — 从 2 个扩展到 4 个核心示例) ──

DEFAULT_EXAMPLES = [
    EXAMPLE_GENE_DISEASE,
    EXAMPLE_METABOLIC_PATHWAY,
    EXAMPLE_EXPRESSION,
    EXAMPLE_REVIEW_NO_RELS,
]
