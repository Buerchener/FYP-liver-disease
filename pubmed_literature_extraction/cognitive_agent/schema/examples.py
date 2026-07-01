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
- tissue: tissues/organs like "liver", "tumor microenvironment"
- cell_type: cell types like "hepatocyte", "immune cell", "T cell"

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
1. Use EXACT text from source — do not paraphrase
2. Gene symbols in UPPERCASE
3. Only extract what the text directly supports
4. If no relationships are found, omit the relationship attributes — DO NOT output empty lists or "null"
5. For review articles without experimental data, still extract entity mentions
6. If the text describes findings in non-human species, mark species accordingly
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
                    {"target_entity": "p53", "target_type": "protein"}
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
                    {"target_entity": "ferroptosis", "target_type": "pathway"}
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
                "associated_with": [
                    {
                        "target_entity": "MASLD",
                        "target_type": "disease",
                        "direction": "decrease",
                        "negated": False,
                        "uncertain": True,
                        "disease_stage": "progression",
                        "evidence": "Nrf2 activation suppressed this effect.",
                    }
                ],
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
                    {"target_entity": "hepatocytes", "target_type": "cell_type"}
                ],
                "associated_with": [
                    {
                        "target_entity": "alcoholic liver disease",
                        "target_type": "disease",
                        "direction": "increase",
                        "negated": False,
                        "uncertain": False,
                        "disease_stage": "ALD",
                        "evidence": "CYP2E1 contributes to oxidative stress in alcoholic liver disease.",
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
                    {"target_entity": "NF-kB", "target_type": "protein"}
                ],
                "participates_in": [
                    {"target_entity": "inflammatory cytokine release", "target_type": "pathway"}
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
            extraction_text="NAFLD",
            attributes={
                "disease_name": "NAFLD",
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
                "associated_with": [
                    {
                        "target_entity": "hepatitis B virus-related HCC",
                        "target_type": "disease",
                        "direction": "increase",
                        "negated": False,
                        "uncertain": False,
                        "disease_stage": "HCC",
                        "evidence": "Network pharmacology identified AKT1 as key hub gene in HBV-related HCC.",
                    }
                ],
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

# ── 所有示例 ──

ALL_EXAMPLES = [
    EXAMPLE_GENE_DISEASE,
    EXAMPLE_METABOLIC_PATHWAY,
    EXAMPLE_EXPRESSION,
    EXAMPLE_PROTEIN_INTERACTION,
    EXAMPLE_REVIEW_NO_RELS,
    EXAMPLE_BIOINFORMATICS,
]

# ── 默认示例集 (v2 — 从 2 个扩展到 4 个核心示例) ──

DEFAULT_EXAMPLES = [
    EXAMPLE_GENE_DISEASE,
    EXAMPLE_METABOLIC_PATHWAY,
    EXAMPLE_EXPRESSION,
    EXAMPLE_REVIEW_NO_RELS,
]
