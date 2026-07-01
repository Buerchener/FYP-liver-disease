#!/usr/bin/env python3
"""
cognitive_agent/schema/entity_classes.py — KG实体类型定义

定义7种核心实体类型及其属性schema，用于 LangExtract extraction_class 映射。
"""

# ── 实体类型 → LangExtract extraction_class 映射 ──

ENTITY_CLASSES: dict[str, dict] = {
    "gene": {
        "label": "Gene",
        "attributes": ["gene_symbol", "species", "normalized_id"],
        "description": "Gene mentioned in text, e.g. TP53, EGFR, NFE2L2, SLC7A11",
        "neo4j_label": "Gene",
        "neo4j_id_property": "gene_id",
        "neo4j_name_properties": ["gene_symbol", "name"],
    },
    "disease": {
        "label": "Disease",
        "attributes": ["disease_name", "disease_stage", "normalized_id"],
        "description": "Disease or disease stage, e.g. HCC, NAFLD, liver fibrosis, MASLD",
        "neo4j_label": "Disease",
        "neo4j_id_property": "disease_id",
        "neo4j_name_properties": ["name", "disease_name"],
    },
    "protein": {
        "label": "Protein",
        "attributes": ["protein_name", "species", "normalized_id"],
        "description": "Protein, e.g. p53, EGFR protein, MAPK14, collagen",
        "neo4j_label": "Protein",
        "neo4j_id_property": "protein_id",
        "neo4j_name_properties": ["name"],
    },
    "pathway": {
        "label": "Pathway",
        "attributes": ["pathway_name", "source_db"],
        "description": "Biological pathway, e.g. ferroptosis, TNF signaling, apoptosis",
        "neo4j_label": "Pathway",
        "neo4j_id_property": "pathway_id",
        "neo4j_name_properties": ["name"],
    },
    "metabolite": {
        "label": "Metabolite",
        "attributes": ["metabolite_name", "hmdb_id"],
        "description": "Small molecule metabolite, e.g. glucose, ferulic acid, glutathione",
        "neo4j_label": "Metabolite",
        "neo4j_id_property": "metabolite_id",
        "neo4j_name_properties": ["name"],
    },
    "tissue": {
        "label": "Tissue",
        "attributes": ["tissue_name"],
        "description": "Tissue or organ, e.g. liver, hepatic tissue, kidney",
        "neo4j_label": "Tissue",
        "neo4j_id_property": "tissue_id",
        "neo4j_name_properties": ["name", "tissue_name"],
    },
    "cell_type": {
        "label": "CellType",
        "attributes": ["cell_type_name"],
        "description": "Cell type, e.g. hepatocyte, Kupffer cell, T cell, stellate cell",
        "neo4j_label": "CellType",
        "neo4j_id_property": "cell_type_id",
        "neo4j_name_properties": ["name", "cell_type_name"],
    },
}

# ── 提取类名 → Neo4j Label 反向映射 ──
EXTRACTION_CLASS_TO_LABEL = {k: v["neo4j_label"] for k, v in ENTITY_CLASSES.items()}
LABEL_TO_EXTRACTION_CLASS = {v["neo4j_label"]: k for k, v in ENTITY_CLASSES.items()}

# ── 实体创建策略 ──
ENTITY_CREATION_POLICY = {
    "Gene": {"min_confidence": 0.8, "require_external_db": True},
    "Protein": {"min_confidence": 0.8, "require_external_db": True},
    "Disease": {"min_confidence": 0.6, "require_external_db": False},
    "Pathway": {"min_confidence": 0.7, "require_external_db": False},
    "Metabolite": {"min_confidence": 0.7, "require_external_db": True},
    "CellType": {"min_confidence": 0.6, "require_external_db": False},
    "Tissue": {"min_confidence": 0.5, "require_external_db": False},
}
