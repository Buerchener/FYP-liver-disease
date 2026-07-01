#!/usr/bin/env python3
"""
cognitive_agent/memory/kg_memory.py — Neo4j 知识图谱记忆体

Agent 的外部动态记忆层：
- 实体查询（精确 + 模糊）
- 关系查询（已有关系检索）
- 知识写入（创建实体/关系/争议标记）
- 索引管理
"""

from __future__ import annotations

from typing import Optional
from neo4j import GraphDatabase, Driver


class KGMemory:
    """Neo4j 知识图谱记忆体 — Agent 的外部动态记忆"""

    # ── 关系属性键名（用于关系提取，不能存入 Neo4j 节点） ──
    RELATION_ATTRIBUTE_KEYS: frozenset[str] = frozenset({
        "associated_with", "encodes", "participates_in",
        "interacts_with", "expressed_in", "prognostic_in",
        "progresses_to", "associated_with_metabolite",
    })

    # ── 通用/模糊术语黑名单（过于泛化，不应作为独立实体） ──
    # 合并了原有生物学术语 + 方法论噪声 + LangExtract 提取伪影
    GENERIC_TERM_BLACKLIST: frozenset[str] = frozenset({
        # ── 原有生物医学泛化词 ──
        "cancer", "tumor", "tumour", "tumors", "tumours",
        "malignancies", "malignancy", "cells", "cell",
        "patients", "patient", "controls", "control",
        "disease", "diseases", "disorder", "disorders",
        "syndrome", "injury", "damage", "lesion", "lesions",
        "inflammation", "stress", "infection", "infections",
        "humans", "human", "mice", "mouse", "rat", "rats",
        "model", "models", "study", "studies", "analysis",
        "expression", "level", "levels", "activity",
        "response", "responses", "effect", "effects",
        "factor", "factors", "mechanism", "mechanisms",
        "pathway", "pathways", "function", "functions",
        "role", "roles", "process", "processes",
        "treatment", "therapy", "therapies",
        "sample", "samples", "data", "result", "results",
        "survival", "prognosis", "outcome", "outcomes",
        "group", "groups", "cohort",
        "liver", "blood", "serum", "plasma", "tissue",
        "normal", "healthy", "control group",
        "in vivo", "in vitro", "clinical",

        # ── 方法论噪声（实验室技术） ──
        "network pharmacology", "molecular docking", "molecular dynamics",
        "gene ontology", "go analysis", "kegg", "kegg pathway",
        "kegg signaling pathway", "gene set enrichment analysis", "gsea",
        "ingenuity pathway analysis", "ipa",
        "string database", "geo database", "tcga", "gtex", "david", "metascape",
        "cytoscape", "clusterprofiler", "reactome", "wikipathways", "biocarta",
        "genecards", "disgenet", "omics", "proteomics", "metabolomics",
        "transcriptomics", "genomics", "bioinformatics analysis",
        "computational analysis", "network analysis", "pathway analysis",
        "functional enrichment", "enrichment analysis",
        "differential expression analysis", "principal component analysis",
        "pca", "hierarchical clustering", "kaplan-meier", "cox regression",
        "logistic regression", "systematic review", "meta-analysis",
        "meta analysis", "literature review", "narrative review", "scoping review",

        # ── LangExtract 提取伪影（泛化描述词） ──
        "gene symbols", "gene symbol", "gene expression", "protein expression",
        "gene targets", "drug targets", "therapeutic targets", "molecular targets",
        "disease-related targets", "disease targets", "therapeutic agents",
        "chemicals", "compounds", "molecules", "drugs", "agents", "biomarkers",
        "candidate genes", "target genes", "key genes", "hub genes",
        "differentially expressed genes", "degs",
        "immunomodulation", "immune regulation", "immune modulation",
        "immunosuppression", "immune activation", "immune evasion",
        "immune microenvironment",  # too generic when alone
        "replication",  # experimental term, not biological pathway
        "ingredient-target-pathway network",  # LangExtract artifact
        "ingredient-target network",  # LangExtract artifact
        "clinical outcomes", "clinical parameters", "biochemical parameters",
        "laboratory parameters", "clinical characteristics", "baseline characteristics",
        "demographic characteristics", "patient characteristics",
        "healthy controls", "control group", "study group", "treatment group",
        "placebo group", "experimental group",
        "signaling pathway", "signaling pathways", "signaling cascade",
        "metabolic pathway", "metabolic pathways", "biological pathway",
        "signal transduction", "cell signaling", "cellular signaling",
        "immune response", "inflammatory response", "oxidative stress response",
        "dna damage response", "unfolded protein response", "upstream regulator",
        "downstream effector", "downstream target",
        "cell proliferation", "cell apoptosis", "cell migration", "cell invasion",
        "cell cycle", "cell death", "cell survival", "cell differentiation",
        "cell growth", "cell viability", "cell senescence",
        "angiogenesis", "metastasis",
        "epithelial-mesenchymal transition", "emt",
        "liver disease", "liver diseases", "chronic liver disease",
        "liver injury", "hepatic injury", "liver damage", "hepatic damage",
        "liver dysfunction", "hepatic dysfunction", "liver failure",
        "liver fibrosis", "hepatic fibrosis", "liver steatosis", "hepatic steatosis",
        "inclusion criteria", "exclusion criteria", "eligibility criteria",
        "primary endpoint", "secondary endpoint", "primary outcome",
        "secondary outcome", "adverse events", "side effects", "safety profile",
        "efficacy", "pharmacokinetics", "pharmacodynamics", "bioavailability",
        "dose-response", "dose response", "concentration-dependent", "time-dependent",
        "immunohistochemical staining", "western blot analysis", "elisa assay",
        "flow cytometry", "mass spectrometry", "chromatography",
        "spectroscopy", "microscopy", "histological analysis", "histopathology",

        # ── 统计/研究方法术语 ──
        "p value", "p-value", "confidence interval", "hazard ratio", "odds ratio",
        "risk ratio", "relative risk", "standard deviation", "standard error",
        "area under curve", "receiver operating characteristic",
        "sensitivity", "specificity", "positive predictive value",
        "negative predictive value",
    })

    # ── 实体 ID 规则：每种实体类型的主键和附加属性 ──
    ENTITY_ID_SCHEMA: dict[str, dict] = {
        "Disease": {
            "id_property": "disease_id",
            "id_prefix": "PROJECT:Disease",
            "name_property": "name",
            "extra_properties": ["disease_name"],
        },
        "Gene": {
            "id_property": "gene_id",
            "id_prefix": "NCBIGene",
            "name_property": "gene_symbol",
            "extra_properties": ["gene_symbol", "name"],
        },
        "Protein": {
            "id_property": "string_protein_id",
            "id_prefix": "PROJECT:Protein",
            "name_property": "preferred_name",
            "extra_properties": ["name", "species_name"],
        },
        "Pathway": {
            "id_property": "pathway_id",
            "id_prefix": "PROJECT:Pathway",
            "name_property": "name",
            "extra_properties": [],
        },
        "Metabolite": {
            "id_property": "metabolite_id",
            "id_prefix": "PROJECT:Metabolite",
            "name_property": "name",
            "extra_properties": [],
        },
        "Tissue": {
            "id_property": "tissue_id",
            "id_prefix": "PROJECT:Tissue",
            "name_property": "name",
            "extra_properties": ["tissue_name"],
        },
        "CellType": {
            "id_property": "cell_type_id",
            "id_prefix": "PROJECT:CellType",
            "name_property": "name",
            "extra_properties": ["cell_type_name"],
        },
    }

    def __init__(self, uri: str, user: str, password: str, database: str = "liver-kg-core-v02"):
        self.uri = uri
        self.database = database
        self._driver: Optional[Driver] = None
        if password:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    # ── 实体查询 ──────────────────────────────────────────

    # Cypher 查询模板（每种实体类型）
    _ENTITY_QUERIES: dict[str, str] = {
        "Gene": """
            MATCH (n:Gene)
            WHERE toLower(n.gene_symbol) = toLower($mention)
               OR toLower(n.name) = toLower($mention)
               OR n.gene_id = $normalized_id
            RETURN elementId(n) AS element_id, labels(n) AS labels,
                   n.gene_id AS node_id, n.gene_symbol AS name
            LIMIT 1
        """,
        "Disease": """
            MATCH (n:Disease)
            WHERE toLower(n.name) = toLower($mention)
               OR toLower(n.disease_name) = toLower($mention)
               OR n.disease_id = $normalized_id
            RETURN elementId(n) AS element_id, labels(n) AS labels,
                   n.disease_id AS node_id, coalesce(n.name, n.disease_name) AS name
            LIMIT 1
        """,
        "Protein": """
            MATCH (n:Protein)
            WHERE toLower(n.name) = toLower($mention)
               OR n.protein_id = $normalized_id
               OR n.string_protein_id = $normalized_id
            RETURN elementId(n) AS element_id, labels(n) AS labels,
                   coalesce(n.string_protein_id, n.protein_id) AS node_id, n.name AS name
            LIMIT 1
        """,
        "Pathway": """
            MATCH (n:Pathway)
            WHERE toLower(n.name) = toLower($mention)
               OR n.pathway_id = $normalized_id
            RETURN elementId(n) AS element_id, labels(n) AS labels,
                   n.pathway_id AS node_id, n.name AS name
            LIMIT 1
        """,
        "Metabolite": """
            MATCH (n:Metabolite)
            WHERE toLower(n.name) = toLower($mention)
               OR n.metabolite_id = $normalized_id
            RETURN elementId(n) AS element_id, labels(n) AS labels,
                   n.metabolite_id AS node_id, n.name AS name
            LIMIT 1
        """,
        "Tissue": """
            MATCH (n:Tissue)
            WHERE toLower(n.name) = toLower($mention)
               OR toLower(n.tissue_name) = toLower($mention)
               OR n.tissue_id = $normalized_id
            RETURN elementId(n) AS element_id, labels(n) AS labels,
                   n.tissue_id AS node_id, coalesce(n.name, n.tissue_name) AS name
            LIMIT 1
        """,
        "CellType": """
            MATCH (n:CellType)
            WHERE toLower(n.name) = toLower($mention)
               OR toLower(n.cell_type_name) = toLower($mention)
               OR n.cell_type_id = $normalized_id
            RETURN elementId(n) AS element_id, labels(n) AS labels,
                   n.cell_type_id AS node_id, coalesce(n.name, n.cell_type_name) AS name
            LIMIT 1
        """,
    }

    def find_entity(self, mention: str, entity_type: str = "", normalized_id: str = "") -> Optional[dict]:
        """在 Neo4j 中查找实体，返回 {element_id, labels, node_id, name} 或 None"""
        if not self._driver:
            return None

        # 尝试所有实体类型，或指定类型
        types_to_try = [entity_type] if entity_type else list(self._ENTITY_QUERIES.keys())

        with self._driver.session(database=self.database) as session:
            for etype in types_to_try:
                cypher = self._ENTITY_QUERIES.get(etype)
                if not cypher:
                    continue
                try:
                    record = session.run(
                        cypher,
                        {"mention": mention, "normalized_id": normalized_id or ""},
                    ).single()
                    if record:
                        return {
                            "element_id": record["element_id"],
                            "labels": record["labels"],
                            "node_id": record["node_id"],
                            "name": record["name"],
                        }
                except Exception:
                    continue
        return None

    def find_entity_fuzzy(self, mention: str, entity_type: str = "") -> Optional[dict]:
        """模糊匹配实体（大小写不敏感 + 双向包含匹配）。

        双向匹配逻辑：
        - 节点名 包含 查询词（如 "HCC" 包含在查询词 "Hepatocellular carcinoma"）
        - 查询词 包含 节点名（如查询词 "HCC" 包含在节点名 "Hepatocellular carcinoma"）
        """
        if not self._driver:
            return None

        types_to_try = [entity_type] if entity_type else list(self._ENTITY_QUERIES.keys())
        mention_lower = mention.lower()

        with self._driver.session(database=self.database) as session:
            for etype in types_to_try:
                label = etype
                try:
                    result = session.run(
                        f"""
                        MATCH (n:{label})
                        WHERE toLower(coalesce(n.name, n.gene_symbol, n.disease_name, '')) CONTAINS $mention
                           OR $mention CONTAINS toLower(coalesce(n.name, n.gene_symbol, n.disease_name, ''))
                           OR (n.gene_symbol IS NOT NULL AND toLower(n.gene_symbol) CONTAINS $mention)
                           OR (n.disease_name IS NOT NULL AND toLower(n.disease_name) CONTAINS $mention)
                           OR (n.tissue_name IS NOT NULL AND toLower(n.tissue_name) CONTAINS $mention)
                           OR (n.cell_type_name IS NOT NULL AND toLower(n.cell_type_name) CONTAINS $mention)
                        RETURN elementId(n) AS element_id, labels(n) AS labels,
                               coalesce(n.gene_id, n.disease_id, n.protein_id, n.pathway_id,
                                        n.metabolite_id, n.tissue_id, n.cell_type_id) AS node_id,
                               coalesce(n.name, n.gene_symbol, n.disease_name) AS name
                        LIMIT 3
                        """,
                        {"mention": mention_lower},
                    )
                    records = list(result)
                    if records:
                        return {
                            "element_id": records[0]["element_id"],
                            "labels": records[0]["labels"],
                            "node_id": records[0]["node_id"],
                            "name": records[0]["name"],
                            "fuzzy_matches": [
                                {"name": r["name"], "node_id": r["node_id"]} for r in records
                            ],
                        }
                except Exception:
                    continue
        return None

    # ── 大小写不敏感查询 ──────────────────────────────────

    def find_entity_by_name_ci(self, name: str, entity_type: str = "") -> Optional[dict]:
        """大小写不敏感的精确名称匹配（用于去重检查）。

        与 find_entity 不同：此方法仅按名称匹配，不检查 ID。
        用于防止 "Immunomodulation" / "immunomodulation" 这类重复。
        """
        if not self._driver:
            return None

        types_to_try = [entity_type] if entity_type else list(self._ENTITY_QUERIES.keys())

        with self._driver.session(database=self.database) as session:
            for label in types_to_try:
                try:
                    result = session.run(
                        f"""
                        MATCH (n:{label})
                        WHERE toLower(coalesce(n.name, n.gene_symbol, n.disease_name)) = toLower($name)
                        RETURN elementId(n) AS element_id
                        LIMIT 1
                        """,
                        {"name": name},
                    ).single()
                    if result:
                        return {
                            "element_id": result["element_id"],
                        }
                except Exception:
                    continue
        return None

    # ── 关系查询 ──────────────────────────────────────────

    def get_relations(self, entity_element_id: str) -> list[dict]:
        """获取某个实体的所有已有关系"""
        if not self._driver:
            return []

        with self._driver.session(database=self.database) as session:
            try:
                result = session.run(
                    """
                    MATCH (n)-[r]->(m)
                    WHERE elementId(n) = $element_id
                    RETURN type(r) AS predicate, labels(m) AS target_labels,
                           coalesce(m.name, m.gene_symbol, m.disease_name) AS target_name
                    UNION
                    MATCH (m)-[r]->(n)
                    WHERE elementId(n) = $element_id
                    RETURN type(r) AS predicate, labels(m) AS target_labels,
                           coalesce(m.name, m.gene_symbol, m.disease_name) AS target_name
                    """,
                    {"element_id": entity_element_id},
                )
                return [
                    {
                        "predicate": rec["predicate"],
                        "target_labels": rec["target_labels"],
                        "target_name": rec["target_name"],
                    }
                    for rec in result
                ]
            except Exception:
                return []

    def check_relation_exists(
        self,
        subject_name: str,
        predicate: str,
        object_name: str,
    ) -> Optional[dict]:
        """检查关系是否已存在（不依赖非标准属性键，避免 schema 警告）"""
        if not self._driver:
            return None

        with self._driver.session(database=self.database) as session:
            try:
                result = session.run(
                    f"""
                    MATCH (s)-[r:{predicate}]->(o)
                    WHERE (toLower(coalesce(s.name, s.gene_symbol, s.disease_name)) = toLower($subject)
                           OR toLower(s.gene_symbol) = toLower($subject))
                      AND (toLower(coalesce(o.name, o.gene_symbol, o.disease_name)) = toLower($object)
                           OR toLower(o.gene_symbol) = toLower($object))
                    RETURN elementId(r) AS rel_element_id
                    LIMIT 1
                    """,
                    {"subject": subject_name, "object": object_name},
                ).single()
                if result:
                    return {
                        "rel_element_id": result["rel_element_id"],
                    }
            except Exception:
                pass
        return None

    # ── 骨干疾病同义词映射（避免创建重复疾病节点） ──
    DISEASE_SYNONYMS: dict[str, str] = {
        # NAFLD
        "nafld": "UMLS:C0400966",
        "non-alcoholic fatty liver disease": "UMLS:C0400966",
        "nonalcoholic fatty liver disease": "UMLS:C0400966",
        "non alcoholic fatty liver disease": "UMLS:C0400966",
        "non-alcoholic fatty liver": "UMLS:C0400966",
        "metabolic dysfunction-associated steatotic liver disease": "UMLS:C0400966",
        "masld": "UMLS:C0400966",
        # NASH
        "nash": "UMLS:C3241937",
        "non-alcoholic steatohepatitis": "UMLS:C3241937",
        "nonalcoholic steatohepatitis": "UMLS:C3241937",
        "non alcoholic steatohepatitis": "UMLS:C3241937",
        "metabolic dysfunction-associated steatohepatitis": "UMLS:C3241937",
        "mash": "UMLS:C3241937",
        # Fibrosis
        "liver fibrosis": "UMLS:C0239946",
        "hepatic fibrosis": "UMLS:C0239946",
        # Cirrhosis
        "cirrhosis": "UMLS:C0023890",
        "liver cirrhosis": "UMLS:C0023890",
        "hepatic cirrhosis": "UMLS:C0023890",
        # HCC
        "hcc": "UMLS:C2239176",
        "hepatocellular carcinoma": "UMLS:C2239176",
        "hepatocellular cancer": "UMLS:C2239176",
        "liver cancer": "UMLS:C2239176",
        "primary liver cancer": "UMLS:C2239176",
        "hepatoma": "UMLS:C2239176",
    }

    # ── 知识写入 ──────────────────────────────────────────

    def _extract_external_id(self, entity_type: str, properties: dict) -> Optional[str]:
        """从 properties 中提取外部数据库 ID (如 HGNC:11998, UMLS:C2239176)。"""
        normalized_id = properties.get("normalized_id", "")
        if not normalized_id:
            return None

        # 按实体类型匹配已知前缀
        prefix_map = {
            "Gene": ["HGNC:", "NCBIGene:"],
            "Disease": ["UMLS:", "MESH:", "DO:"],
            "Protein": ["UniProtKB:", "STRING:"],
            "Pathway": ["KEGG:", "Reactome:", "WP:"],
            "Metabolite": ["HMDB", "HMDB:", "CHEBI:"],
        }

        entity_prefixes = prefix_map.get(entity_type, [])
        for prefix in entity_prefixes:
            if normalized_id.startswith(prefix):
                return normalized_id

        # 不匹配已知前缀，仍尝试使用
        if ":" in normalized_id or normalized_id.isdigit():
            return normalized_id
        return None

    def create_entity(
        self,
        entity_type: str,
        name: str,
        properties: dict,
        evidence: str = "",
        pmid: str = "",
        confidence: float = 0.7,
    ) -> Optional[str]:
        """在 Neo4j 中创建或合并实体节点（使用类型特定的主键和属性 Schema）。

        使用 MERGE 语义：
        - 新节点 → 设置完整属性
        - 已存在 → 仅更新 confidence、evidence、updated_at

        Returns:
            element_id 字符串，或写入失败时返回 None
        """
        if not self._driver:
            return None

        import hashlib
        import time

        # ── 质量检查 0: 过滤通用术语 ──
        name_lower = name.lower().strip()
        if name_lower in self.GENERIC_TERM_BLACKLIST:
            return None  # 静默拒绝，不创建通用实体

        label = entity_type
        id_schema = self.ENTITY_ID_SCHEMA.get(label, {
            "id_property": "node_id",
            "id_prefix": "LLM",
            "name_property": "name",
            "extra_properties": [],
        })

        # ── 质量检查 1: 大小写不敏感去重 ──
        existing_ci = self.find_entity_by_name_ci(name, entity_type=label)
        if existing_ci:
            return existing_ci["element_id"]

        # ── 质量检查 2: 骨干疾病同义词匹配（仅 Disease） ──
        if label == "Disease":
            backbone_umls = self.DISEASE_SYNONYMS.get(name_lower)
            if backbone_umls:
                # 查找骨干疾病节点
                backbone_match = self.find_entity(name, entity_type="Disease", normalized_id=backbone_umls)
                if not backbone_match:
                    backbone_match = self.find_entity("", entity_type="Disease", normalized_id=backbone_umls)
                if backbone_match:
                    return backbone_match["element_id"]

        # 1. 尝试使用外部数据库 ID
        external_id = self._extract_external_id(label, properties)

        # 2. 生成类型特定的主键值 (v2: 不依赖 pmid, 同一概念跨文章共享 ID)
        if external_id:
            id_value = external_id
        else:
            id_base = f"{label}:{name}"  # ← v2: 移除 pmid, 确保跨文章去重
            short_hash = hashlib.sha256(id_base.encode()).hexdigest()[:12]
            id_value = f"{id_schema['id_prefix']}:{short_hash}"

        id_property = id_schema["id_property"]

        # 3. 构建类型特定的属性字典
        now = time.time()
        props = {
            id_property: id_value,
            id_schema["name_property"]: name,
            "source": f"PubMed:{pmid}" if pmid else "LLM_extraction",
            "confidence": confidence,
            "evidence": evidence,
            "created_at": now,
        }

        # 4. 确保 "name" 字段存在（Neo4j Browser 默认显示）
        if "name" not in props:
            props["name"] = name

        # 5. 填充类型特定的额外属性
        for extra in id_schema.get("extra_properties", []):
            if extra not in props:
                props[extra] = name

        # 6. 合并调用者传入的属性（跳过已处理的键 + 关系属性键 + 嵌套结构）
        skip_keys = {"normalized_id", "mention", "type"}
        for k, v in properties.items():
            # 过滤关系属性键（associated_with 等无法存入 Neo4j 的嵌套列表）
            if k in self.RELATION_ATTRIBUTE_KEYS:
                continue
            # 过滤 Neo4j 不支持的嵌套结构（list, dict）
            if isinstance(v, (list, dict)):
                continue
            if k not in skip_keys and k not in props:
                props[k] = v

        # 7. 执行 MERGE（遵守 UNIQUE 约束）
        with self._driver.session(database=self.database) as session:
            try:
                result = session.run(
                    f"""
                    MERGE (n:{label} {{{id_property}: $id_value}})
                    ON CREATE SET n = $props
                    ON MATCH SET
                        n.confidence = CASE
                            WHEN $confidence > coalesce(n.confidence, 0)
                            THEN $confidence ELSE n.confidence
                        END,
                        n.evidence = coalesce(n.evidence, '') + ' | ' + $evidence,
                        n.updated_at = $now,
                        n.updated_by = $pmid
                    RETURN elementId(n) AS element_id
                    """,
                    {
                        "id_value": id_value,
                        "props": props,
                        "confidence": confidence,
                        "evidence": evidence or "",
                        "now": now,
                        "pmid": f"PubMed:{pmid}" if pmid else "LLM_agent",
                    },
                ).single()
                if result:
                    return result["element_id"]
            except Exception as e:
                print(f"    [KG] Failed to MERGE entity {label}:{name} — {e}")
                return None
        return None

    def create_relation(
        self,
        subject_element_id: str,
        predicate: str,
        object_element_id: str,
        properties: dict,
        evidence: str = "",
        pmid: str = "",
        confidence: float = 0.7,
    ) -> Optional[str]:
        """在 Neo4j 中创建新关系"""
        if not self._driver:
            return None

        import hashlib
        import time

        # 生成稳定的 relation_id
        id_payload = f"{predicate}|{pmid}|{subject_element_id}|{object_element_id}|{evidence}"
        relation_id = f"LLM_{predicate}:{hashlib.sha256(id_payload.encode()).hexdigest()[:20]}"

        props = {
            "relation_id": relation_id,
            "source": f"PubMed:{pmid}" if pmid else "LLM_extraction",
            "confidence": confidence,
            "evidence": evidence,
            "created_at": time.time(),
            **properties,
        }

        with self._driver.session(database=self.database) as session:
            try:
                result = session.run(
                    f"""
                    MATCH (s) WHERE elementId(s) = $subj_id
                    MATCH (o) WHERE elementId(o) = $obj_id
                    CREATE (s)-[r:{predicate}]->(o)
                    SET r = $props
                    RETURN elementId(r) AS rel_element_id
                    """,
                    {
                        "subj_id": subject_element_id,
                        "obj_id": object_element_id,
                        "props": props,
                    },
                ).single()
                if result:
                    return result["rel_element_id"]
            except Exception as e:
                print(f"    [KG] Failed to create relation {predicate} — {e}")
                return None
        return None

    def update_relation(
        self,
        rel_element_id: str,
        new_evidence: str,
        new_confidence: float,
        pmid: str = "",
    ) -> bool:
        """更新已有关系的证据和置信度"""
        if not self._driver:
            return False

        with self._driver.session(database=self.database) as session:
            try:
                session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE elementId(r) = $rel_id
                    SET r.evidence = coalesce(r.evidence, '') + ' | ' + $new_evidence,
                        r.confidence = CASE
                            WHEN $new_confidence > coalesce(r.confidence, 0)
                            THEN $new_confidence
                            ELSE r.confidence
                        END,
                        r.updated_at = $timestamp,
                        r.updated_by = $pmid
                    """,
                    {
                        "rel_id": rel_element_id,
                        "new_evidence": new_evidence,
                        "new_confidence": new_confidence,
                        "pmid": f"PubMed:{pmid}" if pmid else "LLM_agent",
                        "timestamp": __import__("time").time(),
                    },
                )
                return True
            except Exception:
                return False

    def mark_disputed(
        self,
        rel_element_id: str,
        dispute_reason: str,
        conflicting_evidence: str,
        pmid: str = "",
    ) -> bool:
        """标记关系为学术争议"""
        if not self._driver:
            return False
        import time

        with self._driver.session(database=self.database) as session:
            try:
                session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE elementId(r) = $rel_id
                    SET r.disputed = true,
                        r.dispute_reason = $reason,
                        r.conflicting_evidence = $conflict_evidence,
                        r.disputed_at = $timestamp,
                        r.disputed_by = $pmid
                    """,
                    {
                        "rel_id": rel_element_id,
                        "reason": dispute_reason,
                        "conflict_evidence": conflicting_evidence,
                        "pmid": f"PubMed:{pmid}" if pmid else "LLM_agent",
                        "timestamp": time.time(),
                    },
                )
                return True
            except Exception:
                return False

    def get_node_counts(self) -> dict[str, int]:
        """获取各类实体数量统计"""
        if not self._driver:
            return {}
        counts = {}
        with self._driver.session(database=self.database) as session:
            for label in ["Gene", "Disease", "Protein", "Pathway", "Metabolite", "Tissue", "CellType"]:
                try:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt").single()
                    counts[label] = result["cnt"] if result else 0
                except Exception:
                    counts[label] = -1
        return counts

    def close(self):
        if self._driver:
            self._driver.close()
