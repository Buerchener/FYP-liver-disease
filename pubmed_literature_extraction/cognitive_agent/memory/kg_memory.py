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

import re
import threading
import unicodedata
from typing import Optional
from urllib.parse import urlparse
from neo4j import GraphDatabase, Driver
from cognitive_agent.schema.entity_classes import ENTITY_CLASSES
from cognitive_agent.schema.write_contract import (
    MAIN_KG_WRITE_CONTRACT,
    NEO4J_IMPORTABLE_PREDICATES,
    is_main_kg_write_signature,
)


ALLOWED_ENTITY_TYPES = frozenset(spec["neo4j_label"] for spec in ENTITY_CLASSES.values())
ALLOWED_RELATION_TYPES = frozenset(NEO4J_IMPORTABLE_PREDICATES)


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

    # These are conditionally valid biological/anatomical concepts.  Their
    # article-level evidence is decided by extraction_quality.py; the storage
    # layer must not silently erase a candidate that already passed that gate.
    GENERIC_TERM_BLACKLIST = GENERIC_TERM_BLACKLIST.difference({
        "inflammation", "injury", "immune response", "immune activation",
        "oxidative stress response", "angiogenesis", "metastasis", "replication",
        "cell proliferation", "cell apoptosis", "cell cycle", "cell death",
        "cell survival", "cell differentiation", "cell growth", "cell senescence",
        "epithelial-mesenchymal transition", "emt", "liver fibrosis",
        "hepatic fibrosis", "liver steatosis", "hepatic steatosis",
        "chronic liver disease", "liver failure", "liver", "blood", "serum", "plasma",
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

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        connection_timeout: float = 10.0,
        allow_isolated_test_writes: bool = False,
    ):
        self.uri = uri
        self.database = database
        self._driver: Optional[Driver] = None
        self._schema_profile_cache: Optional[dict] = None
        self._entity_lookup_cache: dict[tuple[str, str, str], Optional[dict]] = {}
        self._entity_lookup_lock = threading.Lock()
        self._entity_lookup_hits = 0
        self._entity_lookup_misses = 0
        host = urlparse(uri).hostname
        self._local_target = host in {"localhost", "127.0.0.1", "::1"}
        self._write_target_allowed = self._local_target and (
            database == "neo4j" or allow_isolated_test_writes
        )
        if password:
            if not self._local_target:
                raise ValueError("KGMemory connections are restricted to localhost")
            self._driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                connection_timeout=connection_timeout,
                connection_acquisition_timeout=connection_timeout,
            )

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    def get_schema_profile(self, refresh: bool = False) -> dict:
        """Return the actual read schema, cached once per process.

        Queries are built only from labels/properties observed here, avoiding
        Neo4j warnings and false retrieval attempts against an imagined schema.
        """
        if not self._driver:
            return {"labels": {}, "relationship_types": {}, "compatible_entity_types": []}
        if self._schema_profile_cache is not None and not refresh:
            return self._schema_profile_cache
        labels: dict[str, dict] = {}
        relationships: dict[str, int] = {}
        try:
            with self._driver.session(database=self.database) as session:
                for record in session.run(
                    "MATCH (n) UNWIND labels(n) AS label "
                    "RETURN label, count(*) AS count, collect(DISTINCT keys(n)) AS key_sets"
                ):
                    properties = sorted({
                        key for key_set in (record.get("key_sets", []) or [])
                        for key in (key_set or [])
                    })
                    labels[str(record["label"])] = {
                        "count": int(record.get("count", 0) or 0),
                        "properties": properties,
                    }
                for record in session.run(
                    "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count"
                ):
                    relationships[str(record["type"])] = int(record.get("count", 0) or 0)
        except Exception as exc:
            self._schema_profile_cache = {
                "labels": {}, "relationship_types": {}, "compatible_entity_types": [],
                "error": str(exc)[:500],
            }
            return self._schema_profile_cache
        self._schema_profile_cache = {
            "labels": labels,
            "relationship_types": relationships,
            "compatible_entity_types": sorted(set(labels) & ALLOWED_ENTITY_TYPES),
        }
        return self._schema_profile_cache

    def _runtime_entity_fields(self, entity_type: str) -> tuple[list[str], list[str]]:
        profile = self.get_schema_profile()
        label = profile.get("labels", {}).get(entity_type, {})
        existing = set(label.get("properties", []) or [])
        if not existing:
            return [], []
        schema = self.ENTITY_ID_SCHEMA[entity_type]
        name_candidates = [
            schema.get("name_property", ""), "name", "gene_symbol", "disease_name",
            "preferred_name", "pathway_name", "metabolite_name", "tissue_name",
            "cell_type_name",
        ]
        id_candidates = [
            schema.get("id_property", ""), "protein_id", "string_protein_id",
            "normalized_id",
        ]
        names = list(dict.fromkeys(item for item in name_candidates if item in existing))
        identifiers = list(dict.fromkeys(item for item in id_candidates if item in existing))
        return names, identifiers

    @staticmethod
    def _preferred_name(properties: dict | None) -> str:
        properties = properties or {}
        for key in (
            "name", "gene_symbol", "disease_name", "preferred_name",
            "pathway_name", "metabolite_name", "tissue_name", "cell_type_name",
        ):
            value = str(properties.get(key, "") or "").strip()
            if value:
                return value
        return ""

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

        # 未知类型不能进入查询选择路径。
        if entity_type and entity_type not in ALLOWED_ENTITY_TYPES:
            return None
        cache_key = (
            entity_type,
            self.normalize_entity_text(mention),
            str(normalized_id or "").strip(),
        )
        with self._entity_lookup_lock:
            if cache_key in self._entity_lookup_cache:
                self._entity_lookup_hits += 1
                cached = self._entity_lookup_cache[cache_key]
                return dict(cached) if cached is not None else None
        compatible = set(self.get_schema_profile().get("compatible_entity_types", []))
        types_to_try = [entity_type] if entity_type else sorted(compatible)
        types_to_try = [item for item in types_to_try if item in compatible]

        with self._driver.session(database=self.database) as session:
            for etype in types_to_try:
                name_fields, id_fields = self._runtime_entity_fields(etype)
                if not name_fields and not id_fields:
                    continue
                name_checks = [f"toLower(toString(n.`{field}`)) = toLower($mention)" for field in name_fields]
                id_checks = [f"toString(n.`{field}`) = $normalized_id" for field in id_fields]
                checks = [*name_checks, *id_checks]
                name_expr = "coalesce(" + ", ".join(
                    [f"n.`{field}`" for field in name_fields] + ["''"]
                ) + ")"
                id_expr = "coalesce(" + ", ".join(
                    [f"n.`{field}`" for field in id_fields] + ["''"]
                ) + ")"
                cypher = f"""
                    MATCH (n:{etype})
                    WHERE {' OR '.join(checks)}
                    RETURN elementId(n) AS element_id, labels(n) AS labels,
                           {id_expr} AS node_id, {name_expr} AS name
                    LIMIT 1
                """
                try:
                    record = session.run(
                        cypher,
                        {"mention": mention, "normalized_id": normalized_id or ""},
                    ).single()
                    if record:
                        found = {
                            "element_id": record["element_id"],
                            "labels": record["labels"],
                            "node_id": record["node_id"],
                            "name": record["name"],
                        }
                        with self._entity_lookup_lock:
                            self._entity_lookup_cache[cache_key] = found
                            self._entity_lookup_misses += 1
                        return dict(found)
                except Exception:
                    continue
        with self._entity_lookup_lock:
            self._entity_lookup_cache[cache_key] = None
            self._entity_lookup_misses += 1
        return None

    def entity_lookup_cache_stats(self) -> dict:
        """Audit the process-level read-only exact-linking cache."""
        with self._entity_lookup_lock:
            total = self._entity_lookup_hits + self._entity_lookup_misses
            return {
                "entries": len(self._entity_lookup_cache),
                "hits": self._entity_lookup_hits,
                "misses": self._entity_lookup_misses,
                "hit_rate": round(self._entity_lookup_hits / total, 4) if total else 0.0,
            }

    @staticmethod
    def normalize_entity_text(value: str) -> str:
        """Normalize a surface form for deterministic candidate scoring."""
        value = unicodedata.normalize("NFKC", str(value or "")).casefold()
        value = re.sub(r"[-_/]+", " ", value)
        value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
        return " ".join(value.split())

    @classmethod
    def score_entity_candidate(cls, mention: str, candidate_name: str) -> tuple[float, str]:
        """Score a name without treating short arbitrary substrings as matches."""
        query = cls.normalize_entity_text(mention)
        candidate = cls.normalize_entity_text(candidate_name)
        if not query or not candidate:
            return 0.0, "empty"
        if query == candidate:
            return 1.0, "exact_normalized"
        query_tokens = query.split()
        candidate_tokens = candidate.split()
        if len(query_tokens) == 1 and len(query) < 4:
            return 0.0, "short_mention"
        overlap = len(set(query_tokens) & set(candidate_tokens))
        max_tokens = max(len(set(query_tokens)), len(set(candidate_tokens)))
        token_coverage = overlap / max_tokens if max_tokens else 0.0
        # A single generic token is not a synonym for a specific multi-token
        # concept (for example HBV or fibrosis vs HBV-related liver fibrosis).
        if overlap >= 2 and token_coverage >= 0.8:
            return round(0.78 + 0.1 * token_coverage, 3), "token_overlap"
        if min(len(query_tokens), len(candidate_tokens)) == 1 and max(
            len(query_tokens), len(candidate_tokens)
        ) > 1:
            return 0.0, "generic_token_only"
        if query in candidate or candidate in query:
            ratio = min(len(query), len(candidate)) / max(len(query), len(candidate))
            return round(0.55 + 0.25 * ratio, 3), "contains"
        return 0.0, "none"

    def find_entity_fuzzy(self, mention: str, entity_type: str = "") -> Optional[dict]:
        """Return ranked fuzzy candidates; never silently choose the first row."""
        if not self._driver:
            return None
        compatible = set(self.get_schema_profile().get("compatible_entity_types", []))
        types_to_try = [entity_type] if entity_type else sorted(compatible)
        types_to_try = [etype for etype in types_to_try if etype in compatible]
        if entity_type and not types_to_try:
            return None

        candidates = []
        with self._driver.session(database=self.database) as session:
            for etype in types_to_try:
                try:
                    name_fields, id_fields = self._runtime_entity_fields(etype)
                    if not name_fields:
                        continue
                    name_expr = "coalesce(" + ", ".join(
                        [f"n.`{field}`" for field in name_fields] + ["''"]
                    ) + ")"
                    id_expr = "coalesce(" + ", ".join(
                        [f"n.`{field}`" for field in id_fields] + ["''"]
                    ) + ")"
                    result = session.run(
                        f"""
                        MATCH (n:{etype})
                        RETURN elementId(n) AS element_id, labels(n) AS labels,
                               {id_expr} AS node_id,
                               {name_expr} AS name
                        LIMIT 100
                        """,
                    )
                    for record in result:
                        score, match_kind = self.score_entity_candidate(
                            mention, record.get("name", "")
                        )
                        if score:
                            candidates.append({
                                "element_id": record["element_id"],
                                "labels": record["labels"],
                                "node_id": record["node_id"],
                                "name": record["name"],
                                "score": score,
                                "match_kind": match_kind,
                                "entity_type": etype,
                            })
                except Exception:
                    continue
        if not candidates:
            return None
        candidates.sort(key=lambda c: (-c["score"], c["name"] or ""))
        top = candidates[:5]
        return {
            **top[0],
            "fuzzy_matches": top,
            "ambiguous": len(top) > 1 and top[0]["score"] - top[1]["score"] < 0.08,
        }

    # ── 大小写不敏感查询 ──────────────────────────────────

    def find_entity_by_name_ci(self, name: str, entity_type: str = "") -> Optional[dict]:
        """大小写不敏感的精确名称匹配（用于去重检查）。

        与 find_entity 不同：此方法仅按名称匹配，不检查 ID。
        用于防止 "Immunomodulation" / "immunomodulation" 这类重复。
        """
        if not self._driver:
            return None

        compatible = set(self.get_schema_profile().get("compatible_entity_types", []))
        types_to_try = [entity_type] if entity_type else sorted(compatible)
        types_to_try = [label for label in types_to_try if label in compatible]

        with self._driver.session(database=self.database) as session:
            for label in types_to_try:
                name_fields, _ = self._runtime_entity_fields(label)
                if not name_fields:
                    continue
                try:
                    result = session.run(
                        f"""
                        MATCH (n:{label})
                        WHERE any(field IN $name_fields
                                  WHERE toLower(toString(n[field])) = toLower($name))
                        RETURN elementId(n) AS element_id
                        LIMIT 1
                        """,
                        {"name": name, "name_fields": name_fields},
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
                           properties(m) AS target_properties
                    UNION
                    MATCH (m)-[r]->(n)
                    WHERE elementId(n) = $element_id
                    RETURN type(r) AS predicate, labels(m) AS target_labels,
                           properties(m) AS target_properties
                    """,
                    {"element_id": entity_element_id},
                )
                return [
                    {
                        "predicate": rec["predicate"],
                        "target_labels": rec["target_labels"],
                        "target_name": self._preferred_name(rec.get("target_properties")),
                    }
                    for rec in result
                ]
            except Exception:
                return []

    def get_rag_entity_context(
        self,
        entity_element_id: str,
        relation_limit: int = 5,
        evidence_limit: int = 500,
    ) -> dict:
        """Return a bounded, read-only one-hop context for controlled RAG.

        This method never writes and deliberately returns only a small set of
        node aliases and relation provenance fields.  It is not used as current
        article evidence.
        """
        if not self._driver or not entity_element_id:
            return {"synonyms": [], "relations": []}
        relation_limit = max(0, min(int(relation_limit), 20))
        evidence_limit = max(0, min(int(evidence_limit), 2000))
        with self._driver.session(database=self.database) as session:
            try:
                result = session.run(
                    """
                    MATCH (n)
                    WHERE elementId(n) = $element_id
                    OPTIONAL MATCH (n)-[r]-(m)
                    RETURN properties(n) AS node_properties,
                           type(r) AS predicate,
                           CASE WHEN r IS NULL THEN ''
                                WHEN elementId(startNode(r)) = elementId(n) THEN 'outgoing'
                                ELSE 'incoming' END AS edge_orientation,
                           properties(r) AS relation_properties,
                           labels(m) AS target_labels,
                           properties(m) AS target_properties
                    LIMIT $row_limit
                    """,
                    {
                        "element_id": entity_element_id,
                        "evidence_limit": evidence_limit,
                        "row_limit": max(relation_limit, 1),
                    },
                )
                synonyms: list[str] = []
                relations: list[dict] = []
                for record in result:
                    node_properties = record.get("node_properties", {}) or {}
                    raw_synonyms = (
                        node_properties.get("synonyms", [])
                        or node_properties.get("aliases", [])
                        or []
                    )
                    if isinstance(raw_synonyms, str):
                        raw_synonyms = [raw_synonyms]
                    for synonym in raw_synonyms:
                        value = str(synonym or "").strip()
                        if value and value not in synonyms:
                            synonyms.append(value)
                    predicate = record.get("predicate")
                    if predicate and len(relations) < relation_limit:
                        rel_props = record.get("relation_properties", {}) or {}
                        source = str(rel_props.get("source", "") or "")
                        source_pmid = str(
                            rel_props.get("source_pmid", rel_props.get("pmid", "")) or ""
                        )
                        if not source_pmid and source.startswith("PubMed:"):
                            source_pmid = source.split(":", 1)[1]
                        relations.append({
                            "predicate": predicate,
                            "target_name": self._preferred_name(
                                record.get("target_properties")
                            ),
                            "target_type": (record.get("target_labels", []) or [""])[0],
                            "edge_orientation": record.get("edge_orientation", "") or "",
                            "direction": str(rel_props.get("direction", "") or ""),
                            "source_pmid": source_pmid,
                            "source_evidence": str(
                                rel_props.get("evidence", "") or ""
                            )[:evidence_limit],
                        })
                return {"synonyms": synonyms, "relations": relations}
            except Exception:
                return {"synonyms": [], "relations": []}

    def check_relation_exists(
        self,
        subject_name: str,
        predicate: str,
        object_name: str,
        subject_type: str = "",
        object_type: str = "",
    ) -> Optional[dict]:
        """检查关系并返回方向属性；类型过滤避免误配同名节点。"""
        if not self._driver or predicate not in ALLOWED_RELATION_TYPES:
            return None
        # Neo4j emits a warning when a statically named relationship type has
        # never existed in the database.  Such a relationship cannot match,
        # so the schema profile is a lossless early exit.
        existing_relationship_types = set(
            self.get_schema_profile().get("relationship_types", {})
        )
        if predicate not in existing_relationship_types:
            return None

        with self._driver.session(database=self.database) as session:
            try:
                type_filters = ""
                params = {"subject": subject_name, "object": object_name}
                if subject_type in ALLOWED_ENTITY_TYPES:
                    type_filters += f" AND ${'subject_type'} IN labels(s)"
                    params["subject_type"] = subject_type
                if object_type in ALLOWED_ENTITY_TYPES:
                    type_filters += f" AND ${'object_type'} IN labels(o)"
                    params["object_type"] = object_type
                result = session.run(
                    f"""
                    MATCH (s)-[r:{predicate}]->(o)
                    WHERE any(field IN $name_fields
                              WHERE toLower(toString(s[field])) = toLower($subject))
                      AND any(field IN $name_fields
                              WHERE toLower(toString(o[field])) = toLower($object))
                      {type_filters}
                    RETURN elementId(r) AS rel_element_id,
                           properties(r) AS relation_properties
                    LIMIT 1
                    """,
                    {
                        **params,
                        "name_fields": [
                            "name", "gene_symbol", "disease_name", "preferred_name",
                            "pathway_name", "metabolite_name", "tissue_name",
                            "cell_type_name",
                        ],
                    },
                ).single()
                if result:
                    rel_props = result.get("relation_properties", {}) or {}
                    return {
                        "rel_element_id": result["rel_element_id"],
                        "confidence": rel_props.get("confidence", 0.7),
                        "direction": rel_props.get("direction", ""),
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

    def create_entities_batch(self, entities: list[dict], pmid: str = "") -> dict[tuple[str, str], str]:
        """Batch MERGE entities with one UNWIND transaction per label.

        The returned cache is keyed by ``(entity_type, mention.casefold())`` so
        the decision layer can create relations without one lookup per edge.
        """
        if not self._driver or not self._write_target_allowed or not entities:
            return {}
        import hashlib
        import time

        grouped: dict[str, list[dict]] = {}
        for item in entities:
            label = item.get("type", "")
            name = str(item.get("mention", "")).strip()
            if label not in ALLOWED_ENTITY_TYPES or not name:
                continue
            if name.casefold() in self.GENERIC_TERM_BLACKLIST:
                continue
            schema = self.ENTITY_ID_SCHEMA[label]
            props_in = item.get("attributes", item.get("properties", {})) or {}
            external_id = self._extract_external_id(label, props_in)
            id_value = external_id or f"{schema['id_prefix']}:{hashlib.sha256(f'{label}:{name}'.encode()).hexdigest()[:12]}"
            now = time.time()
            props = {
                schema["id_property"]: id_value,
                schema["name_property"]: name,
                "name": name,
                "source": f"PubMed:{pmid}" if pmid else "LLM_extraction",
                "confidence": float(item.get("confidence", 0.7) or 0.7),
                "created_at": now,
            }
            for key, value in props_in.items():
                if key not in self.RELATION_ATTRIBUTE_KEYS and not isinstance(value, (list, dict)) and key not in {"mention", "type", "normalized_id"}:
                    props.setdefault(key, value)
            grouped.setdefault(label, []).append({"id_value": id_value, "props": props, "mention": name, "confidence": props["confidence"]})

        cache: dict[tuple[str, str], str] = {}
        with self._driver.session(database=self.database) as session:
            for label, rows in grouped.items():
                schema = self.ENTITY_ID_SCHEMA[label]
                query = f"""
                UNWIND $rows AS row
                MERGE (n:{label} {{{schema['id_property']}: row.id_value}})
                ON CREATE SET n = row.props
                ON MATCH SET n.confidence = CASE WHEN row.confidence > coalesce(n.confidence, 0) THEN row.confidence ELSE n.confidence END,
                              n.updated_at = row.props.created_at
                RETURN elementId(n) AS element_id, n.{schema['id_property']} AS id_value
                """
                for record in session.run(query, {"rows": rows}):
                    match = next((row for row in rows if row["id_value"] == record["id_value"]), None)
                    if match:
                        cache[(label, match["mention"].casefold())] = record["element_id"]
        return cache

    def create_relations_batch(self, relations: list[dict], pmid: str = "") -> list[str]:
        """Batch idempotent relation MERGE using UNWIND; returns created IDs."""
        if not self._driver or not self._write_target_allowed or not relations:
            return []
        import hashlib
        import time
        grouped: dict[tuple[str, str], list[dict]] = {}
        for rel in relations:
            predicate = str(rel.get("predicate", "") or "").upper()
            subject_type = str(rel.get("subject_type", "") or "")
            object_type = str(rel.get("object_type", "") or "")
            if (
                predicate not in ALLOWED_RELATION_TYPES
                or not rel.get("subject_element_id")
                or not rel.get("object_element_id")
                or not is_main_kg_write_signature(
                    predicate, subject_type, object_type
                )
            ):
                continue
            relation_id_property = MAIN_KG_WRITE_CONTRACT[
                (predicate, subject_type, object_type)
            ]
            relation_key = f"{predicate}|{rel.get('subject_element_id')}|{rel.get('object_element_id')}"
            relation_id = f"LLM_{predicate}:{hashlib.sha256(relation_key.encode()).hexdigest()[:20]}"
            row = dict(rel)
            row.update({
                "relation_id": relation_id,
                relation_id_property: relation_id,
                "source": f"PubMed:{pmid}" if pmid else "LLM_extraction",
                "updated_at": time.time(),
            })
            grouped.setdefault((predicate, relation_id_property), []).append(row)
        if not grouped:
            return []
        written: list[str] = []
        with self._driver.session(database=self.database) as session:
            for (predicate, relation_id_property), rows in grouped.items():
                query = f"""
                UNWIND $rows AS row
                MATCH (s) WHERE elementId(s) = row.subject_element_id
                MATCH (o) WHERE elementId(o) = row.object_element_id
                MERGE (s)-[r:{predicate} {{{relation_id_property}: row.{relation_id_property}}}]->(o)
                ON CREATE SET r = row
                ON MATCH SET r.confidence = CASE WHEN row.confidence > coalesce(r.confidence, 0) THEN row.confidence ELSE r.confidence END,
                              r.updated_at = row.updated_at
                RETURN row.{relation_id_property} AS relation_id
                """
                written.extend(record["relation_id"] for record in session.run(query, {"rows": rows}))
        return written

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
        if not self._driver or not self._write_target_allowed:
            return None

        import hashlib
        import time

        # ── 质量检查 0: 过滤通用术语 ──
        name_lower = name.lower().strip()
        if name_lower in self.GENERIC_TERM_BLACKLIST:
            return None  # 静默拒绝，不创建通用实体

        if entity_type not in ALLOWED_ENTITY_TYPES:
            return None

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
        subject_type: str = "",
        object_type: str = "",
    ) -> Optional[str]:
        """在 Neo4j 中创建新关系"""
        predicate = str(predicate or "").upper()
        if (
            not self._driver
            or not self._write_target_allowed
            or predicate not in ALLOWED_RELATION_TYPES
            or not is_main_kg_write_signature(
                predicate, subject_type, object_type
            )
        ):
            return None

        import hashlib
        import time

        # 生成稳定的 relation_id；不包含 PMID/evidence，便于跨文章聚合同一条关系。
        id_payload = f"{predicate}|{subject_element_id}|{object_element_id}"
        relation_id = f"LLM_{predicate}:{hashlib.sha256(id_payload.encode()).hexdigest()[:20]}"
        relation_id_property = MAIN_KG_WRITE_CONTRACT[
            (predicate, subject_type, object_type)
        ]

        props = {
            "relation_id": relation_id,
            relation_id_property: relation_id,
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
                    MERGE (s)-[r:{predicate} {{{relation_id_property}: $relation_id}}]->(o)
                    ON CREATE SET r = $props
                    ON MATCH SET
                        r.confidence = CASE
                            WHEN $confidence > coalesce(r.confidence, 0)
                            THEN $confidence ELSE r.confidence
                        END,
                        r.evidence = CASE
                            WHEN $evidence = '' OR r.evidence CONTAINS $evidence
                            THEN coalesce(r.evidence, '')
                            WHEN coalesce(r.evidence, '') = ''
                            THEN $evidence
                            ELSE r.evidence + ' | ' + $evidence
                        END,
                        r.updated_at = $updated_at,
                        r.updated_by = $pmid
                    RETURN elementId(r) AS rel_element_id
                    """,
                    {
                        "subj_id": subject_element_id,
                        "obj_id": object_element_id,
                        "relation_id": relation_id,
                        "props": props,
                        "confidence": confidence,
                        "evidence": evidence or "",
                        "updated_at": time.time(),
                        "pmid": f"PubMed:{pmid}" if pmid else "LLM_agent",
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
        if not self._driver or not self._write_target_allowed:
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
        if not self._driver or not self._write_target_allowed:
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
