#!/usr/bin/env python3
"""Import curated DisGeNET liver backbone TSV files into a Neo4j database."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_DATABASE = "liver-kg-core-v02"
DEFAULT_BASE_URL = "http://100.104.181.96:7474"


def read_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def blank_to_none(value: Any) -> Any:
    return None if value == "" else value


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def normalize_diseases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "stage_order": to_int(row.get("stage_order")),
                "name": row.get("stage_code"),
                "external_ids": row.get("disease_vocabularies"),
            }
        )
    return normalized


def normalize_genes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        ncbi_gene_id = row.get("gene_id")
        normalized.append(
            {
                **row,
                "ncbi_gene_id": ncbi_gene_id,
                "gene_id": f"NCBIGene:{ncbi_gene_id}",
                "name": row.get("gene_symbol"),
                "ensembl_gene_ids": row.get("gene_ensembl_ids"),
                "gene_type": row.get("gene_ncbi_type"),
                "gene_dsi": to_float(row.get("gene_dsi")),
                "gene_dpi": to_float(row.get("gene_dpi")),
                "gene_pli": to_float(row.get("gene_pli")),
                "protein_class_names": row.get("gene_protein_class_names"),
            }
        )
    return normalized


def normalize_associations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "stage_order": to_int(row.get("stage_order")),
                "score": to_float(row.get("score")),
                "normalized_score": to_float(row.get("normalized_score")),
                "num_pmids": to_int(row.get("num_pmids")),
                "year_initial": to_int(row.get("year_initial")),
                "year_final": to_int(row.get("year_final")),
                "evidence_index": to_float(row.get("evidence_index")),
                "num_db_snp": to_int(row.get("num_db_snp")),
                "num_clinical_trials": to_int(row.get("num_clinical_trials")),
                "num_chemicals": to_int(row.get("num_chemicals")),
                "num_pmids_with_chemicals": to_int(row.get("num_pmids_with_chemicals")),
                "num_trials_with_chemicals": to_int(row.get("num_trials_with_chemicals")),
                "disgenet_evidence_level": blank_to_none(row.get("disgenet_evidence_level")),
            }
        )
    return normalized


def post_json(base_url: str, database: str, username: str, password: str, statements: list[dict[str, Any]]) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/db/{database}/tx/commit"
    payload = json.dumps({"statements": statements}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    auth = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    request.add_header("Authorization", f"Basic {auth}")
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("errors"):
        raise RuntimeError(json.dumps(result["errors"], indent=2, ensure_ascii=False))
    return result


def chunked(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def query_rows(base_url: str, database: str, username: str, password: str, statement: str) -> list[dict[str, Any]]:
    result = post_json(base_url, database, username, password, [{"statement": statement}])
    columns = result["results"][0]["columns"]
    return [dict(zip(columns, item["row"], strict=True)) for item in result["results"][0]["data"]]


def import_data(
    base_url: str,
    database: str,
    username: str,
    password: str,
    diseases: list[dict[str, Any]],
    genes: list[dict[str, Any]],
    associations: list[dict[str, Any]],
    batch_size: int,
) -> None:
    post_json(
        base_url,
        database,
        username,
        password,
        [
            {"statement": "CREATE CONSTRAINT disease_id_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.disease_id IS UNIQUE"},
            {"statement": "CREATE CONSTRAINT gene_id_unique IF NOT EXISTS FOR (g:Gene) REQUIRE g.gene_id IS UNIQUE"},
            {"statement": "CREATE CONSTRAINT ncbi_gene_id_unique IF NOT EXISTS FOR (g:Gene) REQUIRE g.ncbi_gene_id IS UNIQUE"},
            {
                "statement": (
                    "CREATE CONSTRAINT associated_with_relation_id_unique IF NOT EXISTS "
                    "FOR ()-[r:ASSOCIATED_WITH]-() REQUIRE r.relation_id IS UNIQUE"
                )
            },
        ],
    )

    for batch in chunked(diseases, batch_size):
        post_json(
            base_url,
            database,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MERGE (d:Disease {disease_id: row.disease_id})
SET d.name = row.name,
    d.stage_order = row.stage_order,
    d.disease_name = row.disease_name,
    d.disease_type = row.disease_type,
    d.external_ids = row.external_ids,
    d.disease_classes_msh = row.disease_classes_msh,
    d.source = row.source,
    d.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )

    for batch in chunked(genes, batch_size):
        post_json(
            base_url,
            database,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MERGE (g:Gene {gene_id: row.gene_id})
SET g.ncbi_gene_id = row.ncbi_gene_id,
    g.gene_symbol = row.gene_symbol,
    g.name = row.name,
    g.ensembl_gene_ids = row.ensembl_gene_ids,
    g.gene_type = row.gene_type,
    g.gene_dsi = row.gene_dsi,
    g.gene_dpi = row.gene_dpi,
    g.gene_pli = row.gene_pli,
    g.protein_class_names = row.protein_class_names,
    g.source = row.source,
    g.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )

    for batch in chunked(associations, batch_size):
        post_json(
            base_url,
            database,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MATCH (g:Gene {ncbi_gene_id: row.gene_id})
MATCH (d:Disease {disease_id: row.disease_id})
MERGE (g)-[r:ASSOCIATED_WITH {relation_id: row.relation_id}]->(d)
SET r.source = row.source,
    r.source_record_id = row.assoc_id,
    r.assoc_id = row.assoc_id,
    r.score = row.score,
    r.normalized_score = row.normalized_score,
    r.confidence_score = row.normalized_score,
    r.num_pmids = row.num_pmids,
    r.year_initial = row.year_initial,
    r.year_final = row.year_final,
    r.evidence_index = row.evidence_index,
    r.disgenet_evidence_level = row.disgenet_evidence_level,
    r.score_breakdown = row.score_breakdown,
    r.num_db_snp = row.num_db_snp,
    r.num_clinical_trials = row.num_clinical_trials,
    r.num_chemicals = row.num_chemicals,
    r.num_pmids_with_chemicals = row.num_pmids_with_chemicals,
    r.num_trials_with_chemicals = row.num_trials_with_chemicals,
    r.chemical_evidence = row.chemical_evidence,
    r.tier = row.tier,
    r.tier_reason = row.tier_reason,
    r.evidence_level = row.tier,
    r.api_endpoint = row.api_endpoint,
    r.api_query = row.api_query,
    r.retrieved_at = row.retrieved_at,
    r.validation_status = 'imported',
    r.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/disgenet_liver_backbone/curated"))
    parser.add_argument("--neo4j-base-url", default=os.environ.get("NEO4J_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", DEFAULT_DATABASE))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"))
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    if not args.neo4j_password:
        print("NEO4J_PASSWORD is required", file=sys.stderr)
        return 2

    diseases = normalize_diseases(read_tsv(args.input_dir / "diseases.tsv"))
    genes = normalize_genes(read_tsv(args.input_dir / "genes.tsv"))
    associations = normalize_associations(read_tsv(args.input_dir / "gene_disease_associations.tsv"))
    import_data(
        args.neo4j_base_url,
        args.database,
        args.neo4j_user,
        args.neo4j_password,
        diseases,
        genes,
        associations,
        args.batch_size,
    )
    counts = query_rows(
        args.neo4j_base_url,
        args.database,
        args.neo4j_user,
        args.neo4j_password,
        """
MATCH (d:Disease)
WITH count(d) AS diseases
MATCH (g:Gene)
WITH diseases, count(g) AS genes
MATCH ()-[r:ASSOCIATED_WITH]->()
RETURN diseases, genes, count(r) AS associated_with
""",
    )[0]
    print(json.dumps({"database": args.database, **counts}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
