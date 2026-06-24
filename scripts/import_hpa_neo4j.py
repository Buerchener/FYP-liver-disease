#!/usr/bin/env python3
"""Import selected Human Protein Atlas context for existing Gene nodes."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HPA_BASE_URL = "https://www.proteinatlas.org"
USER_AGENT = "FYP-ontology/0.1 (academic use)"

TISSUE_ZIP = "/download/tsv/rna_tissue_consensus.tsv.zip"
SINGLE_CELL_ZIP = "/download/tsv/rna_single_cell_type.tsv.zip"
CANCER_PROGNOSTIC_ZIP = "/download/tsv/cancer_prognostic_data.tsv.zip"
LIHC_CANCERS = {
    "Liver Hepatocellular Carcinoma (TCGA)": "tcga",
    "Liver Hepatocellular Carcinoma (validation)": "validation",
}


def post_json(url: str, username: str, password: str, statements: list[dict[str, Any]]) -> dict[str, Any]:
    payload = json.dumps({"statements": statements}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    auth = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    request.add_header("Authorization", f"Basic {auth}")
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("errors"):
        raise RuntimeError(json.dumps(result["errors"], indent=2))
    return result


def query_neo4j(url: str, username: str, password: str, statement: str) -> list[dict[str, Any]]:
    result = post_json(url, username, password, [{"statement": statement}])
    columns = result["results"][0]["columns"]
    return [dict(zip(columns, item["row"], strict=True)) for item in result["results"][0]["data"]]


def read_hpa_zip(path: str) -> list[dict[str, str]]:
    request = urllib.request.Request(HPA_BASE_URL + path, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        zipped = response.read()
    archive = zipfile.ZipFile(io.BytesIO(zipped))
    names = archive.namelist()
    if len(names) != 1:
        raise ValueError(f"Expected one file in {path}, found {names}")
    with archive.open(names[0]) as handle:
        text = io.TextIOWrapper(handle, encoding="utf-8")
        return list(csv.DictReader(text, delimiter="\t"))


def read_hpa_json(ensembl_id: str) -> dict[str, Any]:
    request = urllib.request.Request(f"{HPA_BASE_URL}/{ensembl_id}.json", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def chunked(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def classify_prognostic(row: dict[str, str]) -> tuple[str, str, float] | None:
    for column in [
        "validated prognostic - favorable",
        "validated prognostic - unfavorable",
        "potential prognostic - favorable",
        "potential prognostic - unfavorable",
        "unprognostic - favorable",
        "unprognostic - unfavorable",
    ]:
        value = row.get(column)
        if not value:
            continue
        status, direction = column.split(" - ", 1)
        return status, direction, float(value)
    return None


def parse_json_prognostic(value: Any) -> tuple[str, str, float] | None:
    if not isinstance(value, dict):
        return None
    p_value = value.get("p_val")
    if p_value in (None, ""):
        return None
    prognostic = value.get("prognostic") or ""
    direction = value.get("prognostic type") or ""
    if prognostic == "potential prognostic":
        status = "potential prognostic"
    elif prognostic == "prognostic":
        status = "validated prognostic"
    else:
        status = "unprognostic"
    if not direction:
        direction = "not specified"
    return status, direction, float(p_value)


def build_rows(
    genes: list[dict[str, Any]],
    include_single_cell: bool,
    min_single_cell_ncpm: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    genes_by_ensembl = {gene["ensembl_gene_ids"]: gene for gene in genes if gene.get("ensembl_gene_ids")}
    ensembl_ids = set(genes_by_ensembl)

    tissue_nodes: dict[str, dict[str, Any]] = {}
    tissue_expression: list[dict[str, Any]] = []
    for row in read_hpa_zip(TISSUE_ZIP):
        if row.get("Gene") not in ensembl_ids:
            continue
        if row.get("Tissue") != "liver":
            continue
        gene = genes_by_ensembl[row["Gene"]]
        tissue_id = "HPA_TISSUE:liver"
        tissue_nodes[tissue_id] = {
            "tissue_id": tissue_id,
            "name": "liver",
            "tissue_name": "liver",
            "source": "Human Protein Atlas",
        }
        tissue_expression.append(
            {
                "relationship_id": f"HPA_TISSUE_RNA:{gene['gene_id']}|liver",
                "gene_id": gene["gene_id"],
                "gene_symbol": gene["gene_symbol"],
                "ensembl_gene_id": row["Gene"],
                "tissue_id": tissue_id,
                "n_tpm": to_float(row.get("nTPM")),
                "expression_unit": "nTPM",
                "assay": "RNA consensus tissue",
                "source": "Human Protein Atlas",
            }
        )

    cell_type_nodes: dict[str, dict[str, Any]] = {}
    cell_type_expression: list[dict[str, Any]] = []
    if include_single_cell:
        for row in read_hpa_zip(SINGLE_CELL_ZIP):
            if row.get("Gene") not in ensembl_ids:
                continue
            ncpm = to_float(row.get("nCPM")) or 0.0
            if ncpm < min_single_cell_ncpm:
                continue
            gene = genes_by_ensembl[row["Gene"]]
            cell_name = row["Cell type"]
            cell_type_id = f"HPA_CELL_TYPE:{cell_name.lower().replace(' ', '_')}"
            cell_type_nodes[cell_type_id] = {
                "cell_type_id": cell_type_id,
                "name": cell_name,
                "cell_type_name": cell_name,
                "source": "Human Protein Atlas",
            }
            cell_type_expression.append(
                {
                    "relationship_id": f"HPA_CELL_RNA:{gene['gene_id']}|{cell_type_id}",
                    "gene_id": gene["gene_id"],
                    "gene_symbol": gene["gene_symbol"],
                    "ensembl_gene_id": row["Gene"],
                    "cell_type_id": cell_type_id,
                    "n_cpm": ncpm,
                    "expression_unit": "nCPM",
                    "assay": "RNA single cell type",
                    "source": "Human Protein Atlas",
                }
            )

    cancer_prognostic: list[dict[str, Any]] = []
    for row in read_hpa_zip(CANCER_PROGNOSTIC_ZIP):
        ensembl_id = row.get("Gene")
        if ensembl_id not in ensembl_ids:
            continue
        cancer = row.get("Cancer")
        if cancer not in LIHC_CANCERS:
            continue
        parsed = classify_prognostic(row)
        if not parsed:
            continue
        gene = genes_by_ensembl[ensembl_id]
        cohort = LIHC_CANCERS[cancer]
        status, direction, p_value = parsed
        cancer_prognostic.append(
            {
                "relationship_id": f"HPA_LIHC_PROGNOSTIC:{gene['gene_id']}|{cohort}",
                "gene_id": gene["gene_id"],
                "gene_symbol": gene["gene_symbol"],
                "ensembl_gene_id": ensembl_id,
                "disease_id": "UMLS:C2239176",
                "cancer": cancer,
                "cohort": cohort,
                "prognostic_status": status,
                "prognostic_direction": direction,
                "p_value": p_value,
                "source": "Human Protein Atlas",
            }
        )

    return (
        sorted(tissue_nodes.values(), key=lambda item: item["tissue_id"]),
        sorted(tissue_expression, key=lambda item: item["relationship_id"]),
        sorted(cell_type_nodes.values(), key=lambda item: item["cell_type_id"]),
        sorted(cell_type_expression, key=lambda item: item["relationship_id"]),
        sorted(cancer_prognostic, key=lambda item: item["relationship_id"]),
    )


def import_hpa(
    url: str,
    username: str,
    password: str,
    tissues: list[dict[str, Any]],
    tissue_expression: list[dict[str, Any]],
    cell_types: list[dict[str, Any]],
    cell_type_expression: list[dict[str, Any]],
    cancer_prognostic: list[dict[str, Any]],
) -> None:
    for row in tissue_expression:
        row["n_tpm"] = to_float(row.get("n_tpm"))
    for row in cell_type_expression:
        row["n_cpm"] = to_float(row.get("n_cpm"))
    for row in cancer_prognostic:
        row["p_value"] = to_float(row.get("p_value"))

    post_json(
        url,
        username,
        password,
        [
            {"statement": "CREATE CONSTRAINT tissue_id_unique IF NOT EXISTS FOR (t:Tissue) REQUIRE t.tissue_id IS UNIQUE"},
            {
                "statement": (
                    "CREATE CONSTRAINT cell_type_id_unique IF NOT EXISTS "
                    "FOR (c:CellType) REQUIRE c.cell_type_id IS UNIQUE"
                )
            },
            {
                "statement": (
                    "CREATE CONSTRAINT expressed_in_id_unique IF NOT EXISTS "
                    "FOR ()-[r:EXPRESSED_IN]-() REQUIRE r.relationship_id IS UNIQUE"
                )
            },
            {
                "statement": (
                    "CREATE CONSTRAINT prognostic_in_id_unique IF NOT EXISTS "
                    "FOR ()-[r:PROGNOSTIC_IN]-() REQUIRE r.relationship_id IS UNIQUE"
                )
            },
        ],
    )

    for batch in chunked(tissues, 100):
        post_json(
            url,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MERGE (t:Tissue {tissue_id: row.tissue_id})
SET t.name = row.name,
    t.tissue_name = row.tissue_name,
    t.source = row.source,
    t.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )

    for batch in chunked(cell_types, 100):
        post_json(
            url,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MERGE (c:CellType {cell_type_id: row.cell_type_id})
SET c.name = row.name,
    c.cell_type_name = row.cell_type_name,
    c.source = row.source,
    c.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )

    for batch in chunked(tissue_expression, 100):
        post_json(
            url,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MATCH (g:Gene {ncbi_gene_id: row.gene_id})
MATCH (t:Tissue {tissue_id: row.tissue_id})
MERGE (g)-[r:EXPRESSED_IN {relationship_id: row.relationship_id}]->(t)
SET r.source = row.source,
    r.assay = row.assay,
    r.ensembl_gene_id = row.ensembl_gene_id,
    r.n_tpm = row.n_tpm,
    r.expression_unit = row.expression_unit,
    r.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )

    for batch in chunked(cell_type_expression, 100):
        post_json(
            url,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MATCH (g:Gene {ncbi_gene_id: row.gene_id})
MATCH (c:CellType {cell_type_id: row.cell_type_id})
MERGE (g)-[r:EXPRESSED_IN {relationship_id: row.relationship_id}]->(c)
SET r.source = row.source,
    r.assay = row.assay,
    r.ensembl_gene_id = row.ensembl_gene_id,
    r.n_cpm = row.n_cpm,
    r.expression_unit = row.expression_unit,
    r.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )

    for batch in chunked(cancer_prognostic, 100):
        post_json(
            url,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MATCH (g:Gene {ncbi_gene_id: row.gene_id})
MATCH (d:Disease {disease_id: row.disease_id})
MERGE (g)-[r:PROGNOSTIC_IN {relationship_id: row.relationship_id}]->(d)
SET r.source = row.source,
    r.ensembl_gene_id = row.ensembl_gene_id,
    r.cancer = row.cancer,
    r.cohort = row.cohort,
    r.prognostic_status = row.prognostic_status,
    r.prognostic_direction = row.prognostic_direction,
    r.p_value = row.p_value,
    r.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j-url", default="http://100.104.181.96:7474/db/liver-kg-core-v01/tx/commit")
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/hpa_liver_gene_context"))
    parser.add_argument("--include-single-cell", action="store_true")
    parser.add_argument("--min-single-cell-ncpm", type=float, default=100.0)
    parser.add_argument("--from-files", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not args.neo4j_password:
        print("NEO4J_PASSWORD is required", file=sys.stderr)
        return 2

    if args.from_files:
        tissues = read_tsv(args.output_dir / "hpa_tissues.tsv")
        tissue_expression = read_tsv(args.output_dir / "gene_hpa_liver_tissue_expression.tsv")
        cell_types = read_tsv(args.output_dir / "hpa_cell_types.tsv")
        cell_type_expression = read_tsv(args.output_dir / "gene_hpa_cell_type_expression.tsv")
        cancer_prognostic = read_tsv(args.output_dir / "gene_hpa_lihc_prognostic.tsv")
    else:
        genes = query_neo4j(
            args.neo4j_url,
            args.neo4j_user,
            args.neo4j_password,
            """
MATCH (g:Gene)
RETURN g.ncbi_gene_id AS gene_id,
       g.gene_symbol AS gene_symbol,
       g.ensembl_gene_ids AS ensembl_gene_ids
ORDER BY g.gene_symbol
""",
        )
        tissues, tissue_expression, cell_types, cell_type_expression, cancer_prognostic = build_rows(
            genes, args.include_single_cell, args.min_single_cell_ncpm
        )
        write_tsv(args.output_dir / "hpa_tissues.tsv", tissues, ["tissue_id", "name", "tissue_name", "source"])
        write_tsv(
            args.output_dir / "gene_hpa_liver_tissue_expression.tsv",
            tissue_expression,
            [
                "relationship_id",
                "gene_id",
                "gene_symbol",
                "ensembl_gene_id",
                "tissue_id",
                "n_tpm",
                "expression_unit",
                "assay",
                "source",
            ],
        )
        write_tsv(args.output_dir / "hpa_cell_types.tsv", cell_types, ["cell_type_id", "name", "cell_type_name", "source"])
        write_tsv(
            args.output_dir / "gene_hpa_cell_type_expression.tsv",
            cell_type_expression,
            [
                "relationship_id",
                "gene_id",
                "gene_symbol",
                "ensembl_gene_id",
                "cell_type_id",
                "n_cpm",
                "expression_unit",
                "assay",
                "source",
            ],
        )
        write_tsv(
            args.output_dir / "gene_hpa_lihc_prognostic.tsv",
            cancer_prognostic,
            [
                "relationship_id",
                "gene_id",
                "gene_symbol",
                "ensembl_gene_id",
                "disease_id",
                "cancer",
                "cohort",
                "prognostic_status",
                "prognostic_direction",
                "p_value",
                "source",
            ],
        )

    write_metadata(
        args.output_dir / "metadata.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "Human Protein Atlas downloadable TSV",
            "tissue_count": len(tissues),
            "tissue_expression_count": len(tissue_expression),
            "cell_type_count": len(cell_types),
            "cell_type_expression_count": len(cell_type_expression),
            "lihc_prognostic_count": len(cancer_prognostic),
            "include_single_cell": args.include_single_cell or bool(cell_type_expression),
            "min_single_cell_ncpm": args.min_single_cell_ncpm,
        },
    )

    if args.apply:
        import_hpa(
            args.neo4j_url,
            args.neo4j_user,
            args.neo4j_password,
            tissues,
            tissue_expression,
            cell_types,
            cell_type_expression,
            cancer_prognostic,
        )

    print(f"tissues: {len(tissues)}")
    print(f"tissue_expression: {len(tissue_expression)}")
    print(f"cell_types: {len(cell_types)}")
    print(f"cell_type_expression: {len(cell_type_expression)}")
    print(f"lihc_prognostic: {len(cancer_prognostic)}")
    print(f"output_dir: {args.output_dir}")
    print(f"applied: {args.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
