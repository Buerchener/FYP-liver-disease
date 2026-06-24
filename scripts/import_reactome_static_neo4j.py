#!/usr/bin/env python3
"""Import Reactome pathway memberships from a static Ensembl2Reactome mapping file."""

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
from typing import Any, Iterable


DEFAULT_ZIP = Path("/Users/buerchener/Desktop/data_organized_backup_20260616.zip")
DEFAULT_MEMBER = "data_organized_backup_20260616/datasets/hepatology_ontology/ontology_data/Ensembl2Reactome.txt"


def post_json(url: str, username: str, password: str, statements: list[dict[str, Any]]) -> dict[str, Any]:
    payload = json.dumps({"statements": statements}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    auth = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    request.add_header("Authorization", f"Basic {auth}")
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("errors"):
        raise RuntimeError(json.dumps(result["errors"], indent=2, ensure_ascii=False))
    return result


def query_neo4j(url: str, username: str, password: str, statement: str) -> list[dict[str, Any]]:
    result = post_json(url, username, password, [{"statement": statement}])
    columns = result["results"][0]["columns"]
    return [dict(zip(columns, item["row"], strict=True)) for item in result["results"][0]["data"]]


def split_ids(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def iter_mapping_lines(zip_path: Path, member: str) -> Iterable[str]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
            for line in text:
                yield line.rstrip("\n")


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def chunked(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def build_rows(genes: list[dict[str, Any]], zip_path: Path, member: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    id_to_genes: dict[str, list[dict[str, Any]]] = {}
    for gene in genes:
        crossrefs = set()
        crossrefs.update(split_ids(gene.get("ensembl_gene_ids")))
        crossrefs.update(split_ids(gene.get("gene_ensembl_ids")))
        crossrefs.update(split_ids(gene.get("protein_ids_from_disgenet")))
        crossrefs.update(split_ids(gene.get("gene_protein_str_ids")))
        for crossref in crossrefs:
            id_to_genes.setdefault(crossref, []).append(gene)

    pathway_by_id: dict[str, dict[str, Any]] = {}
    membership_by_id: dict[str, dict[str, Any]] = {}
    gene_counts: dict[str, int] = {str(gene["gene_id"]): 0 for gene in genes}
    gene_uniprot_or_ensembl: dict[str, set[str]] = {str(gene["gene_id"]): set() for gene in genes}

    for line in iter_mapping_lines(zip_path, member):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        identifier, stable_id, url, pathway_name, evidence_code, species_name = parts[:6]
        if species_name != "Homo sapiens":
            continue
        if not stable_id.startswith("R-HSA-"):
            continue
        matched_genes = id_to_genes.get(identifier)
        if not matched_genes:
            continue
        pathway_id = f"Reactome:{stable_id}"
        pathway_by_id[pathway_id] = {
            "pathway_id": pathway_id,
            "reactome_stable_id": stable_id,
            "name": pathway_name,
            "source": "Reactome",
            "ncbi_taxon_id": 9606,
            "species_name": species_name,
            "schema_class": "Pathway",
            "is_in_disease": "",
            "has_diagram": "",
            "reactome_url": url,
        }
        for gene in matched_genes:
            gene_id = str(gene["gene_id"])
            relationship_id = f"REACTOME_PARTICIPATION:{gene_id}|{stable_id}"
            membership_by_id[relationship_id] = {
                "relationship_id": relationship_id,
                "gene_id": gene_id,
                "gene_symbol": gene["gene_symbol"],
                "mapping_identifier": identifier,
                "pathway_id": pathway_id,
                "reactome_stable_id": stable_id,
                "evidence_code": evidence_code,
                "source": "Reactome",
            }
            gene_counts[gene_id] += 1
            gene_uniprot_or_ensembl[gene_id].add(identifier)

    mapping_rows = [
        {
            "gene_id": str(gene["gene_id"]),
            "gene_symbol": gene["gene_symbol"],
            "mapping_identifiers": ";".join(sorted(gene_uniprot_or_ensembl[str(gene["gene_id"])])),
            "reactome_membership_count": gene_counts[str(gene["gene_id"])],
            "mapping_status": "mapped" if gene_counts[str(gene["gene_id"])] else "missing",
        }
        for gene in genes
    ]
    return (
        sorted(pathway_by_id.values(), key=lambda row: row["pathway_id"]),
        sorted(membership_by_id.values(), key=lambda row: row["relationship_id"]),
        sorted(mapping_rows, key=lambda row: row["gene_symbol"]),
    )


def import_reactome(url: str, username: str, password: str, pathways: list[dict[str, Any]], memberships: list[dict[str, Any]]) -> None:
    post_json(
        url,
        username,
        password,
        [
            {"statement": "CREATE CONSTRAINT pathway_id_unique IF NOT EXISTS FOR (p:Pathway) REQUIRE p.pathway_id IS UNIQUE"},
            {
                "statement": (
                    "CREATE CONSTRAINT reactome_participation_id_unique IF NOT EXISTS "
                    "FOR ()-[r:PARTICIPATES_IN]-() REQUIRE r.relationship_id IS UNIQUE"
                )
            },
        ],
    )
    for batch in chunked(pathways, 100):
        post_json(
            url,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MERGE (p:Pathway {pathway_id: row.pathway_id})
SET p.name = row.name,
    p.source = row.source,
    p.ncbi_taxon_id = row.ncbi_taxon_id,
    p.species_name = row.species_name,
    p.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )
    for batch in chunked(memberships, 100):
        post_json(
            url,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MATCH (g:Gene {ncbi_gene_id: row.gene_id})
MATCH (p:Pathway {pathway_id: row.pathway_id})
MERGE (g)-[r:PARTICIPATES_IN {relationship_id: row.relationship_id}]->(p)
SET r.source = row.source,
    r.mapping_identifier = row.mapping_identifier,
    r.reactome_stable_id = row.reactome_stable_id,
    r.evidence_code = row.evidence_code,
    r.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j-url", default="http://100.104.181.96:7474/db/liver-kg-core-v02/tx/commit")
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"))
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--zip-member", default=DEFAULT_MEMBER)
    parser.add_argument("--output-dir", type=Path, default=Path("data/reactome_human_liver_genes_v02"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not args.neo4j_password:
        print("NEO4J_PASSWORD is required", file=sys.stderr)
        return 2

    genes = query_neo4j(
        args.neo4j_url,
        args.neo4j_user,
        args.neo4j_password,
        """
MATCH (g:Gene)
RETURN g.ncbi_gene_id AS gene_id,
       g.gene_symbol AS gene_symbol,
       g.ensembl_gene_ids AS ensembl_gene_ids,
       null AS gene_ensembl_ids,
       null AS protein_ids_from_disgenet,
       null AS gene_protein_str_ids
ORDER BY g.gene_symbol
""",
    )
    pathways, memberships, mapping_rows = build_rows(genes, args.zip_path, args.zip_member)
    write_tsv(
        args.output_dir / "reactome_pathways.tsv",
        pathways,
        [
            "pathway_id",
            "reactome_stable_id",
            "name",
            "source",
            "ncbi_taxon_id",
            "species_name",
            "schema_class",
            "reactome_url",
        ],
    )
    write_tsv(
        args.output_dir / "gene_reactome_pathway_memberships.tsv",
        memberships,
        [
            "relationship_id",
            "gene_id",
            "gene_symbol",
            "mapping_identifier",
            "pathway_id",
            "reactome_stable_id",
            "evidence_code",
            "source",
        ],
    )
    write_tsv(
        args.output_dir / "gene_reactome_mapping.tsv",
        mapping_rows,
        ["gene_id", "gene_symbol", "mapping_identifiers", "reactome_membership_count", "mapping_status"],
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": "Reactome static Ensembl2Reactome mapping",
                "gene_count": len(genes),
                "mapped_gene_count": sum(1 for row in mapping_rows if row["mapping_status"] == "mapped"),
                "pathway_count": len(pathways),
                "membership_count": len(memberships),
                "zip_path": str(args.zip_path),
                "zip_member": args.zip_member,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if args.apply:
        import_reactome(args.neo4j_url, args.neo4j_user, args.neo4j_password, pathways, memberships)

    print(f"genes: {len(genes)}")
    print(f"mapped_genes: {sum(1 for row in mapping_rows if row['mapping_status'] == 'mapped')}")
    print(f"pathways: {len(pathways)}")
    print(f"memberships: {len(memberships)}")
    print(f"output_dir: {args.output_dir}")
    print(f"applied: {args.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
