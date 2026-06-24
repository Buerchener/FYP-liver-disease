#!/usr/bin/env python3
"""Import KEGG human pathway memberships for existing Gene nodes."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEGG_BASE_URL = "https://rest.kegg.jp"


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


def get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def parse_kegg_tsv(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        left, right = line.split("\t", 1)
        rows.append((left, right))
    return rows


def fetch_pathway_names(organism: str) -> dict[str, str]:
    text = get_text(f"{KEGG_BASE_URL}/list/pathway/{organism}")
    names: dict[str, str] = {}
    for pathway_id, name in parse_kegg_tsv(text):
        names[pathway_id] = name
        names[f"path:{pathway_id}"] = name
    return names


def fetch_gene_pathway_links(kegg_gene_ids: list[str]) -> list[tuple[str, str]]:
    if not kegg_gene_ids:
        return []
    # KEGG accepts multiple entries joined by "+". Keep chunks conservative.
    links: list[tuple[str, str]] = []
    for index in range(0, len(kegg_gene_ids), 50):
        chunk = kegg_gene_ids[index : index + 50]
        text = get_text(f"{KEGG_BASE_URL}/link/pathway/{'+'.join(chunk)}")
        links.extend(parse_kegg_tsv(text))
    return links


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def chunked(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def normalize_pathway_id(raw_pathway_id: str) -> str:
    return raw_pathway_id.removeprefix("path:")


def build_rows(
    genes: list[dict[str, Any]], organism: str, pathway_names: dict[str, str], links: list[tuple[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    genes_by_kegg = {f"{organism}:{gene['gene_id']}": gene for gene in genes}
    pathway_rows_by_id: dict[str, dict[str, Any]] = {}
    membership_rows: list[dict[str, Any]] = []

    seen_memberships: set[str] = set()
    for kegg_gene_id, raw_pathway_id in links:
        gene = genes_by_kegg.get(kegg_gene_id)
        if not gene:
            continue
        kegg_pathway_id = normalize_pathway_id(raw_pathway_id)
        pathway_id = f"KEGG:{kegg_pathway_id}"
        pathway_name = pathway_names.get(raw_pathway_id) or pathway_names.get(kegg_pathway_id) or ""
        pathway_rows_by_id[pathway_id] = {
            "pathway_id": pathway_id,
            "kegg_pathway_id": kegg_pathway_id,
            "name": pathway_name.removesuffix(" - Homo sapiens (human)"),
            "pathway_name": pathway_name,
            "organism": organism,
            "source": "KEGG",
        }
        relationship_id = f"KEGG_PARTICIPATION:{gene['gene_id']}|{kegg_pathway_id}"
        if relationship_id in seen_memberships:
            continue
        seen_memberships.add(relationship_id)
        membership_rows.append(
            {
                "relationship_id": relationship_id,
                "gene_id": gene["gene_id"],
                "gene_symbol": gene["gene_symbol"],
                "kegg_gene_id": kegg_gene_id,
                "pathway_id": pathway_id,
                "kegg_pathway_id": kegg_pathway_id,
                "source": "KEGG",
            }
        )

    return (
        sorted(pathway_rows_by_id.values(), key=lambda row: row["pathway_id"]),
        sorted(membership_rows, key=lambda row: row["relationship_id"]),
    )


def import_kegg(
    url: str,
    username: str,
    password: str,
    pathways: list[dict[str, Any]],
    memberships: list[dict[str, Any]],
) -> None:
    post_json(
        url,
        username,
        password,
        [
            {
                "statement": (
                    "CREATE CONSTRAINT pathway_id_unique IF NOT EXISTS "
                    "FOR (p:Pathway) REQUIRE p.pathway_id IS UNIQUE"
                )
            },
            {
                "statement": (
                    "CREATE CONSTRAINT kegg_participation_id_unique IF NOT EXISTS "
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
    p.ncbi_taxon_id = 9606,
    p.species_name = 'Homo sapiens',
    p.source = row.source,
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
    r.kegg_gene_id = row.kegg_gene_id,
    r.kegg_pathway_id = row.kegg_pathway_id,
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
    parser.add_argument("--organism", default="hsa")
    parser.add_argument("--output-dir", type=Path, default=Path("data/kegg_human_liver_genes"))
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
RETURN g.ncbi_gene_id AS gene_id, g.gene_symbol AS gene_symbol
ORDER BY g.gene_symbol
""",
    )
    kegg_gene_ids = [f"{args.organism}:{gene['gene_id']}" for gene in genes]
    pathway_names = fetch_pathway_names(args.organism)
    links = fetch_gene_pathway_links(kegg_gene_ids)
    pathways, memberships = build_rows(genes, args.organism, pathway_names, links)

    write_tsv(
        args.output_dir / "kegg_pathways.tsv",
        pathways,
        ["pathway_id", "kegg_pathway_id", "name", "pathway_name", "organism", "source"],
    )
    write_tsv(
        args.output_dir / "gene_kegg_pathway_memberships.tsv",
        memberships,
        [
            "relationship_id",
            "gene_id",
            "gene_symbol",
            "kegg_gene_id",
            "pathway_id",
            "kegg_pathway_id",
            "source",
        ],
    )

    genes_with_pathways = sorted({row["gene_id"] for row in memberships})
    missing_genes = [gene["gene_symbol"] for gene in genes if gene["gene_id"] not in genes_with_pathways]
    write_metadata(
        args.output_dir / "metadata.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "KEGG REST API",
            "organism": args.organism,
            "species_name": "Homo sapiens",
            "gene_count": len(genes),
            "mapped_gene_count": len(genes_with_pathways),
            "missing_gene_symbols": missing_genes,
            "pathway_count": len(pathways),
            "membership_count": len(memberships),
        },
    )

    if args.apply:
        import_kegg(args.neo4j_url, args.neo4j_user, args.neo4j_password, pathways, memberships)

    print(f"genes: {len(genes)}")
    print(f"mapped_genes: {len(genes_with_pathways)}")
    print(f"pathways: {len(pathways)}")
    print(f"memberships: {len(memberships)}")
    print(f"output_dir: {args.output_dir}")
    print(f"applied: {args.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
