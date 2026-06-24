#!/usr/bin/env python3
"""Map existing Gene nodes to STRING proteins and import human PPI edges."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRING_BASE_URL = "https://string-db.org/api/tsv"


def post_json(url: str, username: str, password: str, statements: list[dict[str, Any]]) -> dict[str, Any]:
    payload = json.dumps({"statements": statements}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    auth = f"{username}:{password}".encode("utf-8")
    request.add_header("Authorization", "Basic " + __import__("base64").b64encode(auth).decode("ascii"))
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("errors"):
        raise RuntimeError(json.dumps(result["errors"], indent=2))
    return result


def query_neo4j(url: str, username: str, password: str, statement: str) -> list[dict[str, Any]]:
    result = post_json(url, username, password, [{"statement": statement}])
    records = result["results"][0]["data"]
    columns = result["results"][0]["columns"]
    return [dict(zip(columns, item["row"], strict=True)) for item in records]


def string_tsv(endpoint: str, params: dict[str, str]) -> list[dict[str, str]]:
    data = urllib.parse.urlencode(params).encode("utf-8")
    request = urllib.request.Request(f"{STRING_BASE_URL}/{endpoint}", data=data)
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")
    if not text.strip():
        return []
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def chunked(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def build_mapping_rows(genes: list[dict[str, Any]], species: int) -> list[dict[str, Any]]:
    protein_ids = [row["protein_ids_from_disgenet"] for row in genes if row.get("protein_ids_from_disgenet")]
    protein_mapping = (
        string_tsv(
            "get_string_ids",
            {
                "identifiers": "\r".join(protein_ids),
                "species": str(species),
                "limit": "1",
                "echo_query": "1",
            },
        )
        if protein_ids
        else []
    )
    by_protein_id = {row["queryItem"]: row for row in protein_mapping}

    symbols = [row["gene_symbol"] for row in genes]
    symbol_mapping = string_tsv(
        "get_string_ids",
        {
            "identifiers": "\r".join(symbols),
            "species": str(species),
            "limit": "1",
            "echo_query": "1",
        },
    )
    by_symbol = {
        row["queryItem"]: row
        for row in symbol_mapping
        if row.get("preferredName", "").upper() == row.get("queryItem", "").upper()
    }

    rows: list[dict[str, Any]] = []
    for gene in genes:
        mapping_input = "protein_ids_from_disgenet"
        row = by_protein_id.get(gene.get("protein_ids_from_disgenet"))
        if not row:
            mapping_input = "gene_symbol_exact_preferred_name"
            row = by_symbol.get(gene["gene_symbol"])
        if not row:
            rows.append(
                {
                    "gene_id": gene["gene_id"],
                    "gene_symbol": gene["gene_symbol"],
                    "protein_ids_from_disgenet": gene.get("protein_ids_from_disgenet"),
                    "mapping_status": "missing",
                }
            )
            continue
        rows.append(
            {
                "gene_id": gene["gene_id"],
                "gene_symbol": gene["gene_symbol"],
                "protein_ids_from_disgenet": gene.get("protein_ids_from_disgenet"),
                "string_protein_id": row.get("stringId"),
                "preferred_name": row.get("preferredName"),
                "ncbi_taxon_id": int(row.get("ncbiTaxonId") or species),
                "annotation": row.get("annotation", ""),
                "mapping_status": "mapped",
                "mapping_input": mapping_input,
            }
        )
    return rows


def build_interaction_rows(mapping_rows: list[dict[str, Any]], species: int, required_score: int) -> list[dict[str, Any]]:
    identifiers = [row["string_protein_id"] for row in mapping_rows if row.get("string_protein_id")]
    network = string_tsv(
        "network",
        {
            "identifiers": "\r".join(identifiers),
            "species": str(species),
            "required_score": str(required_score),
            "add_nodes": "0",
        },
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in network:
        protein_a = row["stringId_A"]
        protein_b = row["stringId_B"]
        left, right = sorted([protein_a, protein_b])
        interaction_id = f"STRING:{left}|{right}"
        if interaction_id in seen:
            continue
        seen.add(interaction_id)
        rows.append(
            {
                "interaction_id": interaction_id,
                "string_protein_id_a": left,
                "string_protein_id_b": right,
                "preferred_name_a": row["preferredName_A"],
                "preferred_name_b": row["preferredName_B"],
                "ncbi_taxon_id": int(row.get("ncbiTaxonId") or species),
                "score": to_float(row.get("score")),
                "nscore": to_float(row.get("nscore")),
                "fscore": to_float(row.get("fscore")),
                "pscore": to_float(row.get("pscore")),
                "ascore": to_float(row.get("ascore")),
                "escore": to_float(row.get("escore")),
                "dscore": to_float(row.get("dscore")),
                "tscore": to_float(row.get("tscore")),
                "required_score": required_score,
                "source": "STRING",
            }
        )
    return sorted(rows, key=lambda item: item["interaction_id"])


def import_string(
    url: str,
    username: str,
    password: str,
    mapping_rows: list[dict[str, Any]],
    interaction_rows: list[dict[str, Any]],
) -> None:
    post_json(
        url,
        username,
        password,
        [
            {
                "statement": (
                    "CREATE CONSTRAINT protein_id_unique IF NOT EXISTS "
                    "FOR (p:Protein) REQUIRE p.protein_id IS UNIQUE"
                )
            },
            {
                "statement": (
                    "CREATE CONSTRAINT string_interaction_id_unique IF NOT EXISTS "
                    "FOR ()-[r:INTERACTS_WITH]-() REQUIRE r.interaction_id IS UNIQUE"
                )
            },
        ],
    )

    mapped_rows = [row for row in mapping_rows if row.get("mapping_status") == "mapped"]
    for batch in chunked(mapped_rows, 100):
        post_json(
            url,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MATCH (g:Gene {ncbi_gene_id: row.gene_id})
MERGE (p:Protein {protein_id: row.string_protein_id})
SET p.name = row.preferred_name,
    p.ncbi_taxon_id = row.ncbi_taxon_id,
    p.species_name = 'Homo sapiens',
    p.annotation = row.annotation,
    p.source = 'STRING',
    p.updated_at = datetime()
MERGE (g)-[r:ENCODES]->(p)
SET r.source = 'STRING_mapping',
    r.mapping_input = row.mapping_input,
    r.mapping_status = row.mapping_status,
    r.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )

    for batch in chunked(interaction_rows, 100):
        post_json(
            url,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MATCH (a:Protein {protein_id: row.string_protein_id_a})
MATCH (b:Protein {protein_id: row.string_protein_id_b})
MERGE (a)-[r:INTERACTS_WITH {interaction_id: row.interaction_id}]->(b)
SET r.source = row.source,
    r.ncbi_taxon_id = row.ncbi_taxon_id,
    r.species_name = 'Homo sapiens',
    r.score = row.score,
    r.nscore = row.nscore,
    r.fscore = row.fscore,
    r.pscore = row.pscore,
    r.ascore = row.ascore,
    r.escore = row.escore,
    r.dscore = row.dscore,
    r.tscore = row.tscore,
    r.required_score = row.required_score,
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
    parser.add_argument("--species", type=int, default=9606)
    parser.add_argument("--required-score", type=int, default=700)
    parser.add_argument("--output-dir", type=Path, default=Path("data/string_human_liver_genes"))
    parser.add_argument("--apply", action="store_true", help="Write STRING nodes and relationships to Neo4j.")
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
       null AS protein_ids_from_disgenet
ORDER BY g.gene_symbol
""",
    )
    mapping_rows = build_mapping_rows(genes, args.species)
    interaction_rows = build_interaction_rows(mapping_rows, args.species, args.required_score)

    write_tsv(
        args.output_dir / "gene_string_mapping.tsv",
        mapping_rows,
        [
            "gene_id",
            "gene_symbol",
            "protein_ids_from_disgenet",
            "string_protein_id",
            "preferred_name",
            "ncbi_taxon_id",
            "annotation",
            "mapping_status",
            "mapping_input",
        ],
    )
    write_tsv(
        args.output_dir / "string_interactions.tsv",
        interaction_rows,
        [
            "interaction_id",
            "string_protein_id_a",
            "string_protein_id_b",
            "preferred_name_a",
            "preferred_name_b",
            "ncbi_taxon_id",
            "score",
            "nscore",
            "fscore",
            "pscore",
            "ascore",
            "escore",
            "dscore",
            "tscore",
            "required_score",
            "source",
        ],
    )
    write_metadata(
        args.output_dir / "metadata.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "species": args.species,
            "species_name": "Homo sapiens",
            "required_score": args.required_score,
            "gene_count": len(genes),
            "mapped_gene_count": sum(1 for row in mapping_rows if row.get("mapping_status") == "mapped"),
            "missing_gene_symbols": [
                row["gene_symbol"] for row in mapping_rows if row.get("mapping_status") != "mapped"
            ],
            "interaction_count": len(interaction_rows),
            "source": "STRING API",
        },
    )

    if args.apply:
        import_string(args.neo4j_url, args.neo4j_user, args.neo4j_password, mapping_rows, interaction_rows)

    print(f"genes: {len(genes)}")
    print(f"mapped_genes: {sum(1 for row in mapping_rows if row.get('mapping_status') == 'mapped')}")
    print(f"interactions: {len(interaction_rows)}")
    print(f"output_dir: {args.output_dir}")
    print(f"applied: {args.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
