#!/usr/bin/env python3
"""Import selected HMDB metabolite context for existing Gene nodes.

The importer expects a local HMDB all-metabolites XML file. It does not scrape
HMDB pages, because the public website can block automated small-page queries
and the full dataset is large enough that a streaming local parse is safer.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import urllib.request
import xml.etree.ElementTree as ET


DEFAULT_OUT_DIR = Path("data/hmdb_human_liver_genes")
SCOPES = ("disease-linked", "all-gene-associated")


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


def split_ids(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [item.strip() for item in str(value).replace(",", ";").split(";") if item.strip()]


def merge_semicolon(existing: str, value: str) -> str:
    values = split_ids(existing)
    if value and value not in values:
        values.append(value)
    return ";".join(values)


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split())


def child_text(element: ET.Element | None, name: str, namespace: str) -> str:
    if element is None:
        return ""
    child = element.find(f"{namespace}{name}")
    return clean_text(child.text if child is not None else "")


def open_xml(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        if len(xml_names) != 1:
            raise ValueError(f"Expected exactly one XML file in {path}, found {xml_names}")
        return archive.open(xml_names[0])
    return path.open("rb")


def namespace_for(root_tag: str) -> str:
    if root_tag.startswith("{"):
        return root_tag.split("}", 1)[0] + "}"
    return ""


def parse_hmdb(
    hmdb_xml: Path,
    genes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    genes_by_symbol = {row["gene_symbol"].upper(): row for row in genes if row.get("gene_symbol")}
    genes_by_uniprot: dict[str, dict[str, Any]] = {}
    for gene in genes:
        for uniprot_id in split_ids(gene.get("protein_ids_from_disgenet")):
            genes_by_uniprot[uniprot_id.upper()] = gene

    metabolites_by_id: dict[str, dict[str, Any]] = {}
    gene_links_by_id: dict[str, dict[str, Any]] = {}
    disease_links_by_id: dict[str, dict[str, Any]] = {}

    disease_aliases = {
        "hepatocellular carcinoma": "UMLS:C2239176",
        "liver cancer": "UMLS:C2239176",
        "liver carcinoma": "UMLS:C2239176",
        "non-alcoholic fatty liver disease": "UMLS:C0400966",
        "nonalcoholic fatty liver disease": "UMLS:C0400966",
        "non-alcoholic steatohepatitis": "UMLS:C3241937",
        "nonalcoholic steatohepatitis": "UMLS:C3241937",
        "liver fibrosis": "UMLS:C0239946",
        "hepatic fibrosis": "UMLS:C0239946",
        "cirrhosis": "UMLS:C0023890",
        "liver cirrhosis": "UMLS:C0023890",
    }

    with open_xml(hmdb_xml) as handle:
        context = ET.iterparse(handle, events=("start", "end"))
        _, root = next(context)
        ns = namespace_for(root.tag)
        metabolite_tag = f"{ns}metabolite"

        for event, element in context:
            if event != "end" or element.tag != metabolite_tag:
                continue

            hmdb_id = child_text(element, "accession", ns)
            if not hmdb_id:
                root.clear()
                continue

            matched_genes: dict[str, dict[str, Any]] = {}
            protein_associations = element.find(f"{ns}protein_associations")
            if protein_associations is not None:
                for protein in protein_associations.findall(f"{ns}protein"):
                    gene_name = child_text(protein, "gene_name", ns).upper()
                    uniprot_id = child_text(protein, "uniprot_id", ns).upper()
                    gene = genes_by_symbol.get(gene_name) or genes_by_uniprot.get(uniprot_id)
                    if not gene:
                        continue

                    matched_genes[gene["gene_id"]] = gene
                    protein_accession = child_text(protein, "protein_accession", ns)
                    protein_type = child_text(protein, "protein_type", ns)
                    relationship_id = f"HMDB_GENE_METABOLITE:{gene['gene_id']}|{hmdb_id}"
                    row = gene_links_by_id.setdefault(
                        relationship_id,
                        {
                            "relationship_id": relationship_id,
                            "gene_id": gene["gene_id"],
                            "gene_symbol": gene["gene_symbol"],
                            "hmdb_id": hmdb_id,
                            "hmdb_protein_accessions": "",
                            "uniprot_ids": "",
                            "protein_types": "",
                            "source": "HMDB",
                        },
                    )
                    row["hmdb_protein_accessions"] = merge_semicolon(row["hmdb_protein_accessions"], protein_accession)
                    row["uniprot_ids"] = merge_semicolon(row["uniprot_ids"], uniprot_id)
                    row["protein_types"] = merge_semicolon(row["protein_types"], protein_type)

            if not matched_genes:
                element.clear()
                root.clear()
                continue

            taxonomy = element.find(f"{ns}taxonomy")
            metabolites_by_id[hmdb_id] = {
                "metabolite_id": f"HMDB:{hmdb_id}",
                "hmdb_id": hmdb_id,
                "name": child_text(element, "name", ns),
                "chemical_formula": child_text(element, "chemical_formula", ns),
                "monoisotopic_molecular_weight": child_text(element, "monisotopic_molecular_weight", ns),
                "average_molecular_weight": child_text(element, "average_molecular_weight", ns),
                "kingdom": child_text(taxonomy, "kingdom", ns),
                "super_class": child_text(taxonomy, "super_class", ns),
                "class": child_text(taxonomy, "class", ns),
                "source": "HMDB",
            }

            for gene_id, gene in matched_genes.items():
                relationship_id = f"HMDB_GENE_METABOLITE:{gene_id}|{hmdb_id}"
                gene_links_by_id.setdefault(
                    relationship_id,
                    {
                        "relationship_id": relationship_id,
                        "gene_id": gene_id,
                        "gene_symbol": gene["gene_symbol"],
                        "hmdb_id": hmdb_id,
                        "hmdb_protein_accessions": "",
                        "uniprot_ids": "",
                        "protein_types": "",
                        "source": "HMDB",
                    },
                )

            diseases = element.find(f"{ns}diseases")
            if diseases is not None:
                for disease in diseases.findall(f"{ns}disease"):
                    disease_name = child_text(disease, "name", ns)
                    disease_id = disease_aliases.get(disease_name.lower())
                    if not disease_id:
                        continue
                    relationship_id = f"HMDB_DISEASE_METABOLITE:{disease_id}|{hmdb_id}"
                    disease_links_by_id[relationship_id] = {
                        "relationship_id": relationship_id,
                        "disease_id": disease_id,
                        "disease_name": disease_name,
                        "hmdb_id": hmdb_id,
                        "source": "HMDB",
                    }

            element.clear()
            root.clear()

    return (
        sorted(metabolites_by_id.values(), key=lambda row: row["metabolite_id"]),
        sorted(gene_links_by_id.values(), key=lambda row: row["relationship_id"]),
        sorted(disease_links_by_id.values(), key=lambda row: row["relationship_id"]),
    )


def import_hmdb(
    url: str,
    username: str,
    password: str,
    metabolites: list[dict[str, Any]],
    gene_links: list[dict[str, Any]],
    disease_links: list[dict[str, Any]],
) -> None:
    post_json(
        url,
        username,
        password,
        [
            {
                "statement": (
                    "CREATE CONSTRAINT metabolite_id_unique IF NOT EXISTS "
                    "FOR (m:Metabolite) REQUIRE m.metabolite_id IS UNIQUE"
                )
            },
            {
                "statement": (
                    "CREATE CONSTRAINT hmdb_gene_metabolite_id_unique IF NOT EXISTS "
                    "FOR ()-[r:ASSOCIATED_WITH_METABOLITE]-() REQUIRE r.relationship_id IS UNIQUE"
                )
            },
        ],
    )

    for batch in chunked(metabolites, 100):
        post_json(
            url,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MERGE (m:Metabolite {metabolite_id: row.metabolite_id})
SET m.name = row.name,
    m.chemical_formula = row.chemical_formula,
    m.monoisotopic_molecular_weight = toFloat(row.monoisotopic_molecular_weight),
    m.average_molecular_weight = toFloat(row.average_molecular_weight),
    m.kingdom = row.kingdom,
    m.super_class = row.super_class,
    m.class = row.class,
    m.source = row.source,
    m.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )

    for batch in chunked(gene_links, 100):
        post_json(
            url,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MATCH (g:Gene {ncbi_gene_id: row.gene_id})
MATCH (m:Metabolite {metabolite_id: 'HMDB:' + row.hmdb_id})
MERGE (g)-[r:ASSOCIATED_WITH_METABOLITE {relationship_id: row.relationship_id}]->(m)
SET r.gene_symbol = row.gene_symbol,
    r.hmdb_protein_accessions = row.hmdb_protein_accessions,
    r.uniprot_ids = row.uniprot_ids,
    r.protein_types = row.protein_types,
    r.source = row.source,
    r.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )

    for batch in chunked(disease_links, 100):
        post_json(
            url,
            username,
            password,
            [
                {
                    "statement": """
UNWIND $rows AS row
MATCH (d:Disease {disease_id: row.disease_id})
MATCH (m:Metabolite {metabolite_id: 'HMDB:' + row.hmdb_id})
MERGE (m)-[r:ASSOCIATED_WITH {relationship_id: row.relationship_id}]->(d)
SET r.disease_name = row.disease_name,
    r.source = row.source,
    r.updated_at = datetime()
""",
                    "parameters": {"rows": batch},
                }
            ],
        )


def filter_by_scope(
    scope: str,
    metabolites: list[dict[str, Any]],
    gene_links: list[dict[str, Any]],
    disease_links: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if scope == "all-gene-associated":
        return metabolites, gene_links, disease_links
    if scope != "disease-linked":
        raise ValueError(f"Unsupported HMDB scope: {scope}")

    hmdb_ids = {row["hmdb_id"] for row in disease_links}
    metabolites = [row for row in metabolites if row["hmdb_id"] in hmdb_ids]
    gene_links = [row for row in gene_links if row["hmdb_id"] in hmdb_ids]
    return metabolites, gene_links, disease_links


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmdb-xml", type=Path, required=True, help="Local HMDB metabolites XML, .xml.gz, or .zip")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--neo4j-url", default=os.environ.get("NEO4J_URL"))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"))
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="disease-linked",
        help="HMDB import scope. Default keeps only metabolites explicitly linked to current disease stages.",
    )
    parser.add_argument("--build-only", action="store_true", help="Build TSV files but do not import into Neo4j")
    args = parser.parse_args()

    if not args.hmdb_xml.exists():
        print(f"HMDB XML not found: {args.hmdb_xml}", file=sys.stderr)
        return 2
    if not args.neo4j_url or not args.neo4j_password:
        print("NEO4J_URL and NEO4J_PASSWORD are required.", file=sys.stderr)
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
ORDER BY gene_symbol
""",
    )
    metabolites, gene_links, disease_links = parse_hmdb(args.hmdb_xml, genes)
    raw_counts = {
        "raw_metabolite_count": len(metabolites),
        "raw_gene_metabolite_relationship_count": len(gene_links),
        "raw_disease_metabolite_relationship_count": len(disease_links),
    }
    metabolites, gene_links, disease_links = filter_by_scope(args.scope, metabolites, gene_links, disease_links)

    write_tsv(
        args.out_dir / "hmdb_metabolites.tsv",
        metabolites,
        [
            "metabolite_id",
            "name",
            "chemical_formula",
            "monoisotopic_molecular_weight",
            "average_molecular_weight",
            "kingdom",
            "super_class",
            "class",
            "source",
        ],
    )
    write_tsv(
        args.out_dir / "gene_hmdb_metabolite_associations.tsv",
        gene_links,
        [
            "relationship_id",
            "gene_id",
            "gene_symbol",
            "hmdb_id",
            "hmdb_protein_accessions",
            "uniprot_ids",
            "protein_types",
            "source",
        ],
    )
    write_tsv(
        args.out_dir / "hmdb_disease_metabolite_associations.tsv",
        disease_links,
        ["relationship_id", "disease_id", "disease_name", "hmdb_id", "source"],
    )
    metadata = {
        "source": "HMDB",
        "scope": args.scope,
        "hmdb_xml": str(args.hmdb_xml),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gene_count": len(genes),
        **raw_counts,
        "metabolite_count": len(metabolites),
        "gene_metabolite_relationship_count": len(gene_links),
        "disease_metabolite_relationship_count": len(disease_links),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if not args.build_only:
        import_hmdb(args.neo4j_url, args.neo4j_user, args.neo4j_password, metabolites, gene_links, disease_links)

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
