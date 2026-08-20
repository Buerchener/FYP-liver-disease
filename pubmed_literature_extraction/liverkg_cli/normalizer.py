from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from convert_pubmed_xml_to_jsonl import assess_quality, parse_pubmed_xml

from .pubmed import fetch_pmids


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_input(
    *,
    input_path: Path | None,
    pmids: list[str],
    output_path: Path,
    ncbi_email: str,
    ncbi_tool: str,
    snapshot_dir: Path,
) -> dict[str, Any]:
    if input_path and pmids:
        raise ValueError("local input and --pmid are mutually exclusive")
    if not input_path and not pmids:
        raise ValueError("provide an input file or at least one --pmid")
    if input_path:
        input_path = input_path.expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        suffix = input_path.suffix.lower()
        if suffix == ".jsonl":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(input_path.read_bytes())
            return {
                "kind": "jsonl",
                "source": str(input_path),
                "input_sha256": file_sha256(input_path),
                "normalized_path": str(output_path),
            }
        if suffix == ".xml":
            articles = [assess_quality(item) for item in parse_pubmed_xml(input_path)]
            records = [
                {
                    "pmid": item.get("pmid", ""),
                    "title": item.get("title", ""),
                    "abstract": item.get("abstract", ""),
                    "source": "PubMed",
                    "quality_score": item.get("quality_score", 0),
                }
                for item in articles
            ]
            write_jsonl(records, output_path)
            return {
                "kind": "pubmed_xml",
                "source": str(input_path),
                "input_sha256": file_sha256(input_path),
                "article_count": len(records),
                "normalized_path": str(output_path),
            }
        raise ValueError(f"unsupported input format: {input_path.suffix}")

    articles, stats = fetch_pmids(
        pmids,
        email=ncbi_email,
        tool=ncbi_tool,
        snapshot_dir=snapshot_dir,
    )
    for article in articles:
        assess_quality(article)
    write_jsonl(articles, output_path)
    return {
        "kind": "pmid",
        "requested_pmids": list(dict.fromkeys(pmids)),
        "fetch_stats": stats,
        "normalized_path": str(output_path),
        "input_sha256": file_sha256(output_path),
    }
