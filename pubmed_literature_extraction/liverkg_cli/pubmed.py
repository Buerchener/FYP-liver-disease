from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests


EUTILS_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def parse_pubmed_xml_text(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    articles: list[dict[str, Any]] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = _text(art.find(".//PMID"))
        title = _text(art.find(".//ArticleTitle"))
        abstract_parts: list[str] = []
        for abstract in art.findall(".//AbstractText"):
            label = abstract.get("Label", "")
            text = _text(abstract)
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        if pmid:
            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": " ".join(abstract_parts).strip(),
                "source": "PubMed",
            })
    return articles


def fetch_pmids(
    pmids: list[str],
    *,
    email: str,
    tool: str = "liverkg",
    snapshot_dir: Path,
    rate_limit_s: float = 0.34,
    retries: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not email:
        raise ValueError("NCBI email is required for PMID fetching")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    unique_pmids = list(dict.fromkeys(str(p).strip() for p in pmids if str(p).strip()))
    articles: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, pmid in enumerate(unique_pmids):
        if index:
            time.sleep(rate_limit_s)
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "email": email,
            "tool": tool,
        }
        last_error = ""
        for attempt in range(retries):
            try:
                response = requests.get(EUTILS_FETCH_URL, params=params, timeout=30)
                response.raise_for_status()
                xml_text = response.text
                sha = hashlib.sha256(xml_text.encode("utf-8")).hexdigest()
                (snapshot_dir / f"{pmid}.xml").write_text(xml_text, encoding="utf-8")
                parsed = parse_pubmed_xml_text(xml_text)
                if not parsed:
                    raise ValueError("no PubMedArticle found")
                for article in parsed:
                    article["source_snapshot_sha256"] = sha
                articles.extend(parsed)
                last_error = ""
                break
            except Exception as exc:
                last_error = str(exc)
                time.sleep(min(8, 2 ** attempt))
        if last_error:
            failures.append({"pmid": pmid, "error": last_error})

    stats = {
        "requested": len(unique_pmids),
        "fetched": len(articles),
        "failed": len(failures),
        "failures": failures,
    }
    if failures:
        raise RuntimeError(json.dumps(stats, ensure_ascii=False))
    return articles, stats
