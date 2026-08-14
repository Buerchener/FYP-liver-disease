#!/usr/bin/env python3
"""Pre-register a stratified blind-50 cohort before labels or system outputs exist."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path


STRATA = (
    "clinical", "mechanistic_in_vitro", "human_omics_computational",
    "animal", "review_other",
)
STRATIFICATION_VERSION = "blind50-five-strata-v1"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def article_hash(article: dict) -> str:
    raw = f"{article.get('title', '')}\0{article.get('abstract', '')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def classify(article: dict) -> str:
    text = f"{article.get('title', '')} {article.get('abstract', '')}".casefold()
    if re.search(r"\b(review|meta-analysis|guideline|consensus|commentary|perspective)\b", text):
        return "review_other"
    if re.search(r"\b(mouse|mice|murine|rat|rats|rabbit|animal|xenograft|in vivo)\b", text):
        return "animal"
    if re.search(r"\b(omics|transcriptom\w*|proteom\w*|metabolom\w*|gwas|bioinformatic\w*|computational|machine learning|deep learning)\b", text):
        return "human_omics_computational"
    if re.search(r"\b(in vitro|cell line|hepg2|huh7|organoid|mechanis(?:m|tic))\b", text):
        return "mechanistic_in_vitro"
    if re.search(r"\b(patient|patients|human|cohort|clinical|trial|retrospective|prospective|case-control)\b", text):
        return "clinical"
    return "review_other"


def git_parent() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", type=Path, default=Path("extraction_output/pubmed_converted_500.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("gold_annotations/blind50/blind50_preregistered_seed20260814.json"))
    parser.add_argument("--annotation-template", type=Path, default=Path("gold_annotations/blind50/blind50_annotation_template.jsonl"))
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--start-index", type=int, default=201)
    parser.add_argument("--end-index", type=int, default=500)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.articles.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = rows[args.start_index - 1:args.end_index]
    if len(candidates) != args.end_index - args.start_index + 1:
        raise ValueError("candidate index range is incomplete")
    pools = {name: [] for name in STRATA}
    for source_index, article in enumerate(candidates, start=args.start_index):
        pools[classify(article)].append((source_index, article))
    rng = random.Random(args.seed)
    selected = []
    for stratum in STRATA:
        rng.shuffle(pools[stratum])
        if len(pools[stratum]) < 10:
            raise ValueError(f"stratum {stratum} has only {len(pools[stratum])} candidates")
        selected.extend((stratum, *item) for item in pools[stratum][:10])
    selected.sort(key=lambda item: (STRATA.index(item[0]), str(item[2].get("pmid", ""))))
    records = []
    annotations = []
    for within_stratum, (stratum, source_index, article) in enumerate(selected):
        digest = article_hash(article)
        record = {
            "pmid": str(article.get("pmid", "")),
            "abstract_sha256": digest,
            "stratum": stratum,
            "source_index_1_based": source_index,
        }
        records.append(record)
        annotations.append({
            **record,
            "reviewer_1": "",
            "reviewer_2_required": within_stratum % 10 < 2,
            "in_scope": None,
            "entities": [],
            "relations": [],
            "negative_notes": [],
            "adjudication_status": "UNLABELED",
        })
    manifest = {
        "manifest_version": "agent-v3-blind50-prereg-v1",
        "created_before_blind_outputs": True,
        "labels_inspected": False,
        "frozen": True,
        "seed": args.seed,
        "source_range_1_based": [args.start_index, args.end_index],
        "stratification_version": STRATIFICATION_VERSION,
        "target_per_stratum": 10,
        "second_reviewer_minimum_fraction": 0.20,
        "source_path": str(args.articles),
        "source_sha256": file_hash(args.articles),
        "code_parent_commit": git_parent(),
        "active_rule_bundle_hash": "NOT_YET_FROZEN",
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.annotation_template.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in annotations),
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(args.manifest), "annotation_template": str(args.annotation_template),
        "selected": len(records), "per_stratum": 10,
    }))


if __name__ == "__main__":
    main()
