#!/usr/bin/env python3
"""Freeze the 120/40/40 Agent v3 development split without sklearn."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


TARGETS = {"rule_induction": 120, "validation": 40, "calibration": 40}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def study_bucket(value: str) -> str:
    text = str(value or "").casefold()
    if any(token in text for token in ("review", "commentary", "consensus", "guideline")):
        return "review_other"
    if any(token in text for token in ("mouse", "murine", "rabbit", "animal", "xenograft", "preclinical")):
        return "animal_mechanistic"
    if any(token in text for token in ("cell", "vitro", "hepg2", "hsc", "aml12")):
        return "in_vitro_mechanistic"
    if any(token in text for token in ("omics", "comput", "bioinform", "machine_learning", "model")):
        return "human_omics_computational"
    return "clinical_human"


def stratum(row: dict) -> str:
    relations = row.get("relations", []) or []
    strict_positive = any(item.get("import_ready") for item in relations)
    return "|".join((
        study_bucket(row.get("study_context", "")),
        "in_scope" if row.get("in_scope") else "out_scope",
        "semantic_positive" if relations else "no_relation",
        "strict_positive" if strict_positive else "not_strict",
    ))


def split_rows(rows: list[dict], seed: int) -> dict[str, str]:
    if len(rows) != sum(TARGETS.values()):
        raise ValueError(f"expected exactly 200 gold articles, found {len(rows)}")
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    for group in groups.values():
        rng.shuffle(group)
    # Interleave strata so singleton strata do not all fall into induction.
    ordered: list[tuple[str, dict]] = []
    keys = sorted(groups)
    rng.shuffle(keys)
    while any(groups.values()):
        for key in keys:
            if groups[key]:
                ordered.append((key, groups[key].pop()))
    assigned = {name: 0 for name in TARGETS}
    group_assigned: dict[str, dict[str, int]] = defaultdict(lambda: {name: 0 for name in TARGETS})
    output: dict[str, str] = {}
    for key, row in ordered:
        available = [name for name, target in TARGETS.items() if assigned[name] < target]
        chosen = min(available, key=lambda name: (
            group_assigned[key][name] / TARGETS[name],
            assigned[name] / TARGETS[name],
            name,
        ))
        output[str(row["pmid"])] = chosen
        assigned[chosen] += 1
        group_assigned[key][chosen] += 1
    if assigned != TARGETS:
        raise RuntimeError(f"split allocation failed: {assigned}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("gold_annotations/pubmed_200_gold_v2_strict.jsonl"))
    parser.add_argument("--articles", type=Path, default=Path("extraction_output/pubmed_converted_500.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("gold_annotations/splits/agent_v3_dev_manifest_seed20260814.json"))
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    gold = load_jsonl(args.gold)
    articles = {str(item.get("pmid")): item for item in load_jsonl(args.articles)}
    assignments = split_rows(gold, args.seed)
    records = []
    for row in sorted(gold, key=lambda item: str(item["pmid"])):
        pmid = str(row["pmid"])
        article = articles.get(pmid, {})
        content = f"{article.get('title', row.get('title', ''))}\0{article.get('abstract', '')}"
        records.append({
            "pmid": pmid,
            "split": assignments[pmid],
            "stratum": stratum(row),
            "article_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })
    manifest = {
        "manifest_version": "agent-v3-dev-split-v1",
        "seed": args.seed,
        "frozen": True,
        "targets": TARGETS,
        "gold_source": str(args.gold),
        "gold_sha256": file_hash(args.gold),
        "article_source": str(args.articles),
        "article_source_sha256": file_hash(args.articles),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": TARGETS, "seed": args.seed}))


if __name__ == "__main__":
    main()
