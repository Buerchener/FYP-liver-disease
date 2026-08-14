#!/usr/bin/env python3
"""Compare two Cognitive Agent runs against the same strict gold subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_three_extractors import evaluate, markdown_report


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def duplicate_count(items: list[dict], *, relations: bool) -> int:
    if relations:
        keys = [
            (
                str(x.get("subject", "")).strip().casefold(),
                str(x.get("predicate", "")).strip().upper(),
                str(x.get("object", "")).strip().casefold(),
            )
            for x in items
        ]
    else:
        keys = [
            (str(x.get("mention", "")).strip().casefold(), str(x.get("type", "")))
            for x in items
        ]
    return len(keys) - len(set(keys))


def arm(record: dict, phase: str) -> dict:
    extraction = record.get("phases", {}).get("extraction", {})
    prediction = record.get("phases", {}).get(phase, {})
    entities = prediction.get("entities", []) or []
    relations = prediction.get("relations", []) or []
    warnings = [str(x) for x in extraction.get("warnings", []) or []]
    invalid = sum("invalid json" in x.casefold() or "json parse" in x.casefold() for x in warnings)
    return {
        "prediction": {"entities": entities, "relations": relations},
        "candidate_counts": {
            "entities": len(entities),
            "relations": len(relations),
            "duplicate_entities": duplicate_count(entities, relations=False),
            "duplicate_relations": duplicate_count(relations, relations=True),
        },
        "latency_s": float(record.get("timing", {}).get("total_s", 0.0) or 0.0),
        # LangExtract's OpenAI-compatible provider does not expose token usage.
        "prompt_tokens": 0,
        "output_tokens": 0,
        "attempts": 1 + int(extraction.get("retry_count", 0) or 0),
        "invalid_json_attempts": invalid,
        "error": str(extraction.get("error", "") or ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    run_a = json.loads(args.a.read_text(encoding="utf-8"))
    run_b = json.loads(args.b.read_text(encoding="utf-8"))
    records_a = {str(x["pmid"]): x for x in run_a["records"]}
    records_b = {str(x["pmid"]): x for x in run_b["records"]}
    gold_rows = load_jsonl(args.gold)[: args.limit]
    source = {str(x["pmid"]): x for x in load_jsonl(args.source)}

    gold_eval = {}
    results = []
    missing = []
    for gold in gold_rows:
        pmid = str(gold["pmid"])
        if pmid not in records_a or pmid not in records_b or pmid not in source:
            missing.append(pmid)
            continue
        gold_eval[pmid] = {**gold, "abstract": source[pmid].get("abstract", "")}
        results.append({
            "pmid": pmid,
            "arms": {
                "A_langextract_1_5_raw": arm(records_a[pmid], "extraction"),
                "A_langextract_1_5_final": arm(records_a[pmid], "verification"),
                "B_langextract_1_6_gemini3_raw": arm(records_b[pmid], "extraction"),
                "B_langextract_1_6_gemini3_final": arm(records_b[pmid], "verification"),
            },
        })
    if missing:
        raise SystemExit(f"Missing aligned PMIDs: {missing}")

    metrics = evaluate(results, gold_eval, prices=(0.0, 0.0))
    manifest = {
        "run_id": "langextract_1_5_vs_1_6_gemini3_same50",
        "documents": len(results),
        "model_id": "A=historical model; B=count.gmcli-gemini-3-flash-preview",
        "max_workers": "A=historical; B=6",
        "a_path": str(args.a.resolve()),
        "b_path": str(args.b.resolve()),
        "gold_path": str(args.gold.resolve()),
        "source_path": str(args.source.resolve()),
        "token_cost_note": "Unavailable: LangExtract provider does not expose proxy token usage.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"manifest": manifest, "metrics": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(markdown_report(metrics, manifest), encoding="utf-8")
    print(json.dumps({"documents": len(results), "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
