#!/usr/bin/env python3
"""Evaluate production and shadow-union predictions from one agent result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_three_extractors import _dedupe_count, evaluate, load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path,
        default=Path("extraction_output/pubmed_converted_500.jsonl"),
    )
    parser.add_argument(
        "--gold", type=Path,
        default=Path("gold_annotations/pubmed_50_gold_v1.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.run.read_text(encoding="utf-8"))
    source = {str(item["pmid"]): item for item in load_jsonl(args.source)}
    gold = {
        str(item["pmid"]): {
            **item,
            "abstract": source[str(item["pmid"])].get("abstract", ""),
        }
        for item in load_jsonl(args.gold)
    }
    results = []
    shadow_rows = []
    for record in run.get("records", []):
        pmid = str(record.get("pmid", ""))
        verification = record.get("phases", {}).get("verification", {}) or {}
        collaboration = record.get("phases", {}).get("collaboration", {}) or {}
        production_relations = verification.get("relations", []) or []
        entities = verification.get("entities", []) or []
        shadow_relations = (
            collaboration.get("recovery_partition", {}).get("relations", []) or []
        )
        timing = record.get("timing", {}) or {}
        latency = float(timing.get("total_s", 0.0) or 0.0)
        candidates = record.get("phases", {}).get("extraction", {}) or {}

        def arm(relations: list[dict]) -> dict:
            return {
                "prediction": {"entities": entities, "relations": relations},
                "candidate_counts": {
                    "entities": len(candidates.get("entities", []) or []),
                    "relations": len(candidates.get("relations", []) or []),
                    "duplicate_entities": _dedupe_count(
                        candidates.get("entities", []) or [], "entity"
                    ),
                    "duplicate_relations": _dedupe_count(
                        candidates.get("relations", []) or [], "relation"
                    ),
                },
                "latency_s": latency,
                "prompt_tokens": int(collaboration.get("prompt_tokens", 0) or 0),
                "output_tokens": int(collaboration.get("output_tokens", 0) or 0),
                "attempts": int(bool(collaboration.get("triggered"))),
                "invalid_json_attempts": int(
                    collaboration.get("invalid_json_attempts", 0) or 0
                ),
                "error": collaboration.get("error", ""),
            }

        results.append({
            "pmid": pmid,
            "arms": {
                "production": arm(production_relations),
                "shadow_union": arm([*production_relations, *shadow_relations]),
                "shadow_only": arm(shadow_relations),
            },
        })
        for relation in shadow_relations:
            shadow_rows.append({"pmid": pmid, **relation})

    metrics = evaluate(results, gold, (0.0, 0.0))
    payload = {
        "run": str(args.run),
        "documents": len(results),
        "metrics": metrics,
        "shadow_relations": shadow_rows,
        "notes": {
            "production": "relations exposed to decision/write path",
            "shadow_union": "offline counterfactual: production plus recovered relations",
            "shadow_only": "recovered relations alone",
        },
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
