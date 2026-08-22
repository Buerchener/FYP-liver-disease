#!/usr/bin/env python3
"""Unified evaluator: one scorer, one gold-matching rule, every artifact.

Historical artifacts and new experiment arms are scored with the SAME
alias-aware triple matcher and the SAME gold file, so F1 numbers are directly
comparable.  Three conventions are reported separately:

    all-verified   every relation in the verified output (v9/v10 convention)
    accepted       semantic_status == ACCEPTED only (ablation convention)
    import-ready   import_ready == true (Safe Write convention)

Formats:
    agent    cognitive_agent.agent results JSON (records[].phases.verification)
    arms     benchmark_three_extractors-style {"pmid", "arms": {name: {prediction}}}

Usage:
    python scripts/evaluate_unified.py \
        --items "v9=extraction_output/agent_results_router_v9_final_gold50_20260814.json=agent" \
        --items "aliyun=benchmark_output/aliyun_light_models_gold50_20260812/results.json=arms:deepseek-v4-flash" \
        --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pairwise_judge_experiments import (  # noqa: E402
    GOLD_PATH,
    SOURCE_PATH,
    load_jsonl,
    score_arm,
)


def load_agent_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not records and isinstance(payload, dict):
        for value in payload.values():
            if (
                isinstance(value, list) and value
                and isinstance(value[0], dict) and "pmid" in value[0]
            ):
                records = value
                break
    return records


def load_arms_records(path: Path, arm_name: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("results", payload)
    if isinstance(rows, dict):
        rows = list(rows.values())
    records = []
    for item in rows:
        if not isinstance(item, dict) or "pmid" not in item:
            continue
        arms = item.get("arms", {}) or {}
        arm = arms.get(arm_name)
        if not arm:
            continue
        prediction = arm.get("prediction", {}) or {}
        records.append({
            "pmid": str(item["pmid"]),
            "phases": {"verification": prediction},
            "timing": {"total_s": float(arm.get("latency_s", 0.0) or 0.0)},
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", action="append", default=[],
                        help="name=path=format[:arm] (format: agent | arms)")
    parser.add_argument("--limit", type=int, default=50,
                        help="score the first N gold articles")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmark_output/unified_evaluation.md")
    args = parser.parse_args()
    if not args.items:
        print("[ERROR] provide at least one --items entry")
        return 2

    gold_rows = load_jsonl(GOLD_PATH)[: args.limit]
    gold_by_pmid = {str(item["pmid"]): item for item in gold_rows}
    source_by_pmid = {str(item["pmid"]): item for item in load_jsonl(SOURCE_PATH)}

    rows: list[tuple[str, dict]] = []
    for spec in args.items:
        name, path_s, format_spec = (spec.split("=", 2) + ["", ""])[:3]
        if not name or not path_s or not format_spec:
            print(f"[ERROR] bad --items entry: {spec!r} (want name=path=format[:arm])")
            return 2
        path = Path(path_s)
        if not path.exists():
            print(f"[ERROR] missing artifact: {path}")
            return 2
        parts = format_spec.split(":", 1)
        format_name = parts[0]
        if format_name == "agent":
            records = load_agent_records(path)
        elif format_name == "arms":
            records = load_arms_records(path, parts[1] if len(parts) > 1 else "")
        else:
            print(f"[ERROR] unknown format {format_name!r} for {name}")
            return 2
        metrics = score_arm(records, gold_by_pmid, source_by_pmid)
        rows.append((name, metrics))
        print(f"[SCORED] {name}: {len(records)} records")

    lines = [
        "# Unified evaluation (one scorer, one gold-matching rule)",
        "",
        f"- Gold: `pubmed_200_gold_v2_strict` first {args.limit} articles "
        f"({sum(len(g.get('relations', []) or []) for g in gold_rows)} semantic relations, "
        f"{sum(1 for g in gold_rows if not g.get('relations'))} zero-relation docs)",
        "- Matcher: alias-aware canonical triple (subject, type, predicate, object, type)",
        "- Conventions: all-verified / accepted / import-ready reported separately",
        "",
        "| Item | Sem P/R/F1 (all) | Sem P/R/F1 (acc) | Import P/R/F1 | Evid IoU P | NO_REL FP | Calls (judge/2llm/qwen) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in rows:
        sem = metrics["semantic_relation_all_verified"]
        acc = metrics["semantic_relation_accepted_only"]
        strict = metrics["strict_import_ready"]
        calls = metrics["calls"]
        lines.append(
            f"| {name} | {sem['precision']:.3f}/{sem['recall']:.3f}/{sem['f1']:.3f} "
            f"({sem['tp']}/{sem['fp']}/{sem['fn']}) | "
            f"{acc['precision']:.3f}/{acc['recall']:.3f}/{acc['f1']:.3f} | "
            f"{strict['precision']:.3f}/{strict['recall']:.3f}/{strict['f1']:.3f} | "
            f"{metrics['evidence_span_precision_iou_0_5']:.3f} | "
            f"{metrics['no_relation_false_positive_docs']}/{metrics['zero_relation_docs']} | "
            f"{calls['judge']}/{calls['second_llm']}/{calls['qwen_critic']} |"
        )
    lines.append("")
    report = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
