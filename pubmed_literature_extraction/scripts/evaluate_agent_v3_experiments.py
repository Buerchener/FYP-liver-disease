#!/usr/bin/env python3
"""Article-bootstrap, paired tests, ablations, and cost/coverage curves."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


REQUIRED_ABLATIONS = (
    "legacy", "agent_v2_no_rules", "deepseek_always", "rule_memory",
    "no_qwen_critic", "no_evidence_selector", "no_conformal_router",
    "no_causal_conflict", "no_cache",
)


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("articles", payload) if isinstance(payload, dict) else payload


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def metrics(rows: list[dict]) -> dict:
    totals = {key: sum(float(row.get(key, 0) or 0) for row in rows) for key in (
        "tp", "fp", "fn", "strict_tp", "strict_fp", "evidence_tp", "evidence_fp",
        "dangerous_writes", "aux_calls", "latency_s", "cost",
    )}
    precision = safe_div(totals["tp"], totals["tp"] + totals["fp"])
    recall = safe_div(totals["tp"], totals["tp"] + totals["fn"])
    return {
        "relation_precision": precision,
        "relation_recall": recall,
        "relation_f1": safe_div(2 * precision * recall, precision + recall),
        "strict_precision": safe_div(totals["strict_tp"], totals["strict_tp"] + totals["strict_fp"]),
        "evidence_precision": safe_div(totals["evidence_tp"], totals["evidence_tp"] + totals["evidence_fp"]),
        "dangerous_writes": int(totals["dangerous_writes"]),
        "avg_aux_calls": safe_div(totals["aux_calls"], len(rows)),
        "avg_latency_s": safe_div(totals["latency_s"], len(rows)),
        "total_cost": totals["cost"],
        "article_count": len(rows),
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap(rows: list[dict], *, iterations: int, seed: int) -> dict:
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        values.append(metrics(sample)["relation_f1"])
    return {
        "iterations": iterations,
        "relation_f1_ci95": [percentile(values, 0.025), percentile(values, 0.975)],
    }


def paired_delta(baseline: list[dict], variant: list[dict], *, iterations: int, seed: int) -> dict:
    base_by_id = {str(row["pmid"]): row for row in baseline}
    var_by_id = {str(row["pmid"]): row for row in variant}
    ids = sorted(set(base_by_id) & set(var_by_id))
    if not ids:
        return {"paired_articles": 0, "error": "no common PMID"}
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        sample_ids = [ids[rng.randrange(len(ids))] for _ in ids]
        base = metrics([base_by_id[item] for item in sample_ids])["relation_f1"]
        var = metrics([var_by_id[item] for item in sample_ids])["relation_f1"]
        deltas.append(var - base)
    observed = metrics([var_by_id[item] for item in ids])["relation_f1"] - metrics(
        [base_by_id[item] for item in ids]
    )["relation_f1"]
    return {
        "paired_articles": len(ids), "observed_f1_delta": observed,
        "delta_ci95": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
        "bootstrap_non_negative_probability": safe_div(sum(item >= 0 for item in deltas), len(deltas)),
        "paired_two_sided_p": min(1.0, 2 * safe_div(sum(item <= 0 for item in deltas), len(deltas))),
    }


def coverage_precision_curve(rows: list[dict]) -> list[dict]:
    candidates = [item for row in rows for item in row.get("candidates", []) or []]
    output = []
    for threshold in (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        accepted = [item for item in candidates if float(item.get("confidence", 0)) >= threshold]
        output.append({
            "threshold": threshold,
            "coverage": safe_div(len(accepted), len(candidates)),
            "precision": safe_div(sum(bool(item.get("correct")) for item in accepted), len(accepted)),
            "accepted": len(accepted),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True, help="name=article_metrics.json")
    parser.add_argument("--baseline", default="legacy")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = {}
    for spec in args.variant:
        name, path = spec.split("=", 1)
        variants[name] = load_rows(Path(path))
    missing = sorted(set(REQUIRED_ABLATIONS) - set(variants))
    if missing:
        print(json.dumps({"status": "INCOMPLETE", "missing_ablations": missing}, indent=2))
        return 2
    baseline = variants[args.baseline]
    report = {
        "status": "COMPLETE", "seed": args.seed, "iterations": args.iterations,
        "required_ablations": list(REQUIRED_ABLATIONS), "variants": {},
    }
    for offset, (name, rows) in enumerate(sorted(variants.items())):
        report["variants"][name] = {
            "metrics": metrics(rows),
            "bootstrap": bootstrap(rows, iterations=args.iterations, seed=args.seed + offset),
            "paired_vs_baseline": (
                None if name == args.baseline else paired_delta(
                    baseline, rows, iterations=args.iterations, seed=args.seed + 100 + offset,
                )
            ),
            "coverage_precision_curve": coverage_precision_curve(rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output": str(args.output), "variants": len(variants)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
