#!/usr/bin/env python3
"""Comprehensive article-clustered evaluation for Agent v3 ablations."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_ABLATIONS = (
    "legacy", "agent_v2_no_rules", "deepseek_always", "full_v3",
    "v3_no_rule_memory", "v3_no_qwen_critic", "v3_no_evidence_selector",
    "v3_no_conformal_router", "v3_no_causal_conflict", "v3_no_cache",
)
CORE_BOOTSTRAP_METRICS = (
    "relation_f1", "alias_relation_f1", "family_relation_f1",
    "strict_precision", "strict_f1", "evidence_iou_precision",
    "evidence_iou_f1", "article_exact_rate", "zero_relation_specificity",
)


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("articles", payload) if isinstance(payload, dict) else payload


def safe_div(num: float, den: float, *, empty: float = 0.0) -> float:
    return num / den if den else empty


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = safe_div(tp, tp + fp, empty=1.0)
    recall = safe_div(tp, tp + fn, empty=1.0)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": safe_div(sum(values), len(values)),
        "p50": percentile(values, 0.50), "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95), "p99": percentile(values, 0.99),
        "sum": sum(values),
    }


def calibration_metrics(pairs: list[list[float]]) -> dict[str, float]:
    if not pairs:
        return {"brier": 0.0, "ece_10bin": 0.0, "count": 0}
    brier = sum((float(conf) - int(correct)) ** 2 for conf, correct in pairs) / len(pairs)
    bins: defaultdict[int, list[tuple[float, int]]] = defaultdict(list)
    for conf, correct in pairs:
        bins[min(9, int(max(0.0, min(0.999999, float(conf))) * 10))].append((float(conf), int(correct)))
    ece = sum(
        len(items) / len(pairs) * abs(
            sum(item[0] for item in items) / len(items)
            - sum(item[1] for item in items) / len(items)
        )
        for items in bins.values()
    )
    return {"brier": brier, "ece_10bin": ece, "count": len(pairs)}


def metrics(rows: list[dict]) -> dict[str, Any]:
    sum_fields = (
        "entity_tp", "entity_fp", "entity_fn", "tp", "fp", "fn",
        "alias_tp", "alias_fp", "alias_fn", "family_tp", "family_fp", "family_fn",
        "gene_protein_type_matched", "gene_protein_type_correct",
        "strict_tp", "strict_fp", "strict_fn", "evidence_exact_tp",
        "evidence_iou_tp", "evidence_pred", "evidence_gold", "evidence_contiguous",
        "endpoint_coverage", "trigger_coverage", "direction_confusion",
        "zero_relation_gold", "zero_relation_correct", "hard_negative_fp",
        "dangerous_writes", "schema_violations", "negation_background_method_fp",
        "linking_ambiguous", "linking_total", "actions", "budget_escalations",
        "remote_attempted", "remote_successful", "remote_failed", "remote_retried",
        "aux_calls", "state_changes", "zero_change_calls", "prompt_tokens",
        "output_tokens", "article_exact", "predicate_direction_errors",
        "semantic_accepted", "semantic_review", "semantic_rejected", "semantic_only",
        "candidate_projected", "candidate_attribute_sources", "candidate_top_level_sources",
        "candidate_pair_count", "candidate_verified_count",
    )
    totals = {key: sum(float(row.get(key, 0) or 0) for row in rows) for key in sum_fields}
    entity_p, entity_r, entity_f1 = prf(totals["entity_tp"], totals["entity_fp"], totals["entity_fn"])
    relation_p, relation_r, relation_f1 = prf(totals["tp"], totals["fp"], totals["fn"])
    alias_p, alias_r, alias_f1 = prf(totals["alias_tp"], totals["alias_fp"], totals["alias_fn"])
    family_p, family_r, family_f1 = prf(totals["family_tp"], totals["family_fp"], totals["family_fn"])
    strict_p, strict_r, strict_f1 = prf(totals["strict_tp"], totals["strict_fp"], totals["strict_fn"])
    evidence_exact_p, evidence_exact_r, evidence_exact_f1 = prf(
        totals["evidence_exact_tp"], totals["evidence_pred"] - totals["evidence_exact_tp"],
        totals["evidence_gold"] - totals["evidence_exact_tp"],
    )
    evidence_iou_p, evidence_iou_r, evidence_iou_f1 = prf(
        totals["evidence_iou_tp"], totals["evidence_pred"] - totals["evidence_iou_tp"],
        totals["evidence_gold"] - totals["evidence_iou_tp"],
    )
    predicate_totals: defaultdict[str, Counter] = defaultdict(Counter)
    route_counts: Counter[str] = Counter()
    termination_counts: Counter[str] = Counter()
    phase_latency: defaultdict[str, list[float]] = defaultdict(list)
    confidence_pairs = []
    candidate_sources: Counter[str] = Counter()
    for row in rows:
        for predicate, counts in (row.get("predicate_counts", {}) or {}).items():
            predicate_totals[predicate].update({key: int(value or 0) for key, value in counts.items()})
        route_counts.update(row.get("route_counts", {}) or {})
        termination_counts[str(row.get("termination_reason", "") or "unspecified")] += 1
        for phase, value in (row.get("phase_latency", {}) or {}).items():
            phase_latency[phase].append(float(value or 0.0))
        confidence_pairs.extend(row.get("confidence_correct", []) or [])
        candidate_sources[str(row.get("candidate_source", "unknown"))] += 1
    predicate_metrics = {}
    for predicate, counts in sorted(predicate_totals.items()):
        p, r, f1 = prf(counts["tp"], counts["fp"], counts["fn"])
        predicate_metrics[predicate] = {**dict(counts), "precision": p, "recall": r, "f1": f1}
    active_predicates = [
        item for item in predicate_metrics.values()
        if int(item.get("tp", 0)) + int(item.get("fp", 0)) + int(item.get("fn", 0)) > 0
    ]
    macro_f1 = safe_div(
        sum(float(item["f1"]) for item in active_predicates),
        len(active_predicates),
    )
    latencies = [float(row.get("latency_s", 0.0) or 0.0) for row in rows]
    route_total = sum(route_counts.values())
    result = {
        "article_count": len(rows),
        "entity_endpoint_precision": entity_p, "entity_endpoint_recall": entity_r,
        "entity_endpoint_f1": entity_f1,
        "relation_precision": relation_p, "relation_recall": relation_r,
        "relation_f1": relation_f1, "relation_macro_f1": macro_f1,
        "alias_relation_precision": alias_p, "alias_relation_recall": alias_r,
        "alias_relation_f1": alias_f1,
        "family_relation_precision": family_p, "family_relation_recall": family_r,
        "family_relation_f1": family_f1,
        "gene_protein_type_accuracy": safe_div(
            totals["gene_protein_type_correct"], totals["gene_protein_type_matched"], empty=1.0,
        ),
        "article_exact_rate": safe_div(totals["article_exact"], len(rows)),
        "strict_precision": strict_p, "strict_recall": strict_r, "strict_f1": strict_f1,
        "strict_coverage": safe_div(totals["strict_tp"] + totals["strict_fp"], totals["tp"] + totals["fp"]),
        "evidence_exact_precision": evidence_exact_p, "evidence_exact_recall": evidence_exact_r,
        "evidence_exact_f1": evidence_exact_f1,
        "evidence_iou_precision": evidence_iou_p, "evidence_iou_recall": evidence_iou_r,
        "evidence_iou_f1": evidence_iou_f1,
        "evidence_contiguous_rate": safe_div(totals["evidence_contiguous"], totals["evidence_pred"], empty=1.0),
        "evidence_endpoint_coverage": safe_div(totals["endpoint_coverage"], totals["tp"]),
        "evidence_trigger_coverage": safe_div(totals["trigger_coverage"], totals["tp"]),
        "zero_relation_specificity": safe_div(totals["zero_relation_correct"], totals["zero_relation_gold"], empty=1.0),
        "hard_negative_false_positives": int(totals["hard_negative_fp"]),
        "direction_confusions": int(totals["direction_confusion"]),
        "dangerous_writes": int(totals["dangerous_writes"]),
        "dangerous_write_precision": (
            safe_div(totals["dangerous_writes"], totals["strict_tp"] + totals["strict_fp"])
            if totals["strict_tp"] + totals["strict_fp"] else None
        ),
        "schema_violations": int(totals["schema_violations"]),
        "negation_background_method_false_positives": int(totals["negation_background_method_fp"]),
        "linking_ambiguity_rate": safe_div(totals["linking_ambiguous"], totals["linking_total"]),
        "predicate_metrics": predicate_metrics,
        "risk_routing": {
            "counts": dict(route_counts),
            "rates": {key: safe_div(value, route_total) for key, value in route_counts.items()},
            "total": route_total,
        },
        "calibration": calibration_metrics(confidence_pairs),
        "agent": {
            "avg_actions": safe_div(totals["actions"], len(rows)),
            "budget_escalations": int(totals["budget_escalations"]),
            "termination_counts": dict(termination_counts),
            "state_changes_per_remote_call": safe_div(totals["state_changes"], totals["remote_attempted"]),
            "zero_change_call_rate": safe_div(totals["zero_change_calls"], totals["remote_attempted"]),
        },
        "remote_usage": {
            "attempted": int(totals["remote_attempted"]),
            "successful": int(totals["remote_successful"]),
            "failed": int(totals["remote_failed"]),
            "retried": int(totals["remote_retried"]),
            "avg_aux_calls": safe_div(totals["aux_calls"], len(rows)),
            "prompt_tokens": int(totals["prompt_tokens"]),
            "output_tokens": int(totals["output_tokens"]),
            "total_tokens": int(totals["prompt_tokens"] + totals["output_tokens"]),
        },
        "avg_aux_calls": safe_div(totals["aux_calls"], len(rows)),
        "latency": latency_summary(latencies),
        "phase_latency": {key: latency_summary(values) for key, values in sorted(phase_latency.items())},
        "candidate_sources": dict(candidate_sources),
        "semantic_write_funnel": {
            "semantic_accepted": int(totals["semantic_accepted"]),
            "semantic_review": int(totals["semantic_review"]),
            "semantic_rejected": int(totals["semantic_rejected"]),
            "semantic_only": int(totals["semantic_only"]),
        },
        "candidate_funnel": {
            "top_level_sources": int(totals["candidate_top_level_sources"]),
            "attribute_sources": int(totals["candidate_attribute_sources"]),
            "projected_after_dedup": int(totals["candidate_projected"]),
            "pair_candidates": int(totals["candidate_pair_count"]),
            "verified_candidates": int(totals["candidate_verified_count"]),
        },
        "raw_denominators": {key: int(value) for key, value in totals.items()},
        "cost": {
            "estimated_usd": None,
            "reason": "provider prices not supplied; tokens and request counts reported without invented pricing",
        },
    }
    return result


def bootstrap(rows: list[dict], *, iterations: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {"iterations": 0, "ci95": {}}
    rng = random.Random(seed)
    values = {key: [] for key in CORE_BOOTSTRAP_METRICS}
    for _ in range(iterations):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        result = metrics(sample)
        for key in values:
            values[key].append(float(result[key]))
    return {
        "iterations": iterations,
        "ci95": {key: [percentile(items, 0.025), percentile(items, 0.975)] for key, items in values.items()},
    }


def paired_delta(
    baseline: list[dict], variant: list[dict], *, iterations: int, seed: int,
) -> dict[str, Any]:
    base_by_id = {str(row["pmid"]): row for row in baseline}
    var_by_id = {str(row["pmid"]): row for row in variant}
    ids = sorted(set(base_by_id) & set(var_by_id))
    if not ids:
        return {"paired_articles": 0, "error": "no common PMID"}
    rng = random.Random(seed)
    observed_base, observed_var = metrics([base_by_id[item] for item in ids]), metrics([var_by_id[item] for item in ids])
    deltas = {key: [] for key in CORE_BOOTSTRAP_METRICS}
    latency_deltas = []
    for _ in range(iterations):
        sampled_ids = [ids[rng.randrange(len(ids))] for _ in ids]
        base_result = metrics([base_by_id[item] for item in sampled_ids])
        var_result = metrics([var_by_id[item] for item in sampled_ids])
        for key in deltas:
            deltas[key].append(float(var_result[key]) - float(base_result[key]))
        latency_deltas.append(var_result["latency"]["mean"] - base_result["latency"]["mean"])
    output = {"paired_articles": len(ids), "metrics": {}}
    for key, values in deltas.items():
        observed = float(observed_var[key]) - float(observed_base[key])
        lower_tail = safe_div(sum(item <= 0 for item in values), len(values))
        upper_tail = safe_div(sum(item >= 0 for item in values), len(values))
        output["metrics"][key] = {
            "observed_delta": observed,
            "delta_ci95": [percentile(values, 0.025), percentile(values, 0.975)],
            "non_negative_probability": safe_div(sum(item >= 0 for item in values), len(values)),
            "paired_two_sided_p": min(1.0, 2 * min(lower_tail, upper_tail)),
        }
    output["latency_mean_delta"] = {
        "observed_delta": observed_var["latency"]["mean"] - observed_base["latency"]["mean"],
        "delta_ci95": [percentile(latency_deltas, 0.025), percentile(latency_deltas, 0.975)],
    }
    # Backward-compatible aliases retained for the original v3 research
    # protocol and downstream notebooks.
    relation_delta = output["metrics"]["relation_f1"]
    output.update({
        "observed_f1_delta": relation_delta["observed_delta"],
        "delta_ci95": relation_delta["delta_ci95"],
        "bootstrap_non_negative_probability": relation_delta["non_negative_probability"],
        "paired_two_sided_p": relation_delta["paired_two_sided_p"],
    })
    return output


def coverage_precision_curve(rows: list[dict]) -> list[dict]:
    pairs = [item for row in rows for item in row.get("confidence_correct", []) or []]
    output = []
    for threshold in (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        accepted = [item for item in pairs if float(item[0]) >= threshold]
        output.append({
            "threshold": threshold,
            "coverage": safe_div(len(accepted), len(pairs)),
            "precision": safe_div(sum(int(item[1]) for item in accepted), len(accepted), empty=1.0),
            "accepted": len(accepted),
        })
    return output


def holm_bonferroni(entries: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(entries, key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (count - index)))
        adjusted[name] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True, help="name=article_metrics.json")
    parser.add_argument("--baseline", default="legacy")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260815)
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
    if any(len(rows) != 100 for rows in variants.values()):
        counts = {name: len(rows) for name, rows in variants.items()}
        print(json.dumps({"status": "INCOMPLETE", "article_counts": counts}, indent=2))
        return 2
    baseline = variants[args.baseline]
    full = variants["full_v3"]
    report: dict[str, Any] = {
        "status": "COMPLETE", "seed": args.seed, "iterations": args.iterations,
        "required_ablations": list(REQUIRED_ABLATIONS), "variants": {},
        "multiple_comparison": {},
    }
    p_values_vs_legacy = []
    p_values_vs_full = []
    for offset, name in enumerate(REQUIRED_ABLATIONS):
        rows = variants[name]
        vs_legacy = None if name == args.baseline else paired_delta(
            baseline, rows, iterations=args.iterations, seed=args.seed + 100 + offset,
        )
        vs_full = None if name == "full_v3" else paired_delta(
            full, rows, iterations=args.iterations, seed=args.seed + 300 + offset,
        )
        report["variants"][name] = {
            "metrics": metrics(rows),
            "bootstrap": bootstrap(rows, iterations=args.iterations, seed=args.seed + offset),
            "paired_vs_legacy": vs_legacy,
            "paired_vs_full_v3": vs_full,
            "coverage_precision_curve": coverage_precision_curve(rows),
        }
        if vs_legacy:
            p_values_vs_legacy.append((name, vs_legacy["metrics"]["relation_f1"]["paired_two_sided_p"]))
        if vs_full:
            p_values_vs_full.append((name, vs_full["metrics"]["relation_f1"]["paired_two_sided_p"]))
    report["multiple_comparison"] = {
        "method": "Holm-Bonferroni",
        "relation_f1_vs_legacy_adjusted_p": holm_bonferroni(p_values_vs_legacy),
        "relation_f1_vs_full_v3_adjusted_p": holm_bonferroni(p_values_vs_full),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output": str(args.output), "variants": len(variants)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
