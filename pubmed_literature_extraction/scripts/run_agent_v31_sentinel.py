#!/usr/bin/env python3
"""Offline 20-article gate before any new 100-article/API experiment.

The replay consumes the already frozen Gemini candidates, disables every
auxiliary remote model and Neo4j write, and exercises the repaired local
candidate projector, pair core and verifier.  It is intentionally a development
sentinel, not a publishable held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_agent_v3_experiments import metrics
from scripts.run_agent_v3_ablation100 import (
    ROOT, StatusTracker, article_metric, atomic_json, endpoint_aliases,
    load_jsonl, load_local_env, normalized_triple, run_agent_records, sha256_json,
)


DEFAULT_BASELINE = ROOT / "extraction_output/agent_v3_ablation100_20260815"


def old_records(run_dir: Path) -> dict[str, dict]:
    output = {}
    for path in (run_dir / "arms/legacy").glob("fold*/items/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        output[str(payload["pmid"])] = payload
    return output


def select_sentinel(pmids: list[str], old: dict[str, dict], gold: dict[str, dict], source: dict[str, dict]) -> list[dict]:
    buckets = {"exact_positive": [], "zero_relation": [], "family_or_alias": [], "relation_error": []}
    for pmid in sorted(pmids):
        if pmid not in old:
            continue
        row = article_metric(old[pmid], gold[pmid], source[pmid])
        if not gold[pmid].get("relations"):
            bucket = "zero_relation"
        elif row["article_exact"]:
            bucket = "exact_positive"
        elif row["family_tp"] > row["tp"] or row["alias_tp"] > row["tp"]:
            bucket = "family_or_alias"
        else:
            bucket = "relation_error"
        buckets[bucket].append(pmid)
    selected: list[dict] = []
    used: set[str] = set()
    for bucket in buckets:
        for pmid in buckets[bucket][:5]:
            selected.append({"pmid": pmid, "stratum": bucket})
            used.add(pmid)
    # Some historical runs may not contain five family-normalization rescues.
    # Fill deterministically but retain the observed source stratum in audit.
    for pmid in sorted(pmids):
        if len(selected) >= 20:
            break
        if pmid not in used and pmid in old:
            selected.append({"pmid": pmid, "stratum": "deterministic_fill"})
            used.add(pmid)
    if len(selected) != 20 or len(used) != 20:
        raise RuntimeError("unable to construct a unique 20-article sentinel")
    return selected


def candidate_lattice_family_recall(records: list[dict], gold: dict[str, dict]) -> tuple[float, int, int]:
    matched = total = 0
    for record in records:
        pmid = str(record["pmid"])
        entities = record.get("phases", {}).get("verification", {}).get("entities", []) or []
        aliases, _ = endpoint_aliases(gold[pmid], entities)
        projected = record.get("phases", {}).get("relation_candidate_projection", {}).get("relations", []) or []
        pred_keys = {normalized_triple(item, aliases, family=True) for item in projected}
        for candidate in record.get("phases", {}).get("relation_pair_classification", {}).get("candidates", []) or []:
            for predicate in candidate.get("allowed_predicates", []) or []:
                pred_keys.add(normalized_triple({
                    "subject": candidate.get("subject"), "subject_type": candidate.get("subject_type"),
                    "predicate": predicate, "object": candidate.get("object"),
                    "object_type": candidate.get("object_type"),
                }, aliases, family=True))
        gold_keys = {normalized_triple(item, aliases, family=True) for item in gold[pmid].get("relations", []) or []}
        matched += len(pred_keys & gold_keys)
        total += len(gold_keys)
    return (matched / total if total else 1.0), matched, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "extraction_output/agent_v31_sentinel20")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--with-auxiliary-models", action="store_true")
    args = parser.parse_args()
    baseline = args.baseline_run.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    load_local_env(args.env_file)

    old_manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    frozen = baseline / "frozen_candidates.json"
    gold_rows = load_jsonl(ROOT / "gold_annotations/pubmed_200_gold_v2_strict.jsonl")
    source_rows = load_jsonl(ROOT / "extraction_output/pubmed_converted_500.jsonl")
    gold = {str(item["pmid"]): item for item in gold_rows}
    source = {str(item["pmid"]): item for item in source_rows}
    old = old_records(baseline)
    selection = select_sentinel(old_manifest["evaluation_pmids"], old, gold, source)
    manifest = {
        "protocol": "agent-v3.1-offline-sentinel20-v1",
        "development_only": True,
        "parent_manifest_hash": old_manifest["manifest_hash"],
        "frozen_candidates_sha256": sha256_json(json.loads(frozen.read_text(encoding="utf-8"))),
        "selection": selection,
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    atomic_json(run_dir / "manifest.json", manifest)

    articles = [source[item["pmid"]] for item in selection]
    tracker = StatusTracker(run_dir, total_article_arms=20)
    tracker.start()
    try:
        variant = [
            "--execution-mode", "agent-v2", "--agent-mode", "recall",
            "--pair-classifier-mode", "active", "--rule-memory-mode", "off",
            "--risk-router-mode", "off", "--frozen-candidates", str(frozen),
            "--disable-causal-conflict",
        ]
        if args.with_auxiliary_models:
            variant.extend([
                "--evidence-entailment-mode", "active", "--second-llm-enabled",
                "--second-llm-mode", "conditional",
            ])
        else:
            variant.extend([
                "--evidence-entailment-mode", "off", "--disable-qwen-critic",
            ])
        repaired = run_agent_records(
            task_dir=run_dir / "arms/repaired_local",
            articles=articles, run_name="agent_v31_sentinel20", tracker=tracker,
            max_workers=args.max_workers,
            common=[
                "--extraction-cache-mode", "persistent",
                "--extraction-cache-path", str(run_dir / "sentinel_cache.sqlite3"),
            ],
            variant=variant,
        )
        ids = [item["pmid"] for item in selection]
        baseline_rows = [article_metric(old[pmid], gold[pmid], source[pmid]) for pmid in ids]
        repaired_rows = [article_metric(record, gold[str(record["pmid"])], source[str(record["pmid"])]) for record in repaired]
        baseline_metrics, repaired_metrics = metrics(baseline_rows), metrics(repaired_rows)
        projection_recall, projection_tp, projection_gold = candidate_lattice_family_recall(repaired, gold)
        exact_ids = {item["pmid"] for item in selection if item["stratum"] == "exact_positive"}
        repaired_by_id = {str(item["pmid"]): row for item, row in zip(repaired, repaired_rows)}
        exact_preserved = sum(bool(repaired_by_id[pmid]["article_exact"]) for pmid in exact_ids)
        exact_rate = exact_preserved / len(exact_ids) if exact_ids else 1.0
        local_gates = {
            "bounded_candidate_lattice_family_recall_at_least_0_80": projection_recall >= 0.80,
            "zero_relation_specificity_not_below_historical_sentinel": (
                repaired_metrics["zero_relation_specificity"] >= baseline_metrics["zero_relation_specificity"]
            ),
            "dangerous_writes_zero": repaired_metrics["dangerous_writes"] == 0,
        }
        aux_gates = {
            "family_relation_f1_improves_by_at_least_0_03": (
                repaired_metrics["family_relation_f1"] >= baseline_metrics["family_relation_f1"] + 0.03
            ),
            "family_relation_precision_at_least_0_50": repaired_metrics["family_relation_precision"] >= 0.50,
            "auxiliary_failure_count_zero": repaired_metrics["remote_usage"]["failed"] == 0,
        } if args.with_auxiliary_models else {}
        gates = {**local_gates, **aux_gates}
        passed = all(gates.values())
        report = {
            "status": "PASS" if passed else "FAIL",
            "stage": "auxiliary_sentinel" if args.with_auxiliary_models else "local_candidate_sentinel",
            "manifest_hash": manifest["manifest_hash"], "gates": gates,
            "baseline": baseline_metrics, "repaired": repaired_metrics,
            "bounded_candidate_lattice_family_recall": {
                "value": projection_recall, "matched": projection_tp, "gold": projection_gold,
            },
            "exact_positive_preservation": {
                "value": exact_rate, "preserved": exact_preserved, "total": len(exact_ids),
            },
            "remote_model_calls_expected": "conditional" if args.with_auxiliary_models else 0,
            "safe_to_run_remote_sentinel": bool(not args.with_auxiliary_models and passed),
            "safe_to_expand_to_100": bool(args.with_auxiliary_models and passed),
        }
        atomic_json(run_dir / "sentinel_report.json", report)
        atomic_json(run_dir / "baseline_article_metrics.json", {"articles": baseline_rows})
        atomic_json(run_dir / "repaired_article_metrics.json", {"articles": repaired_rows})
        tracker.close("COMPLETE")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if passed else 3
    except Exception as exc:
        tracker.close("FAILED", str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
