#!/usr/bin/env python3
"""Evaluate legacy rules, improved rules, and selective LLM hybrid on gold profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognitive_agent.hybrid_article_profiler import HybridProfile, profile_article, rule_profile
from cognitive_agent.tool_router import ArticleToolRouter


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def legacy(title: str, abstract: str) -> dict:
    router = ArticleToolRouter()
    p = router.profile(title, abstract)
    plan = router.plan_before_extraction(title, abstract, memory_available=True, rag_enabled=True,
                                         second_llm_enabled=True, reviewer_enabled=False)
    return {"primary_study_type": p.study_type, "has_structured_results": p.has_structured_results,
            "high_extraction_complexity": p.high_complexity, "recommended_route": plan.route}


def score(rows: list[dict], arm: str) -> dict:
    fields = ["primary_study_type", "has_structured_results", "high_extraction_complexity", "recommended_route"]
    metrics = {}
    for field in fields:
        correct = sum(r["gold"][field] == r[arm][field] for r in rows)
        metrics[field] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}
    exact = sum(all(r["gold"][f] == r[arm][f] for f in fields) for r in rows)
    metrics["exact_profile"] = {"correct": exact, "total": len(rows), "accuracy": exact / len(rows)}
    return metrics


def semantic_score(rows: list[dict], arm: str) -> dict:
    if arm == "legacy_rules":
        return {"note": "legacy single-label router does not output modalities, species scope, or evidence posture"}
    species = sum(r["gold_full"]["species_scope"] == r[arm]["species_scope"] for r in rows)
    split_fields = ["evidence_design", "causal_strength", "validation_level"]
    split_accuracy = {
        field: sum(r["gold_full"].get(field) == r[arm].get(field) for r in rows) / len(rows)
        for field in split_fields
    }
    tp = fp = fn = 0
    for row in rows:
        gold = set(row["gold_full"]["secondary_modalities"])
        pred = set(row[arm]["secondary_modalities"])
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "species_scope_accuracy": species / len(rows),
        **{f"{field}_accuracy": value for field, value in split_accuracy.items()},
        "secondary_modality_micro": {"tp": tp, "fp": fp, "fn": fn,
                                     "precision": precision, "recall": recall, "f1": f1},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=ROOT / "gold_annotations/article_profiles_10_gold_v1.json")
    ap.add_argument("--source", type=Path, default=ROOT / "extraction_output/pubmed_converted_500.jsonl")
    ap.add_argument("--output", type=Path, default=ROOT / "benchmark_output/article_profile_hybrid_10.json")
    ap.add_argument("--model-id", default="deepseek-v4-flash")
    ap.add_argument("--force-llm", action="store_true")
    ap.add_argument("--reuse-predictions", type=Path,
                    help="Reuse previously validated hybrid profiles by PMID; routes are recomputed by current code.")
    args = ap.parse_args()
    gold = json.loads(args.gold.read_text())["documents"]
    source = {str(x["pmid"]): x for x in load_jsonl(args.source)}
    reused = {}
    if args.reuse_predictions:
        reused = {str(x["pmid"]): x["hybrid"]
                  for x in json.loads(args.reuse_predictions.read_text()).get("rows", [])}
    rows = []
    for g in gold:
        s = source[str(g["pmid"])]
        improved = rule_profile(s["title"], s["abstract"])
        hybrid = (HybridProfile(**reused[str(g["pmid"])]) if str(g["pmid"]) in reused
                  else profile_article(s["title"], s["abstract"], model_id=args.model_id,
                                       force_llm=args.force_llm))
        legacy_plan = ArticleToolRouter().plan_before_extraction(
            s["title"], s["abstract"], memory_available=True, rag_enabled=True,
            second_llm_enabled=True, reviewer_enabled=False,
        )
        router = ArticleToolRouter()
        improved_plan = router.shadow_plan_before_extraction(
            s["title"], s["abstract"], legacy_plan=legacy_plan,
            memory_available=True, rag_enabled=True, second_llm_enabled=True,
            profile=improved,
        )
        hybrid_plan = router.shadow_plan_before_extraction(
            s["title"], s["abstract"], legacy_plan=legacy_plan,
            memory_available=True, rag_enabled=True, second_llm_enabled=True,
            profile=hybrid,
        )
        improved_dict = improved.to_dict(); improved_dict["recommended_route"] = improved_plan.route
        hybrid_dict = hybrid.to_dict(); hybrid_dict["recommended_route"] = hybrid_plan.route
        gold_core = {"primary_study_type": g["primary_study_type"],
                     "has_structured_results": g["has_structured_results"],
                     "high_extraction_complexity": g["high_extraction_complexity"],
                     "recommended_route": g["recommended_route"]}
        rows.append({"pmid": g["pmid"], "title": s["title"], "gold": gold_core, "gold_full": g,
                     "legacy_rules": legacy(s["title"], s["abstract"]),
                     "improved_rules": improved_dict, "hybrid": hybrid_dict,
                     "shadow_complexity_vector": hybrid_plan.profile.complexity_vector})
        print(g["pmid"], hybrid.source, hybrid.llm_status, flush=True)
    result = {"sample_size": len(rows), "model_id": args.model_id,
              "llm_attempted": sum(r["hybrid"]["llm_status"] != "not_called" for r in rows),
              "llm_calls": sum(r["hybrid"]["llm_status"] == "success" for r in rows),
              "llm_fallbacks": sum(r["hybrid"]["llm_status"].startswith("fallback") for r in rows),
              "llm_call_rate": sum(r["hybrid"]["llm_status"] != "not_called" for r in rows) / len(rows),
              "metrics": {arm: {**score(rows, arm), **semantic_score(rows, arm)}
                          for arm in ("legacy_rules", "improved_rules", "hybrid")},
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
