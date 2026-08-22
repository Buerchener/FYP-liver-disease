#!/usr/bin/env python3
"""10-article tiered-v2 verifier sentinel with cold/warm cache replay.

Development-only.  It consumes frozen Gemini candidates, keeps Neo4j writes
disabled, compares legacy vs tiered-v2, and immediately replays the tiered arm
against the same auxiliary SQLite cache to confirm local warm-cache behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_agent_v3_experiments import metrics
from scripts.run_agent_v3_ablation100 import (
    StatusTracker,
    article_metric,
    atomic_json,
    load_jsonl,
    load_local_env,
    run_agent_records,
    sha256_json,
)


DEFAULT_PMIDS = [
    "41810002", "41799192", "41620901", "41794448", "41650163",
    "41573193", "41902413", "41581151", "41475279", "41719003",
]


def flatten_audits(records: list[dict]) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for record in records:
        phases = record.get("phases", {}) or {}
        pairwise = phases.get("pairwise_judge", {}) or {}
        for audit in pairwise.get("audits", []) or []:
            audits.append({"pmid": record.get("pmid", ""), "tool": "pairwise_judge", **audit})
        critic = ((phases.get("collaboration", {}) or {}).get("critic_audit", {}) or {})
        if critic:
            audits.append({"pmid": record.get("pmid", ""), "tool": "qwen_critic", **critic})
    return audits


def cache_report(records: list[dict]) -> dict[str, Any]:
    audits = flatten_audits(records)
    totals = Counter()
    for audit in audits:
        totals["prompt_tokens"] += int(audit.get("prompt_tokens", 0) or 0)
        totals["output_tokens"] += int(audit.get("output_tokens", 0) or 0)
        totals["provider_cache_read_tokens"] += int(audit.get("provider_cache_read_tokens", 0) or 0)
        totals["provider_cache_miss_tokens"] += int(audit.get("provider_cache_miss_tokens", 0) or 0)
        totals["provider_cache_write_tokens"] += int(audit.get("provider_cache_write_tokens", 0) or 0)
        totals["local_result_hits"] += int(bool(audit.get("local_result_hit", False)))
        totals["provider_prompt_hits"] += int(bool(audit.get("provider_prompt_hit", False)))
        totals["singleflight_shared"] += int(bool(audit.get("singleflight_shared", False)))
    remote_ok = sum(1 for audit in audits if audit.get("status") == "OK")
    return {
        "audit_count": len(audits),
        "remote_ok": remote_ok,
        "totals": dict(totals),
        "provider_cache_hit_rate": (
            round(
                totals["provider_cache_read_tokens"]
                / max(totals["provider_cache_read_tokens"] + totals["provider_cache_miss_tokens"], 1),
                6,
            )
        ),
        "audits": audits,
    }


def relation_audit(records: list[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for relation in (
            record.get("phases", {}).get("verification", {}).get("relations", []) or []
        ):
            rows.append({
                "pmid": record.get("pmid", ""),
                "subject": relation.get("subject", ""),
                "subject_type": relation.get("subject_type", ""),
                "predicate": relation.get("predicate", ""),
                "object": relation.get("object", ""),
                "object_type": relation.get("object_type", ""),
                "factual_status": relation.get("factual_status", ""),
                "semantic_status": relation.get("semantic_status", ""),
                "write_status": relation.get("write_status", ""),
                "claim_role": relation.get("claim_role", ""),
                "import_ready": bool(relation.get("import_ready", False)),
                "quality_flags": relation.get("quality_flags", []),
                "evidence": relation.get("evidence", ""),
                "evidence_spans": relation.get("evidence_spans", []),
            })
    return rows


def arm_flags(policy: str, *, frozen: Path, cache_path: Path, warm: bool = False) -> list[str]:
    flags = [
        "--execution-mode", "agent-v2-shadow",
        "--router-execution-mode", "legacy",
        "--agent-mode", "precision",
        "--disable-causal-conflict",
        "--skip-neo4j-write",
        "--verification-policy", policy,
        "--frozen-candidates", str(frozen),
        "--extraction-cache-mode", "persistent",
        "--extraction-cache-path", str(cache_path),
        "--agent-max-aux-remote-calls", "8",
        "--agent-hard-timeout", "900",
    ]
    if policy == "legacy":
        flags.extend([
            "--pair-classifier-mode", "off",
            "--relation-authority", "legacy",
            "--pairwise-judge-mode", "off",
            "--disable-qwen-critic",
        ])
    else:
        flags.extend([
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--pairwise-judge-claim-gate",
            "--second-llm-enabled",
            "--second-llm-mode", "conditional",
        ])
    return flags


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "benchmark_output/tiered_v2_sentinel10")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--pmids", default=",".join(DEFAULT_PMIDS))
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--frozen-candidates", type=Path, default=ROOT / "benchmark_output/pairwise_judge_frozen_candidates.json")
    args = parser.parse_args()

    load_local_env(args.env_file)
    has_deepseek = bool(
        os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("SECOND_LLM_API_KEY")
    )
    if not has_deepseek:
        raise RuntimeError(
            "tiered-v2 sentinel requires DEEPSEEK_API_KEY or SECOND_LLM_API_KEY "
            "for the DeepSeek adjudicator; refusing to run a fake dual-model experiment"
        )
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    pmids = [item.strip() for item in args.pmids.split(",") if item.strip()]
    source = {str(item["pmid"]): item for item in load_jsonl(ROOT / "extraction_output/pubmed_converted_500.jsonl")}
    gold = {str(item["pmid"]): item for item in load_jsonl(ROOT / "gold_annotations/pubmed_200_gold_v2_strict.jsonl")}
    missing = [pmid for pmid in pmids if pmid not in source or pmid not in gold]
    if missing:
        raise RuntimeError(f"PMIDs missing from source/gold: {missing}")
    articles = [source[pmid] for pmid in pmids]
    manifest = {
        "protocol": "tiered-v2-sentinel10-v1",
        "development_only": True,
        "pmids": pmids,
        "frozen_candidates": str(args.frozen_candidates),
        "frozen_candidates_sha256": sha256_json(json.loads(args.frozen_candidates.read_text(encoding="utf-8"))),
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    atomic_json(run_dir / "manifest.json", manifest)

    tracker = StatusTracker(run_dir, total_article_arms=len(pmids) * 3)
    tracker.start()
    try:
        common: list[str] = []
        legacy = run_agent_records(
            task_dir=run_dir / "arms/A_legacy",
            articles=articles,
            run_name="tiered_v2_A_legacy",
            tracker=tracker,
            max_workers=args.max_workers,
            common=common,
            variant=arm_flags("legacy", frozen=args.frozen_candidates, cache_path=run_dir / "cache.sqlite3"),
        )
        cold = run_agent_records(
            task_dir=run_dir / "arms/B_tiered_cold",
            articles=articles,
            run_name="tiered_v2_B_cold",
            tracker=tracker,
            max_workers=args.max_workers,
            common=common,
            variant=arm_flags("tiered-v2", frozen=args.frozen_candidates, cache_path=run_dir / "cache.sqlite3"),
        )
        warm = run_agent_records(
            task_dir=run_dir / "arms/B_tiered_warm",
            articles=articles,
            run_name="tiered_v2_B_warm",
            tracker=tracker,
            max_workers=args.max_workers,
            common=common,
            variant=arm_flags("tiered-v2", frozen=args.frozen_candidates, cache_path=run_dir / "cache.sqlite3", warm=True),
        )
        legacy_rows = [article_metric(record, gold[str(record["pmid"])], source[str(record["pmid"])]) for record in legacy]
        cold_rows = [article_metric(record, gold[str(record["pmid"])], source[str(record["pmid"])]) for record in cold]
        warm_cache = cache_report(warm)
        cold_cache = cache_report(cold)
        report = {
            "manifest_hash": manifest["manifest_hash"],
            "quality": {
                "legacy": metrics(legacy_rows),
                "tiered_v2_cold": metrics(cold_rows),
            },
            "cache": {
                "cold": cold_cache,
                "warm": warm_cache,
                "warm_local_remote_calls_zero": warm_cache["remote_ok"] == 0
                or warm_cache["totals"].get("local_result_hits", 0) >= warm_cache["remote_ok"],
            },
            "relation_audit": {
                "legacy": relation_audit(legacy),
                "tiered_v2_cold": relation_audit(cold),
                "tiered_v2_warm": relation_audit(warm),
            },
        }
        atomic_json(run_dir / "sentinel_report.json", report)
        atomic_json(run_dir / "legacy_article_metrics.json", {"articles": legacy_rows})
        atomic_json(run_dir / "tiered_v2_article_metrics.json", {"articles": cold_rows})
        tracker.close("COMPLETE")
        print(json.dumps({
            "run_dir": str(run_dir),
            "quality": report["quality"],
            "cache_summary": {
                "cold": cold_cache["totals"],
                "warm": warm_cache["totals"],
                "warm_local_remote_calls_zero": report["cache"]["warm_local_remote_calls_zero"],
            },
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        tracker.close("FAILED", str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
