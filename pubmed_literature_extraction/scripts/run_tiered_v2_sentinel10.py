#!/usr/bin/env python3
"""10-article tiered-v2 verifier sentinel with cold/warm cache replay.

Development-only.  It consumes frozen Gemini candidates, keeps Neo4j writes
disabled, compares legacy vs tiered-v2, and immediately replays the tiered arm
against the same auxiliary SQLite cache to confirm local warm-cache behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_agent_v3_experiments import metrics
from scripts.evaluate_gold200_unified import score_funnel, score_view
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
                "candidate_id": relation.get("candidate_id", ""),
                "candidate_version": relation.get("candidate_version", 1),
                "parent_version": relation.get("parent_version", 0),
                "candidate_lane": relation.get("candidate_lane", ""),
                "subject": relation.get("subject", ""),
                "subject_type": relation.get("subject_type", ""),
                "predicate": relation.get("predicate", ""),
                "object": relation.get("object", ""),
                "object_type": relation.get("object_type", ""),
                "factual_status": relation.get("factual_status", ""),
                "semantic_status": relation.get("semantic_status", ""),
                "write_status": relation.get("write_status", ""),
                "claim_role": relation.get("claim_role", ""),
                "relation_direction": relation.get("relation_direction", "UNKNOWN"),
                "association_sign": relation.get("association_sign", "UNKNOWN"),
                "expression_change": relation.get("expression_change", "UNKNOWN"),
                "activity_change": relation.get("activity_change", "UNKNOWN"),
                "import_ready": bool(relation.get("import_ready", False)),
                "quality_flags": relation.get("quality_flags", []),
                "evidence": relation.get("evidence", ""),
                "evidence_spans": relation.get("evidence_spans", []),
                "evidence_pack": relation.get("evidence_pack", {}),
                "semantic_reasons": relation.get("semantic_reasons", []),
                "write_reasons": relation.get("write_reasons", []),
                "adjudication": relation.get("adjudication", {}),
                "adjudication_verdict": relation.get("adjudication_verdict", ""),
                "adjudication_reason_code": relation.get("adjudication_reason_code", ""),
                "adjudication_confidence": relation.get("adjudication_confidence", 0.0),
                "supporting_span_ids": relation.get("supporting_span_ids", []),
                "relation_card_match": relation.get("relation_card_match", ""),
                "promotion_path": relation.get("promotion_path", ""),
                "support_trigger_match": relation.get("support_trigger_match", ""),
                "support_trigger_reason_codes": relation.get(
                    "support_trigger_reason_codes", []
                ),
                "source_lanes": relation.get("source_lanes", []),
                "type_conflict_group_id": relation.get("type_conflict_group_id", ""),
            })
    return rows


def candidate_audit(records: list[dict]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    totals = Counter()
    for record in records:
        ledger = (
            (record.get("phases", {}) or {}).get("candidate_audit_ledger", {}) or {}
        )
        for row in ledger.get("rows", []) or []:
            rows.append({"pmid": record.get("pmid", ""), **row})
        summary = ledger.get("summary", {}) or {}
        totals.update({
            "projected": int(summary.get("projected_hint_count", 0) or 0),
            "accounted": int(summary.get("accounted_projected_hint_count", 0) or 0),
            "eligible": int(summary.get("eligible_projected_hint_count", 0) or 0),
            "eligible_survived": int(summary.get("eligible_survived_count", 0) or 0),
            "semantic_accepted": int(
                summary.get("semantic_accepted_lineage_count", 0) or 0
            ),
        })
    return {
        "summary": {
            **dict(totals),
            "audit_lineage_accounting": (
                totals["accounted"] / totals["projected"]
                if totals["projected"] else None
            ),
            "eligible_lineage_survival": (
                totals["eligible_survived"] / totals["eligible"]
                if totals["eligible"] else None
            ),
            "semantic_acceptance_survival": (
                totals["semantic_accepted"] / totals["eligible"]
                if totals["eligible"] else None
            ),
            "version_accounting": (
                sum(item.get("disposition") != "UNACCOUNTED" for item in rows) / len(rows)
                if rows else 1.0
            ),
            "recovery_lineage_accounting": (
                sum(item.get("disposition") != "UNACCOUNTED" for item in rows
                    if item.get("candidate_lane") == "recovery")
                / sum(item.get("candidate_lane") == "recovery" for item in rows)
                if any(item.get("candidate_lane") == "recovery" for item in rows)
                else 1.0
            ),
            "r_prefix_lane_mismatch": sum(
                str(item.get("candidate_id", "")).startswith("r-")
                and item.get("candidate_lane") != "recovery" for item in rows
            ),
        },
        "rows": rows,
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[rank], 4)


def aggregate_remote_usage(records: list[dict]) -> dict[str, Any]:
    """Aggregate the authoritative Agent-v2 request ledger by model role."""
    role_map = {
        "pairwise_judge": "pairwise_judge",
        "second_llm_refiner": "deepseek_adjudicator",
        "qwen_edit_critic": "qwen_critic",
    }
    totals: dict[str, Counter] = {
        role: Counter() for role in (
            "pairwise_judge", "deepseek_adjudicator", "qwen_critic",
            "gemini_extraction", "llm_chunk_planner",
        )
    }
    latency_samples: dict[str, list[float]] = defaultdict(list)
    for record in records:
        phase = ((record.get("phases", {}) or {}).get("agent_v2", {}) or {})
        gateway_records = (
            (phase.get("remote_execution", {}) or {}).get("records", []) or []
        )
        if gateway_records:
            for request in gateway_records:
                if request.get("duplicate_of"):
                    continue
                role = role_map.get(request.get("tool", ""), request.get("tool", ""))
                target = totals.setdefault(role, Counter())
                target["logical_steps"] += 1
                target["attempted"] += int(request.get("physical_attempts", 0) or 0)
                target["retried"] += max(
                    0, int(request.get("physical_attempts", 0) or 0) - 1
                )
                target["successful"] += int(
                    request.get("result_status") == "OK"
                    and int(request.get("physical_attempts", 0) or 0) > 0
                )
                target["cached"] += int(bool(request.get("cache_replayed", False)))
                target["timeouts"] += int("timeout" in str(
                    request.get("result_status", "")
                ).casefold())
                if int(request.get("physical_attempts", 0) or 0):
                    latency_samples[role].append(float(request.get("latency_s", 0.0) or 0.0))
                    target["latency_s"] += float(request.get("latency_s", 0.0) or 0.0)
        for tool, usage in (() if gateway_records else (phase.get("remote_usage", {}) or {}).items()):
            if not isinstance(usage, dict):
                continue
            role = role_map.get(tool, tool)
            target = totals.setdefault(role, Counter())
            for field in (
                "attempted", "successful", "retried", "cached", "timeouts",
                "prompt_tokens", "output_tokens", "state_changes", "zero_change_calls",
            ):
                target[field] += int(usage.get(field, 0) or 0)
            target["latency_s"] += float(usage.get("latency_s", 0.0) or 0.0)
            successful = int(usage.get("successful", 0) or 0)
            if successful:
                latency_samples[role].extend([
                    float(usage.get("latency_s", 0.0) or 0.0) / successful
                ] * successful)
        primary = phase.get("primary_extraction", {}) or {}
        primary_requests = int(primary.get("remote_requests", 0) or 0)
        totals["gemini_extraction"]["attempted"] += primary_requests
        totals["gemini_extraction"]["successful"] += primary_requests
        totals["gemini_extraction"]["cached"] += int(
            primary.get("cache_avoided_requests", 0) or 0
        )
        planner = ((record.get("phases", {}) or {}).get("chunk_plan", {}) or {})
        if str(planner.get("strategy", "")).upper() == "LLM_PLANNER":
            planner_requests = int(planner.get("remote_requests", 1) or 1)
            totals["llm_chunk_planner"]["attempted"] += planner_requests
            totals["llm_chunk_planner"]["successful"] += planner_requests
        phases = record.get("phases", {}) or {}
        totals["pairwise_judge"]["invalid_json"] += sum(
            int(item.get("invalid_json_attempts", 0) or 0)
            for item in (phases.get("pairwise_judge", {}) or {}).get("audits", []) or []
        )
        collaboration = phases.get("collaboration", {}) or {}
        totals["deepseek_adjudicator"]["invalid_json"] += int(
            collaboration.get("invalid_json_attempts", 0) or 0
        )
        totals["qwen_critic"]["invalid_json"] += int(
            (collaboration.get("critic_audit", {}) or {}).get(
                "invalid_json_attempts", 0
            ) or 0
        )
    result: dict[str, Any] = {}
    for role, value in totals.items():
        attempted = int(value["attempted"])
        successful = int(value["successful"])
        standard = {
            key: int(value.get(key, 0) or 0)
            for key in (
                "attempted", "successful", "retried", "cached", "timeouts",
                "prompt_tokens", "output_tokens", "state_changes",
                "zero_change_calls", "invalid_json", "logical_steps",
            )
        }
        result[role] = {
            **standard,
            "latency_s": round(float(value.get("latency_s", 0.0) or 0.0), 4),
            "failed": max(0, attempted - successful),
            "latency_percentiles_s": {
                "p50": _percentile(latency_samples[role], 0.50),
                "p90": _percentile(latency_samples[role], 0.90),
                "p95": _percentile(latency_samples[role], 0.95),
                "p99": _percentile(latency_samples[role], 0.99),
                "basis": "per-record tool mean",
            },
        }
    return result


def remote_usage_report(
    records: list[dict], *, reused_from: list[dict] | None = None,
) -> dict[str, Any]:
    incremental = aggregate_remote_usage(records)
    effective = aggregate_remote_usage(reused_from) if reused_from is not None else incremental
    return {
        "effective": effective,
        "incremental_execution": incremental,
        "reused": {
            "document_count": len(records) if reused_from is not None else 0,
            "source": "tiered_v2_cold" if reused_from is not None else "",
            "cached_requests": sum(
                int(item.get("cached", 0) or 0) for item in incremental.values()
            ),
        },
    }


def remote_request_audit_consistent(records: list[dict]) -> dict[str, Any]:
    role_map = {
        "pairwise_judge": "pairwise_judge",
        "second_llm_refiner": "deepseek_adjudicator",
        "qwen_edit_critic": "qwen_critic",
        "langextract_candidate_generator": "gemini_extraction",
        "article_chunker": "llm_chunk_planner",
    }
    trace_counts = Counter()
    for record in records:
        phase = ((record.get("phases", {}) or {}).get("agent_v2", {}) or {})
        for action in phase.get("action_trace", []) or []:
            details = action.get("details", {}) or {}
            local_result = bool(details.get("local_result_hit", False)) or str(
                action.get("cache_status", "") or ""
            ) in {"memory_hit", "persistent_hit"}
            if action.get("remote") and not local_result:
                trace_counts[role_map.get(action.get("tool", ""), action.get("tool", ""))] += 1
    usage = aggregate_remote_usage(records)
    details = {}
    consistent = True
    for role, item in usage.items():
        attempted = int(item.get("attempted", 0) or 0)
        retried = int(item.get("retried", 0) or 0)
        audited = int(trace_counts[role])
        role_ok = attempted == audited + retried
        details[role] = {
            "attempted": attempted,
            "action_trace_requests": audited,
            "retry_attempts": retried,
            "consistent": role_ok,
        }
        consistent = consistent and role_ok
    return {"consistent": consistent, "by_role": details}


def logical_route_manifest(records: list[dict]) -> dict[str, Any]:
    by_pmid: dict[str, list[dict[str, Any]]] = {}
    physical_attempts = 0
    for record in records:
        phase = ((record.get("phases", {}) or {}).get("agent_v2", {}) or {})
        rows = []
        for item in ((phase.get("remote_execution", {}) or {}).get("records", []) or []):
            physical_attempts += int(item.get("physical_attempts", 0) or 0)
            if item.get("duplicate_of"):
                continue
            rows.append({
                "request_id": item.get("request_id", ""),
                "tool": item.get("tool", ""), "stage": item.get("stage", ""),
                "round": item.get("round", 0), "batch": item.get("batch", ""),
                "logical_step": item.get("logical_step", 0),
                "blocked_kind": item.get("blocked_kind", ""),
                "result_status": item.get("result_status", ""),
            })
        by_pmid[str(record.get("pmid", ""))] = rows
    return {
        "by_pmid": by_pmid,
        "logical_request_sequence_hash": sha256_json({
            pmid: [item["request_id"] for item in rows]
            for pmid, rows in sorted(by_pmid.items())
        }),
        "route_decision_hash": sha256_json(by_pmid),
        "physical_attempts": physical_attempts,
    }


def candidate_generation_report(records: list[dict]) -> dict[str, int]:
    totals = Counter()
    for record in records:
        phase = (
            (record.get("phases", {}) or {}).get(
                "relation_pair_classification", {}
            ) or {}
        )
        totals.update({
            "hint_candidates": int(phase.get("hint_candidate_count", 0) or 0),
            "explicit_recovery_candidates": int(
                phase.get("explicit_recovery_candidate_count", 0) or 0
            ),
            "non_explicit_filtered_pairs": int(
                phase.get("non_explicit_filtered_pair_count", 0) or 0
            ),
            "budget_truncated_pairs": int(
                phase.get("budget_truncated_pair_count", 0) or 0
            ),
        })
    return dict(totals)


def provider_failure_count(records: list[dict]) -> int:
    usage = aggregate_remote_usage(records)
    structured = sum(int(item.get("failed", 0) or 0) for item in usage.values())
    legacy = sum(
        int(
            (((record.get("phases", {}) or {}).get("agent_v2", {}) or {})
             .get("remote_usage", {}) or {}).get("failed", 0) or 0
        )
        for record in records
    )
    return structured + legacy


def neo4j_mutation_count(records: list[dict]) -> int:
    """Observed mutations must stay zero in every experimental arm."""
    total = 0
    for record in records:
        execution = (record.get("phases", {}).get("execution", {}) or {})
        actual_fields = (
            "entities_written", "relations_written", "relations_updated_in_neo4j",
        )
        if any(field in execution for field in actual_fields):
            total += sum(int(execution.get(field, 0) or 0) for field in actual_fields)
        elif not bool(execution.get("dry_run", False)):
            # Backward compatibility for genuinely mutating historical rows.
            total += sum(
                int(execution.get(field, 0) or 0)
                for field in ("entities_created", "relations_created", "relations_updated")
            )
    return total


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
        "--agent-max-physical-remote-attempts", "16",
        "--agent-hard-timeout", "900",
    ]
    if policy == "legacy":
        flags.extend([
            "--disable-second-llm",
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
        if os.environ.get("PREDICATE_THRESHOLDS"):
            flags.extend(["--predicate-thresholds", os.environ["PREDICATE_THRESHOLDS"]])
    return flags


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "benchmark_output/tiered_v2_sentinel10")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--pmids", default=",".join(DEFAULT_PMIDS))
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--pilot-count", type=int, default=3)
    parser.add_argument(
        "--pilot-only", action="store_true",
        help="Run only the first cold documents, write pilot_report.json, then stop.",
    )
    parser.add_argument(
        "--skip-legacy", action="store_true",
        help="Run only the tiered-v2 cold/warm arms for the P0 integration gate.",
    )
    parser.add_argument("--frozen-candidates", type=Path, default=ROOT / "benchmark_output/pairwise_judge_frozen_candidates.json")
    parser.add_argument(
        "--gold-path", type=Path,
        default=ROOT / "gold_annotations/pubmed_200_gold_v2_strict.jsonl",
    )
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
    gold = {str(item["pmid"]): item for item in load_jsonl(args.gold_path)}
    missing = [pmid for pmid in pmids if pmid not in source or pmid not in gold]
    if missing:
        raise RuntimeError(f"PMIDs missing from source/gold: {missing}")
    articles = [source[pmid] for pmid in pmids]
    manifest = {
        "protocol": "tiered-v2-monotonic-v5-sentinel10",
        "development_only": True,
        "pmids": pmids,
        "frozen_candidates": str(args.frozen_candidates),
        "frozen_candidates_sha256": sha256_json(json.loads(args.frozen_candidates.read_text(encoding="utf-8"))),
        "gold_path": str(args.gold_path.resolve()),
        "gold_sha256": sha256_json(load_jsonl(args.gold_path)),
        "gold_role": "development_only",
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    atomic_json(run_dir / "manifest.json", manifest)

    arm_count = 2 if args.skip_legacy else 3
    tracker = StatusTracker(
        run_dir,
        total_article_arms=(min(len(pmids), args.pilot_count) if args.pilot_only
                            else len(pmids) * arm_count),
    )
    tracker.start()
    try:
        common: list[str] = []
        legacy = [] if args.skip_legacy else run_agent_records(
            task_dir=run_dir / "arms/A_legacy",
            articles=articles,
            run_name="tiered_v2_A_legacy",
            tracker=tracker,
            max_workers=args.max_workers,
            common=common,
            variant=arm_flags("legacy", frozen=args.frozen_candidates, cache_path=run_dir / "cache.sqlite3"),
        )
        cold_articles = articles[:max(1, args.pilot_count)] if args.pilot_only else articles
        cold = run_agent_records(
            task_dir=run_dir / "arms/B_tiered_cold",
            articles=cold_articles,
            run_name="tiered_v2_B_cold",
            tracker=tracker,
            max_workers=args.max_workers,
            common=common,
            variant=arm_flags("tiered-v2", frozen=args.frozen_candidates, cache_path=run_dir / "cache.sqlite3"),
        )
        if args.pilot_only:
            ledger = candidate_audit(cold)
            rows = relation_audit(cold)
            mean_seconds = sum(
                float((item.get("timing", {}) or {}).get("total_s", 0.0) or 0.0)
                for item in cold
            ) / max(len(cold), 1)
            remaining = max(0, len(articles) - len(cold))
            pilot = {
                "status": "PILOT_COMPLETE",
                "documents": len(cold),
                "pmids": [str(item.get("pmid", "")) for item in cold],
                "candidate_audit": ledger,
                "background_auto_accept": sum(
                    str(item.get("claim_role", "CURRENT_FINDING")).upper()
                    in {"BACKGROUND", "METHOD", "PREDICTION", "PRIOR_WORK", "SPECULATIVE", "OTHER"}
                    and str(item.get("semantic_status", "")).upper() == "ACCEPTED"
                    for item in rows
                ),
                "self_contained_stale_flags": sum(
                    str((item.get("evidence_pack", {}) or {}).get(
                        "support_mode", ""
                    )).upper() == "SELF_CONTAINED"
                    and bool({
                        "cross_sentence", "coreference_only_support",
                        "multi_span_support", "trigger_not_linking_endpoints",
                    } & set(item.get("quality_flags", []) or []))
                    for item in rows
                ),
                "provider_failures": provider_failure_count(cold),
                "neo4j_observed_mutations": neo4j_mutation_count(cold),
                "remote_usage": remote_usage_report(cold),
                "remote_request_audit": remote_request_audit_consistent(cold),
                "candidate_generation": candidate_generation_report(cold),
                "mean_document_seconds": round(mean_seconds, 3),
                "estimated_remaining_seconds": round(
                    mean_seconds * remaining / max(args.max_workers, 1), 1
                ),
                "resume_command_hint": (
                    "rerun this command without --pilot-only using the same --run-dir"
                ),
            }
            pilot["pilot_valid"] = bool(
                ledger["summary"].get("audit_lineage_accounting") == 1
                and ledger["summary"].get("eligible_lineage_survival") == 1
                and pilot["background_auto_accept"] == 0
                and pilot["self_contained_stale_flags"] == 0
                and pilot["provider_failures"] == 0
                and pilot["neo4j_observed_mutations"] == 0
                and pilot["remote_request_audit"]["consistent"]
            )
            atomic_json(run_dir / "pilot_report.json", pilot)
            tracker.close("PILOT_COMPLETE")
            print(json.dumps(pilot, ensure_ascii=False, indent=2))
            return 0
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
        unified_gold_rows = load_jsonl(
            ROOT / "gold_annotations/pubmed_200_gold_v2_candidate_view.jsonl"
        )
        unified_gold = {
            str(item["pmid"]): item for item in unified_gold_rows
            if str(item["pmid"]) in set(pmids)
        }
        unified_source = {pmid: source[pmid] for pmid in pmids}
        cold_funnel = score_funnel(cold, unified_gold, unified_source)
        warm_funnel = score_funnel(warm, unified_gold, unified_source)
        cold_unified = score_view(
            cold, unified_gold, unified_source, "candidate_semantic"
        )
        cold_ledger = candidate_audit(cold)
        warm_ledger = candidate_audit(warm)
        cold_state_hash = sha256_json(relation_audit(cold))
        warm_state_hash = sha256_json(relation_audit(warm))
        finalizer_audits = [
            (
                record.get("phases", {}).get("collaboration", {})
                .get("post_action_finalization", {}) or {}
            )
            for record in [*cold, *warm]
        ]
        post_action_rollback = sum(
            int(item.get("rolled_back_count", 0) or 0) for item in finalizer_audits
        )
        soft_flag_hard_reject = sum(
            int(item.get("soft_flag_hard_reject_count", 0) or 0)
            for item in finalizer_audits
        )
        warm_not_above_cold = warm_cache["remote_ok"] <= cold_cache["remote_ok"]
        provider_failures = {
            "legacy": provider_failure_count(legacy),
            "tiered_v2_cold": provider_failure_count(cold),
            "tiered_v2_warm": provider_failure_count(warm),
        }
        neo4j_mutations = {
            "legacy": neo4j_mutation_count(legacy),
            "tiered_v2_cold": neo4j_mutation_count(cold),
            "tiered_v2_warm": neo4j_mutation_count(warm),
        }
        background_auto_accept = sum(
            str(row.get("claim_role", "CURRENT_FINDING")).upper()
            in {"BACKGROUND", "METHOD", "PREDICTION", "PRIOR_WORK", "SPECULATIVE", "OTHER"}
            and str(row.get("semantic_status", "")).upper() == "ACCEPTED"
            for row in relation_audit(cold)
        )
        stale_self_contained = sum(
            str((row.get("evidence_pack", {}) or {}).get("support_mode", "")).upper()
            == "SELF_CONTAINED"
            and bool({"cross_sentence", "coreference_only_support", "multi_span_support",
                      "trigger_not_linking_endpoints"}
                     & set(row.get("quality_flags", []) or []))
            for row in relation_audit(cold)
        )
        calibration_without_contract = sum(
            str((phase := ((record.get("phases", {}) or {}).get(
                "conformal_risk_router", {}
            ) or {})).get("mode", "off")) == "active"
            and int(phase.get("calibration_size", 0) or 0) == 0
            for record in cold
        )
        accepted_review_retains_raw = (
            int(cold_funnel["accepted_review"]["tp"] or 0)
            >= int(cold_funnel["raw_hint"]["tp"] or 0)
        )
        cold_request_audit = remote_request_audit_consistent(cold)
        warm_request_audit = remote_request_audit_consistent(warm)
        cold_route = logical_route_manifest(cold)
        warm_route = logical_route_manifest(warm)
        filtered_auto_accept = sum(
            bool({"filtered_endpoint", "subject_filtered", "object_filtered"}
                 & set(row.get("quality_flags", []) or []))
            and str(row.get("semantic_status", "")).upper() == "ACCEPTED"
            for row in relation_audit(cold)
        )
        invalid_promotion_accept = sum(
            str(row.get("semantic_status", "")).upper() == "ACCEPTED"
            and str(row.get("promotion_path", "")) not in {"DETERMINISTIC", "ADJUDICATED"}
            for row in relation_audit(cold)
        )
        report = {
            "manifest_hash": manifest["manifest_hash"],
            "quality": {
                "legacy": metrics(legacy_rows) if legacy_rows else {
                    "status": "SKIPPED_FOR_P0",
                },
                "tiered_v2_cold": metrics(cold_rows),
                "tiered_v2_cold_unified": cold_unified,
                "tiered_v2_cold_stage_funnel": cold_funnel,
                "tiered_v2_warm_stage_funnel": warm_funnel,
            },
            "remote_usage": {
                "cold": remote_usage_report(cold),
                "warm": remote_usage_report(warm, reused_from=cold),
                "request_audit": {
                    "cold": cold_request_audit,
                    "warm": warm_request_audit,
                },
                "logical_route": {"cold": cold_route, "warm": warm_route},
            },
            "candidate_generation": {
                "cold": candidate_generation_report(cold),
                "warm": candidate_generation_report(warm),
            },
            "cache": {
                "cold": cold_cache,
                "warm": warm_cache,
                "warm_local_remote_calls_zero": warm_cache["remote_ok"] == 0
                or warm_cache["totals"].get("local_result_hits", 0) >= warm_cache["remote_ok"],
                "warm_remote_calls_not_above_cold": warm_not_above_cold,
            },
            "invariants": {
                "cold_warm_relation_state_equal": cold_state_hash == warm_state_hash,
                "cold_relation_state_hash": cold_state_hash,
                "warm_relation_state_hash": warm_state_hash,
                "post_action_rollback": post_action_rollback,
                "soft_flag_hard_reject": soft_flag_hard_reject,
                "neo4j_write_enabled": False,
                "neo4j_observed_mutations": neo4j_mutations,
                "provider_failures": provider_failures,
                "audit_lineage_accounting": cold_ledger["summary"].get(
                    "audit_lineage_accounting"
                ),
                "eligible_lineage_survival": cold_ledger["summary"].get(
                    "eligible_lineage_survival"
                ),
                "version_accounting": cold_ledger["summary"].get("version_accounting"),
                "recovery_lineage_accounting": cold_ledger["summary"].get(
                    "recovery_lineage_accounting"
                ),
                "r_prefix_lane_mismatch": cold_ledger["summary"].get(
                    "r_prefix_lane_mismatch"
                ),
                "background_auto_accept": background_auto_accept,
                "filtered_endpoint_auto_accept": filtered_auto_accept,
                "invalid_promotion_path_accept": invalid_promotion_accept,
                "self_contained_stale_flags": stale_self_contained,
                "accepted_review_canonical_tp_not_below_raw": accepted_review_retains_raw,
                "semantic_precision_not_below_factual": bool(
                    cold_funnel["semantic_accepted"]["precision"] is not None
                    and cold_funnel["factual_valid"]["precision"] is not None
                    and cold_funnel["semantic_accepted"]["precision"]
                    >= cold_funnel["factual_valid"]["precision"]
                ),
                "cold_warm_logical_sequence_equal": (
                    cold_route["logical_request_sequence_hash"]
                    == warm_route["logical_request_sequence_hash"]
                ),
                "cold_warm_route_decision_equal": (
                    cold_route["route_decision_hash"] == warm_route["route_decision_hash"]
                ),
                "warm_physical_attempts": warm_route["physical_attempts"],
                "conformal_acceptance_without_calibration": calibration_without_contract,
                "remote_usage_request_audit_consistent": bool(
                    cold_request_audit["consistent"]
                    and warm_request_audit["consistent"]
                ),
            },
            "candidate_audit_ledger": {
                "tiered_v2_cold": cold_ledger,
                "tiered_v2_warm": warm_ledger,
            },
            "relation_audit": {
                "legacy": relation_audit(legacy),
                "tiered_v2_cold": relation_audit(cold),
                "tiered_v2_warm": relation_audit(warm),
            },
        }
        report["evaluation_valid"] = bool(
            cold_state_hash == warm_state_hash
            and warm_not_above_cold
            and post_action_rollback == 0
            and soft_flag_hard_reject == 0
            and not any(provider_failures.values())
            and not any(neo4j_mutations.values())
            and cold_ledger["summary"].get("audit_lineage_accounting") == 1
            and cold_ledger["summary"].get("eligible_lineage_survival") == 1
            and cold_ledger["summary"].get("version_accounting") == 1
            and cold_ledger["summary"].get("recovery_lineage_accounting") == 1
            and cold_ledger["summary"].get("r_prefix_lane_mismatch") == 0
            and background_auto_accept == 0
            and filtered_auto_accept == 0
            and invalid_promotion_accept == 0
            and stale_self_contained == 0
            and accepted_review_retains_raw
            and report["invariants"]["semantic_precision_not_below_factual"]
            and report["invariants"]["cold_warm_logical_sequence_equal"]
            and report["invariants"]["cold_warm_route_decision_equal"]
            and warm_route["physical_attempts"] == 0
            and calibration_without_contract == 0
            and cold_request_audit["consistent"]
            and warm_request_audit["consistent"]
            and sum(
                int(item.get("attempted", 0) or 0)
                for item in report["remote_usage"]["warm"][
                    "incremental_execution"
                ].values()
            ) == 0
        )
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
