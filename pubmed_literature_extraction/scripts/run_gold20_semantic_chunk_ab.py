#!/usr/bin/env python3
"""Paired Gold20 live extraction: one-shot versus adaptive source windows."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.aux_model_registry import AuxModelRegistry, AuxModelSpec
from cognitive_agent.article_chunker import ArticleChunker
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.provider_errors import (
    AUTH_ERROR,
    QUOTA_OR_TOKEN_EXHAUSTED,
    classify_provider_error,
)
from cognitive_agent.semantic_chunker import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    SemanticChunkPlanError,
    build_user_prompt,
    parent_sentences,
    plan_from_payload,
    schema_hint,
    source_sha256,
)
from liverkg_cli.config import load_config
from scripts.evaluate_agent_v3_experiments import metrics
from scripts.evaluate_gold200_unified import GOLD_VIEWS, load_jsonl, score_funnel, score_view
from scripts.run_agent_v3_ablation100 import (
    StatusTracker,
    article_metric,
    atomic_json,
    load_local_env,
    relation_map,
    run_agent_records,
    sha256_json,
    verification_relations,
    write_jsonl,
)
from scripts.run_gold200_biored_suite import preflight as provider_preflight


SEED = 20260829
PROTOCOL = "gold20-adaptive-chunk-ab-v2"
GOLD_PATH = ROOT / "gold_annotations/pubmed_200_gold_v2_strict.jsonl"
SOURCE_PATH = ROOT / "extraction_output/pubmed_converted_500.jsonl"
SPLIT_MANIFEST_PATH = ROOT / "gold_annotations/splits/agent_v3_dev_manifest_seed20260814.json"
QUOTAS = ((2, 3), (3, 2), (2, 3), (3, 2))  # positive, zero by length quartile


def article_text(article: dict[str, Any]) -> str:
    return f"TITLE: {article.get('title', '')}\nABSTRACT: {article.get('abstract', '')}"


def stable_rank(pmid: str) -> str:
    return hashlib.sha256(f"{SEED}:{pmid}".encode("utf-8")).hexdigest()


def select_gold20(
    gold_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold = {str(item["pmid"]): item for item in gold_rows}
    source = {str(item["pmid"]): item for item in source_rows}
    candidates = []
    for pmid, item in gold.items():
        if pmid not in source:
            raise ValueError(f"source text missing PMID {pmid}")
        length = len(article_text(source[pmid]))
        candidates.append({
            "pmid": pmid,
            "char_count": length,
            "positive": bool(item.get("relations")),
            "gold_relation_count": len(item.get("relations", []) or []),
        })
    cuts = statistics.quantiles(
        [item["char_count"] for item in candidates], n=4, method="inclusive",
    )

    def quartile(length: int) -> int:
        if length <= cuts[0]:
            return 1
        if length <= cuts[1]:
            return 2
        if length <= cuts[2]:
            return 3
        return 4

    selected: list[dict[str, Any]] = []
    for stratum, (positive_quota, zero_quota) in enumerate(QUOTAS, 1):
        pool = [item for item in candidates if quartile(item["char_count"]) == stratum]
        for positive, quota in ((True, positive_quota), (False, zero_quota)):
            eligible = sorted(
                (item for item in pool if item["positive"] is positive),
                key=lambda item: stable_rank(item["pmid"]),
            )
            if len(eligible) < quota:
                raise ValueError(f"length stratum {stratum} lacks requested class quota")
            for item in eligible[:quota]:
                selected.append({**item, "length_stratum": f"Q{stratum}"})
    selected.sort(key=lambda item: (item["length_stratum"], stable_rank(item["pmid"])))
    if len(selected) != 20 or len({item["pmid"] for item in selected}) != 20:
        raise ValueError("Gold20 selection must contain 20 unique articles")
    if sum(item["positive"] for item in selected) != 10:
        raise ValueError("Gold20 selection must contain ten positive documents")
    manifest = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "gold_path": str(GOLD_PATH),
        "gold_sha256": hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest(),
        "source_path": str(SOURCE_PATH),
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "quartile_cut_points": cuts,
        "quotas_positive_zero": [list(item) for item in QUOTAS],
        "selection": selected,
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    return [source[item["pmid"]] for item in selected], manifest


def acquire_lock(run_dir: Path):
    lock_path = ROOT / ".cache/liverkg_suites/.gold20_semantic_chunk_ab.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        handle.close()
        raise RuntimeError(f"another Gold20 chunk A/B is active (pid={owner})") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()} {run_dir}\n")
    handle.flush()
    return handle


def robust_provider_preflight(cfg) -> dict[str, Any]:
    """Retry the complete canary because compatible proxies can return empty JSON once."""
    started = time.perf_counter()
    last_error = ""
    for attempt in range(1, 6):
        try:
            result = provider_preflight(cfg)
            return {
                **result,
                "attempts": attempt,
                "elapsed_s": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:
            last_error = str(exc)
            category = classify_provider_error(last_error)
            if category in {AUTH_ERROR, QUOTA_OR_TOKEN_EXHAUSTED} or attempt >= 5:
                raise RuntimeError(
                    f"provider preflight failed ({category}) after {attempt} attempts: {last_error}"
                ) from exc
            time.sleep(min(45.0, 2.0 ** attempt))
    raise AssertionError("unreachable provider preflight retry state")


def configure_experiment_critic() -> dict[str, Any]:
    """Use a Qwen model on the reachable experiment proxy without editing .env."""
    original = {
        "model_id": os.environ.get("AUX_CRITIC_MODEL", "qwen3.6-flash"),
        "api_base": os.environ.get("ALIYUN_MAAS_API_BASE", ""),
    }
    selected_model = os.environ.get("GOLD20_QWEN_MODEL", "qwen3.6-plus")
    selected_base = os.environ.get("DEEPSEEK_API_BASE", "")
    selected_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not selected_base or not selected_key:
        raise RuntimeError("reachable proxy credentials are unavailable for Qwen critic")
    os.environ["AUX_CRITIC_MODEL"] = selected_model
    os.environ["ALIYUN_MAAS_API_BASE"] = selected_base
    os.environ["ALIYUN_MAAS_API_KEY"] = selected_key
    return {
        "reason": "configured Aliyun endpoint failed TLS connectivity; use listed Qwen model on reachable proxy",
        "original": original,
        "selected": {"model_id": selected_model, "api_base": selected_base},
        "environment_only": True,
    }


class ExperimentTracker(StatusTracker):
    def __init__(self, run_dir: Path):
        super().__init__(run_dir, total_article_arms=40)
        self.planned = 0
        self.preflight: dict[str, Any] = {}

    def write(self, final_status: str = "RUNNING", error: str = "") -> None:
        with self.write_lock:
            done = self.completed()
            elapsed = max(time.time() - self.started, 0.001)
            rate = done / elapsed
            eta = (self.total - done) / rate if rate > 0 else None
            atomic_json(self.run_dir / "suite_status.json", {
                "status": final_status,
                "pid": os.getpid(),
                "heartbeat_epoch": time.time(),
                "current": self.current,
                "semantic_plans_completed": self.planned,
                "semantic_plans_total": 20,
                "completed_article_arms": done,
                "total_article_arms": self.total,
                "progress": round(done / self.total, 4),
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(eta, 1) if eta is not None else None,
                "preflight": self.preflight,
                "neo4j_write_enabled": False,
                "error": error,
            })


def plan_one_article(
    article: dict[str, Any], plan_dir: Path, spec: AuxModelSpec,
) -> dict[str, Any]:
    pmid = str(article["pmid"])
    text = article_text(article)
    item_path = plan_dir / "items" / f"{pmid}.json"
    if item_path.exists():
        cached = json.loads(item_path.read_text(encoding="utf-8"))
        if cached.get("status") == "OK" and cached.get("source_sha256") == source_sha256(text):
            return cached
    reader = ArticleEvidenceReader()
    units = reader.read(text)
    sentences = parent_sentences(text, units)
    user_prompt = build_user_prompt(
        pmid=pmid, title=str(article.get("title", "")), sentences=sentences,
    )
    previous_error = ""
    started = time.perf_counter()
    for attempt in range(1, 6):
        prompt = user_prompt
        if previous_error:
            prompt += "\nPrevious response was invalid. Correct it without changing sentence IDs: " + previous_error[:240]
        try:
            payload, usage = AuxModelRegistry._openai_call(
                spec,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                schema_hint=schema_hint(),
            )
            plan = plan_from_payload(text, units, payload)
            record = {
                "pmid": pmid,
                "status": "OK",
                "model_id": spec.model_id,
                "api_base": spec.api_base,
                "attempts": attempt,
                "latency_s": round(time.perf_counter() - started, 4),
                **plan.to_dict(),
                "usage": usage,
            }
            atomic_json(item_path, record)
            return record
        except Exception as exc:
            previous_error = str(exc).replace(spec.api_key, "[REDACTED]")
            category = classify_provider_error(previous_error)
            if category in {AUTH_ERROR, QUOTA_OR_TOKEN_EXHAUSTED} or attempt >= 5:
                record = {
                    "pmid": pmid,
                    "status": "FAILED",
                    "model_id": spec.model_id,
                    "attempts": attempt,
                    "latency_s": round(time.perf_counter() - started, 4),
                    "source_sha256": source_sha256(text),
                    "failure_category": category,
                    "error": previous_error[:500],
                }
                atomic_json(item_path, record)
                raise RuntimeError(f"semantic plan failed for PMID {pmid}: {previous_error}") from exc
            delay = min(45.0, 2.0 ** attempt)
            time.sleep(delay)
    raise AssertionError("unreachable semantic chunk retry state")


def create_chunk_manifest(
    articles: list[dict[str, Any]], run_dir: Path, tracker: ExperimentTracker,
) -> Path:
    plan_dir = run_dir / "semantic_plans"
    spec = AuxModelSpec(
        role="primary",
        provider="openai",
        model_id=os.environ.get("AUX_PRIMARY_MODEL", "deepseek-v4-flash"),
        api_base=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        timeout_s=90,
        max_retries=0,
    )
    records: dict[str, dict[str, Any]] = {}
    reader = ArticleEvidenceReader()
    eligible: list[dict[str, Any]] = []
    for article in articles:
        text = article_text(article)
        units = reader.read(text)
        if ArticleChunker.should_use_llm_planner(text, units):
            eligible.append(article)
            continue
        chunks = ArticleChunker().build(text, units)
        pmid = str(article["pmid"])
        records[pmid] = {
            "pmid": pmid,
            "status": "NOT_ELIGIBLE",
            "strategy": chunks[0].strategy,
            "source_sha256": source_sha256(text),
            "attempts": 0,
            "latency_s": 0.0,
            "chunks": [item.to_dict() for item in chunks],
            "usage": {},
        }
    if eligible and not spec.configured:
        raise RuntimeError("DeepSeek semantic chunk planner is not configured")
    tracker.set("semantic_chunk_planning")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(plan_one_article, item, plan_dir, spec): item for item in eligible}
        for future in as_completed(futures):
            record = future.result()
            records[str(record["pmid"])] = record
            tracker.planned = len(records)
            tracker.write()
    tracker.planned = len(records)
    tracker.write()
    manifest = {
        "protocol": PROTOCOL,
        "prompt_version": PROMPT_VERSION,
        "model_id": spec.model_id,
        "created_at_epoch": time.time(),
        "articles": {pmid: records[pmid] for pmid in sorted(records)},
    }
    manifest["manifest_hash"] = sha256_json({k: v for k, v in manifest.items() if k != "created_at_epoch"})
    path = plan_dir / "semantic_chunk_manifest.json"
    atomic_json(path, manifest)
    return path


def common_agent_args(cfg) -> list[str]:
    args = [
        "--execution-mode", "agent-v2",
        "--agent-budget-profile", "quality",
        "--agent-mode", "precision",
        "--agent-max-aux-remote-calls", "6",
        "--agent-soft-timeout", "420",
        "--agent-hard-timeout", "900",
        "--rule-memory-mode", "shadow",
        "--evidence-entailment-mode", "active",
        "--risk-router-mode", "off",
        "--pair-classifier-mode", "active",
        "--relation-authority", "unified-active",
        "--verification-policy", "tiered-v2",
        "--pairwise-judge-mode", "active",
        "--pairwise-judge-model", cfg.second_llm_model_id,
        "--pairwise-judge-max-pairs", "24",
        "--pairwise-judge-max-calls", "1",
        "--pairwise-judge-min-confidence", "0.70",
        "--second-llm-enabled",
        "--second-llm-mode", "conditional",
        "--second-llm-model-id", cfg.second_llm_model_id,
        "--second-llm-api-base", cfg.second_llm_api_base,
        "--second-llm-timeout", "120",
        "--golden-shot-max-examples", "4",
        "--disable-causal-conflict",
    ]
    if cfg.rule_bundle:
        args.extend(["--rule-bundle", cfg.rule_bundle])
    if cfg.conformal_calibration:
        args.extend(["--conformal-calibration", cfg.conformal_calibration])
    if os.environ.get("PREDICATE_THRESHOLDS"):
        args.extend(["--predicate-thresholds", os.environ["PREDICATE_THRESHOLDS"]])
    return args


def raw_article_metric(record: dict[str, Any], gold: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    raw = copy.deepcopy(record)
    extraction = raw.get("phases", {}).get("extraction", {}) or {}
    raw.setdefault("phases", {})["verification"] = {
        "entities": extraction.get("entities", []) or [],
        "relations": [
            {**item, "semantic_status": "ACCEPTED"}
            for item in extraction.get("relations", []) or []
        ],
    }
    return article_metric(raw, gold, source)


def _latency_percentiles(values: list[float]) -> dict[str, float | None]:
    ordered = sorted(float(item) for item in values if float(item) >= 0)
    if not ordered:
        return {key: None for key in ("p50", "p90", "p95", "p99")}
    return {
        f"p{percent}": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * percent / 100) - 1)]
        for percent in (50, 90, 95, 99)
    }


def _usage_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    model_calls = Counter()
    role_calls = Counter()
    role_latencies: dict[str, list[float]] = defaultdict(list)
    failures = Counter()
    for record in records:
        extraction = record.get("phases", {}).get("extraction", {}) or {}
        totals["gemini_requests_estimated"] += int(extraction.get("chunk_count", 1) or 1) + int(extraction.get("retry_count", 0) or 0)
        role_calls["gemini_extraction"] += int(extraction.get("chunk_count", 1) or 1) + int(extraction.get("retry_count", 0) or 0)
        totals["extraction_retries"] += int(extraction.get("retry_count", 0) or 0)
        pairwise = record.get("phases", {}).get("pairwise_judge", {}) or {}
        for audit in pairwise.get("audits", []) or []:
            totals["remote_requests"] += 1
            if audit.get("model_id"):
                model_calls[str(audit.get("model_id"))] += 1
            role_calls["deepseek_pairwise_judge"] += 1
            if audit.get("latency_s") is not None:
                role_latencies["deepseek_pairwise_judge"].append(float(audit.get("latency_s", 0) or 0))
            for key in ("prompt_tokens", "output_tokens", "provider_cache_read_tokens", "provider_cache_miss_tokens"):
                totals[key] += int(audit.get(key, 0) or 0)
            if str(audit.get("status")) not in {"OK", "SKIPPED"}:
                failures[str(audit.get("status") or "UNKNOWN")] += 1
        collaboration = record.get("phases", {}).get("collaboration", {}) or {}
        if collaboration.get("triggered"):
            totals["remote_requests"] += 1
            model_calls[str(collaboration.get("model_id") or "second_llm")] += 1
            role_calls["deepseek_adjudicator"] += 1
            if collaboration.get("latency_s") is not None:
                role_latencies["deepseek_adjudicator"].append(float(collaboration.get("latency_s", 0) or 0))
            totals["prompt_tokens"] += int(collaboration.get("prompt_tokens", 0) or 0)
            totals["output_tokens"] += int(collaboration.get("output_tokens", 0) or 0)
            totals["retries"] += sum(
                int(item.get("retry_count", 0) or 0)
                for item in collaboration.get("rounds", []) or []
            )
        collaboration_status = str(collaboration.get("status", "") or "")
        if collaboration_status not in {
            "", "OK", "NOT_TRIGGERED", "SKIPPED_BY_ROUTER", "DISABLED",
        }:
            failures[collaboration_status] += 1
        if any("json" in str(item).casefold() for item in collaboration.get("warnings", []) or []):
            totals["invalid_json"] += 1
        evidence = record.get("phases", {}).get("evidence_entailment", {}).get("audit", {}) or {}
        for role in ("deepseek", "qwen"):
            audit = evidence.get(role, {}) or {}
            if audit.get("model_id"):
                totals["remote_requests"] += 1
                model_calls[str(audit.get("model_id"))] += 1
                role_name = f"{role}_critic" if role == "qwen" else "deepseek_entailment"
                role_calls[role_name] += 1
                if audit.get("latency_s") is not None:
                    role_latencies[role_name].append(float(audit.get("latency_s", 0) or 0))
            for key in ("prompt_tokens", "output_tokens", "provider_cache_read_tokens", "provider_cache_miss_tokens"):
                totals[key] += int(audit.get(key, 0) or 0)
            if audit and str(audit.get("status")) not in {"OK", "SKIPPED", ""}:
                failures[str(audit.get("status"))] += 1
    return {
        **dict(totals),
        "model_calls": dict(model_calls),
        "requests_by_role": dict(role_calls),
        "latency_seconds_by_role": {
            role: _latency_percentiles(values)
            for role, values in sorted(role_latencies.items())
        },
        "failure_statuses": dict(failures),
        "gemini_usage": "usage_unavailable" if records else "not_run",
    }


def arm_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    executed = [
        record for record in records
        if not bool((record.get("phases", {}).get("paired_design", {}) or {}).get("extraction_reused"))
    ]
    reused = [record for record in records if record not in executed]
    effective = _usage_totals(records)
    incremental = _usage_totals(executed)
    return {
        **effective,
        "effective_arm": effective,
        "incremental_execution": incremental,
        "reused": {
            "document_count": len(reused),
            "source_arm": "A_one_shot" if reused else "",
            "pmids": [str(item.get("pmid", "")) for item in reused],
        },
        "executed_document_count": len(executed),
    }


def bootstrap_f1_difference(a_rows: list[dict], b_rows: list[dict], *, samples: int = 5000) -> dict[str, Any]:
    def f1(rows: list[dict], indexes: list[int]) -> float:
        tp = sum(int(rows[i].get("tp", 0) or 0) for i in indexes)
        fp = sum(int(rows[i].get("fp", 0) or 0) for i in indexes)
        fn = sum(int(rows[i].get("fn", 0) or 0) for i in indexes)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if len(a_rows) != len(b_rows) or not a_rows:
        return {"samples": 0, "error": "paired rows unavailable"}
    rng = random.Random(SEED + 31)
    indexes = list(range(len(a_rows)))
    a_predictions = sum(int(item.get("tp", 0) or 0) + int(item.get("fp", 0) or 0) for item in a_rows)
    b_predictions = sum(int(item.get("tp", 0) or 0) + int(item.get("fp", 0) or 0) for item in b_rows)
    observed = f1(b_rows, indexes) - f1(a_rows, indexes)
    deltas = []
    for _ in range(samples):
        draw = [rng.randrange(len(indexes)) for _ in indexes]
        deltas.append(f1(b_rows, draw) - f1(a_rows, draw))
    deltas.sort()
    return {
        "samples": samples,
        "observed_b_minus_a": observed,
        "percentile_95": [deltas[int(samples * 0.025)], deltas[min(samples - 1, int(samples * 0.975))]],
        "development_sample_only": True,
        "comparison_valid": bool(a_predictions or b_predictions),
        "invalid_reason": "both_arms_have_zero_predictions" if not (a_predictions or b_predictions) else "",
    }


def build_report(
    run_dir: Path, sample_manifest: dict[str, Any], chunk_manifest_path: Path,
    arm_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    selected_ids = [item["pmid"] for item in sample_manifest["selection"]]
    gold_rows = load_jsonl(GOLD_PATH)
    source_rows = load_jsonl(SOURCE_PATH)
    gold = {str(item["pmid"]): item for item in gold_rows if str(item["pmid"]) in selected_ids}
    source = {str(item["pmid"]): item for item in source_rows if str(item["pmid"]) in selected_ids}
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "development_only": True,
        "sample_manifest_hash": sample_manifest["manifest_hash"],
        "semantic_chunk_manifest": str(chunk_manifest_path),
        "arms": {},
    }
    chunk_manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    chunk_totals = Counter()
    for record in chunk_manifest.get("articles", {}).values():
        usage = record.get("usage", {}) or {}
        chunk_totals["requests"] += int(record.get("attempts", 0) or 0)
        chunk_totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        chunk_totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        chunk_totals["provider_cache_read_tokens"] += int(
            usage.get("provider_cache_read_tokens", 0) or 0
        )
        chunk_totals["provider_cache_miss_tokens"] += int(
            usage.get("provider_cache_miss_tokens", 0) or 0
        )
        chunk_totals["latency_milliseconds"] += int(
            1000 * float(record.get("latency_s", 0.0) or 0.0)
        )
    report["semantic_chunk_planner_usage"] = {
        **dict(chunk_totals),
        "model_id": chunk_manifest.get("model_id", ""),
        "eligible_article_count": sum(
            item.get("status") == "OK"
            for item in chunk_manifest.get("articles", {}).values()
        ),
        "ineligible_article_count": sum(
            item.get("status") == "NOT_ELIGIBLE"
            for item in chunk_manifest.get("articles", {}).values()
        ),
        "latency_s": round(chunk_totals["latency_milliseconds"] / 1000, 3),
    }
    article_rows: dict[str, list[dict]] = {}
    stage_article_rows: dict[str, dict[str, list[dict]]] = {}
    for arm, records in arm_records.items():
        records = sorted(records, key=lambda item: selected_ids.index(str(item.get("pmid", ""))))
        arm_records[arm] = records
        rows = [article_metric(record, gold[str(record["pmid"])], source[str(record["pmid"])]) for record in records]
        raw_rows = [raw_article_metric(record, gold[str(record["pmid"])], source[str(record["pmid"])]) for record in records]
        article_rows[arm] = rows
        stage_article_rows[arm] = {stage: [] for stage in (
            "raw_hint", "factual_valid", "semantic_accepted", "accepted_review", "import_ready",
        )}
        for record in records:
            pmid = str(record["pmid"])
            per_article = score_funnel([record], {pmid: gold[pmid]}, {pmid: source[pmid]})
            for stage in stage_article_rows[arm]:
                stage_article_rows[arm][stage].append(per_article[stage])
        views = {}
        for view, path in GOLD_VIEWS.items():
            view_gold = {
                str(item["pmid"]): item for item in load_jsonl(path)
                if str(item["pmid"]) in selected_ids
            }
            views[view] = {
                "metrics": score_view(records, view_gold, source, view),
                "review_inclusive": score_view(records, view_gold, source, view, include_review=True),
            }
            if view == "candidate_semantic":
                views[view]["stage_funnel"] = score_funnel(records, view_gold, source)
        report["arms"][arm] = {
            "record_count": len(records),
            "raw_langextract": metrics(raw_rows),
            "final_agent": metrics(rows),
            "unified_views": views,
            "usage": arm_usage(records),
            "chunking_strategy_counts": dict(Counter(
                str((record.get("phases", {}).get("reader", {}).get("chunking", {}) or {}).get("strategy", "unknown"))
                for record in records
            )),
        }
        atomic_json(run_dir / "metrics" / f"{arm}_article_metrics.json", {"articles": rows})

    all_records = [record for records in arm_records.values() for record in records]
    finalizer_audits = [
        record.get("phases", {}).get("collaboration", {}).get(
            "post_action_finalization", {}
        ) or {}
        for record in all_records
    ]
    pair_cap_violations = []
    for record in all_records:
        phases = record.get("phases", {}) or {}
        projected = (phases.get("relation_candidate_projection", {}) or {}).get("relations", []) or []
        hinted_pairs = {
            (
                str(item.get("subject", "")).casefold(),
                str(item.get("subject_type", "")),
                str(item.get("object", "")).casefold(),
                str(item.get("object_type", "")),
            )
            for item in projected
        }
        candidate_count = int(
            (phases.get("relation_pair_classification", {}) or {}).get("candidate_count", 0)
            or 0
        )
        if candidate_count > len(hinted_pairs) + 24:
            pair_cap_violations.append({
                "pmid": record.get("pmid", ""),
                "candidate_count": candidate_count,
                "unique_hint_pairs": len(hinted_pairs),
            })
    report["verification_reconciliation_invariants"] = {
        "post_action_rollback": sum(
            int(item.get("rolled_back_count", 0) or 0) for item in finalizer_audits
        ),
        "soft_flag_hard_reject": sum(
            int(item.get("soft_flag_hard_reject_count", 0) or 0)
            for item in finalizer_audits
        ),
        "pair_candidate_cap_violations": pair_cap_violations,
    }

    report["paired_bootstrap"] = {
        stage: bootstrap_f1_difference(
            stage_article_rows["A_one_shot"][stage],
            stage_article_rows["B_semantic_chunk"][stage],
        )
        for stage in stage_article_rows["A_one_shot"]
    }

    split_payload = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    split_by_pmid = {
        str(item.get("pmid", "")): str(item.get("split", ""))
        for item in split_payload.get("records", [])
    }
    calibration_seen = {pmid for pmid in selected_ids if split_by_pmid.get(pmid) == "calibration"}
    report["development_strata"] = {
        "calibration_seen_pmids": sorted(calibration_seen),
        "calibration_seen_count": len(calibration_seen),
        "non_calibration_dev_count": len(selected_ids) - len(calibration_seen),
        "held_out_claim_allowed": False,
        "arms": {},
    }
    for arm, records in arm_records.items():
        by_pmid = {str(item.get("pmid", "")): item for item in records}
        report["development_strata"]["arms"][arm] = {}
        for label, pmids in {
            "calibration_seen": calibration_seen,
            "non_calibration_dev": set(selected_ids) - calibration_seen,
        }.items():
            subset_records = [by_pmid[pmid] for pmid in selected_ids if pmid in pmids]
            subset_gold = {pmid: gold[pmid] for pmid in pmids}
            subset_source = {pmid: source[pmid] for pmid in pmids}
            report["development_strata"]["arms"][arm][label] = score_funnel(
                subset_records, subset_gold, subset_source,
            )
    ledger = []
    a_by_id = {str(item["pmid"]): item for item in arm_records["A_one_shot"]}
    b_by_id = {str(item["pmid"]): item for item in arm_records["B_semantic_chunk"]}
    for selected in sample_manifest["selection"]:
        pmid = selected["pmid"]
        gold_keys = set(relation_map(gold[pmid].get("relations", []) or []))
        a_keys = set(relation_map(verification_relations(a_by_id[pmid])))
        b_keys = set(relation_map(verification_relations(b_by_id[pmid])))
        ledger.append({
            **selected,
            "title": source[pmid].get("title", ""),
            "abstract": source[pmid].get("abstract", ""),
            "gold_relations": gold[pmid].get("relations", []) or [],
            "a_relations": verification_relations(a_by_id[pmid]),
            "b_relations": verification_relations(b_by_id[pmid]),
            "b_chunking": (
                b_by_id[pmid].get("phases", {}).get("reader", {}).get("chunking", {})
                or {}
            ),
            "tp_rescued_by_b": len((b_keys & gold_keys) - a_keys),
            "tp_lost_by_b": len((a_keys & gold_keys) - b_keys),
            "new_fp_in_b": len((b_keys - gold_keys) - a_keys),
        })
    changed_pmids = {
        pmid for pmid, record in b_by_id.items()
        if str(
            (record.get("phases", {}).get("reader", {}).get("chunking", {}) or {}).get(
                "strategy", "one_shot"
            )
        ) != "one_shot"
    }
    report["paired_design"] = {
        "unchanged_one_shot_reuses_arm_a": True,
        "changed_pmids": sorted(changed_pmids),
        "changed_document_count": len(changed_pmids),
        "unchanged_document_count": len(selected_ids) - len(changed_pmids),
    }
    changed_indexes = [index for index, pmid in enumerate(selected_ids) if pmid in changed_pmids]
    report["changed_doc_paired_bootstrap"] = {
        stage: bootstrap_f1_difference(
            [stage_article_rows["A_one_shot"][stage][index] for index in changed_indexes],
            [stage_article_rows["B_semantic_chunk"][stage][index] for index in changed_indexes],
        )
        for stage in stage_article_rows["A_one_shot"]
    }
    atomic_json(run_dir / "paired_audit_ledger.json", {"articles": ledger})
    return report


def write_markdown(run_dir: Path, report: dict[str, Any]) -> None:
    def triple(metric: dict[str, Any]) -> str:
        return "/".join(
            "undefined" if metric.get(key) is None else f"{float(metric[key]):.3f}"
            for key in ("precision", "recall", "f1")
        )

    lines = [
        "# Gold20 one-shot vs adaptive chunk A/B",
        "",
        "> Development-only length-stratified sample; do not claim external generalization.",
        "",
        "| Arm | Semantic-accepted P/R/F1 | Raw-hint P/R/F1 | Review coverage P/R/F1 | Incremental docs | Effective / incremental Gemini requests |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, item in report["arms"].items():
        funnel = item["unified_views"]["candidate_semantic"]["stage_funnel"]
        semantic = funnel["semantic_accepted"]
        raw = funnel["raw_hint"]
        review = funnel["accepted_review"]
        usage = item["usage"]
        lines.append(
            f"| {arm} | {triple(semantic)} | {triple(raw)} | {triple(review)} "
            f"| {usage.get('executed_document_count', 0)} "
            f"| {usage.get('effective_arm', {}).get('gemini_requests_estimated', 0)} / "
            f"{usage.get('incremental_execution', {}).get('gemini_requests_estimated', 0)} |"
        )
    bootstrap = report["paired_bootstrap"]["semantic_accepted"]
    lines.extend([
        "",
        f"- Primary paired semantic-accepted F1 delta B-A: `{bootstrap.get('observed_b_minus_a')}`",
        f"- Paired bootstrap 95% interval: `{bootstrap.get('percentile_95')}`",
        f"- Comparison valid: `{bootstrap.get('comparison_valid')}`",
        f"- Calibration-seen documents: `{report.get('development_strata', {}).get('calibration_seen_count')}`",
        f"- Changed documents: `{report.get('paired_design', {}).get('changed_document_count')}`",
        f"- Evaluation valid: `{report.get('evaluation_valid')}`",
    ])
    (run_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(run_dir: Path, env_file: Path, *, skip_preflight: bool = False) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(run_dir)
    load_local_env(env_file)
    tracker = ExperimentTracker(run_dir)
    tracker.start()
    try:
        frozen_paths = [GOLD_PATH, *GOLD_VIEWS.values()]
        frozen_checksums_before = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in frozen_paths
        }
        critic_routing = configure_experiment_critic()
        atomic_json(run_dir / "critic_routing.json", critic_routing)
        cfg = load_config()
        gold_rows, source_rows = load_jsonl(GOLD_PATH), load_jsonl(SOURCE_PATH)
        articles, sample_manifest = select_gold20(gold_rows, source_rows)
        atomic_json(run_dir / "sample_manifest.json", sample_manifest)
        write_jsonl(run_dir / "input_gold20.jsonl", articles)
        if not skip_preflight:
            tracker.set("provider_preflight")
            tracker.preflight = robust_provider_preflight(cfg)
            tracker.write()
        chunk_manifest_path = create_chunk_manifest(articles, run_dir, tracker)

        base = common_agent_args(cfg)
        arm_specs = {
            "A_one_shot": [
                *base,
                "--disable-chunked-extraction",
                "--extraction-cache-mode", "persistent",
                "--extraction-cache-path", str(run_dir / "cache/arm_a.sqlite3"),
                "--candidate-store-mode", "sqlite",
                "--candidate-store-path", str(run_dir / "candidates/arm_a.sqlite3"),
            ],
            "B_semantic_chunk": [
                *base,
                "--semantic-chunk-manifest", str(chunk_manifest_path),
                "--extraction-cache-mode", "persistent",
                "--extraction-cache-path", str(run_dir / "cache/arm_b.sqlite3"),
                "--candidate-store-mode", "sqlite",
                "--candidate-store-path", str(run_dir / "candidates/arm_b.sqlite3"),
            ],
        }
        invariant = sha256_json(base)
        atomic_json(run_dir / "arm_config_audit.json", {
            "common_config_hash": invariant,
            "only_intended_differences": [
                "chunking_strategy", "semantic_chunk_manifest", "local_cache_paths",
            ],
            "neo4j_write_enabled": False,
        })
        tracker.set("paired_full_agent_arms")
        arm_records: dict[str, list[dict[str, Any]]] = {}
        a_records = run_agent_records(
            task_dir=run_dir / "arms/A_one_shot",
            articles=articles,
            run_name="gold20_A_one_shot",
            tracker=tracker,
            common=[],
            variant=arm_specs["A_one_shot"],
            max_workers=1,
        )
        arm_records["A_one_shot"] = a_records
        atomic_json(run_dir / "arms/A_one_shot/records.json", {"records": a_records})

        # A strict paired design changes only documents that actually require
        # windows.  Short one-shot documents reuse Arm A byte-for-byte at the
        # extraction/system-output level, eliminating stochastic re-extraction
        # as a chunking confound.
        reader = ArticleEvidenceReader()
        chunker = ArticleChunker()
        changed_articles = []
        for article in articles:
            text = article_text(article)
            chunks = chunker.build(text, reader.read(text))
            if len(chunks) > 1 or chunks[0].strategy != "one_shot":
                changed_articles.append(article)
        changed_ids = {str(item["pmid"]) for item in changed_articles}
        b_items = run_dir / "arms/B_semantic_chunk/items"
        b_items.mkdir(parents=True, exist_ok=True)
        a_by_id = {str(item["pmid"]): item for item in a_records}
        for article in articles:
            pmid = str(article["pmid"])
            if pmid in changed_ids:
                continue
            reused = copy.deepcopy(a_by_id[pmid])
            reused.setdefault("phases", {})["paired_design"] = {
                "reused_from_arm": "A_one_shot",
                "reason": "adaptive_strategy_remained_one_shot",
                "extraction_reused": True,
            }
            atomic_json(b_items / f"{pmid}.json", reused)

        if changed_articles:
            run_agent_records(
                task_dir=run_dir / "arms/B_semantic_chunk",
                articles=changed_articles,
                run_name="gold20_B_semantic_chunk_changed",
                tracker=tracker,
                common=[],
                variant=arm_specs["B_semantic_chunk"],
                max_workers=1,
            )
        b_records = [
            json.loads((b_items / f"{article['pmid']}.json").read_text(encoding="utf-8"))
            for article in articles
        ]
        arm_records["B_semantic_chunk"] = b_records
        atomic_json(run_dir / "arms/B_semantic_chunk/records.json", {"records": b_records})

        report = build_report(run_dir, sample_manifest, chunk_manifest_path, arm_records)
        frozen_checksums_after = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in frozen_paths
        }
        all_complete = all(len(items) == 20 for items in arm_records.values())
        provider_failures = sum(
            sum(item["usage"].get("failure_statuses", {}).values())
            for item in report["arms"].values()
        )
        dangerous = sum(
            int(item["final_agent"].get("dangerous_writes", 0) or 0)
            for item in report["arms"].values()
        )
        invariants = report["verification_reconciliation_invariants"]
        frozen_unchanged = frozen_checksums_before == frozen_checksums_after
        report["evaluation_valid"] = bool(
            all_complete and provider_failures == 0 and dangerous == 0
            and invariants["post_action_rollback"] == 0
            and invariants["soft_flag_hard_reject"] == 0
            and not invariants["pair_candidate_cap_violations"]
            and frozen_unchanged
        )
        report["validity"] = {
            "all_40_article_arms_complete": all_complete,
            "provider_failure_count": provider_failures,
            "dangerous_writes": dangerous,
            "neo4j_write_enabled": False,
            "frozen_gold_unchanged": frozen_unchanged,
            "frozen_checksums_before": frozen_checksums_before,
            "frozen_checksums_after": frozen_checksums_after,
        }
        atomic_json(run_dir / "comparison_report.json", report)
        write_markdown(run_dir, report)
        tracker.close("COMPLETE")
        return 0
    except Exception as exc:
        tracker.close("FAILED", str(exc)[:1000])
        raise
    finally:
        lock.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    return run_experiment(args.run_dir.resolve(), args.env_file.resolve(), skip_preflight=args.skip_preflight)


if __name__ == "__main__":
    raise SystemExit(main())
