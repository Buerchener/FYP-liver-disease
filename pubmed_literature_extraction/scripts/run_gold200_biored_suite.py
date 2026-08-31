#!/usr/bin/env python3
"""Run the reproducible Gold200 then BioRED Test-100 evaluation suite."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import fcntl
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.aux_model_registry import AuxModelRegistry, AuxModelSpec
from cognitive_agent.provider_errors import AUTH_ERROR, QUOTA_OR_TOKEN_EXHAUSTED
from liverkg_cli.config import load_config
from liverkg_cli.env import load_project_env
from liverkg_cli.normalizer import normalize_input
from liverkg_cli.runs import RunSpec, run_dir, save_run_spec, update_status
from liverkg_cli.security import get_secret
from liverkg_cli.worker import main as worker_main


GOLD_INPUT = ROOT / "extraction_output" / "pubmed_converted_500.jsonl"
BIORED_TEST = ROOT / ".cache/research/biored/dataset/BioRED/Test.PubTator"
RUNS_ROOT = ROOT / ".cache/liverkg_runs"
SUITE_ROOT = ROOT / ".cache/liverkg_suites"


def acquire_suite_lock():
    """Keep at most one costly Gold200/BioRED suite active per checkout."""
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = SUITE_ROOT / ".suite.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        handle.close()
        raise RuntimeError(f"another evaluation suite is already active (pid={owner})")
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def suite_status(path: Path, **updates: Any) -> None:
    current: dict[str, Any] = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(updates)
    current["updated_at"] = time.time()
    write_json(path, current)


def local_runtime_env() -> dict[str, str]:
    # The project .env is an intentional experiment-level override.  It must
    # win over inherited terminals and historical workstream credentials.
    env = os.environ.copy()
    env.update(load_project_env(ROOT))
    env["LIVERKG_RUNS_DIR"] = str(RUNS_ROOT)
    return env


def preflight(cfg) -> dict[str, Any]:
    import langextract as lx
    from openai import OpenAI
    from langextract.factory import ModelConfig

    from cognitive_agent.golden_examples import GoldenExampleSelector
    from cognitive_agent.schema.examples import KG_EXTRACTION_PROMPT
    from cognitive_agent.timeout_openai_provider import LiverKGTimeoutOpenAIModel  # noqa: F401

    primary_key = get_secret("gemini_api_key")
    if not primary_key:
        raise RuntimeError("primary Gemini-compatible API key is not configured")
    client = OpenAI(api_key=primary_key, base_url=cfg.api_base, timeout=30, max_retries=0)
    models = client.models.list()
    model_ids = sorted(str(item.id) for item in models.data)
    if cfg.model_id not in model_ids:
        raise RuntimeError(f"configured primary model is not listed by provider: {cfg.model_id}")
    completion = client.chat.completions.create(
        model=cfg.model_id,
        messages=[{"role": "user", "content": "Return only {\"ok\":true}."}],
        temperature=0,
        max_tokens=16,
        response_format={"type": "json_object"},
    )
    if not completion.choices or not completion.choices[0].message.content:
        raise RuntimeError("primary completion returned no content")

    canary_pmids = {"41810002": "review", "41581151": "mechanistic"}
    canary_articles: dict[str, dict[str, Any]] = {}
    with GOLD_INPUT.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            pmid = str(item.get("pmid", ""))
            if pmid in canary_pmids:
                canary_articles[pmid] = item
            if len(canary_articles) == len(canary_pmids):
                break
    if set(canary_articles) != set(canary_pmids):
        raise RuntimeError("primary extraction canary articles are missing")

    canary_model = ModelConfig(
        provider="liverkg_timeout_openai",
        model_id=cfg.model_id,
        provider_kwargs={
            "api_key": primary_key,
            "base_url": cfg.api_base,
            "temperature": 0.0,
            "connect_timeout_s": 15,
            "request_timeout_s": 60,
            "max_output_tokens": 4096,
            "reasoning_effort": "minimal",
        },
    )
    selector = GoldenExampleSelector()
    canary_results = []
    for pmid, study_type in canary_pmids.items():
        item = canary_articles[pmid]
        text = f"TITLE: {item.get('title', '')}\nABSTRACT: {item.get('abstract', '')}"
        selection = selector.select(
            text, study_type, max_examples=4, document_id=pmid,
        )
        canary_started = time.monotonic()
        extracted = lx.extract(
            text_or_documents=[lx.data.Document(document_id=pmid, text=text)],
            prompt_description=(
                KG_EXTRACTION_PROMPT
                + "\n\nRuntime strategy: use balanced extraction and prefer "
                "RESULTS/CONCLUSION evidence."
            ),
            examples=selection.examples,
            config=canary_model,
            temperature=0,
            max_workers=1,
            use_schema_constraints=False,
            # Compatible Gemini proxies may wrap otherwise valid JSON in a
            # single markdown fence.  Production ExtractionKernel accepts both
            # fenced and raw objects, so the canary must exercise the same path.
            fence_output=True,
            show_progress=False,
            extraction_passes=1,
            max_char_buffer=4000,
        )
        documents = extracted if isinstance(extracted, list) else [extracted]
        extraction_count = sum(
            len(getattr(document, "extractions", []) or [])
            for document in documents
        )
        if extraction_count <= 0:
            raise RuntimeError(f"primary extraction canary returned no entities for PMID {pmid}")
        canary_results.append({
            "pmid": pmid,
            "elapsed_seconds": round(time.monotonic() - canary_started, 3),
            "extraction_count": extraction_count,
            "example_name": selection.names[0],
        })

    registry = AuxModelRegistry([
        AuxModelSpec("judge", "openai", cfg.second_llm_model_id, cfg.second_llm_api_base, get_secret("deepseek_api_key"), timeout_s=45),
        AuxModelSpec("critic", "openai", cfg.aux_critic_model, cfg.qwen_api_base, get_secret("qwen_api_key"), timeout_s=45),
    ])
    for role in ("judge", "critic"):
        if not registry.configured(role):
            raise RuntimeError(f"{role} credentials are not configured")
        result = registry.call_json(
            role,
            system_prompt="Return strict JSON only.",
            user_prompt="Return {\"ok\":true}.",
            schema_hint={"ok": "boolean"},
        )
        if result.status != "OK":
            raise RuntimeError(f"{role} preflight failed: {result.error}")
    return {
        "primary_model": cfg.model_id,
        "primary_api_base": cfg.api_base,
        "listed_model_count": len(model_ids),
        "primary_usage": getattr(completion, "usage", None).model_dump() if getattr(completion, "usage", None) else {},
        "primary_extraction_canaries": canary_results,
        "aux_usage": registry.audit()["usage"],
    }


def gold_spec(run_id: str, cfg) -> RunSpec:
    current_run_dir = run_dir(run_id)
    manifest = normalize_input(
        input_path=GOLD_INPUT,
        pmids=[],
        output_path=current_run_dir / "inputs" / "input.jsonl",
        ncbi_email=cfg.ncbi_email,
        ncbi_tool=cfg.ncbi_tool,
        snapshot_dir=current_run_dir / "snapshots",
    )
    return RunSpec(
        run_id=run_id,
        profile="quality",
        input_path=str(GOLD_INPUT),
        normalized_input_path=str(current_run_dir / "inputs" / "input.jsonl"),
        output_dir="extraction_output",
        limit=200,
        dry_run=True,
        write_neo4j=False,
        max_workers=2,
        extraction_inner_max_workers=1,
        model_id=cfg.model_id,
        api_base=cfg.api_base,
        neo4j_uri=cfg.neo4j_uri,
        neo4j_user=cfg.neo4j_user,
        neo4j_database=cfg.neo4j_database,
        second_llm_enabled=True,
        second_llm_model_id=cfg.second_llm_model_id,
        second_llm_api_base=cfg.second_llm_api_base,
        second_llm_mode="conditional",
        aux_primary_model=cfg.second_llm_model_id,
        aux_critic_model=cfg.aux_critic_model,
        rule_bundle=cfg.rule_bundle,
        conformal_calibration=cfg.conformal_calibration,
        agent_args={
            "execution_mode": "agent-v2",
            "agent_budget_profile": "quality",
            "rule_memory_mode": "shadow",
            "evidence_entailment_mode": "active",
            "risk_router_mode": "off",
            "pair_classifier_mode": "active",
            "relation_authority": "unified-active",
            "max_workers": 2,
            "extraction_inner_max_workers": 1,
            "extraction_cache_mode": "persistent",
            "golden_shot_max_examples": 4,
            "verification_policy": "tiered-v2",
            "pairwise_judge_mode": "active",
            "pairwise_judge_model": cfg.second_llm_model_id,
            "pairwise_judge_max_pairs": 24,
            "pairwise_judge_max_calls": 1,
            "pairwise_judge_min_confidence": 0.7,
            "agent_max_aux_remote_calls": 6,
            "agent_soft_timeout": 420,
            "agent_hard_timeout": 900,
            "second_llm_timeout": 120,
            "disable_causal_conflict": True,
            "post_eval_gold200": True,
        },
        input_manifest=manifest,
        project_root=str(ROOT),
    )


def gold_failure_categories(run_id: str) -> dict[str, int]:
    summary_path = run_dir(run_id) / "audit_relation_ledger" / "summary.json"
    if not summary_path.exists():
        return {}
    return dict(json.loads(summary_path.read_text(encoding="utf-8")).get("provider_failure_categories", {}))


def main() -> int:
    started = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_dir = SUITE_ROOT / f"gold200_biored_{timestamp}"
    status_path = suite_dir / "suite_status.json"
    env = local_runtime_env()
    os.environ.update(env)
    cfg = load_config()
    gold_run_id = f"gold200_tieredv2_{timestamp}"
    biored_dir = ROOT / "benchmark_output" / f"biored_given_entity_test_{timestamp}"
    suite_status(
        status_path,
        state="preflight", gold_run_id=gold_run_id, biored_output_dir=str(biored_dir),
        estimated_total_seconds=7 * 3600, neo4j_write_enabled=False,
    )
    try:
        suite_lock = acquire_suite_lock()
    except RuntimeError as exc:
        suite_status(
            status_path, state="blocked_existing_suite", error=str(exc),
            finished_at=time.time(),
        )
        return 2
    try:
        suite_status(status_path, preflight=preflight(cfg))
    except Exception as exc:
        suite_status(status_path, state="preflight_failed", error=str(exc), finished_at=time.time())
        return 1

    spec = gold_spec(gold_run_id, cfg)
    save_run_spec(spec)
    update_status(gold_run_id, state="queued", log_path=str(run_dir(gold_run_id) / "logs" / "liverkg.log"))
    suite_status(status_path, state="gold200_running", stage_started_at=time.time())
    gold_code = worker_main([gold_run_id])
    result_path = ROOT / "extraction_output" / f"agent_results_{gold_run_id}.json"
    audit_dir = run_dir(gold_run_id) / "audit_relation_ledger"
    if result_path.exists():
        subprocess.run([
            sys.executable, str(ROOT / "scripts/audit_gold200_relation_ledger.py"),
            "--result", str(result_path), "--output-dir", str(audit_dir),
        ], cwd=ROOT, check=True)
    categories = gold_failure_categories(gold_run_id)
    suite_status(
        status_path, state="gold200_finished", gold_exit_code=gold_code,
        gold_provider_failure_categories=categories, gold_finished_at=time.time(),
    )
    if gold_code != 0 or categories.get(AUTH_ERROR, 0) or categories.get(QUOTA_OR_TOKEN_EXHAUSTED, 0):
        reason = "gold_exit_failure" if gold_code else "skipped_upstream_quota_or_auth_error"
        suite_status(status_path, state=reason, biored_state="skipped", finished_at=time.time())
        return gold_code or 0

    suite_status(status_path, state="biored_running", stage_started_at=time.time())
    command = [
        sys.executable, str(ROOT / "scripts/run_biored_llm_benchmark.py"),
        "--pubtator", str(BIORED_TEST), "--output-dir", str(biored_dir),
        "--cache-path", str(suite_dir / "biored_cache.sqlite3"),
        "--limit", "100", "--max-workers", "2",
    ]
    biored_code = subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
    suite_status(
        status_path,
        state="completed" if not biored_code else "biored_failed",
        biored_exit_code=biored_code, finished_at=time.time(),
        elapsed_seconds=round(time.time() - started, 2),
    )
    return biored_code


if __name__ == "__main__":
    raise SystemExit(main())
