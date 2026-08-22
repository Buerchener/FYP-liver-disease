from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import PROFILES
from .env import merged_runtime_env
from .normalizer import write_jsonl
from .runs import RunSpec, run_dir, update_status
from .security import get_secret


def validate_write_safety(spec: RunSpec, *, confirmed: bool) -> None:
    if not spec.write_neo4j:
        return
    if not confirmed:
        raise PermissionError("Neo4j writes require explicit confirmation")
    host = urlparse(spec.neo4j_uri).hostname
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise PermissionError("--write-neo4j is restricted to localhost Neo4j")
    if spec.neo4j_database != "neo4j":
        raise PermissionError("--write-neo4j is restricted to database 'neo4j'")
    if not get_secret("neo4j_password"):
        raise PermissionError("Neo4j writes require NEO4J_PASSWORD or keyring secret")


def _runtime_env(spec: RunSpec) -> dict[str, str]:
    env = merged_runtime_env()
    secrets = {
        "GEMINI_API_KEY": get_secret("gemini_api_key"),
        "NEO4J_PASSWORD": get_secret("neo4j_password"),
        "DEEPSEEK_API_KEY": get_secret("deepseek_api_key"),
        "SECOND_LLM_API_KEY": get_secret("deepseek_api_key"),
        "QWEN_API_KEY": get_secret("qwen_api_key"),
        "DASHSCOPE_API_KEY": get_secret("qwen_api_key"),
        "ALIYUN_MAAS_API_KEY": get_secret("qwen_api_key"),
    }
    for key, value in secrets.items():
        env.setdefault(key, value)
    env["NEO4J_URI"] = spec.neo4j_uri
    env["NEO4J_USER"] = spec.neo4j_user
    env["NEO4J_DATABASE"] = spec.neo4j_database
    if spec.second_llm_enabled:
        env["SECOND_LLM_ENABLED"] = "true"
    return env


def agent_command(spec: RunSpec, *, resume: bool = False) -> list[str]:
    profile_args = dict(PROFILES.get(spec.profile, PROFILES["quality"]))
    profile_args.update(spec.agent_args)

    input_path = Path(spec.normalized_input_path)
    if resume:
        checkpoint_dir = run_dir(spec.run_id) / "checkpoints"
        completed = {path.stem for path in checkpoint_dir.glob("*.json")}
        if completed and input_path.exists():
            remaining: list[dict[str, Any]] = []
            with input_path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if str(record.get("pmid", "")) not in completed:
                        remaining.append(record)
            resumed_input = run_dir(spec.run_id) / "inputs" / "input_remaining.jsonl"
            write_jsonl(remaining, resumed_input)
            input_path = resumed_input

    cmd = [
        sys.executable,
        "-m",
        "cognitive_agent.agent",
        "--input",
        str(input_path),
        "--limit",
        str(spec.limit),
        "--run-id",
        spec.run_id,
        "--output-dir",
        spec.output_dir,
        "--api-base",
        spec.api_base,
        "--model-id",
        spec.model_id,
        "--neo4j-uri",
        spec.neo4j_uri,
        "--neo4j-user",
        spec.neo4j_user,
        "--neo4j-database",
        spec.neo4j_database,
        "--max-workers",
        str(profile_args.get("max_workers", spec.max_workers)),
        "--extraction-inner-max-workers",
        str(profile_args.get("extraction_inner_max_workers", spec.extraction_inner_max_workers)),
        "--execution-mode",
        profile_args.get("execution_mode", "agent-v2"),
        "--agent-budget-profile",
        profile_args.get("agent_budget_profile", "quality"),
        "--verification-policy",
        profile_args.get("verification_policy", "legacy"),
        "--rule-memory-mode",
        profile_args.get("rule_memory_mode", "active"),
        "--evidence-entailment-mode",
        profile_args.get("evidence_entailment_mode", "active"),
        "--risk-router-mode",
        profile_args.get("risk_router_mode", "active"),
        "--pair-classifier-mode",
        profile_args.get("pair_classifier_mode", "active"),
        "--relation-authority",
        profile_args.get("relation_authority", "unified-active"),
        "--extraction-cache-mode",
        profile_args.get("extraction_cache_mode", "persistent"),
        "--candidate-store-mode",
        profile_args.get("candidate_store_mode", "sqlite"),
        "--candidate-store-path",
        profile_args.get(
            "candidate_store_path",
            str(run_dir(spec.run_id) / "candidate_relations.sqlite3"),
        ),
        "--aux-primary-model",
        spec.aux_primary_model,
        "--aux-critic-model",
        spec.aux_critic_model,
        "--article-checkpoint-dir",
        str(run_dir(spec.run_id) / "checkpoints"),
    ]
    optional_value_flags = {
        "agent_max_aux_remote_calls": "--agent-max-aux-remote-calls",
        "agent_soft_timeout": "--agent-soft-timeout",
        "agent_hard_timeout": "--agent-hard-timeout",
        "pairwise_judge_mode": "--pairwise-judge-mode",
        "pairwise_judge_model": "--pairwise-judge-model",
        "pairwise_judge_max_pairs": "--pairwise-judge-max-pairs",
        "pairwise_judge_max_calls": "--pairwise-judge-max-calls",
        "pairwise_judge_min_confidence": "--pairwise-judge-min-confidence",
        "second_llm_timeout": "--second-llm-timeout",
    }
    for key, flag in optional_value_flags.items():
        if key in profile_args and profile_args[key] is not None:
            cmd += [flag, str(profile_args[key])]
    optional_switch_flags = {
        "pairwise_judge_claim_gate": "--pairwise-judge-claim-gate",
        "pairwise_judge_hinted_only": "--pairwise-judge-hinted-only",
        "disable_causal_conflict": "--disable-causal-conflict",
        "disable_qwen_critic": "--disable-qwen-critic",
    }
    for key, flag in optional_switch_flags.items():
        if profile_args.get(key):
            cmd.append(flag)
    if spec.rule_bundle:
        cmd += ["--rule-bundle", spec.rule_bundle]
    if spec.conformal_calibration:
        cmd += ["--conformal-calibration", spec.conformal_calibration]
    if spec.write_neo4j:
        cmd.append("--write-neo4j")
    else:
        cmd.append("--skip-neo4j-write")
    if spec.second_llm_enabled:
        cmd += [
            "--second-llm-enabled",
            "--second-llm-model-id",
            spec.second_llm_model_id,
            "--second-llm-api-base",
            spec.second_llm_api_base,
            "--second-llm-mode",
            spec.second_llm_mode,
        ]
    return cmd


def run_agent(spec: RunSpec, *, resume: bool = False) -> int:
    update_status(spec.run_id, state="running", pid=os.getpid(), command="cognitive_agent.agent")
    cmd = agent_command(spec, resume=resume)
    env = _runtime_env(spec)
    proc = subprocess.Popen(cmd, cwd=spec.project_root, env=env)
    update_status(spec.run_id, child_pid=proc.pid)
    code = proc.wait()
    state = "succeeded" if code == 0 else "failed"
    update_status(
        spec.run_id,
        state=state,
        exit_code=code,
        result_path=str(Path(spec.output_dir) / f"agent_results_{spec.run_id}.json"),
        report_path=str(Path(spec.output_dir) / f"agent_report_{spec.run_id}.json"),
    )
    return code
