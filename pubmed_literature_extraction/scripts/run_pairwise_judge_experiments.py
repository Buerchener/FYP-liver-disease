#!/usr/bin/env python3
"""Round-3 ablation ladder on frozen candidate snapshots.

Arms (all dry-run, no Neo4j, identical frozen LangExtract candidates):

    A baseline               legacy path, judge/second-LLM off
    B +entity recovery       coverage critic adds missing endpoint entities
    C +Claim Gate only       gate blocks non-direct-finding pairs; DIRECT_FINDING
                             survivors keep the local backend prediction
    D +predicate judge       Stage B predicate decision for gate survivors
    E +error-pattern few-shot  same-class hard-negative demonstrations
    F +conditional DeepSeek  bounded second-model adjudication (Qwen off)
    G +Qwen critic           disagreement critic on top of F

The preregistered blind-50 is never touched.  Evaluation uses the frozen
development gold; per-arm reports include semantic P/R/F1 (all-verified,
accepted-only, import-ready), per-predicate P/R/F1, evidence IoU precision,
NO_RELATION false positives, and LLM call/token/latency cost.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.abbreviation_detector import AbbreviationDetector
from cognitive_agent.extraction_quality import locate_contiguous, normalize_surface


GOLD_PATH = ROOT / "gold_annotations/pubmed_200_gold_v2_strict.jsonl"
SOURCE_PATH = ROOT / "extraction_output/pubmed_converted_500.jsonl"
FROZEN_RUN = ROOT / "extraction_output/agent_results_router_v10_four_layer_live_gold100_20260814.json"
VENV_PYTHON = ROOT / ".venv-cognitive/bin/python"
PREDICATES = (
    "ASSOCIATED_WITH", "ASSOCIATED_WITH_METABOLITE", "ENCODES", "EXPRESSED_IN",
    "INTERACTS_WITH", "PARTICIPATES_IN", "PROGNOSTIC_IN", "PROGRESSES_TO",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass
class ArmSpec:
    name: str
    flags: list[str] = field(default_factory=list)


def arm_specs(*, evaluated_pmids: list[str]) -> dict[str, ArmSpec]:
    exclude = ",".join(evaluated_pmids)
    common = [
        # The Controller stays in shadow mode: it enforces remote budgets and
        # cache accounting without changing the legacy router or write path.
        "--execution-mode", "agent-v2-shadow",
        "--router-execution-mode", "legacy",
        "--agent-mode", "precision",
        "--disable-causal-conflict",
        "--skip-neo4j-write",
        "--relation-authority", "legacy",
    ]
    return {
        "A_baseline": ArmSpec("A_baseline", [
            *common,
            "--pair-classifier-mode", "off",
            "--pairwise-judge-mode", "off",
        ]),
        # B shares A's production path; its candidate-pair recall comes from
        # the offline diagnostic over the same frozen snapshot.
        "B_lattice": ArmSpec("B_lattice", [
            *common,
            "--pair-classifier-mode", "shadow",
            "--pairwise-judge-mode", "off",
        ]),
        "C_judge": ArmSpec("C_judge", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--evidence-entailment-mode", "off",
        ]),
        "D_fewshot": ArmSpec("D_fewshot", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--evidence-entailment-mode", "off",
            "--few-shot-mode", "retrieval",
            "--few-shot-pool", str(GOLD_PATH),
            "--few-shot-source", str(SOURCE_PATH),
            "--few-shot-exclude-pmids", exclude,
        ]),
        "E_deepseek": ArmSpec("E_deepseek", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--evidence-entailment-mode", "off",
            "--few-shot-mode", "retrieval",
            "--few-shot-pool", str(GOLD_PATH),
            "--few-shot-source", str(SOURCE_PATH),
            "--few-shot-exclude-pmids", exclude,
            "--second-llm-enabled", "--second-llm-mode", "conditional",
            "--disable-qwen-critic",
        ]),
        "F_qwen": ArmSpec("F_qwen", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--evidence-entailment-mode", "off",
            "--few-shot-mode", "retrieval",
            "--few-shot-pool", str(GOLD_PATH),
            "--few-shot-source", str(SOURCE_PATH),
            "--few-shot-exclude-pmids", exclude,
            "--second-llm-enabled", "--second-llm-mode", "conditional",
        ]),
        # G isolates the candidate-volume effect: the judge only sees pairs
        # with an extractor hint or an explicit trigger, so bare co-occurrence
        # pairs abstain instead of being over-accepted.
        "G_hinted_only": ArmSpec("G_hinted_only", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--pairwise-judge-hinted-only",
            "--evidence-entailment-mode", "off",
        ]),
        # H is the aliyun-style reference on this frozen set: legacy relations
        # plus bounded DeepSeek keep/reject, no pair lattice, no judge.
        "H_legacy_adjudicated": ArmSpec("H_legacy_adjudicated", [
            *common,
            "--pair-classifier-mode", "off",
            "--pairwise-judge-mode", "off",
            "--second-llm-enabled", "--second-llm-mode", "conditional",
            "--disable-qwen-critic",
        ]),
        # ── Round-3 ablation ladder (cumulative): B..G build on each other ──
        # B: + entity recovery (coverage critic, training-free)
        "R3_B_recovery": ArmSpec("R3_B_recovery", [
            *common,
            "--pair-classifier-mode", "off",
            "--pairwise-judge-mode", "off",
            "--entity-recovery-mode", "active",
        ]),
        # C: + Claim Gate only (predicate stage off: DIRECT_FINDING survivors
        # keep their local backend prediction — isolates the gate's own effect)
        "R3_C_gate_only": ArmSpec("R3_C_gate_only", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--evidence-entailment-mode", "off",
            "--pairwise-judge-claim-gate",
            "--pairwise-judge-predicate-off",
            "--entity-recovery-mode", "active",
        ]),
        # D: + Stage B predicate judge on the gate survivors
        "R3_D_gate_predicate": ArmSpec("R3_D_gate_predicate", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--evidence-entailment-mode", "off",
            "--pairwise-judge-claim-gate",
            "--entity-recovery-mode", "active",
        ]),
        # E: + error-pattern few-shot (same-class hard negatives, no training)
        "R3_E_error_pattern": ArmSpec("R3_E_error_pattern", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--evidence-entailment-mode", "off",
            "--pairwise-judge-claim-gate",
            "--entity-recovery-mode", "active",
            "--few-shot-mode", "error_pattern",
            "--few-shot-pool", str(GOLD_PATH),
            "--few-shot-source", str(SOURCE_PATH),
            "--few-shot-exclude-pmids", exclude,
        ]),
        # F: + conditional DeepSeek adjudication (router-gated, Qwen off)
        "R3_F_deepseek": ArmSpec("R3_F_deepseek", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--evidence-entailment-mode", "off",
            "--pairwise-judge-claim-gate",
            "--entity-recovery-mode", "active",
            "--few-shot-mode", "error_pattern",
            "--few-shot-pool", str(GOLD_PATH),
            "--few-shot-source", str(SOURCE_PATH),
            "--few-shot-exclude-pmids", exclude,
            "--second-llm-enabled", "--second-llm-mode", "conditional",
            "--disable-qwen-critic",
        ]),
        # G: + disagreement Qwen critic on top of F
        "R3_G_qwen": ArmSpec("R3_G_qwen", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--evidence-entailment-mode", "off",
            "--pairwise-judge-claim-gate",
            "--entity-recovery-mode", "active",
            "--few-shot-mode", "error_pattern",
            "--few-shot-pool", str(GOLD_PATH),
            "--few-shot-source", str(SOURCE_PATH),
            "--few-shot-exclude-pmids", exclude,
            "--second-llm-enabled", "--second-llm-mode", "conditional",
        ]),
        # ── Round-4 ablation (Claim Gate v2, unified authority, no legacy bypass) ──
        # A: baseline — explicit unified relation path (judge off, no claim
        # gate).  Legacy remains a separate, compatibility-preserving arm.
        "R4_A_baseline": ArmSpec("R4_A_baseline", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "off",
        ]),
        # B: + Claim Gate v2 (two-stage: ASSERTED | claim_role)
        "R4_B_claim_gate": ArmSpec("R4_B_claim_gate", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--evidence-entailment-mode", "off",
            "--pairwise-judge-claim-gate",
        ]),
        # C: + Predicate Judge on ASSERTED survivors
        "R4_C_predicate": ArmSpec("R4_C_predicate", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--evidence-entailment-mode", "off",
            "--pairwise-judge-claim-gate",
            # predicate_stage defaults to True
        ]),
        # D: + independent second-model verification (CONFIRM/REJECT only)
        "R4_D_second_llm": ArmSpec("R4_D_second_llm", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--evidence-entailment-mode", "off",
            "--pairwise-judge-claim-gate",
            "--second-llm-enabled", "--second-llm-mode", "conditional",
            "--disable-qwen-critic",
        ]),
        # E: + disagreement-triggered Qwen critic
        "R4_E_critic": ArmSpec("R4_E_critic", [
            *common,
            "--pair-classifier-mode", "active",
            "--relation-authority", "unified-active",
            "--pairwise-judge-mode", "active",
            "--evidence-entailment-mode", "off",
            "--pairwise-judge-claim-gate",
            "--second-llm-enabled", "--second-llm-mode", "conditional",
        ]),
    }


def build_snapshot(records: list[dict]) -> Path:
    snapshot = ROOT / "benchmark_output" / "pairwise_judge_frozen_candidates.json"
    payload = {
        "candidates": {
            str(item.get("pmid", "")): dict(item.get("phases", {}).get("extraction", {}) or {})
            for item in records
            if item.get("pmid")
        },
    }
    snapshot.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot


def run_arm(
    arm: ArmSpec, *, snapshot: Path, limit: int, run_id: str, max_workers: int,
    output_dir: Path, env: dict[str, str],
) -> Path:
    args = [
        str(VENV_PYTHON), "-m", "cognitive_agent.agent",
        "--input", str(SOURCE_PATH),
        "--limit", str(limit),
        "--run-id", run_id,
        "--output-dir", str(output_dir),
        "--frozen-candidates", str(snapshot),
        "--max-workers", str(max_workers),
        "--extraction-inner-max-workers", "1",
        *arm.flags,
    ]
    print(f"[ARM START] {arm.name} run_id={run_id}", flush=True)
    completed = subprocess.run(
        args, env=env, capture_output=True, text=True, timeout=3 * 3600,
    )
    if completed.returncode != 0:
        print(f"[ARM FAILED] {arm.name} rc={completed.returncode}", flush=True)
        print(completed.stdout[-4000:], flush=True)
        print(completed.stderr[-4000:], flush=True)
        raise RuntimeError(f"arm {arm.name} failed with rc={completed.returncode}")
    print(f"[ARM DONE] {arm.name}", flush=True)
    return output_dir / f"agent_results_{run_id}.json"


# ──────────────────────────── scoring ────────────────────────────

def _aliases(gold: dict, text: str) -> dict[str, set[tuple[str, str]]]:
    mapping: dict[str, set[tuple[str, str]]] = {}
    for ent in gold.get("entities", []) or []:
        canonical = normalize_surface(ent.get("canonical", ent.get("mention", "")))
        typed = (canonical, str(ent.get("type", "")))
        for value in (ent.get("mention", ""), ent.get("canonical", "")):
            mapping.setdefault(normalize_surface(value), set()).add(typed)
    detected = AbbreviationDetector().detect(text)
    for short, long_form in detected.abbr_to_long.items():
        short_n, long_n = normalize_surface(short), normalize_surface(long_form)
        for typed in mapping.get(short_n, set()) | mapping.get(long_n, set()):
            mapping.setdefault(short_n, set()).add(typed)
            mapping.setdefault(long_n, set()).add(typed)
    return mapping


def _canonical_endpoint(value: str, etype: str, aliases: dict) -> str:
    surface = normalize_surface(value)
    same_type = [key[0] for key in aliases.get(surface, set()) if key[1] == etype]
    return same_type[0] if len(same_type) == 1 else surface


def _relation_key(item: dict, aliases: dict) -> tuple[str, str, str]:
    return (
        _canonical_endpoint(str(item.get("subject", "")), str(item.get("subject_type", "")), aliases),
        str(item.get("predicate", "")).upper(),
        _canonical_endpoint(str(item.get("object", "")), str(item.get("object_type", "")), aliases),
    )


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def score_arm(records: list[dict], gold_by_pmid: dict, source_by_pmid: dict) -> dict[str, Any]:
    rel = Counter()
    rel_accepted = Counter()
    strict = Counter()
    entity = Counter()
    evidence_correct = 0
    evidence_predictions = 0
    source_contiguous = 0
    predicate_counts: dict[str, Counter] = defaultdict(Counter)
    zero_relation_docs = 0
    zero_relation_fp_docs = 0
    judge_calls = 0
    second_llm_calls = 0
    qwen_calls = 0
    judge_tokens = [0, 0]
    llm_tokens = [0, 0]
    latencies: list[float] = []

    for record in records:
        pmid = str(record.get("pmid", ""))
        gold = gold_by_pmid.get(pmid)
        if gold is None:
            continue
        source = source_by_pmid.get(pmid, {})
        text = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"
        aliases = _aliases(gold, text)
        phases = record.get("phases", {}) or {}
        verification = phases.get("verification", {}) or {}
        pred_entities = {
            (normalize_surface(item.get("mention", "")), str(item.get("type", item.get("entity_type", ""))))
            for item in (verification.get("entities", []) or [])
        }
        gold_entities = {
            (normalize_surface(e.get("canonical", e.get("mention", ""))), str(e.get("type", "")))
            for e in (gold.get("entities", []) or [])
        }
        entity["tp"] += len(gold_entities & pred_entities)
        entity["fp"] += len(pred_entities - gold_entities)
        entity["fn"] += len(gold_entities - pred_entities)

        relations = verification.get("relations", []) or []
        gold_relations = gold.get("relations", []) or []
        gold_keys = {_relation_key(item, aliases) for item in gold_relations}
        gold_strict_keys = {_relation_key(item, aliases) for item in gold_relations if item.get("import_ready")}
        pred_keys = {_relation_key(item, aliases) for item in relations}
        accepted_keys = {
            _relation_key(item, aliases) for item in relations
            if str(item.get("semantic_status", "") or "ACCEPTED").upper() == "ACCEPTED"
        }
        pred_strict_keys = {_relation_key(item, aliases) for item in relations if item.get("import_ready")}
        rel["tp"] += len(gold_keys & pred_keys)
        rel["fp"] += len(pred_keys - gold_keys)
        rel["fn"] += len(gold_keys - pred_keys)
        rel_accepted["tp"] += len(gold_keys & accepted_keys)
        rel_accepted["fp"] += len(accepted_keys - gold_keys)
        rel_accepted["fn"] += len(gold_keys - accepted_keys)
        strict["tp"] += len(gold_strict_keys & pred_strict_keys)
        strict["fp"] += len(pred_strict_keys - gold_strict_keys)
        strict["fn"] += len(gold_strict_keys - pred_strict_keys)

        if not gold_relations:
            zero_relation_docs += 1
            zero_relation_fp_docs += int(bool(pred_keys))

        for predicate in PREDICATES:
            gold_p = {key for key in gold_keys if key[1] == predicate}
            pred_p = {key for key in pred_keys if key[1] == predicate}
            predicate_counts[predicate].update({
                "tp": len(gold_p & pred_p),
                "fp": len(pred_p - gold_p),
                "fn": len(gold_p - pred_p),
            })

        gold_by_key: dict[tuple, list[dict]] = defaultdict(list)
        for relation in gold_relations:
            gold_by_key[_relation_key(relation, aliases)].append(relation)
        for relation in relations:
            evidence_predictions += 1
            evidence = str(relation.get("evidence", "") or "")
            contiguous, ps, pe = locate_contiguous(evidence, text)
            source_contiguous += int(contiguous)
            if not contiguous:
                continue
            key = _relation_key(relation, aliases)
            for gold_relation in gold_by_key.get(key, []):
                ok, gs, ge = locate_contiguous(str(gold_relation.get("evidence", "") or ""), text)
                if not ok:
                    continue
                intersection = max(0, min(pe, ge) - max(ps, gs))
                union = max(pe, ge) - min(ps, gs)
                if union and intersection / union >= 0.5:
                    evidence_correct += 1
                    break

        judge_phase = phases.get("pairwise_judge", {}) or {}
        for audit in judge_phase.get("audits", []) or []:
            if audit.get("status") == "OK":
                judge_calls += 1
                judge_tokens[0] += int(audit.get("prompt_tokens", 0) or 0)
                judge_tokens[1] += int(audit.get("output_tokens", 0) or 0)
        collaboration = phases.get("collaboration", {}) or {}
        for round_item in collaboration.get("rounds", []) or []:
            if round_item.get("triggered"):
                second_llm_calls += 1
                llm_tokens[0] += int(round_item.get("prompt_tokens", 0) or 0)
                llm_tokens[1] += int(round_item.get("output_tokens", 0) or 0)
        critic_audit = (phases.get("collaboration", {}) or {}).get("critic_audit", {}) or {}
        if critic_audit.get("status") not in {None, "NOT_TRIGGERED", "UNCONFIGURED"}:
            qwen_calls += 1
        timing = record.get("timing", {}) or {}
        latencies.append(float(
            timing.get("total_s", record.get("total_time_s", 0.0)) or 0.0
        ))

    predicate_metrics = {}
    for predicate in PREDICATES:
        counts = predicate_counts.get(predicate, Counter())
        predicate_metrics[predicate] = _prf(counts["tp"], counts["fp"], counts["fn"])

    # Macro-averaged P/R/F1 over positive relation classes, following the
    # ANCHOR-RE / SemRepGS protocol (no-rel excluded, each positive predicate
    # gets equal weight regardless of instance count).
    _preds = [pm for pm in predicate_metrics.values() if pm["tp"] + pm["fp"] + pm["fn"] > 0]
    macro = {
        "precision": sum(pm["precision"] for pm in _preds) / len(_preds) if _preds else 0.0,
        "recall": sum(pm["recall"] for pm in _preds) / len(_preds) if _preds else 0.0,
        "f1": sum(pm["f1"] for pm in _preds) / len(_preds) if _preds else 0.0,
    }

    def percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        rank = (len(ordered) - 1) * p
        low, high = math.floor(rank), math.ceil(rank)
        if low == high:
            return ordered[low]
        return ordered[low] * (high - rank) + ordered[high] * (rank - low)

    return {
        "entity_endpoint": _prf(entity["tp"], entity["fp"], entity["fn"]),
        "semantic_relation_all_verified": _prf(rel["tp"], rel["fp"], rel["fn"]),
        "semantic_relation_accepted_only": _prf(
            rel_accepted["tp"], rel_accepted["fp"], rel_accepted["fn"]
        ),
        "strict_import_ready": _prf(strict["tp"], strict["fp"], strict["fn"]),
        # ANCHOR-RE 同款口径：对所有 positive 关系类的 macro 平均。
        "macro_semantic_relation_all_verified": macro,
        "macro_semantic_relation_accepted_only": {
            "precision": sum(pm["precision"] for pm in _preds) / len(_preds) if _preds else 0.0,
            "recall": sum(pm["recall"] for pm in _preds) / len(_preds) if _preds else 0.0,
            "f1": sum(pm["f1"] for pm in _preds) / len(_preds) if _preds else 0.0,
        },
        "evidence_span_precision_iou_0_5": (
            evidence_correct / evidence_predictions if evidence_predictions else 1.0
        ),
        "evidence_source_contiguous_rate": (
            source_contiguous / evidence_predictions if evidence_predictions else 1.0
        ),
        "no_relation_false_positive_docs": zero_relation_fp_docs,
        "zero_relation_docs": zero_relation_docs,
        "predicate_metrics": predicate_metrics,
        "calls": {
            "judge": judge_calls,
            "second_llm": second_llm_calls,
            "qwen_critic": qwen_calls,
            "judge_calls_per_article": judge_calls / max(1, len(records)),
            "second_llm_calls_per_article": second_llm_calls / max(1, len(records)),
        },
        "tokens": {
            "judge_prompt": judge_tokens[0], "judge_output": judge_tokens[1],
            "second_llm_prompt": llm_tokens[0], "second_llm_output": llm_tokens[1],
        },
        "latency": {
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
    }


def markdown(arm_results: dict[str, dict], lattice_summary: dict | None) -> str:
    lines = [
        "# Pairwise-judge experiment comparison (frozen 100-doc dev gold)",
        "",
        "All arms share identical frozen LangExtract candidates and the same",
        "deterministic verifier.  Dry-run only; no Neo4j writes; blind-50 untouched.",
        "",
    ]
    if lattice_summary:
        lines += [
            "## Candidate lattice recall (offline diagnostic)",
            "",
            f"- Old clause-only lattice: {lattice_summary.get('candidate_pair_recall_old', 0):.1%}",
            f"- New sentence + adjacent windows: {lattice_summary.get('candidate_pair_recall_new', 0):.1%}",
            f"- Delta: {lattice_summary.get('recall_delta', 0):+.1%}",
            "",
        ]
    lines += [
        "## Headline metrics",
        "",
        "| Arm | Semantic P/R/F1 (all verified) | Semantic P/R/F1 (accepted) | Import-ready P/R/F1 | Evidence IoU P | NO_REL FP docs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in arm_results.items():
        sem = metrics["semantic_relation_all_verified"]
        acc = metrics["semantic_relation_accepted_only"]
        strict = metrics["strict_import_ready"]
        lines.append(
            f"| {name} | {sem['precision']:.3f}/{sem['recall']:.3f}/{sem['f1']:.3f} | "
            f"{acc['precision']:.3f}/{acc['recall']:.3f}/{acc['f1']:.3f} | "
            f"{strict['precision']:.3f}/{strict['recall']:.3f}/{strict['f1']:.3f} | "
            f"{metrics['evidence_span_precision_iou_0_5']:.3f} | "
            f"{metrics['no_relation_false_positive_docs']}/{metrics['zero_relation_docs']} |"
        )
    lines += [
        "",
        "## Per-predicate semantic F1 (all verified)",
        "",
        "| Predicate | " + " | ".join(arm_results) + " |",
        "|---|" + "---:|" * len(arm_results),
    ]
    for predicate in PREDICATES:
        row = [predicate]
        for metrics in arm_results.values():
            value = metrics["predicate_metrics"].get(predicate, {})
            row.append(f"{value.get('precision', 0):.2f}/{value.get('recall', 0):.2f}/{value.get('f1', 0):.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## LLM calls, tokens and latency",
        "",
        "| Arm | Judge calls | 2nd-LLM calls | Qwen calls | Judge tok in/out | 2nd-LLM tok in/out | Latency p50/p95 s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in arm_results.items():
        calls = metrics["calls"]
        tokens = metrics["tokens"]
        latency = metrics["latency"]
        lines.append(
            f"| {name} | {calls['judge']} | {calls['second_llm']} | {calls['qwen_critic']} | "
            f"{tokens['judge_prompt']}/{tokens['judge_output']} | "
            f"{tokens['second_llm_prompt']}/{tokens['second_llm_output']} | "
            f"{latency['p50']:.1f}/{latency['p95']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="A,B,C,D,E,F")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_output")
    parser.add_argument("--run-id", default="pairwise_judge_experiments")
    parser.add_argument("--skip-diagnostic", action="store_true")
    parser.add_argument(
        "--frozen-run", type=Path, default=FROZEN_RUN,
        help="agent results JSON whose phases.extraction become the frozen snapshot",
    )
    parser.add_argument(
        "--frozen-snapshot", type=Path, default=None,
        help="reuse an existing frozen-candidates JSON instead of rebuilding "
        "from --frozen-run (guarantees bit-identical candidates across reruns)",
    )
    args = parser.parse_args()

    if args.frozen_snapshot:
        frozen_payload = json.loads(args.frozen_snapshot.read_text(encoding="utf-8"))
        frozen_records = [
            {"pmid": pmid, "phases": {"extraction": extraction}}
            for pmid, extraction in frozen_payload.get("candidates", {}).items()
        ][: args.limit]
        snapshot = args.frozen_snapshot
    else:
        frozen_payload = json.loads(args.frozen_run.read_text(encoding="utf-8"))
        frozen_records = frozen_payload.get("records", []) if isinstance(frozen_payload, dict) else frozen_payload
        frozen_records = frozen_records[: args.limit]
        snapshot = build_snapshot(frozen_records)

    gold_rows = load_jsonl(GOLD_PATH)[: args.limit]
    gold_by_pmid = {str(item["pmid"]): item for item in gold_rows}
    source_by_pmid = {str(item["pmid"]): item for item in load_jsonl(SOURCE_PATH)}
    evaluated_pmids = sorted(gold_by_pmid)

    env = dict(os.environ)
    env.setdefault("DEEPSEEK_API_KEY", "")
    env.setdefault("GEMINI_API_KEY", "")

    out_dir = args.output_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    all_specs = arm_specs(evaluated_pmids=evaluated_pmids)
    letter_to_name = {
        "A": "A_baseline", "B": "R3_B_recovery", "C": "R3_C_gate_only",
        "D": "R3_D_gate_predicate", "E": "R3_E_error_pattern",
        "F": "R3_F_deepseek", "G": "R3_G_qwen",
        # Round-4 arms (current focus; select via --arms A,B,C,D,E,F)
        "A4": "R4_A_baseline", "B4": "R4_B_claim_gate",
        "C4": "R4_C_predicate", "D4": "R4_D_second_llm",
        "E4": "R4_E_critic",
    }
    selected = [
        letter_to_name.get(name.strip().upper(), name.strip())
        for name in args.arms.split(",") if name.strip()
    ]

    lattice_summary = None
    if not args.skip_diagnostic:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "diagnose_candidate_misses",
            str(ROOT / "scripts" / "diagnose_candidate_misses.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        diag_rows = []
        for record in frozen_records:
            pmid = str(record.get("pmid", ""))
            if pmid not in gold_by_pmid or pmid not in source_by_pmid:
                continue
            gold = gold_by_pmid[pmid]
            source = source_by_pmid[pmid]
            text = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"
            extraction = record.get("phases", {}).get("extraction", {}) or {}
            old_config = module.PairClassifierConfig(
                include_parent_sentences=False, include_adjacent_windows=False,
                max_candidates=64,
            )
            new_config = module.PairClassifierConfig(
                include_parent_sentences=True, include_adjacent_windows=True,
                max_candidates=128,
            )
            diag_rows.extend(module.diagnose_article(
                pmid, text, extraction,
                gold.get("relations", []) or [], gold.get("entities", []) or [],
                old_config, new_config,
            ))
        lattice_summary = module.summarize(diag_rows)

    arm_results: dict[str, dict] = {}
    arm_records: dict[str, list[dict]] = {}
    for name in selected:
        arm = all_specs[name]
        results_path = run_arm(
            arm, snapshot=snapshot, limit=args.limit,
            run_id=f"pairwise_{name}_{args.run_id}", max_workers=args.max_workers,
            output_dir=out_dir, env=env,
        )
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        records = payload.get("records", payload.get("history", []))
        if isinstance(payload, dict) and "records" not in payload and "history" not in payload:
            records = payload.get("articles", [])
        if not records:
            records = []
        # Agent results store per-article records under 'records'.
        if not records and isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list) and value and isinstance(value[0], dict) and "pmid" in value[0]:
                    records = value
                    break
        arm_records[name] = records
        arm_results[name] = score_arm(records, gold_by_pmid, source_by_pmid)

    (out_dir / "metrics.json").write_text(
        json.dumps({"arms": arm_results, "lattice": lattice_summary},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(markdown(arm_results, lattice_summary), encoding="utf-8")
    print(markdown(arm_results, lattice_summary))
    print(f"[DONE] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
