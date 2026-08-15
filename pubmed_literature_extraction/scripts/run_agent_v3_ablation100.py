#!/usr/bin/env python3
"""Leakage-controlled, resumable 100-article Agent v3 ablation experiment.

The runner freezes primary candidates, fits fold-specific rule/calibration
artifacts without reading the held-out fold, executes ten ablations, performs
three warm replays, and writes article-level metrics for the statistical
evaluator.  Neo4j writes are never enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.aux_model_registry import AuxModelRegistry
from cognitive_agent.conformal_router import (
    CalibrationExample,
    ConformalCalibration,
    RiskFeatures,
)
from cognitive_agent.hybrid_article_profiler import rule_profile
from cognitive_agent.rule_learning import RuleLearner
from cognitive_agent.rule_memory import (
    ErrorCard,
    RuleBundle,
    RulePromotionGate,
    SoftRule,
    render_rule_context,
)


SEED = 20260815
ARMS = (
    "legacy",
    "agent_v2_no_rules",
    "deepseek_always",
    "full_v3",
    "v3_no_rule_memory",
    "v3_no_qwen_critic",
    "v3_no_evidence_selector",
    "v3_no_conformal_router",
    "v3_no_causal_conflict",
    "v3_no_cache",
)
HARD_FLAGS = {
    "schema_mismatch", "invalid_schema", "missing_endpoint", "endpoint_not_grounded",
    "evidence_not_contiguous", "evidence_not_in_source", "hard_negation",
    "background_only", "method_only", "safe_write_blocked",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The status heartbeat and the main runner can write concurrently.  A fixed
    # ``.tmp`` name lets one writer replace the other writer's temporary file.
    # Give every writer its own staging path; os.replace remains atomic.
    temp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temp.replace(path)


def sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("'\"")


def norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").split())


def triple(relation: dict) -> tuple[str, str, str, str, str]:
    return (
        norm(relation.get("subject")),
        str(relation.get("subject_type", "")),
        str(relation.get("predicate", "")),
        norm(relation.get("object")),
        str(relation.get("object_type", "")),
    )


def relation_map(relations: list[dict]) -> dict[tuple[str, str, str, str, str], dict]:
    return {triple(item): item for item in relations}


def article_labels(gold: dict, source: dict) -> set[str]:
    profile = rule_profile(source.get("title", ""), source.get("abstract", ""))
    relations = gold.get("relations", []) or []
    labels = {
        f"study:{profile.primary_study_type}",
        "relation:positive" if relations else "relation:zero",
        "strict:positive" if any(item.get("import_ready") for item in relations) else "strict:zero",
        "scope:in" if gold.get("in_scope") else "scope:out",
    }
    labels.update(f"predicate:{item.get('predicate', 'UNKNOWN')}" for item in relations)
    return labels


def iterative_select(
    rows: list[dict], size: int, *, labels_by_id: dict[str, set[str]], seed: int,
) -> list[dict]:
    """Deterministic multilabel selection with rare-label preservation."""
    if size >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    tie = {str(row["pmid"]): rng.random() for row in rows}
    frequencies = Counter(label for row in rows for label in labels_by_id[str(row["pmid"])])
    targets = {
        label: min(count, max(1, round(count * size / len(rows))))
        for label, count in frequencies.items()
    }
    selected: list[dict] = []
    counts: Counter[str] = Counter()
    remaining = list(rows)
    while len(selected) < size:
        def score(row: dict) -> tuple[float, float, str]:
            pmid = str(row["pmid"])
            gain = sum(
                max(0, targets[label] - counts[label]) / max(1, frequencies[label])
                for label in labels_by_id[pmid]
            )
            return gain, tie[pmid], pmid
        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        counts.update(labels_by_id[str(chosen["pmid"])])
    return selected


def make_folds(
    rows: list[dict], *, labels_by_id: dict[str, set[str]], seed: int,
) -> list[list[dict]]:
    rng = random.Random(seed)
    tie = {str(row["pmid"]): rng.random() for row in rows}
    frequencies = Counter(label for row in rows for label in labels_by_id[str(row["pmid"])])
    ordered = sorted(
        rows,
        key=lambda row: (
            -sum(1 / max(1, frequencies[label]) for label in labels_by_id[str(row["pmid"])]),
            tie[str(row["pmid"])],
        ),
    )
    folds: list[list[dict]] = [[] for _ in range(5)]
    counts = [Counter() for _ in range(5)]
    for row in ordered:
        labels = labels_by_id[str(row["pmid"])]
        candidates = [index for index in range(5) if len(folds[index]) < 20]
        index = min(
            candidates,
            key=lambda item: (
                sum(counts[item][label] / max(1, frequencies[label]) for label in labels),
                len(folds[item]),
                item,
            ),
        )
        folds[index].append(row)
        counts[index].update(labels)
    if [len(fold) for fold in folds] != [20] * 5:
        raise RuntimeError("fold construction failed to produce five 20-article folds")
    return folds


def build_manifest(gold_rows: list[dict], source_rows: list[dict]) -> dict:
    source_by_id = {str(item["pmid"]): item for item in source_rows}
    old50 = {str(item["pmid"]) for item in gold_rows[:50]}
    pool = [item for item in gold_rows if str(item["pmid"]) not in old50]
    labels_by_id = {
        str(item["pmid"]): article_labels(item, source_by_id[str(item["pmid"])])
        for item in gold_rows
    }
    selected = iterative_select(pool, 100, labels_by_id=labels_by_id, seed=SEED)
    folds = make_folds(selected, labels_by_id=labels_by_id, seed=SEED + 1)
    all_by_id = {str(item["pmid"]): item for item in gold_rows}
    fold_payloads = []
    for fold_index, test_rows in enumerate(folds):
        test_ids = {str(item["pmid"]) for item in test_rows}
        training = [item for item in gold_rows if str(item["pmid"]) not in test_ids]
        calibration = iterative_select(
            training, 30, labels_by_id=labels_by_id, seed=SEED + 100 + fold_index,
        )
        calibration_ids = {str(item["pmid"]) for item in calibration}
        validation_pool = [item for item in training if str(item["pmid"]) not in calibration_ids]
        validation = iterative_select(
            validation_pool, 30, labels_by_id=labels_by_id, seed=SEED + 200 + fold_index,
        )
        validation_ids = {str(item["pmid"]) for item in validation}
        induction = [
            item for item in training
            if str(item["pmid"]) not in calibration_ids | validation_ids
        ]
        if len(induction) != 120:
            raise RuntimeError("fold learning split must be 120/30/30")
        fold_payloads.append({
            "fold": fold_index + 1,
            "test_pmids": [str(item["pmid"]) for item in test_rows],
            "induction_pmids": [str(item["pmid"]) for item in induction],
            "validation_pmids": [str(item["pmid"]) for item in validation],
            "calibration_pmids": [str(item["pmid"]) for item in calibration],
        })
    selected_ids = [str(item["pmid"]) for item in selected]
    manifest = {
        "experiment": "agent-v3-ablation100-crossfit-v1",
        "seed": SEED,
        "created_at": utc_now(),
        "evaluation_pmids": selected_ids,
        "excluded_historical_gold50_pmids": sorted(old50),
        "golden_example_pmids": ["41650163", "41482383"],
        "folds": fold_payloads,
        "arms": list(ARMS),
        "source_hashes": {
            pmid: hashlib.sha256(
                (source_by_id[pmid].get("title", "") + "\n" + source_by_id[pmid].get("abstract", "")).encode("utf-8")
            ).hexdigest()
            for pmid in selected_ids
        },
        "gold_hash": sha256_json([all_by_id[pmid] for pmid in selected_ids]),
        "rules": {
            "test_labels_hidden_until_predictions_frozen": True,
            "fold_learning_sizes": {"induction": 120, "validation": 30, "calibration": 30},
        },
    }
    manifest["manifest_hash"] = sha256_json({k: v for k, v in manifest.items() if k != "created_at"})
    return manifest


def validate_manifest(manifest: dict, blind_manifest: Path) -> None:
    evaluation = manifest["evaluation_pmids"]
    excluded = set(manifest["excluded_historical_gold50_pmids"])
    if len(evaluation) != 100 or len(set(evaluation)) != 100:
        raise ValueError("evaluation manifest must contain 100 unique PMIDs")
    if set(evaluation) & excluded:
        raise ValueError("evaluation set overlaps historical gold50")
    if set(evaluation) & set(manifest["golden_example_pmids"]):
        raise ValueError("evaluation set overlaps prompt Golden Examples")
    if blind_manifest.exists():
        blind = json.loads(blind_manifest.read_text(encoding="utf-8"))
        blind_ids = set()
        for key in ("pmids", "articles", "records", "selected"):
            for item in blind.get(key, []) if isinstance(blind, dict) else []:
                blind_ids.add(str(item.get("pmid", item)) if isinstance(item, dict) else str(item))
        if set(evaluation) & blind_ids:
            raise ValueError("evaluation set overlaps preregistered blind50")
    for fold in manifest["folds"]:
        test = set(fold["test_pmids"])
        train = set(fold["induction_pmids"] + fold["validation_pmids"] + fold["calibration_pmids"])
        if len(test) != 20 or len(train) != 180 or test & train:
            raise ValueError(f"fold {fold['fold']} leakage or size violation")


class StatusTracker:
    def __init__(self, run_dir: Path, total_article_arms: int):
        self.run_dir = run_dir
        self.total = total_article_arms
        self.started = time.time()
        self.current = "initializing"
        self.stop_event = threading.Event()
        self.write_lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def set(self, current: str) -> None:
        self.current = current
        self.write()

    def completed(self) -> int:
        roots = [self.run_dir / "arms", self.run_dir / "warm_replays"]
        return sum(1 for root in roots if root.exists() for _ in root.rglob("items/*.json"))

    def write(self, final_status: str = "RUNNING", error: str = "") -> None:
        with self.write_lock:
            done = self.completed()
            elapsed = max(time.time() - self.started, 0.001)
            rate = done / elapsed
            eta = (self.total - done) / rate if rate > 0 else None
            atomic_json(self.run_dir / "status.json", {
                "status": final_status,
                "pid": os.getpid(),
                "heartbeat": utc_now(),
                "current": self.current,
                "completed_article_arms": done,
                "total_article_arms": self.total,
                "progress": round(done / self.total, 4),
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(eta, 1) if eta is not None else None,
                "error": error,
            })

    def _loop(self) -> None:
        while not self.stop_event.wait(15):
            self.write()

    def close(self, status: str, error: str = "") -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.write(status, error)


def preflight() -> dict:
    from openai import OpenAI

    required = (
        "GEMINI_API_KEY", "GEMINI_API_BASE", "GEMINI_MODEL",
        "DEEPSEEK_API_KEY", "ALIYUN_MAAS_API_KEY", "ALIYUN_MAAS_API_BASE",
    )
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise RuntimeError("missing required environment variables: " + ", ".join(missing))
    output: dict[str, Any] = {"checked_at": utc_now()}
    started = time.perf_counter()
    main = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url=os.environ["GEMINI_API_BASE"], timeout=30,
    )
    response = main.chat.completions.create(
        model=os.environ["GEMINI_MODEL"], temperature=0,
        messages=[{"role": "user", "content": "Return only JSON: {\"ok\":true}"}],
        response_format={"type": "json_object"},
    )
    json.loads(str(response.choices[0].message.content or "{}"))
    output["primary"] = {
        "status": "OK", "model": os.environ["GEMINI_MODEL"],
        "api_base": os.environ["GEMINI_API_BASE"],
        "latency_s": round(time.perf_counter() - started, 3),
    }
    registry = AuxModelRegistry.from_environment(
        primary_model=os.environ.get("AUX_PRIMARY_MODEL", "deepseek-v4-flash"),
        critic_model=os.environ.get("AUX_CRITIC_MODEL", "qwen3.6-flash"), timeout_s=30,
    )
    for role in ("primary", "critic"):
        result = registry.call_json(
            role,
            system_prompt="Return JSON only with one boolean field named ok.",
            user_prompt='Return {"ok": true}.', schema_hint={"ok": "boolean"},
        )
        if result.status != "OK" or result.payload.get("ok") is not True:
            raise RuntimeError(f"{role} preflight failed: {result.error or result.payload}")
        output["deepseek" if role == "primary" else "qwen"] = {
            "status": "OK", "model": result.model_id, "latency_s": result.latency_s,
        }
    return output


def record_ok(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return not payload.get("error") and bool(payload.get("phases", {}).get("verification"))
    except Exception:
        return False


def run_agent_records(
    *, task_dir: Path, articles: list[dict], run_name: str, tracker: StatusTracker,
    common: list[str], variant: list[str], max_workers: int,
) -> list[dict]:
    items = task_dir / "items"
    items.mkdir(parents=True, exist_ok=True)
    expected = {str(item["pmid"]): item for item in articles}
    tracker.set(run_name)
    for attempt in range(1, 4):
        missing = [
            article for pmid, article in expected.items()
            if not record_ok(items / f"{pmid}.json")
        ]
        if not missing:
            break
        input_path = task_dir / f"input_attempt{attempt}.jsonl"
        write_jsonl(input_path, missing)
        output_dir = task_dir / f"agent_output_attempt{attempt}"
        command = [
            str(ROOT / ".venv-cognitive/bin/python"), "-m", "cognitive_agent.agent",
            "--input", str(input_path), "--limit", str(len(missing)),
            "--run-id", f"{run_name}_a{attempt}", "--output-dir", str(output_dir),
            "--skip-neo4j-write", "--api-base", os.environ["GEMINI_API_BASE"],
            "--model-id", os.environ["GEMINI_MODEL"], "--max-workers", str(max_workers),
            "--extraction-inner-max-workers", "1", "--reflection-interval", "0",
            "--article-checkpoint-dir", str(items), "--disable-shadow-router",
            "--aux-primary-model", os.environ.get("AUX_PRIMARY_MODEL", "deepseek-v4-flash"),
            "--aux-critic-model", os.environ.get("AUX_CRITIC_MODEL", "qwen3.6-flash"),
            *common, *variant,
        ]
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
        if result.returncode and not any(record_ok(items / f"{item['pmid']}.json") for item in missing):
            raise RuntimeError(f"agent task {run_name} failed without progress (exit {result.returncode})")
    missing_ids = [pmid for pmid in expected if not record_ok(items / f"{pmid}.json")]
    if missing_ids:
        raise RuntimeError(f"agent task {run_name} incomplete after retries: {missing_ids[:10]}")
    return [json.loads((items / f"{pmid}.json").read_text(encoding="utf-8")) for pmid in expected]


def verification_relations(record: dict) -> list[dict]:
    return record.get("phases", {}).get("verification", {}).get("relations", []) or []


def quick_counts(records: list[dict], gold_by_id: dict[str, dict]) -> dict:
    rows = []
    totals = Counter()
    for record in records:
        pmid = str(record["pmid"])
        pred = relation_map(verification_relations(record))
        gold = relation_map(gold_by_id[pmid].get("relations", []) or [])
        pred_strict = {key for key, value in pred.items() if value.get("import_ready")}
        gold_strict = {key for key, value in gold.items() if value.get("import_ready")}
        row = {
            "tp": len(set(pred) & set(gold)), "fp": len(set(pred) - set(gold)),
            "fn": len(set(gold) - set(pred)),
            "strict_tp": len(pred_strict & gold_strict),
            "strict_fp": len(pred_strict - gold_strict),
            "strict_fn": len(gold_strict - pred_strict),
            "dangerous_writes": sum(
                bool(set(item.get("quality_flags", []) or []) & HARD_FLAGS)
                for item in pred.values() if item.get("import_ready")
            ),
        }
        rows.append(row)
        totals.update(row)
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    strict_precision = totals["strict_tp"] / max(totals["strict_tp"] + totals["strict_fp"], 1)
    return {
        "rows": rows, "relation_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "strict_precision": strict_precision, "dangerous_writes": totals["dangerous_writes"],
        "errors": totals["fp"] + totals["fn"],
    }


def bootstrap_non_negative(base_rows: list[dict], variant_rows: list[dict], seed: int) -> float:
    rng = random.Random(seed)
    non_negative = 0
    for _ in range(1000):
        indexes = [rng.randrange(len(base_rows)) for _ in base_rows]
        def f1(rows: list[dict]) -> float:
            tp = sum(rows[i]["tp"] for i in indexes)
            fp = sum(rows[i]["fp"] for i in indexes)
            fn = sum(rows[i]["fn"] for i in indexes)
            p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            return 2 * p * r / max(p + r, 1e-12)
        non_negative += f1(variant_rows) >= f1(base_rows)
    return non_negative / 1000


def gold_error_cards(records: list[dict], gold_by_id: dict[str, dict]) -> list[ErrorCard]:
    cards: list[ErrorCard] = []
    for record in records:
        pmid = str(record["pmid"])
        pred = relation_map(verification_relations(record))
        gold = relation_map(gold_by_id[pmid].get("relations", []) or [])
        for index, key in enumerate(sorted(set(pred) - set(gold))):
            item = pred[key]
            cards.append(ErrorCard(
                error_id=f"fp-{pmid}-{index}", pmid=pmid, category="false_positive",
                observed={k: item.get(k, "") for k in (
                    "subject_type", "predicate", "object_type", "direction", "evidence_role"
                )}, reason_codes=list(item.get("quality_flags", []) or []), split="induction",
            ))
        for index, key in enumerate(sorted(set(gold) - set(pred))):
            item = gold[key]
            cards.append(ErrorCard(
                error_id=f"fn-{pmid}-{index}", pmid=pmid, category="false_negative",
                expected={k: item.get(k, "") for k in (
                    "subject_type", "predicate", "object_type", "direction"
                )}, reason_codes=["gold_relation_not_recovered"], split="induction",
            ))
    categories: defaultdict[str, list[ErrorCard]] = defaultdict(list)
    for card in cards:
        categories[card.category].append(card)
    limited = []
    for category in sorted(categories):
        limited.extend(categories[category][:60])
    return limited[:120]


def write_bundle(path: Path, bundle: RuleBundle, audit: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    atomic_json(path / "active_rules.json", bundle.to_dict())
    (path / "RULE_CONTEXT.md").write_text(render_rule_context(bundle), encoding="utf-8")
    atomic_json(path / "rule_learning_audit.json", audit)


def fit_fold_artifacts(
    *, fold: dict, fold_dir: Path, by_id: dict[str, dict], gold_by_id: dict[str, dict],
    snapshot_records: dict[str, dict], snapshot_path: Path, tracker: StatusTracker,
    common: list[str], max_workers: int, manifest_hash: str,
) -> tuple[Path, Path]:
    final_bundle = fold_dir / "rules" / "active_rules.json"
    final_calibration = fold_dir / "conformal_calibration.json"
    if final_bundle.exists() and final_calibration.exists():
        return final_bundle, final_calibration
    induction_records = [snapshot_records[pmid] for pmid in fold["induction_pmids"]]
    cards = gold_error_cards(induction_records, gold_by_id)
    write_jsonl(fold_dir / "error_cards.jsonl", [asdict(item) for item in cards])
    validation_articles = [by_id[pmid] for pmid in fold["validation_pmids"]]
    calibration_articles = [by_id[pmid] for pmid in fold["calibration_pmids"]]
    base_variant = [
        "--execution-mode", "agent-v2", "--pair-classifier-mode", "active",
        "--rule-memory-mode", "off", "--evidence-entailment-mode", "off",
        "--risk-router-mode", "off", "--frozen-candidates", str(snapshot_path),
    ]
    baseline_val = run_agent_records(
        task_dir=fold_dir / "learning" / "baseline_validation",
        articles=validation_articles, run_name=f"fold{fold['fold']}_baseline_val",
        tracker=tracker, common=common, variant=base_variant, max_workers=max_workers,
    )
    baseline_cal = run_agent_records(
        task_dir=fold_dir / "learning" / "baseline_calibration",
        articles=calibration_articles, run_name=f"fold{fold['fold']}_baseline_cal",
        tracker=tracker, common=common, variant=base_variant, max_workers=max_workers,
    )
    base_val_metrics = quick_counts(baseline_val, gold_by_id)
    base_cal_metrics = quick_counts(baseline_cal, gold_by_id)
    registry = AuxModelRegistry.from_environment(
        primary_model=os.environ.get("AUX_PRIMARY_MODEL", "deepseek-v4-flash"),
        critic_model=os.environ.get("AUX_CRITIC_MODEL", "qwen3.6-flash"), timeout_s=90,
    )
    learner = RuleLearner(registry)
    bundle = RuleBundle(metadata={"fold": fold["fold"], "manifest_hash": manifest_hash})
    audits = []
    low_gain_rounds = 0
    best_f1 = base_val_metrics["relation_f1"]
    calibration_records = baseline_cal
    for round_index in range(1, 4):
        tracker.set(f"fold{fold['fold']}:rule_induction_round{round_index}")
        candidates, primary_result, rejected = learner.induce(cards)
        if primary_result.status == "PROTOCOL_INVALID":
            raise RuntimeError(
                f"fold {fold['fold']} rule induction protocol invalid: "
                f"{len(rejected)} rejected rules"
            )
        critic_results = []
        approved: list[SoftRule] = []
        for rule in candidates:
            critique = learner.critique(rule)
            critic_results.append(critique.to_dict())
            objections = critique.payload.get("safety_objections", []) if critique.status == "OK" else []
            rule.critic_model = critique.model_id
            rule.critic_approved = bool(
                critique.status == "OK" and critique.payload.get("approved") and not objections
            )
            if rule.critic_approved:
                rule.status = "shadow"
                approved.append(rule)
        candidate_bundle = RuleBundle(
            revision=bundle.revision + 1, status="shadow",
            rules=[*bundle.rules, *approved], previous_bundle_hash=bundle.bundle_hash,
            metadata={"fold": fold["fold"], "round": round_index, "manifest_hash": manifest_hash},
        )
        round_dir = fold_dir / "learning" / f"round{round_index}"
        write_bundle(round_dir / "candidate", candidate_bundle, {})
        rule_common = [
            "--execution-mode", "agent-v2", "--pair-classifier-mode", "active",
            "--evidence-entailment-mode", "off", "--risk-router-mode", "off",
            "--frozen-candidates", str(snapshot_path), "--rule-bundle",
            str(round_dir / "candidate" / "active_rules.json"),
        ]
        run_agent_records(
            task_dir=round_dir / "shadow", articles=validation_articles,
            run_name=f"fold{fold['fold']}_round{round_index}_shadow", tracker=tracker,
            common=common, variant=[*rule_common, "--rule-memory-mode", "shadow"],
            max_workers=max_workers,
        )
        for rule in candidate_bundle.rules:
            rule.status = "active"
        write_bundle(round_dir / "active_candidate", candidate_bundle, {})
        active_args = [
            *rule_common[:-1], str(round_dir / "active_candidate" / "active_rules.json"),
            "--rule-memory-mode", "active",
        ]
        variant_val = run_agent_records(
            task_dir=round_dir / "validation", articles=validation_articles,
            run_name=f"fold{fold['fold']}_round{round_index}_validation", tracker=tracker,
            common=common, variant=active_args, max_workers=max_workers,
        )
        variant_cal = run_agent_records(
            task_dir=round_dir / "calibration", articles=calibration_articles,
            run_name=f"fold{fold['fold']}_round{round_index}_calibration", tracker=tracker,
            common=common, variant=active_args, max_workers=max_workers,
        )
        val_metrics = quick_counts(variant_val, gold_by_id)
        cal_metrics = quick_counts(variant_cal, gold_by_id)
        validation_gate = {
            "errors_fixed": max(0, base_val_metrics["errors"] - val_metrics["errors"]),
            "new_regressions": max(0, val_metrics["errors"] - base_val_metrics["errors"]),
            "dangerous_writes": val_metrics["dangerous_writes"],
            "strict_precision_delta": val_metrics["strict_precision"] - base_val_metrics["strict_precision"],
        }
        calibration_gate = {
            "bootstrap_non_negative_probability": bootstrap_non_negative(
                base_cal_metrics["rows"], cal_metrics["rows"], SEED + round_index + fold["fold"] * 10,
            )
        }
        promoted = []
        promotion_rejections = list(rejected)
        for rule in approved:
            allowed, reasons = RulePromotionGate.decide(
                rule, validation=validation_gate, calibration=calibration_gate,
                shadow_completed=True,
            )
            rule.metrics = {**validation_gate, **calibration_gate}
            rule.status = "active" if allowed else "rejected"
            if allowed:
                promoted.append(rule)
            else:
                promotion_rejections.append({"rule_id": rule.rule_id, "reasons": reasons})
        gain = val_metrics["relation_f1"] - best_f1
        audits.append({
            "round": round_index, "primary": primary_result.to_dict(),
            "critics": critic_results, "candidate_count": len(candidates),
            "approved_count": len(approved), "promoted_count": len(promoted),
            "validation": validation_gate, "calibration": calibration_gate,
            "relation_f1": val_metrics["relation_f1"], "gain_vs_best": gain,
            "rejected": promotion_rejections,
        })
        if promoted:
            bundle = RuleBundle(
                revision=bundle.revision + 1, status="active",
                rules=[*bundle.rules, *promoted], previous_bundle_hash=bundle.bundle_hash,
                metadata={"fold": fold["fold"], "round": round_index, "manifest_hash": manifest_hash},
            )
            calibration_records = variant_cal
        if gain >= 0.005:
            best_f1 = val_metrics["relation_f1"]
            low_gain_rounds = 0
        else:
            low_gain_rounds += 1
        if not candidates or low_gain_rounds >= 2:
            break
    write_bundle(fold_dir / "rules", bundle, {
        "status": "COMPLETE", "rounds": audits, "fold": fold["fold"],
        "test_pmids_seen": [], "manifest_hash": manifest_hash,
    })
    calibration_examples = []
    for record in calibration_records:
        pmid = str(record["pmid"])
        gold_keys = set(relation_map(gold_by_id[pmid].get("relations", []) or []))
        gold_strict_keys = {
            key for key, value in relation_map(
                gold_by_id[pmid].get("relations", []) or []
            ).items() if value.get("import_ready")
        }
        study_type = rule_profile(by_id[pmid].get("title", ""), by_id[pmid].get("abstract", "")).primary_study_type
        verification = record.get("phases", {}).get("verification", {}) or {}
        semantic = float(verification.get("summary", {}).get("semantic_score", 0.0) or 0.0)
        for relation in verification.get("relations", []) or []:
            features = RiskFeatures(
                candidate_id=str(relation.get("candidate_id") or sha256_json(triple(relation))[:12]),
                relation_score=float(relation.get("relation_probability") or relation.get("classifier_confidence") or semantic),
                evidence_score=float(relation.get("evidence_confidence") or max(0.0, 1.0 - 0.25 * max(0, int(relation.get("evidence_level", 1) or 1) - 1))),
                verifier_passed=bool(relation.get("schema_valid") and relation.get("evidence_contiguous") and not relation.get("negated")),
                verifier_flags=list(relation.get("quality_flags", []) or []),
                local_label=str(relation.get("predicate", "")),
                rule_support=len(relation.get("rule_matches", []) or []),
                rule_conflict=any(item.get("action") in {"REJECT", "ABSTAIN", "REVIEW"} for item in relation.get("rule_matches", []) or []),
                study_type=study_type, predicate=str(relation.get("predicate", "unknown")),
                section=str(relation.get("evidence_role", "ABSTRACT")),
                semantic_only=True,
            )
            calibration_examples.append(CalibrationExample.from_features(
                features, error=triple(relation) not in gold_keys,
                risk_target="semantic",
            ))
            write_features = RiskFeatures(**{
                **features.__dict__,
                "verifier_passed": bool(relation.get("import_ready")),
                "semantic_only": False,
            })
            calibration_examples.append(CalibrationExample.from_features(
                write_features,
                error=(
                    not bool(relation.get("import_ready"))
                    or triple(relation) not in gold_strict_keys
                ),
                risk_target="write",
            ))
    calibration = ConformalCalibration(
        examples=calibration_examples,
        version=f"agent-v3-conformal-fold{fold['fold']}-v2-dual-risk",
        source_manifest_hash=manifest_hash,
    )
    atomic_json(final_calibration, calibration.to_dict())
    return final_bundle, final_calibration


def arm_args(
    arm: str, *, snapshot_path: Path, bundle: Path, calibration: Path, cache_path: Path,
) -> tuple[list[str], list[str]]:
    cache_mode = "off" if arm == "v3_no_cache" else "persistent"
    common = ["--extraction-cache-mode", cache_mode, "--extraction-cache-path", str(cache_path)]
    frozen = [] if arm == "v3_no_cache" else ["--frozen-candidates", str(snapshot_path)]
    if arm == "legacy":
        return common, [*frozen, "--execution-mode", "legacy", "--pair-classifier-mode", "off", "--rule-memory-mode", "off", "--evidence-entailment-mode", "off", "--risk-router-mode", "off"]
    if arm == "agent_v2_no_rules":
        return common, [*frozen, "--execution-mode", "agent-v2", "--pair-classifier-mode", "active", "--rule-memory-mode", "off", "--evidence-entailment-mode", "off", "--risk-router-mode", "off", "--second-llm-enabled"]
    if arm == "deepseek_always":
        return common, [*frozen, "--execution-mode", "agent-v2", "--pair-classifier-mode", "active", "--rule-memory-mode", "off", "--evidence-entailment-mode", "active", "--risk-router-mode", "off", "--second-llm-enabled", "--second-llm-mode", "always", "--disable-qwen-critic"]
    args = [
        *frozen, "--execution-mode", "agent-v2", "--pair-classifier-mode", "active",
        "--rule-memory-mode", "active", "--rule-bundle", str(bundle),
        "--evidence-entailment-mode", "active", "--risk-router-mode", "active",
        "--conformal-calibration", str(calibration), "--second-llm-enabled",
        "--second-llm-mode", "conditional",
        "--agent-mode", "recall",
    ]
    if arm == "v3_no_rule_memory":
        index = args.index("--rule-memory-mode")
        args[index + 1] = "off"
    elif arm == "v3_no_qwen_critic":
        args.append("--disable-qwen-critic")
    elif arm == "v3_no_evidence_selector":
        args.append("--disable-evidence-selector")
    elif arm == "v3_no_conformal_router":
        index = args.index("--risk-router-mode")
        args[index + 1] = "off"
    elif arm == "v3_no_causal_conflict":
        args.append("--disable-causal-conflict")
    return common, args


def char_iou(source: str, first: str, second: str) -> float:
    a, b = source.find(first), source.find(second)
    if a < 0 or b < 0:
        return 0.0
    a_end, b_end = a + len(first), b + len(second)
    overlap = max(0, min(a_end, b_end) - max(a, b))
    union = max(a_end, b_end) - min(a, b)
    return overlap / union if union else 0.0


def article_metric(record: dict, gold: dict, source: dict) -> dict:
    relations = verification_relations(record)
    pred = relation_map(relations)
    gold_relations = gold.get("relations", []) or []
    gold_map = relation_map(gold_relations)
    pred_entities = {
        (norm(item.get("mention")), str(item.get("type", item.get("entity_type", ""))))
        for item in record.get("phases", {}).get("verification", {}).get("entities", []) or []
    }
    gold_entities = {(norm(item.get("canonical", item.get("mention"))), str(item.get("type", ""))) for item in gold.get("entities", []) or []}
    pred_strict = {key for key, value in pred.items() if value.get("import_ready")}
    gold_strict = {key for key, value in gold_map.items() if value.get("import_ready")}
    evidence_exact = evidence_iou = endpoint_coverage = trigger_coverage = 0
    direction_confusion = 0
    matched = set(pred) & set(gold_map)
    full_source = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"
    for key in matched:
        p, g = pred[key], gold_map[key]
        pe, ge = str(p.get("evidence", "")), str(g.get("evidence", ""))
        evidence_exact += bool(pe and pe == ge)
        evidence_iou += bool(pe and ge and char_iou(full_source, pe, ge) >= 0.5)
        endpoint_coverage += bool(norm(p.get("subject")) in norm(pe) and norm(p.get("object")) in norm(pe))
        trigger_coverage += bool(p.get("evidence_trigger") or p.get("trigger_span"))
        direction_confusion += bool(g.get("direction") and p.get("direction") != g.get("direction"))
    routes = record.get("phases", {}).get("conformal_risk_router", {}).get("routes", []) or []
    route_counts = Counter(str(item.get("decision", "UNKNOWN")) for item in routes)
    verification = record.get("phases", {}).get("verification", {}) or {}
    entities = verification.get("entities", []) or []
    ambiguous = sum(bool(item.get("ambiguity_reason") or len(item.get("candidates", []) or []) > 1) for item in entities)
    collaboration = record.get("phases", {}).get("collaboration", {}) or {}
    evidence_audit = record.get("phases", {}).get("evidence_entailment", {}).get("audit", {}) or {}
    actions = record.get("phases", {}).get("agent_v2", {}).get("action_trace", []) or []
    predicate_counts = {}
    for predicate in sorted({item.get("predicate") for item in [*relations, *gold_relations] if item.get("predicate")}):
        pred_keys = {key for key in pred if key[2] == predicate}
        gold_keys = {key for key in gold_map if key[2] == predicate}
        predicate_counts[predicate] = {
            "tp": len(pred_keys & gold_keys), "fp": len(pred_keys - gold_keys),
            "fn": len(gold_keys - pred_keys), "support": len(gold_keys),
        }
    confidence_correct = []
    for key, item in pred.items():
        confidence = float(item.get("relation_probability") or item.get("classifier_confidence") or 0.5)
        confidence_correct.append([confidence, int(key in gold_map)])
    timing = record.get("timing", {}) or {}
    remote_results = [
        evidence_audit.get("deepseek", {}) or {}, evidence_audit.get("qwen", {}) or {}, collaboration,
    ]
    attempted = sum(int(bool(item.get("attempts") or item.get("triggered"))) for item in remote_results)
    successful = sum(int(item.get("status") == "OK") for item in remote_results)
    failed = sum(int(item.get("status") in {"FALLBACK", "ERROR"}) for item in remote_results)
    state_changes = sum(int(collaboration.get("merge", {}).get(key, 0) or 0) for key in ("relation_additions", "relation_edits", "relation_rejections"))
    return {
        "pmid": str(record["pmid"]),
        "entity_tp": len(pred_entities & gold_entities), "entity_fp": len(pred_entities - gold_entities), "entity_fn": len(gold_entities - pred_entities),
        "tp": len(set(pred) & set(gold_map)), "fp": len(set(pred) - set(gold_map)), "fn": len(set(gold_map) - set(pred)),
        "strict_tp": len(pred_strict & gold_strict), "strict_fp": len(pred_strict - gold_strict), "strict_fn": len(gold_strict - pred_strict),
        "evidence_exact_tp": evidence_exact, "evidence_iou_tp": evidence_iou,
        "evidence_pred": len(relations), "evidence_gold": len(gold_relations),
        "evidence_contiguous": sum(bool(item.get("evidence") and str(item.get("evidence")) in full_source) for item in relations),
        "endpoint_coverage": endpoint_coverage, "trigger_coverage": trigger_coverage,
        "direction_confusion": direction_confusion,
        "zero_relation_gold": int(not gold_relations), "zero_relation_correct": int(not gold_relations and not relations),
        "hard_negative_fp": len(relations) if not gold_relations else 0,
        "dangerous_writes": sum(bool(set(item.get("quality_flags", []) or []) & HARD_FLAGS) for item in relations if item.get("import_ready")),
        "schema_violations": sum(not bool(item.get("schema_valid", True)) for item in relations),
        "negation_background_method_fp": sum(bool(set(item.get("quality_flags", []) or []) & {"hard_negation", "background_only", "method_only"}) for item in relations),
        "linking_ambiguous": ambiguous, "linking_total": len(entities),
        "route_counts": dict(route_counts), "route_total": len(routes),
        "confidence_correct": confidence_correct,
        "actions": len(actions),
        "budget_escalations": len(record.get("phases", {}).get("agent_v2", {}).get("budget_escalations", []) or []),
        "termination_reason": record.get("phases", {}).get("agent_v2", {}).get("termination", {}).get("reason", ""),
        "remote_attempted": attempted, "remote_successful": successful, "remote_failed": failed,
        "remote_retried": sum(int(item.get("invalid_json_attempts", 0) or 0) for item in remote_results),
        "aux_calls": attempted, "state_changes": state_changes,
        "zero_change_calls": int(attempted > 0 and state_changes == 0),
        "prompt_tokens": sum(int(item.get("prompt_tokens", 0) or 0) for item in remote_results),
        "output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in remote_results),
        "latency_s": float(timing.get("total_s", 0.0) or 0.0),
        "phase_latency": {key: float(value or 0.0) for key, value in timing.items() if key.endswith("_s")},
        "predicate_counts": predicate_counts,
        "candidate_source": record.get("phases", {}).get("extraction", {}).get("candidate_source", "unknown"),
        "predicate_direction_errors": direction_confusion,
        "article_exact": int(set(pred) == set(gold_map)),
    }


def prediction_hash(records: list[dict]) -> str:
    payload = [{
        "pmid": str(record["pmid"]),
        "entities": record.get("phases", {}).get("verification", {}).get("entities", []),
        "relations": verification_relations(record),
    } for record in records]
    return sha256_json(payload)


def write_reports(run_dir: Path, manifest: dict, warm_audit: dict) -> None:
    metrics_path = run_dir / "metrics" / "ablation_report.json"
    if not metrics_path.exists():
        return
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    def table(language: str) -> str:
        zh = language == "zh"
        title = "# Agent v3：100 篇五折交叉拟合消融结果" if zh else "# Agent v3: 100-article cross-fitted ablation results"
        note = (
            "本报告使用 LLM 严格裁决内部金标，不替代专家盲审。" if zh
            else "This report uses an internally LLM-adjudicated gold set and does not replace expert blind review."
        )
        lines = [title, "", note, "", f"- Manifest: `{manifest['manifest_hash']}`", f"- Seed: `{manifest['seed']}`", f"- Warm replay consistent: `{warm_audit.get('all_consistent', False)}`", "", "| Arm | Relation P/R/F1 | Strict P/R/F1 | Evidence IoU P/R/F1 | P95 latency | Aux/article |", "|---|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            m = report["variants"][arm]["metrics"]
            lines.append(
                f"| {arm} | {m['relation_precision']:.3f}/{m['relation_recall']:.3f}/{m['relation_f1']:.3f} | "
                f"{m['strict_precision']:.3f}/{m['strict_recall']:.3f}/{m['strict_f1']:.3f} | "
                f"{m['evidence_iou_precision']:.3f}/{m['evidence_iou_recall']:.3f}/{m['evidence_iou_f1']:.3f} | "
                f"{m['latency']['p95']:.2f}s | {m['avg_aux_calls']:.2f} |"
            )
        return "\n".join(lines) + "\n"
    (run_dir / "RESULTS.zh-CN.md").write_text(table("zh"), encoding="utf-8")
    (run_dir / "RESULTS.md").write_text(table("en"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / "runner.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
    except FileExistsError:
        raise SystemExit(f"runner lock already exists: {lock}")
    (run_dir / "runner.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    tracker = StatusTracker(run_dir, total_article_arms=1300)
    tracker.start()
    try:
        load_local_env(args.env_file)
        gold_path = ROOT / "gold_annotations/pubmed_200_gold_v2_strict.jsonl"
        source_path = ROOT / "extraction_output/pubmed_converted_500.jsonl"
        blind_path = ROOT / "gold_annotations/blind50/blind50_preregistered_seed20260814.json"
        gold_rows = load_jsonl(gold_path)
        source_rows = load_jsonl(source_path)
        if len(gold_rows) != 200:
            raise RuntimeError("strict gold file must contain exactly 200 articles")
        source_by_id = {str(item["pmid"]): item for item in source_rows}
        gold_by_id = {str(item["pmid"]): item for item in gold_rows}
        by_id = {pmid: source_by_id[pmid] for pmid in gold_by_id}
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = build_manifest(gold_rows, source_rows)
            atomic_json(manifest_path, manifest)
        validate_manifest(manifest, blind_path)
        tracker.set("three_model_preflight")
        preflight_payload = preflight()
        atomic_json(run_dir / "preflight.json", preflight_payload)
        common_cache = run_dir / "cache" / "primary.sqlite3"
        common = ["--extraction-cache-mode", "persistent", "--extraction-cache-path", str(common_cache)]
        tracker.set("freeze_primary_candidates_all200")
        snapshot_records_list = run_agent_records(
            task_dir=run_dir / "candidate_freeze", articles=[by_id[str(item["pmid"])] for item in gold_rows],
            run_name="candidate_freeze_all200", tracker=tracker, common=common,
            variant=["--execution-mode", "legacy", "--pair-classifier-mode", "off", "--rule-memory-mode", "off", "--evidence-entailment-mode", "off", "--risk-router-mode", "off"],
            max_workers=args.max_workers,
        )
        snapshot_records = {str(item["pmid"]): item for item in snapshot_records_list}
        snapshot_path = run_dir / "frozen_candidates.json"
        if not snapshot_path.exists():
            candidates = {
                pmid: record.get("phases", {}).get("extraction", {})
                for pmid, record in snapshot_records.items()
            }
            atomic_json(snapshot_path, {
                "version": "agent-v3-frozen-candidates-v1",
                "manifest_hash": manifest["manifest_hash"],
                "model": os.environ["GEMINI_MODEL"],
                "api_base": os.environ["GEMINI_API_BASE"],
                "candidates": candidates,
                "candidate_hash": sha256_json(candidates),
            })
        bundles: dict[int, Path] = {}
        calibrations: dict[int, Path] = {}
        for fold in manifest["folds"]:
            fold_dir = run_dir / "folds" / f"fold{fold['fold']}"
            bundle, calibration = fit_fold_artifacts(
                fold=fold, fold_dir=fold_dir, by_id=by_id, gold_by_id=gold_by_id,
                snapshot_records=snapshot_records, snapshot_path=snapshot_path,
                tracker=tracker, common=common, max_workers=args.max_workers,
                manifest_hash=manifest["manifest_hash"],
            )
            bundles[fold["fold"]] = bundle
            calibrations[fold["fold"]] = calibration
        arm_records: dict[str, list[dict]] = {arm: [] for arm in ARMS}
        full_by_fold: dict[int, list[dict]] = {}
        for fold in manifest["folds"]:
            fold_number = fold["fold"]
            test_articles = [by_id[pmid] for pmid in fold["test_pmids"]]
            for arm in ARMS:
                cache_path = run_dir / "cache" / f"fold{fold_number}_{arm}.sqlite3"
                arm_common, variant = arm_args(
                    arm, snapshot_path=snapshot_path, bundle=bundles[fold_number],
                    calibration=calibrations[fold_number], cache_path=cache_path,
                )
                records = run_agent_records(
                    task_dir=run_dir / "arms" / arm / f"fold{fold_number}",
                    articles=test_articles, run_name=f"fold{fold_number}_{arm}", tracker=tracker,
                    common=arm_common, variant=variant, max_workers=args.max_workers,
                )
                arm_records[arm].extend(records)
                if arm == "full_v3":
                    full_by_fold[fold_number] = records
        warm_audit = {"folds": {}, "all_consistent": True}
        for fold in manifest["folds"]:
            fold_number = fold["fold"]
            test_articles = [by_id[pmid] for pmid in fold["test_pmids"]]
            expected_hash = prediction_hash(full_by_fold[fold_number])
            hashes = []
            for replay in range(1, 4):
                cache_path = run_dir / "cache" / f"fold{fold_number}_full_v3.sqlite3"
                replay_common, replay_variant = arm_args(
                    "full_v3", snapshot_path=snapshot_path, bundle=bundles[fold_number],
                    calibration=calibrations[fold_number], cache_path=cache_path,
                )
                records = run_agent_records(
                    task_dir=run_dir / "warm_replays" / f"replay{replay}" / f"fold{fold_number}",
                    articles=test_articles, run_name=f"warm{replay}_fold{fold_number}", tracker=tracker,
                    common=replay_common, variant=replay_variant, max_workers=args.max_workers,
                )
                hashes.append(prediction_hash(records))
                if any(record.get("phases", {}).get("extraction", {}).get("candidate_source") != "frozen_snapshot" for record in records):
                    raise RuntimeError("warm replay attempted live primary extraction")
            consistent = all(item == expected_hash for item in hashes)
            warm_audit["folds"][str(fold_number)] = {
                "expected_hash": expected_hash, "replay_hashes": hashes,
                "consistent": consistent, "primary_remote_calls": 0,
            }
            warm_audit["all_consistent"] &= consistent
        atomic_json(run_dir / "warm_replay_audit.json", warm_audit)
        if not warm_audit["all_consistent"]:
            raise RuntimeError("warm replay result hashes are inconsistent")
        metric_specs = []
        metrics_dir = run_dir / "metrics" / "article_metrics"
        for arm, records in arm_records.items():
            rows = [
                article_metric(record, gold_by_id[str(record["pmid"])], by_id[str(record["pmid"])])
                for record in records
            ]
            if len(rows) != 100:
                raise RuntimeError(f"arm {arm} has {len(rows)} rows instead of 100")
            path = metrics_dir / f"{arm}.json"
            atomic_json(path, {"articles": rows})
            metric_specs.extend(["--variant", f"{arm}={path}"])
        tracker.set("bootstrap_and_reports")
        evaluator = ROOT / "scripts/evaluate_agent_v3_experiments.py"
        report_path = run_dir / "metrics" / "ablation_report.json"
        result = subprocess.run([
            str(ROOT / ".venv-cognitive/bin/python"), str(evaluator), *metric_specs,
            "--baseline", "legacy", "--iterations", "10000", "--seed", str(SEED),
            "--output", str(report_path),
        ], cwd=ROOT, env=os.environ.copy(), check=False)
        if result.returncode:
            raise RuntimeError(f"statistical evaluator failed with exit {result.returncode}")
        write_reports(run_dir, manifest, warm_audit)
        atomic_json(run_dir / "completion.json", {
            "status": "COMPLETE", "finished_at": utc_now(),
            "manifest_hash": manifest["manifest_hash"], "arms": list(ARMS),
            "articles_per_arm": 100, "dangerous_write_guard": "dry_run_only",
        })
        tracker.close("COMPLETE")
        return 0
    except Exception as exc:
        tracker.close("FAILED", str(exc)[:1000])
        atomic_json(run_dir / "failure.json", {"failed_at": utc_now(), "error": str(exc)})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
