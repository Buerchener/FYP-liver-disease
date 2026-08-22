#!/usr/bin/env python3
"""Given-entity-pair evaluation, aligned with the ANCHOR-RE / SemRepGS protocol.

The task is reformulated as relation *classification* over gold entity pairs:
every gold (subject, object) pair is one sample, and the model's prediction for
that pair is compared against the gold relation class(es).  This isolates the
relation decision protocol (lattice -> claim gate -> predicate judge ->
verifier) from NER / candidate-recall error, which is exactly what ANCHOR-RE
codes by starting from gold entity pairs.

Protocol (mirrors ANCHOR-RE, Section 3.8):
- unit          = gold entity pair (subject, object), per document
- no_rel        = a gold entity pair with NO relation in gold          (excluded here,
                 reported separately as zero-pair count)
- TP            = model predicts a positive predicate for the pair and gold has that predicate
- FP            = model predicts a positive predicate for the pair but gold has NO such
                 predicate (or a different one)
- FN            = gold has a predicate for the pair but the model predicts nothing
                 (or a different predicate)
- micro P/R/F1  = aggregate TP/FP/FN over all pairs
- macro P/R/F1  = per-predicate P/R/F1 averaged over the positive predicates

Evaluation is run against a frozen agent result (records[].phases.verification)
and the v2-strict gold.  Entity matching is alias-aware (reuses score_arm's
canonical endpoint logic).

Usage:
    python3 scripts/evaluate_entity_pair.py \
        --pred path/to/agent_results.json \
        [--gold gold_annotations/pubmed_200_gold_v2_strict.jsonl] \
        [--limit 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.extraction_quality import normalize_surface  # noqa: E402
from cognitive_agent.abbreviation_detector import AbbreviationDetector  # noqa: E402
from scripts.run_pairwise_judge_experiments import (  # noqa: E402
    GOLD_PATH, PREDICATES, load_jsonl, _prf, _aliases, _canonical_endpoint,
)

# Predicates that ANCHOR-RE/SemRepGS-style protocols treat as positive (all of
# our schema predicates are positive; no_rel is the implicit "no relation").
POSITIVE_PREDICATES = set(PREDICATES)


def _load_gold(path: Path) -> dict[str, dict]:
    return {str(row["pmid"]): row for row in load_jsonl(path)}


def _gold_pairs(gold: dict, aliases: dict) -> dict[tuple, str]:
    """gold entity pairs that carry at least one positive relation.

    Returns {canonical_pair: predicate}.  A pair with multiple predicates keeps
    the first (multiclass simplification, matching ANCHOR-RE's single-label
    framing).
    """
    pairs: dict[tuple, str] = {}
    for rel in gold.get("relations", []) or []:
        pred = str(rel.get("predicate", "")).upper()
        if pred not in POSITIVE_PREDICATES:
            continue
        key = _canonical_pair(rel, aliases)
        if key not in pairs:
            pairs[key] = pred
    return pairs


def _canonical_pair(rel: dict, aliases: dict) -> tuple[str, str]:
    return (
        _canonical_endpoint(str(rel.get("subject", "")), str(rel.get("subject_type", "")), aliases),
        _canonical_endpoint(str(rel.get("object", "")), str(rel.get("object_type", "")), aliases),
    )


def _predicted_pairs(relations: list[dict], aliases: dict) -> dict[tuple, str]:
    """Model-predicted positive (subject, object) pairs -> predicate."""
    pairs: dict[tuple, str] = {}
    for rel in relations:
        pred = str(rel.get("predicate", "")).upper()
        if pred not in POSITIVE_PREDICATES:
            continue
        key = _canonical_pair(rel, aliases)
        if key not in pairs:
            pairs[key] = pred
    return pairs


def score_entity_pair(
    records: list[dict], gold_by_pmid: dict, source_by_pmid: dict,
) -> dict[str, object]:
    rel = Counter()
    pred_counts: dict[str, Counter] = defaultdict(Counter)
    zero_pair_docs = 0
    num_gold_pairs = 0
    num_pred_pairs = 0

    for record in records:
        pmid = str(record.get("pmid", ""))
        gold = gold_by_pmid.get(pmid)
        if gold is None:
            continue
        source = source_by_pmid.get(pmid, {})
        text = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"
        aliases = _aliases(gold, text)
        gold_pairs = _gold_pairs(gold, aliases)
        phases = record.get("phases", {}) or {}
        verification = phases.get("verification", {}) or {}
        pred_pairs = _predicted_pairs(verification.get("relations", []) or [], aliases)

        num_gold_pairs += len(gold_pairs)
        num_pred_pairs += len(pred_pairs)

        if not gold_pairs:
            zero_pair_docs += 1

        # ANCHOR-RE-style entity-pair TP/FP/FN.
        for key, gold_pred in gold_pairs.items():
            model_pred = pred_pairs.get(key)
            if model_pred is None:
                rel["fn"] += 1
                pred_counts[gold_pred]["fn"] += 1
            elif model_pred == gold_pred:
                rel["tp"] += 1
                pred_counts[gold_pred]["tp"] += 1
            else:
                # Model predicts a different positive predicate for the pair.
                rel["fp"] += 1
                rel["fn"] += 1
                pred_counts[gold_pred]["fn"] += 1
                pred_counts[model_pred]["fp"] += 1
        # Model-predicted pairs that have NO gold relation at all.
        for key, model_pred in pred_pairs.items():
            if key not in gold_pairs:
                rel["fp"] += 1
                pred_counts[model_pred]["fp"] += 1

    predicate_metrics = {
        pred: _prf(c["tp"], c["fp"], c["fn"]) for pred, c in pred_counts.items()
    }
    _preds = list(predicate_metrics.values())
    macro = (
        {
            "precision": sum(pm["precision"] for pm in _preds) / len(_preds),
            "recall": sum(pm["recall"] for pm in _preds) / len(_preds),
            "f1": sum(pm["f1"] for pm in _preds) / len(_preds),
        }
        if _preds else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    )

    return {
        "protocol": "given-entity-pair (ANCHOR-RE/SemRepGS aligned)",
        "unit": "gold entity pair",
        "micro": _prf(rel["tp"], rel["fp"], rel["fn"]),
        "macro": macro,
        "num_gold_pairs": num_gold_pairs,
        "num_predicted_pairs": num_pred_pairs,
        "zero_relation_pair_docs": zero_pair_docs,
        "predicate_metrics": predicate_metrics,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="agent_results JSON")
    ap.add_argument("--gold", type=Path, default=GOLD_PATH)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    gold = _load_gold(args.gold)
    gold_rows = list(gold.values())[: args.limit]
    gold_by_pmid = {str(g["pmid"]): g for g in gold_rows}
    source_by_pmid = {
        str(item["pmid"]): item
        for item in load_jsonl(ROOT / "extraction_output" / "pubmed_converted_500.jsonl")
    }
    evaluated_pmids = set(gold_by_pmid)

    payload = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    records = [r for r in records if str(r.get("pmid", "")) in evaluated_pmids]

    metrics = score_entity_pair(records, gold_by_pmid, source_by_pmid)
    pretty = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(pretty, encoding="utf-8")
    print(pretty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())