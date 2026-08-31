#!/usr/bin/env python3
"""Fit precision-first adjudication thresholds on the frozen calibration split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.schema.predicate_cards import RELATION_CARDS
from scripts.evaluate_gold200_unified import aliases, load_jsonl, records_from_result, relation_key


CANDIDATE_THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_threshold(
    rows: list[tuple[float, bool]], *, min_precision: float, min_predictions: int,
) -> tuple[float | None, dict[str, Any]]:
    eligible: list[tuple[int, float, int, int]] = []
    audits = []
    for threshold in CANDIDATE_THRESHOLDS:
        selected = [label for confidence, label in rows if confidence >= threshold]
        tp = sum(selected)
        total = len(selected)
        precision = tp / total if total else None
        audits.append({
            "threshold": threshold, "predictions": total, "tp": tp,
            "precision": precision,
        })
        if total >= min_predictions and precision is not None and precision >= min_precision:
            eligible.append((total, -threshold, tp, len(eligible)))
    if not eligible:
        return None, {"selected": None, "candidates": audits}
    total, negative_threshold, _, _ = max(eligible)
    selected_threshold = -negative_threshold
    return selected_threshold, {
        "selected": selected_threshold,
        "selected_predictions": total,
        "candidates": audits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--gold", type=Path,
        default=ROOT / "gold_annotations/pubmed_200_gold_v2_candidate_view.jsonl",
    )
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "extraction_output/pubmed_converted_500.jsonl",
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        default=ROOT / "gold_annotations/splits/agent_v3_dev_manifest_seed20260814.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--min-predictions", type=int, default=5)
    args = parser.parse_args()

    for path in (args.records, args.gold, args.source, args.split_manifest):
        if "blind50" in str(path.resolve()).casefold():
            raise ValueError("Blind50 is forbidden for predicate threshold fitting")

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    calibration_pmids = {
        str(item.get("pmid", ""))
        for item in manifest.get("records", [])
        if item.get("split") == "calibration"
    }
    records = records_from_result(args.records)
    unexpected = sorted({str(item.get("pmid", "")) for item in records} - calibration_pmids)
    if unexpected:
        raise ValueError(f"non-calibration PMID detected: {unexpected[:5]}")

    gold_by_pmid = {str(item.get("pmid", "")): item for item in load_jsonl(args.gold)}
    source_by_pmid = {str(item.get("pmid", "")): item for item in load_jsonl(args.source)}
    labelled: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for record in records:
        pmid = str(record.get("pmid", ""))
        gold = gold_by_pmid.get(pmid, {"relations": [], "entities": []})
        source = source_by_pmid.get(pmid, {})
        text = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"
        alias_map = aliases(gold, text)
        gold_keys = {relation_key(item, alias_map) for item in gold.get("relations", []) or []}
        relations = (
            record.get("phases", {}).get("verification", {}).get("relations", []) or []
        )
        for relation in relations:
            verdict = str(relation.get("adjudication_verdict", "") or "").upper()
            reason = str(relation.get("adjudication_reason_code", "") or "").upper()
            confidence = float(relation.get("adjudication_confidence", 0.0) or 0.0)
            if verdict != "SUPPORTED" or reason != "EXPLICIT_DIRECT_RELATION":
                continue
            predicate = str(relation.get("predicate", "") or "").upper()
            if predicate not in RELATION_CARDS:
                continue
            labelled[predicate].append((confidence, relation_key(relation, alias_map) in gold_keys))

    global_rows = [item for rows in labelled.values() for item in rows]
    global_threshold, global_audit = choose_threshold(
        global_rows,
        min_precision=args.min_precision,
        min_predictions=args.min_predictions,
    )
    if global_threshold is None:
        global_threshold = 0.90
        global_fallback = "insufficient_global_precision_support"
    else:
        global_fallback = ""

    thresholds: dict[str, float] = {}
    predicate_audit: dict[str, Any] = {}
    for predicate in sorted(RELATION_CARDS):
        selected, audit = choose_threshold(
            labelled.get(predicate, []),
            min_precision=args.min_precision,
            min_predictions=args.min_predictions,
        )
        thresholds[predicate] = selected if selected is not None else global_threshold
        predicate_audit[predicate] = {
            **audit,
            "examples": len(labelled.get(predicate, [])),
            "fallback": "" if selected is not None else "global",
        }

    output = {
        "artifact_type": "predicate_adjudication_thresholds",
        "version": "predicate-thresholds-v1",
        "thresholds": thresholds,
        "global_threshold": global_threshold,
        "global_fallback": global_fallback,
        "minimum_precision": args.min_precision,
        "minimum_predictions": args.min_predictions,
        "calibration_pmids": len({str(item.get("pmid", "")) for item in records}),
        "manifest_sha256": sha256(args.split_manifest),
        "gold_sha256": sha256(args.gold),
        "records_sha256": sha256(args.records),
        "global_audit": global_audit,
        "predicate_audit": predicate_audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "thresholds": thresholds,
        "calibration_pmids": output["calibration_pmids"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
