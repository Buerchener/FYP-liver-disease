#!/usr/bin/env python3
"""Produce a relation-by-relation audit ledger for a completed Gold-200 run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.extraction_quality import normalize_surface
from cognitive_agent.provider_errors import classify_provider_error
from scripts.evaluate_gold200_unified import (
    SOURCE_PATH, aliases, load_jsonl, records_from_result, relation_key, semantic_kept,
)

GOLD_PATH = ROOT / "gold_annotations/pubmed_200_gold_v2_candidate_view.jsonl"
SYMMETRIC = frozenset({"INTERACTS_WITH"})


def endpoint_key(value: str, kind: str, alias_map: dict) -> tuple[str, str]:
    normalized = normalize_surface(value)
    candidates = [item for item in alias_map.get(normalized, set()) if item[1] == kind]
    return candidates[0] if len(candidates) == 1 else (normalized, kind)


def pair_matches(candidate: dict, gold_key: tuple, alias_map: dict) -> bool:
    left = endpoint_key(str(candidate.get("subject", "")), str(candidate.get("subject_type", "")), alias_map)
    right = endpoint_key(str(candidate.get("object", "")), str(candidate.get("object_type", "")), alias_map)
    expected = ((gold_key[0], gold_key[1]), (gold_key[3], gold_key[4]))
    if (left, right) == expected:
        return True
    return gold_key[2] in SYMMETRIC and (right, left) == expected


def compact(items: list[dict], limit: int = 4) -> list[dict]:
    keys = (
        "candidate_id", "subject", "subject_type", "predicate", "object", "object_type",
        "label", "confidence", "relation_probability", "no_relation_probability", "reason_codes",
        "semantic_status", "factual_status", "write_status", "write_contract_valid", "quality_flags",
        "evidence", "error", "prediction",
    )
    return [{key: item.get(key) for key in keys if key in item} for item in items[:limit]]


def classify_gold_relation(
    gold_relation: dict, phases: dict, alias_map: dict,
) -> tuple[str, dict[str, Any]]:
    target = relation_key(gold_relation, alias_map)
    raw_relations = list((phases.get("extraction", {}) or {}).get("relations", []) or [])
    extraction_error = str((phases.get("extraction", {}) or {}).get("error", "") or "")
    projected = list((phases.get("relation_candidate_projection", {}) or {}).get("relations", []) or [])
    verification = list((phases.get("verification", {}) or {}).get("relations", []) or [])
    pair_phase = phases.get("relation_pair_classification", {}) or {}
    candidates = list(pair_phase.get("candidates", []) or [])
    predictions = {str(item.get("candidate_id", "")): item for item in pair_phase.get("predictions", []) or []}
    raw_matches = [item for item in raw_relations if relation_key(item, alias_map) == target]
    projected_matches = [item for item in projected if relation_key(item, alias_map) == target]
    final_matches = [item for item in verification if relation_key(item, alias_map) == target and semantic_kept(item)]
    candidate_matches = [item for item in candidates if pair_matches(item, target, alias_map)]
    candidate_audit = [
        {**item, **({"prediction": predictions.get(str(item.get("candidate_id", "")), {})})}
        for item in candidate_matches
    ]

    raw_entities = list((phases.get("extraction", {}) or {}).get("entities", []) or [])
    source_entities = {
        endpoint_key(str(item.get("mention", "")), str(item.get("type", "")), alias_map)
        for item in raw_entities
    }
    expected_entities = {(target[0], target[1]), (target[3], target[4])}
    surface_entities = {normalize_surface(str(item.get("mention", ""))) for item in raw_entities}
    endpoints_present = expected_entities <= source_entities
    surfaces_present = {
        normalize_surface(str(gold_relation.get("subject", ""))),
        normalize_surface(str(gold_relation.get("object", ""))),
    } <= surface_entities

    if extraction_error:
        stage = "PRIMARY_EXTRACTION_FAILED"
    elif final_matches:
        stage = "TP_FINAL_SEMANTIC"
    elif projected_matches or raw_matches:
        stage = "DROPPED_AFTER_RAW_OR_PROJECTED_TRIPLE"
    elif candidate_matches:
        labels = {str(predictions.get(str(item.get("candidate_id", "")), {}).get("label", "")) for item in candidate_matches}
        stage = "PAIR_CLASSIFIER_NO_RELATION" if labels <= {"NO_RELATION", ""} else "PAIR_PREDICATE_DIVERGENCE_OR_DROPPED"
    elif endpoints_present:
        stage = "PAIR_NOT_GENERATED_FROM_PRESENT_ENDPOINTS"
    elif surfaces_present:
        stage = "ENDPOINT_TYPE_MISMATCH"
    else:
        stage = "GOLD_ENDPOINT_MISSING_FROM_RAW_EXTRACTION"
    details = {
        "raw_matches": compact(raw_matches), "projected_matches": compact(projected_matches),
        "final_matches": compact(final_matches), "pair_candidates": compact(candidate_audit),
        "raw_endpoint_count": len(raw_entities), "endpoints_present_with_gold_types": endpoints_present,
        "endpoint_surfaces_present": surfaces_present,
        "primary_extraction_error": extraction_error,
        "final_relation_count": len(verification),
        "core_selection": phases.get("relation_core_selection", {}),
    }
    return stage, details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    gold_rows = load_jsonl(GOLD_PATH)
    source_by_pmid = {str(item["pmid"]): item for item in load_jsonl(SOURCE_PATH)}
    records = {str(item.get("pmid", "")): item for item in records_from_result(args.result)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    relation_rows, prediction_rows, document_rows = [], [], []
    stages = Counter()
    for gold in gold_rows:
        pmid = str(gold["pmid"])
        record = records.get(pmid, {})
        phases = record.get("phases", {}) or {}
        source = source_by_pmid.get(pmid, {})
        text = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"
        alias_map = aliases(gold, text)
        gold_keys = {relation_key(item, alias_map) for item in gold.get("relations", []) or []}
        final_relations = list((phases.get("verification", {}) or {}).get("relations", []) or [])
        final_keys = {relation_key(item, alias_map) for item in final_relations if semantic_kept(item)}
        document_rows.append({
            "pmid": pmid, "title": gold.get("title", ""), "article_text": text,
            "gold_relation_count": len(gold_keys), "final_semantic_relation_count": len(final_keys),
            "raw_relation_count": len((phases.get("extraction", {}) or {}).get("relations", []) or []),
            "projected_relation_count": len((phases.get("relation_candidate_projection", {}) or {}).get("relations", []) or []),
            "pair_candidate_count": int((phases.get("relation_pair_classification", {}) or {}).get("candidate_count", 0) or 0),
            "final_relations": compact(final_relations, limit=999),
        })
        for relation in gold.get("relations", []) or []:
            stage, details = classify_gold_relation(relation, phases, alias_map)
            stages[stage] += 1
            relation_rows.append({
                "pmid": pmid, "title": gold.get("title", ""), "article_text": text,
                "gold_relation": relation, "stage": stage, "details": details,
            })
        for relation in final_relations:
            key = relation_key(relation, alias_map)
            prediction_rows.append({
                "pmid": pmid, "prediction": relation, "outcome": "TP" if key in gold_keys and semantic_kept(relation) else "FP_OR_REJECTED_AUDIT",
                "gold_keys": [list(item) for item in sorted(gold_keys)],
            })

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_jsonl(args.output_dir / "gold_relation_ledger.jsonl", relation_rows)
    write_jsonl(args.output_dir / "prediction_ledger.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "document_ledger.jsonl", document_rows)
    primary_failures = stages["PRIMARY_EXTRACTION_FAILED"]
    provider_failure_categories = Counter(
        classify_provider_error(str(row["details"].get("primary_extraction_error", "")))
        for row in relation_rows if row["stage"] == "PRIMARY_EXTRACTION_FAILED"
    )
    evaluation_valid = primary_failures * 2 < len(relation_rows)
    summary = {
        "gold_relations": len(relation_rows),
        "stages": dict(stages),
        "final_prediction_rows": len(prediction_rows),
        "evaluation_valid": evaluation_valid,
        "invalidity_reason": (
            "primary extraction failed for at least half of gold relations"
            if not evaluation_valid else ""
        ),
        "provider_failure_categories": dict(provider_failure_categories),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gold-200 relation audit",
        "",
        f"- Gold relations: {len(relation_rows)}",
        f"- Evaluation valid for model-quality metrics: {evaluation_valid}",
        f"- Validity note: {summary['invalidity_reason'] or 'none'}",
        f"- Provider failure categories: {json.dumps(summary['provider_failure_categories'])}",
        "",
        "| Stage | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {stage} | {count} |" for stage, count in stages.most_common())
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    detail_lines = ["# Gold-200 relation-by-relation audit", ""]
    for index, row in enumerate(relation_rows, start=1):
        gold = row["gold_relation"]
        details = row["details"]
        detail_lines.extend([
            f"## {index}. PMID {row['pmid']} - {row['stage']}",
            "",
            f"- Title: {row['title']}",
            (
                "- Gold: "
                f"{gold.get('subject')} [{gold.get('subject_type')}] "
                f"--{gold.get('predicate')}--> "
                f"{gold.get('object')} [{gold.get('object_type')}]"
            ),
            f"- Gold evidence: {gold.get('evidence', '')}",
            f"- Primary extraction error: {details.get('primary_extraction_error') or 'none'}",
            f"- Raw matching triples: {json.dumps(details.get('raw_matches', []), ensure_ascii=False)}",
            f"- Projected matching triples: {json.dumps(details.get('projected_matches', []), ensure_ascii=False)}",
            f"- Pair candidates: {json.dumps(details.get('pair_candidates', []), ensure_ascii=False)}",
            f"- Final matching triples: {json.dumps(details.get('final_matches', []), ensure_ascii=False)}",
            "",
        ])
    (args.output_dir / "relation_by_relation_audit.md").write_text(
        "\n".join(detail_lines), encoding="utf-8"
    )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
