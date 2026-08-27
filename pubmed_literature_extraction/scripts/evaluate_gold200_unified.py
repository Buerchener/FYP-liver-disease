#!/usr/bin/env python3
"""Score a 200-document LiverKG run under one explicit metric contract.

The gold annotation is relation-centric, so relation-level TN is undefined:
the negative relation universe is not enumerated.  This scorer therefore uses
the standard positive-class typed-triple P/R/F1 for relation extraction and
reports TN only for the separately defined document-level relation-presence
task.  Candidate, main-KG-contract and safe-write targets are kept separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.abbreviation_detector import AbbreviationDetector
from cognitive_agent.extraction_quality import normalize_surface


GOLD_VIEWS = {
    "candidate_semantic": ROOT / "gold_annotations/pubmed_200_gold_v2_candidate_view.jsonl",
    "main_kg_write_contract": ROOT / "gold_annotations/pubmed_200_gold_v2_write_contract.jsonl",
    "strict_import_ready": ROOT / "gold_annotations/pubmed_200_gold_v2_strict_import_ready.jsonl",
}
SOURCE_PATH = ROOT / "extraction_output/pubmed_converted_500.jsonl"
SYMMETRIC_PREDICATES = frozenset({"ASSOCIATED_WITH", "INTERACTS_WITH"})


def cell_subset_signature(value: str) -> tuple[str, str] | None:
    """Normalize exact numbered cell-subset paraphrases for scoring only."""
    surface = normalize_surface(value)
    identifier = r"(?:\d+[a-z0-9]*|[a-z]+\d[a-z0-9]*)"
    prefix = re.fullmatch(
        rf"(?:subpopulation|subset|subgroup|cluster)s?\s+({identifier})\s+(.+)",
        surface,
    )
    suffix = re.fullmatch(
        rf"(.+?)\s+(?:subpopulation|subset|subgroup|cluster)s?\s+({identifier})",
        surface,
    )
    if prefix:
        subset_id, cell_text = prefix.group(1), prefix.group(2)
    elif suffix:
        cell_text, subset_id = suffix.group(1), suffix.group(2)
    else:
        return None
    families = (
        "macrophage", "endothelial cell", "stellate cell", "kupffer cell",
        "neutrophil", "monocyte", "natural killer cell", "nk cell",
        "regulatory t cell", "t cell", "b cell", "cholangiocyte",
        "fibroblast", "hepatocyte",
    )
    family = next((item for item in families if item in cell_text), "")
    return (family, subset_id) if family else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records_from_result(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("records", "history", "articles"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, list):
            return value
    return []


def aliases(gold: dict[str, Any], text: str) -> dict[str, set[tuple[str, str]]]:
    mapping: dict[str, set[tuple[str, str]]] = {}
    for entity in gold.get("entities", []) or []:
        canonical = normalize_surface(entity.get("canonical", entity.get("mention", "")))
        typed = (canonical, str(entity.get("type", "")))
        for surface in (entity.get("mention", ""), entity.get("canonical", "")):
            mapping.setdefault(normalize_surface(surface), set()).add(typed)
    detected = AbbreviationDetector().detect(text)
    for short, long_form in detected.abbr_to_long.items():
        short_key = normalize_surface(short)
        long_key = normalize_surface(long_form)
        for typed in mapping.get(short_key, set()) | mapping.get(long_key, set()):
            mapping.setdefault(short_key, set()).add(typed)
            mapping.setdefault(long_key, set()).add(typed)
    return mapping


def canonical_endpoint(value: str, entity_type: str, alias_map: dict) -> str:
    surface = normalize_surface(value)
    exact = [canonical for canonical, kind in alias_map.get(surface, set()) if kind == entity_type]
    if len(exact) == 1:
        return exact[0]
    if entity_type == "CellType":
        signature = cell_subset_signature(surface)
        if signature:
            matches = {
                canonical
                for alias_surface, typed_values in alias_map.items()
                if cell_subset_signature(alias_surface) == signature
                for canonical, kind in typed_values
                if kind == entity_type
            }
            if len(matches) == 1:
                return next(iter(matches))
    return surface


def relation_key(relation: dict[str, Any], alias_map: dict) -> tuple[str, str, str, str, str]:
    subject_type = str(relation.get("subject_type", ""))
    object_type = str(relation.get("object_type", ""))
    predicate = str(relation.get("predicate", "")).upper()
    subject = canonical_endpoint(str(relation.get("subject", "")), subject_type, alias_map)
    obj = canonical_endpoint(str(relation.get("object", "")), object_type, alias_map)
    key = (subject, subject_type, predicate, obj, object_type)
    if predicate in SYMMETRIC_PREDICATES and subject_type == object_type:
        reverse = (obj, object_type, predicate, subject, subject_type)
        return min(key, reverse)
    return key


def semantic_kept(relation: dict[str, Any]) -> bool:
    """Positive semantic prediction: ACCEPTED only, REVIEW is abstention."""
    return (
        str(relation.get("scope_status", "IN_SCOPE")).upper() != "OUT_OF_SCOPE"
        and "article_out_of_scope" not in set(relation.get("quality_flags", []) or [])
        and
        str(relation.get("factual_status", "VALID")).upper() != "REJECTED"
        and str(relation.get("semantic_status", "ACCEPTED")).upper() == "ACCEPTED"
    )


def semantic_candidate_kept(relation: dict[str, Any]) -> bool:
    """Candidate coverage view: ACCEPTED plus HUMAN_REVIEW, never REJECTED."""
    return (
        str(relation.get("scope_status", "IN_SCOPE")).upper() != "OUT_OF_SCOPE"
        and "article_out_of_scope" not in set(relation.get("quality_flags", []) or [])
        and
        str(relation.get("factual_status", "VALID")).upper() != "REJECTED"
        and str(relation.get("semantic_status", "ACCEPTED")).upper() in {"ACCEPTED", "REVIEW"}
    )


def predicted_for_view(
    relations: list[dict[str, Any]], view: str, *, include_review: bool = False,
) -> list[dict[str, Any]]:
    keep = semantic_candidate_kept if include_review else semantic_kept
    kept = [relation for relation in relations if keep(relation)]
    if view == "candidate_semantic":
        return kept
    if view == "main_kg_write_contract":
        return [relation for relation in kept if relation.get("write_contract_valid") is True]
    return [
        relation for relation in kept
        if relation.get("write_contract_valid") is True
        and (
            str(relation.get("write_status", "")).upper() == "IMPORT_READY"
            or relation.get("import_ready") is True
        )
    ]


def prf(counts: Counter) -> dict[str, float | int]:
    tp, fp, fn = (int(counts[key]) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def score_view(
    records: list[dict[str, Any]], gold_by_pmid: dict[str, dict[str, Any]],
    source_by_pmid: dict[str, dict[str, Any]], view: str, *, include_review: bool = False,
) -> dict[str, Any]:
    counts: Counter = Counter()
    by_predicate: dict[str, Counter] = defaultdict(Counter)
    document = Counter()
    direction = Counter()
    missing_pmids: list[str] = []

    record_by_pmid = {str(record.get("pmid", "")): record for record in records}
    for pmid, gold in gold_by_pmid.items():
        record = record_by_pmid.get(pmid)
        if record is None:
            missing_pmids.append(pmid)
            relations: list[dict[str, Any]] = []
        else:
            relations = list((record.get("phases", {}) or {}).get("verification", {}).get("relations", []) or [])
        source = source_by_pmid.get(pmid, {})
        text = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"
        alias_map = aliases(gold, text)
        gold_relations = list(gold.get("relations", []) or [])
        gold_by_key = {relation_key(item, alias_map): item for item in gold_relations}
        pred_by_key = {
            relation_key(item, alias_map): item
            for item in predicted_for_view(relations, view, include_review=include_review)
        }
        gold_keys, pred_keys = set(gold_by_key), set(pred_by_key)
        counts.update({
            "tp": len(gold_keys & pred_keys),
            "fp": len(pred_keys - gold_keys),
            "fn": len(gold_keys - pred_keys),
        })
        for predicate in sorted({key[2] for key in gold_keys | pred_keys}):
            gold_p = {key for key in gold_keys if key[2] == predicate}
            pred_p = {key for key in pred_keys if key[2] == predicate}
            by_predicate[predicate].update({
                "tp": len(gold_p & pred_p), "fp": len(pred_p - gold_p), "fn": len(gold_p - pred_p),
            })
        for key in gold_keys & pred_keys:
            if str(gold_by_key[key].get("direction", "unknown")) == str(pred_by_key[key].get("direction", "unknown")):
                direction["correct"] += 1
            direction["matched"] += 1

        gold_present, pred_present = bool(gold_keys), bool(pred_keys)
        if gold_present and pred_present:
            document["tp"] += 1
        elif pred_present:
            document["fp"] += 1
        elif gold_present:
            document["fn"] += 1
        else:
            document["tn"] += 1

    predicate_metrics = {predicate: prf(value) for predicate, value in sorted(by_predicate.items())}
    active = [value for value in predicate_metrics.values() if value["tp"] + value["fp"] + value["fn"]]
    macro_f1 = sum(float(value["f1"]) for value in active) / len(active) if active else 0.0
    return {
        "typed_directed_triple_micro": prf(counts),
        "positive_predicate_macro_f1": macro_f1,
        "per_predicate": predicate_metrics,
        "document_relation_presence": {**prf(document), "tn": int(document["tn"])},
        "direction_agreement_given_triple_tp": direction["correct"] / direction["matched"] if direction["matched"] else 1.0,
        "missing_result_pmids": missing_pmids,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    records = records_from_result(args.result)
    source_by_pmid = {str(row["pmid"]): row for row in load_jsonl(SOURCE_PATH)}
    report: dict[str, Any] = {
        "metric_contract_version": "liverkg-gold200-v1",
        "result": str(args.result),
        "result_record_count": len(records),
        "relation_level_tn": "undefined: gold is relation-centric and does not enumerate negative triples",
        "views": {},
    }
    for view, path in GOLD_VIEWS.items():
        rows = load_jsonl(path)
        gold_by_pmid = {str(row["pmid"]): row for row in rows}
        report["views"][view] = {
            "gold_path": str(path), "gold_sha256": sha256(path),
            "gold_documents": len(rows),
            "gold_relations": sum(len(row.get("relations", []) or []) for row in rows),
            "metrics": score_view(records, gold_by_pmid, source_by_pmid, view),
        }
        if view == "candidate_semantic":
            report["views"][view]["review_inclusive_coverage_metrics"] = score_view(
                records, gold_by_pmid, source_by_pmid, view, include_review=True,
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Gold-200 unified evaluation", "", "- Relation metric: typed, directed positive triples; same-type `ASSOCIATED_WITH` and `INTERACTS_WITH` are symmetric.", "- Primary metrics count `ACCEPTED` only. Candidate coverage including `REVIEW` is reported separately in JSON and is not precision.", "- Relation-level TN is undefined for this relation-centric gold. TN below is document-level relation-presence TN.", "", "| View | Gold relations | Micro P/R/F1 (TP/FP/FN) | Positive macro F1 | Document TP/FP/TN/FN | Direction agreement |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in report["views"].items():
        metric = item["metrics"]
        micro = metric["typed_directed_triple_micro"]
        doc = metric["document_relation_presence"]
        lines.append(
            f"| {name} | {item['gold_relations']} | {micro['precision']:.3f}/{micro['recall']:.3f}/{micro['f1']:.3f} ({micro['tp']}/{micro['fp']}/{micro['fn']}) | {metric['positive_predicate_macro_f1']:.3f} | {doc['tp']}/{doc['fp']}/{doc['tn']}/{doc['fn']} | {metric['direction_agreement_given_triple_tp']:.3f} |"
        )
    lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.output_json)
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
