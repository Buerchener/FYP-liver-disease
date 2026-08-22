#!/usr/bin/env python3
"""Diagnose why gold relations never reach the candidate pair lattice.

Offline, deterministic, no API calls.  For every gold relation it classifies
the first blocking reason into:

    endpoint_missing    neither/both endpoint entities were not extracted
    alias_gap           endpoint mention exists in text but the extracted
                        entity's alias set never merges with the gold form
    schema_mask         the (subject_type, object_type) pair has no allowed
                        predicate under RELATION_SIGNATURES
    cross_clause        both endpoints co-occur in one sentence but no single
                        clause contains both
    cross_sentence      endpoints sit in adjacent sentences
    document_level      endpoints are farther apart than adjacent sentences
    truncation          the pair exists in the lattice but ranks beyond the cap
    covered             the pair is present in the lattice (not a miss)

It reports old (clause-only, cap 64) vs new (sentence + adjacent windows,
cap 128) candidate-pair recall so the lattice change can be evaluated alone.
"""

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

from cognitive_agent.abbreviation_detector import AbbreviationDetector
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.extraction_quality import normalize_surface
from cognitive_agent.relation_pair_classifier import (
    BioREDPairClassifier,
    PairClassifierConfig,
)
from cognitive_agent.schema.relation_signatures import RELATION_SIGNATURES


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def alias_set(entity: dict, abbr_map: Any) -> set[str]:
    values = [
        str(entity.get("mention", "") or ""),
        *(str(item or "") for item in (entity.get("canonical_mentions", []) or [])),
    ]
    mention = str(entity.get("mention", "") or "")
    if abbr_map and mention:
        values.extend([
            abbr_map.resolve_to_long(mention),
            abbr_map.resolve_to_short(mention),
        ])
    return {normalize_surface(value) for value in values if str(value or "").strip()}


def gold_endpoint_aliases(gold_entity: dict, abbr_map: Any) -> set[str]:
    values = [
        str(gold_entity.get("canonical", "") or ""),
        str(gold_entity.get("mention", "") or ""),
        *(str(item or "") for item in (gold_entity.get("aliases", []) or [])),
    ]
    mention = str(gold_entity.get("mention", "") or "")
    canonical = str(gold_entity.get("canonical", "") or "")
    if abbr_map:
        values.extend([
            abbr_map.resolve_to_long(mention),
            abbr_map.resolve_to_short(mention),
            abbr_map.resolve_to_long(canonical),
            abbr_map.resolve_to_short(canonical),
        ])
    return {normalize_surface(value) for value in values if str(value or "").strip()}


def find_gold_entity(gold_entities: list[dict], surface: str, entity_type: str) -> dict | None:
    wanted = normalize_surface(surface)
    for entity in gold_entities:
        if str(entity.get("type", "")) != entity_type:
            continue
        for value in (
            entity.get("canonical"), entity.get("mention"), *(entity.get("aliases", []) or [])
        ):
            if normalize_surface(value) == wanted:
                return entity
    return None


def extracted_match(
    entities: list[dict], gold_aliases: set[str], entity_type: str, abbr_map: Any,
) -> tuple[str, str]:
    """Return (status, detail) for one gold endpoint vs extracted entities."""
    for entity in entities:
        etype = str(entity.get("type", entity.get("entity_type", "")) or "")
        if etype and etype != entity_type:
            continue
        aliases = alias_set(entity, abbr_map)
        if aliases & gold_aliases:
            return "ok", str(entity.get("mention", "") or "")
        # Same mention but wrong extracted type is an endpoint-quality miss.
        for alias in aliases:
            if alias in gold_aliases:
                return "type_mismatch", f"extracted as {etype}"
    # Endpoint might be in the text but never extracted.
    return "missing", ""


def endpoint_alias_variants(value: str, abbr_map: Any) -> set[str]:
    variants = {normalize_surface(value)}
    if abbr_map and value:
        variants.add(normalize_surface(abbr_map.resolve_to_long(value)))
        variants.add(normalize_surface(abbr_map.resolve_to_short(value)))
    return {item for item in variants if item}


def pair_covered(
    candidates: list[Any], subject: str, subject_type: str, object_: str, object_type: str,
    subject_aliases: set[str], object_aliases: set[str], abbr_map: Any,
) -> bool:
    """Alias-aware pair coverage: the lattice stores extracted mentions, gold
    stores canonical surfaces; both sides are expanded through the article
    abbreviation map before comparison."""
    wanted_sub = subject_aliases | endpoint_alias_variants(subject, abbr_map)
    wanted_obj = object_aliases | endpoint_alias_variants(object_, abbr_map)
    for item in candidates:
        get = item.get if isinstance(item, dict) else lambda key, default="": getattr(item, key, default)
        cand_sub = endpoint_alias_variants(str(get("subject", "")), abbr_map)
        cand_obj = endpoint_alias_variants(str(get("object", "")), abbr_map)
        if not wanted_sub & cand_sub:
            continue
        if str(get("subject_type", "")) != subject_type:
            continue
        if not wanted_obj & cand_obj:
            continue
        if str(get("object_type", "")) != object_type:
            continue
        return True
    return False


def first_window_kind(
    text: str, subject_aliases: set[str], object_aliases: set[str],
) -> str:
    reader = ArticleEvidenceReader()
    units = reader.read(text)
    parents = ArticleEvidenceReader.parent_units(text, units)
    adjacent = ArticleEvidenceReader.adjacent_sentence_windows(text, parents)

    def covers(window) -> bool:
        found_sub = found_obj = False
        for alias in subject_aliases:
            pattern = " ".join(alias.split())
            if pattern and pattern in " ".join(window.text.casefold().split()):
                found_sub = True
                break
        for alias in object_aliases:
            pattern = " ".join(alias.split())
            if pattern and pattern in " ".join(window.text.casefold().split()):
                found_obj = True
                break
        return found_sub and found_obj

    for unit in units:
        if covers(unit):
            return "same_clause"
    for parent in parents:
        if covers(parent):
            return "cross_clause"
    for window in adjacent:
        if covers(window):
            return "cross_sentence"
    return "document_level"


def diagnose_article(
    pmid: str,
    text: str,
    extraction: dict,
    gold_relations: list[dict],
    gold_entities: list[dict],
    old_config: PairClassifierConfig,
    new_config: PairClassifierConfig,
) -> list[dict]:
    entities = extraction.get("entities", []) or []
    relations = extraction.get("relations", []) or []
    reader = ArticleEvidenceReader()
    units = reader.read(text)
    abbr_map = AbbreviationDetector().detect(text)

    old_classifier = BioREDPairClassifier(old_config)
    new_classifier = BioREDPairClassifier(new_config)
    uncapped_config = PairClassifierConfig(
        include_parent_sentences=new_config.include_parent_sentences,
        include_adjacent_windows=new_config.include_adjacent_windows,
        max_candidates=100_000,
    )
    uncapped_classifier = BioREDPairClassifier(uncapped_config)
    old_candidates, old_truncated = old_classifier.build_candidates(
        entities, relations, units, source_text=text
    )
    new_candidates, new_truncated = new_classifier.build_candidates(
        entities, relations, units, source_text=text
    )
    uncapped_candidates, _ = uncapped_classifier.build_candidates(
        entities, relations, units, source_text=text
    )

    rows: list[dict] = []
    for relation in gold_relations:
        predicate = str(relation.get("predicate", "")).upper()
        subject = str(relation.get("subject", "") or "")
        object_ = str(relation.get("object", "") or "")
        subject_type = str(relation.get("subject_type", "") or "")
        object_type = str(relation.get("object_type", "") or "")

        # ── endpoint extraction check ──
        gold_sub = find_gold_entity(gold_entities, subject, subject_type)
        gold_obj = find_gold_entity(gold_entities, object_, object_type)
        sub_aliases = gold_endpoint_aliases(gold_sub, abbr_map) if gold_sub else {normalize_surface(subject)}
        obj_aliases = gold_endpoint_aliases(gold_obj, abbr_map) if gold_obj else {normalize_surface(object_)}
        sub_status, sub_detail = extracted_match(entities, sub_aliases, subject_type, abbr_map)
        obj_status, obj_detail = extracted_match(entities, obj_aliases, object_type, abbr_map)

        def covered(candidates: list[Any]) -> bool:
            return (
                pair_covered(
                    candidates, subject, subject_type, object_, object_type,
                    sub_aliases, obj_aliases, abbr_map,
                )
                or pair_covered(
                    candidates, object_, object_type, subject, subject_type,
                    obj_aliases, sub_aliases, abbr_map,
                )
            )

        covered_old = covered(old_candidates)
        covered_new = covered(new_candidates)
        covered_uncapped = covered(uncapped_candidates)

        reason = ""
        detail = ""
        if sub_status != "ok" or obj_status != "ok":
            reason = (
                "alias_gap"
                if (sub_status == "type_mismatch" or obj_status == "type_mismatch")
                else "endpoint_missing"
            )
            detail = f"subject={sub_status}:{sub_detail} object={obj_status}:{obj_detail}"
        elif not covered_uncapped:
            # ── schema mask ──
            allowed_forward = [
                p for p, pairs in RELATION_SIGNATURES.items()
                if (subject_type, object_type) in pairs
            ]
            allowed_reverse = [
                p for p, pairs in RELATION_SIGNATURES.items()
                if (object_type, subject_type) in pairs
            ]
            if not allowed_forward and not allowed_reverse:
                reason = "schema_mask"
                detail = f"({subject_type}, {object_type}) has no allowed predicate"
            else:
                kind = first_window_kind(text, sub_aliases, obj_aliases)
                if kind in {"cross_clause", "cross_sentence", "document_level"}:
                    reason = kind
                else:
                    # Same clause, endpoints extracted, aliases merged, schema
                    # allowed, yet no candidate: mention-form/boundary miss in
                    # the pairing regex.
                    reason = "same_clause_not_paired"
        elif covered_uncapped and not covered_new:
            reason = "truncation"
            detail = f"old_truncated={old_truncated} new_truncated={new_truncated}"
        else:
            reason = "covered"

        hinted = any(
            normalize_surface(item.get("subject", "")) in sub_aliases
            and normalize_surface(item.get("object", "")) in obj_aliases
            for item in relations
        )
        rows.append({
            "pmid": pmid,
            "triple": f"{subject}|{subject_type}|{predicate}|{object_}|{object_type}",
            "predicate": predicate,
            "reason": reason,
            "detail": detail,
            "covered_old_lattice": covered_old,
            "covered_new_lattice": covered_new,
            "langextract_hint": hinted,
            "import_ready": bool(relation.get("import_ready")),
        })
    return rows


def summarize(rows: list[dict]) -> dict[str, Any]:
    total = len(rows)
    reasons = Counter(item["reason"] for item in rows)
    old_recall = sum(item["covered_old_lattice"] for item in rows) / total if total else 0.0
    new_recall = sum(item["covered_new_lattice"] for item in rows) / total if total else 0.0
    per_predicate: dict[str, dict[str, Any]] = {}
    for predicate in sorted({item["predicate"] for item in rows}):
        subset = [item for item in rows if item["predicate"] == predicate]
        per_predicate[predicate] = {
            "count": len(subset),
            "old_recall": sum(item["covered_old_lattice"] for item in subset) / len(subset),
            "new_recall": sum(item["covered_new_lattice"] for item in subset) / len(subset),
            "reasons": dict(Counter(item["reason"] for item in subset)),
        }
    hinted_of_covered = sum(
        item["langextract_hint"] and item["covered_new_lattice"] for item in rows
    )
    summary: dict[str, Any] = {
        "gold_relation_count": total,
        "reason_counts": dict(reasons),
        "candidate_pair_recall_old": round(old_recall, 4),
        "candidate_pair_recall_new": round(new_recall, 4),
        "recall_delta": round(new_recall - old_recall, 4),
        "covered_with_langextract_hint": hinted_of_covered,
        "per_predicate": per_predicate,
    }
    if any("covered_with_recovery" in item for item in rows):
        recovery_recall = sum(item["covered_with_recovery"] for item in rows) / total
        endpoint_miss_base = reasons.get("endpoint_missing", 0) + reasons.get("alias_gap", 0)
        endpoint_miss_after = sum(
            1 for item in rows
            if item.get("reason_with_recovery") in {"endpoint_missing", "alias_gap"}
        )
        summary.update({
            "candidate_pair_recall_with_recovery": round(recovery_recall, 4),
            "recovery_recall_delta": round(recovery_recall - new_recall, 4),
            "endpoint_miss_base": endpoint_miss_base,
            "endpoint_miss_after_recovery": endpoint_miss_after,
            "endpoint_miss_recovered": endpoint_miss_base - endpoint_miss_after,
        })
    return summary


def markdown(summary: dict[str, Any], rows: list[dict]) -> str:
    lines = [
        "# Candidate lattice miss diagnosis",
        "",
        f"- Gold relations analysed: {summary['gold_relation_count']}",
        f"- Candidate-pair recall (old clause-only lattice, cap 64): {summary['candidate_pair_recall_old']:.1%}",
        f"- Candidate-pair recall (new sentence + adjacent windows, cap 128): {summary['candidate_pair_recall_new']:.1%}",
        f"- Recall delta: {summary['recall_delta']:+.1%}",
        "",
        "## Miss reason distribution (new lattice)",
        "",
        "| Reason | Count | Share |",
        "|---|---:|---:|",
    ]
    total = summary["gold_relation_count"] or 1
    for reason, count in sorted(
        summary["reason_counts"].items(), key=lambda item: -item[1]
    ):
        lines.append(f"| {reason} | {count} | {count / total:.1%} |")
    if "candidate_pair_recall_with_recovery" in summary:
        lines += [
            f"- Candidate-pair recall with entity recovery: {summary['candidate_pair_recall_with_recovery']:.1%}",
            f"- Recovery delta: {summary['recovery_recall_delta']:+.1%} "
            f"(endpoint misses {summary['endpoint_miss_base']} → {summary['endpoint_miss_after_recovery']}; "
            f"recovered {summary['endpoint_miss_recovered']})",
        ]
    lines += [
        "",
        "## Per-predicate candidate-pair recall",
        "",
        "| Predicate | Count | Old recall | New recall | Miss reasons |",
        "|---|---:|---:|---:|---|",
    ]
    for predicate, info in summary["per_predicate"].items():
        reasons = ", ".join(f"{key}:{value}" for key, value in sorted(info["reasons"].items()))
        lines.append(
            f"| {predicate} | {info['count']} | {info['old_recall']:.1%} | "
            f"{info['new_recall']:.1%} | {reasons} |"
        )
    lines += ["", "## Missed triples (new lattice)", ""]
    for item in sorted(rows, key=lambda row: (row["reason"], row["triple"])):
        if item["reason"] != "covered":
            lines.append(
                f"- `{item['triple']}` — {item['reason']}"
                + (f" ({item['detail']})" if item.get("detail") else "")
                + (", import_ready" if item["import_ready"] else "")
                + (
                    f" → recovered: {item['reason_with_recovery']}"
                    if item.get("covered_with_recovery") is not None
                    else ""
                )
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", required=True, type=Path,
        help="frozen agent results JSON with phases.extraction per article",
    )
    parser.add_argument(
        "--gold", type=Path,
        default=ROOT / "gold_annotations/pubmed_200_gold_v2_strict.jsonl",
    )
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "extraction_output/pubmed_converted_500.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_output")
    parser.add_argument("--run-id", default="candidate_miss_diagnostic")
    parser.add_argument("--limit", type=int, default=0, help="analyse first N articles")
    parser.add_argument(
        "--with-recovery", action="store_true",
        help="additionally merge phases.entity_recovery.recovered entities into the "
        "inventory and report the hypothetical candidate-pair recall with recovery",
    )
    args = parser.parse_args()

    payload = json.loads(args.run.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    source_by_pmid = {str(item["pmid"]): item for item in load_jsonl(args.source)}
    gold_by_pmid = {str(item["pmid"]): item for item in load_jsonl(args.gold)}

    old_config = PairClassifierConfig(
        include_parent_sentences=False, include_adjacent_windows=False,
        max_candidates=64,
    )
    new_config = PairClassifierConfig(
        include_parent_sentences=True, include_adjacent_windows=True,
        max_candidates=128,
    )

    rows: list[dict] = []
    for record in records[: args.limit or None]:
        pmid = str(record.get("pmid", ""))
        if pmid not in gold_by_pmid or pmid not in source_by_pmid:
            continue
        gold = gold_by_pmid[pmid]
        source = source_by_pmid[pmid]
        text = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"
        extraction = record.get("phases", {}).get("extraction", {}) or {}
        base_rows = diagnose_article(
            pmid, text, extraction,
            gold.get("relations", []) or [],
            gold.get("entities", []) or [],
            old_config, new_config,
        )
        if args.with_recovery:
            recovery = record.get("phases", {}).get("entity_recovery", {}) or {}
            recovered = list(recovery.get("recovered", []) or [])
            recovered_extraction = {
                **extraction,
                "entities": list(extraction.get("entities", []) or []) + recovered,
            }
            recovery_rows = diagnose_article(
                pmid, text, recovered_extraction,
                gold.get("relations", []) or [],
                gold.get("entities", []) or [],
                old_config, new_config,
            )
            by_triple = {item["triple"]: item for item in recovery_rows}
            for row in base_rows:
                counterpart = by_triple.get(row["triple"], {})
                row["covered_with_recovery"] = bool(
                    counterpart.get("covered_new_lattice", False)
                )
                row["reason_with_recovery"] = str(
                    counterpart.get("reason", row["reason"])
                )
        rows.extend(base_rows)

    summary = summarize(rows)
    out_dir = args.output_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "miss_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(markdown(summary, rows), encoding="utf-8")
    print(markdown(summary, rows))
    print(f"[DONE] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
