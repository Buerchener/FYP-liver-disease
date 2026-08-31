#!/usr/bin/env python3
"""Audit BioRED factual-to-semantic routing without changing predictions.

The ledger keeps official exact-match status separate from a source-support
assessment.  It is diagnostic only: Dev labels are never emitted as runtime
rules or model inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from cognitive_agent.biored_adapter import BioREDDocument, parse_biored_pubtator


def relation_key(item: dict[str, Any]) -> tuple[str, str, str]:
    left, right = sorted((str(item.get("arg1_id", "")), str(item.get("arg2_id", ""))))
    return left, right, str(item.get("relation_type", ""))


def _best_candidate(
    item: dict[str, Any], factual: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_id = str(item.get("candidate_id", "") or "")
    same_id = [
        value for value in factual
        if candidate_id and str(value.get("candidate_id", "") or "") == candidate_id
    ]
    if same_id:
        return max(same_id, key=lambda value: int(value.get("candidate_version", 1) or 1))
    same_key = [value for value in factual if relation_key(value) == relation_key(item)]
    return max(
        same_key, key=lambda value: int(value.get("candidate_version", 1) or 1),
        default=item,
    )


def _endpoint_support(pack: dict[str, Any]) -> tuple[str, list[str]]:
    owners = [
        item for item in list(pack.get("spans", []) or [])
        if str(item.get("role", "") or "") == "OWNER"
    ]
    closed = [
        str(item.get("span_id", "") or "") for item in owners
        if bool(item.get("subject_covered")) and bool(item.get("object_covered"))
    ]
    if closed:
        return "SINGLE_OWNER_ENDPOINT_CLOSED", [closed[0]]
    subject = [item for item in owners if bool(item.get("subject_covered"))]
    object_ = [item for item in owners if bool(item.get("object_covered"))]
    if subject and object_:
        ids = list(dict.fromkeys([
            str(subject[0].get("span_id", "") or ""),
            str(object_[0].get("span_id", "") or ""),
        ]))
        return "MULTI_OWNER_ENDPOINT_CLOSED", ids
    return "ENDPOINT_SUPPORT_UNRESOLVED", []


def _judge_failure(
    candidate: dict[str, Any], audit: dict[tuple[str, int], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    candidate_id = str(candidate.get("candidate_id", "") or "")
    version = int(candidate.get("candidate_version", 1) or 1)
    adjudication = dict(candidate.get("label_adjudication", {}) or {})
    judge = dict(adjudication.get("judge", {}) or {})
    row = dict(audit.get((candidate_id, version), {}) or {})
    verdict = str(judge.get("verdict", "") or "").upper()
    asserted = str(judge.get("relation_asserted", "") or "").upper()
    target = str(judge.get("recommended_relation_type", "") or "")
    current = str(candidate.get("relation_type", "") or "")
    try:
        confidence = float(judge.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    reason_codes = list(row.get("reason_codes", []) or [])
    if not judge:
        failure = "MISSING_JUDGE_DECISION"
    elif verdict not in {"KEEP", "EDIT"}:
        failure = "JUDGE_ABSTAINED"
    elif asserted == "NO":
        failure = "JUDGE_RELATION_REJECTED"
    elif target and target != current:
        failure = "JUDGE_LABEL_CONFLICT"
    elif confidence < 0.90:
        failure = "JUDGE_LOW_CONFIDENCE"
    elif "adjudicator_span_support_insufficient" in reason_codes:
        failure = "JUDGE_SPAN_SUPPORT_INSUFFICIENT"
    else:
        failure = "ROUTING_REVIEW_OTHER"
    return failure, {
        "verdict": verdict,
        "relation_asserted": asserted,
        "recommended_relation_type": target,
        "confidence": confidence,
        "supporting_span_ids": list(judge.get("supporting_span_ids", []) or []),
        "reason_code": str(judge.get("reason_code", "") or ""),
        "routing_reason_codes": reason_codes,
        "decision": str(adjudication.get("decision", "") or ""),
    }


def _routing_gates(
    candidate: dict[str, Any], judge: dict[str, Any],
    adjudication_audit: dict[tuple[str, int], dict[str, Any]],
) -> list[str]:
    """Return orthogonal routing failures instead of a lossy first-failure label."""
    candidate_id = str(candidate.get("candidate_id", "") or "")
    version = int(candidate.get("candidate_version", 1) or 1)
    row = dict(adjudication_audit.get((candidate_id, version), {}) or {})
    verdict = str(judge.get("verdict", "") or "").upper()
    asserted = str(judge.get("relation_asserted", "") or "").upper()
    target = str(judge.get("recommended_relation_type", "") or "")
    current = str(candidate.get("relation_type", "") or "")
    try:
        confidence = float(judge.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    gates: list[str] = []
    if not judge:
        gates.append("MISSING_JUDGE_DECISION")
    if judge and verdict not in {"KEEP", "EDIT"}:
        gates.append("JUDGE_ABSTAINED")
    if asserted == "NO":
        gates.append("JUDGE_RELATION_REJECTED")
    elif asserted == "AMBIGUOUS":
        gates.append("JUDGE_RELATION_AMBIGUOUS")
    if target and target != current:
        gates.append("JUDGE_LABEL_CONFLICT")
    if confidence < 0.90:
        gates.append("JUDGE_LOW_CONFIDENCE")
    reason_codes = set(row.get("reason_codes", []) or [])
    if "adjudicator_span_support_insufficient" in reason_codes:
        gates.append("JUDGE_SPAN_SUPPORT_INSUFFICIENT")
    if "adjudicator_invalid_target_label" in reason_codes:
        gates.append("JUDGE_INVALID_TARGET_LABEL")
    if "adjudicator_type_signature_mismatch" in reason_codes:
        gates.append("JUDGE_TYPE_SIGNATURE_MISMATCH")
    return sorted(set(gates))


def _concept(document: BioREDDocument, concept_id: str) -> dict[str, Any]:
    concept = document.concepts.get(concept_id)
    return {
        "concept_id": concept_id,
        "entity_type": concept.entity_type if concept else "",
        "mentions": [item.text for item in concept.mentions] if concept else [],
    }


def _audit_row(
    *, scope: str, source: dict[str, Any], candidate: dict[str, Any],
    document: BioREDDocument, gold: set[tuple[str, str, str]],
    adjudication_audit: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    key = relation_key(source)
    exact_gold = key in gold
    pack = dict(candidate.get("evidence_pack", {}) or {})
    endpoint_mode, endpoint_span_ids = _endpoint_support(pack)
    routing_failure, judge = _judge_failure(candidate, adjudication_audit)
    routing_gates = _routing_gates(candidate, judge, adjudication_audit)
    flags = sorted(set(candidate.get("quality_flags", []) or []))
    pair = key[:2]
    gold_pair_labels = sorted({
        relation.relation_type for relation in document.relations
        if relation.pair == pair
    })
    mismatch_kind = (
        "TP" if exact_gold else "WRONG_LABEL" if gold_pair_labels else "EXTRA_PAIR"
    )
    candidate_relation_type = str(candidate.get("relation_type", "") or "")
    candidate_key = (*pair, candidate_relation_type)
    candidate_exact_gold = candidate_key in gold
    candidate_mismatch_kind = (
        "TP" if candidate_exact_gold
        else "WRONG_LABEL" if gold_pair_labels else "EXTRA_PAIR"
    )
    adjudication = dict(candidate.get("label_adjudication", {}) or {})
    critic = dict(adjudication.get("critic", {}) or {})
    source_support_assessment = (
        "GOLD_EXACT_CONFIRMED" if exact_gold
        else "MODEL_SUPPORTED_GOLD_MISMATCH"
        if judge["verdict"] == "KEEP" and judge["relation_asserted"] == "YES"
        else "ALTERNATIVE_LABEL_PROPOSED"
        if judge["recommended_relation_type"]
        and judge["recommended_relation_type"] != str(candidate.get("relation_type", ""))
        else "MODEL_REJECTED"
        if judge["relation_asserted"] == "NO"
        else "UNRESOLVED_GOLD_MISMATCH"
    )
    spans = [
        {
            "span_id": str(item.get("span_id", "") or ""),
            "sentence_id": str(item.get("sentence_id", "") or ""),
            "role": str(item.get("role", "") or ""),
            "alignment_status": str(item.get("alignment_status", "") or ""),
            "subject_covered": bool(item.get("subject_covered")),
            "object_covered": bool(item.get("object_covered")),
            "trigger_match": str(item.get("trigger_match", "NONE") or "NONE"),
            "text": str(item.get("text", "") or ""),
        }
        for item in list(pack.get("spans", []) or [])
    ]
    return {
        "scope": scope,
        "pmid": document.pmid,
        "candidate_id": str(candidate.get("candidate_id", "") or ""),
        "candidate_version": int(candidate.get("candidate_version", 1) or 1),
        "pair_candidate_id": str(candidate.get("pair_candidate_id", "") or ""),
        "label_key": list(key),
        "original_label_key": list(relation_key(source)),
        "official_exact_status": "TP" if exact_gold else "FP_EXACT_MISMATCH",
        "official_mismatch_kind": mismatch_kind,
        "candidate_label_key": list(candidate_key),
        "candidate_official_exact_status": (
            "TP" if candidate_exact_gold else "FP_EXACT_MISMATCH"
        ),
        "candidate_official_mismatch_kind": candidate_mismatch_kind,
        "gold_pair_labels": gold_pair_labels,
        "current_label_matches_gold_pair": key[2] in gold_pair_labels,
        "judge_target_matches_gold_pair": (
            str(judge.get("recommended_relation_type", "") or "") in gold_pair_labels
        ),
        "source_support_assessment": source_support_assessment,
        "arg1": _concept(document, key[0]),
        "arg2": _concept(document, key[1]),
        "relation_type": key[2],
        "candidate_relation_type": candidate_relation_type,
        "novelty": str(candidate.get("novelty", "Unknown") or "Unknown"),
        "confidence": float(candidate.get("confidence", 0.0) or 0.0),
        "evidence_quote": str(candidate.get("evidence_quote", "") or ""),
        "evidence_window": str(candidate.get("evidence_window", "") or ""),
        "source_traceable": bool(pack.get("source_traceable")),
        "endpoint_support_mode": endpoint_mode,
        "endpoint_support_span_ids": endpoint_span_ids,
        "predicate_support_mode": str(candidate.get("support_mode", "UNRESOLVED") or "UNRESOLVED"),
        "relation_card_match": str(candidate.get("relation_card_match", "NONE") or "NONE"),
        "quality_flags": flags,
        "routing_failure": routing_failure,
        "routing_gates": routing_gates,
        "judge": judge,
        "critic": {
            "verdict": str(critic.get("verdict", "") or "").upper(),
            "relation_asserted": str(critic.get("relation_asserted", "") or "").upper(),
            "recommended_relation_type": str(
                critic.get("recommended_relation_type", "") or ""
            ),
            "confidence": float(critic.get("confidence", 0.0) or 0.0),
            "supporting_span_ids": list(critic.get("supporting_span_ids", []) or []),
            "reason_code": str(critic.get("reason_code", "") or ""),
        },
        "evidence_spans": spans,
    }


def _rows_for_scope(
    *, scope: str, source_field: str, record: dict[str, Any],
    document: BioREDDocument,
) -> list[dict[str, Any]]:
    accepted = {relation_key(item) for item in record.get("framework_relations", []) or []}
    factual = list(record.get("factual_relations", []) or [])
    gold = {item.label_key for item in document.relations}
    adjudication_audit = {
        (
            str(item.get("candidate_id", "") or ""),
            int(item.get("candidate_version_before", 1) or 1),
        ): dict(item)
        for item in record.get("native_label_adjudication_audit", []) or []
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for source in record.get(source_field, []) or []:
        if relation_key(source) in accepted:
            continue
        candidate = _best_candidate(source, factual)
        identity = (*relation_key(source), str(candidate.get("candidate_id", "") or ""))
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(_audit_row(
            scope=scope, source=source, candidate=candidate,
            document=document, gold=gold, adjudication_audit=adjudication_audit,
        ))
    return rows


def _counter(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get(field, "")) for item in rows).items()))


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        by_scope[item["scope"]].append(item)
    scopes = {}
    for scope, values in sorted(by_scope.items()):
        type_only = [item for item in values if "native_relation_card_type_only" in item["quality_flags"]]
        type_only_tp = [item for item in type_only if item["official_exact_status"] == "TP"]
        type_only_fp = [item for item in type_only if item["official_exact_status"] != "TP"]
        scopes[scope] = {
            "removed_total": len(values),
            "official_status": _counter(values, "official_exact_status"),
            "routing_failures": _counter(values, "routing_failure"),
            "relation_types": _counter(values, "relation_type"),
            "type_only": {
                "total": len(type_only),
                "tp": len(type_only_tp),
                "fp_exact_mismatch": len(type_only_fp),
                "tp_routing_failures": _counter(type_only_tp, "routing_failure"),
                "fp_routing_failures": _counter(type_only_fp, "routing_failure"),
                "tp_relation_types": _counter(type_only_tp, "relation_type"),
                "fp_relation_types": _counter(type_only_fp, "relation_type"),
                "tp_endpoint_support_modes": _counter(type_only_tp, "endpoint_support_mode"),
                "fp_endpoint_support_modes": _counter(type_only_fp, "endpoint_support_mode"),
                "fp_source_support_assessment": _counter(
                    type_only_fp, "source_support_assessment",
                ),
                "official_mismatch_kinds": _counter(type_only, "official_mismatch_kind"),
                "tp_routing_gates": dict(sorted(Counter(
                    gate for item in type_only_tp for gate in item.get("routing_gates", [])
                ).items())),
                "fp_routing_gates": dict(sorted(Counter(
                    gate for item in type_only_fp for gate in item.get("routing_gates", [])
                ).items())),
                "wrong_label_judge_matches_gold": sum(
                    bool(item.get("judge_target_matches_gold_pair"))
                    for item in type_only
                    if item.get("official_mismatch_kind") == "WRONG_LABEL"
                ),
            },
        }
    return {
        "audit_contract_version": "biored-semantic-routing-audit-v2",
        "audit_scope": (
            "Official exact-match status is separated from source-support assessment; "
            "the latter is diagnostic and never changes benchmark labels."
        ),
        "row_count": len(rows),
        "scopes": scopes,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# BioRED semantic routing audit",
        "",
        summary["audit_scope"],
        "",
    ]
    for scope, values in summary["scopes"].items():
        type_only = values["type_only"]
        lines.extend([
            f"## {scope}", "",
            f"- Removed total: {values['removed_total']}",
            f"- Exact status: `{json.dumps(values['official_status'], sort_keys=True)}`",
            f"- Type-only: {type_only['total']} (TP {type_only['tp']}, exact-mismatch FP {type_only['fp_exact_mismatch']})",
            f"- Type-only TP routing: `{json.dumps(type_only['tp_routing_failures'], sort_keys=True)}`",
            f"- Type-only FP routing: `{json.dumps(type_only['fp_routing_failures'], sort_keys=True)}`",
            f"- Type-only TP labels: `{json.dumps(type_only['tp_relation_types'], sort_keys=True)}`",
            f"- Type-only FP labels: `{json.dumps(type_only['fp_relation_types'], sort_keys=True)}`",
            f"- Type-only FP source audit: `{json.dumps(type_only['fp_source_support_assessment'], sort_keys=True)}`",
            f"- Type-only exact mismatch kinds: `{json.dumps(type_only['official_mismatch_kinds'], sort_keys=True)}`",
            f"- Type-only TP orthogonal gates: `{json.dumps(type_only['tp_routing_gates'], sort_keys=True)}`",
            f"- Type-only FP orthogonal gates: `{json.dumps(type_only['fp_routing_gates'], sort_keys=True)}`",
            f"- Wrong-label cases where judge target matches Gold pair label: {type_only['wrong_label_judge_matches_gold']}",
            "",
        ])
    return "\n".join(lines)


def render_primary_type_only_markdown(rows: list[dict[str, Any]]) -> str:
    values = [
        item for item in rows
        if item.get("scope") == "primary_to_semantic"
        and "native_relation_card_type_only" in item.get("quality_flags", [])
    ]
    lines = [
        "# BioRED primary-to-semantic type-only item ledger",
        "",
        "Every removed type-only candidate is listed once. Gold is used only for this post-hoc audit.",
        "Official exact mismatch and model source-support opinion remain separate columns.",
        "",
        "| # | PMID | endpoints | predicted | Gold pair labels | exact audit | judge | conf. | gates | evidence |",
        "|---:|---|---|---|---|---|---|---:|---|---|",
    ]
    for index, item in enumerate(values, start=1):
        left = "/".join(item["arg1"].get("mentions", [])[:2]) or item["arg1"]["concept_id"]
        right = "/".join(item["arg2"].get("mentions", [])[:2]) or item["arg2"]["concept_id"]
        evidence = str(item.get("evidence_quote", "") or item.get("evidence_window", ""))
        evidence = " ".join(evidence.split())[:180]
        cells = [
            str(index), str(item.get("pmid", "")),
            f"{left} [{item['arg1'].get('entity_type', '')}] ↔ {right} [{item['arg2'].get('entity_type', '')}]",
            str(item.get("relation_type", "")),
            ", ".join(item.get("gold_pair_labels", []) or ["NO_RELATION"]),
            str(item.get("official_mismatch_kind", "")),
            f"{item['judge'].get('relation_asserted', '')}/{item['judge'].get('recommended_relation_type', '')}",
            f"{float(item['judge'].get('confidence', 0.0) or 0.0):.2f}",
            ", ".join(item.get("routing_gates", [])), evidence,
        ]
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pubtator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    documents = {item.pmid: item for item in parse_biored_pubtator(args.pubtator)}
    rows: list[dict[str, Any]] = []
    for record in payload.get("records", []) or []:
        document = documents[str(record.get("pmid", ""))]
        rows.extend(_rows_for_scope(
            scope="primary_to_semantic", source_field="primary_relations",
            record=record, document=document,
        ))
        rows.extend(_rows_for_scope(
            scope="factual_to_semantic", source_field="factual_relations",
            record=record, document=document,
        ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output_dir / "routing_audit.jsonl"
    with ledger_path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = build_summary(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(
        render_markdown(summary) + "\n", encoding="utf-8",
    )
    primary_type_only = [
        item for item in rows
        if item.get("scope") == "primary_to_semantic"
        and "native_relation_card_type_only" in item.get("quality_flags", [])
    ]
    (args.output_dir / "primary_type_only_items.md").write_text(
        render_primary_type_only_markdown(rows) + "\n", encoding="utf-8",
    )
    csv_fields = [
        "pmid", "candidate_id", "candidate_version", "official_mismatch_kind",
        "relation_type", "candidate_relation_type", "candidate_official_mismatch_kind",
        "gold_pair_labels", "arg1_id", "arg1_type", "arg1_mentions",
        "arg2_id", "arg2_type", "arg2_mentions", "endpoint_support_mode",
        "routing_failure", "routing_gates", "judge_relation_asserted",
        "judge_recommended_relation_type", "judge_confidence", "judge_reason_code",
        "critic_relation_asserted", "critic_recommended_relation_type",
        "critic_confidence", "evidence_quote",
    ]
    with (args.output_dir / "primary_type_only_items.csv").open(
        "w", encoding="utf-8", newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for item in primary_type_only:
            writer.writerow({
                "pmid": item["pmid"], "candidate_id": item["candidate_id"],
                "candidate_version": item["candidate_version"],
                "official_mismatch_kind": item["official_mismatch_kind"],
                "relation_type": item["relation_type"],
                "candidate_relation_type": item["candidate_relation_type"],
                "candidate_official_mismatch_kind": item[
                    "candidate_official_mismatch_kind"
                ],
                "gold_pair_labels": ";".join(item["gold_pair_labels"]),
                "arg1_id": item["arg1"]["concept_id"],
                "arg1_type": item["arg1"]["entity_type"],
                "arg1_mentions": ";".join(item["arg1"]["mentions"]),
                "arg2_id": item["arg2"]["concept_id"],
                "arg2_type": item["arg2"]["entity_type"],
                "arg2_mentions": ";".join(item["arg2"]["mentions"]),
                "endpoint_support_mode": item["endpoint_support_mode"],
                "routing_failure": item["routing_failure"],
                "routing_gates": ";".join(item["routing_gates"]),
                "judge_relation_asserted": item["judge"]["relation_asserted"],
                "judge_recommended_relation_type": item["judge"]["recommended_relation_type"],
                "judge_confidence": item["judge"]["confidence"],
                "judge_reason_code": item["judge"]["reason_code"],
                "critic_relation_asserted": item["critic"]["relation_asserted"],
                "critic_recommended_relation_type": item["critic"]["recommended_relation_type"],
                "critic_confidence": item["critic"]["confidence"],
                "evidence_quote": item["evidence_quote"],
            })
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
