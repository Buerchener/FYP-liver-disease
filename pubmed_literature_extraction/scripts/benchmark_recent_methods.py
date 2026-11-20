#!/usr/bin/env python3
"""Paired regression checks on three hand-selected real PubMed abstracts.

These cases verify mechanism behavior only. They are neither a representative
sample nor evidence of generalized accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cognitive_agent.schema.schema_profile import schema_profile_from_dict
from cognitive_agent.verifier import KGVerifier


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "extraction_output" / "pubmed_converted_500.jsonl"


class OfflineGraph:
    is_connected = False


def source_record(pmid: str) -> dict:
    with SOURCES.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if str(record.get("pmid")) == pmid:
                return record
    raise ValueError(f"PMID {pmid} not found in {SOURCES}")


def source_text(record: dict) -> str:
    return f"TITLE: {record['title']}\nABSTRACT: {record['abstract']}"


def custom_profile(name: str, entity_types: list[str], predicate: str, signatures: list[list[str]],
                   explicit_patterns: list[str]) -> Any:
    return schema_profile_from_dict({
        "name": name, "contract_version": "benchmark-v1",
        "semantic_target": "RELATION_TRUTH", "entity_types": entity_types,
        "predicates": {predicate: {
            "allowed_signatures": signatures,
            "explicit_patterns": explicit_patterns,
            "weak_patterns": [], "exclusion_patterns": [],
        }},
    })


def snapshot(rel: Any) -> dict:
    return {
        "factual_status": rel.factual_status,
        "semantic_status": rel.semantic_status,
        "import_ready": bool(rel.import_ready),
        "semantic_reasons": list(rel.semantic_reasons),
        "quality_flags": list(rel.quality_flags),
        "evidence": rel.evidence,
        "contrastive_evidence": rel.contrastive_evidence or {},
        "argument_competition": rel.argument_competition or {},
    }


def verify(record: dict, candidate: dict, entities: list[dict], *,
           evidence_statistics: bool = True,
           evidence_group_binding: bool = True,
           profile: Any = None) -> Any:
    text = source_text(record)
    verifier = KGVerifier(
        OfflineGraph(), verification_policy="tiered-v2",
        evidence_statistics=evidence_statistics,
        evidence_group_binding=evidence_group_binding,
        schema_profile=profile,
    )
    return verifier.verify(entities, [candidate], pmid=str(record["pmid"]), text=text).relations[0]


def assert_trace(record: dict, trace: dict) -> None:
    text = source_text(record)
    start, end = int(trace["char_start"]), int(trace["char_end"])
    if text[start:end] != trace["text"]:
        raise AssertionError(f"trace offset mismatch at {start}:{end}: {trace.get('text')!r}")


def argument_case() -> dict:
    record = source_record("41948344")
    text = source_text(record)
    quote = (
        "scRNA-seq validated cell-type-specific expression patterns of CCND1 "
        "(epithelial cells, endothelial cells, hepatocytes, macrophages) and "
        "IL7R (T/NK cells)."
    )
    if quote not in text:
        raise AssertionError("CCND1 source quote no longer matches the frozen local abstract")
    entities = [{"mention": "CCND1", "type": "Gene", "attributes": {}},
                {"mention": "IL7R", "type": "Gene", "attributes": {}}]
    entities.extend({"mention": x, "type": "CellType", "attributes": {}}
                    for x in ("epithelial cells", "endothelial cells", "hepatocytes", "macrophages", "T/NK cells"))
    profile = None  # Use the project's real LiverKG profile and write contract.
    candidates = {
        "wrong_CCND1_to_T_NK": {
            "subject": "CCND1", "subject_type": "Gene", "predicate": "EXPRESSED_IN",
            "object": "T/NK cells", "object_type": "CellType",
            "evidence": quote, "claim_role": "CURRENT_FINDING",
        },
        "supported_CCND1_to_hepatocytes": {
            "subject": "CCND1", "subject_type": "Gene", "predicate": "EXPRESSED_IN",
            "object": "hepatocytes", "object_type": "CellType",
            "evidence": quote, "claim_role": "CURRENT_FINDING",
        },
    }
    results = {}
    for name, candidate in candidates.items():
        before = verify(record, candidate, entities, evidence_group_binding=False, profile=profile)
        after = verify(record, candidate, entities, profile=profile)
        output = {"baseline_arguments_off": snapshot(before), "arguments_on": snapshot(after)}
        competition = after.argument_competition or {}
        if name.startswith("wrong_"):
            rival = competition.get("replacement") or {}
            argument_spans = competition.get("arguments") or {}
            for trace in (argument_spans.get("subject"), argument_spans.get("object"),
                          competition.get("group"), competition.get("trigger")):
                if trace:
                    assert_trace(record, trace)
            source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            output["expected"] = "baseline import-ready; rival IL7R bound to T/NK; REVIEW and not import-ready"
            output["passed"] = (
                competition.get("status") == "COMPETING_ARGUMENT"
                and competition.get("binding_mode") == "PARENTHETICAL_GROUP"
                and rival.get("mention") == "IL7R" and rival.get("type") == "Gene"
                and competition.get("source_sha256") == source_hash
                and before.semantic_status == "ACCEPTED" and before.import_ready
                and after.semantic_status == "REVIEW" and not after.import_ready
                and bool(competition.get("evidence", {}).get("text"))
            )
            if competition.get("evidence"):
                assert_trace(record, competition["evidence"])
        else:
            output["expected"] = "no false competitor; ACCEPTED/import-ready in both arms"
            output["passed"] = (
                not competition
                and before.import_ready == after.import_ready
                and before.semantic_status == after.semantic_status
                and before.semantic_status == "ACCEPTED" and before.import_ready
            )
        results[name] = output
    return {
        "pmid": "41948344", "article_title": record["title"],
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "candidate_evidence_quote": quote,
        "cases": results,
    }


def statistics_case() -> dict:
    record = source_record("41610839")
    text = source_text(record)
    positive = (
        "In MASH patients without T2DM, semaglutide improved fibrosis stage "
        "(RR = 1.72, 95% CI 1.21-2.44, p = 0.002, N comparisons = 4, I2 = 3%, 474 participants)"
    )
    negative = (
        "In MASH patients with T2DM, semaglutide did not improve fibrosis stage "
        "(RR = 1.21, 95% CI 0.94-1.56, p = 0.13; I2 = 0%, N comparisons = 4, 646 participants)"
    )
    for quote in (positive, negative):
        if quote not in text:
            raise AssertionError(f"semaglutide source quote missing: {quote}")
    profile = custom_profile(
        "SemaglutideFibrosisBenchmark", ["Intervention", "Outcome"], "IMPROVES",
        [["Intervention", "Outcome"]], [r"\bimprov(?:e|es|ed)\b"],
    )
    candidate = {
        "subject": "semaglutide", "subject_type": "Intervention", "predicate": "IMPROVES",
        "object": "fibrosis stage", "object_type": "Outcome", "evidence": positive,
        "claim_role": "CURRENT_FINDING",
    }
    entities = [{"mention": "semaglutide", "type": "Intervention", "attributes": {}},
                {"mention": "fibrosis stage", "type": "Outcome", "attributes": {}}]
    before = verify(record, candidate, entities, evidence_statistics=False, profile=profile)
    after = verify(record, candidate, entities, evidence_statistics=True, profile=profile)
    comparison = after.contrastive_evidence or {}
    statistical = (comparison.get("inconclusive") or {}).get("statistical_evidence") or {}
    estimate = statistical.get("estimate") or {}
    counter = comparison.get("counterevidence") or {}
    inconclusive = comparison.get("inconclusive") or {}
    support = comparison.get("support") or {}
    for trace in (support, inconclusive, counter, estimate):
        if trace.get("text"):
            assert_trace(record, trace)
    expected_estimate = (
        estimate.get("measure") == "RR" and estimate.get("point") == 1.21
        and estimate.get("lower") == 0.94 and estimate.get("upper") == 1.56
        and estimate.get("null") == 1.0
    )
    return {
        "pmid": "41610839", "article_title": record["title"],
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "fixed_positive_quote": positive, "opposing_cohort_quote": negative,
        "baseline_statistics_off": snapshot(before), "statistics_on": snapshot(after),
        "expected": "statistically inconclusive; both variants remain REVIEW and not import-ready",
        "passed": (
            expected_estimate
            and after.semantic_status == "REVIEW" and not after.import_ready
            and before.semantic_status == "REVIEW" and not before.import_ready
            and comparison.get("scope_status") == "DIFFERENT_SCOPE"
            and (before.contrastive_evidence or {}).get("status") == "CONTESTED"
            and before.contrastive_evidence.get("scope_status") == "DIFFERENT_SCOPE"
            and comparison.get("source_sha256") == hashlib.sha256(text.encode("utf-8")).hexdigest()
            and comparison.get("status") == "INCONCLUSIVE"
        ),
    }


def endpoint_control_case() -> dict:
    record = source_record("41846463")
    text = source_text(record)
    quote = "RRT on days 1 to 3 was associated with higher hazard of LT"
    if quote not in text:
        raise AssertionError("endpoint-control source quote missing")
    profile = custom_profile(
        "RRTOutcomeBenchmark", ["Exposure", "Outcome"], "ASSOCIATED_WITH",
        [["Exposure", "Outcome"]], [r"\bassociated with\b"],
    )
    candidate = {
        "subject": "RRT on days 1 to 3", "subject_type": "Exposure",
        "predicate": "ASSOCIATED_WITH", "object": "higher hazard of LT",
        "object_type": "Outcome", "evidence": quote, "claim_role": "CURRENT_FINDING",
    }
    entities = [{"mention": candidate[k], "type": candidate[f"{k}_type"], "attributes": {}}
                for k in ("subject", "object")]
    before = verify(record, candidate, entities, evidence_statistics=False, profile=profile)
    after = verify(record, candidate, entities, evidence_statistics=True, profile=profile)
    output = {"baseline_statistics_off": snapshot(before), "statistics_on": snapshot(after)}
    output["expected"] = "the mortality counterstatement is endpoint-mismatched; statistics do not alter it"
    output["passed"] = (
        before.semantic_status == after.semantic_status
        and before.import_ready == after.import_ready
        and before.semantic_status == after.semantic_status == "ACCEPTED"
        and not before.contrastive_evidence and not after.contrastive_evidence
    )
    return {
        "pmid": "41846463", "article_title": record["title"],
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "candidate_quote": quote, "case": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    report = {
        "title": "Hand-selected real-source mechanism regression",
        "warning": "Three selected PubMed records; no generalized accuracy or novelty claim.",
        "argument_competition": argument_case(),
        "statistical_opposition": statistics_case(),
        "endpoint_hard_control": endpoint_control_case(),
    }
    checks = [
        case["passed"] for case in report["argument_competition"]["cases"].values()
    ] + [report["statistical_opposition"]["passed"],
         report["endpoint_hard_control"]["case"]["passed"]]
    report["summary"] = {"checks_passed": sum(checks), "checks_total": len(checks), "passed": all(checks)}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(report["title"])
        print(report["warning"])
        print(f"Pass: {report['summary']['checks_passed']}/{report['summary']['checks_total']}")
        for name, case in report["argument_competition"]["cases"].items():
            print(f"{name}: {'PASS' if case['passed'] else 'FAIL'}")
        for name in ("statistical_opposition", "endpoint_hard_control"):
            print(f"{name}: {'PASS' if report[name].get('passed', report[name].get('case', {}).get('passed')) else 'FAIL'}")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
