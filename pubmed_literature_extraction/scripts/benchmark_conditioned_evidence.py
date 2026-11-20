#!/usr/bin/env python3
"""Paired factorial ablation for scope-aware and argument-aware evidence.

Synthetic cases check mechanism behavior only; they are not labelled accuracy
data. Scope and argument switches are varied independently while the base
candidate and positive evidence quote remain fixed within each experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cognitive_agent.verifier import KGVerifier
from scripts.benchmark_evidence_contrast import OUTPUT, load_archive_cases


class OfflineKG:
    is_connected = False


@dataclass(frozen=True)
class Case:
    name: str
    support: str
    counter_context: str
    expected_scope: str

    @property
    def source(self) -> str:
        return f"TITLE: Cohort analysis\nABSTRACT: RESULTS: {self.support} {self.counter_context}"


SCOPE_CASES = (
    Case("same_cohort", "In cohort A, TP53 was associated with HCC.",
         "In cohort A, TP53 was not associated with HCC.", "SHARED_SCOPE"),
    Case("different_cohort", "In cohort A, TP53 was associated with HCC.",
         "In cohort B, TP53 was not associated with HCC.", "DIFFERENT_SCOPE"),
    Case("unknown_cohort", "In cohort A, TP53 was associated with HCC.",
         "In an unreported subgroup, TP53 was not associated with HCC.", "UNRESOLVED_SCOPE"),
    Case("missing_counter_scope", "In cohort A, TP53 was associated with HCC.",
         "TP53 was not associated with HCC.", "UNRESOLVED_SCOPE"),
    Case("multiple_counter_scopes", "In cohort A, TP53 was associated with HCC.",
         "In cohort A and cohort B, TP53 was not associated with HCC.", "UNRESOLVED_SCOPE"),
    Case("near_different_far_same", "In cohort A, TP53 was associated with HCC.",
         "In cohort B, TP53 was not associated with HCC. In cohort A, TP53 was not associated with HCC.", "SHARED_SCOPE"),
    Case("different_analysis_models", "In model 1, TP53 was associated with HCC.",
         "In model 2, TP53 was not associated with HCC.", "DIFFERENT_SCOPE"),
    Case("partial_scope", "In cohort A, TP53 was associated with HCC in model 1.",
         "In cohort A, TP53 was not associated with HCC.", "UNRESOLVED_SCOPE"),
    Case("sex_strata", "In men, TP53 was associated with HCC.",
         "In women, TP53 was not associated with HCC.", "DIFFERENT_SCOPE"),
)


@dataclass(frozen=True)
class ArgumentCase:
    name: str
    evidence_quote: str
    expected_competing: bool
    subject: str = "TP53"
    object: str = "HCC"
    object_type: str = "Disease"
    predicate: str = "ASSOCIATED_WITH"
    decoys: tuple[tuple[str, str], ...] = ()

    @property
    def source(self) -> str:
        return f"TITLE: Relation study\nABSTRACT: RESULTS: {self.evidence_quote}"


ARGUMENT_CASES = (
    ArgumentCase("direct_target_control", "TP53 was associated with HCC.", False),
    ArgumentCase("replacement_subject", "TP53 was measured; KRAS was associated with HCC.",
                 True, decoys=(("KRAS", "Gene"),)),
    ArgumentCase("replacement_object", "TP53 was associated with cirrhosis; HCC was recorded.",
                 True, decoys=(("cirrhosis", "Disease"),)),
    ArgumentCase("expression_decoy", "CYP2E1 was measured; CYP3A4 was expressed in hepatocytes.",
                 True, subject="CYP2E1", object="hepatocytes", object_type="CellType",
                 predicate="EXPRESSED_IN", decoys=(("CYP3A4", "Gene"),)),
    ArgumentCase("coordination_other_endpoint", "TP53 and KRAS were associated with HCC.",
                 False, decoys=(("KRAS", "Gene"),)),
    ArgumentCase("different_typed_endpoint", "TP53 was measured; KRAS was expressed in hepatocytes.",
                 False, decoys=(("KRAS", "Gene"), ("hepatocytes", "CellType"))),
    ArgumentCase("no_explicit_same_predicate", "TP53 was measured; KRAS predicted HCC.",
                 False, decoys=(("KRAS", "Gene"),)),
    ArgumentCase("negated_rival", "TP53 was measured; KRAS was not associated with HCC.",
                 False, decoys=(("KRAS", "Gene"),)),
)


def relation() -> dict[str, Any]:
    return {
        "subject": "TP53", "subject_type": "Gene", "predicate": "ASSOCIATED_WITH",
        "object": "HCC", "object_type": "Disease",
        "evidence": "TP53 was associated with HCC.",
        "direction": "positive", "negated": False, "uncertain": False,
    }


def entities(extra: dict[str, str] | None = None) -> list[dict]:
    known = {"TP53": "Gene", "HCC": "Disease", **(extra or {})}
    return [{"mention": mention, "type": kind, "attributes": {}}
            for mention, kind in known.items()]


def verify(text: str, *, scope: bool, arguments: bool):
    verifier = KGVerifier(
        OfflineKG(), verification_policy="tiered-v2",
        evidence_contrast=True, evidence_scope=scope, evidence_arguments=arguments,
    )
    return verifier.verify(entities(), [relation()], pmid="synthetic", text=text).relations[0]


def scope_readout(case: Case, enabled: bool, arguments: bool = False) -> dict:
    rel = verify(case.source, scope=enabled, arguments=arguments)
    contrast = rel.contrastive_evidence or {}
    return {
        "semantic_status": rel.semantic_status,
        "import_ready": bool(rel.import_ready),
        "scope_status": contrast.get("scope_status"),
        "scope_pair_counts": contrast.get("scope_pair_counts"),
        "scoped_assertions": contrast.get("scoped_assertions"),
        "legacy_status": contrast.get("status"),
    }


def argument_readout(case: ArgumentCase, enabled: bool, scope: bool = False) -> dict:
    verifier = KGVerifier(
        OfflineKG(), verification_policy="tiered-v2",
        evidence_contrast=True, evidence_scope=scope, evidence_arguments=enabled,
    )
    candidate = {
        **relation(), "evidence": case.evidence_quote, "predicate": case.predicate,
        "subject": case.subject, "object": case.object, "object_type": case.object_type,
        "direction": "unknown" if case.predicate == "EXPRESSED_IN" else "positive",
    }
    article_entities = [
        {"mention": mention, "type": kind, "attributes": {}}
        for mention, kind in ((case.subject, "Gene"), (case.object, case.object_type), *case.decoys)
    ]
    rel = verifier.verify(article_entities, [candidate], pmid=f"synthetic:{case.name}", text=case.source).relations[0]
    competition = rel.argument_competition
    witness = competition.get("evidence")
    witness_aligned = witness is not None and (
        case.source[witness["char_start"]:witness["char_end"]] == witness["text"]
    )
    return {
        "semantic_status": rel.semantic_status,
        "import_ready": bool(rel.import_ready),
        "argument_status": competition.get("status"),
        "witness_aligned": witness_aligned,
        "argument_competition": competition,
    }


def build_report() -> tuple[dict, bool]:
    scope_rows = []
    scope_pass = True
    for case in SCOPE_CASES:
        old = scope_readout(case, False)
        new = scope_readout(case, True)
        argument_only = scope_readout(case, False, arguments=True)
        combined = scope_readout(case, True, arguments=True)
        no_write_change = old["import_ready"] == new["import_ready"]
        status_pass = new["scope_status"] == case.expected_scope
        interaction_pass = (
            combined["scope_status"] == new["scope_status"]
            and argument_only["scope_status"] is None
        )
        scope_pass &= no_write_change and status_pass and interaction_pass
        scope_rows.append({
            "name": case.name, "expected_scope": case.expected_scope,
            "baseline": old, "scope_enabled": new,
            "argument_only": argument_only,
            "both_enabled": combined,
            "scope_status_pass": status_pass,
            "factor_interaction_preserves_scope": interaction_pass,
            "import_ready_unchanged": no_write_change,
        })

    argument_rows = []
    argument_pass = True
    for case in ARGUMENT_CASES:
        old = argument_readout(case, False)
        new = argument_readout(case, True)
        scope_only = argument_readout(case, False, scope=True)
        combined = argument_readout(case, True, scope=True)
        expected_status = "COMPETING_ARGUMENT" if case.expected_competing else None
        status_pass = new["argument_status"] == expected_status
        safe_review = not case.expected_competing or (
            old["semantic_status"] == "ACCEPTED" and old["import_ready"]
            and new["semantic_status"] == "REVIEW" and not new["import_ready"]
            and new["witness_aligned"]
        )
        unchanged_control = case.expected_competing or old["import_ready"] == new["import_ready"]
        interaction_pass = (
            combined["argument_status"] == new["argument_status"]
            and scope_only["argument_status"] is None
        )
        argument_pass &= status_pass and safe_review and unchanged_control and interaction_pass
        argument_rows.append({
            "name": case.name, "expected_competing": case.expected_competing,
            "baseline": old, "arguments_enabled": new,
            "scope_only": scope_only,
            "both_enabled": combined,
            "argument_status_pass": status_pass,
            "factor_interaction_preserves_argument": interaction_pass,
            "write_safety_pass": safe_review,
            "control_import_ready_unchanged": unchanged_control,
        })

    report = {
        "title": "Conditioned evidence factorial mechanism ablation",
        "warning": "Synthetic mechanism checks only; these cases are not labelled accuracy data.",
        "design": "Offline tiered-v2. Scope and argument toggles are independent; each case fixes candidate and positive quote.",
        "scope": {
            "metrics": {
                "expected_shared": sum(c.expected_scope == "SHARED_SCOPE" for c in SCOPE_CASES),
                "observed_shared": sum(r["scope_enabled"]["scope_status"] == "SHARED_SCOPE" for r in scope_rows),
                "expected_different": sum(c.expected_scope == "DIFFERENT_SCOPE" for c in SCOPE_CASES),
                "observed_different": sum(r["scope_enabled"]["scope_status"] == "DIFFERENT_SCOPE" for r in scope_rows),
                "expected_unresolved": sum(c.expected_scope == "UNRESOLVED_SCOPE" for c in SCOPE_CASES),
                "observed_unresolved": sum(r["scope_enabled"]["scope_status"] == "UNRESOLVED_SCOPE" for r in scope_rows),
                "scope_import_ready_flips": sum(not r["import_ready_unchanged"] for r in scope_rows),
            },
            "pass": scope_pass, "cases": scope_rows,
        },
        "arguments": {
            "metrics": {
                "expected_competing": sum(c.expected_competing for c in ARGUMENT_CASES),
                "observed_competing": sum(r["arguments_enabled"]["argument_status"] == "COMPETING_ARGUMENT" for r in argument_rows),
                "unintended_competition_controls": sum(
                    not c.expected_competing and r["arguments_enabled"]["argument_status"] == "COMPETING_ARGUMENT"
                    for c, r in zip(ARGUMENT_CASES, argument_rows)
                ),
                "argument_control_import_ready_flips": sum(
                    not r["control_import_ready_unchanged"] for r in argument_rows
                ),
                "competing_cases_left_import_ready": sum(
                    c.expected_competing and r["arguments_enabled"]["import_ready"]
                    for c, r in zip(ARGUMENT_CASES, argument_rows)
                ),
            },
            "pass": argument_pass, "cases": argument_rows,
        },
    }
    return report, scope_pass and argument_pass


def replay_archive(path: Path) -> dict:
    cases, article_count, digest = load_archive_cases(path, OUTPUT / "pubmed_converted_500.jsonl")
    baseline = KGVerifier(OfflineKG(), verification_policy="tiered-v2",
                          evidence_scope=False, evidence_arguments=False)
    proposed = KGVerifier(OfflineKG(), verification_policy="tiered-v2")
    counts = {
        "articles": article_count, "candidates": 0, "qualified_contrasts": 0,
        "argument_competitions": 0, "semantic_changes": 0, "write_promotions": 0,
    }
    for case in cases:
        before = baseline.verify(case["entities"], case["relations"], pmid=case["pmid"], text=case["text"])
        after = proposed.verify(case["entities"], case["relations"], pmid=case["pmid"], text=case["text"])
        for prior, current in zip(before.relations, after.relations, strict=True):
            counts["candidates"] += 1
            counts["qualified_contrasts"] += bool(current.contrastive_evidence.get("scoped_assertions"))
            counts["argument_competitions"] += bool(current.argument_competition)
            counts["semantic_changes"] += prior.semantic_status != current.semantic_status
            counts["write_promotions"] += not prior.import_ready and current.import_ready
    return {**counts, "archive_sha256": digest, "labels": "unlabelled replay"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--replay-archive", type=Path, help="optionally replay the first 30 archived articles")
    args = parser.parse_args()
    report, passed = build_report()
    if args.replay_archive is not None:
        replay = replay_archive(args.replay_archive)
        report["frozen_archive_replay"] = replay
        passed = passed and replay["write_promotions"] == 0
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(report["title"])
        for feature in ("scope", "arguments"):
            metrics = report[feature]["metrics"]
            print(feature, json.dumps(metrics, ensure_ascii=False))
        if "frozen_archive_replay" in report:
            print("unlabelled replay", json.dumps(report["frozen_archive_replay"], ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
