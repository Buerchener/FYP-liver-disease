#!/usr/bin/env python3
"""Paired, offline ablation for same-article evidence contrast.

This is a mechanism check, not an accuracy benchmark: the synthetic cases have
declared expected routing, but are not human-labelled biomedical data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cognitive_agent.verifier import KGVerifier


class OfflineKG:
    is_connected = False


SUPPORT = "TP53 was associated with HCC."
SUBJECT = {"mention": "TP53", "type": "Gene", "attributes": {}}
DISEASE = {"mention": "HCC", "type": "Disease", "attributes": {}}
ALT = {"mention": "ALT", "type": "Gene", "attributes": {}}


@dataclass(frozen=True)
class Fixture:
    name: str
    subject: str
    subject_type: str
    predicate: str
    obj: str
    object_type: str
    support: str
    surrounding: str
    expected_contested: bool
    control: bool

    @property
    def source(self) -> str:
        return f"TITLE: Synthetic study\nABSTRACT: RESULTS: {self.support} {self.surrounding}"


def fixture(name, surrounding, *, expected=False, control=True,
            subject="TP53", subject_type="Gene", predicate="ASSOCIATED_WITH",
            obj="HCC", object_type="Disease", support="TP53 was associated with HCC."):
    return Fixture(name, subject, subject_type, predicate, obj, object_type,
                   support, surrounding, expected, control)


FIXTURES = (
    fixture("positive_only", ""),
    fixture("same_pair_denial", "However, TP53 was not associated with HCC.", expected=True, control=False),
    fixture("same_pair_positive_repeat", "TP53 was also associated with HCC."),
    fixture("different_endpoint_denial", "TP53 was not associated with ALT."),
    fixture("same_clause_other_endpoint_hard_negative", "TP53 was not associated with ALT and HCC was measured."),
    fixture("independent_claim_denial", "HCC was not associated with AFP."),
    fixture("methods_denial", "METHODS: TP53 was not associated with HCC."),
    fixture("background_denial", "BACKGROUND: TP53 was not associated with HCC."),
    fixture("not_only_control", "TP53 was not only associated with HCC but also with AFP."),
    fixture("protein_interaction_denial", "However, TP53 did not interact with BRCA1.",
            expected=True, control=False, subject="TP53", subject_type="Protein",
            predicate="INTERACTS_WITH", obj="BRCA1", object_type="Protein",
            support="TP53 interacts with BRCA1."),
    fixture("gene_cell_expression_denial", "However, CYP2E1 was not expressed in hepatocytes.",
            expected=True, control=False, subject="CYP2E1", subject_type="Gene",
            predicate="EXPRESSED_IN", obj="hepatocytes", object_type="CellType",
            support="CYP2E1 was expressed in hepatocytes."),
)

OUTPUT = Path(__file__).resolve().parents[1] / "extraction_output"


def load_archive_cases(archive_path: Path, sources_path: Path, limit: int = 30) -> tuple[list[dict], int, str]:
    """Load a bounded frozen extraction archive and its matching source text."""
    sources = {}
    with sources_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                source = json.loads(line)
                sources[str(source["pmid"])] = source
    payload = json.loads(archive_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])[:limit]
    cases = []
    for record in records:
        pmid = str(record.get("pmid", ""))
        source = sources.get(pmid)
        if source is None:
            raise ValueError(f"Missing frozen source for PMID {pmid}")
        extraction = (record.get("phases", {}).get("extraction", {}) or {})
        entities = extraction.get("entities", []) or []
        relations = extraction.get("relations", []) or []
        if relations:
            cases.append({
                "pmid": pmid, "entities": entities, "relations": relations,
                "text": f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}",
            })
    return cases, len(records), hashlib.sha256(archive_path.read_bytes()).hexdigest()


def replay_archive(cases: list[dict], *, records_examined: int, archive_sha256: str) -> dict:
    before_verifier = KGVerifier(OfflineKG(), verification_policy="tiered-v2", evidence_contrast=False)
    after_verifier = KGVerifier(OfflineKG(), verification_policy="tiered-v2", evidence_contrast=True)
    totals = {"archive_records_examined": records_examined,
              "articles_with_candidates": len(cases), "candidates": 0, "status_changes": 0,
              "new_reviews": 0, "import_ready_flips": 0, "contested_after": 0}
    for case in cases:
        before = before_verifier.verify(case["entities"], case["relations"], pmid=case["pmid"], text=case["text"])
        after = after_verifier.verify(case["entities"], case["relations"], pmid=case["pmid"], text=case["text"])
        for old, new in zip(before.relations, after.relations, strict=True):
            totals["candidates"] += 1
            totals["status_changes"] += old.semantic_status != new.semantic_status
            totals["new_reviews"] += old.semantic_status != "REVIEW" and new.semantic_status == "REVIEW"
            totals["import_ready_flips"] += old.import_ready != new.import_ready
            totals["contested_after"] += bool(getattr(new, "contrastive_evidence", None))
    return {
        "note": "Exploratory replay of frozen, unlabelled candidates; counts gate changes only, not accuracy.",
        "archive_sha256": archive_sha256,
        **totals,
    }


def run(enabled: bool) -> list[dict]:
    verifier = KGVerifier(
        OfflineKG(), verification_policy="tiered-v2", evidence_contrast=enabled,
    )
    output = []
    for case in FIXTURES:
        entities = [
            {"mention": case.subject, "type": case.subject_type, "attributes": {}},
            {"mention": case.obj, "type": case.object_type, "attributes": {}},
        ]
        # Negative-control endpoints are present as entities too, so the test
        # specifically checks pair scoping, rather than mention resolution.
        extras = {"ALT": "Gene", "AFP": "Protein", "BRCA1": "Protein"}
        for mention, entity_type in extras.items():
            if mention in case.source and mention not in {case.subject, case.obj}:
                entities.append({"mention": mention, "type": entity_type, "attributes": {}})
        relation = {
            "subject": case.subject, "subject_type": case.subject_type,
            "predicate": case.predicate, "object": case.obj,
            "object_type": case.object_type, "evidence": case.support,
            "direction": "positive", "negated": False, "uncertain": False,
        }
        result = verifier.verify(entities, [relation], pmid=f"synthetic:{case.name}", text=case.source)
        rel = result.relations[0]
        output.append({
            "name": case.name, "expected_contested": case.expected_contested,
            "control": case.control, "factual_status": rel.factual_status,
            "semantic_status": rel.semantic_status,
            "import_ready": rel.import_ready,
            "quality_flags": list(rel.quality_flags),
            "contrastive_evidence": getattr(rel, "contrastive_evidence", None),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--replay-archive", type=Path,
                        help="optionally replay a frozen extraction JSON archive (exploratory; no labels)")
    parser.add_argument("--replay-sources", type=Path, default=OUTPUT / "pubmed_converted_500.jsonl",
                        help="JSONL source archive matching --replay-archive")
    parser.add_argument("--replay-limit", type=int, default=30,
                        help="maximum archive records to replay (default: 30)")
    args = parser.parse_args()
    if args.replay_limit < 1:
        parser.error("--replay-limit must be positive")
    baseline, contrast = run(False), run(True)
    pairs = []
    for case, before, after in zip(FIXTURES, baseline, contrast):
        # Guard the paired design: candidate relation and quoted positive support
        # are constants; only surrounding source sentences vary.
        assert case.support in case.source
        pairs.append({
            "name": case.name, "expected_contested": case.expected_contested,
            "candidate": f"{case.subject}:{case.subject_type} {case.predicate} {case.obj}:{case.object_type}",
            "control": case.control,
            "baseline": before, "contrast": after,
            "newly_contested": before["semantic_status"] != "REVIEW" and after["semantic_status"] == "REVIEW",
            "import_ready_delta": int(after["import_ready"]) - int(before["import_ready"]),
        })
    intended = sum(
        p["expected_contested"]
        and p["baseline"]["semantic_status"] == "ACCEPTED"
        and p["contrast"]["semantic_status"] == "REVIEW"
        and bool(p["contrast"]["contrastive_evidence"])
        and not p["contrast"]["import_ready"]
        for p in pairs
    )
    false_review = sum(p["control"] and p["contrast"]["semantic_status"] == "REVIEW" for p in pairs)
    report = {
        "title": "Synthetic paired mechanism ablation (not an accuracy estimate)",
        "design": "Offline tiered-v2; each pair fixes its typed candidate and positive quote, varying only surrounding context.",
        "summary": {
            "expected_conflict_cases": sum(p["expected_contested"] for p in pairs),
            "expected_conflicts_routed_to_review": intended,
            "control_cases": sum(p["control"] for p in pairs),
            "unintended_control_reviews": false_review,
            "newly_contested_pairs": sum(p["newly_contested"] for p in pairs),
            "import_ready_deltas": sum(p["import_ready_delta"] for p in pairs),
        },
        "cases": pairs,
    }
    if args.replay_archive:
        archive_cases, records_examined, archive_sha256 = load_archive_cases(
            args.replay_archive, args.replay_sources, args.replay_limit,
        )
        report["frozen_archive_replay"] = replay_archive(
            archive_cases, records_examined=records_examined, archive_sha256=archive_sha256,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else _pretty(report))
    return int(intended != report["summary"]["expected_conflict_cases"] or false_review > 0)


def _pretty(report: dict) -> str:
    s = report["summary"]
    lines = [report["title"], report["design"], "",
             f"Conflicts routed to review: {s['expected_conflicts_routed_to_review']}/{s['expected_conflict_cases']}",
             f"Unintended control reviews: {s['unintended_control_reviews']}/{s['control_cases']}",
             f"Newly contested pairs: {s['newly_contested_pairs']}; summed import-ready deltas: {s['import_ready_deltas']}", "",
             "Per-case: name | baseline status→contrast status | import-ready Δ"]
    lines.extend(f"{p['name']} | {p['baseline']['semantic_status']}→{p['contrast']['semantic_status']} | {p['import_ready_delta']:+d}" for p in report["cases"])
    if archive := report.get("frozen_archive_replay"):
        lines.extend((
            "",
            archive["note"],
            f"{archive['candidates']} frozen candidates from {archive['archive_records_examined']} articles.",
            f"Semantic decisions changed: {archive['status_changes']}. "
            f"Import eligibility changed: {archive['import_ready_flips']}.",
        ))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
