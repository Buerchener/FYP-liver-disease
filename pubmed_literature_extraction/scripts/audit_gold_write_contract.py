#!/usr/bin/env python3
"""Audit gold annotations against the current Neo4j write contract.

The gold file may intentionally contain broad literature relations.  This
script separates those candidate-level annotations from relations that match
the current eight writeable LiverKG relation patterns.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


WRITE_CONTRACT = frozenset({
    ("ASSOCIATED_WITH", "Gene", "Disease"),
    ("ASSOCIATED_WITH", "Metabolite", "Disease"),
    ("PROGNOSTIC_IN", "Gene", "Disease"),
    ("PROGRESSES_TO", "Disease", "Disease"),
    ("ENCODES", "Gene", "Protein"),
    ("INTERACTS_WITH", "Protein", "Protein"),
    ("PARTICIPATES_IN", "Gene", "Pathway"),
    ("EXPRESSED_IN", "Gene", "Tissue"),
    ("EXPRESSED_IN", "Gene", "CellType"),
    ("ASSOCIATED_WITH_METABOLITE", "Gene", "Metabolite"),
})


def _relation_signature(relation: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(relation.get("predicate", "") or "").upper(),
        str(relation.get("subject_type", "") or ""),
        str(relation.get("object_type", "") or ""),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicate_counts: Counter[str] = Counter()
    signature_counts: Counter[tuple[str, str, str]] = Counter()
    outside_counts: Counter[tuple[str, str, str]] = Counter()
    relation_total = 0
    write_contract_total = 0
    strict_import_ready_total = 0
    outside_examples: list[dict[str, Any]] = []
    for row in rows:
        for relation in row.get("relations", []) or []:
            relation_total += 1
            signature = _relation_signature(relation)
            predicate_counts[signature[0]] += 1
            signature_counts[signature] += 1
            if signature in WRITE_CONTRACT:
                write_contract_total += 1
                if bool(relation.get("import_ready", False)):
                    strict_import_ready_total += 1
            else:
                outside_counts[signature] += 1
                if len(outside_examples) < 40:
                    outside_examples.append({
                        "pmid": row.get("pmid", ""),
                        "title": row.get("title", ""),
                        "signature": list(signature),
                        "relation": {
                            key: relation.get(key)
                            for key in (
                                "subject", "subject_type", "predicate",
                                "object", "object_type", "evidence", "claim_role",
                            )
                        },
                    })
    return {
        "article_count": len(rows),
        "relation_count": relation_total,
        "write_contract_relation_count": write_contract_total,
        "strict_import_ready_write_contract_relation_count": strict_import_ready_total,
        "candidate_only_relation_count": relation_total - write_contract_total,
        "predicate_counts": dict(sorted(predicate_counts.items())),
        "signature_counts": {
            "|".join(signature): count
            for signature, count in signature_counts.most_common()
        },
        "candidate_only_signature_counts": {
            "|".join(signature): count
            for signature, count in outside_counts.most_common()
        },
        "outside_examples": outside_examples,
        "recommendation": (
            "Do not delete candidate-only gold relations. Add an evaluation "
            "view: write_contract_gold for strict import-ready scoring and "
            "candidate_gold for semantic candidate recall/review scoring."
        ),
    }


def write_jsonl(
    rows: list[dict[str, Any]], path: Path, *,
    write_contract_only: bool,
    strict_import_ready_only: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            output = dict(row)
            relations = []
            for relation in row.get("relations", []) or []:
                rel = dict(relation)
                in_contract = _relation_signature(rel) in WRITE_CONTRACT
                rel["gold_write_status"] = "WRITE_CONTRACT" if in_contract else "CANDIDATE_ONLY"
                if write_contract_only and not in_contract:
                    continue
                if strict_import_ready_only and not (in_contract and bool(rel.get("import_ready", False))):
                    continue
                relations.append(rel)
            output["relations"] = relations
            handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--candidate-view", type=Path, default=None)
    parser.add_argument("--write-contract-view", type=Path, default=None)
    parser.add_argument("--strict-import-view", type=Path, default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.gold)
    report = audit(rows)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if args.candidate_view:
        write_jsonl(rows, args.candidate_view, write_contract_only=False)
    if args.write_contract_view:
        write_jsonl(rows, args.write_contract_view, write_contract_only=True)
    if args.strict_import_view:
        write_jsonl(
            rows, args.strict_import_view,
            write_contract_only=True,
            strict_import_ready_only=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
