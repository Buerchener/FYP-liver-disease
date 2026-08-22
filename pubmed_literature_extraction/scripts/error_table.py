#!/usr/bin/env python3
"""Round-3 error table.

Two questions this answers with data:

1. Claim Gate P/R vs gold: for every gold relation, is its candidate pair
   present in the gate table (gate_table in phases.pairwise_judge) and did it
   get DIRECT_FINDING?  For non-gold pairs, how many were allowed through?
2. Gold NO_RELATION → predicted ASSOCIATED_WITH* false positives, bucketed by
   provenance (claim_gate_* / judge_* / legacy) and by deterministic error
   signatures of the evidence sentence (cohort/background/measurement/...).

Read-only analysis over existing agent results JSONs; no API calls.
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

from cognitive_agent.few_shot_retriever import detect_error_signatures  # noqa: E402
from run_pairwise_judge_experiments import (  # noqa: E402
    _aliases,
    _canonical_endpoint,
    _relation_key,
    load_jsonl,
)

GOLD_PATH = ROOT / "gold_annotations/pubmed_200_gold_v2_strict.jsonl"
SOURCE_PATH = ROOT / "extraction_output/pubmed_converted_500.jsonl"
AW_PREDICATES = {"ASSOCIATED_WITH", "ASSOCIATED_WITH_METABOLITE"}


def gate_key(row: dict, aliases: dict) -> tuple[str, str]:
    """(subject canonical, object canonical) for a gate_table row."""
    return (
        _canonical_endpoint(str(row.get("subject", "")), str(row.get("subject_type", "")), aliases),
        _canonical_endpoint(str(row.get("object", "")), str(row.get("object_type", "")), aliases),
    )


def gold_key(relation: dict, aliases: dict) -> tuple[str, str]:
    key = _relation_key(relation, aliases)
    return key[0], key[2]


def analyze_arm(records: list[dict], gold_by_pmid: dict, source_by_pmid: dict) -> dict[str, Any]:
    # ── Claim Gate P/R vs gold ──
    gate_candidates = 0          # gold pairs that appear as gate candidates
    gate_passed = 0              # ... and got DIRECT_FINDING
    gate_blocked: Counter = Counter()      # ... and got <status>
    gate_allow_tp = 0            # DIRECT_FINDING rows matching a gold pair
    gate_allow_fp = 0            # DIRECT_FINDING rows matching no gold pair
    gate_total = 0               # rows with any claim_status
    not_candidate: Counter = Counter()     # gold miss reasons (proxy: empty)

    # ── AW false positives on zero-relation docs ──
    zero_docs = 0
    aw_fp_rows: list[dict] = []
    all_aw_fp_rows: list[dict] = []
    signature_counts: Counter = Counter()
    provenance_counts: Counter = Counter()

    for record in records:
        pmid = str(record.get("pmid", ""))
        gold = gold_by_pmid.get(pmid)
        if gold is None:
            continue
        source = source_by_pmid.get(pmid, {})
        text = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"
        aliases = _aliases(gold, text)
        gold_relations = gold.get("relations", []) or []
        gold_pairs = {gold_key(item, aliases) for item in gold_relations}
        phases = record.get("phases", {}) or {}
        verification = phases.get("verification", {}) or {}
        relations = verification.get("relations", []) or []

        gate_table = (phases.get("pairwise_judge", {}) or {}).get("gate_table", []) or []
        gate_rows = []
        for row in gate_table:
            gate_rows.append((gate_key(row, aliases), row))
            status = str(row.get("claim_status", "") or "")
            if not status:
                continue
            gate_total += 1
            if status == "DIRECT_FINDING":
                if gate_key(row, aliases) in gold_pairs:
                    gate_allow_tp += 1
                else:
                    gate_allow_fp += 1
        row_by_pair = dict(gate_rows)
        for pair in gold_pairs:
            row = row_by_pair.get(pair)
            if row is None:
                not_candidate["not_a_candidate"] += 1
                continue
            gate_candidates += 1
            status = str(row.get("claim_status", "") or "")
            if status == "DIRECT_FINDING":
                gate_passed += 1
            elif status:
                gate_blocked[status] += 1
            else:
                gate_blocked["unjudged"] += 1

        # Predicted keys (alias-aware, all-verified口径) and provenance.
        pred_by_key: dict[tuple[str, str], dict] = {}
        for relation in relations:
            key = _relation_key(relation, aliases)
            pred_by_key.setdefault((key[0], key[2]), relation)

        aw_preds = [
            relation for relation in relations
            if str(relation.get("predicate", "")).upper() in AW_PREDICATES
        ]
        is_zero_doc = not gold_relations
        if is_zero_doc:
            zero_docs += 1
        for relation in aw_preds:
            flags = set(relation.get("quality_flags", []) or [])
            evidence = str(relation.get("evidence", "") or "")
            signatures = detect_error_signatures(evidence)
            provenance = (
                "gate" if any(flag.startswith("claim_gate_") for flag in flags)
                else "judge" if "pairwise_judge" in flags
                else "legacy"
            )
            provenance_counts[provenance] += 1
            for signature in signatures:
                signature_counts[signature] += 1
            row = {
                "pmid": pmid,
                "subject": relation.get("subject", ""),
                "object": relation.get("object", ""),
                "predicate": relation.get("predicate", ""),
                "semantic_status": relation.get("semantic_status", ""),
                "flags": sorted(flags),
                "evidence": evidence[:220],
                "signatures": signatures,
                "provenance": provenance,
                "gate_status": "",
            }
            pair = (relation.get("subject", ""), relation.get("object", ""))
            gate_row = row_by_pair.get(pair)
            if gate_row is not None:
                row["gate_status"] = gate_row.get("claim_status", "")
            all_aw_fp_rows.append(row)
            if is_zero_doc:
                aw_fp_rows.append(row)

    gate_precision = gate_allow_tp / max(gate_allow_tp + gate_allow_fp, 1)
    gate_recall = gate_passed / max(gate_candidates, 1)
    return {
        "gate": {
            "total_decisions": gate_total,
            "direct_finding_pass": gate_allow_tp + gate_allow_fp,
            "direct_finding_tp": gate_allow_tp,
            "direct_finding_fp": gate_allow_fp,
            "precision": round(gate_precision, 4),
            "gold_candidates": gate_candidates,
            "gold_passed": gate_passed,
            "recall": round(gate_recall, 4),
            "blocked_by_status": dict(gate_blocked),
            "not_candidate": dict(not_candidate),
        },
        "aw_fp": {
            "total_aw_fp": len(all_aw_fp_rows),
            "zero_doc_aw_fp": len(aw_fp_rows),
            "zero_docs": zero_docs,
            "provenance": dict(provenance_counts),
            "signatures": dict(signature_counts),
            "rows": aw_fp_rows,
        },
    }


def markdown(arm_results: dict[str, dict]) -> str:
    lines = [
        "# Round-3 error table: Claim Gate P/R and ASSOCIATED_WITH FP forensics",
        "",
        "Gold = v2-strict dev gold.  All-verified口径 unless noted.  Dry-run arms only.",
        "",
        "## Claim Gate vs gold pairs",
        "",
        "| Arm | Gate decisions | DIRECT_FINDING (TP/FP) | Gate P | Gold pairs as candidates | Passed (R) | Blocked by status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, metrics in arm_results.items():
        gate = metrics["gate"]
        lines.append(
            f"| {name} | {gate['total_decisions']} | "
            f"{gate['direct_finding_tp']}/{gate['direct_finding_fp']} | "
            f"{gate['precision']:.3f} | {gate['gold_candidates']} | "
            f"{gate['gold_passed']} ({gate['recall']:.3f}) | "
            f"{gate['blocked_by_status']} |"
        )
    lines += [
        "",
        "## ASSOCIATED_WITH* false positives by provenance",
        "",
        "| Arm | AW FP total | AW FP on zero-relation docs | legacy / judge / gate-code | Evidence error signatures |",
        "|---|---:|---:|---:|---|",
    ]
    for name, metrics in arm_results.items():
        fp = metrics["aw_fp"]
        provenance = fp["provenance"]
        lines.append(
            f"| {name} | {fp['total_aw_fp']} | {fp['zero_doc_aw_fp']} "
            f"(/{fp['zero_docs']} docs) | "
            f"{provenance.get('legacy', 0)} / {provenance.get('judge', 0)} / "
            f"{provenance.get('gate', 0)} | {fp['signatures']} |"
        )
    lines += ["", "## Zero-relation docs: predicted AW FPs (per-arm detail)", ""]
    for name, metrics in arm_results.items():
        rows = metrics["aw_fp"]["rows"]
        if not rows:
            lines.append(f"**{name}**: none\n")
            continue
        lines.append(f"**{name}** ({len(rows)}):\n")
        lines.append(
            "| PMID | Subject → Object | Status | Provenance | Gate | Signatures | Evidence |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for row in sorted(rows, key=lambda item: (item["pmid"], item["subject"])):
            lines.append(
                f"| {row['pmid']} | {row['subject']} → {row['object']} | "
                f"{row['semantic_status']} | {row['provenance']} | {row['gate_status'] or '-'} | "
                f"{','.join(row['signatures']) or '-'} | {row['evidence']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arms", required=True,
        help="name=path pairs, comma separated, e.g. "
        "D=benchmark_output/x/agent_results_D.json",
    )
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_output")
    parser.add_argument("--run-id", default="error_table")
    args = parser.parse_args()

    gold_by_pmid = {str(item["pmid"]): item for item in load_jsonl(args.gold)}
    source_by_pmid = {str(item["pmid"]): item for item in load_jsonl(args.source)}

    arm_results: dict[str, dict] = {}
    for item in args.arms.split(","):
        name, _, path = item.partition("=")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records = payload.get("records", payload.get("history", []))
        if not records and isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list) and value and isinstance(value[0], dict) and "pmid" in value[0]:
                    records = value
                    break
        arm_results[name] = analyze_arm(records, gold_by_pmid, source_by_pmid)
        print(f"[ANALYZED] {name}: {len(records)} records", flush=True)

    out_dir = args.output_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "error_table.json").write_text(
        json.dumps(arm_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "error_table.md").write_text(markdown(arm_results), encoding="utf-8")
    print(markdown(arm_results))
    print(f"[DONE] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
