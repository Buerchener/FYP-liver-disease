#!/usr/bin/env python3
"""Offline P0 integration replay for completed Sentinel item records.

The replay never calls a provider and never writes Neo4j.  It rebuilds the
protected hint lane from stable candidate IDs, reapplies the persisted claim
gate, then runs the current verifier/reconciliation/finalizer contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.agent import (
    bind_pairwise_gate_to_hints,
    build_candidate_audit_ledger,
)
from cognitive_agent.collaborative_extractor import CollaborativeExtractor
from cognitive_agent.verifier import KGVerifier
from scripts.evaluate_gold200_unified import load_jsonl, score_funnel, score_view


class OfflineKG:
    is_connected = False


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    prior_items = args.prior_run.resolve() / "arms/B_tiered_cold/items"
    item_paths = sorted(prior_items.glob("*.json"))
    if not item_paths:
        raise SystemExit(f"no prior Sentinel items found under {prior_items}")
    source_path = ROOT / "extraction_output/pubmed_converted_500.jsonl"
    gold_path = ROOT / "gold_annotations/pubmed_200_gold_v2_candidate_view.jsonl"
    checksum_before = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    source_all = {str(item["pmid"]): item for item in load_jsonl(source_path)}
    gold_all = {str(item["pmid"]): item for item in load_jsonl(gold_path)}

    verifier = KGVerifier(OfflineKG(), verification_policy="tiered-v2")
    finalizer = CollaborativeExtractor()
    replayed: list[dict] = []
    for path in item_paths:
        original = json.loads(path.read_text(encoding="utf-8"))
        pmid = str(original.get("pmid", ""))
        phases = original.get("phases", {}) or {}
        extraction = phases.get("extraction", {}) or {}
        projected = list(
            (phases.get("relation_candidate_projection", {}) or {}).get(
                "relations", []
            ) or []
        )
        pair = phases.get("relation_pair_classification", {}) or {}
        gated_hints = bind_pairwise_gate_to_hints(
            projected,
            pair_candidates=list(pair.get("candidates", []) or []),
            pair_predictions=list(pair.get("predictions", []) or []),
            gate_table=list(
                (phases.get("pairwise_judge", {}) or {}).get("gate_table", []) or []
            ),
        )
        old_core = list(
            (phases.get("relation_core_selection", {}) or {}).get("relations", []) or []
        )
        recovery = [
            copy.deepcopy(item) for item in old_core
            if str(item.get("candidate_lane", "extracted_hint")) == "recovery"
        ]
        prior_verified_by_key = {
            (
                str(item.get("candidate_id", "") or ""),
                max(1, int(item.get("candidate_version", 1) or 1)),
            ): item
            for item in (phases.get("verification", {}) or {}).get("relations", []) or []
            if str(item.get("candidate_id", "") or "")
        }
        # Preserve model declarations/provenance, but never carry forward
        # derived verifier flags or statuses from the prior policy version.
        model_fields = {
            "adjudication_verdict", "adjudication_reason_code",
            "adjudication_confidence", "supporting_span_ids", "adjudication",
            "adjudication_model_id", "pairwise_gate", "pair_candidate_id",
        }
        for relation in [*gated_hints, *recovery]:
            key = (
                str(relation.get("candidate_id", "") or ""),
                max(1, int(relation.get("candidate_version", 1) or 1)),
            )
            previous = prior_verified_by_key.get(key, {})
            for field in model_fields:
                if previous.get(field) not in (None, "", [], {}):
                    relation[field] = copy.deepcopy(previous[field])
            preserved_model_flags = set(previous.get("quality_flags", []) or []) & {
                "adjudicator_entailed", "qwen_critic_approved",
                "dual_model_entailed", "critic_approved",
            }
            relation["quality_flags"] = sorted(set(
                relation.get("quality_flags", []) or []
            ) | preserved_model_flags)
        core = [*gated_hints, *recovery]
        source = source_all[pmid]
        text = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"
        initial = verifier.verify(
            list(extraction.get("entities", []) or []), core, pmid=pmid, text=text,
        )
        finalized, pass_audit = finalizer.finalize_after_reverification(
            core, initial.to_dict(), source_text=text,
        )
        final = verifier.verify(
            list(extraction.get("entities", []) or []), finalized,
            pmid=pmid, text=text,
        )
        finalization_audit = {"passes": [{**pass_audit, "pass": 1}]}
        ledger = build_candidate_audit_ledger(
            projected_relations=projected,
            core_relations=core,
            final_relations=[item.to_dict() for item in final.relations],
            finalization_audit=finalization_audit,
            verified_candidates=[item.to_dict() for item in initial.relations],
            pair_candidates=list(pair.get("candidates", []) or []),
            pair_predictions=list(pair.get("predictions", []) or []),
            gate_table=list(
                (phases.get("pairwise_judge", {}) or {}).get("gate_table", []) or []
            ),
        )
        replayed.append({
            "pmid": pmid,
            "phases": {
                "extraction": extraction,
                "relation_candidate_projection": {"relations": projected},
                "relation_core_selection": {"relations": core},
                "verification": final.to_dict(),
                "candidate_audit_ledger": ledger,
                "collaboration": {"post_action_finalization": {
                    **finalization_audit,
                    "rolled_back_count": 0,
                    "soft_flag_hard_reject_count": 0,
                }},
            },
        })

    pmids = {str(item["pmid"]) for item in replayed}
    gold = {pmid: gold_all[pmid] for pmid in pmids}
    source = {pmid: source_all[pmid] for pmid in pmids}
    funnel = score_funnel(replayed, gold, source)
    rows = [
        relation
        for record in replayed
        for relation in record["phases"]["verification"]["relations"]
    ]
    ledgers = [record["phases"]["candidate_audit_ledger"] for record in replayed]
    projected_count = sum(
        item["summary"]["projected_hint_count"] for item in ledgers
    )
    accounted_count = sum(
        item["summary"]["accounted_projected_hint_count"] for item in ledgers
    )
    eligible_count = sum(
        item["summary"]["eligible_projected_hint_count"] for item in ledgers
    )
    survived_count = sum(
        item["summary"]["eligible_survived_count"] for item in ledgers
    )
    all_ledger_rows = [
        row for ledger in ledgers for row in ledger.get("rows", []) or []
    ]
    stale = sum(
        str((item.get("evidence_pack", {}) or {}).get("support_mode", "")).upper()
        == "SELF_CONTAINED"
        and bool({
            "cross_sentence", "coreference_only_support", "multi_span_support",
            "trigger_not_linking_endpoints",
        } & set(item.get("quality_flags", []) or []))
        for item in rows
    )
    report = {
        "policy": verifier.policy.contract_version,
        "mode": "offline_no_provider_no_neo4j",
        "source_run": str(args.prior_run.resolve()),
        "record_count": len(replayed),
        "metrics": score_view(replayed, gold, source, "candidate_semantic"),
        "stage_funnel": funnel,
        "invariants": {
            "audit_lineage_accounting": (
                accounted_count / projected_count if projected_count else None
            ),
            "eligible_lineage_survival": (
                survived_count / eligible_count if eligible_count else None
            ),
            "background_auto_accept": sum(
                str(item.get("claim_role", "CURRENT_FINDING")).upper()
                in {"BACKGROUND", "METHOD", "PREDICTION", "PRIOR_WORK", "SPECULATIVE", "OTHER"}
                and str(item.get("semantic_status", "")).upper() == "ACCEPTED"
                for item in rows
            ),
            "filtered_endpoint_auto_accept": sum(
                bool({"filtered_endpoint", "subject_filtered", "object_filtered"}
                     & set(item.get("quality_flags", []) or []))
                and str(item.get("semantic_status", "")).upper() == "ACCEPTED"
                for item in rows
            ),
            "invalid_promotion_path_accept": sum(
                str(item.get("semantic_status", "")).upper() == "ACCEPTED"
                and str(item.get("promotion_path", "")) not in {"DETERMINISTIC", "ADJUDICATED"}
                for item in rows
            ),
            "version_accounting": (
                sum(item.get("disposition") != "UNACCOUNTED" for item in all_ledger_rows)
                / len(all_ledger_rows) if all_ledger_rows else 1.0
            ),
            "recovery_lineage_accounting": (
                sum(item.get("disposition") != "UNACCOUNTED" for item in all_ledger_rows
                    if item.get("candidate_lane") == "recovery")
                / sum(item.get("candidate_lane") == "recovery" for item in all_ledger_rows)
                if any(item.get("candidate_lane") == "recovery" for item in all_ledger_rows)
                else 1.0
            ),
            "r_prefix_lane_mismatch": sum(
                str(item.get("candidate_id", "")).startswith("r-")
                and item.get("candidate_lane") != "recovery"
                for item in all_ledger_rows
            ),
            "lineage_binding_missing": sum(
                "lineage_binding_missing" in set(item.get("quality_flags", []) or [])
                for item in rows
            ),
            "self_contained_stale_flags": stale,
            "accepted_review_canonical_tp_not_below_raw": (
                funnel["accepted_review"]["tp"] >= funnel["raw_hint"]["tp"]
            ),
            "semantic_precision_not_below_factual": (
                funnel["semantic_accepted"]["precision"] is not None
                and funnel["factual_valid"]["precision"] is not None
                and funnel["semantic_accepted"]["precision"]
                >= funnel["factual_valid"]["precision"]
            ),
            "post_action_rollback": 0,
            "soft_flag_hard_reject": 0,
            "provider_failures": 0,
            "neo4j_mutations": 0,
            "frozen_gold_checksum_unchanged": (
                checksum_before == hashlib.sha256(gold_path.read_bytes()).hexdigest()
            ),
        },
        "records": replayed,
    }
    report["evaluation_valid"] = all([
        report["invariants"]["audit_lineage_accounting"] == 1,
        report["invariants"]["eligible_lineage_survival"] == 1,
        report["invariants"]["background_auto_accept"] == 0,
        report["invariants"]["filtered_endpoint_auto_accept"] == 0,
        report["invariants"]["invalid_promotion_path_accept"] == 0,
        report["invariants"]["version_accounting"] == 1,
        report["invariants"]["recovery_lineage_accounting"] == 1,
        report["invariants"]["r_prefix_lane_mismatch"] == 0,
        report["invariants"]["lineage_binding_missing"] == 0,
        report["invariants"]["self_contained_stale_flags"] == 0,
        report["invariants"]["accepted_review_canonical_tp_not_below_raw"],
        report["invariants"]["semantic_precision_not_below_factual"],
        report["invariants"]["frozen_gold_checksum_unchanged"],
    ])
    atomic_json(args.output.resolve(), report)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "evaluation_valid": report["evaluation_valid"],
        "invariants": report["invariants"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["evaluation_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
