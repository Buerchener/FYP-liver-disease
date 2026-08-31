#!/usr/bin/env python3
"""Offline BioRED semantic-routing replay over frozen candidate/model records.

Dev relations are removed from documents before evidence/reconciliation work and
are read only after predictions are frozen for post-hoc scoring.  No provider,
cache, Neo4j, or source experiment record is mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from cognitive_agent.biored_adapter import (
    apply_native_label_adjudication,
    enrich_native_relation,
    parse_biored_pubtator,
    registry_from_training,
)


def relation_key(item: dict[str, Any]) -> tuple[str, str, str]:
    left, right = sorted((str(item.get("arg1_id", "")), str(item.get("arg2_id", ""))))
    return left, right, str(item.get("relation_type", ""))


def score(
    records: list[dict[str, Any]], documents: list[Any], *, field: str,
) -> dict[str, Any]:
    by_pmid = {str(item["pmid"]): item for item in records}
    tp = fp = fn = 0
    for document in documents:
        gold = {item.label_key for item in document.relations}
        predicted = {
            relation_key(item)
            for item in by_pmid.get(document.pmid, {}).get(field, []) or []
        }
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and precision + recall else 0.0
    )
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pubtator", type=Path, required=True)
    parser.add_argument("--training-pubtator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.report.read_text(encoding="utf-8"))
    documents = parse_biored_pubtator(args.pubtator)
    by_pmid = {item.pmid: item for item in documents}
    registry, registry_manifest = registry_from_training(args.training_pubtator)
    replay_records: list[dict[str, Any]] = []
    transition_ledger: list[dict[str, Any]] = []

    for source_record in source.get("records", []) or []:
        pmid = str(source_record.get("pmid", ""))
        scoring_document = by_pmid[pmid]
        routing_document = replace(scoring_document, relations=[])
        replayed: list[dict[str, Any]] = []
        for frozen in source_record.get("factual_relations", []) or []:
            lane = str(frozen.get("candidate_lane", "recovery") or "recovery")
            candidate = enrich_native_relation(
                dict(frozen), routing_document, registry, lane=lane,
            )
            adjudication = dict(frozen.get("label_adjudication", {}) or {})
            judge = dict(adjudication.get("judge", {}) or {})
            critic = dict(adjudication.get("critic", {}) or {})
            output, audit = apply_native_label_adjudication(
                [candidate], routing_document, registry,
                [judge] if judge else [], [critic] if critic else [],
                require_critic=True, allow_model_label_edits=False,
            )
            replayed.append(output[0])
            transition_ledger.append({
                "pmid": pmid,
                "candidate_id": candidate.get("candidate_id", ""),
                "arg1_id": candidate.get("arg1_id", ""),
                "arg2_id": candidate.get("arg2_id", ""),
                "candidate_version_before": candidate.get("candidate_version", 1),
                "semantic_status_before": frozen.get("semantic_status", "REVIEW"),
                "semantic_status_after": output[0].get("semantic_status", "REVIEW"),
                "relation_type_before": frozen.get("relation_type", ""),
                "relation_type_after": output[0].get("relation_type", ""),
                "audit": audit[0] if audit else {},
            })

        by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in replayed:
            key = relation_key(item)
            previous = by_key.get(key)
            if previous is None or int(item.get("candidate_version", 1) or 1) > int(
                previous.get("candidate_version", 1) or 1
            ):
                by_key[key] = item
        factual = list(by_key.values())
        semantic = [item for item in factual if item.get("semantic_status") == "ACCEPTED"]
        replay_records.append({
            "pmid": pmid, "factual_relations": factual,
            "framework_relations": semantic,
        })

    factual_metrics = score(replay_records, documents, field="factual_relations")
    semantic_metrics = score(replay_records, documents, field="framework_relations")
    old_semantic = dict(
        source.get("framework_metrics", source.get("metrics", {})).get(
            "pair_relation_micro", {}
        ) or {}
    )
    for item in transition_ledger:
        document = by_pmid[item["pmid"]]
        gold = {relation.label_key for relation in document.relations}
        after_key = (*sorted((item["arg1_id"], item["arg2_id"])), item["relation_type_after"])
        item["posthoc_official_exact_status_after"] = "TP" if after_key in gold else "FP"
    promotions = [
        item for item in transition_ledger
        if item["semantic_status_before"] != "ACCEPTED"
        and item["semantic_status_after"] == "ACCEPTED"
    ]
    promotion_tp = sum(
        item["posthoc_official_exact_status_after"] == "TP" for item in promotions
    )
    promotion_fp = len(promotions) - promotion_tp
    report = {
        "protocol": "biored-semantic-routing-offline-replay-v2",
        "source_report": str(args.report),
        "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "pubtator_sha256": hashlib.sha256(args.pubtator.read_bytes()).hexdigest(),
        "training_manifest": registry_manifest,
        "dev_gold_used_during_routing": False,
        "remote_calls": 0,
        "neo4j_mutations": 0,
        "factual_metrics": factual_metrics,
        "semantic_metrics": semantic_metrics,
        "source_semantic_metrics": old_semantic,
        "semantic_delta": {
            key: semantic_metrics.get(key, 0) - old_semantic.get(key, 0)
            for key in ("tp", "fp", "fn", "f1")
            if isinstance(old_semantic.get(key, 0), (int, float))
        },
        "transition_counts": {
            "review_to_accepted": sum(
                item["semantic_status_before"] != "ACCEPTED"
                and item["semantic_status_after"] == "ACCEPTED"
                for item in transition_ledger
            ),
            "accepted_to_review": sum(
                item["semantic_status_before"] == "ACCEPTED"
                and item["semantic_status_after"] != "ACCEPTED"
                for item in transition_ledger
            ),
            "label_edits": sum(
                item["relation_type_before"] != item["relation_type_after"]
                for item in transition_ledger
            ),
            "promoted_tp": promotion_tp,
            "promoted_fp": promotion_fp,
            "promotion_precision": (
                promotion_tp / len(promotions) if promotions else None
            ),
        },
        "records": replay_records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    with (args.output_dir / "transition_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for item in transition_ledger:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.output_dir / "report.md").write_text(
        "# BioRED semantic routing offline replay\n\n"
        f"- Source semantic: `{json.dumps(old_semantic, sort_keys=True)}`\n"
        f"- Replayed semantic: `{json.dumps(semantic_metrics, sort_keys=True)}`\n"
        f"- Delta: `{json.dumps(report['semantic_delta'], sort_keys=True)}`\n"
        f"- Transitions: `{json.dumps(report['transition_counts'], sort_keys=True)}`\n"
        "- Remote calls: `0`; Neo4j mutations: `0`; Dev Gold used during routing: `false`\n",
        encoding="utf-8",
    )
    print(args.output_dir / "report.json")


if __name__ == "__main__":
    main()
