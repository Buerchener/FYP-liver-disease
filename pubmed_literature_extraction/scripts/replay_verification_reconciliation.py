#!/usr/bin/env python3
"""Offline replay of frozen Gold20 candidates through tiered-v2 reconciliation."""

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

from cognitive_agent.collaborative_extractor import CollaborativeConfig, CollaborativeExtractor
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.extraction_quality import prepare_extraction
from cognitive_agent.relation_contract import RelationCandidateProjector
from cognitive_agent.relation_pair_classifier import BioREDPairClassifier, PairClassifierConfig
from cognitive_agent.schema.predicate_cards import load_predicate_thresholds
from cognitive_agent.verifier import KGVerifier
from scripts.evaluate_gold200_unified import (
    GOLD_VIEWS,
    SOURCE_PATH,
    aliases,
    load_jsonl,
    relation_key,
    score_funnel,
    score_view,
)


class OfflineKG:
    is_connected = False


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def text_for(source: dict) -> str:
    return f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"


def _surface_key(relation: dict) -> tuple[str, ...]:
    """Match a legacy review row back to its projected extracted hint."""
    return tuple(
        str(relation.get(field, "") or "").strip().casefold()
        for field in ("subject", "subject_type", "predicate", "object", "object_type")
    )


def _legacy_replay_candidates(record: dict, projected: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Restore candidate identity and old structured decisions without inventing claims.

    The 2026-08-29 cache predates EvidencePack span IDs.  Its collaboration
    phase nevertheless retained the reviewed relation, explicit reason code,
    confidence, and critic approval.  We overlay those review rows onto the
    corresponding extracted hints, then bind the decision to newly source-
    aligned span IDs in a second verification pass.
    """
    collaboration = (record.get("phases", {}) or {}).get("collaboration", {}) or {}
    review_candidates = list(collaboration.get("review_candidates", []) or [])
    decisions = {
        str(item.get("candidate_id", "") or ""): copy.deepcopy(item)
        for item in collaboration.get("review_decisions", []) or []
        if str(item.get("candidate_id", "") or "")
    }
    by_key = {_surface_key(item): copy.deepcopy(item) for item in review_candidates}
    candidates: list[dict] = []
    used_review_ids: set[str] = set()
    for item in projected:
        relation = copy.deepcopy(item)
        reviewed = by_key.get(_surface_key(item))
        if reviewed:
            relation.update(copy.deepcopy(reviewed))
            relation["provenance"] = sorted(set(
                list(item.get("provenance", []) or [])
                + list(reviewed.get("provenance", []) or [])
                + ["legacy_collaboration_replay"]
            ))
            used_review_ids.add(str(reviewed.get("candidate_id", "") or ""))
        candidates.append(relation)
    # Recovery candidates and edited review candidates need not occur in the
    # extraction projection.  Retain them as audit candidates rather than
    # silently dropping a model-recovered lineage.
    for reviewed in review_candidates:
        candidate_id = str(reviewed.get("candidate_id", "") or "")
        if candidate_id in used_review_ids:
            continue
        relation = copy.deepcopy(reviewed)
        relation["provenance"] = sorted(set(
            list(relation.get("provenance", []) or [])
            + ["legacy_collaboration_replay"]
        ))
        candidates.append(relation)
    return candidates, decisions


def _bind_legacy_adjudications(
    relations: list[dict], decisions: dict[str, dict], *, model_id: str,
) -> tuple[list[dict], list[dict]]:
    """Bind cached decisions to verified source spans for monotonic replay."""
    output: list[dict] = []
    migration_audit: list[dict] = []
    for item in relations:
        relation = copy.deepcopy(item)
        candidate_id = str(relation.get("candidate_id", "") or "")
        decision = decisions.get(candidate_id)
        if not decision:
            output.append(relation)
            continue
        action = str(decision.get("action", "") or "").upper()
        reason_code = str(decision.get("reason_code", "") or "").upper()
        supported = action in {"KEEP", "ACCEPT"} and reason_code == "EXPLICIT_DIRECT_RELATION"
        spans = list((relation.get("evidence_pack", {}) or {}).get("spans", []) or [])
        supporting_span_ids = [
            str(span.get("span_id", "") or "") for span in spans if span.get("span_id")
        ] if supported else []
        verdict = "SUPPORTED" if supported else "NOT_SUPPORTED"
        adjudication = {
            **copy.deepcopy(decision),
            "verdict": verdict,
            "reason_code": reason_code,
            "supporting_span_ids": supporting_span_ids,
            "model_id": model_id,
            "migration": "legacy_evidence_unit_to_evidence_pack_v1",
        }
        relation.update({
            "adjudication_verdict": verdict,
            "adjudication_reason_code": reason_code,
            "adjudication_confidence": float(decision.get("confidence", 0.0) or 0.0),
            "supporting_span_ids": supporting_span_ids,
            "adjudication_model_id": model_id,
            "adjudication": adjudication,
        })
        if supported:
            relation.setdefault("quality_flags", []).append("adjudicator_entailed")
        if supported and bool(decision.get("critic_approved", False)):
            relation.setdefault("quality_flags", []).extend([
                "qwen_critic_approved", "dual_model_entailed",
            ])
        relation["quality_flags"] = sorted(set(relation.get("quality_flags", []) or []))
        migration_audit.append({
            "candidate_id": candidate_id,
            "action": action,
            "verdict": verdict,
            "reason_code": reason_code,
            "supporting_span_ids": supporting_span_ids,
            "source_traceable": bool((relation.get("evidence_pack", {}) or {}).get("source_traceable")),
        })
        output.append(relation)
    return output, migration_audit


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _restore_current_lineage(
    relations: list[dict], projected: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Migrate old pair-overwritten IDs without changing relation content."""
    projected_by_key: dict[tuple[str, ...], list[dict]] = {}
    for item in projected:
        projected_by_key.setdefault(_surface_key(item), []).append(item)
    output: list[dict] = []
    audit: list[dict] = []
    for item in relations:
        relation = copy.deepcopy(item)
        if str(relation.get("candidate_lane", "extracted_hint")) == "recovery":
            output.append(relation)
            continue
        matches = projected_by_key.get(_surface_key(relation), [])
        if not matches:
            output.append(relation)
            continue
        source = matches[0]
        source_id = str(source.get("candidate_id", "") or "")
        old_id = str(relation.get("candidate_id", "") or "")
        if source_id and source_id != old_id:
            relation["candidate_id"] = source_id
            relation["pair_candidate_id"] = str(
                relation.get("pair_candidate_id", "")
                or (old_id if old_id.startswith("p-") else "")
            )
            relation["source_candidate_ids"] = sorted(set([
                source_id,
                *(str(value) for value in relation.get("source_candidate_ids", []) or []),
            ]))
            audit.append({
                "old_candidate_id": old_id,
                "candidate_id": source_id,
                "pair_candidate_id": relation.get("pair_candidate_id", ""),
                "reason": "pair_id_restored_to_projection_lineage",
            })
        output.append(relation)
    return output, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predicate-thresholds", type=Path)
    args = parser.parse_args()

    prior = args.prior_run.resolve()
    sample = json.loads((prior / "sample_manifest.json").read_text(encoding="utf-8"))
    pmids = {str(item["pmid"]) for item in sample["selection"]}
    source = {
        str(item["pmid"]): item for item in load_jsonl(SOURCE_PATH)
        if str(item["pmid"]) in pmids
    }
    gold_path = GOLD_VIEWS["candidate_semantic"]
    gold = {
        str(item["pmid"]): item for item in load_jsonl(gold_path)
        if str(item["pmid"]) in pmids
    }
    checksum_before = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    thresholds = load_predicate_thresholds(args.predicate_thresholds) if args.predicate_thresholds else {}
    verifier = KGVerifier(
        OfflineKG(), verification_policy="tiered-v2", predicate_thresholds=thresholds,
    )
    finalizer = CollaborativeExtractor(CollaborativeConfig(verification_policy="tiered-v2"))
    projector = RelationCandidateProjector()
    pair_classifier = BioREDPairClassifier(PairClassifierConfig(
        enabled=True, mode="active", backend="deterministic",
        max_candidates=128, max_recovery_candidates=24,
        max_recovery_per_owner_sentence=6,
    ))
    reader = ArticleEvidenceReader()
    report = {
        "policy": verifier.policy.contract_version,
        "predicate_thresholds": str(args.predicate_thresholds.resolve()) if args.predicate_thresholds else None,
        "arms": {},
        "frozen_gold": str(gold_path),
    }

    for arm in ("A_one_shot", "B_semantic_chunk"):
        records = json.loads((prior / "arms" / arm / "records.json").read_text(encoding="utf-8"))["records"]
        raw_fake = []
        replay_fake = []
        audits = []
        pair_cap_violations = []
        hard_rejected_gold = []
        migration_audits = []
        source_hashes = []
        extraction_hashes = []
        projection_hashes = []
        context_only_duplicates = 0
        self_contained_cross_sentence = 0
        self_contained_stale_linkage = 0
        pmid_41948344 = []
        projected_hint_lineages = 0
        retained_hint_lineages = 0
        for record in records:
            pmid = str(record["pmid"])
            text = text_for(source[pmid])
            extraction = record["phases"]["extraction"]
            entities = copy.deepcopy(extraction.get("entities", []) or [])
            raw_relations = copy.deepcopy(extraction.get("relations", []) or [])
            source_hashes.append((pmid, hashlib.sha256(text.encode("utf-8")).hexdigest()))
            extraction_hashes.append((pmid, _json_hash(raw_relations)))
            raw_fake.append({"pmid": pmid, "phases": {"verification": {"relations": raw_relations}}})
            projection = projector.project(entities, raw_relations, text=text)
            projection_hashes.append((pmid, _json_hash(projection.relations)))
            prepared = prepare_extraction(entities, projection.relations, text=text)
            pair_result = pair_classifier.classify(
                entities=prepared.entities,
                relations=prepared.relations,
                units=reader.read(text),
                source_text=text,
            )
            hinted_pairs = {
                (
                    item.subject.casefold(), item.subject_type,
                    item.object.casefold(), item.object_type,
                )
                for item in pair_result.candidates if item.source_predicates
            }
            if len(pair_result.candidates) > len(hinted_pairs) + 24:
                pair_cap_violations.append({
                    "pmid": pmid,
                    "candidate_count": len(pair_result.candidates),
                    "hint_pair_count": len(hinted_pairs),
                })
            existing_verified = list(
                (record.get("phases", {}).get("verification", {}) or {}).get("relations", []) or []
            )
            if existing_verified:
                adjudicated, lineage_audit = _restore_current_lineage(
                    existing_verified, projection.relations,
                )
                migration_audit = [{"lineage": item} for item in lineage_audit]
            else:
                replay_candidates, decisions = _legacy_replay_candidates(record, projection.relations)
                # First pass creates exact source-aligned EvidencePack span IDs.
                aligned = verifier.verify(entities, replay_candidates, pmid=pmid, text=text)
                collaboration = record.get("phases", {}).get("collaboration", {}) or {}
                adjudicated, migration_audit = _bind_legacy_adjudications(
                    aligned.to_dict().get("relations", []),
                    decisions,
                    model_id=str(collaboration.get("model_id", "") or "legacy-cached-adjudicator"),
                )
            checked = verifier.verify(entities, adjudicated, pmid=pmid, text=text)
            finalized, audit = finalizer.finalize_after_reverification(
                adjudicated, checked.to_dict(), source_text=text,
            )
            final_checked = verifier.verify(entities, finalized, pmid=pmid, text=text)
            final_relations = final_checked.to_dict().get("relations", [])
            projected_ids = {
                str(item.get("candidate_id", "") or "")
                for item in projection.relations
                if str(item.get("candidate_id", "") or "")
            }
            retained_ids = {
                candidate_id
                for relation in final_relations
                for candidate_id in {
                    str(relation.get("candidate_id", "") or ""),
                    *(str(value) for value in relation.get("merged_candidate_ids", []) or []),
                    *(str(value) for value in relation.get("source_candidate_ids", []) or []),
                    *(
                        str(instance.get("candidate_id", "") or "")
                        for instance in relation.get("claim_instances", []) or []
                        if isinstance(instance, dict)
                    ),
                }
                if candidate_id
            }
            projected_hint_lineages += len(projected_ids)
            retained_hint_lineages += len(projected_ids & retained_ids)
            for relation in final_relations:
                pack = relation.get("evidence_pack", {}) or {}
                if (
                    pack.get("support_mode") == "SELF_CONTAINED"
                    and "cross_sentence" in set(relation.get("quality_flags", []) or [])
                ):
                    self_contained_cross_sentence += 1
                if (
                    pack.get("support_mode") == "SELF_CONTAINED"
                    and "trigger_not_linking_endpoints" in set(relation.get("quality_flags", []) or [])
                ):
                    self_contained_stale_linkage += 1
                if str(relation.get("evidence_role", "")).upper() == "CONTEXT":
                    context_only_duplicates += 1
                if (
                    pmid == "41948344"
                    and relation.get("subject") == "CCND1"
                    and relation.get("predicate") == "EXPRESSED_IN"
                    and relation.get("object") in {
                        "endothelial cells", "epithelial cells", "hepatocytes", "macrophages",
                    }
                ):
                    pmid_41948344.append({
                        "object": relation.get("object"),
                        "candidate_id": relation.get("candidate_id"),
                        "support_mode": pack.get("support_mode"),
                        "support_sentence_ids": pack.get("support_sentence_ids", []),
                        "minimal_support_span_ids": pack.get("minimal_support_span_ids", []),
                        "semantic_status": relation.get("semantic_status"),
                        "write_status": relation.get("write_status"),
                        "quality_flags": relation.get("quality_flags", []),
                    })
            replay_fake.append({
                "pmid": pmid,
                "phases": {
                    "extraction": extraction,
                    "relation_candidate_projection": {"relations": adjudicated},
                    "relation_core_selection": {"production_relations": adjudicated},
                    "verification": final_checked.to_dict(),
                },
            })
            audits.append({
                "pmid": pmid,
                **audit,
                "final_relations": final_checked.to_dict().get("relations", []),
            })
            migration_audits.append({"pmid": pmid, "relations": migration_audit})
            alias_map = aliases(gold[pmid], text)
            gold_keys = {relation_key(item, alias_map) for item in gold[pmid].get("relations", []) or []}
            for relation in final_checked.to_dict().get("relations", []):
                if (
                    relation_key(relation, alias_map) in gold_keys
                    and str(relation.get("factual_status", "")).upper() == "REJECTED"
                ):
                    hard_rejected_gold.append({
                        "pmid": pmid,
                        "candidate_id": relation.get("candidate_id", ""),
                        "quality_flags": relation.get("quality_flags", []),
                    })
        raw_score = score_view(raw_fake, gold, source, "candidate_semantic", include_review=True)
        accepted_score = score_view(replay_fake, gold, source, "candidate_semantic", include_review=False)
        replay_score = score_view(replay_fake, gold, source, "candidate_semantic", include_review=True)
        funnel = score_funnel(replay_fake, gold, source)
        accepted_micro = accepted_score["typed_directed_triple_micro"]
        factual_micro = funnel["factual_valid"]
        accepted_precision = accepted_micro.get("precision")
        factual_precision = factual_micro.get("precision")
        accepted_not_worse = bool(
            accepted_precision is not None
            and (factual_precision is None or accepted_precision >= factual_precision)
        )
        report["arms"][arm] = {
            "raw": raw_score,
            "semantic_accepted": accepted_score,
            "accepted_review": replay_score,
            "funnel": funnel,
            "acceptance_precision_not_below_factual": accepted_not_worse,
            "post_action_rollback": sum(int(item.get("rolled_back_count", 0) or 0) for item in audits),
            "soft_flag_hard_reject": sum(int(item.get("soft_flag_hard_reject_count", 0) or 0) for item in audits),
            "hard_rejected_gold": hard_rejected_gold,
            "pair_cap_violations": pair_cap_violations,
            "article_audits": audits,
            "adjudication_migration_audits": migration_audits,
            "source_manifest_hash": _json_hash(source_hashes),
            "raw_extraction_manifest_hash": _json_hash(extraction_hashes),
            "candidate_projection_manifest_hash": _json_hash(projection_hashes),
            "context_only_duplicate_count": context_only_duplicates,
            "self_contained_cross_sentence_count": self_contained_cross_sentence,
            "self_contained_stale_linkage_count": self_contained_stale_linkage,
            "pmid_41948344": pmid_41948344,
            "structural_hint_lineage_survival": (
                retained_hint_lineages / projected_hint_lineages
                if projected_hint_lineages else None
            ),
            "structural_hint_lineages": {
                "retained": retained_hint_lineages,
                "projected": projected_hint_lineages,
            },
        }
    checksum_after = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    report["frozen_gold_sha256_before"] = checksum_before
    report["frozen_gold_sha256_after"] = checksum_after
    report["frozen_gold_unchanged"] = checksum_before == checksum_after
    report["passed"] = all(
        arm["post_action_rollback"] == 0
        and arm["soft_flag_hard_reject"] == 0
        and not arm["hard_rejected_gold"]
        and not arm["pair_cap_violations"]
        and arm["context_only_duplicate_count"] == 0
        and arm["self_contained_cross_sentence_count"] == 0
        and arm["self_contained_stale_linkage_count"] == 0
        and (
            arm["semantic_accepted"]["typed_directed_triple_micro"]["tp"]
            + arm["semantic_accepted"]["typed_directed_triple_micro"]["fp"]
        ) > 0
        and arm["acceptance_precision_not_below_factual"]
        and (
            arm["funnel"]["derived"]["raw_canonical_tp_survival"] is None
            or arm["funnel"]["derived"]["raw_canonical_tp_survival"] <= 1.0
        )
        and arm["funnel"]["accepted_review"]["tp"]
            == arm["funnel"]["raw_hint"]["tp"]
        and arm["structural_hint_lineage_survival"] == 1.0
        for arm in report["arms"].values()
    ) and report["frozen_gold_unchanged"]
    atomic_json(args.output.resolve(), report)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "passed": report["passed"],
        "arms": {
            name: {
                "raw": item["raw"]["typed_directed_triple_micro"],
                "semantic_accepted": item["semantic_accepted"]["typed_directed_triple_micro"],
                "accepted_review": item["accepted_review"]["typed_directed_triple_micro"],
                "raw_tp_survival": item["funnel"]["derived"]["raw_canonical_tp_survival"],
                "pair_cap_violations": len(item["pair_cap_violations"]),
            }
            for name, item in report["arms"].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
