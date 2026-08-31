#!/usr/bin/env python3
"""Run BioRED's given-entity relation benchmark with an auditable LLM cascade.

This is deliberately separate from the LiverKG ontology and Neo4j pipeline.
The official BioRED concept IDs are supplied to the model (the given-entity
task); only BioRED relation labels and novelty are predicted.  Positive-class
metrics exclude NO_RELATION, as in common relation-extraction reporting.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.aux_model_registry import AuxModelRegistry, AuxModelSpec, StructuredModelResult
from cognitive_agent.extraction_cache import LightweightExtractionCache
from cognitive_agent.provider_errors import (
    AUTH_ERROR, QUOTA_OR_TOKEN_EXHAUSTED, classify_provider_error,
)
from liverkg_cli.config import load_config
from liverkg_cli.env import load_project_env
from liverkg_cli.security import get_secret
from cognitive_agent.biored_adapter import (
    apply_native_label_adjudication,
    apply_train_signature_prior_guard,
    audit_false_negatives,
    build_native_label_review_items,
    build_native_pair_candidates,
    enrich_native_relation,
    guard_biored_test,
    parse_biored_pubtator,
    registry_from_training,
    seed_native_lineage,
    stable_dev_partition,
    validate_native_predictions,
)

RELATION_TYPES = (
    "Association", "Bind", "Comparison", "Conversion", "Cotreatment",
    "Drug_Interaction", "Negative_Correlation", "Positive_Correlation",
)
NOVELTY = ("Novel", "No", "Unknown")


@dataclass
class Concept:
    concept_id: str
    entity_type: str
    mentions: list[str] = field(default_factory=list)


@dataclass
class Relation:
    left: str
    right: str
    relation_type: str
    novelty: str

    @property
    def pair(self) -> tuple[str, str]:
        return tuple(sorted((self.left, self.right)))

    @property
    def label_key(self) -> tuple[str, str, str]:
        return (*self.pair, self.relation_type)

    @property
    def novelty_key(self) -> tuple[str, str, str, str]:
        return (*self.pair, self.relation_type, self.novelty)


@dataclass
class Document:
    pmid: str
    text: str
    concepts: dict[str, Concept]
    relations: list[Relation]


def parse_pubtator(path: Path) -> list[Document]:
    documents: list[Document] = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 2 or "|t|" not in lines[0] or "|a|" not in lines[1]:
            continue
        pmid, title = lines[0].split("|t|", 1)
        _, abstract = lines[1].split("|a|", 1)
        concepts: dict[str, Concept] = {}
        relations: list[Relation] = []
        for line in lines[2:]:
            fields = line.split("\t")
            if len(fields) == 6 and fields[1].isdigit():
                _, _, _, mention, entity_type, raw_ids = fields
                for concept_id in raw_ids.split(","):
                    concept = concepts.setdefault(concept_id, Concept(concept_id, entity_type))
                    if mention not in concept.mentions:
                        concept.mentions.append(mention)
            elif len(fields) == 5 and fields[1] in RELATION_TYPES:
                _, relation_type, left, right, novelty = fields
                relations.append(Relation(left, right, relation_type, novelty if novelty in NOVELTY else "Unknown"))
        documents.append(Document(pmid, f"{title}\n{abstract}", concepts, relations))
    return documents


def model_result_dict(result: StructuredModelResult) -> dict[str, Any]:
    return result.to_dict()


def _mention_texts(concept: Any) -> list[str]:
    return [
        str(getattr(item, "text", item) or "")
        for item in list(getattr(concept, "mentions", []) or [])
        if str(getattr(item, "text", item) or "")
    ]


def normalise_relations(
    payload: dict[str, Any], concepts: dict[str, Any], source_text: str = "",
    *, candidate_pairs: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    values = payload.get("relations", []) if isinstance(payload, dict) else []
    seen: set[tuple[str, str, str]] = set()
    clean: list[dict[str, Any]] = []
    for item in values if isinstance(values, list) else []:
        if not isinstance(item, dict):
            continue
        left, right = str(item.get("arg1_id", "")), str(item.get("arg2_id", ""))
        relation_type = str(item.get("relation_type", ""))
        if left not in concepts or right not in concepts or left == right or relation_type not in RELATION_TYPES:
            continue
        pair_candidate_id = str(item.get("pair_candidate_id", "") or "")
        pair_candidate = (candidate_pairs or {}).get(pair_candidate_id)
        if candidate_pairs is not None:
            if pair_candidate is None:
                # Models occasionally omit the stable ID.  Pair identity is
                # still unambiguous, so recover the ID by exact official IDs;
                # never use list position.
                exact = [
                    value for value in candidate_pairs.values()
                    if tuple(sorted((value["arg1_id"], value["arg2_id"])))
                    == tuple(sorted((left, right)))
                ]
                pair_candidate = exact[0] if len(exact) == 1 else None
            if pair_candidate is None:
                continue
            if tuple(sorted((left, right))) != tuple(sorted((
                pair_candidate["arg1_id"], pair_candidate["arg2_id"],
            ))):
                continue
            if relation_type not in set(pair_candidate.get("allowed_relation_types", []) or []):
                continue
            pair_candidate_id = str(pair_candidate["pair_candidate_id"])
        key = (*sorted((left, right)), relation_type)
        if key in seen:
            continue
        seen.add(key)
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        evidence_quote = str(item.get("evidence_quote", "") or "").strip()
        evidence_start = source_text.find(evidence_quote) if source_text and evidence_quote else -1
        endpoint_closed = bool(
            evidence_quote
            and any(mention.casefold() in evidence_quote.casefold() for mention in _mention_texts(concepts[left]))
            and any(mention.casefold() in evidence_quote.casefold() for mention in _mention_texts(concepts[right]))
        )
        role = str(item.get("claim_role", "OTHER") or "OTHER").upper()
        if role not in {
            "CURRENT_FINDING", "PRIOR_WORK", "BACKGROUND", "METHOD",
            "PREDICTION", "SPECULATIVE", "OTHER",
        }:
            role = "OTHER"
        clean.append({
            "arg1_id": left, "arg2_id": right, "relation_type": relation_type,
            "novelty": str(item.get("novelty", "Unknown")) if str(item.get("novelty", "Unknown")) in NOVELTY else "Unknown",
            "confidence": confidence,
            "evidence_quote": evidence_quote,
            "evidence_start": evidence_start,
            "evidence_end": evidence_start + len(evidence_quote) if evidence_start >= 0 else -1,
            "evidence_source_traceable": evidence_start >= 0,
            "evidence_endpoint_closed": endpoint_closed,
            "factual_status": "VALID" if evidence_start >= 0 else "REVIEW",
            "semantic_status": "ACCEPTED" if evidence_start >= 0 and endpoint_closed else "REVIEW",
            "claim_role": role,
            "pair_candidate_id": pair_candidate_id,
            "candidate_id": str((pair_candidate or {}).get("candidate_id", "") or ""),
            "candidate_version": 1,
            "candidate_lane": "recovery" if pair_candidate is not None else "extracted_hint",
            "source_lanes": ["recovery" if pair_candidate is not None else "extracted_hint"],
            "owner_sentence_ids": list((pair_candidate or {}).get("owner_sentence_ids", []) or []),
            "evidence_window": str((pair_candidate or {}).get("evidence_window", "") or ""),
            "write_status": "BLOCKED",
            "write_reasons": ["benchmark_no_write"],
        })
    if candidate_pairs is not None:
        best_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for item in clean:
            pair = tuple(sorted((item["arg1_id"], item["arg2_id"])))
            previous = best_by_pair.get(pair)
            if previous is None or (
                float(item.get("confidence", 0.0) or 0.0), item["relation_type"]
            ) > (
                float(previous.get("confidence", 0.0) or 0.0), previous["relation_type"]
            ):
                best_by_pair[pair] = item
        return list(best_by_pair.values())
    return clean


def call_cached(
    cache: LightweightExtractionCache, registry: AuxModelRegistry, role: str,
    system: str, user: str, schema: dict[str, Any],
) -> tuple[StructuredModelResult, str]:
    key_payload = {
        "harness": "biored-given-entity-v2", "role": role,
        "model": registry.specs[role].model_id, "system": system, "user": user, "schema": schema,
    }
    key = "biored:" + hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()

    def factory() -> tuple[dict[str, Any], float]:
        result = registry.call_json(role, system_prompt=system, user_prompt=user, schema_hint=schema)
        return model_result_dict(result), result.latency_s

    payload, status = cache.get_or_compute(key, factory, cacheable=lambda item: item.get("status") == "OK")
    fields = StructuredModelResult.__dataclass_fields__
    return StructuredModelResult(**{key: value for key, value in payload.items() if key in fields}), status


def normalise_label_decisions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    values = payload.get("decisions", []) if isinstance(payload, dict) else []
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in values if isinstance(values, list) else []:
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("candidate_id", "") or "")
        try:
            version = max(1, int(raw.get("candidate_version", 1) or 1))
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0)))
            relation_confidence = max(0.0, min(1.0, float(
                raw.get("relation_confidence", confidence) or 0.0
            )))
            label_confidence = max(0.0, min(1.0, float(
                raw.get("label_confidence", confidence) or 0.0
            )))
        except (TypeError, ValueError):
            continue
        key = (candidate_id, version)
        if not candidate_id or key in seen:
            continue
        seen.add(key)
        output.append({
            "candidate_id": candidate_id,
            "candidate_version": version,
            "verdict": str(raw.get("verdict", "ABSTAIN") or "ABSTAIN").upper(),
            "recommended_relation_type": str(raw.get("recommended_relation_type", "") or ""),
            "relation_asserted": str(raw.get("relation_asserted", "AMBIGUOUS") or "AMBIGUOUS").upper(),
            "confidence": confidence,
            "relation_confidence": relation_confidence,
            "label_confidence": label_confidence,
            "supporting_span_ids": [
                str(item) for item in raw.get("supporting_span_ids", []) or [] if str(item)
            ],
            "reason_code": str(raw.get("reason_code", "") or ""),
            "support_scope": str(raw.get("support_scope", "") or "").upper(),
        })
    return output


def score(records: list[dict[str, Any]], documents: list[Document]) -> dict[str, Any]:
    by_pmid = {str(record["pmid"]): record for record in records}
    label_counts, novelty_counts, document_counts = Counter(), Counter(), Counter()
    per_label: dict[str, Counter] = defaultdict(Counter)
    unavailable: list[str] = []
    for doc in documents:
        record = by_pmid.get(doc.pmid)
        if not record or record.get("status") != "OK":
            unavailable.append(doc.pmid)
            predicted: list[dict[str, Any]] = []
        else:
            predicted = record.get("relations", []) or []
        gold_label = {item.label_key for item in doc.relations}
        gold_novelty = {item.novelty_key for item in doc.relations}
        pred_label = {(*sorted((item["arg1_id"], item["arg2_id"])), item["relation_type"]) for item in predicted}
        pred_novelty = {(*sorted((item["arg1_id"], item["arg2_id"])), item["relation_type"], item["novelty"]) for item in predicted}
        label_counts.update(tp=len(gold_label & pred_label), fp=len(pred_label - gold_label), fn=len(gold_label - pred_label))
        novelty_counts.update(tp=len(gold_novelty & pred_novelty), fp=len(pred_novelty - gold_novelty), fn=len(gold_novelty - pred_novelty))
        for label in RELATION_TYPES:
            gold_l = {item for item in gold_label if item[2] == label}
            pred_l = {item for item in pred_label if item[2] == label}
            per_label[label].update(tp=len(gold_l & pred_l), fp=len(pred_l - gold_l), fn=len(gold_l - pred_l))
        if gold_label and pred_label:
            document_counts["tp"] += 1
        elif pred_label:
            document_counts["fp"] += 1
        elif gold_label:
            document_counts["fn"] += 1
        else:
            document_counts["tn"] += 1

    def prf(values: Counter) -> dict[str, float | int]:
        tp, fp, fn = (int(values[key]) for key in ("tp", "fp", "fn"))
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * p * r / (p + r) if p is not None and p + r else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}

    per_label_metrics = {label: prf(values) for label, values in per_label.items()}
    macro_items = [
        item for item in per_label_metrics.values()
        if item["tp"] + item["fn"] > 0
    ]
    macro_f1 = sum(item["f1"] for item in macro_items) / len(macro_items) if macro_items else None
    return {
        "pair_relation_micro": prf(label_counts),
        "pair_relation_novelty_micro": prf(novelty_counts),
        "positive_relation_macro_f1": macro_f1,
        "per_relation_type": per_label_metrics,
        "document_relation_presence": {**prf(document_counts), "tn": int(document_counts["tn"])},
        "unavailable_pmids": unavailable,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pubtator", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--pmids", default="",
        help="Optional comma-separated diagnostic document IDs; selection only, never labels",
    )
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument(
        "--dev-slice", choices=("all", "calibration", "eval"), default="all",
        help="Select the fixed sha256-ranked BioRED Dev calibration20 or dev-eval80 slice",
    )
    parser.add_argument(
        "--candidate-mode", choices=("free", "concept-pair", "hybrid"),
        default="free",
        help="free preserves the legacy Gemini generator; concept-pair classifies official ID pairs; hybrid adds pair recovery",
    )
    parser.add_argument(
        "--training-pubtator", type=Path,
        default=ROOT / ".cache/research/biored/dataset/BioRED/Train.PubTator",
    )
    parser.add_argument("--max-concept-pairs", type=int, default=160)
    parser.add_argument("--concept-pair-batch-size", type=int, default=48)
    parser.add_argument(
        "--label-review-batch-size", type=int, default=12,
        help="Pair-focused native-label review micro-batch size",
    )
    parser.add_argument("--concept-pair-sentence-distance", type=int, default=2)
    parser.add_argument("--max-recovery-pairs", type=int, default=24)
    parser.add_argument(
        "--label-adjudication", choices=("off", "single", "dual"), default="dual",
        help=(
            "Recheck BioRED native labels by candidate ID and EvidencePack. "
            "dual requires independent DeepSeek/Qwen agreement before a versioned relabel"
        ),
    )
    parser.add_argument(
        "--allow-model-label-edits", action="store_true",
        help=(
            "Enable dual-model label edits. Off by default until a Train-only "
            "transition calibration artifact passes its precision gate."
        ),
    )
    parser.add_argument(
        "--frozen-primary-records", type=Path,
        help="Offline verification replay from an existing report.json or records.jsonl",
    )
    args = parser.parse_args()
    guard_biored_test(args.pubtator, args.release_manifest)
    all_documents = parse_biored_pubtator(args.pubtator)
    full_calibration_docs: list[Any] = []
    full_dev_eval_docs: list[Any] = list(all_documents)
    if args.pubtator.name.casefold() == "dev.pubtator":
        full_calibration_docs, full_dev_eval_docs = stable_dev_partition(
            all_documents, calibration_size=min(20, len(all_documents)),
        )
    selected_documents = (
        full_calibration_docs if args.dev_slice == "calibration"
        else full_dev_eval_docs if args.dev_slice == "eval"
        else all_documents
    )
    requested_pmids = [item.strip() for item in str(args.pmids).split(",") if item.strip()]
    if requested_pmids:
        available = {item.pmid: item for item in selected_documents}
        missing = [item for item in requested_pmids if item not in available]
        if missing:
            raise ValueError(f"requested PMIDs are not in selected slice: {missing}")
        documents = [available[item] for item in requested_pmids][:args.limit]
    else:
        documents = selected_documents[:args.limit]
    native_registry, native_registry_manifest = registry_from_training(args.training_pubtator)
    native_schema_profile_manifest = native_registry.to_schema_profile(
        source=f"train_sha256:{native_registry_manifest['training_sha256']}",
    ).manifest()
    selected_pmids = {item.pmid for item in documents}
    calibration_docs = [item for item in full_calibration_docs if item.pmid in selected_pmids]
    dev_eval_docs = [item for item in full_dev_eval_docs if item.pmid in selected_pmids]
    # The project-local experiment configuration must win over stale terminal
    # or keychain values, just like the main evaluation suite preflight.
    os.environ.update(load_project_env(ROOT))
    cfg = load_config()
    primary_key, deepseek_key, qwen_key = (get_secret(name) for name in ("gemini_api_key", "deepseek_api_key", "qwen_api_key"))
    retry_options = {
        "max_retries": max(0, int(os.environ.get("AUX_MODEL_MAX_RETRIES", "5"))),
        "retry_base_delay_s": max(0.0, float(os.environ.get("AUX_MODEL_RETRY_BASE_DELAY_S", "2"))),
        "retry_max_delay_s": max(0.0, float(os.environ.get("AUX_MODEL_RETRY_MAX_DELAY_S", "45"))),
    }
    registry = AuxModelRegistry([
        AuxModelSpec("primary", "openai", cfg.model_id, cfg.api_base, primary_key, timeout_s=120, **retry_options),
        AuxModelSpec("recovery", "openai", cfg.model_id, cfg.api_base, primary_key, timeout_s=120, **retry_options),
        AuxModelSpec("judge", "openai", cfg.second_llm_model_id, cfg.second_llm_api_base, deepseek_key, timeout_s=120, **retry_options),
        AuxModelSpec("critic", "openai", cfg.aux_critic_model, cfg.qwen_api_base, qwen_key, timeout_s=120, **retry_options),
    ])
    required_roles = {"primary"} if not args.frozen_primary_records else set()
    if args.label_adjudication in {"single", "dual"}:
        required_roles.add("judge")
    if args.label_adjudication == "dual":
        required_roles.add("critic")
    if not all(registry.configured(role) for role in required_roles):
        raise RuntimeError(
            "missing configured roles for this run: "
            + ", ".join(sorted(role for role in required_roles if not registry.configured(role)))
        )
    if args.frozen_primary_records and args.candidate_mode == "hybrid" and not registry.configured("recovery"):
        raise RuntimeError("Gemini recovery credentials must be configured for frozen hybrid replay")
    schema = {"relations": [{"candidate_id": "stable candidate id when supplied", "pair_candidate_id": "stable pair id when supplied", "arg1_id": "string", "arg2_id": "string", "relation_type": "BioRED label", "novelty": "Novel|No|Unknown", "confidence": "0..1", "evidence_quote": "exact source quote", "supporting_span_ids": ["EvidencePack span ids"], "claim_role": "CURRENT_FINDING|PRIOR_WORK|BACKGROUND|METHOD|PREDICTION|SPECULATIVE|OTHER"}]}
    system = (
        "You perform BioRED given-entity relation extraction. Use only supplied concept IDs. "
        f"Allowed relation_type values: {', '.join(RELATION_TYPES)}. "
        "Return only direct article-supported relation rows and omit NO_RELATION cases. "
        "The word positive here never means Positive_Correlation; choose that label only from target-pair semantics. "
        "Do not infer co-occurrence. "
        "Novel means the article presents the relation as new; use No only when the article clearly marks it as prior knowledge, otherwise Unknown. Return strict JSON."
    )
    pair_system = (
        system
        + " Classify only the supplied CANDIDATE_PAIRS. Return no row for NO_RELATION. "
        "Copy pair_candidate_id exactly and choose relation_type only from that pair's allowed_relation_types. "
        "For the TARGET PAIR, use Positive_Correlation only when one endpoint increases, promotes, causes, "
        "elevates, worsens, or increases the risk of the other; use Negative_Correlation only when one endpoint "
        "decreases, inhibits, attenuates, protects against, improves, or reduces the risk of the other. "
        "A change word elsewhere in the sentence does not determine the target pair's sign. Use Association "
        "when a direct relation is asserted but its effect between the target endpoints is unsigned. "
        "Each pair includes BioRED-Train type-signature label priors. Treat a >=0.90 dominant prior as the "
        "default when evidence is not explicit, but let direct target-pair evidence override the prior. "
        "Use these native relation cards and their boundaries:\n"
        + json.dumps(native_registry.prompt_cards(), ensure_ascii=False, separators=(",", ":"))
    )
    label_schema = {"decisions": [{
        "candidate_id": "copy exactly", "candidate_version": "copy exactly",
        "verdict": "KEEP|EDIT|ABSTAIN",
        "relation_asserted": "YES|NO|AMBIGUOUS",
        "relation_confidence": "0..1 confidence that the target pair has a direct/allowed BioRED relation",
        "recommended_relation_type": "one allowed BioRED label",
        "label_confidence": "0..1 confidence in the recommended label conditional on relation=YES",
        "confidence": "legacy overall 0..1", "supporting_span_ids": ["OWNER span ids copied exactly"],
        "support_scope": "DIRECT|MULTI_SENTENCE|DERIVED_BY_GUIDELINE|NONE",
        "reason_code": "concise boundary reason",
    }]}
    label_system = (
        "You are a conservative BioRED native relation and label adjudicator following the official annotation guideline. "
        "For EACH TARGET PAIR, first ignore the untrusted current label and independently decide whether BioRED would annotate "
        "a relation (YES, NO, or AMBIGUOUS). Then, only conditional on YES, choose the official label and score relation truth "
        "confidence separately from label confidence. Use the pair-local OWNER spans as citeable evidence and the full abstract "
        "only to resolve document-level or guideline-derived context. Copy candidate identity and cited span IDs exactly. A cue such as "
        "increase/decrease counts only when it semantically connects the two TARGET endpoints, not when it describes "
        "another entity in the same sentence. Association is a direct but unsigned relation; Positive_Correlation is "
        "a target-pair increase/promotion/causation or increased risk; Negative_Correlation is a target-pair "
        "decrease/inhibition/protection/improvement or reduced risk. "
        "A pathogenic variant stated to cause, predispose to, or increase susceptibility to a disease is "
        "Positive_Correlation; do not downgrade it to Association merely because the effect is not numeric. "
        "Bind requires physical binding, Cotreatment joint administration, Drug_Interaction an asserted pharmacokinetic/pharmacodynamic interaction, Conversion actual "
        "conversion, and Comparison an explicit comparison. KEEP the current label when it is already best, EDIT only "
        "when another allowed label is supported, and ABSTAIN when relation truth or label remains ambiguous. The supplied "
        "signature_contract contains the endpoint-specific official boundary and Train-only label priors; priors are tie-breakers, "
        "not evidence. Return one decision per item."
    )
    frozen_records: dict[str, dict[str, Any]] = {}
    if args.frozen_primary_records:
        if args.candidate_mode not in {"free", "hybrid"}:
            raise ValueError("frozen primary replay supports --candidate-mode free or hybrid")
        raw = args.frozen_primary_records.read_text(encoding="utf-8")
        if args.frozen_primary_records.suffix == ".jsonl":
            values = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            payload = json.loads(raw)
            values = list(payload.get("records", []) if isinstance(payload, dict) else payload)
        frozen_records = {str(item.get("pmid", "")): item for item in values}
    lock = threading.Lock()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    cache = LightweightExtractionCache(
        mode="persistent", path=str(args.cache_path),
        persistent_max_entries=2000, persistent_max_mb=200, ttl_days=30,
    )
    abort_event = threading.Event()
    abort_reason = {"value": ""}

    def run_document(doc: Any) -> dict[str, Any]:
        if abort_event.is_set():
            return {"pmid": doc.pmid, "status": "SKIPPED_GLOBAL_PROVIDER_FAILURE", "error": abort_reason["value"], "relations": []}
        entities = [
            {"id": key, "type": value.entity_type, "mentions": sorted(_mention_texts(value))}
            for key, value in sorted(doc.concepts.items())
        ]
        user = "ARTICLE:\n" + doc.text + "\n\nENTITIES:\n" + json.dumps(entities, ensure_ascii=False, separators=(",", ":"))
        audits: dict[str, Any] = {}
        generation_metrics: dict[str, int] = {
            "hint_candidates": 0,
            "official_pair_candidates": 0,
            "concept_pair_batches": 0,
            "explicit_recovery_candidates": 0,
            "non_explicit_filtered_pairs": 0,
            "budget_truncated_pairs": 0,
        }
        candidates: list[dict[str, Any]] = []
        pair_candidates: list[dict[str, Any]] = []

        if args.frozen_primary_records:
            frozen = frozen_records.get(doc.pmid)
            if frozen is None:
                return {"pmid": doc.pmid, "status": "PRIMARY_REPLAY_MISSING", "relations": []}
            candidates = [dict(item) for item in (
                frozen.get("primary_relations", frozen.get("raw_relations", [])) or []
            )]
            audits.update(primary_replay=True, primary_replay_source=str(args.frozen_primary_records))
        elif args.candidate_mode in {"free", "hybrid"}:
            primary, primary_cache = call_cached(cache, registry, "primary", system, user, schema)
            audits.update(primary=primary.to_dict(), primary_cache=primary_cache)
            if primary.status != "OK":
                failure_category = classify_provider_error(primary.error)
                if failure_category in {AUTH_ERROR, QUOTA_OR_TOKEN_EXHAUSTED}:
                    abort_reason["value"] = failure_category
                    abort_event.set()
                return {"pmid": doc.pmid, "status": "PRIMARY_FAILED", "error": primary.error, "failure_category": failure_category, "primary": primary.to_dict(), "cache": primary_cache, "relations": []}
            candidates = normalise_relations(primary.payload, doc.concepts, doc.text)

        if args.candidate_mode == "concept-pair":
            pair_candidates, pair_metrics = build_native_pair_candidates(
                doc, native_registry, explicit_only=False,
                max_candidates=args.max_concept_pairs,
                max_sentence_distance=args.concept_pair_sentence_distance,
            )
            generation_metrics["budget_truncated_pairs"] += pair_metrics["budget_truncated_pairs"]
            generation_metrics["non_explicit_filtered_pairs"] += pair_metrics["filtered_nonlocal_pairs"]
            generation_metrics["official_pair_candidates"] = len(pair_candidates)
            generation_metrics["explicit_recovery_candidates"] = sum(
                item.get("support_match") in {"EXPLICIT", "WEAK"}
                for item in pair_candidates
            )
            if pair_candidates:
                batch_size = max(1, int(args.concept_pair_batch_size))
                batches = [
                    pair_candidates[index:index + batch_size]
                    for index in range(0, len(pair_candidates), batch_size)
                ]
                generation_metrics["concept_pair_batches"] = len(batches)
                batch_audits: list[dict[str, Any]] = []
                candidates = []
                for batch_index, batch in enumerate(batches, start=1):
                    pair_map = {item["pair_candidate_id"]: item for item in batch}
                    pair_user = (
                        user
                        + f"\n\nBATCH {batch_index}/{len(batches)}. Classify every supplied pair independently."
                        + "\n\nCANDIDATE_PAIRS:\n"
                        + json.dumps(batch, ensure_ascii=False, separators=(",", ":"))
                    )
                    primary, primary_cache = call_cached(
                        cache, registry, "primary", pair_system, pair_user, schema,
                    )
                    batch_audits.append({
                        "batch_index": batch_index,
                        "pair_candidate_ids": list(pair_map),
                        "result": primary.to_dict(),
                        "cache": primary_cache,
                    })
                    if primary.status != "OK":
                        audits["primary_batches"] = batch_audits
                        failure_category = classify_provider_error(primary.error)
                        if failure_category in {AUTH_ERROR, QUOTA_OR_TOKEN_EXHAUSTED}:
                            abort_reason["value"] = failure_category
                            abort_event.set()
                        return {
                            "pmid": doc.pmid, "status": "PRIMARY_FAILED",
                            "error": primary.error, "failure_category": failure_category,
                            "primary_batches": batch_audits, "relations": [],
                        }
                    candidates.extend(normalise_relations(
                        primary.payload, doc.concepts, doc.text,
                        candidate_pairs=pair_map,
                    ))
                audits["primary_batches"] = batch_audits

        if args.candidate_mode == "hybrid":
            existing_pairs = {
                tuple(sorted((item["arg1_id"], item["arg2_id"]))) for item in candidates
            }
            pair_candidates, pair_metrics = build_native_pair_candidates(
                doc, native_registry, existing_pairs=existing_pairs,
                explicit_only=True, max_candidates=args.max_recovery_pairs,
            )
            generation_metrics["explicit_recovery_candidates"] = len(pair_candidates)
            generation_metrics["official_pair_candidates"] = len(pair_candidates)
            generation_metrics["non_explicit_filtered_pairs"] += (
                pair_metrics["filtered_nonlocal_pairs"] + pair_metrics["filtered_no_explicit_support"]
            )
            generation_metrics["budget_truncated_pairs"] += pair_metrics["budget_truncated_pairs"]
            if pair_candidates:
                pair_map = {item["pair_candidate_id"]: item for item in pair_candidates}
                recovery_user = user + "\n\nCANDIDATE_PAIRS:\n" + json.dumps(pair_candidates, ensure_ascii=False, separators=(",", ":"))
                recovery, recovery_cache = call_cached(cache, registry, "recovery", pair_system, recovery_user, schema)
                audits.update(recovery=recovery.to_dict(), recovery_cache=recovery_cache)
                if recovery.status == "OK":
                    recovered = normalise_relations(
                        recovery.payload, doc.concepts, doc.text, candidate_pairs=pair_map,
                    )
                    for item in recovered:
                        item["candidate_lane"] = "recovery"
                    candidates.extend(recovered)

        generation_metrics["hint_candidates"] = sum(
            str(item.get("candidate_lane", "extracted_hint") or "extracted_hint") != "recovery"
            for item in candidates
        )
        candidates = [
            seed_native_lineage(
                item, doc,
                lane=(
                    "recovery"
                    if str(item.get("candidate_lane", "extracted_hint") or "extracted_hint") == "recovery"
                    else "extracted_hint"
                ),
            )
            for item in candidates
        ]
        validated, validation_audit = validate_native_predictions(candidates, doc, native_registry)
        enriched: list[dict[str, Any]] = []
        for item in validated:
            lane = str(item.get("candidate_lane", "extracted_hint") or "extracted_hint")
            enriched.append(enrich_native_relation(
                item, doc, native_registry,
                lane="recovery" if lane == "recovery" else "extracted_hint",
            ))

        enriched, train_signature_prior_audit = apply_train_signature_prior_guard(
            enriched, doc, native_registry,
        )
        label_adjudication_audit: list[dict[str, Any]] = []
        label_reviews = build_native_label_review_items(enriched, doc, native_registry)
        if args.label_adjudication != "off" and label_reviews:
            review_batch_size = max(1, int(args.label_review_batch_size))
            review_batches = [
                label_reviews[index:index + review_batch_size]
                for index in range(0, len(label_reviews), review_batch_size)
            ]
            judge_decisions: list[dict[str, Any]] = []
            judge_batch_audits: list[dict[str, Any]] = []
            for batch_index, review_batch in enumerate(review_batches, start=1):
                label_user = (
                    "DOCUMENT TITLE AND ABSTRACT:\n" + doc.text
                    + f"\n\nPAIR-FOCUSED REVIEW BATCH {batch_index}/{len(review_batches)}. "
                    "Decide each pair independently; do not transfer a cue or decision between candidates."
                    + "\n\nCANDIDATES TO ADJUDICATE:\n"
                    + json.dumps(review_batch, ensure_ascii=False, separators=(",", ":"))
                )
                judge_result, judge_cache = call_cached(
                    cache, registry, "judge", label_system, label_user, label_schema,
                )
                judge_batch_audits.append({
                    "batch_index": batch_index,
                    "candidate_ids": [item["candidate_id"] for item in review_batch],
                    "result": judge_result.to_dict(), "cache": judge_cache,
                })
                if judge_result.status != "OK":
                    audits["label_judge_batches"] = judge_batch_audits
                    failure_category = classify_provider_error(judge_result.error)
                    return {
                        "pmid": doc.pmid, "status": "LABEL_JUDGE_FAILED",
                        "error": judge_result.error,
                        "failure_category": failure_category,
                        "relations": [], "audits": audits,
                    }
                judge_decisions.extend(normalise_label_decisions(judge_result.payload))
            audits["label_judge_batches"] = judge_batch_audits
            critic_decisions: list[dict[str, Any]] = []
            proposed_edit_keys = {
                (str(item.get("candidate_id", "")), int(item.get("candidate_version", 1) or 1))
                for item in judge_decisions
                if str(item.get("verdict", "")).upper() == "EDIT"
            }
            if args.label_adjudication == "dual" and proposed_edit_keys:
                critic_reviews = [
                    item for item in label_reviews
                    if (str(item["candidate_id"]), int(item["candidate_version"])) in proposed_edit_keys
                ]
                critic_batches = [
                    critic_reviews[index:index + review_batch_size]
                    for index in range(0, len(critic_reviews), review_batch_size)
                ]
                critic_batch_audits: list[dict[str, Any]] = []
                for batch_index, critic_batch in enumerate(critic_batches, start=1):
                    critic_user = (
                        "Independently adjudicate these proposed-to-change candidates. The first judge's answer "
                        "is intentionally hidden; treat the current label as untrusted.\n\nDOCUMENT TITLE AND ABSTRACT:\n"
                        + doc.text
                        + f"\n\nPAIR-FOCUSED CRITIC BATCH {batch_index}/{len(critic_batches)}."
                        + "\n\nCANDIDATES TO ADJUDICATE:\n"
                        + json.dumps(critic_batch, ensure_ascii=False, separators=(",", ":"))
                    )
                    critic_result, critic_cache = call_cached(
                        cache, registry, "critic", label_system, critic_user, label_schema,
                    )
                    critic_batch_audits.append({
                        "batch_index": batch_index,
                        "candidate_ids": [item["candidate_id"] for item in critic_batch],
                        "result": critic_result.to_dict(), "cache": critic_cache,
                    })
                    if critic_result.status != "OK":
                        audits["label_critic_batches"] = critic_batch_audits
                        failure_category = classify_provider_error(critic_result.error)
                        return {
                            "pmid": doc.pmid, "status": "LABEL_CRITIC_FAILED",
                            "error": critic_result.error,
                            "failure_category": failure_category,
                            "relations": [], "audits": audits,
                        }
                    critic_decisions.extend(normalise_label_decisions(critic_result.payload))
                audits["label_critic_batches"] = critic_batch_audits
            enriched, label_adjudication_audit = apply_native_label_adjudication(
                enriched, doc, native_registry, judge_decisions, critic_decisions,
                require_critic=args.label_adjudication == "dual",
                allow_model_label_edits=args.allow_model_label_edits,
            )

        # Reconcile by typed native edge.  A hint remains representative when
        # recovery independently finds the same edge; both claim instances and
        # lanes remain auditable.
        by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in enriched:
            key = (*sorted((item["arg1_id"], item["arg2_id"])), item["relation_type"])
            previous = by_key.get(key)
            if previous is None:
                item["claim_instances"] = [dict(item)]
                by_key[key] = item
                continue
            instances = list(previous.get("claim_instances", []) or [dict(previous)])
            instances.append(dict(item))
            representative = previous
            if previous.get("candidate_lane") == "recovery" and item.get("candidate_lane") == "extracted_hint":
                representative = item
            representative["claim_instances"] = instances
            representative["source_lanes"] = sorted({
                str(value.get("candidate_lane", "")) for value in instances
            } - {""})
            representative["merged_candidate_ids"] = sorted({
                str(value.get("candidate_id", "")) for value in instances
            } - {"", str(representative.get("candidate_id", ""))})
            by_key[key] = representative
        relations = list(by_key.values())
        primary_hash = hashlib.sha256(json.dumps(
            candidates, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        framework_relations = [
            item for item in relations if item.get("semantic_status") == "ACCEPTED"
        ]
        output_candidate_keys = {
            str(item.get("pair_candidate_id", "")) for item in relations
            if str(item.get("pair_candidate_id", ""))
        }
        ledger = [
            {
                "candidate_id": item.get("candidate_id", ""),
                "candidate_version": item.get("candidate_version", 1),
                "candidate_lane": item.get("candidate_lane", ""),
                "pair_candidate_id": item.get("pair_candidate_id", ""),
                "disposition": (
                    "KEPT" if item.get("semantic_status") == "ACCEPTED"
                    else "SEMANTIC_AUDIT_ONLY"
                ),
                "reason_codes": list(item.get("quality_flags", []) or []),
            }
            for item in relations
        ]
        ledger.extend({
            "candidate_id": item["candidate_id"],
            "candidate_version": 1,
            "candidate_lane": "recovery",
            "pair_candidate_id": item["pair_candidate_id"],
            "disposition": "GATE_NOT_ASSERTED",
            "reason_codes": ["pair_model_no_positive_relation"],
        } for item in pair_candidates if item["pair_candidate_id"] not in output_candidate_keys)
        ledger.extend({
            "candidate_id": str((item.get("candidate", {}) or {}).get("candidate_id", "")),
            "candidate_version": int((item.get("candidate", {}) or {}).get("candidate_version", 1) or 1),
            "candidate_lane": str((item.get("candidate", {}) or {}).get("candidate_lane", "")),
            "pair_candidate_id": str((item.get("candidate", {}) or {}).get("pair_candidate_id", "")),
            "disposition": "FACTUAL_DISCARD",
            "reason_codes": list(item.get("reason_codes", []) or []),
        } for item in validation_audit)
        ledger.extend({
            "candidate_id": str(item.get("candidate_id", "")),
            "candidate_version": int(item.get("candidate_version_before", 1) or 1),
            "candidate_lane": "recovery",
            "pair_candidate_id": "",
            "disposition": "SUPERSEDED",
            "reason_codes": [str(item.get("reason_code", "train_signature_dominant_label"))],
        } for item in train_signature_prior_audit)
        return {
            "pmid": doc.pmid, "status": "OK",
            "candidate_mode": args.candidate_mode,
            "primary_output_hash": primary_hash,
            "primary_relations": candidates,
            "raw_relations": relations,
            "factual_relations": relations,
            "framework_relations": framework_relations,
            "relations": framework_relations,
            "candidate_generation_metrics": generation_metrics,
            "pair_candidate_audit": pair_candidates,
            "candidate_audit_ledger": ledger,
            "native_validation_audit": validation_audit,
            "train_signature_prior_audit": train_signature_prior_audit,
            "native_label_review_items": label_reviews,
            "native_label_adjudication_audit": label_adjudication_audit,
            "current_finding_diagnostic": {
                "predicted_current_finding": sum(
                    item.get("claim_role") == "CURRENT_FINDING" for item in relations
                ),
                "predicted_non_current": sum(
                    item.get("claim_role") != "CURRENT_FINDING" for item in relations
                ),
                "gold_recall": None,
            },
            "audits": audits,
        }

    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(2, args.max_workers))) as pool:
        futures = {pool.submit(run_document, document): document.pmid for document in documents}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            with lock:
                records.append(record)
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    cache.close()
    records.sort(key=lambda item: item["pmid"])
    failure_categories = Counter(
        item.get("failure_category") or classify_provider_error(str(item.get("error", "")))
        for item in records if item.get("status") != "OK"
    )
    token_or_quota_error = bool(failure_categories[QUOTA_OR_TOKEN_EXHAUSTED])
    auth_error = bool(failure_categories[AUTH_ERROR])
    usage = {role: dict(value) for role, value in registry.usage.items()}
    usage_unavailable_roles = [
        role for role, values in usage.items()
        if int(values.get("successful", 0) or 0) and not (
            int(values.get("prompt_tokens", 0) or 0) or int(values.get("output_tokens", 0) or 0)
        )
    ]
    completed = sum(item.get("status") == "OK" for item in records)
    if token_or_quota_error:
        run_status = "incomplete_token_or_quota_error"
    elif auth_error:
        run_status = "incomplete_auth_error"
    elif completed == len(documents):
        run_status = "completed"
    else:
        run_status = "incomplete_provider_error"
    fn_ledger, fn_summary = audit_false_negatives(
        records, documents, relation_field="primary_relations",
    )
    with (args.output_dir / "false_negative_audit.jsonl").open("w", encoding="utf-8") as handle:
        for item in fn_ledger:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    removed_tp_ledger: list[dict[str, Any]] = []
    removed_fp = 0
    by_document = {item.pmid: item for item in documents}
    for record in records:
        document = by_document[str(record["pmid"])]
        gold = {item.label_key for item in document.relations}
        accepted = {
            (*sorted((item["arg1_id"], item["arg2_id"])), item["relation_type"])
            for item in record.get("framework_relations", []) or []
        }
        factual_by_key = {
            (*sorted((item["arg1_id"], item["arg2_id"])), item["relation_type"]): item
            for item in record.get("factual_relations", []) or []
        }
        for item in record.get("primary_relations", []) or []:
            key = (*sorted((item["arg1_id"], item["arg2_id"])), item["relation_type"])
            if key in accepted:
                continue
            if key in gold:
                audited = factual_by_key.get(key, item)
                removed_tp_ledger.append({
                    "pmid": record["pmid"], "label_key": list(key),
                    "candidate_id": audited.get("candidate_id", ""),
                    "support_mode": audited.get("support_mode", "UNRESOLVED"),
                    "relation_card_match": audited.get("relation_card_match", "NONE"),
                    "quality_flags": audited.get("quality_flags", []),
                    "evidence_pack": audited.get("evidence_pack", {}),
                })
            else:
                removed_fp += 1
    (args.output_dir / "verification_removed_tp_audit.json").write_text(
        json.dumps(removed_tp_ledger, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    candidate_generation_totals = Counter()
    label_adjudication_totals = Counter()
    label_adjudication_transitions: dict[str, Counter] = defaultdict(Counter)
    train_signature_prior_transitions: dict[str, Counter] = defaultdict(Counter)
    generated_pair_spaces: dict[str, dict[tuple[str, str], set[str]]] = {}
    projected_candidate_ids: set[str] = set()
    projected_candidate_versions: set[tuple[str, int]] = set()
    projected_pair_ids: set[str] = set()
    ledger_candidate_ids: set[str] = set()
    ledger_candidate_versions: set[tuple[str, int]] = set()
    ledger_pair_ids: set[str] = set()
    r_prefix_lane_mismatch = lineage_binding_missing = 0
    for record in records:
        candidate_generation_totals.update(record.get("candidate_generation_metrics", {}) or {})
        for item in record.get("native_label_adjudication_audit", []) or []:
            decision = str(item.get("decision", "REVIEW") or "REVIEW")
            label_adjudication_totals[decision] += 1
            original = str(item.get("original_relation_type", "") or "")
            final = str(item.get("final_relation_type", "") or "")
            if original and final:
                label_adjudication_transitions[original][final] += 1
        for item in record.get("train_signature_prior_audit", []) or []:
            original = str(item.get("original_relation_type", "") or "")
            final = str(item.get("final_relation_type", "") or "")
            if original and final:
                train_signature_prior_transitions[original][final] += 1
        spaces: dict[tuple[str, str], set[str]] = {}
        for item in record.get("pair_candidate_audit", []) or []:
            pair = tuple(sorted((str(item.get("arg1_id", "")), str(item.get("arg2_id", "")))))
            spaces.setdefault(pair, set()).update(item.get("allowed_relation_types", []) or [])
        generated_pair_spaces[str(record.get("pmid", ""))] = spaces
        for item in record.get("primary_relations", []) or []:
            candidate_id = str(item.get("candidate_id", "") or "")
            if candidate_id:
                projected_candidate_ids.add(candidate_id)
                projected_candidate_versions.add((
                    candidate_id, int(item.get("candidate_version", 1) or 1),
                ))
            r_prefix_lane_mismatch += int(
                candidate_id.startswith("r-") and item.get("candidate_lane") != "recovery"
            )
            lineage_binding_missing += int(
                "lineage_binding_missing" in set(item.get("quality_flags", []) or [])
            )
        for item in record.get("pair_candidate_audit", []) or []:
            projected_pair_ids.add(str(item.get("pair_candidate_id", "") or ""))
        for item in record.get("candidate_audit_ledger", []) or []:
            ledger_candidate_id = str(item.get("candidate_id", "") or "")
            ledger_candidate_ids.add(ledger_candidate_id)
            if ledger_candidate_id:
                ledger_candidate_versions.add((
                    ledger_candidate_id,
                    int(item.get("candidate_version", 1) or 1),
                ))
            ledger_pair_ids.add(str(item.get("pair_candidate_id", "") or ""))
    projected_pair_ids.discard("")
    ledger_candidate_ids.discard("")
    ledger_pair_ids.discard("")
    fn_pair_covered = fn_label_space_covered = 0
    for item in fn_ledger:
        pair = tuple(sorted((str(item["arg1_id"]), str(item["arg2_id"]))))
        space = generated_pair_spaces.get(str(item["pmid"]), {}).get(pair)
        if space is not None:
            fn_pair_covered += 1
            if str(item["gold_relation_type"]) in space:
                fn_label_space_covered += 1
    hint_framework_records = [
        {
            **record,
            "relations": [
                item for item in record.get("framework_relations", []) or []
                if "extracted_hint" in set(item.get("source_lanes", []) or [item.get("candidate_lane", "")])
            ],
        }
        for record in records
    ]
    recovery_framework_records = [
        {
            **record,
            "relations": [
                item for item in record.get("framework_relations", []) or []
                if "recovery" in set(item.get("source_lanes", []) or [item.get("candidate_lane", "")])
            ],
        }
        for record in records
    ]
    hint_framework_metrics = score(hint_framework_records, documents)
    recovery_framework_metrics = score(recovery_framework_records, documents)
    all_framework_metrics = score([
        {**item, "relations": item.get("framework_relations", [])} for item in records
    ], documents)
    report = {
        "protocol": "BioRED-native-v8 pair-local-context/microbatch/dual-confidence-adjudication; no LiverKG mapping; Neo4j blocked",
        "candidate_mode": args.candidate_mode,
        "dev_slice": args.dev_slice,
        "requested_pmids": requested_pmids,
        "label_adjudication_mode": args.label_adjudication,
        "concept_pair_config": {
            "max_candidates_per_document": args.max_concept_pairs,
            "batch_size": args.concept_pair_batch_size,
            "max_sentence_distance": args.concept_pair_sentence_distance,
        },
        "label_review_config": {
            "contract_version": "biored-label-adjudication-v2",
            "batch_size": args.label_review_batch_size,
            "relation_confidence_threshold": 0.90,
            "label_confidence_threshold": 0.80,
            "edit_confidence_threshold": 0.90,
            "model_label_edits_enabled": args.allow_model_label_edits,
            "full_abstract_context": True,
            "pair_local_evidence": True,
        },
        "native_registry_manifest": native_registry_manifest,
        "schema_profile_manifest": native_schema_profile_manifest,
        "split": args.pubtator.name, "documents_requested": len(documents), "documents_completed": completed,
        "run_status": run_status, "evaluation_valid": completed == len(documents),
        "failure_categories": dict(failure_categories), "usage": usage,
        "usage_unavailable_roles": usage_unavailable_roles,
        "metrics": score(records, documents),
        "raw_metrics": score([
            {**item, "relations": item.get("primary_relations", item.get("raw_relations", []))}
            for item in records
        ], documents),
        "factual_metrics": score([
            {**item, "relations": item.get("factual_relations", item.get("raw_relations", []))}
            for item in records
        ], documents),
        "framework_metrics": score([
            {**item, "relations": item.get("framework_relations", item.get("relations", []))}
            for item in records
        ], documents),
        "false_negative_audit": fn_summary,
        "verification_delta_audit": {
            "removed_true_positives": len(removed_tp_ledger),
            "removed_false_positives": removed_fp,
            "removed_tp_ledger": "verification_removed_tp_audit.json",
        },
        "candidate_generation_metrics": dict(candidate_generation_totals),
        "native_label_adjudication": {
            "contract_version": "biored-label-adjudication-v2",
            "decision_counts": dict(label_adjudication_totals),
            "transitions": {
                label: dict(values)
                for label, values in sorted(label_adjudication_transitions.items())
            },
            "automatic_regex_relabel_disabled": True,
            "automatic_model_relabel_disabled": not args.allow_model_label_edits,
            "train_signature_prior_transitions": {
                label: dict(values)
                for label, values in sorted(train_signature_prior_transitions.items())
            },
            "train_cue_audit": {
                "association_precision": 0.7207207207207207,
                "positive_correlation_precision": 0.3422263109475621,
                "negative_correlation_precision": 0.2913752913752914,
                "note": "Train-only owner-local cue audit; lexical cues route review but never relabel",
            },
        },
        "lineage_audit": {
            "lineage_contract_version": "lineage-v2",
            "projected_candidate_count": len(projected_candidate_ids),
            "projected_pair_candidate_count": len(projected_pair_ids),
            "audit_lineage_accounting": (
                len(projected_candidate_ids & ledger_candidate_ids) / len(projected_candidate_ids)
                if projected_candidate_ids else 1.0
            ),
            "version_accounting": (
                len(projected_candidate_versions & ledger_candidate_versions)
                / len(projected_candidate_versions)
                if projected_candidate_versions else 1.0
            ),
            "pair_candidate_accounting": (
                len(projected_pair_ids & ledger_pair_ids) / len(projected_pair_ids)
                if projected_pair_ids else 1.0
            ),
            "unaccounted_candidate_ids": sorted(projected_candidate_ids - ledger_candidate_ids),
            "unaccounted_pair_candidate_ids": sorted(projected_pair_ids - ledger_pair_ids),
            "r_prefix_lane_mismatch": r_prefix_lane_mismatch,
            "lineage_binding_missing": lineage_binding_missing,
        },
        "candidate_recovery_oracle_audit": {
            "gold_labels_not_used_for_generation": True,
            "false_negatives": len(fn_ledger),
            "pair_covered": fn_pair_covered,
            "label_in_allowed_space": fn_label_space_covered,
            "pair_coverage": fn_pair_covered / len(fn_ledger) if fn_ledger else None,
            "label_space_coverage": fn_label_space_covered / len(fn_ledger) if fn_ledger else None,
        },
        "lane_metrics": {
            "hint_only": hint_framework_metrics,
            "recovery_only": recovery_framework_metrics,
            "combined": all_framework_metrics,
            "recovery_tp_gain": (
                all_framework_metrics["pair_relation_micro"]["tp"]
                - hint_framework_metrics["pair_relation_micro"]["tp"]
            ),
            "recovery_fp_added": (
                all_framework_metrics["pair_relation_micro"]["fp"]
                - hint_framework_metrics["pair_relation_micro"]["fp"]
            ),
        },
        "current_finding_diagnostic": {
            "predicted_current_finding": sum(
                int((item.get("current_finding_diagnostic", {}) or {}).get(
                    "predicted_current_finding", 0
                )) for item in records
            ),
            "predicted_non_current": sum(
                int((item.get("current_finding_diagnostic", {}) or {}).get(
                    "predicted_non_current", 0
                )) for item in records
            ),
            "gold_recall": None,
        },
        "dev_partition": {
            "method": "sha256_pmid_rank_v1",
            "calibration_seen_pmids": [item.pmid for item in calibration_docs],
            "dev_eval_pmids": [item.pmid for item in dev_eval_docs],
            "calibration_metrics": score(records, calibration_docs) if calibration_docs else None,
            "dev_eval_metrics": score(records, dev_eval_docs) if dev_eval_docs else None,
            "raw_calibration_metrics": score([
                {**item, "relations": item.get("primary_relations", [])} for item in records
            ], calibration_docs) if calibration_docs else None,
            "raw_dev_eval_metrics": score([
                {**item, "relations": item.get("primary_relations", [])} for item in records
            ], dev_eval_docs) if dev_eval_docs else None,
        },
        "official_test_run": args.pubtator.name.casefold() == "test.pubtator",
        "neo4j_mutations": 0,
        "records": records,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = report["metrics"]["pair_relation_micro"]
    raw_summary = report["raw_metrics"]["pair_relation_micro"]
    factual_summary = report["factual_metrics"]["pair_relation_micro"]
    (args.output_dir / "report.md").write_text(
        "# BioRED given-entity benchmark\n\n"
        f"- Status: `{report['run_status']}`\n- Completed: {report['documents_completed']}/{len(documents)}\n"
        f"- Candidate mode: `{args.candidate_mode}`\n"
        f"- Label adjudication: `{args.label_adjudication}`; decisions `{json.dumps(report['native_label_adjudication']['decision_counts'], ensure_ascii=False)}`\n"
        f"- Evaluation valid: `{report['evaluation_valid']}`\n"
        f"- Provider failures: `{json.dumps(report['failure_categories'], ensure_ascii=False)}`\n"
        f"- Raw pair+relation P/R/F1: {raw_summary['precision'] if raw_summary['precision'] is not None else 'undefined'}/{raw_summary['recall']:.3f}/{raw_summary['f1']:.3f} ({raw_summary['tp']}/{raw_summary['fp']}/{raw_summary['fn']})\n"
        f"- Factual pair+relation P/R/F1: {factual_summary['precision'] if factual_summary['precision'] is not None else 'undefined'}/{factual_summary['recall']:.3f}/{factual_summary['f1']:.3f} ({factual_summary['tp']}/{factual_summary['fp']}/{factual_summary['fn']})\n"
        f"- Semantic accepted P/R/F1: {summary['precision'] if summary['precision'] is not None else 'undefined'}/{summary['recall']:.3f}/{summary['f1']:.3f} ({summary['tp']}/{summary['fp']}/{summary['fn']})\n"
        f"- Pair+relation+novelty micro F1: {report['metrics']['pair_relation_novelty_micro']['f1']:.3f}\n"
        f"- Positive relation macro F1: {report['metrics']['positive_relation_macro_f1']:.3f}\n"
        f"- FN audit: `{json.dumps(report['false_negative_audit']['category_counts'], ensure_ascii=False)}`\n"
        f"- Lineage accounting: `{report['lineage_audit']['audit_lineage_accounting']}`; pair accounting: `{report['lineage_audit']['pair_candidate_accounting']}`\n",
        encoding="utf-8",
    )
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
