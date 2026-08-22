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
from liverkg_cli.security import get_secret

RELATION_TYPES = (
    "Association", "Bind", "Comparison", "Conversion", "Cotreatment",
    "Drug_Interaction", "Negative_Correlation", "Positive_Correlation",
)
NOVELTY = ("Novel", "No")


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
                relations.append(Relation(left, right, relation_type, novelty if novelty in NOVELTY else "No"))
        documents.append(Document(pmid, f"{title}\n{abstract}", concepts, relations))
    return documents


def model_result_dict(result: StructuredModelResult) -> dict[str, Any]:
    return result.to_dict()


def normalise_relations(payload: dict[str, Any], concepts: dict[str, Concept]) -> list[dict[str, Any]]:
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
        key = (*sorted((left, right)), relation_type)
        if key in seen:
            continue
        seen.add(key)
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        clean.append({
            "arg1_id": left, "arg2_id": right, "relation_type": relation_type,
            "novelty": str(item.get("novelty", "No")) if str(item.get("novelty", "No")) in NOVELTY else "No",
            "confidence": confidence,
        })
    return clean


def call_cached(
    cache: LightweightExtractionCache, registry: AuxModelRegistry, role: str,
    system: str, user: str, schema: dict[str, Any],
) -> tuple[StructuredModelResult, str]:
    key_payload = {
        "harness": "biored-given-entity-v1", "role": role,
        "model": registry.specs[role].model_id, "system": system, "user": user, "schema": schema,
    }
    key = "biored:" + hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()

    def factory() -> tuple[dict[str, Any], float]:
        result = registry.call_json(role, system_prompt=system, user_prompt=user, schema_hint=schema)
        return model_result_dict(result), result.latency_s

    payload, status = cache.get_or_compute(key, factory, cacheable=lambda item: item.get("status") == "OK")
    fields = StructuredModelResult.__dataclass_fields__
    return StructuredModelResult(**{key: value for key, value in payload.items() if key in fields}), status


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
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}

    per_label_metrics = {label: prf(values) for label, values in per_label.items()}
    macro_f1 = sum(item["f1"] for item in per_label_metrics.values()) / len(per_label_metrics)
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
    parser.add_argument("--cache-path", type=Path, required=True)
    args = parser.parse_args()
    documents = parse_pubtator(args.pubtator)[:args.limit]
    cfg = load_config()
    primary_key, deepseek_key, qwen_key = (get_secret(name) for name in ("gemini_api_key", "deepseek_api_key", "qwen_api_key"))
    retry_options = {
        "max_retries": max(0, int(os.environ.get("AUX_MODEL_MAX_RETRIES", "5"))),
        "retry_base_delay_s": max(0.0, float(os.environ.get("AUX_MODEL_RETRY_BASE_DELAY_S", "2"))),
        "retry_max_delay_s": max(0.0, float(os.environ.get("AUX_MODEL_RETRY_MAX_DELAY_S", "45"))),
    }
    registry = AuxModelRegistry([
        AuxModelSpec("primary", "openai", cfg.model_id, cfg.api_base, primary_key, timeout_s=120, **retry_options),
        AuxModelSpec("judge", "openai", cfg.second_llm_model_id, cfg.second_llm_api_base, deepseek_key, timeout_s=120, **retry_options),
        AuxModelSpec("critic", "openai", cfg.aux_critic_model, cfg.qwen_api_base, qwen_key, timeout_s=120, **retry_options),
    ])
    if not all(registry.configured(role) for role in ("primary", "judge", "critic")):
        raise RuntimeError("primary, DeepSeek, and Qwen credentials must all be configured")
    schema = {"relations": [{"arg1_id": "string", "arg2_id": "string", "relation_type": "BioRED label", "novelty": "Novel|No", "confidence": "0..1"}]}
    system = (
        "You perform BioRED given-entity relation extraction. Use only supplied concept IDs. "
        f"Allowed relation_type values: {', '.join(RELATION_TYPES)}. "
        "Return only direct article-supported positive relations. Do not infer co-occurrence. "
        "Novel means the article presents the relation as new; otherwise use No. Return strict JSON."
    )
    lock = threading.Lock()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    cache = LightweightExtractionCache(
        mode="persistent", path=str(args.cache_path),
        persistent_max_entries=2000, persistent_max_mb=200, ttl_days=30,
    )
    abort_event = threading.Event()
    abort_reason = {"value": ""}

    def run_document(doc: Document) -> dict[str, Any]:
        if abort_event.is_set():
            return {"pmid": doc.pmid, "status": "SKIPPED_GLOBAL_PROVIDER_FAILURE", "error": abort_reason["value"], "relations": []}
        entities = [{"id": key, "type": value.entity_type, "mentions": sorted(value.mentions)} for key, value in sorted(doc.concepts.items())]
        user = "ARTICLE:\n" + doc.text + "\n\nENTITIES:\n" + json.dumps(entities, ensure_ascii=False, separators=(",", ":"))
        primary, primary_cache = call_cached(cache, registry, "primary", system, user, schema)
        if primary.status != "OK":
            failure_category = classify_provider_error(primary.error)
            if failure_category in {AUTH_ERROR, QUOTA_OR_TOKEN_EXHAUSTED}:
                abort_reason["value"] = failure_category
                abort_event.set()
            return {"pmid": doc.pmid, "status": "PRIMARY_FAILED", "error": primary.error, "failure_category": failure_category, "primary": primary.to_dict(), "cache": primary_cache, "relations": []}
        relations = normalise_relations(primary.payload, doc.concepts)
        uncertain = [item for item in relations if item["confidence"] < 0.85]
        audits: dict[str, Any] = {"primary": primary.to_dict(), "primary_cache": primary_cache}
        if uncertain:
            judge_user = user + "\n\nPRIMARY_CANDIDATES_TO_REVIEW:\n" + json.dumps(uncertain, separators=(",", ":"))
            judge, judge_cache = call_cached(cache, registry, "judge", system, judge_user, schema)
            audits.update(judge=judge.to_dict(), judge_cache=judge_cache)
            if judge.status == "OK":
                reviewed = normalise_relations(judge.payload, doc.concepts)
                if {(*sorted((item["arg1_id"], item["arg2_id"])), item["relation_type"]) for item in reviewed} != {(*sorted((item["arg1_id"], item["arg2_id"])), item["relation_type"]) for item in uncertain}:
                    critic_user = judge_user + "\n\nDEEPSEEK_REVIEW:\n" + json.dumps(reviewed, separators=(",", ":"))
                    critic, critic_cache = call_cached(cache, registry, "critic", system, critic_user, schema)
                    audits.update(critic=critic.to_dict(), critic_cache=critic_cache)
                    if critic.status == "OK":
                        reviewed = normalise_relations(critic.payload, doc.concepts)
                by_key = {(*sorted((item["arg1_id"], item["arg2_id"])), item["relation_type"]): item for item in relations if item not in uncertain}
                by_key.update({(*sorted((item["arg1_id"], item["arg2_id"])), item["relation_type"]): item for item in reviewed})
                relations = list(by_key.values())
        return {"pmid": doc.pmid, "status": "OK", "relations": relations, "audits": audits}

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
    report = {
        "protocol": "BioRED given-entity relation and novelty extraction; no LiverKG schema mapping; no Neo4j writes",
        "split": args.pubtator.name, "documents_requested": len(documents), "documents_completed": completed,
        "run_status": run_status, "evaluation_valid": completed == len(documents),
        "failure_categories": dict(failure_categories), "usage": usage,
        "usage_unavailable_roles": usage_unavailable_roles,
        "metrics": score(records, documents), "records": records,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = report["metrics"]["pair_relation_micro"]
    (args.output_dir / "report.md").write_text(
        "# BioRED given-entity benchmark\n\n"
        f"- Status: `{report['run_status']}`\n- Completed: {report['documents_completed']}/{len(documents)}\n"
        f"- Evaluation valid: `{report['evaluation_valid']}`\n"
        f"- Provider failures: `{json.dumps(report['failure_categories'], ensure_ascii=False)}`\n"
        f"- Pair+relation micro P/R/F1: {summary['precision']:.3f}/{summary['recall']:.3f}/{summary['f1']:.3f} ({summary['tp']}/{summary['fp']}/{summary['fn']})\n"
        f"- Pair+relation+novelty micro F1: {report['metrics']['pair_relation_novelty_micro']['f1']:.3f}\n"
        f"- Positive relation macro F1: {report['metrics']['positive_relation_macro_f1']:.3f}\n",
        encoding="utf-8",
    )
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
