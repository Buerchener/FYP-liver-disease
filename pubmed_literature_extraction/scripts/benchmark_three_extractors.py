#!/usr/bin/env python3
"""Run the 50-paper, three-arm extraction benchmark without Neo4j writes.

Arms:
  original_agent: LangExtract candidates + current deterministic verifier.
  langextract_llm_hybrid: the same LangExtract call + always-on LLM correction
    + deterministic re-verification.
  deepseek_direct: one strict-JSON model call + deterministic verifier.

The gold set is relation-centric.  Entity scores therefore measure annotated
relation endpoints/core diseases, not exhaustive NER recall.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langextract.factory import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.abbreviation_detector import AbbreviationDetector
from cognitive_agent.collaborative_extractor import (
    COLLABORATION_JSON_SCHEMA,
    CollaborativeConfig,
    CollaborativeExtractor,
)
from cognitive_agent.extraction_kernel import ExtractionKernel
from cognitive_agent.extraction_quality import locate_contiguous, normalize_surface
from cognitive_agent.schema.entity_classes import ENTITY_CLASSES
from cognitive_agent.schema.relation_signatures import RELATION_SIGNATURES
from cognitive_agent.verifier import KGVerifier


ENTITY_TYPES = sorted(item["label"] for item in ENTITY_CLASSES.values())
PREDICATES = sorted(RELATION_SIGNATURES)
DIRECTIONS = ["positive", "negative", "increase", "decrease", "none", "unknown"]

DIRECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entities", "relations"],
    "properties": {
        "entities": {
            "type": "array",
            "maxItems": 60,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["mention", "type", "normalized_id", "confidence"],
                "properties": {
                    "mention": {"type": "string"},
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                    "normalized_id": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "relations": {
            "type": "array",
            "maxItems": 60,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "subject", "subject_type", "predicate", "object", "object_type",
                    "direction", "evidence", "negated", "uncertain", "species",
                    "confidence",
                ],
                "properties": {
                    "subject": {"type": "string"},
                    "subject_type": {"type": "string", "enum": ENTITY_TYPES},
                    "predicate": {"type": "string", "enum": PREDICATES},
                    "object": {"type": "string"},
                    "object_type": {"type": "string", "enum": ENTITY_TYPES},
                    "direction": {"type": "string", "enum": DIRECTIONS},
                    "evidence": {"type": "string"},
                    "negated": {"type": "boolean"},
                    "uncertain": {"type": "boolean"},
                    "species": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}


class OfflineKG:
    """Explicitly disconnected KG: evaluation must not depend on or write Neo4j."""

    is_connected = False


@dataclass
class ModelCall:
    payload: dict
    latency_s: float
    prompt_tokens: int
    output_tokens: int
    attempts: int
    invalid_json_attempts: int
    error: str = ""


def _usage_count(response: Any, field: str) -> int:
    usage = getattr(response, "usage_metadata", None)
    value = getattr(usage, field, 0) if usage is not None else 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        return "\n".join(lines[1:]).strip()
    return text


def call_json_model(
    *, provider: str, api_key: str, api_base: str, model_id: str, prompt: str,
    schema: dict, max_output_tokens: int = 8192, retries: int = 1,
) -> ModelCall:
    """Direct JSON call with measured invalid output and retry behavior."""
    started = time.perf_counter()
    invalid = 0
    last_error = ""
    prompt_tokens = 0
    output_tokens = 0
    for attempt in range(1, retries + 2):
        try:
            if provider == "openai":
                from openai import OpenAI

                client = OpenAI(api_key=api_key, base_url=api_base, timeout=120.0)
                schema_instruction = json.dumps(schema, ensure_ascii=False)
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a biomedical knowledge extraction expert. "
                                "Return strict JSON only. Your JSON must satisfy this schema: "
                                + schema_instruction
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_output_tokens,
                    response_format={"type": "json_object"},
                )
                raw_text = str(response.choices[0].message.content or "")
                usage = getattr(response, "usage", None)
                prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            else:
                from google import genai
                from google.genai import types

                kwargs: dict[str, Any] = {"api_key": api_key}
                if api_base:
                    kwargs["http_options"] = {"base_url": api_base, "timeout": 120000}
                client = genai.Client(**kwargs)
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=max_output_tokens,
                        response_mime_type="application/json",
                        response_json_schema=schema,
                    ),
                )
                raw_text = str(getattr(response, "text", "") or "")
                prompt_tokens += _usage_count(response, "prompt_token_count")
                output_tokens += _usage_count(response, "candidates_token_count")
            try:
                payload = json.loads(_strip_fence(raw_text))
                if not isinstance(payload, dict):
                    raise ValueError("top-level JSON is not an object")
                return ModelCall(
                    payload=payload,
                    latency_s=time.perf_counter() - started,
                    prompt_tokens=prompt_tokens or math.ceil(len(prompt) / 4),
                    output_tokens=output_tokens or math.ceil(len(raw_text) / 4),
                    attempts=attempt,
                    invalid_json_attempts=invalid,
                )
            except Exception as exc:
                invalid += 1
                last_error = f"invalid JSON: {exc}"
        except Exception as exc:
            last_error = str(exc)
        if attempt <= retries:
            time.sleep(min(2 ** (attempt - 1), 3))
    return ModelCall(
        payload={"entities": [], "relations": []},
        latency_s=time.perf_counter() - started,
        prompt_tokens=prompt_tokens or math.ceil(len(prompt) / 4),
        output_tokens=output_tokens,
        attempts=retries + 1,
        invalid_json_attempts=invalid,
        error=last_error[:500],
    )


def direct_prompt(text: str, pmid: str) -> str:
    signatures = {
        predicate: sorted([list(pair) for pair in pairs])
        for predicate, pairs in RELATION_SIGNATURES.items()
    }
    return f"""You are a biomedical curator extracting a liver-disease knowledge graph.
Read only the supplied title and abstract. Return strict JSON matching the response schema.

Entity types: {', '.join(ENTITY_TYPES)}.
Predicates and allowed endpoint signatures: {json.dumps(signatures, ensure_ascii=False)}.

Rules:
- Extract specific, explicitly mentioned biomedical entities; no generic method/statistical/category terms.
- Gene means the text explicitly discusses a gene; Protein means a protein/product/function. Do not swap them from background knowledge.
- Extract a relation only when the article text explicitly supports that exact subject, predicate and object.
- Do not turn background, objectives, methods, screening, docking, database prediction, speculation or negated claims into findings.
- evidence must be one verbatim continuous quote from the supplied text and cover both endpoints plus the relation trigger/direction.
- Do not paraphrase evidence and do not use outside knowledge.
- Include relation endpoints in entities. Return empty arrays when there is no supported item.

PMID: {pmid}
SOURCE TEXT:
{text[:16000]}
"""


def normalize_direct(payload: dict, text: str) -> tuple[list[dict], list[dict]]:
    entities = []
    for item in payload.get("entities", []) if isinstance(payload.get("entities"), list) else []:
        if not isinstance(item, dict):
            continue
        mention = str(item.get("mention", "") or "").strip()
        etype = str(item.get("type", "") or "")
        if not mention or etype not in ENTITY_TYPES:
            continue
        grounded, start, end = locate_contiguous(mention, text)
        entities.append({
            "mention": mention, "type": etype,
            "attributes": {"normalized_id": str(item.get("normalized_id", "") or "")},
            "confidence": item.get("confidence", 0.0), "grounded": grounded,
            "source_span": text[start:end] if grounded else "",
            "char_start": start, "char_end": end,
        })
    relations = []
    for item in payload.get("relations", []) if isinstance(payload.get("relations"), list) else []:
        if not isinstance(item, dict):
            continue
        relations.append({
            "subject": str(item.get("subject", "") or "").strip(),
            "subject_type": str(item.get("subject_type", "") or ""),
            "predicate": str(item.get("predicate", "") or "").upper(),
            "object": str(item.get("object", "") or "").strip(),
            "object_type": str(item.get("object_type", "") or ""),
            "direction": str(item.get("direction", "unknown") or "unknown").lower(),
            "evidence": str(item.get("evidence", "") or ""),
            "negated": bool(item.get("negated", False)),
            "uncertain": bool(item.get("uncertain", False)),
            "species": str(item.get("species", "") or ""),
            "confidence": item.get("confidence", 0.0),
        })
    return entities, relations


def _dedupe_count(items: list[dict], kind: str) -> int:
    seen = set()
    duplicates = 0
    for item in items:
        if kind == "entity":
            key = (normalize_surface(item.get("mention")), str(item.get("type", item.get("entity_type", ""))))
        else:
            key = (
                normalize_surface(item.get("subject")), str(item.get("subject_type", "")),
                str(item.get("predicate", "")).upper(), normalize_surface(item.get("object")),
                str(item.get("object_type", "")),
            )
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _verified_payload(verified: Any) -> dict:
    value = verified.to_dict()
    return {"entities": value["entities"], "relations": value["relations"], "summary": value["summary"]}


def process_article(
    gold: dict, source: dict, kernel: ExtractionKernel, verifier: KGVerifier,
    provider: str, api_key: str, api_base: str, model_id: str,
) -> dict:
    pmid = str(gold["pmid"])
    text = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"

    # Arm 1: current/original agent core.
    t0 = time.perf_counter()
    raw = kernel.extract(text=text, document_id=pmid)
    initial = verifier.verify(raw.entities, raw.relations, pmid=pmid, text=text)
    original_latency = time.perf_counter() - t0
    lx_prompt_tokens = math.ceil(len(text) / 4)
    lx_output_tokens = math.ceil(len(json.dumps(raw.to_dict(), ensure_ascii=False)) / 4)
    lx_invalid = int(bool(raw.error) or any("parse" in str(x).casefold() for x in raw.warnings))

    # Arm 2: same LangExtract candidates, second LLM correction, re-verification.
    hybrid_started = time.perf_counter()
    hybrid_calls: list[ModelCall] = []

    def hybrid_generate(prompt: str) -> str:
        call = call_json_model(
            provider=provider, api_key=api_key, api_base=api_base,
            model_id=model_id, prompt=prompt,
            schema=COLLABORATION_JSON_SCHEMA, max_output_tokens=8192,
        )
        hybrid_calls.append(call)
        if call.error:
            raise RuntimeError(call.error)
        return json.dumps(call.payload, ensure_ascii=False)

    collaborator = CollaborativeExtractor(
        CollaborativeConfig(enabled=True, model_id=model_id, mode="always"),
        generate=hybrid_generate,
    )
    collaboration = collaborator.collaborate(
        text=text, extraction=raw.to_dict(), verification=initial.to_dict(), pmid=pmid,
    )
    merged = collaborator.merge(
        raw_entities=raw.entities, raw_relations=raw.relations,
        initial_verification=initial.to_dict(), collaboration=collaboration,
    )
    hybrid_verified = verifier.verify(merged.entities, merged.relations, pmid=pmid, text=text)
    hybrid_extra_latency = time.perf_counter() - hybrid_started
    h_call = hybrid_calls[0] if hybrid_calls else ModelCall({}, hybrid_extra_latency, 0, 0, 0, 0, collaboration.error)

    # Arm 3: strict JSON direct extraction, no LangExtract.
    prompt = direct_prompt(text, pmid)
    direct_call = call_json_model(
        provider=provider, api_key=api_key, api_base=api_base,
        model_id=model_id, prompt=prompt,
        schema=DIRECT_SCHEMA, max_output_tokens=8192,
    )
    direct_entities, direct_relations = normalize_direct(direct_call.payload, text)
    direct_verify_started = time.perf_counter()
    direct_verified = verifier.verify(direct_entities, direct_relations, pmid=pmid, text=text)
    direct_verify_latency = time.perf_counter() - direct_verify_started

    def arm(raw_entities: list[dict], raw_relations: list[dict], verified: Any, **meta: Any) -> dict:
        return {
            "raw_entities": raw_entities,
            "raw_relations": raw_relations,
            "prediction": _verified_payload(verified),
            "candidate_counts": {
                "entities": len(raw_entities), "relations": len(raw_relations),
                "duplicate_entities": _dedupe_count(raw_entities, "entity"),
                "duplicate_relations": _dedupe_count(raw_relations, "relation"),
            },
            **meta,
        }

    return {
        "pmid": pmid,
        "title": source.get("title", ""),
        "arms": {
            "original_agent": arm(
                raw.entities, raw.relations, initial,
                latency_s=original_latency, prompt_tokens=lx_prompt_tokens,
                output_tokens=lx_output_tokens, attempts=1 + raw.retry_count,
                invalid_json_attempts=lx_invalid, error=raw.error,
            ),
            "langextract_llm_hybrid": arm(
                merged.entities, merged.relations, hybrid_verified,
                latency_s=original_latency + hybrid_extra_latency,
                prompt_tokens=lx_prompt_tokens + h_call.prompt_tokens,
                output_tokens=lx_output_tokens + h_call.output_tokens,
                attempts=(1 + raw.retry_count) + h_call.attempts,
                invalid_json_attempts=lx_invalid + h_call.invalid_json_attempts,
                error=collaboration.error,
                collaboration=collaboration.to_dict(), merge=merged.to_dict(),
            ),
            "deepseek_direct": arm(
                direct_entities, direct_relations, direct_verified,
                latency_s=direct_call.latency_s + direct_verify_latency,
                prompt_tokens=direct_call.prompt_tokens, output_tokens=direct_call.output_tokens,
                attempts=direct_call.attempts,
                invalid_json_attempts=direct_call.invalid_json_attempts,
                error=direct_call.error,
            ),
        },
    }


def _aliases(gold: dict, text: str) -> dict[str, set[tuple[str, str]]]:
    mapping: dict[str, set[tuple[str, str]]] = {}
    entities = gold.get("entities", [])
    for ent in entities:
        canonical = normalize_surface(ent.get("canonical", ent.get("mention", "")))
        typed = (canonical, str(ent.get("type", "")))
        for value in (ent.get("mention", ""), ent.get("canonical", "")):
            mapping.setdefault(normalize_surface(value), set()).add(typed)
    detected = AbbreviationDetector().detect(text)
    for short, long_form in detected.abbr_to_long.items():
        short_n, long_n = normalize_surface(short), normalize_surface(long_form)
        for typed in mapping.get(short_n, set()) | mapping.get(long_n, set()):
            mapping.setdefault(short_n, set()).add(typed)
            mapping.setdefault(long_n, set()).add(typed)
    return mapping


def _canonical_entity(item: dict, aliases: dict[str, set[tuple[str, str]]]) -> tuple[str, str]:
    surface = normalize_surface(item.get("mention", ""))
    etype = str(item.get("type", item.get("entity_type", "")))
    same_type = [key for key in aliases.get(surface, set()) if key[1] == etype]
    return same_type[0] if len(same_type) == 1 else (surface, etype)


def _canonical_endpoint(value: str, etype: str, aliases: dict[str, set[tuple[str, str]]]) -> str:
    surface = normalize_surface(value)
    same_type = [key[0] for key in aliases.get(surface, set()) if key[1] == etype]
    return same_type[0] if len(same_type) == 1 else surface


def _relation_key(item: dict, aliases: dict[str, set[tuple[str, str]]]) -> tuple[str, str, str]:
    return (
        _canonical_endpoint(str(item.get("subject", "")), str(item.get("subject_type", "")), aliases),
        str(item.get("predicate", "")).upper(),
        _canonical_endpoint(str(item.get("object", "")), str(item.get("object_type", "")), aliases),
    )


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def evaluate(results: list[dict], gold_by_pmid: dict[str, dict], prices: tuple[float, float]) -> dict:
    arms = next(iter(results))["arms"].keys()
    output = {}
    for arm_name in arms:
        ent_tp = ent_fp = ent_fn = 0
        rel_tp = rel_fp = rel_fn = 0
        strict_tp = strict_fp = strict_fn = 0
        gene_protein_correct = gene_protein_matched = 0
        evidence_correct = evidence_predictions = source_contiguous = 0
        cand_total = dup_total = 0
        latencies: list[float] = []
        prompt_tokens = output_tokens = attempts = invalid = failed = 0
        for result in results:
            pmid = result["pmid"]
            gold = gold_by_pmid[pmid]
            source_text = f"TITLE: {gold.get('title', '')}\nABSTRACT: {gold.get('abstract', '')}"
            aliases = _aliases(gold, source_text)
            arm = result["arms"][arm_name]
            pred = arm["prediction"]

            gold_entities = {
                (normalize_surface(e.get("canonical", e.get("mention", ""))), str(e.get("type", "")))
                for e in gold.get("entities", [])
            }
            pred_entities = {_canonical_entity(e, aliases) for e in pred.get("entities", [])}
            ent_tp += len(gold_entities & pred_entities)
            ent_fp += len(pred_entities - gold_entities)
            ent_fn += len(gold_entities - pred_entities)

            for entity in pred.get("entities", []):
                surface = normalize_surface(entity.get("mention", ""))
                molecular = [x for x in aliases.get(surface, set()) if x[1] in {"Gene", "Protein"}]
                if molecular:
                    gene_protein_matched += 1
                    gene_protein_correct += int(str(entity.get("type", "")) in {x[1] for x in molecular})

            gold_relations = {_relation_key(r, aliases) for r in gold.get("relations", [])}
            gold_strict = {_relation_key(r, aliases) for r in gold.get("relations", []) if r.get("import_ready")}
            pred_relations = {_relation_key(r, aliases) for r in pred.get("relations", [])}
            pred_strict = {_relation_key(r, aliases) for r in pred.get("relations", []) if r.get("import_ready")}
            rel_tp += len(gold_relations & pred_relations)
            rel_fp += len(pred_relations - gold_relations)
            rel_fn += len(gold_relations - pred_relations)
            strict_tp += len(gold_strict & pred_strict)
            strict_fp += len(pred_strict - gold_strict)
            strict_fn += len(gold_strict - pred_strict)

            gold_by_key = {}
            for relation in gold.get("relations", []):
                gold_by_key.setdefault(_relation_key(relation, aliases), []).append(relation)
            for relation in pred.get("relations", []):
                evidence_predictions += 1
                evidence = str(relation.get("evidence", "") or "")
                contiguous, ps, pe = locate_contiguous(evidence, source_text)
                source_contiguous += int(contiguous)
                if not contiguous:
                    continue
                key = _relation_key(relation, aliases)
                matched = False
                for gold_relation in gold_by_key.get(key, []):
                    ok, gs, ge = locate_contiguous(str(gold_relation.get("evidence", "")), source_text)
                    if not ok:
                        continue
                    intersection = max(0, min(pe, ge) - max(ps, gs))
                    union = max(pe, ge) - min(ps, gs)
                    if union and intersection / union >= 0.5:
                        matched = True
                        break
                evidence_correct += int(matched)

            counts = arm["candidate_counts"]
            cand_total += counts["entities"] + counts["relations"]
            dup_total += counts["duplicate_entities"] + counts["duplicate_relations"]
            latencies.append(float(arm.get("latency_s", 0.0)))
            prompt_tokens += int(arm.get("prompt_tokens", 0))
            output_tokens += int(arm.get("output_tokens", 0))
            attempts += int(arm.get("attempts", 0))
            invalid += int(arm.get("invalid_json_attempts", 0))
            failed += int(bool(arm.get("error")))

        input_price, output_price = prices
        output[arm_name] = {
            "entity_endpoint_core_disease": _prf(ent_tp, ent_fp, ent_fn),
            "gene_protein_type_accuracy": gene_protein_correct / gene_protein_matched if gene_protein_matched else None,
            "gene_protein_type_counts": {"correct": gene_protein_correct, "matched": gene_protein_matched},
            "semantic_relation_triples": _prf(rel_tp, rel_fp, rel_fn),
            "strict_import_ready_relation_triples": _prf(strict_tp, strict_fp, strict_fn),
            "evidence_span_precision_iou_0_5": evidence_correct / evidence_predictions if evidence_predictions else 1.0,
            "evidence_source_contiguous_rate": source_contiguous / evidence_predictions if evidence_predictions else 1.0,
            "evidence_counts": {"correct": evidence_correct, "predicted_relations": evidence_predictions},
            "duplicate_candidate_rate": dup_total / cand_total if cand_total else 0.0,
            "duplicate_counts": {"duplicates": dup_total, "candidates": cand_total},
            "latency_seconds": {
                "mean": statistics.mean(latencies) if latencies else 0.0,
                "p50": _percentile(latencies, 0.50), "p95": _percentile(latencies, 0.95),
                "sum_article_latency": sum(latencies),
            },
            "usage_and_cost": {
                "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
                "estimated_cost_usd": (prompt_tokens * input_price + output_tokens * output_price) / 1_000_000,
                "price_reference_usd_per_million": {"input": input_price, "output": output_price},
                "cost_kind": "reference_estimate_not_proxy_invoice",
            },
            "invalid_json_attempt_rate": invalid / attempts if attempts else 0.0,
            "invalid_json_attempts": invalid, "model_attempts": attempts,
            "article_failure_rate": failed / len(results), "failed_articles": failed,
        }
    return output


def markdown_report(metrics: dict, manifest: dict) -> str:
    rows = []
    for name, m in metrics.items():
        e, r, s = m["entity_endpoint_core_disease"], m["semantic_relation_triples"], m["strict_import_ready_relation_triples"]
        rows.append(
            f"| {name} | {e['precision']:.3f}/{e['recall']:.3f}/{e['f1']:.3f} | "
            f"{(m['gene_protein_type_accuracy'] if m['gene_protein_type_accuracy'] is not None else 0):.3f} | "
            f"{r['precision']:.3f}/{r['recall']:.3f}/{r['f1']:.3f} | "
            f"{s['precision']:.3f}/{s['recall']:.3f}/{s['f1']:.3f} | "
            f"{m['evidence_span_precision_iou_0_5']:.3f} | {m['duplicate_candidate_rate']:.3f} | "
            f"{m['latency_seconds']['p50']:.1f}/{m['latency_seconds']['p95']:.1f} | "
            f"${m['usage_and_cost']['estimated_cost_usd']:.4f} | {m['invalid_json_attempt_rate']:.3f} |"
        )
    return f"""# Three-extractor benchmark

- Run: `{manifest['run_id']}`
- Documents: {manifest['documents']}
- Model: `{manifest['model_id']}`
- Workers: {manifest['max_workers']}
- Gold: relation-centric (45 semantic relations; 22 import-ready relations).
- Cost: reference estimate, not the third-party proxy invoice.

| Arm | Entity P/R/F1* | Gene/Protein type acc. | Semantic relation P/R/F1 | Import-ready relation P/R/F1 | Evidence precision | Duplicate rate | Latency p50/p95 s | Est. cost | Invalid JSON rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

* Entity labels contain relation endpoints plus core diseases; they are not exhaustive NER labels.
Evidence precision requires a matched triple, continuous source evidence, and character-span IoU >= 0.5 with gold evidence.
"""


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=ROOT / "gold_annotations/pubmed_50_gold_v1.jsonl")
    parser.add_argument("--source", type=Path, default=ROOT / "extraction_output/pubmed_converted_500.jsonl")
    parser.add_argument("--output-root", type=Path, default=ROOT / "benchmark_output")
    parser.add_argument("--run-id", default=f"three_extractors_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--provider", choices=("gemini", "openai"), default=os.environ.get("BENCHMARK_PROVIDER", "openai"))
    parser.add_argument("--model-id", default=os.environ.get("BENCHMARK_MODEL_ID", os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")))
    parser.add_argument("--api-base", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--input-price-per-million", type=float, default=0.14)
    parser.add_argument("--output-price-per-million", type=float, default=0.28)
    args = parser.parse_args()

    api_key = (
        os.environ.get("DEEPSEEK_API_KEY", "")
        if args.provider == "openai"
        else os.environ.get("GEMINI_API_KEY", "") or os.environ.get("LLM_API_KEY", "")
    )
    if not api_key:
        print(f"[ERROR] API key is not configured for provider={args.provider}", flush=True)
        return 2
    gold_rows = load_jsonl(args.gold)
    source_rows = load_jsonl(args.source)
    source_by_pmid = {str(row["pmid"]): row for row in source_rows}
    if len(gold_rows) != 50 or any(str(row["pmid"]) not in source_by_pmid for row in gold_rows):
        print("[ERROR] gold/source PMID alignment failed", flush=True)
        return 2

    run_dir = args.output_root / args.run_id
    items_dir = run_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": args.run_id, "started_at": datetime.now(timezone.utc).isoformat(),
        "documents": len(gold_rows), "max_workers": args.max_workers,
        "provider": args.provider, "model_id": args.model_id, "api_base": args.api_base,
        "gold_path": str(args.gold.resolve()), "source_path": str(args.source.resolve()),
        "neo4j_reads": False, "neo4j_writes": False,
        "arm_definitions": {
            "original_agent": "LangExtract candidates + current deterministic verifier; Phase B/C off",
            "langextract_llm_hybrid": "same LangExtract call + always-on LLM correction + deterministic re-verification",
            "deepseek_direct": "strict JSON direct DeepSeek extraction without LangExtract + deterministic verifier",
        },
        "cost_note": "Reference estimate using configured per-token prices; not a proxy invoice.",
    }
    atomic_json(run_dir / "manifest.json", manifest)
    print(f"[START] run={args.run_id} docs=50 workers={args.max_workers} model={args.model_id}", flush=True)

    provider_kwargs: dict[str, Any] = {"api_key": api_key, "temperature": 0.0}
    if args.provider == "openai":
        provider_kwargs["base_url"] = args.api_base
    else:
        provider_kwargs["http_options"] = {"base_url": args.api_base}
    model_config = ModelConfig(
        provider=args.provider, model_id=args.model_id,
        provider_kwargs=provider_kwargs,
    )
    kernel = ExtractionKernel(model_config)
    verifier = KGVerifier(OfflineKG())
    results: list[dict] = []
    lock = threading.Lock()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_article, gold, source_by_pmid[str(gold["pmid"])], kernel, verifier,
                args.provider, api_key, args.api_base, args.model_id,
            ): gold
            for gold in gold_rows
        }
        for future in as_completed(futures):
            gold = futures[future]
            pmid = str(gold["pmid"])
            try:
                result = future.result()
            except Exception as exc:
                result = {"pmid": pmid, "fatal_error": str(exc)[:1000], "arms": {}}
            with lock:
                (items_dir / f"{pmid}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                results.append(result)
                completed += 1
                atomic_json(run_dir / "progress.json", {
                    "completed": completed, "total": len(gold_rows), "last_pmid": pmid,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                print(f"[PROGRESS] {completed}/50 PMID={pmid}", flush=True)

    valid = [item for item in results if item.get("arms")]
    gold_eval = {}
    for gold in gold_rows:
        source = source_by_pmid[str(gold["pmid"])]
        gold_eval[str(gold["pmid"])] = {**gold, "abstract": source.get("abstract", "")}
    metrics = evaluate(
        valid, gold_eval,
        (args.input_price_per_million, args.output_price_per_million),
    ) if valid else {}
    results.sort(key=lambda item: next((i for i, g in enumerate(gold_rows) if str(g["pmid"]) == item["pmid"]), 999))
    atomic_json(run_dir / "predictions.json", results)
    atomic_json(run_dir / "metrics.json", metrics)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_documents"] = len(valid)
    manifest["fatal_failures"] = len(results) - len(valid)
    atomic_json(run_dir / "manifest.json", manifest)
    (run_dir / "metrics.md").write_text(markdown_report(metrics, manifest), encoding="utf-8")
    print(f"[DONE] completed={len(valid)}/50 report={run_dir / 'metrics.md'}", flush=True)
    return 0 if len(valid) == 50 else 1


if __name__ == "__main__":
    raise SystemExit(main())
