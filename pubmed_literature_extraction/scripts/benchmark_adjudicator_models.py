#!/usr/bin/env python3
"""Compare second-stage adjudicators on one fixed 50-document candidate set.

The upstream LangExtract output is loaded from an existing agent run and never
regenerated.  Every model therefore receives the same entities, relations,
evidence units, deterministic verification, and bounded recovery lattice.
Neo4j is disabled: this benchmark measures the adjudicator, not changing graph
state or graph-retrieval coverage.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.abbreviation_detector import AbbreviationDetector
from cognitive_agent.agentic_controller import AgenticArticleController
from cognitive_agent.collaborative_extractor import (
    CollaborativeConfig,
    CollaborativeExtractor,
)
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.verifier import KGVerifier
from scripts.benchmark_three_extractors import _dedupe_count, evaluate, load_jsonl


class OfflineKG:
    is_connected = False


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    api_base: str
    api_key: str


@dataclass
class PreparedArticle:
    pmid: str
    title: str
    text: str
    raw_entities: list[dict]
    raw_relations: list[dict]
    repaired_relations: list[dict]
    initial_verification: dict
    recovery_candidates: list[dict]
    rag_context: dict


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low, high = int(index), min(int(index) + 1, len(values) - 1)
    weight = index - low
    return values[low] * (1 - weight) + values[high] * weight


def prepare_articles(run_path: Path, source_path: Path) -> list[PreparedArticle]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    source_by_pmid = {str(item["pmid"]): item for item in load_jsonl(source_path)}
    verifier = KGVerifier(OfflineKG())
    detector = AbbreviationDetector()
    reader = ArticleEvidenceReader()
    controller = AgenticArticleController(max_recovery_candidates=12, max_repairs=16)
    prepared: list[PreparedArticle] = []
    records = sorted(run.get("records", []), key=lambda item: item.get("_batch_index", 9999))
    for record in records:
        pmid = str(record.get("pmid", ""))
        source = source_by_pmid[pmid]
        title = str(source.get("title", "") or "")
        text = f"TITLE: {title}\nABSTRACT: {source.get('abstract', '')}"
        extraction = record.get("phases", {}).get("extraction", {})
        entities = extraction.get("entities", []) or []
        relations = extraction.get("relations", []) or []
        units = reader.read(text)
        abbreviations = detector.detect(text)
        plan = controller.plan(text, entities, relations, abbreviations, units)
        initial = verifier.verify(entities, plan.repaired_relations, pmid=pmid, text=text)
        controller.add_recovery_observation(
            plan=plan,
            text=text,
            verified_entities=[item.to_dict() for item in initial.entities],
            verified_relations=[item.to_dict() for item in initial.relations],
            abbr_map=abbreviations,
            units=units,
        )
        prepared.append(PreparedArticle(
            pmid=pmid,
            title=title,
            text=text,
            raw_entities=entities,
            raw_relations=relations,
            repaired_relations=plan.repaired_relations,
            initial_verification=initial.to_dict(),
            recovery_candidates=plan.recovery_candidates,
            rag_context=record.get("phases", {}).get("rag_context", {}) or {},
        ))
    return prepared


def adjudicate(article: PreparedArticle, spec: ModelSpec) -> dict:
    started = time.perf_counter()
    verifier = KGVerifier(OfflineKG())
    extractor = CollaborativeExtractor(CollaborativeConfig(
        enabled=True,
        provider="openai",
        api_key=spec.api_key,
        api_base=spec.api_base,
        model_id=spec.model_id,
        mode="conditional",
        timeout=90.0,
        max_output_tokens=None,
        thinking_enabled=False,
    ))
    collaboration = extractor.collaborate(
        text=article.text,
        extraction={
            "entities": article.raw_entities,
            "relations": article.raw_relations,
            "error": "",
            "warnings": [],
        },
        verification=article.initial_verification,
        pmid=article.pmid,
        rag_context={
            "usage_policy": article.rag_context.get("usage_policy", {}),
            "entity_contexts": article.rag_context.get("entity_contexts", []),
        },
        recovery_candidates=article.recovery_candidates,
    )
    merged = extractor.merge(
        raw_entities=article.raw_entities,
        raw_relations=article.repaired_relations,
        initial_verification=article.initial_verification,
        collaboration=collaboration,
    )
    verified = verifier.verify(
        merged.entities, merged.relations, pmid=article.pmid, text=article.text
    )
    finalization_passes: list[dict] = []
    for pass_index in range(2):
        relations, audit = extractor.finalize_after_reverification(
            merged.relations, verified.to_dict(), source_text=article.text
        )
        audit["pass"] = pass_index + 1
        changed = bool(
            audit["rolled_back_count"]
            or audit["duplicate_relations_removed"]
            or audit["symmetric_orientation_changes"]
        )
        finalization_passes.append(audit)
        if not changed:
            break
        merged.relations = relations
        verified = verifier.verify(
            merged.entities, merged.relations, pmid=article.pmid, text=article.text
        )

    entities, relations = article.raw_entities, article.raw_relations
    return {
        "pmid": article.pmid,
        "title": article.title,
        "prediction": {
            "entities": verified.to_dict().get("entities", []),
            "relations": verified.to_dict().get("relations", []),
            "summary": verified.to_dict().get("summary", {}),
        },
        "candidate_counts": {
            "entities": len(entities),
            "relations": len(relations),
            "duplicate_entities": _dedupe_count(entities, "entity"),
            "duplicate_relations": _dedupe_count(relations, "relation"),
        },
        "latency_s": time.perf_counter() - started,
        "prompt_tokens": collaboration.prompt_tokens,
        "output_tokens": collaboration.output_tokens,
        "attempts": int(collaboration.triggered),
        "invalid_json_attempts": collaboration.invalid_json_attempts,
        "error": collaboration.error,
        "status": collaboration.status,
        "triggered": collaboration.triggered,
        "review_decision_count": len(collaboration.review_decisions),
        "relation_additions": merged.relation_additions,
        "relation_edits": merged.relation_edits,
        "relation_rejections": merged.relation_rejections,
        "finalization_passes": finalization_passes,
    }


def model_specs(names: list[str]) -> list[ModelSpec]:
    aliyun_key = os.environ.get("ALIYUN_MAAS_API_KEY", "")
    aliyun_base = os.environ.get("ALIYUN_MAAS_API_BASE", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    specs: list[ModelSpec] = []
    for name in names:
        if name == "deepseek-v4-flash":
            specs.append(ModelSpec(name, name, "https://api.deepseek.com", deepseek_key))
        else:
            specs.append(ModelSpec(name, name, aliyun_base, aliyun_key))
    missing = [item.name for item in specs if not item.api_key or not item.api_base]
    if missing:
        raise SystemExit(f"missing local API configuration for: {', '.join(missing)}")
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-run",
        type=Path,
        default=ROOT / "extraction_output/agent_results_agent_arch_v4_gold50_20260812.json",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "extraction_output/pubmed_converted_500.jsonl",
    )
    parser.add_argument(
        "--gold", type=Path,
        default=ROOT / "gold_annotations/pubmed_50_gold_v1.jsonl",
    )
    parser.add_argument(
        "--models",
        default=(
            "deepseek-v4-flash,qwen-flash,qwen3.5-flash,"
            "qwen3-30b-a3b-instruct-2507"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--run-id", default=f"adjudicator_models_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_output")
    args = parser.parse_args()

    names = [item.strip() for item in args.models.split(",") if item.strip()]
    specs = model_specs(names)
    prepared = prepare_articles(args.input_run, args.source)
    if len(prepared) != 50:
        raise SystemExit(f"expected 50 fixed articles, found {len(prepared)}")

    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    all_by_pmid: dict[str, dict] = {
        item.pmid: {"pmid": item.pmid, "title": item.title, "arms": {}}
        for item in prepared
    }
    model_wall: dict[str, float] = {}
    for spec in specs:
        print(f"[MODEL START] {spec.name}", flush=True)
        started = time.perf_counter()
        completed = 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(adjudicate, article, spec): article for article in prepared}
            for future in as_completed(futures):
                article = futures[future]
                try:
                    arm = future.result()
                except Exception as exc:
                    arm = {
                        "prediction": {"entities": [], "relations": [], "summary": {}},
                        "candidate_counts": {
                            "entities": len(article.raw_entities),
                            "relations": len(article.raw_relations),
                            "duplicate_entities": _dedupe_count(article.raw_entities, "entity"),
                            "duplicate_relations": _dedupe_count(article.raw_relations, "relation"),
                        },
                        "latency_s": 0.0, "prompt_tokens": 0, "output_tokens": 0,
                        "attempts": 1, "invalid_json_attempts": 0,
                        "error": str(exc)[:500], "status": "EXCEPTION", "triggered": True,
                    }
                all_by_pmid[article.pmid]["arms"][spec.name] = arm
                completed += 1
                if completed % 10 == 0:
                    print(f"[MODEL PROGRESS] {spec.name} {completed}/50", flush=True)
        model_wall[spec.name] = time.perf_counter() - started
        print(f"[MODEL DONE] {spec.name} {model_wall[spec.name]:.1f}s", flush=True)

    results = [all_by_pmid[item.pmid] for item in prepared]
    source_by_pmid = {str(item["pmid"]): item for item in load_jsonl(args.source)}
    gold_by_pmid = {
        str(item["pmid"]): {
            **item,
            "abstract": source_by_pmid[str(item["pmid"])].get("abstract", ""),
        }
        for item in load_jsonl(args.gold)
    }
    metrics = evaluate(results, gold_by_pmid, (0.0, 0.0))
    for name, value in metrics.items():
        value["latency_seconds"]["wall_clock"] = model_wall[name]
        value["usage_and_cost"]["estimated_cost_usd"] = None
        value["usage_and_cost"]["cost_kind"] = "workspace_price_not_available"

    manifest = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": len(prepared),
        "max_workers": args.max_workers,
        "input_run": str(args.input_run.resolve()),
        "fixed_upstream_candidates": True,
        "neo4j_reads": False,
        "neo4j_writes": False,
        "thinking_enabled": False,
        "models": names,
        "comparison_note": (
            "All models adjudicated identical stored LangExtract candidates. "
            "Costs are omitted because custom MaaS workspace pricing was not exposed."
        ),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    header = (
        "| Model | Entity P/R/F1 | Gene/Protein | Relation P/R/F1 | "
        "Import-ready P/R/F1 | Evidence | Duplicate | p50/p95 | Wall | Tokens in/out | "
        "Invalid JSON | Failures |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for name in names:
        value = metrics[name]
        entity = value["entity_endpoint_core_disease"]
        relation = value["semantic_relation_triples"]
        strict = value["strict_import_ready_relation_triples"]
        latency = value["latency_seconds"]
        usage = value["usage_and_cost"]
        rows.append(
            f"| {name} | {entity['precision']:.3f}/{entity['recall']:.3f}/{entity['f1']:.3f} | "
            f"{(value['gene_protein_type_accuracy'] or 0):.3f} | "
            f"{relation['precision']:.3f}/{relation['recall']:.3f}/{relation['f1']:.3f} | "
            f"{strict['precision']:.3f}/{strict['recall']:.3f}/{strict['f1']:.3f} | "
            f"{value['evidence_span_precision_iou_0_5']:.3f} | "
            f"{value['duplicate_candidate_rate']:.3f} | "
            f"{latency['p50']:.2f}/{latency['p95']:.2f} | {latency['wall_clock']:.1f}s | "
            f"{usage['prompt_tokens']}/{usage['output_tokens']} | "
            f"{value['invalid_json_attempts']} | {value['failed_articles']} |"
        )
    report = (
        f"# Fixed-candidate adjudicator benchmark\n\n"
        f"- Run: `{args.run_id}`\n"
        f"- Documents: 50\n"
        f"- Workers per model: {args.max_workers}\n"
        f"- Upstream candidates fixed: yes\n"
        f"- Neo4j reads/writes: no/no\n"
        f"- Cost: unavailable for the custom MaaS workspace; tokens are reported.\n\n"
        + header + "\n" + "\n".join(rows) + "\n"
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"[DONE] {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
