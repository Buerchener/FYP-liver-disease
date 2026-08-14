#!/usr/bin/env python3
"""Deterministically replay legacy and shadow routing on saved extraction state."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognitive_agent.tool_router import ArticleToolRouter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--source", type=Path,
        default=ROOT / "extraction_output/pubmed_converted_500.jsonl",
    )
    ap.add_argument("--memory-available", action="store_true")
    args = ap.parse_args()
    run = json.loads(args.run.read_text())
    source_by_pmid = {}
    if args.source.exists():
        for line in args.source.read_text().splitlines():
            if line.strip():
                article = json.loads(line)
                source_by_pmid[str(article.get("pmid", ""))] = article
    router = ArticleToolRouter()
    rows = []
    for record in run.get("records", []):
        phases = record.get("phases", {})
        article = source_by_pmid.get(str(record.get("pmid", "")), {})
        title = article.get("title", record.get("title", ""))
        abstract = article.get("abstract", "")
        legacy_pre_payload = phases.get("tool_plan_pre", {})
        old_shadow = phases.get("shadow_tool_plan_pre", {})
        if not old_shadow:
            continue
        from cognitive_agent.tool_router import ArticleProfile, ToolDecision, ToolPlan
        def restore(payload: dict) -> ToolPlan:
            p = payload["profile"]
            profile = ArticleProfile(
                study_type=p.get("study_type", "other"), char_count=p.get("char_count", 0),
                sentence_count=p.get("sentence_count", 0), max_sentence_words=p.get("max_sentence_words", 0),
                entity_signal_count=p.get("entity_signal_count", 0), mechanistic_signal_count=p.get("mechanistic_signal_count", 0),
                evidence_signal_count=p.get("evidence_signal_count", 0), has_structured_results=p.get("has_structured_results", False),
                high_complexity=p.get("high_complexity", False), reason_codes=tuple(p.get("reason_codes", [])),
                secondary_modalities=tuple(p.get("secondary_modalities", [])), species_scope=p.get("species_scope", "unclear"),
                evidence_design=p.get("evidence_design", "unclear"), causal_strength=p.get("causal_strength", "unclear"),
                validation_level=p.get("validation_level", "unclear"), profile_confidence=p.get("profile_confidence", 0.0),
                profile_source=p.get("profile_source", "rules"), complexity_vector=p.get("complexity_vector", {}),
            )
            plan = ToolPlan(
                payload.get("stage", "post_verification"), payload.get("route", "STANDARD"), profile,
                reason_codes=payload.get("reason_codes", []),
                plan_status=payload.get("plan_status", "production"),
                legacy_route=payload.get("legacy_route", ""),
                early_stop_reasons=payload.get("early_stop_reasons", []),
                routing_version=payload.get("routing_version", "legacy"),
                layer_trace=payload.get("four_layer_trace", {}),
            )
            plan.decisions = {
                d.get("tool", "unknown"): ToolDecision(
                    d.get("tool", "unknown"), d.get("decision", "SKIP"), d.get("reason", ""),
                    d.get("cost_class", "low"), d.get("expected_value", 0.0),
                    d.get("budget_level", "minimal"), d.get("plan_status", "production"),
                    d.get("hard_masked", False), d.get("candidate_pool", False),
                    d.get("utility", {}),
                ) for d in payload.get("tools", [])
            }
            return plan
        legacy_post = restore(phases.get("tool_plan_post", legacy_pre_payload))
        if abstract:
            legacy_pre = restore(legacy_pre_payload)
            shadow_pre = router.shadow_plan_before_extraction(
                title, abstract, legacy_plan=legacy_pre,
                memory_available=args.memory_available, rag_enabled=True,
                second_llm_enabled=True,
            )
        else:
            shadow_pre = restore(old_shadow)
        post = router.shadow_plan_after_verification(
            shadow_pre, legacy_post, phases.get("extraction", {}), phases.get("verification", {}),
            memory_available=args.memory_available, rag_enabled=True, second_llm_enabled=True,
            recovery_candidate_count=len(phases.get("agent_controller", {}).get("recovery_candidates", []) or []),
        )
        rows.append({"pmid": record.get("pmid"), "legacy": legacy_post.to_dict(), "shadow": post.to_dict()})
    route_counts = Counter(r["shadow"]["route"] for r in rows)
    tool_counts = Counter(t for r in rows for t in r["shadow"].get("called_tools", []))
    legacy_route_counts = Counter(r["legacy"]["route"] for r in rows)
    legacy_tool_counts = Counter(t for r in rows for t in r["legacy"].get("called_tools", []))
    mask_counts = Counter(
        tool for row in rows for tool in (
            row["shadow"].get("four_layer_trace", {})
            .get("layer_1_safety_mask", {}).get("masked_tools", {})
        )
    )
    candidate_counts = Counter(
        tool for row in rows for tool, selected in (
            row["shadow"].get("four_layer_trace", {})
            .get("layer_2_candidate_pool", {}).get("tools", {})
        ).items() if selected
    )
    optional_tools = {
        "context_memory", "article_chunker", "neo4j_rag", "second_llm_refiner",
        "relation_recovery", "causal_reasoner", "conflict_resolver",
    }
    legacy_optional = sum(legacy_tool_counts.get(tool, 0) for tool in optional_tools)
    shadow_optional = sum(tool_counts.get(tool, 0) for tool in optional_tools)
    remote_or_multiwindow = {"article_chunker", "neo4j_rag", "second_llm_refiner"}
    legacy_expensive = sum(legacy_tool_counts.get(tool, 0) for tool in remote_or_multiwindow)
    shadow_expensive = sum(tool_counts.get(tool, 0) for tool in remote_or_multiwindow)
    payload = {
        "routing_version": ArticleToolRouter.ROUTING_VERSION,
        "documents": len(rows), "route_counts": dict(route_counts),
        "deep_rate": route_counts["DEEP"] / max(len(rows), 1),
        "tool_call_counts": dict(tool_counts),
        "legacy_route_counts": dict(legacy_route_counts),
        "legacy_tool_call_counts": dict(legacy_tool_counts),
        "safety_mask_counts": dict(mask_counts),
        "candidate_pool_counts": dict(candidate_counts),
        "counterfactual_call_reduction": {
            "optional_calls_legacy": legacy_optional,
            "optional_calls_four_layer": shadow_optional,
            "optional_reduction_rate": round(
                (legacy_optional - shadow_optional) / max(legacy_optional, 1), 4
            ),
            "remote_or_multiwindow_calls_legacy": legacy_expensive,
            "remote_or_multiwindow_calls_four_layer": shadow_expensive,
            "remote_or_multiwindow_reduction_rate": round(
                (legacy_expensive - shadow_expensive) / max(legacy_expensive, 1), 4
            ),
            "measurement_note": (
                "offline counterfactual plan replay; not measured API latency or quality"
            ),
        },
        "production_execution_unchanged": True,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
