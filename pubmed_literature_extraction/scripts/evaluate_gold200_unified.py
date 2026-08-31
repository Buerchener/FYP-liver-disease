#!/usr/bin/env python3
"""Score a 200-document LiverKG run under one explicit metric contract.

The gold annotation is relation-centric, so relation-level TN is undefined:
the negative relation universe is not enumerated.  This scorer therefore uses
the standard positive-class typed-triple P/R/F1 for relation extraction and
reports TN only for the separately defined document-level relation-presence
task.  Candidate, main-KG-contract and safe-write targets are kept separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.abbreviation_detector import AbbreviationDetector
from cognitive_agent.extraction_quality import normalize_surface
from cognitive_agent.relation_contract import normalize_relation_semantics


GOLD_VIEWS = {
    "candidate_semantic": ROOT / "gold_annotations/pubmed_200_gold_v2_candidate_view.jsonl",
    "main_kg_write_contract": ROOT / "gold_annotations/pubmed_200_gold_v2_write_contract.jsonl",
    "strict_import_ready": ROOT / "gold_annotations/pubmed_200_gold_v2_strict_import_ready.jsonl",
}
AUDIT_V3_PATH = ROOT / "gold_annotations/pubmed_200_gold_v3_audit_draft.jsonl"
GOLD_PROFILES = frozenset({"frozen-v2", "audit-v3"})
SOURCE_PATH = ROOT / "extraction_output/pubmed_converted_500.jsonl"
SYMMETRIC_PREDICATES = frozenset({"ASSOCIATED_WITH", "INTERACTS_WITH"})


def cell_subset_signature(value: str) -> tuple[str, str] | None:
    """Normalize exact numbered cell-subset paraphrases for scoring only."""
    surface = normalize_surface(value)
    identifier = r"(?:\d+[a-z0-9]*|[a-z]+\d[a-z0-9]*)"
    prefix = re.fullmatch(
        rf"(?:subpopulation|subset|subgroup|cluster)s?\s+({identifier})\s+(.+)",
        surface,
    )
    suffix = re.fullmatch(
        rf"(.+?)\s+(?:subpopulation|subset|subgroup|cluster)s?\s+({identifier})",
        surface,
    )
    if prefix:
        subset_id, cell_text = prefix.group(1), prefix.group(2)
    elif suffix:
        cell_text, subset_id = suffix.group(1), suffix.group(2)
    else:
        return None
    families = (
        "macrophage", "endothelial cell", "stellate cell", "kupffer cell",
        "neutrophil", "monocyte", "natural killer cell", "nk cell",
        "regulatory t cell", "t cell", "b cell", "cholangiocyte",
        "fibroblast", "hepatocyte",
    )
    family = next((item for item in families if item in cell_text), "")
    return (family, subset_id) if family else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def direction_kind(relation: dict[str, Any]) -> str:
    explicit = str(relation.get("direction_semantics", "") or "").upper()
    if explicit:
        return explicit
    return {
        "positive": "ASSOCIATION_SIGN", "negative": "ASSOCIATION_SIGN",
        "increase": "CHANGE_DIRECTION", "decrease": "CHANGE_DIRECTION",
        "none": "NON_DIRECTIONAL", "unknown": "UNKNOWN",
    }.get(str(relation.get("direction", "") or "").casefold(), "INVALID")


def gold_rows_for_profile(profile: str, view: str) -> tuple[list[dict[str, Any]], Path]:
    if profile == "frozen-v2":
        path = GOLD_VIEWS[view]
        return load_jsonl(path), path
    path = AUDIT_V3_PATH
    rows = load_jsonl(path)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        output = dict(row)
        relations = []
        for relation in row.get("relations", []) or []:
            in_contract = str(relation.get("gold_write_status", "")) == "WRITE_CONTRACT"
            if view == "main_kg_write_contract" and not in_contract:
                continue
            if view == "strict_import_ready" and not (
                in_contract and bool(relation.get("import_ready"))
            ):
                continue
            relations.append(relation)
        output["relations"] = relations
        filtered.append(output)
    return filtered, path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records_from_result(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("records", "history", "articles"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, list):
            return value
    return []


def _surface_variants(value: str) -> set[str]:
    raw = str(value or "").strip()
    variants = {normalize_surface(raw)}
    # Keep parenthetical normalization deliberately narrow: only a trailing
    # acronym definition is stripped, never arbitrary disease qualifiers.
    parenthetical = re.fullmatch(
        r"(.+?)\s*\(([A-Za-z][A-Za-z0-9-]{1,14})\)\s*", raw
    )
    if parenthetical:
        variants.add(normalize_surface(parenthetical.group(1)))
        variants.add(normalize_surface(parenthetical.group(2)))
    variants.add(normalize_surface(re.sub(r"[-‐‑‒–—]", " ", raw)))
    greek = {
        "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
        "κ": "kappa", "λ": "lambda", "μ": "mu",
    }
    replaced = raw
    for symbol, name in greek.items():
        replaced = replaced.replace(symbol, name).replace(symbol.upper(), name)
    variants.add(normalize_surface(replaced))
    return {item for item in variants if item}


def _explicit_parenthetical_pairs(text: str) -> list[tuple[str, str]]:
    """Recover source-declared aliases missed by acronym heuristics.

    A preceding parenthesis is a strong boundary for coordinated definitions,
    e.g. ``CD155) and T cell ... domains (TIGIT)``.
    """
    pairs: list[tuple[str, str]] = []
    for match in re.finditer(r"\(([A-Za-z][A-Za-z0-9-]{1,14})\)", text):
        short = match.group(1)
        prefix = text[:match.start()]
        boundary = max(
            prefix.rfind("."), prefix.rfind(";"), prefix.rfind(":"),
            prefix.rfind("\n"), prefix.rfind(")"),
        )
        phrase = prefix[boundary + 1:].strip(" ,:-")
        phrase = re.sub(r"^(?:and|or)\s+", "", phrase, flags=re.IGNORECASE)
        if 2 <= len(phrase.split()) <= 12 and len(phrase) <= 100:
            pairs.append((short, phrase))
    for match in re.finditer(
        r"\b([A-Za-z][A-Za-z0-9-]{1,14})\s*\(([^()]{3,160})\)", text
    ):
        short, long_form = match.group(1), match.group(2).strip()
        if 2 <= len(long_form.split()) <= 18:
            pairs.append((short, long_form))
    return list(dict.fromkeys(pairs))


def _credible_parenthetical_alias(short: str, long_form: str) -> bool:
    short_letters = re.sub(r"[^A-Za-z0-9]", "", short).casefold()
    words = re.findall(r"[A-Za-z0-9]+", long_form)
    if not (2 <= len(short_letters) <= 14 and 2 <= len(words) <= 12):
        return False
    if re.search(r"[.;:=<>]", long_form):
        return False
    return True


def aliases(
    gold: dict[str, Any], text: str, audit: dict[str, list] | None = None,
) -> dict[str, set[tuple[str, str]]]:
    mapping: dict[str, set[tuple[str, str]]] = {}
    for entity in gold.get("entities", []) or []:
        canonical = normalize_surface(entity.get("canonical", entity.get("mention", "")))
        typed = (canonical, str(entity.get("type", "")))
        for surface in (entity.get("mention", ""), entity.get("canonical", "")):
            for variant in _surface_variants(str(surface or "")):
                mapping.setdefault(variant, set()).add(typed)
    # The evaluator accepts only source-explicit parenthetical declarations.
    # Broader abbreviation heuristics remain useful to extraction but are too
    # permissive for a benchmark canonicalizer.
    detected_pairs = [
        item for item in _explicit_parenthetical_pairs(text)
        if _credible_parenthetical_alias(*item)
    ]
    for short, long_form in dict.fromkeys(detected_pairs):
        short_keys = _surface_variants(short)
        long_keys = _surface_variants(long_form)
        typed_values = {
            typed for key in short_keys | long_keys for typed in mapping.get(key, set())
        }
        if audit is not None:
            audit.setdefault("detected_pairs", []).append({
                "short": short, "long_form": long_form,
            })
        if not typed_values:
            if audit is not None:
                audit.setdefault("unresolved_long_forms", []).append(long_form)
            continue
        types = {kind for _, kind in typed_values}
        if len(types) > 1:
            if audit is not None:
                audit.setdefault("type_conflict_pairs", []).append({
                    "short": short, "long_form": long_form, "types": sorted(types),
                })
            continue
        for key in short_keys | long_keys:
            mapping.setdefault(key, set()).update(typed_values)
        if audit is not None:
            audit.setdefault("accepted_pairs", []).append({
                "short": short, "long_form": long_form,
                "type": next(iter(types)),
            })
    return mapping


def canonical_endpoint(value: str, entity_type: str, alias_map: dict) -> str:
    surfaces = _surface_variants(value)
    surface = normalize_surface(value)
    exact = {
        canonical
        for variant in surfaces
        for canonical, kind in alias_map.get(variant, set())
        if kind == entity_type
    }
    if len(exact) == 1:
        return next(iter(exact))
    if entity_type == "CellType":
        signature = cell_subset_signature(surface)
        if signature:
            matches = {
                canonical
                for alias_surface, typed_values in alias_map.items()
                if cell_subset_signature(alias_surface) == signature
                for canonical, kind in typed_values
                if kind == entity_type
            }
            if len(matches) == 1:
                return next(iter(matches))
    return surface


def canonical_endpoint_untyped(value: str, alias_map: dict) -> str:
    """Canonicalize an endpoint without forgiving predicate errors."""
    matches = {
        canonical
        for variant in _surface_variants(value)
        for canonical, _kind in alias_map.get(variant, set())
    }
    if len(matches) == 1:
        return next(iter(matches))
    parenthetical = re.fullmatch(
        r"(.+?)\s*\(([A-Za-z][A-Za-z0-9-]{1,14})\)\s*", str(value or "").strip()
    )
    return normalize_surface(parenthetical.group(1) if parenthetical else value)


def relation_key(relation: dict[str, Any], alias_map: dict) -> tuple[str, str, str, str, str]:
    subject_type = str(relation.get("subject_type", ""))
    object_type = str(relation.get("object_type", ""))
    predicate = str(relation.get("predicate", "")).upper()
    subject = canonical_endpoint(str(relation.get("subject", "")), subject_type, alias_map)
    obj = canonical_endpoint(str(relation.get("object", "")), object_type, alias_map)
    key = (subject, subject_type, predicate, obj, object_type)
    if predicate in SYMMETRIC_PREDICATES and subject_type == object_type:
        reverse = (obj, object_type, predicate, subject, subject_type)
        return min(key, reverse)
    return key


def untyped_relation_key(
    relation: dict[str, Any], alias_map: dict,
) -> tuple[str, str, str]:
    """Diagnostic key that ignores endpoint types but retains the predicate."""
    predicate = str(relation.get("predicate", "")).upper()
    subject = canonical_endpoint_untyped(str(relation.get("subject", "")), alias_map)
    obj = canonical_endpoint_untyped(str(relation.get("object", "")), alias_map)
    key = (subject, predicate, obj)
    if predicate in SYMMETRIC_PREDICATES:
        return min(key, (obj, predicate, subject))
    return key


def semantic_kept(relation: dict[str, Any]) -> bool:
    """Positive semantic prediction: ACCEPTED only, REVIEW is abstention."""
    return (
        str(relation.get("scope_status", "IN_SCOPE")).upper() != "OUT_OF_SCOPE"
        and "article_out_of_scope" not in set(relation.get("quality_flags", []) or [])
        and
        str(relation.get("factual_status", "VALID")).upper() != "REJECTED"
        and str(relation.get("semantic_status", "ACCEPTED")).upper() == "ACCEPTED"
    )


def semantic_candidate_kept(relation: dict[str, Any]) -> bool:
    """Candidate coverage view: ACCEPTED plus HUMAN_REVIEW, never REJECTED."""
    return (
        str(relation.get("scope_status", "IN_SCOPE")).upper() != "OUT_OF_SCOPE"
        and "article_out_of_scope" not in set(relation.get("quality_flags", []) or [])
        and
        str(relation.get("factual_status", "VALID")).upper() != "REJECTED"
        and str(relation.get("semantic_status", "ACCEPTED")).upper() in {"ACCEPTED", "REVIEW"}
    )


def predicted_for_view(
    relations: list[dict[str, Any]], view: str, *, include_review: bool = False,
) -> list[dict[str, Any]]:
    keep = semantic_candidate_kept if include_review else semantic_kept
    kept = [relation for relation in relations if keep(relation)]
    if view == "candidate_semantic":
        return kept
    if view == "main_kg_write_contract":
        return [relation for relation in kept if relation.get("write_contract_valid") is True]
    return [
        relation for relation in kept
        if relation.get("write_contract_valid") is True
        and (
            str(relation.get("write_status", "")).upper() == "IMPORT_READY"
            or relation.get("import_ready") is True
        )
    ]


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054,
) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        ) / denominator
    )
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def prf(counts: Counter) -> dict[str, Any]:
    tp, fp, fn = (int(counts[key]) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if tp == fp == fn == 0:
        f1 = None
    elif precision is None or recall is None:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "precision_wilson95": wilson_interval(tp, tp + fp),
        "recall_wilson95": wilson_interval(tp, tp + fn),
    }


def token_iou(left: str, right: str) -> float:
    a = set(re.findall(r"[A-Za-z0-9_-]+", str(left or "").casefold()))
    b = set(re.findall(r"[A-Za-z0-9_-]+", str(right or "").casefold()))
    return len(a & b) / len(a | b) if a | b else 0.0


def semantic_component(relation: dict[str, Any], field: str) -> str:
    normalized = normalize_relation_semantics(relation)
    return str(normalized.get(field, "UNKNOWN") or "UNKNOWN").upper()


def relations_at_stage(record: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    phases = record.get("phases", {}) or {}
    verified = list((phases.get("verification", {}) or {}).get("relations", []) or [])
    if stage == "raw_hint":
        return list((phases.get("extraction", {}) or {}).get("relations", []) or [])
    if stage == "all_candidates":
        return list(
            (phases.get("relation_core_selection", {}) or {}).get("relations", [])
            or (phases.get("relation_candidate_projection", {}) or {}).get("relations", [])
        )
    if stage == "factual_valid":
        return [item for item in verified if str(item.get("factual_status", "VALID")).upper() != "REJECTED"]
    if stage == "semantic_accepted":
        return [item for item in verified if semantic_kept(item)]
    if stage == "accepted_review":
        return [item for item in verified if semantic_candidate_kept(item)]
    if stage == "import_ready":
        return [
            item for item in verified
            if str(item.get("write_status", "")).upper() == "IMPORT_READY"
            or item.get("import_ready") is True
        ]
    raise ValueError(f"unknown funnel stage: {stage}")


def semantic_risk_coverage(rows: list[tuple[float, bool]]) -> dict[str, Any]:
    """Return selective prediction risk over factual-valid candidates."""
    if not rows:
        return {"population": 0, "points": [], "aurc": None}
    ordered = sorted(rows, key=lambda item: item[0], reverse=True)
    points: list[dict[str, Any]] = []
    n = len(ordered)
    cutoffs = sorted({max(1, math.ceil(n * fraction / 10)) for fraction in range(1, 11)})
    for count in cutoffs:
        selected = ordered[:count]
        correct = sum(int(label) for _, label in selected)
        precision = correct / count
        points.append({
            "coverage": count / n,
            "selected": count,
            "precision": precision,
            "risk": 1.0 - precision,
            "minimum_confidence": selected[-1][0],
        })
    previous_coverage = 0.0
    previous_risk = points[0]["risk"]
    area = 0.0
    for point in points:
        width = point["coverage"] - previous_coverage
        area += width * (previous_risk + point["risk"]) / 2.0
        previous_coverage = point["coverage"]
        previous_risk = point["risk"]
    return {"population": n, "points": points, "aurc": area}


def format_metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def score_funnel(
    records: list[dict[str, Any]], gold_by_pmid: dict[str, dict[str, Any]],
    source_by_pmid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    record_by_pmid = {str(item.get("pmid", "")): item for item in records}
    stages = (
        "raw_hint", "all_candidates", "factual_valid",
        "semantic_accepted", "accepted_review", "import_ready",
    )
    output: dict[str, Any] = {}
    stage_tp_keys: dict[str, set[tuple[Any, ...]]] = {stage: set() for stage in stages}
    projected_hint_tp_ids: set[tuple[str, str]] = set()
    final_accepted_tp_ids: set[tuple[str, str]] = set()
    accepted_recovery_tp: set[tuple[Any, ...]] = set()
    risk_rows: list[tuple[float, bool]] = []
    factual_candidate_count = 0
    accepted_candidate_count = 0
    human_review_count = 0
    lineage_totals = Counter()
    for stage in stages:
        counts = Counter()
        untyped_counts = Counter()
        prediction_count = 0
        for pmid, gold in gold_by_pmid.items():
            source = source_by_pmid.get(pmid, {})
            text = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"
            alias_map = aliases(gold, text)
            gold_keys = {relation_key(item, alias_map) for item in gold.get("relations", []) or []}
            relations = relations_at_stage(record_by_pmid.get(pmid, {}), stage)
            pred_keys = {relation_key(item, alias_map) for item in relations}
            gold_untyped = {
                untyped_relation_key(item, alias_map)
                for item in gold.get("relations", []) or []
            }
            pred_untyped = {
                untyped_relation_key(item, alias_map) for item in relations
            }
            stage_tp_keys[stage].update((pmid, *key) for key in gold_keys & pred_keys)
            prediction_count += len(pred_keys)
            counts.update({
                "tp": len(gold_keys & pred_keys),
                "fp": len(pred_keys - gold_keys),
                "fn": len(gold_keys - pred_keys),
            })
            untyped_counts.update({
                "tp": len(gold_untyped & pred_untyped),
                "fp": len(pred_untyped - gold_untyped),
                "fn": len(gold_untyped - pred_untyped),
            })
        output[stage] = {
            **prf(counts),
            "prediction_count": prediction_count,
            "untyped_endpoint_diagnostic": prf(untyped_counts),
        }

    for pmid, gold in gold_by_pmid.items():
        source = source_by_pmid.get(pmid, {})
        text = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"
        alias_map = aliases(gold, text)
        gold_keys = {relation_key(item, alias_map) for item in gold.get("relations", []) or []}
        record = record_by_pmid.get(pmid, {})
        phases = record.get("phases", {}) or {}
        ledger_summary = (
            (phases.get("candidate_audit_ledger", {}) or {}).get("summary", {}) or {}
        )
        lineage_totals.update({
            "projected": int(ledger_summary.get("projected_hint_count", 0) or 0),
            "accounted": int(
                ledger_summary.get("accounted_projected_hint_count", 0) or 0
            ),
            "eligible": int(
                ledger_summary.get("eligible_projected_hint_count", 0) or 0
            ),
            "eligible_survived": int(
                ledger_summary.get("eligible_survived_count", 0) or 0
            ),
            "semantic_accepted": int(
                ledger_summary.get("semantic_accepted_lineage_count", 0) or 0
            ),
        })
        projected = list((phases.get("relation_candidate_projection", {}) or {}).get("relations", []) or [])
        verified = list((phases.get("verification", {}) or {}).get("relations", []) or [])
        for relation in projected:
            key = relation_key(relation, alias_map)
            candidate_id = str(relation.get("candidate_id", "") or "")
            if candidate_id and key in gold_keys and str(relation.get("candidate_lane", "extracted_hint")) == "extracted_hint":
                projected_hint_tp_ids.add((pmid, candidate_id))
        for relation in verified:
            key = relation_key(relation, alias_map)
            is_gold = key in gold_keys
            factual = str(relation.get("factual_status", "VALID")).upper() != "REJECTED"
            accepted = str(relation.get("semantic_status", "")).upper() == "ACCEPTED"
            if factual:
                factual_candidate_count += 1
                confidence = float(
                    relation.get("semantic_confidence", 0.0)
                    or relation.get("adjudication_confidence", 0.0)
                    or relation.get("classifier_confidence", 0.0)
                    or relation.get("evidence_confidence", 0.0)
                    or 0.0
                )
                risk_rows.append((confidence, is_gold))
            if str(relation.get("write_status", "")).upper() == "HUMAN_REVIEW":
                human_review_count += 1
            if not accepted:
                continue
            accepted_candidate_count += 1
            candidate_ids = {
                str(relation.get("candidate_id", "") or ""),
                *(str(item) for item in relation.get("merged_candidate_ids", []) or []),
                *(str(item) for item in relation.get("source_candidate_ids", []) or []),
                *(
                    str(item.get("candidate_id", "") or "")
                    for item in relation.get("claim_instances", []) or []
                    if isinstance(item, dict)
                ),
            } - {""}
            if is_gold:
                final_accepted_tp_ids.update((pmid, item) for item in candidate_ids)
                canonical = (pmid, *key)
                source_lanes = {
                    str(relation.get("candidate_lane", "extracted_hint") or "extracted_hint"),
                    *(str(item) for item in relation.get("source_lanes", []) or []),
                    *(
                        str(item.get("candidate_lane", "") or "")
                        for item in relation.get("claim_instances", []) or []
                        if isinstance(item, dict)
                    ),
                }
                if "recovery" in source_lanes:
                    accepted_recovery_tp.add(canonical)

    raw_tp_keys = stage_tp_keys["raw_hint"]
    accepted_tp_keys = stage_tp_keys["semantic_accepted"]
    factual_tp_keys = stage_tp_keys["factual_valid"]
    surviving_raw = raw_tp_keys & accepted_tp_keys
    surviving_lineages = projected_hint_tp_ids & final_accepted_tp_ids
    raw_count = output["raw_hint"]["prediction_count"]
    candidate_count = output["all_candidates"]["prediction_count"]
    output["derived"] = {
        "raw_canonical_tp_survival": len(surviving_raw) / len(raw_tp_keys) if raw_tp_keys else None,
        "candidate_id_lineage_tp_survival": (
            len(surviving_lineages) / len(projected_hint_tp_ids)
            if projected_hint_tp_ids else None
        ),
        "audit_lineage_accounting": (
            lineage_totals["accounted"] / lineage_totals["projected"]
            if lineage_totals["projected"] else None
        ),
        "eligible_lineage_survival": (
            lineage_totals["eligible_survived"] / lineage_totals["eligible"]
            if lineage_totals["eligible"] else None
        ),
        "semantic_acceptance_survival": (
            lineage_totals["semantic_accepted"] / lineage_totals["eligible"]
            if lineage_totals["eligible"] else None
        ),
        "recovery_tp_gain": len(accepted_recovery_tp - raw_tp_keys),
        "net_tp_gain": len(accepted_tp_keys) - len(surviving_raw),
        "factual_to_semantic_reconciliation_tp_loss": len(factual_tp_keys - accepted_tp_keys),
        "candidate_expansion_rate": candidate_count / raw_count if raw_count else None,
        "soft_flag_tp_loss": len(factual_tp_keys - accepted_tp_keys),
        "auto_accept_coverage": (
            accepted_candidate_count / factual_candidate_count if factual_candidate_count else None
        ),
        "human_review_burden": (
            human_review_count / factual_candidate_count if factual_candidate_count else None
        ),
    }
    output["semantic_risk_coverage"] = semantic_risk_coverage(risk_rows)
    return output


def score_view(
    records: list[dict[str, Any]], gold_by_pmid: dict[str, dict[str, Any]],
    source_by_pmid: dict[str, dict[str, Any]], view: str, *, include_review: bool = False,
) -> dict[str, Any]:
    counts: Counter = Counter()
    untyped_counts: Counter = Counter()
    by_predicate: dict[str, Counter] = defaultdict(Counter)
    document = Counter()
    direction = Counter()
    component_counts: dict[str, Counter] = defaultdict(Counter)
    component_confusion: dict[str, Counter] = defaultdict(Counter)
    evidence_metrics = Counter()
    missing_pmids: list[str] = []
    alias_audit_total: dict[str, list] = defaultdict(list)

    record_by_pmid = {str(record.get("pmid", "")): record for record in records}
    for pmid, gold in gold_by_pmid.items():
        record = record_by_pmid.get(pmid)
        if record is None:
            missing_pmids.append(pmid)
            relations: list[dict[str, Any]] = []
        else:
            relations = list((record.get("phases", {}) or {}).get("verification", {}).get("relations", []) or [])
        source = source_by_pmid.get(pmid, {})
        text = f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"
        alias_audit: dict[str, list] = {}
        alias_map = aliases(gold, text, alias_audit)
        for category, values in alias_audit.items():
            alias_audit_total[category].extend([
                {"pmid": pmid, **value} if isinstance(value, dict)
                else {"pmid": pmid, "value": value}
                for value in values
            ])
        gold_relations = list(gold.get("relations", []) or [])
        gold_by_key = {relation_key(item, alias_map): item for item in gold_relations}
        pred_by_key = {
            relation_key(item, alias_map): item
            for item in predicted_for_view(relations, view, include_review=include_review)
        }
        gold_keys, pred_keys = set(gold_by_key), set(pred_by_key)
        gold_untyped = {untyped_relation_key(item, alias_map) for item in gold_relations}
        pred_untyped = {
            untyped_relation_key(item, alias_map)
            for item in predicted_for_view(relations, view, include_review=include_review)
        }
        counts.update({
            "tp": len(gold_keys & pred_keys),
            "fp": len(pred_keys - gold_keys),
            "fn": len(gold_keys - pred_keys),
        })
        untyped_counts.update({
            "tp": len(gold_untyped & pred_untyped),
            "fp": len(pred_untyped - gold_untyped),
            "fn": len(gold_untyped - pred_untyped),
        })
        for predicate in sorted({key[2] for key in gold_keys | pred_keys}):
            gold_p = {key for key in gold_keys if key[2] == predicate}
            pred_p = {key for key in pred_keys if key[2] == predicate}
            by_predicate[predicate].update({
                "tp": len(gold_p & pred_p), "fp": len(pred_p - gold_p), "fn": len(gold_p - pred_p),
            })
        for key in gold_keys & pred_keys:
            gold_direction = str(gold_by_key[key].get("direction", "unknown"))
            pred_direction = str(pred_by_key[key].get("direction", "unknown"))
            if gold_direction == pred_direction:
                direction["correct"] += 1
            direction["matched"] += 1
            for field in (
                "relation_direction", "association_sign",
                "expression_change", "activity_change",
            ):
                gold_value = semantic_component(gold_by_key[key], field)
                pred_value = semantic_component(pred_by_key[key], field)
                if gold_value != "UNKNOWN":
                    component_counts[field]["annotated"] += 1
                    if pred_value != "UNKNOWN":
                        component_counts[field]["covered"] += 1
                        if pred_value == gold_value:
                            component_counts[field]["correct"] += 1
                    component_confusion[field][f"{gold_value}->{pred_value}"] += 1
            gold_evidence = str(gold_by_key[key].get("evidence", "") or "")
            pred_evidence = str(pred_by_key[key].get("evidence", "") or "")
            if gold_evidence:
                evidence_metrics["annotated"] += 1
                evidence_metrics["exact"] += int(
                    normalize_surface(gold_evidence) == normalize_surface(pred_evidence)
                )
                evidence_metrics["token_iou_sum"] += token_iou(gold_evidence, pred_evidence)
            pack = pred_by_key[key].get("evidence_pack", {}) or {}
            evidence_metrics["pack_present"] += int(bool(pack.get("spans")))
            evidence_metrics["endpoint_covered"] += int(bool(
                pack.get("support_subject_covered") and pack.get("support_object_covered")
            ))
            evidence_metrics["trigger_covered"] += int(bool(pack.get("support_trigger_covered")))
            evidence_metrics["owner_covered"] += int(bool(pack.get("support_sentence_ids")))

        gold_present, pred_present = bool(gold_keys), bool(pred_keys)
        if gold_present and pred_present:
            document["tp"] += 1
        elif pred_present:
            document["fp"] += 1
        elif gold_present:
            document["fn"] += 1
        else:
            document["tn"] += 1

    predicate_metrics = {predicate: prf(value) for predicate, value in sorted(by_predicate.items())}
    # True predicate macro: only predicates with Gold support participate.
    # A Gold-supported predicate with no predictions contributes F1=0; an
    # empty-Gold/empty-prediction predicate is absent rather than perfect.
    active = [value for value in predicate_metrics.values() if value["tp"] + value["fn"]]
    macro_f1 = (
        sum(float(value["f1"] or 0.0) for value in active) / len(active)
        if active else None
    )
    return {
        "typed_directed_triple_micro": prf(counts),
        "untyped_endpoint_diagnostic": {
            **prf(untyped_counts),
            "diagnostic_only": True,
            "note": "endpoint types ignored; predicate and orientation retained",
        },
        "positive_predicate_macro_f1": macro_f1,
        "per_predicate": predicate_metrics,
        "document_relation_presence": {**prf(document), "tn": int(document["tn"])},
        "direction_agreement_given_triple_tp": direction["correct"] / direction["matched"] if direction["matched"] else None,
        "direction_by_semantics": {
            "deprecated": True,
            "replacement": "direction_components",
            "reason": "legacy direction conflated relation direction, sign, expression and activity change",
        },
        "direction_components": {
            field: {
                "annotated_triple_tp": int(component_counts[field]["annotated"]),
                "coverage": (
                    component_counts[field]["covered"] / component_counts[field]["annotated"]
                    if component_counts[field]["annotated"] else None
                ),
                "conditional_accuracy": (
                    component_counts[field]["correct"] / component_counts[field]["covered"]
                    if component_counts[field]["covered"] else None
                ),
                "confusion": dict(sorted(component_confusion[field].items())),
            }
            for field in (
                "relation_direction", "association_sign",
                "expression_change", "activity_change",
            )
        },
        "evidence": {
            "annotated_triple_tp": int(evidence_metrics["annotated"]),
            "exact_accuracy": (
                evidence_metrics["exact"] / evidence_metrics["annotated"]
                if evidence_metrics["annotated"] else None
            ),
            "mean_token_iou": (
                evidence_metrics["token_iou_sum"] / evidence_metrics["annotated"]
                if evidence_metrics["annotated"] else None
            ),
            "evidence_pack_presence": evidence_metrics["pack_present"] / direction["matched"] if direction["matched"] else None,
            "endpoint_coverage": evidence_metrics["endpoint_covered"] / direction["matched"] if direction["matched"] else None,
            "trigger_coverage": evidence_metrics["trigger_covered"] / direction["matched"] if direction["matched"] else None,
            "owner_coverage": evidence_metrics["owner_covered"] / direction["matched"] if direction["matched"] else None,
        },
        "alias_audit": {
            category: values for category, values in sorted(alias_audit_total.items())
        },
        "missing_result_pmids": missing_pmids,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--gold-profile", choices=sorted(GOLD_PROFILES), default="frozen-v2")
    parser.add_argument(
        "--allow-incomplete-audit-gold", action="store_true",
        help="Allow audit-v3 draft rows that still require expert adjudication.",
    )
    args = parser.parse_args()

    records = records_from_result(args.result)
    source_by_pmid = {str(row["pmid"]): row for row in load_jsonl(SOURCE_PATH)}
    report: dict[str, Any] = {
        "metric_contract_version": "liverkg-gold200-tiered-v4",
        "result": str(args.result),
        "result_record_count": len(records),
        "gold_profile": args.gold_profile,
        "gold_role": "development_only",
        "final_external_test": "BioRED official test",
        "relation_level_tn": "undefined: gold is relation-centric and does not enumerate negative triples",
        "views": {},
    }
    for view in GOLD_VIEWS:
        rows, path = gold_rows_for_profile(args.gold_profile, view)
        if args.gold_profile == "audit-v3" and not args.allow_incomplete_audit_gold:
            unresolved = sum(not bool(row.get("adjudication_complete")) for row in rows)
            if unresolved:
                raise SystemExit(
                    f"audit-v3 is incomplete for {unresolved} documents; "
                    "pass --allow-incomplete-audit-gold for development diagnostics only"
                )
        gold_by_pmid = {str(row["pmid"]): row for row in rows}
        positive_docs = [row for row in rows if row.get("relations")]
        max_doc_relations = max((len(row.get("relations", [])) for row in rows), default=0)
        gold_relation_count = sum(len(row.get("relations", []) or []) for row in rows)
        report["views"][view] = {
            "gold_path": str(path), "gold_sha256": sha256(path),
            "gold_documents": len(rows),
            "gold_relations": gold_relation_count,
            "gold_positive_documents": len(positive_docs),
            "max_relations_in_one_gold_document": max_doc_relations,
            "small_sample_warning": (
                "fewer than 30 positive relations; treat F1 as a safety regression gate"
                if gold_relation_count < 30 else ""
            ),
            "metrics": score_view(records, gold_by_pmid, source_by_pmid, view),
        }
        if view == "candidate_semantic":
            report["views"][view]["review_inclusive_coverage_metrics"] = score_view(
                records, gold_by_pmid, source_by_pmid, view, include_review=True,
            )
            report["views"][view]["stage_funnel"] = score_funnel(
                records, gold_by_pmid, source_by_pmid,
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Gold-200 unified evaluation", "", f"- Gold profile: `{args.gold_profile}` (development only).", "- Final external test: BioRED official test.", "- Relation metric: typed, directed positive triples; same-type `ASSOCIATED_WITH` and `INTERACTS_WITH` are symmetric.", "- Primary metrics count `ACCEPTED` only. Candidate coverage including `REVIEW` is reported separately in JSON and is not automatic-write precision.", "- Relation-level TN is undefined for this relation-centric gold. TN below is document-level relation-presence TN.", "", "| View | Gold relations | Micro P/R/F1 (TP/FP/FN) | Positive macro F1 | Document TP/FP/TN/FN | Direction agreement |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in report["views"].items():
        metric = item["metrics"]
        micro = metric["typed_directed_triple_micro"]
        doc = metric["document_relation_presence"]
        lines.append(
            f"| {name} | {item['gold_relations']} | {format_metric(micro['precision'])}/{format_metric(micro['recall'])}/{format_metric(micro['f1'])} ({micro['tp']}/{micro['fp']}/{micro['fn']}) | {format_metric(metric['positive_predicate_macro_f1'])} | {doc['tp']}/{doc['fp']}/{doc['tn']}/{doc['fn']} | {format_metric(metric['direction_agreement_given_triple_tp'])} |"
        )
    lines.append("")
    strict = report["views"]["strict_import_ready"]
    if strict["small_sample_warning"]:
        lines.append(
            f"> Strict import-ready warning: {strict['small_sample_warning']}; "
            f"{strict['gold_relations']} relations occur in {strict['gold_positive_documents']} documents "
            f"(maximum {strict['max_relations_in_one_gold_document']} in one document)."
        )
        lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.output_json)
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
