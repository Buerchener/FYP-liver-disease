#!/usr/bin/env python3
"""Versioned, leakage-aware quality audit for the Gold200 development set.

The frozen v2 file is never rewritten.  Static audit artifacts may be checked
in, while model-review payloads and caches stay below ``benchmark_output``.
Model reviewers see the source article and the frozen annotation, but never a
system prediction.  Their output is advisory and cannot promote a draft to an
adjudicated gold set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.aux_model_registry import (  # noqa: E402
    AuxModelRegistry,
    AuxModelSpec,
    StructuredModelResult,
)
from cognitive_agent.schema.relation_signatures import (  # noqa: E402
    ALLOWED_DIRECTIONS,
    LITERATURE_CANDIDATE_SIGNATURES,
)
from cognitive_agent.schema.write_contract import (  # noqa: E402
    MAIN_KG_WRITE_CONTRACT_VERSION,
    is_main_kg_write_signature,
)
from liverkg_cli.env import load_project_env  # noqa: E402


GOLD_PATH = ROOT / "gold_annotations/pubmed_200_gold_v2_strict.jsonl"
SOURCE_PATH = ROOT / "extraction_output/pubmed_converted_500.jsonl"
TRACKED_DRAFT_PATH = ROOT / "gold_annotations/pubmed_200_gold_v3_audit_draft.jsonl"
TRACKED_LEDGER_PATH = ROOT / "gold_annotations/pubmed_200_gold_v3_audit_ledger.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "benchmark_output/gold200_quality_audit_v3"

AUDIT_CONTRACT_VERSION = "gold200-quality-audit-v3"
PROMPT_VERSION = "gold200-blind-review-v1"
CLAIM_ROLES = frozenset({
    "CURRENT_FINDING", "BACKGROUND", "METHOD", "PREDICTION", "PRIOR_WORK", "OTHER",
})
DIRECTION_SEMANTICS = {
    "positive": "ASSOCIATION_SIGN",
    "negative": "ASSOCIATION_SIGN",
    "increase": "CHANGE_DIRECTION",
    "decrease": "CHANGE_DIRECTION",
    "none": "NON_DIRECTIONAL",
    "unknown": "UNKNOWN",
}
STRONG_RELATION_CUES = re.compile(
    r"\b(associated with|correlated with|interacts? with|binds? to|encoded by|encodes|"
    r"express(?:ed|ion) in|progress(?:es|ed)? to|prognostic (?:factor|marker)|"
    r"participates? in)\b",
    re.IGNORECASE,
)
NEGATION_CUES = re.compile(
    r"\b(no association|not associated|did not|does not|failed to|without evidence)\b",
    re.IGNORECASE,
)
UNCERTAINTY_CUES = re.compile(
    r"\b(may|might|could|potential|possibly|suggest(?:s|ed)?|predicted|prediction)\b",
    re.IGNORECASE,
)
NON_CURRENT_CONTEXT = re.compile(
    r"(review|commentary|background|prior|prediction|computational|case_report)",
    re.IGNORECASE,
)
L1_RELATION_FLAGS = frozenset({
    "historical_import_ready",
    "main_kg_write_contract",
    "negation_cue",
    "uncertainty_cue",
    "non_current_study_context",
    "same_endpoint_pair_multilabel",
    "rare_predicate",
})
L1_DOCUMENT_FLAGS = L1_RELATION_FLAGS | frozenset({"zero_relation_with_strong_cue"})


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def article_text(source: dict[str, Any]) -> str:
    return f"TITLE: {source.get('title', '')}\nABSTRACT: {source.get('abstract', '')}"


def sentence_index(text: str, start: int) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", text[: max(0, start)]))


def locate_evidence(evidence: str, text: str) -> list[dict[str, Any]]:
    start = text.find(evidence)
    if start < 0:
        return []
    return [{
        "text": evidence,
        "start": start,
        "end": start + len(evidence),
        "sentence_index": sentence_index(text, start),
    }]


def direction_semantics(direction: str) -> str:
    return DIRECTION_SEMANTICS.get(str(direction or "").casefold(), "INVALID")


def relation_signature(relation: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(relation.get("predicate", "")).upper(),
        str(relation.get("subject_type", "")),
        str(relation.get("object_type", "")),
    )


def relation_id(pmid: str, index: int, relation: dict[str, Any]) -> str:
    digest = stable_hash({
        "pmid": pmid,
        "index": index,
        "signature": relation_signature(relation),
        "subject": relation.get("subject", ""),
        "object": relation.get("object", ""),
        "evidence": relation.get("evidence", ""),
    })[:16]
    return f"{pmid}:r{index + 1}:{digest}"


def infer_claim_role(row: dict[str, Any], relation: dict[str, Any]) -> tuple[str, list[str]]:
    context = str(row.get("study_context", ""))
    reason = str(relation.get("exclusion_reason", ""))
    evidence = str(relation.get("evidence", ""))
    combined = f"{context} {reason}".casefold()
    reasons: list[str] = []
    if "prior" in combined:
        return "PRIOR_WORK", ["prior_work_marker"]
    if "background" in combined or "review" in context.casefold():
        return "BACKGROUND", ["background_or_review_context"]
    if "computational" in combined or "prediction" in combined or "predicted" in evidence.casefold():
        return "PREDICTION", ["prediction_or_computational_context"]
    if re.search(r"\b(method|we used|was performed|were retrieved|docking)\b", evidence, re.I):
        return "METHOD", ["method_marker"]
    reasons.append("default_current_finding_requires_adjudication")
    return "CURRENT_FINDING", reasons


def endpoint_surfaces(row: dict[str, Any], relation: dict[str, Any], side: str) -> set[str]:
    canonical = str(relation.get(side, ""))
    entity_type = str(relation.get(f"{side}_type", ""))
    values = {canonical.casefold()}
    for entity in row.get("entities", []) or []:
        if (
            str(entity.get("type", "")) == entity_type
            and str(entity.get("canonical", entity.get("mention", ""))).casefold()
            == canonical.casefold()
        ):
            values.add(str(entity.get("mention", "")).casefold())
            values.add(str(entity.get("canonical", "")).casefold())
    return {value for value in values if value}


def relation_risk_flags(
    row: dict[str, Any], relation: dict[str, Any], *, pair_multilabel: bool,
    predicate_count: int,
) -> list[str]:
    flags: list[str] = []
    evidence = str(relation.get("evidence", ""))
    signature = relation_signature(relation)
    if bool(relation.get("import_ready")):
        flags.append("historical_import_ready")
    if is_main_kg_write_signature(signature[0], signature[1], signature[2]):
        flags.append("main_kg_write_contract")
    if NEGATION_CUES.search(evidence):
        flags.append("negation_cue")
    if UNCERTAINTY_CUES.search(evidence):
        flags.append("uncertainty_cue")
    if NON_CURRENT_CONTEXT.search(str(row.get("study_context", ""))):
        flags.append("non_current_study_context")
    if pair_multilabel:
        flags.append("same_endpoint_pair_multilabel")
    if predicate_count <= 3:
        flags.append("rare_predicate")
    evidence_folded = evidence.casefold()
    for side in ("subject", "object"):
        if not any(surface in evidence_folded for surface in endpoint_surfaces(row, relation, side)):
            flags.append(f"{side}_alias_not_literal_in_evidence")
    return list(dict.fromkeys(flags))


def static_audit(
    gold_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_by_pmid = {str(row.get("pmid", "")): row for row in source_rows[:200]}
    predicate_counts = Counter(
        str(relation.get("predicate", ""))
        for row in gold_rows for relation in row.get("relations", []) or []
    )
    errors: list[dict[str, Any]] = []
    document_ledger: list[dict[str, Any]] = []
    relation_ledger: list[dict[str, Any]] = []
    draft_rows: list[dict[str, Any]] = []

    seen_pmids: set[str] = set()
    for doc_index, row in enumerate(gold_rows):
        pmid = str(row.get("pmid", ""))
        source = source_by_pmid.get(pmid)
        if pmid in seen_pmids:
            errors.append({"pmid": pmid, "code": "duplicate_pmid"})
        seen_pmids.add(pmid)
        if source is None:
            errors.append({"pmid": pmid, "code": "source_record_missing"})
            source = {}
        expected_source = source_rows[doc_index] if doc_index < min(200, len(source_rows)) else {}
        if str(expected_source.get("pmid", "")) != pmid:
            errors.append({
                "pmid": pmid, "code": "source_order_mismatch",
                "expected_pmid": str(expected_source.get("pmid", "")),
            })
        if str(source.get("title", "")) != str(row.get("title", "")):
            errors.append({"pmid": pmid, "code": "source_title_mismatch"})
        text = article_text(source)
        source_digest = stable_hash({
            "pmid": pmid, "title": source.get("title", ""), "abstract": source.get("abstract", ""),
        })
        pair_counts = Counter(
            (
                str(rel.get("subject", "")).casefold(), str(rel.get("subject_type", "")),
                str(rel.get("object", "")).casefold(), str(rel.get("object_type", "")),
            )
            for rel in row.get("relations", []) or []
        )
        draft = dict(row)
        draft_relations: list[dict[str, Any]] = []
        doc_reasons: set[str] = set()
        relation_ids: list[str] = []
        for rel_index, relation in enumerate(row.get("relations", []) or []):
            current = dict(relation)
            rid = relation_id(pmid, rel_index, current)
            relation_ids.append(rid)
            signature = relation_signature(current)
            allowed_pairs = LITERATURE_CANDIDATE_SIGNATURES.get(signature[0], set())
            if (signature[1], signature[2]) not in allowed_pairs:
                errors.append({"pmid": pmid, "relation_id": rid, "code": "candidate_signature_invalid"})
            direction = str(current.get("direction", ""))
            if direction not in ALLOWED_DIRECTIONS:
                errors.append({"pmid": pmid, "relation_id": rid, "code": "direction_invalid"})
            spans = locate_evidence(str(current.get("evidence", "")), text)
            if not spans:
                errors.append({"pmid": pmid, "relation_id": rid, "code": "evidence_not_in_source"})
            pair_key = (
                str(current.get("subject", "")).casefold(), str(current.get("subject_type", "")),
                str(current.get("object", "")).casefold(), str(current.get("object_type", "")),
            )
            flags = relation_risk_flags(
                row, current, pair_multilabel=pair_counts[pair_key] > 1,
                predicate_count=predicate_counts[signature[0]],
            )
            doc_reasons.update(flags)
            l1_flags = [flag for flag in flags if flag in L1_RELATION_FLAGS]
            claim_role, role_reasons = infer_claim_role(row, current)
            adjudication_status = "EXPERT_REVIEW_REQUIRED" if l1_flags else "PENDING_REVIEW"
            relation_ledger.append({
                "audit_contract_version": AUDIT_CONTRACT_VERSION,
                "pmid": pmid,
                "document_index": doc_index,
                "relation_id": rid,
                "source_sha256": source_digest,
                "current_annotation": current,
                "suggested_interface_fields": {
                    "claim_role": claim_role,
                    "claim_role_reasons": role_reasons,
                    "direction_semantics": direction_semantics(direction),
                    "evidence_spans": spans,
                    "gold_write_status": (
                        "WRITE_CONTRACT"
                        if is_main_kg_write_signature(*signature)
                        else "CANDIDATE_ONLY"
                    ),
                },
                "risk_flags": flags,
                "audit_tier": "L1_HIGH_RISK" if l1_flags else "L2_FULL",
                "adjudication_status": adjudication_status,
                "reviewer_decisions": {},
                "proposed_change": None,
            })
            augmented = dict(current)
            augmented.update({
                "relation_id": rid,
                "claim_role": claim_role,
                "evidence_spans": spans,
                "direction_semantics": direction_semantics(direction),
                "gold_write_status": (
                    "WRITE_CONTRACT"
                    if is_main_kg_write_signature(*signature)
                    else "CANDIDATE_ONLY"
                ),
                "adjudication_status": adjudication_status,
                "adjudication_reasons": flags + role_reasons,
            })
            draft_relations.append(augmented)

        zero_cues = [] if row.get("relations") else STRONG_RELATION_CUES.findall(text)
        if zero_cues:
            doc_reasons.add("zero_relation_with_strong_cue")
        if not row.get("relations") and bool(row.get("in_scope")):
            doc_reasons.add("in_scope_zero_relation")
        l1_doc_reasons = sorted(flag for flag in doc_reasons if flag in L1_DOCUMENT_FLAGS)
        document_status = "EXPERT_REVIEW_REQUIRED" if l1_doc_reasons else "PENDING_REVIEW"
        document_ledger.append({
            "audit_contract_version": AUDIT_CONTRACT_VERSION,
            "pmid": pmid,
            "document_index": doc_index,
            "source_sha256": source_digest,
            "relation_ids": relation_ids,
            "relation_count": len(relation_ids),
            "in_scope": bool(row.get("in_scope")),
            "zero_relation": not bool(relation_ids),
            "strong_relation_cue_count": len(zero_cues),
            "risk_flags": sorted(doc_reasons),
            "audit_tier": "L1_HIGH_RISK" if l1_doc_reasons else "L2_FULL",
            "adjudication_status": document_status,
            "reviewer_decisions": {},
        })
        draft["relations"] = draft_relations
        draft["gold_version"] = "gold200-v3-audit-draft"
        draft["gold_development_only"] = True
        draft["source_sha256"] = source_digest
        draft["adjudication_status"] = document_status
        draft["adjudication_complete"] = False
        draft_rows.append(draft)

    if len(gold_rows) != 200:
        errors.append({"code": "document_count_mismatch", "actual": len(gold_rows), "expected": 200})
    if len(seen_pmids) != len(gold_rows):
        errors.append({"code": "unique_pmid_count_mismatch", "actual": len(seen_pmids)})
    summary = {
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "documents": len(gold_rows),
        "relations": len(relation_ledger),
        "zero_relation_documents": sum(item["zero_relation"] for item in document_ledger),
        "in_scope_zero_relation_documents": sum(
            item["zero_relation"] and item["in_scope"] for item in document_ledger
        ),
        "l1_documents": sum(item["audit_tier"] == "L1_HIGH_RISK" for item in document_ledger),
        "l1_relations": sum(item["audit_tier"] == "L1_HIGH_RISK" for item in relation_ledger),
        "historical_import_ready_relations": sum(
            bool(item["current_annotation"].get("import_ready")) for item in relation_ledger
        ),
        "write_contract_relations": sum(
            item["suggested_interface_fields"]["gold_write_status"] == "WRITE_CONTRACT"
            for item in relation_ledger
        ),
        "claim_role_missing_in_frozen": sum(
            not bool(item["current_annotation"].get("claim_role")) for item in relation_ledger
        ),
        "structural_errors": errors,
        "structural_valid": not errors,
        "risk_flag_counts": dict(Counter(
            flag for item in relation_ledger for flag in item["risk_flags"]
        )),
    }
    return document_ledger, relation_ledger, draft_rows, summary


SYSTEM_PROMPT = """You are an independent biomedical relation-annotation auditor.
Review only the supplied PubMed title/abstract and frozen Gold annotation. You
must not assume an extractor prediction. Lexical co-occurrence alone is not a
relation. Preserve supported background, method and prediction relations as
semantic candidates, but distinguish them from current findings and from safe
main-KG writes. Apply scoped negation to the target clause only.

Allowed entity types: Gene, Protein, Disease, Pathway, Metabolite, Tissue,
CellType. Allowed predicates: ASSOCIATED_WITH, PROGNOSTIC_IN, PROGRESSES_TO,
ENCODES, INTERACTS_WITH, PARTICIPATES_IN, EXPRESSED_IN,
ASSOCIATED_WITH_METABOLITE. Claim roles: CURRENT_FINDING, BACKGROUND, METHOD,
PREDICTION, PRIOR_WORK, OTHER. Directions retain distinct semantics:
positive/negative are association signs; increase/decrease are change
directions; none is non-directional; unknown is unresolved.

Return one JSON object. Review each supplied relation_id, identify likely
missing relations only when both endpoints and the predicate are supported by
source text, and quote exact contiguous source evidence. Model opinions are
advisory; do not claim expert adjudication."""

REVIEW_SCHEMA = {
    "document_decision": {
        "in_scope": True,
        "zero_relation_annotation": "CONFIRMED|LIKELY_MISSING|UNSURE",
        "reason": "string",
    },
    "relation_reviews": [{
        "relation_id": "string",
        "decision": "CONFIRM|REVISE|REMOVE|UNSURE",
        "claim_role": "CURRENT_FINDING|BACKGROUND|METHOD|PREDICTION|PRIOR_WORK|OTHER",
        "direction": "positive|negative|increase|decrease|none|unknown",
        "import_ready": "YES|NO|UNSURE",
        "source_quote": "exact quote",
        "reason": "string",
    }],
    "missing_relations": [{
        "subject": "string", "subject_type": "string", "predicate": "string",
        "object": "string", "object_type": "string", "direction": "string",
        "claim_role": "string", "source_quote": "exact quote", "reason": "string",
    }],
    "expert_review_required": True,
}


def review_user_prompt(
    row: dict[str, Any], source: dict[str, Any], relation_rows: list[dict[str, Any]],
) -> str:
    frozen_relations = []
    by_id = {item["relation_id"]: item for item in relation_rows}
    for relation in row.get("relations", []) or []:
        match = next(
            (item for item in relation_rows if item["current_annotation"] == relation),
            None,
        )
        payload = dict(relation)
        payload["relation_id"] = match["relation_id"] if match else ""
        frozen_relations.append(payload)
    annotation = {
        "in_scope": row.get("in_scope"),
        "study_context": row.get("study_context"),
        "entities": row.get("entities", []),
        "relations": frozen_relations,
        "negative_notes": row.get("negative_notes", []),
    }
    del by_id
    return (
        f"PMID: {row.get('pmid', '')}\n"
        f"TITLE: {source.get('title', '')}\n"
        f"ABSTRACT: {source.get('abstract', '')}\n\n"
        "FROZEN GOLD ANNOTATION TO AUDIT:\n"
        + json.dumps(annotation, ensure_ascii=False, sort_keys=True)
    )


@dataclass(frozen=True)
class Reviewer:
    name: str
    spec: AuxModelSpec


class ReviewCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS gold_reviews (
                cache_key TEXT PRIMARY KEY,
                reviewer TEXT NOT NULL,
                model_id TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.connection.commit()
        self.lock = threading.Lock()

    def get(self, key: str, *, retry_invalid: bool = False) -> dict[str, Any] | None:
        status_clause = "AND status='OK'" if retry_invalid else ""
        with self.lock:
            row = self.connection.execute(
                f"SELECT result_json FROM gold_reviews WHERE cache_key=? {status_clause}", (key,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, reviewer: Reviewer, result: dict[str, Any]) -> None:
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO gold_reviews VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key, reviewer.name, reviewer.spec.model_id,
                    str(result.get("status", "")),
                    json.dumps(result, ensure_ascii=False, sort_keys=True), time.time(),
                ),
            )
            self.connection.commit()

    def close(self) -> None:
        with self.lock:
            self.connection.close()


def configured_reviewers(env: dict[str, str], selection: str) -> list[Reviewer]:
    common = {
        "timeout_s": float(env.get("GOLD_AUDIT_TIMEOUT_S", "120")),
        "max_retries": int(env.get("GOLD_AUDIT_MAX_RETRIES", "4")),
        "retry_base_delay_s": float(env.get("GOLD_AUDIT_RETRY_BASE_DELAY_S", "2")),
        "retry_max_delay_s": float(env.get("GOLD_AUDIT_RETRY_MAX_DELAY_S", "45")),
    }
    reviewers: list[Reviewer] = []
    if selection in {"both", "gemini"}:
        reviewers.append(Reviewer("gemini", AuxModelSpec(
            role="primary", provider="openai",
            model_id=env.get("GEMINI_MODEL", "gemini-3.7-flash"),
            api_base=env.get("GEMINI_API_BASE", ""),
            api_key=env.get("GEMINI_API_KEY", ""), **common,
        )))
    if selection in {"both", "deepseek"}:
        reviewers.append(Reviewer("deepseek", AuxModelSpec(
            role="judge", provider="openai",
            model_id=env.get("SECOND_LLM_MODEL_ID", env.get("AUX_PRIMARY_MODEL", "deepseek-v4-flash")),
            api_base=env.get("SECOND_LLM_API_BASE", env.get("DEEPSEEK_API_BASE", "")),
            api_key=env.get("SECOND_LLM_API_KEY", env.get("DEEPSEEK_API_KEY", "")), **common,
        )))
    missing = [reviewer.name for reviewer in reviewers if not reviewer.spec.configured]
    if missing:
        raise RuntimeError("unconfigured reviewers: " + ", ".join(missing))
    return reviewers


def validate_review_payload(payload: dict[str, Any], relation_ids: set[str], text: str) -> list[str]:
    errors: list[str] = []
    reviews = payload.get("relation_reviews", [])
    if not isinstance(reviews, list):
        return ["relation_reviews_not_list"]
    returned_ids: set[str] = set()
    for item in reviews:
        rid = str(item.get("relation_id", ""))
        returned_ids.add(rid)
        if rid not in relation_ids:
            errors.append(f"unknown_relation_id:{rid}")
        if str(item.get("claim_role", "")) not in CLAIM_ROLES:
            errors.append(f"invalid_claim_role:{rid}")
        if str(item.get("direction", "")) not in ALLOWED_DIRECTIONS:
            errors.append(f"invalid_direction:{rid}")
        quote = str(item.get("source_quote", ""))
        if quote and quote not in text:
            errors.append(f"source_quote_not_exact:{rid}")
    if returned_ids != relation_ids:
        errors.append("relation_review_coverage_mismatch")
    missing = payload.get("missing_relations", [])
    if not isinstance(missing, list):
        errors.append("missing_relations_not_list")
    else:
        for index, item in enumerate(missing):
            quote = str(item.get("source_quote", ""))
            if not quote or quote not in text:
                errors.append(f"missing_relation_quote_not_exact:{index}")
            signature = (
                str(item.get("predicate", "")).upper(),
                str(item.get("subject_type", "")), str(item.get("object_type", "")),
            )
            if (signature[1], signature[2]) not in LITERATURE_CANDIDATE_SIGNATURES.get(signature[0], set()):
                errors.append(f"missing_relation_signature_invalid:{index}")
    return errors


def reject_invalid_missing_proposals(
    payload: dict[str, Any], errors: list[str],
) -> tuple[dict[str, Any], list[str], int]:
    """Drop invalid additive proposals while preserving valid relation reviews.

    A reviewer is allowed to brainstorm a bad missing relation; that suggestion
    must not invalidate its otherwise complete audit of frozen relation IDs.
    Errors concerning existing relations remain critical.
    """
    rejected_indices: set[int] = set()
    critical: list[str] = []
    for error in errors:
        if error.startswith("missing_relation_"):
            try:
                rejected_indices.add(int(error.rsplit(":", 1)[1]))
            except (TypeError, ValueError):
                critical.append(error)
        else:
            critical.append(error)
    sanitized = dict(payload)
    missing = list(payload.get("missing_relations", []) or [])
    sanitized["missing_relations"] = [
        item for index, item in enumerate(missing) if index not in rejected_indices
    ]
    return sanitized, critical, len(rejected_indices)


def call_reviewer(
    reviewer: Reviewer, row: dict[str, Any], source: dict[str, Any],
    relation_rows: list[dict[str, Any]], cache: ReviewCache, *, retry_invalid: bool = False,
) -> dict[str, Any]:
    prompt = review_user_prompt(row, source, relation_rows)
    cache_key = stable_hash({
        "prompt_version": PROMPT_VERSION,
        "reviewer": reviewer.name,
        "model_id": reviewer.spec.model_id,
        "system": SYSTEM_PROMPT,
        "user": prompt,
        "schema": REVIEW_SCHEMA,
    })
    cached = cache.get(cache_key, retry_invalid=retry_invalid)
    if cached is not None:
        cached["local_result_hit"] = True
        return cached
    registry = AuxModelRegistry([reviewer.spec])
    result: StructuredModelResult = registry.call_json(
        reviewer.spec.role,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        schema_hint=REVIEW_SCHEMA,
    )
    text = article_text(source)
    relation_ids = {item["relation_id"] for item in relation_rows}
    validation_errors = (
        validate_review_payload(result.payload, relation_ids, text)
        if result.status == "OK" else []
    )
    sanitized_payload, critical_errors, rejected_missing = reject_invalid_missing_proposals(
        result.payload, validation_errors,
    )
    payload = {
        "pmid": str(row.get("pmid", "")),
        "reviewer": reviewer.name,
        "model_id": reviewer.spec.model_id,
        "status": "INVALID_RESPONSE" if critical_errors else result.status,
        "review": sanitized_payload,
        "validation_errors": validation_errors,
        "critical_validation_errors": critical_errors,
        "rejected_missing_relation_proposals": rejected_missing,
        "error": result.error,
        "latency_s": result.latency_s,
        "attempts": result.attempts,
        "prompt_tokens": result.prompt_tokens,
        "output_tokens": result.output_tokens,
        "provider_cache_read_tokens": result.provider_cache_read_tokens,
        "provider_cache_miss_tokens": result.provider_cache_miss_tokens,
        "local_result_hit": False,
    }
    cache.put(cache_key, reviewer, payload)
    return payload


def select_documents(
    stage: str, gold_rows: list[dict[str, Any]], document_ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if stage == "l2":
        return list(gold_rows)
    high_risk = {
        item["pmid"] for item in document_ledger
        if item["audit_tier"] == "L1_HIGH_RISK"
    }
    return [row for row in gold_rows if str(row.get("pmid", "")) in high_risk]


def _review_consensus(decisions: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    if not decisions:
        return "UNREVIEWED", None
    if len(decisions) == 1:
        return "SINGLE_REVIEW", None
    values = list(decisions.values())
    fields = ("decision", "claim_role", "direction", "import_ready")
    first = {field: values[0].get(field) for field in fields}
    if all({field: item.get(field) for field in fields} == first for item in values[1:]):
        return "CONSENSUS", first
    return "DISAGREEMENT", None


def build_review_ledgers(
    document_ledger: list[dict[str, Any]], relation_ledger: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    valid_by_pmid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    attempted_pmids = {str(result.get("pmid", "")) for result in results}
    for result in results:
        if result.get("status") == "OK":
            valid_by_pmid[str(result.get("pmid", ""))][str(result.get("reviewer", ""))] = result

    relation_reviews: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for reviewers in valid_by_pmid.values():
        for reviewer, result in reviewers.items():
            for decision in result.get("review", {}).get("relation_reviews", []) or []:
                relation_reviews[str(decision.get("relation_id", ""))][reviewer] = decision

    enriched_relations: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    consensus_counts: Counter[str] = Counter()
    for item in relation_ledger:
        output = dict(item)
        decisions = relation_reviews.get(item["relation_id"], {})
        status, consensus = _review_consensus(decisions)
        consensus_counts[status] += 1
        output["reviewer_decisions"] = decisions
        output["model_review_status"] = status
        output["model_consensus"] = consensus
        enriched_relations.append(output)
        high_risk = item["audit_tier"] == "L1_HIGH_RISK"
        confirmed = bool(consensus and consensus.get("decision") == "CONFIRM")
        if high_risk or (item["pmid"] in attempted_pmids and not confirmed):
            flags = set(item.get("risk_flags", []))
            priority = 0 if "historical_import_ready" in flags else (
                1 if "main_kg_write_contract" in flags else 2
            )
            queue.append({
                "item_type": "RELATION",
                "item_id": item["relation_id"],
                "pmid": item["pmid"],
                "priority": priority,
                "current_annotation": item["current_annotation"],
                "risk_flags": item["risk_flags"],
                "reviewer_decisions": decisions,
                "model_review_status": status,
                "model_consensus": consensus,
                "required_action": "EXPERT_REVIEW_REQUIRED",
            })

    enriched_documents: list[dict[str, Any]] = []
    missing_groups: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    zero_consensus = Counter()
    for item in document_ledger:
        output = dict(item)
        reviewers = valid_by_pmid.get(item["pmid"], {})
        document_decisions = {
            reviewer: result.get("review", {}).get("document_decision", {})
            for reviewer, result in reviewers.items()
        }
        output["reviewer_decisions"] = document_decisions
        zero_values = {
            str(value.get("zero_relation_annotation", ""))
            for value in document_decisions.values()
        }
        zero_status = (
            "UNREVIEWED" if not zero_values else
            "CONSENSUS" if len(document_decisions) >= 2 and len(zero_values) == 1 else
            "SINGLE_REVIEW" if len(document_decisions) == 1 else "DISAGREEMENT"
        )
        output["model_review_status"] = zero_status
        zero_consensus[zero_status] += 1
        enriched_documents.append(output)
        if item.get("zero_relation") and item["pmid"] in attempted_pmids and (
            item.get("audit_tier") == "L1_HIGH_RISK"
            or zero_values - {"CONFIRMED"}
            or zero_status != "CONSENSUS"
        ):
            queue.append({
                "item_type": "ZERO_RELATION_DOCUMENT",
                "item_id": f"{item['pmid']}:zero",
                "pmid": item["pmid"],
                "priority": 1 if "zero_relation_with_strong_cue" in item.get("risk_flags", []) else 2,
                "risk_flags": item.get("risk_flags", []),
                "reviewer_decisions": document_decisions,
                "model_review_status": zero_status,
                "required_action": "EXPERT_REVIEW_REQUIRED",
            })
        for reviewer, result in reviewers.items():
            for proposal in result.get("review", {}).get("missing_relations", []) or []:
                key = (
                    item["pmid"], str(proposal.get("subject", "")).casefold(),
                    str(proposal.get("subject_type", "")),
                    str(proposal.get("predicate", "")).upper(),
                    str(proposal.get("object", "")).casefold(),
                    str(proposal.get("object_type", "")),
                )
                grouped = missing_groups.setdefault(key, {
                    "item_type": "MISSING_RELATION_PROPOSAL",
                    "item_id": f"{item['pmid']}:missing:{stable_hash(key)[:16]}",
                    "pmid": item["pmid"], "priority": 2,
                    "reviewer_proposals": {},
                    "required_action": "EXPERT_REVIEW_REQUIRED",
                })
                grouped["reviewer_proposals"][reviewer] = proposal

    for proposal in missing_groups.values():
        proposals = list(proposal["reviewer_proposals"].values())
        comparison_fields = (
            "subject", "subject_type", "predicate", "object", "object_type",
            "direction", "claim_role",
        )
        signatures = {
            tuple(str(item.get(field, "")).casefold() for field in comparison_fields)
            for item in proposals
        }
        proposal["model_review_status"] = (
            "CONSENSUS" if len(proposals) >= 2 and len(signatures) == 1 else
            "DISAGREEMENT" if len(proposals) >= 2 else "SINGLE_REVIEW"
        )
        queue.append(proposal)
    queue.sort(key=lambda item: (int(item["priority"]), item["pmid"], item["item_id"]))
    summary = {
        "relation_model_review_status": dict(consensus_counts),
        "document_model_review_status": dict(zero_consensus),
        "missing_relation_proposals": len(missing_groups),
        "missing_relation_consensus": sum(
            item["model_review_status"] == "CONSENSUS" for item in missing_groups.values()
        ),
        "adjudication_queue_items": len(queue),
        "automatic_gold_changes": 0,
    }
    return enriched_documents, enriched_relations, queue, summary


def render_summary(summary: dict[str, Any]) -> str:
    review = summary.get("remote_review", {})
    return "\n".join([
        "# Gold200 quality audit",
        "",
        f"- Contract: `{summary['audit_contract_version']}`",
        f"- Frozen Gold SHA-256: `{summary['frozen_gold_sha256']}`",
        f"- Documents / relations: {summary['documents']} / {summary['relations']}",
        f"- L1 documents / relations: {summary['l1_documents']} / {summary['l1_relations']}",
        f"- Structural valid: {summary['structural_valid']}",
        f"- Frozen relations missing claim_role: {summary['claim_role_missing_in_frozen']}",
        f"- Remote stage: {review.get('stage', 'not_run')}",
        f"- Remote reviews OK / invalid / failed: {review.get('ok', 0)} / {review.get('invalid', 0)} / {review.get('failed', 0)}",
        f"- Prompt / output tokens: {review.get('prompt_tokens', 0)} / {review.get('output_tokens', 0)}",
        "",
        "The frozen v2 file is a development benchmark. Model reviews are advisory;",
        "unresolved and import-ready records require human or domain-expert adjudication.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("static", "l1", "l2"), default="static")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reviewers", choices=("both", "gemini", "deepseek"), default="both")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--emit-tracked-draft", action="store_true")
    parser.add_argument(
        "--retry-invalid", action="store_true",
        help="Retry cached INVALID_RESPONSE entries; default warm replay is fully local.",
    )
    args = parser.parse_args()

    gold_rows = load_jsonl(args.gold)
    source_rows = load_jsonl(args.source)
    document_ledger, relation_ledger, draft_rows, summary = static_audit(gold_rows, source_rows)
    summary.update({
        "frozen_gold_path": str(args.gold),
        "frozen_gold_sha256": sha256_file(args.gold),
        "source_path": str(args.source),
        "source_sha256": sha256_file(args.source),
        "write_contract_version": MAIN_KG_WRITE_CONTRACT_VERSION,
        "gold_role": "development_only",
        "final_external_test": "BioRED official test",
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "document_ledger.jsonl", document_ledger)
    write_jsonl(args.output_dir / "relation_ledger.jsonl", relation_ledger)
    write_jsonl(args.output_dir / "gold200_v3_audit_draft.jsonl", draft_rows)
    if args.emit_tracked_draft:
        write_jsonl(TRACKED_DRAFT_PATH, draft_rows)
        write_jsonl(TRACKED_LEDGER_PATH, relation_ledger)

    if args.stage != "static":
        env = load_project_env(ROOT)
        reviewers = configured_reviewers(env, args.reviewers)
        selected = select_documents(args.stage, gold_rows, document_ledger)
        if args.max_documents > 0:
            selected = selected[: args.max_documents]
        source_by_pmid = {str(row.get("pmid", "")): row for row in source_rows}
        relations_by_pmid: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in relation_ledger:
            relations_by_pmid[item["pmid"]].append(item)
        cache = ReviewCache(args.output_dir / "blind_review_cache.sqlite3")
        results: list[dict[str, Any]] = []
        futures = []
        with ThreadPoolExecutor(max_workers=max(1, min(2, args.max_workers))) as executor:
            for row in selected:
                pmid = str(row.get("pmid", ""))
                for reviewer in reviewers:
                    futures.append(executor.submit(
                        call_reviewer, reviewer, row, source_by_pmid.get(pmid, {}),
                        relations_by_pmid.get(pmid, []), cache,
                        retry_invalid=args.retry_invalid,
                    ))
            for future in as_completed(futures):
                results.append(future.result())
        cache.close()
        results.sort(key=lambda item: (item["pmid"], item["reviewer"]))
        write_jsonl(args.output_dir / f"{args.stage}_model_reviews.jsonl", results)
        enriched_documents, enriched_relations, queue, review_summary = build_review_ledgers(
            document_ledger, relation_ledger, results,
        )
        write_jsonl(args.output_dir / "document_ledger_with_reviews.jsonl", enriched_documents)
        write_jsonl(args.output_dir / "relation_ledger_with_reviews.jsonl", enriched_relations)
        write_jsonl(args.output_dir / "adjudication_queue.jsonl", queue)
        statuses = Counter(item["status"] for item in results)
        summary["remote_review"] = {
            "stage": args.stage,
            "documents_selected": len(selected),
            "review_requests": len(results),
            "ok": statuses["OK"],
            "invalid": statuses["INVALID_RESPONSE"],
            "failed": len(results) - statuses["OK"] - statuses["INVALID_RESPONSE"],
            "local_result_hits": sum(bool(item.get("local_result_hit")) for item in results),
            "remote_requests": sum(not bool(item.get("local_result_hit")) for item in results),
            "remote_prompt_tokens": sum(
                int(item.get("prompt_tokens", 0))
                for item in results if not item.get("local_result_hit")
            ),
            "remote_output_tokens": sum(
                int(item.get("output_tokens", 0))
                for item in results if not item.get("local_result_hit")
            ),
            "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in results),
            "output_tokens": sum(int(item.get("output_tokens", 0)) for item in results),
            "provider_cache_read_tokens": sum(
                int(item.get("provider_cache_read_tokens", 0)) for item in results
            ),
            "provider_cache_miss_tokens": sum(
                int(item.get("provider_cache_miss_tokens", 0)) for item in results
            ),
            "rejected_missing_relation_proposals": sum(
                int(item.get("rejected_missing_relation_proposals", 0)) for item in results
            ),
            **review_summary,
        }
    else:
        summary["remote_review"] = {"stage": "not_run"}

    write_json(args.output_dir / "manifest.json", {
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "frozen_gold_sha256": summary["frozen_gold_sha256"],
        "source_sha256": summary["source_sha256"],
        "stage": args.stage,
        "reviewers": args.reviewers if args.stage != "static" else "none",
        "max_workers": max(1, min(2, args.max_workers)),
        "gold_role": "development_only",
        "final_external_test": "BioRED official test",
    })
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(render_summary(summary), encoding="utf-8")
    print(args.output_dir)
    return 0 if summary["structural_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
