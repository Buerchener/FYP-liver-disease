"""Select safe, presentation-ready fields from extraction result JSON."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _text(value: Any) -> str:
    return str(value) if value is not None else ""


def _status(value: Any) -> str:
    return _text(value).upper() or "UNKNOWN"


def _optional_bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _spans(relation: Mapping[str, Any]) -> list[dict[str, Any]]:
    pack = relation.get("evidence_pack")
    candidates = pack.get("spans") if isinstance(pack, Mapping) else None
    if not isinstance(candidates, list) or not candidates:
        candidates = relation.get("evidence_spans")
    if not isinstance(candidates, list):
        return []

    spans = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        text = candidate.get("text", candidate.get("evidence", ""))
        if not text:
            continue
        span = {
            "text": _text(text),
            "role": _text(candidate.get("role")) or "UNKNOWN",
            "alignment_status": _status(candidate.get("alignment_status")),
        }
        for key in ("char_start", "char_end"):
            value = candidate.get(key)
            if isinstance(value, int):
                span[key] = value
        spans.append(span)
    return spans


def _relation_data(
    relation: Mapping[str, Any], schema_name: str, schema_hash: str,
) -> dict[str, Any]:
    relation_schema_name = _text(relation.get("schema_profile_name")) or schema_name
    relation_schema_hash = _text(relation.get("schema_profile_sha256")) or schema_hash
    semantic_reasons = relation.get("semantic_reasons")
    if semantic_reasons is None:
        semantic_reasons = relation.get("review_reasons")

    return {
        "subject": _text(relation.get("subject")),
        "subject_type": _text(relation.get("subject_type")),
        "predicate": _text(relation.get("predicate")),
        "object": _text(relation.get("object")),
        "object_type": _text(relation.get("object_type")),
        "evidence": _text(relation.get("evidence")),
        "semantic_status": _status(relation.get("semantic_status")),
        "factual_status": _status(relation.get("factual_status")),
        "write_status": _status(relation.get("write_status")),
        "candidate_schema_valid": _optional_bool(
            relation.get("candidate_schema_valid"), relation.get("schema_valid"),
        ),
        "write_contract_valid": _optional_bool(relation.get("write_contract_valid")),
        "import_ready": _optional_bool(relation.get("import_ready")),
        "semantic_reasons": _string_list(semantic_reasons),
        "write_reasons": _string_list(relation.get("write_reasons")),
        "quality_flags": _string_list(relation.get("quality_flags")),
        "spans": _spans(relation),
        "schema_name": relation_schema_name,
        "schema_hash": relation_schema_hash,
    }


def build_review_data(payload: dict[str, Any], *, demo: bool = False) -> dict[str, Any]:
    """Return a minimal review payload from verified relation records only.

    Raw extraction proposals are deliberately excluded. Missing legacy statuses
    remain ``UNKNOWN`` and missing write booleans remain ``None`` so the review
    interface cannot imply a verification decision that the archive lacks.
    """
    if not isinstance(payload, dict):
        raise ValueError("Expected an agent results JSON object with a 'records' array.")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Invalid agent results JSON. Expected 'records' to be an array.")

    report = payload.get("report")
    report = report if isinstance(report, Mapping) else {}
    profile = report.get("schema_profile")
    profile = profile if isinstance(profile, Mapping) else {}
    schema_name = _text(profile.get("name"))
    schema_hash = _text(profile.get("manifest_sha256"))

    articles = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        phases = record.get("phases")
        phases = phases if isinstance(phases, Mapping) else {}
        verification = phases.get("verification")
        verification = verification if isinstance(verification, Mapping) else {}
        raw_relations = verification.get("relations")
        relations = [
            _relation_data(item, schema_name, schema_hash)
            for item in raw_relations
            if isinstance(item, Mapping)
        ] if isinstance(raw_relations, list) else []
        articles.append({
            "pmid": _text(record.get("pmid")),
            "title": _text(record.get("title")),
            "error": _text(record.get("error")),
            "relations": relations,
        })

    config = payload.get("config")
    config = config if isinstance(config, Mapping) else {}
    return {
        "run_id": _text(report.get("run_id")) or _text(config.get("run_id")),
        "schema_name": schema_name,
        "schema_hash": schema_hash,
        "mode": "demo" if demo else "run",
        "articles": articles,
    }
