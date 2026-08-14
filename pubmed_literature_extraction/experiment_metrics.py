"""Shared experiment metric normalization for v1 baseline and v2/v3 agent."""
from __future__ import annotations

from typing import Any


def pct(num: int | float, den: int | float) -> float | None:
    return round(float(num) / float(den) * 100, 1) if den else None


def normalize_run(*, version: str, report: dict[str, Any], provider_errors: int = 0) -> dict[str, Any]:
    """Normalize both report shapes without turning provider failure into quality zero."""
    if version == "v1":
        records = int(report.get("records", 0))
        relations = int(report.get("relations", 0))
        schema_valid = int(report.get("schema_valid_relations", 0))
        ready = int(report.get("import_ready_relations", 0))
        errors = int(report.get("error_count", 0))
        return _base(records, report.get("entities", 0), relations, schema_valid, ready, errors, provider_errors,
                     report.get("review_relations", 0), report.get("zero_relation_articles", 0))
    records = int(report.get("total_articles", 0))
    extraction = report.get("extraction", {})
    decisions = report.get("decisions", {})
    quality = report.get("quality", {})
    relations = int(extraction.get("total_relations_extracted", 0))
    errors = int(quality.get("error_count", 0))
    if provider_errors or errors and not relations:
        schema_valid = ready = None
    else:
        schema_valid = int(extraction.get("schema_valid_relations", decisions.get("schema_valid", 0)))
        ready = int(decisions.get("total_import_ready", 0))
    return _base(records, extraction.get("total_entities_extracted", 0), relations, schema_valid, ready,
                 errors, provider_errors, decisions.get("discarded", 0), report.get("zero_relation_articles", 0))


def _base(records, entities, relations, schema_valid, ready, errors, provider_errors, review, zero):
    return {
        "records": records, "successful_records": records if not provider_errors else None,
        "entities": entities, "relations": relations, "schema_valid": schema_valid,
        "schema_valid_rate": pct(schema_valid, relations), "import_ready": ready,
        "import_ready_rate": pct(ready, relations), "review_or_discard": review,
        "zero_relation_articles": zero, "error_count": errors,
        "provider_error_count": provider_errors, "quality_status": "invalid_provider" if provider_errors else "ok",
    }
