"""Version-aware candidate lineage helpers.

The live tiered pipeline must never infer identity from list position.  This
module is intentionally dependency-light so every stage can normalize and
bind the same flat lineage contract without importing the Agent.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable


LINEAGE_CONTRACT_VERSION = "lineage-v2"
LANES = frozenset({"extracted_hint", "recovery"})


def lineage_key(value: dict[str, Any]) -> tuple[str, int] | None:
    candidate_id = str(value.get("candidate_id", "") or "")
    if not candidate_id:
        return None
    try:
        version = max(1, int(value.get("candidate_version", 1) or 1))
    except (TypeError, ValueError):
        version = 1
    return candidate_id, version


def normalize_lineage(
    value: dict[str, Any], *, lane: str | None = None,
    allow_legacy_lane_inference: bool = False,
) -> dict[str, Any]:
    """Return a copy carrying every lineage-v2 field.

    New live outputs must supply a lane.  The r-prefix inference exists only
    for read-only legacy cache replay and is explicitly audited.
    """
    output = copy.deepcopy(value)
    flags = set(output.get("quality_flags", []) or [])
    resolved_lane = str(lane or output.get("candidate_lane", "") or "")
    candidate_id = str(output.get("candidate_id", "") or "")
    if resolved_lane not in LANES and allow_legacy_lane_inference and candidate_id:
        resolved_lane = "recovery" if candidate_id.startswith("r-") else "extracted_hint"
        flags.add("lineage_migration_inferred")
    if resolved_lane not in LANES:
        resolved_lane = ""
        flags.add("lineage_binding_missing")

    try:
        version = max(1, int(output.get("candidate_version", 1) or 1))
    except (TypeError, ValueError):
        version = 1
        flags.add("lineage_version_invalid")
    try:
        parent_version = max(0, int(output.get("parent_version", 0) or 0))
    except (TypeError, ValueError):
        parent_version = 0
        flags.add("lineage_version_invalid")

    output.update({
        "lineage_contract_version": LINEAGE_CONTRACT_VERSION,
        "candidate_id": candidate_id,
        "candidate_version": version,
        "parent_candidate_id": str(output.get("parent_candidate_id", "") or ""),
        "parent_version": parent_version,
        "candidate_lane": resolved_lane,
        "pair_candidate_id": str(output.get("pair_candidate_id", "") or ""),
        "source_candidate_ids": _strings(
            output.get("source_candidate_ids", []) or ([candidate_id] if candidate_id else [])
        ),
        "source_lanes": _strings(
            output.get("source_lanes", []) or ([resolved_lane] if resolved_lane else [])
        ),
        "merged_candidate_ids": _strings(output.get("merged_candidate_ids", []) or []),
        "edit_reason_code": str(output.get("edit_reason_code", "") or ""),
        "quality_flags": sorted(flags),
    })
    return output


def edited_version(value: dict[str, Any], *, reason_code: str) -> dict[str, Any]:
    """Create the next version of a materially edited candidate."""
    output = normalize_lineage(value, allow_legacy_lane_inference=True)
    old_version = output["candidate_version"]
    output["parent_candidate_id"] = output["candidate_id"]
    output["parent_version"] = old_version
    output["candidate_version"] = old_version + 1
    output["edit_reason_code"] = str(reason_code or "candidate_edit")
    return output


def bind_by_lineage(
    candidates: Iterable[dict[str, Any]], decisions: Iterable[dict[str, Any]],
    *, legacy_index_compatibility: bool = False,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    """Bind decisions by (candidate_id, version), never by live array index.

    The optional index compatibility applies only when *both* records lack an
    ID.  It is therefore safe for historical records and impossible to use as
    a fallback for a partially identified live decision.
    """
    candidate_rows = list(candidates)
    decision_rows = list(decisions)
    candidate_keys = {key for row in candidate_rows if (key := lineage_key(row))}
    bound: dict[tuple[str, int], dict[str, Any]] = {}
    unbound: list[dict[str, Any]] = []
    for index, raw in enumerate(decision_rows):
        decision = copy.deepcopy(raw)
        key = lineage_key(decision)
        if key and key in candidate_keys:
            bound[key] = decision
            continue
        if (
            legacy_index_compatibility and not key and index < len(candidate_rows)
            and lineage_key(candidate_rows[index]) is None
        ):
            decision["_legacy_index_bound"] = True
            continue
        decision.setdefault("quality_flags", []).append("lineage_binding_missing")
        decision["quality_flags"] = sorted(set(decision["quality_flags"]))
        unbound.append(decision)
    return bound, unbound


def _strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})
