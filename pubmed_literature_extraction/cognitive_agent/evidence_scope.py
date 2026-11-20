"""Bounded, source-backed scope witnesses for evidence assertions.

Scope labels are descriptive report strata. In particular, different cohort or
model labels do not prove disjoint populations and never authorize promotion.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


_COHORT_ROLES = re.compile(
    r"\b(?P<value>external\s+validation|discovery|validation|training|testing|"
    r"test|control|case|replication|development)\s+cohort\b",
    re.IGNORECASE,
)
_SCOPE_LABEL = r"(?:[A-Z]|\d{1,3}|[A-Z][a-z]{2,})"
_COHORT_IDS = re.compile(rf"(?i:\bcohort)\s+(?P<value>{_SCOPE_LABEL})\b")
_COHORT_LIST = re.compile(
    rf"(?i:\bcohorts?)\s+(?P<values>{_SCOPE_LABEL}"
    rf"(?:\s*(?:,|and|or|/|&)\s*(?:cohorts?\s+)?{_SCOPE_LABEL})+)\b"
)
_PATIENT_CRITERION = re.compile(
    r"(?i:\b(?:in|among)\s+(?:[\w-]+\s+){0,3}(?:patients|participants)\s+(?P<polarity>without|with))\s+"
    r"(?P<criterion>[A-Z][A-Z0-9_.-]{1,})\s*,",
)
_SEX = re.compile(
    r"\b(?:in|among)\s+(?:the\s+)?(?P<value>men|women|males|females)\b|"
    r"\b(?P<binary>male|female)\s+(?:participants|patients|subjects|individuals|adults)\b",
    re.IGNORECASE,
)
_SEX_LIST = re.compile(
    r"(?i:\b(?:in|among)\s+(?:the\s+)?)(?P<values>(?:men|women|males|females)"
    r"(?:\s*(?:,|and|or|/|&)\s*(?:men|women|males|females))+\b)"
)
_MODEL_ID = re.compile(r"(?i:(?:analysis\s+)?model)\s+(?P<value>[A-Z]|\d{1,3})\b")
_MODEL_LIST = re.compile(
    rf"(?i:\bmodels?)\s+(?P<values>{_SCOPE_LABEL}"
    rf"(?:\s*(?:,|and|or|/|&)\s*(?:models?\s+)?{_SCOPE_LABEL})+)\b"
)
_ADJUSTMENT = re.compile(
    r"(?i:\bmodel(?:\s+(?:[A-Z]|\d{1,3}))?|\banalysis)\s+"
    r"(?P<value>unadjusted|adjusted)\b|"
    r"\b(?P<prefix>unadjusted|adjusted)\s+(?i:model|analysis)"
    r"(?:\s+(?:[A-Z]|\d{1,3}))?\b"
)
_LABEL_TOKEN = re.compile(rf"\b{_SCOPE_LABEL}\b")


def _value(dimension: str, raw: str) -> str:
    folded = re.sub(r"\s+", " ", raw.strip().casefold())
    if dimension == "sex":
        return "male" if folded in {"men", "males", "male"} else "female"
    if dimension in {"cohort_id", "model_id"}:
        return folded.upper()
    return folded


def extract_scope_witnesses(text: str, *, char_offset: int = 0) -> list[dict[str, Any]]:
    """Extract explicit cohort, sex, model-ID and adjustment witnesses.

    Every returned span is an exact slice of ``text`` shifted by char_offset.
    Unknown values are not inferred from nearby sentences or article metadata.
    """
    found: list[dict[str, Any]] = []

    def add(dimension: str, raw_value: str, start: int, end: int) -> None:
        found.append({
            "dimension": dimension,
            "value": _value(dimension, raw_value),
            "text": text[start:end],
            "char_start": char_offset + start,
            "char_end": char_offset + end,
        })

    for match in _COHORT_ROLES.finditer(text):
        add("cohort_role", match.group("value"), *match.span())

    for pattern in (_COHORT_IDS, _COHORT_LIST):
        for match in pattern.finditer(text):
            if pattern is _COHORT_IDS:
                value_start, value_end = match.span("value")
                add("cohort_id", match.group("value"), match.start(), value_end)
            else:
                values_start, _ = match.span("values")
                for token in _LABEL_TOKEN.finditer(match.group("values")):
                    start, end = values_start + token.start(), values_start + token.end()
                    add("cohort_id", token.group(), start, end)

    for match in _PATIENT_CRITERION.finditer(text):
        criterion = match.group("criterion").casefold()
        polarity = match.group("polarity").casefold()
        start = match.start("polarity")
        end = match.end("criterion")
        add(f"patient_criterion:{criterion}", polarity, start, end)

    for match in _SEX.finditer(text):
        value_group = "value" if match.group("value") else "binary"
        start, end = match.span(value_group)
        add("sex", match.group(value_group), start, end)
    for match in _SEX_LIST.finditer(text):
        values_start, _ = match.span("values")
        for token in re.finditer(r"(?i)men|women|males|females", match.group("values")):
            start, end = values_start + token.start(), values_start + token.end()
            add("sex", token.group(), start, end)

    for pattern in (_MODEL_ID, _MODEL_LIST):
        for match in pattern.finditer(text):
            if pattern is _MODEL_ID:
                add("model_id", match.group("value"), *match.span())
            else:
                values_start, _ = match.span("values")
                for token in _LABEL_TOKEN.finditer(match.group("values")):
                    start, end = values_start + token.start(), values_start + token.end()
                    add("model_id", token.group(), start, end)

    for match in _ADJUSTMENT.finditer(text):
        group = "value" if match.group("value") else "prefix"
        start, end = match.span(group)
        add("adjustment", match.group(group), start, end)

    # A role and ID are distinct dimensions ("discovery cohort A").
    # Enumerated labels retain multiple same-dimension witnesses so comparison
    # can conservatively return UNRESOLVED_SCOPE.
    unique = {
        (item["dimension"], item["value"], item["char_start"], item["char_end"]): item
        for item in found
    }
    return sorted(unique.values(), key=lambda item: (item["char_start"], item["char_end"], item["dimension"]))


def normalize_scope(witnesses: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return normalized dimension values without discarding ambiguity."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for witness in witnesses:
        grouped[str(witness["dimension"])].add(str(witness["value"]))
    return {key: sorted(values) for key, values in sorted(grouped.items())}


def compare_scope(
    left_witnesses: list[dict[str, Any]], right_witnesses: list[dict[str, Any]],
) -> str:
    """Compare only explicit same-dimension, single-valued scope witnesses."""
    left = normalize_scope(left_witnesses)
    right = normalize_scope(right_witnesses)
    if not left or not right or left.keys() != right.keys():
        return "UNRESOLVED_SCOPE"
    if any(len(values) != 1 for values in (*left.values(), *right.values())):
        return "UNRESOLVED_SCOPE"
    if all(left[key] == right[key] for key in left):
        return "SHARED_SCOPE"
    # A single changed label is a distinct reported stratum, never proof that
    # the underlying samples are disjoint. Multiple changed dimensions remain
    # unresolved because their joint interpretation is ambiguous.
    differing = [key for key in left if left[key] != right[key]]
    if len(differing) != 1:
        return "UNRESOLVED_SCOPE"
    return "DIFFERENT_SCOPE"
