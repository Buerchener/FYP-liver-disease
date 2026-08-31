#!/usr/bin/env python3
"""Source-grounded, multi-span evidence packs for relation candidates."""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable

from cognitive_agent.evidence_units import ArticleEvidenceReader, EvidenceUnit
from cognitive_agent.extraction_quality import locate_contiguous, normalize_surface
from cognitive_agent.schema.predicate_cards import predicate_support_match


@dataclass(frozen=True)
class EvidenceSpan:
    span_id: str
    text: str
    char_start: int
    char_end: int
    sentence_id: str
    role: str = "OWNER"
    alignment_status: str = "MATCH_EXACT"
    subject_covered: bool = False
    object_covered: bool = False
    trigger_covered: bool = False
    trigger_match: str = "NONE"
    trigger_reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class EvidencePack:
    source_sha256: str
    spans: list[EvidenceSpan] = field(default_factory=list)
    owner_sentence_ids: list[str] = field(default_factory=list)
    context_sentence_ids: list[str] = field(default_factory=list)
    alias_resolution: dict[str, list[str]] = field(default_factory=dict)
    coreference_path: list[str] = field(default_factory=list)
    subject_covered: bool = False
    object_covered: bool = False
    trigger_covered: bool = False
    source_traceable: bool = False
    support_mode: str = "UNRESOLVED"
    minimal_support_span_ids: list[str] = field(default_factory=list)
    context_span_ids: list[str] = field(default_factory=list)
    support_sentence_ids: list[str] = field(default_factory=list)
    support_subject_covered: bool = False
    support_object_covered: bool = False
    support_trigger_covered: bool = False
    support_trigger_match: str = "NONE"
    support_trigger_reason_codes: list[str] = field(default_factory=list)
    resolution_steps: list[str] = field(default_factory=list)
    support_closure_reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "spans": [item.to_dict() for item in self.spans],
            "owner_sentence_ids": list(self.owner_sentence_ids),
            "context_sentence_ids": list(self.context_sentence_ids),
            "alias_resolution": dict(self.alias_resolution),
            "coreference_path": list(self.coreference_path),
            "subject_covered": self.subject_covered,
            "object_covered": self.object_covered,
            "trigger_covered": self.trigger_covered,
            "source_traceable": self.source_traceable,
            "support_mode": self.support_mode,
            "minimal_support_span_ids": list(self.minimal_support_span_ids),
            "context_span_ids": list(self.context_span_ids),
            "support_sentence_ids": list(self.support_sentence_ids),
            "support_subject_covered": self.support_subject_covered,
            "support_object_covered": self.support_object_covered,
            "support_trigger_covered": self.support_trigger_covered,
            "support_trigger_match": self.support_trigger_match,
            "support_trigger_reason_codes": list(self.support_trigger_reason_codes),
            "resolution_steps": list(self.resolution_steps),
            "support_closure_reason_codes": list(self.support_closure_reason_codes),
        }


def _aliases(relation: dict, role: str, aliases_by_canonical: dict[str, list[str]]) -> list[str]:
    value = str(relation.get(role, "") or "").strip()
    family = str(relation.get(f"{role}_family", "") or "").strip()
    candidates = [
        value,
        family,
        *(aliases_by_canonical.get(value, []) or []),
        *(aliases_by_canonical.get(family, []) or []),
    ]
    return list(dict.fromkeys(item for item in candidates if item))


def _contains_alias(text: str, aliases: list[str]) -> bool:
    normalized = normalize_surface(text)
    return any(normalize_surface(alias) in normalized for alias in aliases if alias)


COREFERENCE_RE = re.compile(
    r"(?i)\b(?:it|its|they|them|their|these|those|both|the former|the latter|"
    r"this (?:gene|protein|factor|pathway|disease|condition)|"
    r"these (?:genes|proteins|factors|pathways|diseases|conditions))\b"
)


def _directly_contains(text: str, value: str) -> bool:
    return bool(value and normalize_surface(value) in normalize_surface(text))


def _looks_like_abbreviation(value: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", str(value or ""))
    return bool(2 <= len(compact) <= 12 and any(char.isupper() for char in str(value)))


def _token_similarity(left: str, right: str) -> float:
    left_tokens = re.findall(r"\w+", left.casefold())
    right_tokens = re.findall(r"\w+", right.casefold())
    if not left_tokens or not right_tokens:
        return 0.0
    return SequenceMatcher(None, left_tokens, right_tokens, autojunk=False).ratio()


class EvidencePackBuilder:
    """Build at most three exact/fuzzy source spans without semantic vetoes."""

    def __init__(
        self,
        fuzzy_threshold: float = 0.75,
        max_spans: int = 3,
        *,
        support_matcher: Callable[..., dict[str, Any]] | None = None,
    ):
        self.fuzzy_threshold = max(0.5, min(1.0, float(fuzzy_threshold)))
        self.max_spans = max(1, min(3, int(max_spans)))
        self.reader = ArticleEvidenceReader()
        # LiverKG remains the default.  External given-entity benchmarks can
        # inject their native predicate registry without mapping labels into
        # the LiverKG ontology.
        self.support_matcher = support_matcher or predicate_support_match

    def _support_match(
        self,
        relation: dict,
        evidence: str,
        subject_aliases: list[str],
        object_aliases: list[str],
    ) -> dict[str, Any]:
        return self.support_matcher(
            str(relation.get("predicate", "") or ""),
            str(relation.get("subject_type", "") or ""),
            str(relation.get("object_type", "") or ""),
            evidence,
            subject_aliases=subject_aliases,
            object_aliases=object_aliases,
        )

    @staticmethod
    def _sentence_role(relation: dict, sentence_id: str) -> str:
        owners = set(relation.get("owner_sentence_ids", []) or [])
        contexts = set(relation.get("context_sentence_ids", []) or [])
        if sentence_id in contexts and sentence_id not in owners:
            return "CONTEXT"
        return "OWNER"

    def _span(
        self,
        unit: EvidenceUnit,
        relation: dict,
        subject_aliases: list[str],
        object_aliases: list[str],
        alignment_status: str,
    ) -> EvidenceSpan:
        support = self._support_match(
            relation, unit.text, subject_aliases, object_aliases,
        )
        trigger_match = str(support.get("match", "NONE"))
        return EvidenceSpan(
            span_id=f"ep-{unit.parent_sentence_id}-{unit.char_start}-{unit.char_end}",
            text=unit.text,
            char_start=unit.char_start,
            char_end=unit.char_end,
            sentence_id=unit.parent_sentence_id,
            role=self._sentence_role(relation, unit.parent_sentence_id),
            alignment_status=alignment_status,
            subject_covered=_contains_alias(unit.text, subject_aliases),
            object_covered=_contains_alias(unit.text, object_aliases),
            trigger_covered=trigger_match in {"EXPLICIT", "WEAK"},
            trigger_match=trigger_match,
            trigger_reason_codes=tuple(support.get("reason_codes", []) or []),
        )

    @staticmethod
    def _support_rank(
        spans: tuple[EvidenceSpan, ...], primary_span_ids: set[str],
    ) -> tuple[Any, ...]:
        return (
            len(spans),
            -max(
                ({"EXPLICIT": 2, "WEAK": 1}.get(item.trigger_match, 0) for item in spans),
                default=0,
            ),
            sum(item.alignment_status != "MATCH_EXACT" for item in spans),
            -sum(item.span_id in primary_span_ids for item in spans),
            sum(max(0, item.char_end - item.char_start) for item in spans),
            tuple(item.sentence_id for item in spans),
        )

    def _finalize_support(
        self,
        pack: EvidencePack,
        relation: dict,
        *,
        primary_span_ids: set[str],
    ) -> None:
        """Choose the deterministic minimal owner-only support closure."""
        owner_spans = [item for item in pack.spans if item.role == "OWNER"]
        pack.context_span_ids = [item.span_id for item in pack.spans if item.role == "CONTEXT"]
        subject_aliases = list((pack.alias_resolution or {}).get("subject", []) or [])
        object_aliases = list((pack.alias_resolution or {}).get("object", []) or [])
        predicate = str(relation.get("predicate", "") or "").upper()

        self_contained = [
            (item,) for item in owner_spans
            if (
                item.subject_covered
                and item.object_covered
                and item.trigger_match in {"EXPLICIT", "WEAK"}
            )
        ]
        candidates: list[tuple[EvidenceSpan, ...]] = list(self_contained)
        if not candidates:
            for size in range(2, min(self.max_spans, len(owner_spans)) + 1):
                for combo in itertools.combinations(owner_spans, size):
                    if (
                        any(item.subject_covered for item in combo)
                        and any(item.object_covered for item in combo)
                        and any(item.trigger_covered for item in combo)
                    ):
                        candidates.append(combo)
                if candidates:
                    break

        if not candidates:
            pack.support_mode = "UNRESOLVED"
            pack.support_closure_reason_codes = ["owner_support_not_closed"]
            return

        chosen = tuple(sorted(
            min(candidates, key=lambda items: self._support_rank(items, primary_span_ids)),
            key=lambda item: (item.char_start, item.char_end, item.span_id),
        ))
        pack.minimal_support_span_ids = [item.span_id for item in chosen]
        pack.support_sentence_ids = list(dict.fromkeys(item.sentence_id for item in chosen))
        pack.support_subject_covered = any(item.subject_covered for item in chosen)
        pack.support_object_covered = any(item.object_covered for item in chosen)
        pack.support_trigger_covered = any(item.trigger_covered for item in chosen)
        trigger_rank = {"CONFLICT": -1, "NONE": 0, "WEAK": 1, "EXPLICIT": 2}
        pack.support_trigger_match = max(
            (item.trigger_match for item in chosen),
            key=lambda value: trigger_rank.get(value, 0),
            default="NONE",
        )
        pack.support_trigger_reason_codes = list(dict.fromkeys(
            reason for item in chosen for reason in item.trigger_reason_codes
        ))

        resolution_steps: set[str] = set()
        support_text = " ".join(item.text for item in chosen)
        for role in ("subject", "object"):
            value = str(relation.get(role, "") or "")
            aliases = list((pack.alias_resolution or {}).get(role, []) or [])
            if not _directly_contains(support_text, value) and _contains_alias(support_text, aliases):
                resolution_steps.add(
                    "ABBREVIATION"
                    if any(_looks_like_abbreviation(alias) for alias in aliases)
                    else "ALIAS"
                )
        if bool(set(relation.get("quality_flags", []) or []) & {"composite_endpoint", "schema_gap"}):
            resolution_steps.add("COMPOSITE_ENDPOINT")

        trigger_spans = [item for item in chosen if item.trigger_covered]
        endpoint_missing_in_trigger_span = bool(trigger_spans) and not any(
            item.subject_covered and item.object_covered for item in trigger_spans
        )
        if len(chosen) > 1 and endpoint_missing_in_trigger_span and COREFERENCE_RE.search(support_text):
            resolution_steps.add("COREFERENCE")

        pack.resolution_steps = sorted(resolution_steps)
        if len(chosen) == 1:
            pack.support_mode = "SELF_CONTAINED"
            pack.support_closure_reason_codes = ["single_owner_span_closes_relation"]
        elif "COREFERENCE" in resolution_steps:
            pack.support_mode = "COREFERENCE"
            pack.support_closure_reason_codes = ["coreference_resolved_across_owner_spans"]
            pack.coreference_path = list(pack.support_sentence_ids)
        else:
            pack.support_mode = "MULTI_SPAN"
            pack.support_closure_reason_codes = ["multiple_owner_spans_required"]

    def build(
        self,
        relation: dict,
        *,
        text: str,
        aliases_by_canonical: dict[str, list[str]] | None = None,
    ) -> EvidencePack:
        aliases_by_canonical = aliases_by_canonical or {}
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
        pack = EvidencePack(source_sha256=source_hash)
        subject_aliases = _aliases(relation, "subject", aliases_by_canonical)
        object_aliases = _aliases(relation, "object", aliases_by_canonical)
        pack.alias_resolution = {
            "subject": subject_aliases,
            "object": object_aliases,
        }
        if not text:
            return pack

        units = self.reader.read(text)
        parents = self.reader.parent_units(text, units)
        selected: list[tuple[EvidenceUnit, str]] = []
        primary_ranges: set[tuple[int, int]] = set()
        evidence_values = list(dict.fromkeys(
            str(item or "").strip()
            for item in [
                relation.get("evidence", ""),
                *(relation.get("evidence_candidates", []) or []),
            ]
            if str(item or "").strip()
        ))

        for evidence in evidence_values[: self.max_spans]:
            grounded, start, end = locate_contiguous(evidence, text)
            if grounded:
                containing = next(
                    (item for item in parents if item.char_start <= start and end <= item.char_end),
                    None,
                )
                if containing is not None:
                    exact_units = [containing]
                else:
                    # A quote may cover multiple source sentences.  Preserve
                    # their sentence identities instead of synthesizing one
                    # pseudo-span that would look self-contained.
                    exact_units = [
                        item for item in parents
                        if item.char_end > start and item.char_start < end
                    ]
                    if not exact_units:
                        exact_units = [EvidenceUnit(
                            unit_id="evidence", section="ABSTRACT", text=text[start:end],
                            char_start=start, char_end=end, parent_sentence_id="evidence",
                        )]
                for exact_unit in exact_units:
                    selected.append((exact_unit, "MATCH_EXACT"))
                    primary_ranges.add((exact_unit.char_start, exact_unit.char_end))
            else:
                fuzzy = max(parents, key=lambda item: _token_similarity(evidence, item.text), default=None)
                score = _token_similarity(evidence, fuzzy.text) if fuzzy else 0.0
                if fuzzy and score >= self.fuzzy_threshold:
                    selected.append((fuzzy, "MATCH_FUZZY"))
                    primary_ranges.add((fuzzy.char_start, fuzzy.char_end))

        # Repair incomplete/empty quotes from source sentences.  First prefer a
        # single sentence closing both endpoints; then use an adjacent pair.
        both = [
            item for item in parents
            if _contains_alias(item.text, subject_aliases)
            and _contains_alias(item.text, object_aliases)
        ]
        both.sort(key=lambda item: (
            -{"EXPLICIT": 2, "WEAK": 1}.get(
                self._support_match(
                    relation, item.text, subject_aliases, object_aliases,
                ).get("match", "NONE"),
                0,
            ),
            len(item.text), item.char_start,
        ))
        for item in both:
            selected.append((item, "MATCH_EXACT"))

        ordered = sorted(parents, key=lambda item: item.char_start)
        for index, first in enumerate(ordered[:-1]):
            second = ordered[index + 1]
            combined = first.text + " " + second.text
            if not (
                _contains_alias(combined, subject_aliases)
                and _contains_alias(combined, object_aliases)
            ):
                continue
            if self._support_match(
                relation, combined, subject_aliases, object_aliases,
            ).get("match") not in {"EXPLICIT", "WEAK"}:
                continue
            selected.extend(((first, "MATCH_EXACT"), (second, "MATCH_EXACT")))

        # Document-level relations often state the predicate next to one
        # endpoint while naming the other endpoint in a non-adjacent title or
        # setup sentence.  Preserve the model's exact quote, then add the
        # nearest owner sentence for each still-missing endpoint.  The final
        # support search remains capped at three spans and therefore cannot
        # turn arbitrary whole-document co-occurrence into an unbounded pack.
        selected_ranges = [(item.char_start, item.char_end) for item, _ in selected]
        anchor = min(
            (abs(item.char_start - start) for item, _ in selected for start, _ in primary_ranges),
            default=0,
        )
        for aliases in (subject_aliases, object_aliases):
            endpoint_parents = [
                item for item in parents if _contains_alias(item.text, aliases)
            ]
            endpoint_parents.sort(key=lambda item: (
                min(
                    (abs(item.char_start - start) for start, _ in primary_ranges),
                    default=anchor,
                ),
                len(item.text),
                item.char_start,
            ))
            if endpoint_parents:
                candidate = endpoint_parents[0]
                if (candidate.char_start, candidate.char_end) not in selected_ranges:
                    selected.append((candidate, "MATCH_EXACT"))
                    selected_ranges.append((candidate.char_start, candidate.char_end))

        unique: dict[tuple[int, int], tuple[EvidenceUnit, str]] = {}
        for unit, status in selected:
            key = (unit.char_start, unit.char_end)
            previous = unique.get(key)
            if previous is None or previous[1] != "MATCH_EXACT":
                unique[key] = (unit, status)
        ranked = sorted(unique.values(), key=lambda item: (
            -{"EXPLICIT": 2, "WEAK": 1}.get(
                self._support_match(
                    relation, item[0].text, subject_aliases, object_aliases,
                ).get("match", "NONE"),
                0,
            ),
            item[1] != "MATCH_EXACT",
            (item[0].char_start, item[0].char_end) not in primary_ranges,
            not (
                _contains_alias(item[0].text, subject_aliases)
                and _contains_alias(item[0].text, object_aliases)
            ),
            item[0].char_start,
        ))[:self.max_spans]
        pack.spans = [
            self._span(unit, relation, subject_aliases, object_aliases, status)
            for unit, status in ranked
        ]
        pack.owner_sentence_ids = list(dict.fromkeys(
            item.sentence_id for item in pack.spans if item.role == "OWNER"
        ))
        pack.context_sentence_ids = list(dict.fromkeys(
            item.sentence_id for item in pack.spans if item.role == "CONTEXT"
        ))
        pack.subject_covered = any(item.subject_covered for item in pack.spans)
        pack.object_covered = any(item.object_covered for item in pack.spans)
        pack.trigger_covered = any(item.trigger_covered for item in pack.spans)
        pack.source_traceable = bool(pack.spans)
        primary_span_ids = {
            item.span_id for item in pack.spans
            if (item.char_start, item.char_end) in primary_ranges
        }
        self._finalize_support(pack, relation, primary_span_ids=primary_span_ids)
        return pack
