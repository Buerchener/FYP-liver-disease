#!/usr/bin/env python3
"""Evidence-first selection and closed-label entailment for Agent v3.

Every selected span is a contiguous substring of the source.  Auxiliary models
may classify bounded candidates, but cannot rewrite evidence or introduce an
entity endpoint.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_agent.aux_model_registry import AuxModelRegistry
from cognitive_agent.evidence_units import EvidenceUnit


ENTAILED = "ENTAILED"
CONTRADICTED = "CONTRADICTED"
NOT_ENOUGH_INFORMATION = "NOT_ENOUGH_INFORMATION"
ENTAILMENT_LABELS = frozenset({ENTAILED, CONTRADICTED, NOT_ENOUGH_INFORMATION})
ASSERTIVE_SECTIONS = frozenset({"RESULT", "RESULTS", "CONCLUSION", "CONCLUSIONS", "DISCUSSION"})
NON_ASSERTIVE_SECTIONS = frozenset({
    "BACKGROUND", "INTRODUCTION", "OBJECTIVE", "OBJECTIVES", "AIM", "AIMS",
    "PURPOSE", "METHOD", "METHODS", "MATERIALS_AND_METHODS",
})
NEGATION_RE = re.compile(r"\b(?:no|not|neither|without|failed to|did not|lack(?:ed|s)?)\b", re.I)

TRIGGER_PATTERNS: dict[str, tuple[str, ...]] = {
    "ENCODES": (r"\bencod(?:e|es|ed|ing)\b",),
    "PROGNOSTIC_IN": (r"\bprognos(?:is|tic)\b", r"\bsurvival\b", r"\brecurren\w*\b"),
    "PROGRESSES_TO": (r"\bprogress\w*\s+(?:in)?to\b", r"\bevolv\w*\s+into\b"),
    "INTERACTS_WITH": (
        r"\binteract\w*\s+with\b", r"\bbind\w*\s+(?:to|with)\b",
        r"\bcross[- ]?talk\b", r"\bcell(?:ular)?[- ]cell communication\b",
        r"\bjuxtapos\w*\b",
    ),
    "PARTICIPATES_IN": (r"\bparticipat\w*\s+in\b", r"\b(?:regulat|mediat|activat|inhibit)\w*\b"),
    "EXPRESSED_IN": (
        r"\bexpress\w*\s+(?:in|by|within)\b",
        r"\b(?:high|low)?\s*expression\s+of\b.{0,100}\b(?:in|within)\b",
        r"\bsource\s+of\b",
        r"\blocali[sz]\w*\s+(?:in|to)\b",
    ),
    "ASSOCIATED_WITH_METABOLITE": (r"\bmetabol\w*\b", r"\b(?:associat|correlat)\w*\b"),
    "ASSOCIATED_WITH": (r"\bassociat\w*\b", r"\bcorrelat\w*\b", r"\blinked\s+to\b", r"\brelated\s+to\b"),
}


@dataclass(frozen=True)
class SelectedEvidence:
    candidate_id: str
    unit_id: str
    section: str
    text: str
    char_start: int
    char_end: int
    subject_span: tuple[int, int]
    object_span: tuple[int, int]
    trigger_span: tuple[int, int] | None
    local_label: str
    evidence_confidence: float
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntailmentDecision:
    candidate_id: str
    label: str
    confidence: float
    source: str
    quoted_span: str = ""
    quote_char_start: int = -1
    quote_char_end: int = -1
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceSelector:
    """Find the shortest exact unit containing both endpoints and a trigger."""

    @staticmethod
    def _mention_spans(text: str, mentions: list[str]) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for mention in dict.fromkeys(str(item).strip() for item in mentions if str(item).strip()):
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(mention) + r"(?![A-Za-z0-9])", re.I)
            spans.extend((match.start(), match.end()) for match in pattern.finditer(text))
        return sorted(set(spans))

    @staticmethod
    def _trigger_spans(text: str, predicate: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for pattern in TRIGGER_PATTERNS.get(predicate, ()):
            spans.extend((match.start(), match.end()) for match in re.finditer(pattern, text, re.I))
        return sorted(set(spans))

    @staticmethod
    def _minimal_window(
        subject_spans: list[tuple[int, int]], object_spans: list[tuple[int, int]],
        trigger_spans: list[tuple[int, int]],
        *, text: str = "", other_spans: list[tuple[int, int]] | None = None,
    ) -> tuple[int, int, tuple[int, int], tuple[int, int], tuple[int, int] | None, bool]:
        """Return the smallest pair-local span without borrowing a third endpoint's cue.

        Association wording is often present in a sentence for a *different*
        entity pair.  A trigger is therefore usable only when no other known
        entity mention in the same clause is closer to it than either proposed
        endpoint.  Ambiguous triggers are removed, but the endpoint pair is
        retained as NEI for bounded review.
        """
        other_spans = list(other_spans or [])

        def distance(left: tuple[int, int], right: tuple[int, int]) -> int:
            if left[1] < right[0]:
                return right[0] - left[1]
            if right[1] < left[0]:
                return left[0] - right[1]
            return 0

        def clause_bounds(trigger: tuple[int, int]) -> tuple[int, int]:
            if not text:
                return 0, 0
            starts = [text.rfind(mark, 0, trigger[0]) for mark in (".", ";", "\n")]
            start = max(starts) + 1
            ends = [
                index for index in (text.find(mark, trigger[1]) for mark in (".", ";", "\n"))
                if index >= 0
            ]
            return start, min(ends) if ends else len(text)

        def attached(
            subject: tuple[int, int], obj: tuple[int, int], trigger: tuple[int, int],
        ) -> bool:
            if not other_spans:
                return True
            clause_start, clause_end = clause_bounds(trigger)
            endpoint_distance = max(distance(trigger, subject), distance(trigger, obj))
            for other in other_spans:
                if other in {subject, obj} or not (clause_start <= other[0] < clause_end):
                    continue
                if distance(trigger, other) < endpoint_distance:
                    return False
            return True

        usable: list[tuple[int, int]] = []
        ambiguous_trigger = False
        for trigger in trigger_spans:
            if any(attached(subject, obj, trigger) for subject in subject_spans for obj in object_spans):
                usable.append(trigger)
            else:
                ambiguous_trigger = True
        triggers: list[tuple[int, int] | None] = usable or [None]
        candidates = []
        for subject in subject_spans:
            for obj in object_spans:
                for trigger in triggers:
                    if trigger is not None and not attached(subject, obj, trigger):
                        continue
                    spans = [subject, obj, *([trigger] if trigger else [])]
                    start = min(span[0] for span in spans)
                    end = max(span[1] for span in spans)
                    candidates.append((end - start, start, end, subject, obj, trigger))
        _, start, end, subject, obj, trigger = min(candidates)
        return start, end, subject, obj, trigger, ambiguous_trigger and trigger is None

    def select(
        self, *, candidate_id: str, subject_mentions: list[str], object_mentions: list[str],
        predicate: str, units: list[EvidenceUnit], source_text: str,
        other_mentions: list[str] | None = None,
    ) -> SelectedEvidence | None:
        selections: list[tuple[tuple[Any, ...], SelectedEvidence]] = []
        for unit in units:
            subjects = self._mention_spans(unit.text, subject_mentions)
            objects = self._mention_spans(unit.text, object_mentions)
            if not subjects or not objects:
                continue
            triggers = self._trigger_spans(unit.text, predicate)
            others = self._mention_spans(unit.text, list(other_mentions or []))
            start, end, subject, obj, trigger, ambiguous_trigger = self._minimal_window(
                subjects, objects, triggers, text=unit.text, other_spans=others,
            )
            prefix_start = max(0, start - 32)
            prefix_negations = list(NEGATION_RE.finditer(unit.text[prefix_start:start]))
            if prefix_negations:
                start = prefix_start + prefix_negations[-1].start()
            absolute_start, absolute_end = unit.char_start + start, unit.char_start + end
            span = source_text[absolute_start:absolute_end]
            if span != unit.text[start:end]:
                continue
            reasons = ["both_endpoints", "continuous_source_span", "minimal_sufficient_span"]
            if trigger:
                reasons.append("explicit_predicate_trigger")
            elif ambiguous_trigger:
                reasons.append("trigger_attachment_ambiguous")
            if unit.section in ASSERTIVE_SECTIONS:
                reasons.append("assertive_section")
            if NEGATION_RE.search(span):
                label, confidence = CONTRADICTED, 0.96
                reasons.append("negation_signal")
            elif not trigger or unit.section in NON_ASSERTIVE_SECTIONS:
                label, confidence = NOT_ENOUGH_INFORMATION, 0.58 if trigger else 0.42
                reasons.append("non_assertive_or_missing_trigger")
            else:
                label, confidence = ENTAILED, 0.92
            selected = SelectedEvidence(
                candidate_id=candidate_id, unit_id=unit.unit_id, section=unit.section,
                text=span, char_start=absolute_start, char_end=absolute_end,
                subject_span=(unit.char_start + subject[0], unit.char_start + subject[1]),
                object_span=(unit.char_start + obj[0], unit.char_start + obj[1]),
                trigger_span=(
                    (unit.char_start + trigger[0], unit.char_start + trigger[1]) if trigger else None
                ),
                local_label=label, evidence_confidence=confidence, reason_codes=reasons,
            )
            # A valid trigger dominates endpoint-only co-occurrence; then prefer
            # result/conclusion evidence and the shortest contiguous span.
            rank = (not bool(trigger), unit.section not in ASSERTIVE_SECTIONS, len(span), unit.unit_id)
            selections.append((rank, selected))
        return min(selections, key=lambda item: item[0])[1] if selections else None


class EvidenceEntailmentEngine:
    """Local-first triage; DeepSeek handles only NEI and Qwen only conflicts."""

    def __init__(self, registry: AuxModelRegistry | None = None):
        self.registry = registry

    @staticmethod
    def _align_quote(source_text: str, quote: str) -> tuple[int, int]:
        quote = str(quote or "").strip()
        if not quote:
            return -1, -1
        start = source_text.find(quote)
        return (start, start + len(quote)) if start >= 0 else (-1, -1)

    def assess(
        self, selections: list[SelectedEvidence], *, source_text: str,
        candidate_payloads: dict[str, dict[str, Any]], allow_remote: bool,
    ) -> tuple[list[EntailmentDecision], dict[str, Any]]:
        decisions: dict[str, EntailmentDecision] = {}
        uncertain = []
        for item in selections:
            if item.local_label != NOT_ENOUGH_INFORMATION:
                decisions[item.candidate_id] = EntailmentDecision(
                    item.candidate_id, item.local_label, item.evidence_confidence,
                    "local_rule", item.text, item.char_start, item.char_end, item.reason_codes,
                )
            else:
                uncertain.append(item)
        audit: dict[str, Any] = {
            "local_decisions": len(decisions), "remote_candidates": len(uncertain),
            "deepseek": {}, "qwen": {}, "closed_labels": sorted(ENTAILMENT_LABELS),
        }
        if uncertain and allow_remote and self.registry and self.registry.configured("primary"):
            bounded = [{
                "candidate_id": item.candidate_id,
                "subject": candidate_payloads.get(item.candidate_id, {}).get("subject", ""),
                "predicate": candidate_payloads.get(item.candidate_id, {}).get("predicate", ""),
                "object": candidate_payloads.get(item.candidate_id, {}).get("object", ""),
                "evidence": item.text,
                "section": item.section,
            } for item in uncertain]
            result = self.registry.call_json(
                "primary",
                system_prompt=(
                    "Classify only the supplied bounded biomedical relation candidates as "
                    "ENTAILED, CONTRADICTED, or NOT_ENOUGH_INFORMATION. Do not add entities, "
                    "relations, or rewrite quotes. Return JSON only."
                ),
                user_prompt=str({"candidates": bounded}),
                schema_hint={"decisions": [{"candidate_id": "", "label": "", "quote": ""}]},
            )
            audit["deepseek"] = result.to_dict()
            if result.status == "OK":
                by_id = {item.candidate_id: item for item in uncertain}
                for raw in result.payload.get("decisions", []) or []:
                    candidate_id = str(raw.get("candidate_id", ""))
                    label = str(raw.get("label", "")).upper()
                    quote = str(raw.get("quote", ""))
                    selected = by_id.get(candidate_id)
                    start, end = self._align_quote(source_text, quote)
                    if not selected or label not in ENTAILMENT_LABELS or start < 0:
                        continue
                    if not (selected.char_start <= start < end <= selected.char_end):
                        continue
                    decisions[candidate_id] = EntailmentDecision(
                        candidate_id, label, 0.72, "deepseek", quote, start, end,
                        ["remote_bounded_adjudication", "quote_realigned"],
                    )
                conflicts = [
                    item for item in uncertain
                    if decisions.get(item.candidate_id)
                    and decisions[item.candidate_id].label == CONTRADICTED
                    and candidate_payloads.get(item.candidate_id, {}).get("predicate")
                ]
                if conflicts and self.registry.configured("critic"):
                    critic = self.registry.call_json(
                        "critic",
                        system_prompt=(
                            "Independently review only the supplied evidence-versus-predicate "
                            "conflicts. Use ENTAILED, CONTRADICTED, or NOT_ENOUGH_INFORMATION. "
                            "Quote an unchanged substring and return JSON only."
                        ),
                        user_prompt=str({"conflicts": [{
                            **candidate_payloads[item.candidate_id],
                            "candidate_id": item.candidate_id,
                            "evidence": item.text,
                            "deepseek_label": decisions[item.candidate_id].label,
                        } for item in conflicts]}),
                        schema_hint={"decisions": [{"candidate_id": "", "label": "", "quote": ""}]},
                    )
                    audit["qwen"] = critic.to_dict()
                    if critic.status == "OK":
                        conflict_by_id = {item.candidate_id: item for item in conflicts}
                        for raw in critic.payload.get("decisions", []) or []:
                            candidate_id = str(raw.get("candidate_id", ""))
                            label = str(raw.get("label", "")).upper()
                            quote = str(raw.get("quote", ""))
                            selected = conflict_by_id.get(candidate_id)
                            start, end = self._align_quote(source_text, quote)
                            if (not selected or label not in ENTAILMENT_LABELS or start < 0
                                    or not selected.char_start <= start < end <= selected.char_end):
                                continue
                            if label != decisions[candidate_id].label:
                                label = NOT_ENOUGH_INFORMATION
                            decisions[candidate_id] = EntailmentDecision(
                                candidate_id, label, 0.75, "qwen_conflict_critic",
                                quote, start, end,
                                ["dual_model_conflict_review", "quote_realigned"],
                            )
        for item in uncertain:
            decisions.setdefault(item.candidate_id, EntailmentDecision(
                item.candidate_id, NOT_ENOUGH_INFORMATION, item.evidence_confidence,
                "local_abstention", item.text, item.char_start, item.char_end,
                [*item.reason_codes, "remote_unavailable_or_invalid"],
            ))
        return [decisions[item.candidate_id] for item in selections], audit
