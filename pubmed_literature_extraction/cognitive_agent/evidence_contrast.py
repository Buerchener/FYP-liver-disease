"""Conservative intra-article support versus scoped-refutation detection.

Only explicit predicate rules can establish either side.  Background, method,
objective, and discussion sections are excluded.  Candidate pairs are ranked
first by shared sentence, then by source distance, then by source order.
"""

from __future__ import annotations

import re
from typing import Any

from cognitive_agent.evidence_pack import ArticleEvidenceIndex
from cognitive_agent.schema.schema_profile import SchemaProfile, _alias_spans


_CLAUSE_BREAK = re.compile(
    r"\s*(?:;\s*|,\s*(?=(?:but|however|whereas|while|and)\b)|"
    r"\b(?:but|however|whereas|while)\b[\s,]*)",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(?:not|never|no\s+longer|failed\s+to|fails\s+to)\b", re.I)
_POSSIBLY_NEGATED = re.compile(r"\b(?:not(?!\s+only\b)|never|no|failed\s+to|fails\s+to)\b", re.I)
_NEGATION_GAP = re.compile(
    r"\s*(?:(?:statistically|significantly|independently|consistently|"
    r"directly|detectably|strongly|clearly)\s+){0,3}", re.I,
)
_OBJECT_PREFIX = re.compile(
    r"\s*(?:(?:the|a|an|either|both|of|with|within|to|by|in)\s+){0,3}", re.I,
)
_WORD = re.compile(r"\w+", re.UNICODE)


def _clauses(parent_text: str, start: int):
    cursor = 0
    breaks = [match.span() for match in _CLAUSE_BREAK.finditer(parent_text)]
    breaks.append((len(parent_text), len(parent_text)))
    for end, next_start in breaks:
        fragment = parent_text[cursor:end]
        clause = fragment.strip()
        if clause:
            left = cursor + len(fragment) - len(fragment.lstrip())
            yield clause, start + left, start + left + len(clause)
        cursor = next_start


def _eligible_section(section: str) -> bool:
    normalized = re.sub(r"[^A-Z]+", "_", str(section or "").upper()).strip("_")
    return normalized in {"ABSTRACT", "RESULT", "RESULTS", "CONCLUSION", "CONCLUSIONS"}


def _trigger_spans(patterns: tuple[str, ...], text: str) -> list[tuple[int, int]]:
    spans = []
    for pattern in patterns:
        spans.extend(
            (match.start(), match.end())
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
        )
    return sorted(set(spans))


def _scoped_negation(clause: str, trigger_start: int) -> tuple[int, int] | None:
    """Find a negator that directly scopes a configured trigger (at most 3 words)."""
    matches = _NEGATION.finditer(clause, 0, trigger_start)
    return next((
        match.span() for match in matches
        if _NEGATION_GAP.fullmatch(clause[match.end():trigger_start])
    ), None)


def _direct_arguments(
    left: str, right: str, subject_aliases: list[str], object_aliases: list[str],
) -> bool:
    # An unrelated object or conjunction between the trigger and an endpoint
    # cannot be repaired into a denial just because the endpoint occurs later.
    return any(
        len(_WORD.findall(left[end:])) <= 6
        and not re.search(r"[,;:]|\b(?:and|or)\b", left[end:], flags=re.I)
        for _, end in _alias_spans(left, subject_aliases)
    ) and any(
        _OBJECT_PREFIX.fullmatch(right[:start])
        for start, _ in _alias_spans(right, object_aliases)
    )


def _trace(text: str, start: int, end: int, sentence_id: str, section: str) -> dict[str, Any]:
    return {
        "text": text[start:end],
        "char_start": start,
        "char_end": end,
        "sentence_id": sentence_id,
        "section": section,
    }


def find_evidence_conflict(
    relation: dict,
    *,
    source_index: ArticleEvidenceIndex,
    profile: SchemaProfile,
    subject_aliases: list[str],
    object_aliases: list[str],
) -> dict[str, Any]:
    """Return paired explicit support and scoped negation for the same typed edge.

    Generic exclusion cues are deliberately ignored as negation evidence.  A
    cue counts only when it immediately modifies one of the predicate's
    configured explicit trigger patterns and removing that cue leaves explicit
    support for both supplied endpoints in the same clause.
    """
    predicate = str(relation.get("predicate", "") or "")
    subject_type = str(relation.get("subject_type", "") or "")
    object_type = str(relation.get("object_type", "") or "")
    rule = profile.rule(predicate)
    if (
        rule is None
        or not rule.explicit_patterns
        or not profile.valid(predicate, subject_type, object_type)
        or not _NEGATION.search(source_index.text)
    ):
        return {}

    supports: list[dict[str, Any]] = []
    counters: list[dict[str, Any]] = []
    for parent in source_index.parents:
        if not _eligible_section(parent.section):
            continue
        for clause, start, end in _clauses(parent.text, parent.char_start):
            spans = _trigger_spans(rule.explicit_patterns, clause)
            if not spans:
                continue
            trace = _trace(source_index.text, start, end, parent.parent_sentence_id, parent.section)
            scoped = [
                (trigger_start, trigger_end, cue)
                for trigger_start, trigger_end in spans
                if (cue := _scoped_negation(clause, trigger_start)) is not None
            ]
            if not scoped:
                if _POSSIBLY_NEGATED.search(clause):
                    continue
                match = profile.support_match(
                    predicate, subject_type, object_type, clause,
                    subject_aliases=subject_aliases,
                    object_aliases=object_aliases,
                )
                if match.get("match") == "EXPLICIT":
                    supports.append(trace)
                continue

            for trigger_start, trigger_end, (mask_start, mask_end) in scoped:
                left, right = clause[:mask_start], clause[trigger_end:]
                attaches = _direct_arguments(left, right, subject_aliases, object_aliases)
                if not attaches and (rule.symmetric or rule.relation_direction == "NON_DIRECTIONAL"):
                    attaches = _direct_arguments(left, right, object_aliases, subject_aliases)
                if not attaches:
                    continue
                # The profile must confirm the same typed relation after
                # masking precisely the source-aligned negation cue.
                unnegated = clause[:mask_start] + (" " * (mask_end - mask_start)) + clause[mask_end:]
                unnegated_match = profile.support_match(
                    predicate, subject_type, object_type, unnegated,
                    subject_aliases=subject_aliases,
                    object_aliases=object_aliases,
                )
                if unnegated_match.get("match") == "EXPLICIT":
                    counters.append({
                        **trace,
                        "negation_start": start + mask_start,
                        "negation_end": start + mask_end,
                        "trigger_start": start + trigger_start,
                        "trigger_end": start + trigger_end,
                    })
                    break

    pairs = (
        (support, counter)
        for support in supports
        for counter in counters
        if support["char_start"] != counter["char_start"]
        or support["char_end"] != counter["char_end"]
    )
    closest = min(
        pairs,
        default=None,
        key=lambda pair: (
            0 if pair[0]["sentence_id"] == pair[1]["sentence_id"] else 1,
            abs(pair[0]["char_start"] - pair[1]["char_start"]),
            pair[0]["char_start"],
            pair[1]["char_start"],
        ),
    )
    if closest is None:
        return {}
    support, counter = closest
    return {"status": "CONTESTED", "support": support, "counterevidence": counter}
