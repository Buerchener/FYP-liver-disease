"""Compare source-aligned support, explicit opposition, and inconclusive findings."""

from __future__ import annotations

import re
from typing import Any

from cognitive_agent.evidence_pack import ArticleEvidenceIndex
from cognitive_agent.schema.schema_profile import SchemaProfile, _alias_spans
from cognitive_agent.evidence_scope import (
    compare_scope as compare_scope_status,
    extract_scope_witnesses,
    normalize_scope,
)
from cognitive_agent.evidence_statistics import classify_opposition


_CLAUSE_BREAK = re.compile(
    r"\s*(?:;\s*|,\s*(?=(?:but|however|whereas|while|and)\b)|"
    r"\b(?:but|however|whereas|while)\b[\s,]*)",
    re.IGNORECASE,
)
_PARENTHETICAL = re.compile(r"\([^()]*\)")
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
    parentheticals = [match.span() for match in _PARENTHETICAL.finditer(parent_text)]
    breaks = [
        match.span() for match in _CLAUSE_BREAK.finditer(parent_text)
        if not any(left < match.start() < right for left, right in parentheticals)
    ]
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
    return normalized in {
        "ABSTRACT", "RESULT", "RESULTS", "FINDING", "FINDINGS",
        "CONCLUSION", "CONCLUSIONS",
    }


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


def _direct_argument_spans(
    left: str, right: str, subject_aliases: list[str], object_aliases: list[str],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    # An unrelated object or conjunction between the trigger and an endpoint
    # cannot be repaired into a denial just because the endpoint occurs later.
    subject = max(
        (
            span for span in _alias_spans(left, subject_aliases)
            if len(_WORD.findall(left[span[1]:])) <= 6
            and not re.search(r"[,;:]|\b(?:and|or)\b", left[span[1]:], flags=re.I)
        ),
        key=lambda span: (span[1], -span[0]), default=None,
    )
    object_ = next((
        span for span in _alias_spans(right, object_aliases)
        if _OBJECT_PREFIX.fullmatch(right[:span[0]])
    ), None)
    return (subject, object_) if subject is not None and object_ is not None else None


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
    compare_scope: bool = True,
    interpret_statistics: bool = True,
) -> dict[str, Any]:
    """Pair explicit support with a denial or non-confirmation of the same typed edge.

    The opposing witness must use a scoped negator with a configured trigger
    and close both typed arguments after masking the negator. A confidence
    interval is interpreted only when attached to the bound right argument.
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
    opposing: dict[str, list[dict[str, Any]]] = {"COUNTEREVIDENCE": [], "INCONCLUSIVE": []}
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
                arguments = _direct_argument_spans(left, right, subject_aliases, object_aliases)
                if arguments is None and (rule.symmetric or rule.relation_direction == "NON_DIRECTIONAL"):
                    arguments = _direct_argument_spans(left, right, object_aliases, subject_aliases)
                if arguments is None:
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
                    opposition = {
                        **trace,
                        "negation_start": start + mask_start,
                        "negation_end": start + mask_end,
                        "trigger_start": start + trigger_start,
                        "trigger_end": start + trigger_end,
                    }
                    polarity = "COUNTEREVIDENCE"
                    if interpret_statistics:
                        assessment = classify_opposition(
                            clause, char_offset=start,
                            negation=(mask_start, mask_end),
                            trigger=(trigger_start, trigger_end),
                            right_argument_end=trigger_end + arguments[1][1],
                        )
                        polarity = assessment["polarity"]
                        opposition["statistical_evidence"] = assessment
                    opposing[polarity].append(opposition)
                    break

    if not supports or not any(opposing.values()):
        return {}
    # Keep a direct denial as the primary witness when both kinds occur.
    # Within its class, prefer scope agreement and proximity as before.
    polarity = next(kind for kind, witnesses in opposing.items() if witnesses)
    counters = opposing[polarity]
    if compare_scope:
        for trace in (*supports, *opposing["COUNTEREVIDENCE"], *opposing["INCONCLUSIVE"]):
            trace["scope_witnesses"] = extract_scope_witnesses(
                trace["text"], char_offset=trace["char_start"],
            )
    status_priority = {
        "SHARED_SCOPE": 0,
        "UNRESOLVED_SCOPE": 1,
        "DIFFERENT_SCOPE": 2,
    }
    counts = dict.fromkeys(status_priority, 0)
    best_pair = None
    best_rank = None
    for support in supports:
        for counter in counters:
            status = compare_scope_status(
                support["scope_witnesses"], counter["scope_witnesses"],
            ) if compare_scope else "UNRESOLVED_SCOPE"
            counts[status] += 1
            rank = (
                status_priority[status],
                support["sentence_id"] != counter["sentence_id"],
                abs(support["char_start"] - counter["char_start"]),
                support["char_start"], counter["char_start"],
            )
            if best_rank is None or rank < best_rank:
                best_pair = support, counter, status
                best_rank = rank
    if best_pair is None:
        return {}
    support, counter, status = best_pair
    status_key, witness_key = {
        "COUNTEREVIDENCE": ("CONTESTED", "counterevidence"),
        "INCONCLUSIVE": ("INCONCLUSIVE", "inconclusive"),
    }[polarity]
    result = {"status": status_key, "support": support, witness_key: counter}
    if not compare_scope:
        return result

    result["source_sha256"] = source_index.source_sha256
    result["scope_status"] = status
    # Pair counts describe the displayed polarity. All witnesses remain below.
    result["scope_pair_counts"] = counts

    groups: dict[tuple, dict[str, Any]] = {}
    for polarity, assertions in (("SUPPORT", supports), *opposing.items()):
        for assertion in assertions:
            scope = normalize_scope(assertion.get("scope_witnesses", []))
            signature = tuple((dimension, tuple(values)) for dimension, values in scope.items())
            comparable = bool(signature and all(len(values) == 1 for _, values in signature))
            # Missing or multivalued conditions do not identify a shared stratum.
            key = (signature, polarity, -1 if comparable else assertion["char_start"])
            group = groups.setdefault(key, {
                "scope": scope,
                "polarity": polarity,
                "explicit_single_scope": comparable,
                "assertions": [],
            })
            group["assertions"].append({**assertion, "role": polarity.casefold()})
    result["scoped_assertions"] = [
        {**group, "assertions": sorted(
            group["assertions"], key=lambda item: (item["char_start"], item["char_end"]),
        )}
        for _, group in sorted(groups.items())
    ]
    return result
