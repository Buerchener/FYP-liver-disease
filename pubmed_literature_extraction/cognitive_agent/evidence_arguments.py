"""Challenge an explicit relation with a source-grounded alternative argument."""

from __future__ import annotations

from typing import Any

from cognitive_agent.evidence_contrast import (
    _POSSIBLY_NEGATED,
    _clauses,
    _direct_argument_spans,
    _eligible_section,
    _trace,
    _trigger_spans,
)
from cognitive_agent.evidence_pack import ArticleEvidenceIndex
from cognitive_agent.extraction_quality import locate_mention, normalize_surface
from cognitive_agent.schema.schema_profile import PredicateRule, SchemaProfile


def _has_mention(text: str, aliases: list[str]) -> bool:
    return any(locate_mention(alias, text)[0] for alias in aliases)


def _bound_arguments(
    clause: str, trigger: tuple[int, int], subjects: list[str], objects: list[str],
    rule: PredicateRule,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    start, end = trigger
    left, right = clause[:start], clause[end:]
    pair = _direct_argument_spans(left, right, subjects, objects)
    if pair is not None:
        subject, object_ = pair
        return subject, (end + object_[0], end + object_[1])
    if rule.symmetric or rule.relation_direction == "NON_DIRECTIONAL":
        reverse = _direct_argument_spans(left, right, objects, subjects)
        if reverse is not None:
            object_, subject = reverse
            return (end + subject[0], end + subject[1]), object_
    return None


def find_argument_competitor(
    relation: dict,
    *,
    source_index: ArticleEvidenceIndex,
    profile: SchemaProfile,
    entity_aliases: dict[tuple[str, str], list[str]],
    quote: str,
) -> dict[str, Any]:
    """Return an explicit rival when a broad quote only supports another pair locally.

    The quote and predicate stay fixed. Only one typed endpoint can change, to
    a different grounded entity already present in that quote. The rival needs
    a clause-local trigger attached to both arguments, without a direct witness
    for the original pair anywhere in the quote.
    """
    quote = str(quote or "").strip()
    if not quote:
        return {}
    text = source_index.text
    start = text.find(quote)
    if start < 0 or text.find(quote, start + 1) >= 0:
        return {}
    parent = source_index.containing_parent(start, start + len(quote))
    if parent is None or not _eligible_section(parent.section):
        return {}

    predicate = str(relation.get("predicate") or "")
    subject = str(relation.get("subject") or ""), str(relation.get("subject_type") or "")
    object_ = str(relation.get("object") or ""), str(relation.get("object_type") or "")
    rule = profile.rule(predicate)
    if rule is None or not rule.explicit_patterns or not rule.valid(subject[1], object_[1]):
        return {}
    if subject not in entity_aliases or object_ not in entity_aliases:
        return {}
    subject_aliases = [subject[0], *entity_aliases[subject]]
    object_aliases = [object_[0], *entity_aliases[object_]]

    def explicit(fragment: str, left: list[str], right: list[str]) -> bool:
        return profile.support_match(
            predicate, subject[1], object_[1], fragment,
            subject_aliases=left, object_aliases=right,
        ).get("match") == "EXPLICIT"

    clauses = list(_clauses(quote, start))
    if len(clauses) < 2 or not explicit(quote, subject_aliases, object_aliases):
        return {}
    if any(explicit(clause, subject_aliases, object_aliases) for clause, _, _ in clauses):
        return {}
    if not (_has_mention(quote, subject_aliases) and _has_mention(quote, object_aliases)):
        return {}

    # Iterate in source order, and make ties independent of raw entity order.
    for clause, clause_start, clause_end in clauses:
        if _POSSIBLY_NEGATED.search(clause):
            continue
        triggers = _trigger_spans(rule.explicit_patterns, clause)
        if not triggers:
            continue
        for role, target, target_aliases, held_aliases in (
            ("subject", subject, subject_aliases, object_aliases),
            ("object", object_, object_aliases, subject_aliases),
        ):
            if _has_mention(clause, target_aliases) or not _has_mention(clause, held_aliases):
                continue
            target_surfaces = {normalize_surface(alias) for alias in target_aliases}
            for (mention, entity_type), aliases in sorted(entity_aliases.items()):
                if entity_type != target[1]:
                    continue
                rival_aliases = [mention, *aliases]
                if target_surfaces.intersection(map(normalize_surface, rival_aliases)):
                    continue
                if not _has_mention(clause, rival_aliases):
                    continue
                left, right = (
                    (rival_aliases, held_aliases) if role == "subject"
                    else (held_aliases, rival_aliases)
                )
                anchored = next((
                    (item, spans)
                    for item in triggers
                    if (spans := _bound_arguments(clause, item, left, right, rule)) is not None
                ), None)
                if anchored is not None and explicit(clause, left, right):
                    (trigger_start, trigger_end), arguments = anchored
                    return {
                        "status": "COMPETING_ARGUMENT",
                        "source_sha256": source_index.source_sha256,
                        "role": role,
                        "replacement": {"mention": mention, "type": entity_type},
                        "target": {"mention": target[0], "type": target[1]},
                        "evidence": _trace(
                            text, clause_start, clause_end, parent.parent_sentence_id, parent.section,
                        ),
                        "trigger": {
                            "text": clause[trigger_start:trigger_end],
                            "char_start": clause_start + trigger_start,
                            "char_end": clause_start + trigger_end,
                        },
                        "arguments": {
                            key: {
                                "text": clause[arg_start:arg_end],
                                "char_start": clause_start + arg_start,
                                "char_end": clause_start + arg_end,
                            }
                            for key, (arg_start, arg_end) in zip(("subject", "object"), arguments)
                        },
                    }
    return {}
