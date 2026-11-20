"""Challenge an explicit relation with a source-grounded alternative argument."""

from __future__ import annotations

import re
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
from cognitive_agent.evidence_units import EvidenceUnit
from cognitive_agent.extraction_quality import locate_mention, normalize_surface
from cognitive_agent.schema.schema_profile import PredicateRule, SchemaProfile, _alias_spans

_PAREN_GROUP = re.compile(r"(?P<owner>[^(),;]+?)\s*\((?P<body>[^();]*)\)")
_GROUP_CONNECTOR = re.compile(r"\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)", re.I)
_GROUP_TRAILING = re.compile(r"\s*[.!?]?\s*$")


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


def _parenthetical_group_competitor(
    subject: tuple[str, str], object_: tuple[str, str], *,
    quote: str, quote_start: int, parent: EvidenceUnit, source_index: ArticleEvidenceIndex,
    entity_aliases: dict[tuple[str, str], list[str]], trigger: tuple[int, int],
) -> dict[str, Any]:
    """Resolve a shared trigger followed by contiguous typed owner–list groups."""
    if _POSSIBLY_NEGATED.search(quote[:trigger[0]]):
        return {}
    owners = {
        name: [name, *aliases] for (name, kind), aliases in entity_aliases.items()
        if kind == subject[1]
    }
    object_aliases = [object_[0], *entity_aliases[object_]]
    other_objects = [
        [name, *aliases] for (name, kind), aliases in entity_aliases.items()
        if kind == object_[1] and name != object_[0]
    ]
    seen_owners: set[str] = set()
    witnesses: list[tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    cursor = trigger[1]
    while True:
        group = _PAREN_GROUP.match(quote, cursor)
        if group is None:
            return {}
        raw_owner = group.group("owner")
        owner = raw_owner.strip()
        matching_owners = [
            name for name, aliases in owners.items() if (0, len(owner)) in _alias_spans(owner, aliases)
        ]
        if len(matching_owners) != 1 or matching_owners[0] in seen_owners:
            return {}
        name = matching_owners[0]
        seen_owners.add(name)
        owner_start = group.start("owner") + len(raw_owner) - len(raw_owner.lstrip())

        # Resolve only the queried object. Other list members can be unlabelled.
        # A partial mention could be a nested list or a qualified assertion.
        item_start = group.start("body")
        for raw_item in group.group("body").split(","):
            item = raw_item.strip()
            if not item:
                return {}
            matches = _alias_spans(item, object_aliases)
            if matches:
                if matches != [(0, len(item))] or any(
                    (0, len(item)) in _alias_spans(item, aliases) for aliases in other_objects
                ):
                    return {}
                start = item_start + len(raw_item) - len(raw_item.lstrip())
                witnesses.append((
                    name, (owner_start, owner_start + len(owner)),
                    (start, start + len(item)), group.span("body"),
                ))
            item_start += len(raw_item) + 1

        cursor = group.end()
        if _GROUP_TRAILING.fullmatch(quote[cursor:]):
            break
        connector = _GROUP_CONNECTOR.match(quote, cursor)
        if connector is None:
            return {}
        cursor = connector.end()

    if subject[0] not in seen_owners or len(witnesses) != 1:
        return {}
    rival, owner_span, object_span, group_span = witnesses[0]
    if rival == subject[0]:
        return {}

    def trace(span: tuple[int, int]) -> dict[str, Any]:
        start, end = span
        return {"text": quote[start:end], "char_start": quote_start + start, "char_end": quote_start + end}

    return {
        "status": "COMPETING_ARGUMENT",
        "binding_mode": "PARENTHETICAL_GROUP",
        "source_sha256": source_index.source_sha256,
        "role": "subject",
        "replacement": {"mention": rival, "type": subject[1]},
        "target": {"mention": subject[0], "type": subject[1]},
        "evidence": _trace(
            source_index.text, quote_start, quote_start + len(quote),
            parent.parent_sentence_id, parent.section,
        ),
        "trigger": trace(trigger),
        "arguments": {"subject": trace(owner_span), "object": trace(object_span)},
        "group": trace(group_span),
    }


def find_argument_competitor(
    relation: dict,
    *,
    source_index: ArticleEvidenceIndex,
    profile: SchemaProfile,
    entity_aliases: dict[tuple[str, str], list[str]],
    quote: str,
    compare_groups: bool = True,
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

    if compare_groups and rule.parenthetical_group_binding:
        group_triggers = _trigger_spans(rule.explicit_patterns, quote)
        if len(group_triggers) == 1:
            grouped = _parenthetical_group_competitor(
                subject, object_, quote=quote, quote_start=start, parent=parent,
                source_index=source_index, entity_aliases=entity_aliases, trigger=group_triggers[0],
            )
            if grouped:
                return grouped

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
