#!/usr/bin/env python3
"""Canonical predicate semantics shared by extraction and verification.

The cards describe the literature candidate contract.  They do not grant main
KG write permission; that remains the responsibility of ``write_contract``.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .relation_signatures import LITERATURE_CANDIDATE_SIGNATURES


CARD_MATCHES = frozenset({"EXPLICIT", "TYPE_ONLY", "CONFLICT", "NOT_APPLICABLE"})
SUPPORT_MATCHES = frozenset({"EXPLICIT", "WEAK", "NONE", "CONFLICT"})


@dataclass(frozen=True)
class RelationCard:
    predicate: str
    description: str
    relation_direction: str
    association_sign_applicable: bool
    expression_change_applicable: bool
    activity_change_applicable: bool
    trigger_patterns: tuple[str, ...]
    high_precision_patterns: tuple[str, ...]
    exclusion_patterns: tuple[str, ...] = ()
    confusable_predicates: tuple[str, ...] = ()
    boundary_note: str = ""

    @property
    def allowed_signatures(self) -> frozenset[tuple[str, str]]:
        return frozenset(LITERATURE_CANDIDATE_SIGNATURES.get(self.predicate, set()))

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "definition": self.description,
            "allowed_signatures": [list(item) for item in sorted(self.allowed_signatures)],
            "relation_direction": self.relation_direction,
            "association_sign_applicable": self.association_sign_applicable,
            "expression_change_applicable": self.expression_change_applicable,
            "activity_change_applicable": self.activity_change_applicable,
            "confusable_predicates": list(self.confusable_predicates),
            "boundary_note": self.boundary_note,
        }


RELATION_CARDS: dict[str, RelationCard] = {
    "ASSOCIATED_WITH": RelationCard(
        "ASSOCIATED_WITH",
        "The text explicitly states an association, correlation, risk relation, or linked change between the endpoints.",
        "NON_DIRECTIONAL", True, True, True,
        (
            r"associated with", r"association (?:between|with)", r"correlat(?:ed|ion)",
            r"(?:^|[\s-])associated(?:\s|$)", r"characteri[sz]ed by", r"caused by",
            r"linked to", r"related to", r"contribut(?:es?|ed) to", r"promot(?:es?|ed)",
            r"suppress(?:es|ed)", r"reduc(?:es|ed)", r"increas(?:es|ed)",
            r"decreas(?:es|ed)",
            r"attenuat(?:es|ed)", r"ameliorat(?:es|ed)", r"protect(?:s|ed) against",
        ),
        (
            r"\bassociat\w+ (?:with|between)\b", r"\bcorrelat\w+ (?:with|between)\b",
            r"\blinked to\b", r"\brelated to\b", r"\brisk factor (?:for|of)\b",
            r"\bindependent(?:ly)? associated\b",
        ),
        confusable_predicates=("PROGNOSTIC_IN", "PROGRESSES_TO", "PARTICIPATES_IN"),
        boundary_note="Mere co-occurrence, measurement in a cohort, or an expression location is insufficient.",
    ),
    "ASSOCIATED_WITH_METABOLITE": RelationCard(
        "ASSOCIATED_WITH_METABOLITE",
        "The text explicitly associates a gene or protein endpoint with a metabolite endpoint.",
        "NON_DIRECTIONAL", True, False, False,
        (r"associated with", r"correlat(?:ed|ion)", r"interact(?:s|ed|ion) with"),
        (r"\b(?:metabolic|metabolite) association\b", r"\bassociat\w+ with\b", r"\bcorrelat\w+ with\b"),
        confusable_predicates=("ASSOCIATED_WITH", "INTERACTS_WITH"),
        boundary_note="One endpoint must be a grounded Metabolite; a generic metabolic phenotype is not sufficient.",
    ),
    "INTERACTS_WITH": RelationCard(
        "INTERACTS_WITH",
        "The text explicitly reports molecular binding, physical interaction, cell communication, or supported crosstalk.",
        "NON_DIRECTIONAL", False, False, True,
        (r"interact(?:s|ed|ion) with", r"interactions? between", r"bind(?:s|ing|bound) to", r"cross[- ]?talk", r"cell(?:ular)?[- ]cell communication", r"juxtapos\w*"),
        (r"\binteract\w* with\b", r"\binteractions? between\b", r"\bbind\w* (?:to|with)\b", r"\bcross[- ]?talk\b", r"\bcell(?:ular)?[- ]cell communication\b", r"\bjuxtapos\w*\b"),
        confusable_predicates=("ASSOCIATED_WITH",),
        boundary_note="Co-expression or pathway co-membership without interaction evidence is insufficient.",
    ),
    "ENCODES": RelationCard(
        "ENCODES", "A gene explicitly encodes the protein product.",
        "SUBJECT_TO_OBJECT", False, False, False,
        (r"encod(?:es|ed)",), (r"\bencod\w+",),
        boundary_note="The subject must be Gene and the object Protein; naming similarity alone is insufficient.",
    ),
    "PARTICIPATES_IN": RelationCard(
        "PARTICIPATES_IN",
        "A gene, protein, cell type, or metabolite explicitly participates in, mediates, or regulates a named pathway.",
        "SUBJECT_TO_OBJECT", False, False, True,
        (r"participat(?:es|ed) in", r"involved in", r"mediates?", r"\bvia\b", r"regulat\w+", r"core .{0,80}pathway genes?"),
        (r"\bparticipat\w* in\b", r"\bmediat\w+", r"\bplays? a role in\b", r"\bregulat\w+\b", r"\bcore\b.{0,80}\bpathway genes?\b"),
        (r"\b(?:gene set enrichment|enrichment analysis|gsea)\b",),
        ("ASSOCIATED_WITH",),
        "An enrichment result without an endpoint-specific assertion is insufficient.",
    ),
    "EXPRESSED_IN": RelationCard(
        "EXPRESSED_IN",
        "The text explicitly locates gene or protein expression in a grounded tissue or cell type.",
        "SUBJECT_TO_OBJECT", False, True, False,
        (r"express(?:ed|ion|ion level|ion levels|ion of).{0,100}\b(?:in|by|within)\b", r"expression patterns? of", r"overexpress(?:ed|ion).{0,100}\b(?:in|by|within)\b", r"\bsource of\b", r"localized in", r"present in"),
        (r"\bexpress\w+ (?:in|by|within)\b", r"\bexpression patterns? of\b", r"\b(?:high|low)?\s*expression\s+of\b.{0,100}\b(?:in|within)\b", r"\bsource\s+of\b", r"\bpresent in\b", r"\blocali[sz]\w+ (?:in|to)\b"),
        confusable_predicates=("ASSOCIATED_WITH",),
        boundary_note="A change in expression without a Tissue/CellType location is not EXPRESSED_IN.",
    ),
    "PROGNOSTIC_IN": RelationCard(
        "PROGNOSTIC_IN",
        "A gene or protein marker explicitly predicts prognosis, survival, recurrence, mortality, or clinical outcome in a disease.",
        "SUBJECT_TO_OBJECT", False, True, False,
        (r"prognostic", r"predict(?:s|ed) survival", r"associated with survival", r"recurren\w*", r"mortality"),
        (r"\bprognos\w+", r"\bpredict\w* (?:of|for)? (?:survival|outcome|mortality|recurrence)\b", r"\bassociated with (?:overall )?(?:survival|outcome)\b"),
        (r"\bpotential target for (?:diagnosis|treatment|therapy)\b",),
        ("ASSOCIATED_WITH",),
        "Diagnostic or therapeutic-target language without an outcome relation is insufficient.",
    ),
    "PROGRESSES_TO": RelationCard(
        "PROGRESSES_TO", "One disease, condition, or stage explicitly progresses, develops, or evolves into another.",
        "SUBJECT_TO_OBJECT", False, False, False,
        (r"progress(?:es|ed|ion) to", r"develop(?:s|ed) into", r"evolv(?:es|ed) into", r"progressive form of"),
        (r"\bprogress\w* (?:in)?to\b", r"\bdevelop\w* into\b", r"\bevolv\w* into\b", r"\bprogressive form of\b"),
        confusable_predicates=("ASSOCIATED_WITH",),
        boundary_note="Comorbidity or elevated risk without a transition assertion is insufficient.",
    ),
}


PREDICATE_TRIGGERS = {
    predicate: card.trigger_patterns for predicate, card in RELATION_CARDS.items()
}
HIGH_PRECISION_PREDICATE_TRIGGERS = {
    predicate: card.high_precision_patterns for predicate, card in RELATION_CARDS.items()
}


def relation_card(predicate: str) -> RelationCard | None:
    return RELATION_CARDS.get(str(predicate or "").upper())


def relation_description(predicate: str, subject_type: str = "", object_type: str = "") -> str:
    card = relation_card(predicate)
    if not card:
        return str(predicate or "").upper()
    if card.predicate == "INTERACTS_WITH" and subject_type == object_type == "CellType":
        return (
            "The text explicitly reports cell-cell crosstalk, communication, or a "
            "functionally supported spatial interaction; mere co-occurrence is insufficient."
        )
    return card.description


def _surface_pattern(value: str) -> re.Pattern[str] | None:
    tokens = re.findall(r"\w+", str(value or ""), flags=re.UNICODE)
    if not tokens:
        return None
    return re.compile(
        r"(?<!\w)" + r"(?:[\W_]+)".join(re.escape(token) for token in tokens) + r"(?!\w)",
        re.IGNORECASE,
    )


def _alias_spans(text: str, aliases: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for alias in aliases:
        pattern = _surface_pattern(alias)
        if pattern is not None:
            spans.extend((match.start(), match.end()) for match in pattern.finditer(text))
    return sorted(set(spans))


def _trigger_spans(patterns: tuple[str, ...], text: str) -> list[tuple[int, int]]:
    return [
        (match.start(), match.end())
        for pattern in patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]


def _trigger_links(
    trigger_spans: list[tuple[int, int]],
    subject_spans: list[tuple[int, int]],
    object_spans: list[tuple[int, int]],
    *,
    predicate: str,
) -> bool:
    for subject_span in subject_spans:
        for object_span in object_spans:
            subject_center = sum(subject_span) / 2
            object_center = sum(object_span) / 2
            low, high = sorted((subject_center, object_center))
            if any(end >= low and start <= high for start, end in trigger_spans):
                return True
            if predicate == "EXPRESSED_IN" and object_center > subject_center:
                if any(0 <= subject_span[0] - end <= 60 for _, end in trigger_spans):
                    return True
    return False


def predicate_support_match(
    predicate: str,
    subject_type: str,
    object_type: str,
    evidence: str,
    *,
    subject_aliases: list[str] | None = None,
    object_aliases: list[str] | None = None,
) -> dict[str, Any]:
    """Return one shared predicate-specific support decision.

    ``WEAK`` is source-grounded structural support and may close an
    EvidencePack.  Only ``EXPLICIT`` is sufficient for deterministic semantic
    promotion.  Callers therefore no longer maintain separate trigger tables
    or endpoint-linking interpretations.
    """
    card = relation_card(predicate)
    if card is None:
        return {"match": "NONE", "reason_codes": ["unknown_predicate"]}
    if (str(subject_type), str(object_type)) not in card.allowed_signatures:
        return {"match": "NONE", "reason_codes": ["type_signature_mismatch"]}
    text = str(evidence or "")
    exclusions = [
        pattern for pattern in card.exclusion_patterns
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    if exclusions:
        return {
            "match": "CONFLICT",
            "reason_codes": ["predicate_exclusion_cue"],
            "matched_exclusions": exclusions,
        }

    subject_spans = _alias_spans(text, list(subject_aliases or []))
    object_spans = _alias_spans(text, list(object_aliases or []))
    endpoints_present = bool(subject_spans and object_spans)
    explicit_spans = _trigger_spans(card.high_precision_patterns, text)
    weak_spans = _trigger_spans(card.trigger_patterns, text)
    explicit_links = endpoints_present and _trigger_links(
        explicit_spans, subject_spans, object_spans, predicate=card.predicate,
    )
    weak_links = endpoints_present and _trigger_links(
        weak_spans, subject_spans, object_spans, predicate=card.predicate,
    )

    # Coordinated expression is common in biomedical abstracts: the second
    # gene/protein can be separated from the expression noun by a conjunction.
    if card.predicate == "EXPRESSED_IN" and endpoints_present:
        expression_location = re.search(
            r"\b(?:high|low|increased|decreased)?\s*(?:expression|overexpression)\b"
            r".{0,180}\b(?:in|within|by)\b",
            text,
            flags=re.IGNORECASE,
        )
        explicit_links = bool(explicit_links or expression_location)

    # A differential expression statement tied to a disease cohort is an
    # explicit gene/protein--disease association, not bare co-occurrence.
    if card.predicate == "ASSOCIATED_WITH" and endpoints_present:
        cohort_change = re.search(
            r"\b(?:expression|levels?)\b.{0,180}"
            r"\b(?:increas|decreas|elevat|reduc|higher|lower)\w*\b.{0,160}"
            r"\b(?:patients?|participants?|subjects?|tissues?|samples?)\b",
            text,
            flags=re.IGNORECASE,
        )
        expression_change_in_endpoint = re.search(
            r"\b(?:expression|levels?)\b.{0,120}"
            r"\b(?:increas|decreas|elevat|reduc|higher|lower)\w*\b.{0,80}\bin\b",
            text,
            flags=re.IGNORECASE,
        )
        explicit_links = bool(explicit_links or cohort_change or expression_change_in_endpoint)

    if explicit_links:
        return {"match": "EXPLICIT", "reason_codes": ["explicit_predicate_support"]}
    if weak_links:
        return {"match": "WEAK", "reason_codes": ["weak_predicate_support"]}
    # External closure is only meaningful when one endpoint is absent from
    # this span (typically a resolved pronoun or neighbouring owner span).
    # When both endpoints are present but the trigger sits outside their
    # local span, returning WEAK would incorrectly bless unrelated clauses.
    if (
        (explicit_spans or weak_spans)
        and (subject_spans or object_spans)
        and not endpoints_present
    ):
        return {
            "match": "WEAK",
            "reason_codes": ["trigger_present_endpoint_closure_external"],
        }
    reasons = ["predicate_support_absent"]
    if (explicit_spans or weak_spans) and endpoints_present:
        reasons.append("trigger_not_linking_endpoints")
    if not endpoints_present:
        reasons.append("support_endpoints_not_closed")
    return {"match": "NONE", "reason_codes": reasons}


def match_relation_card(
    predicate: str,
    subject_type: str,
    object_type: str,
    evidence: str,
    *,
    endpoints_linked: bool = True,
) -> dict[str, Any]:
    """Return a deterministic card match plus auditable reason codes."""
    card = relation_card(predicate)
    if card is None:
        return {"match": "NOT_APPLICABLE", "reason_codes": ["unknown_predicate"]}
    if (str(subject_type), str(object_type)) not in card.allowed_signatures:
        return {"match": "NOT_APPLICABLE", "reason_codes": ["type_signature_mismatch"]}
    text = str(evidence or "")
    exclusions = [pattern for pattern in card.exclusion_patterns if re.search(
        pattern, text, flags=re.IGNORECASE
    )]
    if exclusions:
        return {"match": "CONFLICT", "reason_codes": ["predicate_exclusion_cue"]}
    explicit = any(re.search(
        pattern, text, flags=re.IGNORECASE
    ) for pattern in card.high_precision_patterns)
    if explicit and endpoints_linked:
        return {"match": "EXPLICIT", "reason_codes": ["explicit_predicate_support"]}
    reasons = ["type_signature_only"]
    if explicit and not endpoints_linked:
        reasons.append("trigger_not_linking_endpoints")
    return {"match": "TYPE_ONLY", "reason_codes": reasons}


def load_predicate_thresholds(path: str | Path | None) -> dict[str, float]:
    """Load a read-only calibrated threshold artifact."""
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(payload.get("artifact_type", "")) != "predicate_adjudication_thresholds":
        raise ValueError("invalid predicate threshold artifact_type")
    thresholds = payload.get("thresholds", {}) or {}
    return {
        str(predicate).upper(): max(0.0, min(1.0, float(value)))
        for predicate, value in thresholds.items()
        if str(predicate).upper() in RELATION_CARDS
    }
