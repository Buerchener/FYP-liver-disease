#!/usr/bin/env python3
"""Deterministic article-local demonstration selection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from cognitive_agent.schema.examples import (
    EXAMPLE_BIOINFORMATICS,
    EXAMPLE_EXPRESSION,
    EXAMPLE_GENE_DISEASE,
    EXAMPLE_GOLD_CLINICAL_MULTI,
    EXAMPLE_GOLD_OTUD5,
    EXAMPLE_METABOLIC_PATHWAY,
    EXAMPLE_PROTEIN_INTERACTION,
    EXAMPLE_REVIEW_NO_RELS,
)


GOLDEN_EXAMPLE_VERSION = "dynamic-golden-shot-v2"


@dataclass(frozen=True)
class GoldenExample:
    name: str
    example: object
    study_types: frozenset[str]
    cues: tuple[str, ...]
    negative_relation_example: bool = False
    source_pmids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class GoldenSelection:
    examples: list
    names: list[str]
    reasons: list[str]


POSITIVE_EXAMPLES = (
    GoldenExample(
        "gold_otud5_interaction_expression", EXAMPLE_GOLD_OTUD5,
        frozenset({"mechanistic", "human_omics", "clinical"}),
        (r"interact\w*", r"express\w*", r"macrophage|cell subset"),
        source_pmids=frozenset({"41650163"}),
    ),
    GoldenExample(
        "gold_clinical_multi_association", EXAMPLE_GOLD_CLINICAL_MULTI,
        frozenset({"clinical", "other", "review"}),
        (r"associated with", r"risk of", r"cohort|patients?"),
        source_pmids=frozenset({"41482383"}),
    ),
    GoldenExample(
        "expression_localization", EXAMPLE_EXPRESSION,
        frozenset({"mechanistic", "human_omics", "clinical", "in_vitro"}),
        (r"express\w*", r"hepatocytes?|tissue|cell"),
    ),
    GoldenExample(
        "metabolic_pathway", EXAMPLE_METABOLIC_PATHWAY,
        frozenset({"animal", "mechanistic", "human_omics"}),
        (r"pathway|ferroptosis|apoptosis", r"mice|mouse|murine"),
    ),
    GoldenExample(
        "protein_interaction", EXAMPLE_PROTEIN_INTERACTION,
        frozenset({"mechanistic", "in_vitro", "animal"}),
        (r"interact\w*|bind\w*", r"protein|signaling"),
    ),
    GoldenExample(
        "gene_disease_and_encodes", EXAMPLE_GENE_DISEASE,
        frozenset({"clinical", "mechanistic", "other"}),
        (r"associated with|progress\w*", r"encod\w*|gene"),
    ),
)

NEGATIVE_EXAMPLES = (
    GoldenExample(
        "negative_computational_no_relation", EXAMPLE_BIOINFORMATICS,
        frozenset({"computational", "human_omics", "mechanistic"}),
        (r"network pharmacology|molecular docking|bioinformatics|screening",),
        True,
    ),
    GoldenExample(
        "negative_review_no_asserted_relation", EXAMPLE_REVIEW_NO_RELS,
        frozenset({"review", "other", "clinical"}),
        (r"review|summari[sz]\w*|background",),
        True,
    ),
)


class GoldenExampleSelector:
    """Pick relevant demonstrations while respecting the provider prompt budget."""

    def select(
        self,
        text: str,
        study_type: str,
        *,
        max_examples: int = 4,
        document_id: str = "",
    ) -> GoldenSelection:
        max_examples = max(1, min(4, int(max_examples)))
        low = str(text or "").casefold()

        def score(item: GoldenExample) -> tuple[float, list[str]]:
            value = 0.0
            reasons: list[str] = []
            if study_type in item.study_types:
                value += 2.0
                reasons.append(f"study_type_{study_type}")
            cue_hits = sum(bool(re.search(pattern, low, re.IGNORECASE)) for pattern in item.cues)
            if cue_hits:
                value += 1.5 * cue_hits
                reasons.append(f"cue_hits_{cue_hits}")
            return value, reasons

        ranked = []
        eligible_positives = [
            item for item in POSITIVE_EXAMPLES
            if str(document_id) not in item.source_pmids
        ]
        for index, item in enumerate(eligible_positives):
            value, reasons = score(item)
            ranked.append((-value, index, item, reasons or ["coverage_diversity"]))
        ranked.sort(key=lambda row: (row[0], row[1]))
        negative_ranked = []
        for index, item in enumerate(NEGATIVE_EXAMPLES):
            value, reasons = score(item)
            negative_ranked.append((-value, index, item, reasons or ["boundary_control"]))
        negative_ranked.sort(key=lambda row: (row[0], row[1]))
        if max_examples == 1:
            best_positive = ranked[0]
            best_negative = negative_ranked[0]
            prefer_boundary = study_type in {"computational", "review"}
            chosen = [
                best_negative
                if best_negative[0] < best_positive[0]
                or (best_negative[0] == best_positive[0] and prefer_boundary)
                else best_positive
            ]
        else:
            negative_count = (
                min(2, max_examples - 1)
                if study_type in {"computational", "review"}
                else 1
            )
            positives = ranked[: max_examples - negative_count]
            chosen = [*positives, *negative_ranked[:negative_count]]
        return GoldenSelection(
            examples=[row[2].example for row in chosen],
            names=[row[2].name for row in chosen],
            reasons=[f"{row[2].name}:{','.join(row[3])}" for row in chosen],
        )
