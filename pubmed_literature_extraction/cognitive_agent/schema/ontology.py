"""Shared ontology, import policy, and evidence classification for all extractors."""
from __future__ import annotations

from typing import Any

from .relation_signatures import LITERATURE_CANDIDATE_SIGNATURES
from .write_contract import is_main_kg_write_signature


ONTOLOGY_VERSION = "liver-kg-ontology-v1"

# Direct means the edge itself states a disease/prognosis/progression claim.
DIRECT_DISEASE_SIGNATURES = {
    (predicate, signature)
    for predicate, signatures in LITERATURE_CANDIDATE_SIGNATURES.items()
    for signature in signatures
    if "Disease" in signature and predicate in {"ASSOCIATED_WITH", "PROGNOSTIC_IN", "PROGRESSES_TO"}
}

CONTEXT_RELATIONS = {
    "INTERACTS_WITH", "PARTICIPATES_IN", "EXPRESSED_IN", "ASSOCIATED_WITH_METABOLITE",
}


def relation_schema_valid(predicate: str, subject_type: str, object_type: str) -> bool:
    return (
        (subject_type, object_type)
        in LITERATURE_CANDIDATE_SIGNATURES.get(predicate, set())
    )


def relation_write_policy(
    predicate: str,
    subject_type: str = "",
    object_type: str = "",
) -> str:
    if is_main_kg_write_signature(predicate, subject_type, object_type):
        return "write_candidate"
    if relation_schema_valid(predicate, subject_type, object_type):
        return "identify_only"
    return "reject"


def classify_evidence(
    predicate: str,
    subject_type: str,
    object_type: str,
    *,
    source: str = "",
    inferred: bool = False,
    uncertain: bool = False,
) -> tuple[str, str]:
    """Return stable evidence class and a human-readable basis.

    ``source`` is intentionally not used as the sole classifier: the same
    predicate can be direct or contextual depending on its endpoints.
    """
    if inferred or uncertain:
        return "inferred_or_hypothesis", "inference_or_uncertain_claim"
    if (predicate, (subject_type, object_type)) in DIRECT_DISEASE_SIGNATURES:
        if predicate == "PROGNOSTIC_IN":
            return "direct_disease_evidence", "disease_prognosis_edge"
        if predicate == "PROGRESSES_TO":
            return "direct_disease_evidence", "explicit_disease_progression_edge"
        return "direct_disease_evidence", "molecule_or_pathway_disease_edge"
    if predicate in CONTEXT_RELATIONS or "Disease" not in (subject_type, object_type):
        return "contextual_background", f"{source or 'knowledge_graph'}_context"
    return "contextual_background", f"{source or 'knowledge_graph'}_association_context"


def annotate_relation_evidence(rel: dict[str, Any], *, source: str = "PubMed") -> dict[str, Any]:
    evidence_class, basis = classify_evidence(
        rel.get("predicate", ""), rel.get("subject_type", ""), rel.get("object_type", ""),
        source=source, inferred=bool(rel.get("inferred")), uncertain=bool(rel.get("uncertain")),
    )
    rel["evidence_class"] = evidence_class
    rel["evidence_basis"] = basis
    rel["write_policy"] = relation_write_policy(
        rel.get("predicate", ""),
        rel.get("subject_type", ""),
        rel.get("object_type", ""),
    )
    return rel
