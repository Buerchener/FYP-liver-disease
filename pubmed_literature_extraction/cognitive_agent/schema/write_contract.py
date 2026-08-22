"""Frozen write contract for the liver-kg-core-v02 Neo4j graph.

The literature extractor may understand more relations than the current main
graph can represent.  This module is the sole authority for automatic Neo4j
writes; widening candidate extraction must never widen this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .entity_classes import ENTITY_CLASSES


MAIN_KG_WRITE_CONTRACT_VERSION = "liver-kg-core-v02-contract-v1"

# (predicate, subject Neo4j label, object Neo4j label) -> relation identifier
# property used by the existing main graph.  This is deliberately closed.
MAIN_KG_WRITE_CONTRACT: dict[tuple[str, str, str], str] = {
    ("ASSOCIATED_WITH", "Gene", "Disease"): "relation_id",
    ("ASSOCIATED_WITH", "Metabolite", "Disease"): "relationship_id",
    ("PROGNOSTIC_IN", "Gene", "Disease"): "relationship_id",
    ("PROGRESSES_TO", "Disease", "Disease"): "progression_id",
    ("ENCODES", "Gene", "Protein"): "relationship_id",
    ("INTERACTS_WITH", "Protein", "Protein"): "interaction_id",
    ("PARTICIPATES_IN", "Gene", "Pathway"): "relationship_id",
    ("EXPRESSED_IN", "Gene", "Tissue"): "relationship_id",
    ("EXPRESSED_IN", "Gene", "CellType"): "relationship_id",
    ("ASSOCIATED_WITH_METABOLITE", "Gene", "Metabolite"): "relationship_id",
}

NEO4J_IMPORTABLE_PREDICATES = frozenset(
    predicate for predicate, _, _ in MAIN_KG_WRITE_CONTRACT
)

# Retained for older import paths where every predicate has one stable ID
# property.  New write paths use write_contract_assessment() per typed edge.
RELATION_ID_PROPERTY: dict[str, str] = {}
for (predicate, _, _), relation_id_property in MAIN_KG_WRITE_CONTRACT.items():
    RELATION_ID_PROPERTY.setdefault(predicate, relation_id_property)


@dataclass(frozen=True)
class WriteContractAssessment:
    valid: bool
    reasons: tuple[str, ...]
    version: str = MAIN_KG_WRITE_CONTRACT_VERSION
    relation_id_property: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "write_contract_valid": self.valid,
            "schema_gap_reasons": list(self.reasons),
            "write_contract_version": self.version,
            "relation_id_property": self.relation_id_property,
        }


def _value(relation: Any, key: str, default: str = "") -> str:
    if isinstance(relation, Mapping):
        return str(relation.get(key, default) or "").strip()
    return str(getattr(relation, key, default) or "").strip()


def write_contract_assessment(
    relation: Any,
    *,
    standardized_entities: Mapping[tuple[str, str], Any] | None = None,
) -> WriteContractAssessment:
    """Assess a final relation against the frozen main-KG contract.

    Endpoint grounding remains a factual verifier responsibility.  The adapter
    only checks that a grounded, standardised endpoint can be represented by
    the main graph and that the typed edge has a stable relation ID property.
    """
    predicate = _value(relation, "predicate").upper()
    subject = _value(relation, "subject")
    obj = _value(relation, "object")
    subject_type = _value(relation, "subject_type")
    object_type = _value(relation, "object_type")
    reasons: list[str] = []

    if not subject:
        reasons.append("subject_endpoint_unwritable")
    if not obj:
        reasons.append("object_endpoint_unwritable")
    if subject_type not in {spec["neo4j_label"] for spec in ENTITY_CLASSES.values()}:
        reasons.append("unsupported_subject_endpoint_type")
    if object_type not in {spec["neo4j_label"] for spec in ENTITY_CLASSES.values()}:
        reasons.append("unsupported_object_endpoint_type")
    if standardized_entities is not None:
        if (subject, subject_type) not in standardized_entities:
            reasons.append("subject_endpoint_not_standardized")
        if (obj, object_type) not in standardized_entities:
            reasons.append("object_endpoint_not_standardized")

    signature = (predicate, subject_type, object_type)
    relation_id_property = MAIN_KG_WRITE_CONTRACT.get(signature, "")
    if not relation_id_property:
        if predicate not in NEO4J_IMPORTABLE_PREDICATES:
            reasons.append("unsupported_main_kg_predicate")
        else:
            reasons.append("unsupported_main_kg_signature")
    if not relation_id_property:
        reasons.append("missing_main_kg_relation_id_property")

    return WriteContractAssessment(
        valid=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        relation_id_property=relation_id_property,
    )


class SchemaAdapter:
    """Apply the main-KG write contract after semantic verification."""

    def assess(
        self,
        relation: Any,
        *,
        standardized_entities: Mapping[tuple[str, str], Any] | None = None,
    ) -> WriteContractAssessment:
        return write_contract_assessment(
            relation,
            standardized_entities=standardized_entities,
        )

    def apply(
        self,
        relation: Any,
        *,
        standardized_entities: Mapping[tuple[str, str], Any] | None = None,
    ) -> WriteContractAssessment:
        assessment = self.assess(
            relation,
            standardized_entities=standardized_entities,
        )
        relation.write_contract_valid = assessment.valid
        relation.schema_gap_reasons = list(assessment.reasons)
        relation.write_contract_version = assessment.version

        if assessment.valid:
            return assessment

        # A schema gap is not a claim about the truth of the article.  Keep
        # factual/semantic hard rejections intact; all other candidates remain
        # available in the review ledger and never reach Neo4j.
        relation.import_ready = False
        if (
            str(getattr(relation, "factual_status", "")).upper() == "REJECTED"
            or str(getattr(relation, "semantic_status", "")).upper() == "REJECTED"
        ):
            relation.write_status = "BLOCKED"
        elif (
            str(getattr(relation, "factual_status", "")).upper() == "REVIEW"
            or str(getattr(relation, "semantic_status", "")).upper() == "REVIEW"
        ):
            relation.write_status = "HUMAN_REVIEW"
        else:
            relation.write_status = "SEMANTIC_ONLY"
        return assessment


def is_main_kg_write_signature(
    predicate: str,
    subject_type: str,
    object_type: str,
) -> bool:
    return (str(predicate or "").upper(), subject_type, object_type) in MAIN_KG_WRITE_CONTRACT
