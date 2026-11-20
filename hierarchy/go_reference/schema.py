"""Input contract and evidence-grounded output assembly for GO references."""

import json
import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

Short = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
Definition = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=6000)]
GoID = Annotated[str, StringConstraints(pattern=r"^GO:[0-9]{7}$")]
Status = Literal["MATCHED", "REFERENCE_ONLY", "NO_MATCH", "UNCERTAIN", "LOOKUP_FAILED"]
Predicate = Literal["is_a", "part_of", "has_part", "regulates", "positively_regulates", "negatively_regulates"]
PREDICATES = {"is_a", "part_of", "has_part", "regulates", "positively_regulates", "negatively_regulates"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CandidateParent(StrictModel):
    concept_id: Short
    name: Short
    definition: Definition | None = None
    go_id: GoID | None = None


class Concept(StrictModel):
    concept_id: Short
    name: Short
    definition: Definition
    candidate_parents: list[CandidateParent | Short] | None = Field(default=None, max_length=50)
    taxonomy_context: JsonValue = None

    @model_validator(mode="after")
    def limit_context(self):
        if len(json.dumps(self.model_dump(), ensure_ascii=False, allow_nan=False)) > 24000:
            raise ValueError("Input exceeds 24,000 characters")
        return self


# Only these judgments are supplied by the model. All ontology facts are hydrated
# from the successful OLS transcript, including labels, definitions and directions.
REASONS = {
    "MATCHED": {"same_scope": "The concept denotes the same biological process, function or component as the cited term."},
    "REFERENCE_ONLY": {
        "different_scope": "The cited term is relevant, but the concept and term have different scopes.",
        "contextual": "The GO term describes biology mentioned by the concept. It serves as contextual evidence.",
    },
    "NO_MATCH": {
        "outside_go": "The concept itself falls outside the scope of GO.",
        "no_suitable_hit": "The lookup returned no suitable GO reference for the concept as defined.",
    },
    "UNCERTAIN": {
        "ambiguous": "The evidence leaves multiple interpretations of the concept open.",
        "insufficient": "The available evidence is insufficient for a reliable mapping.",
    },
}


class Decision(StrictModel):
    status: Literal["MATCHED", "REFERENCE_ONLY", "NO_MATCH", "UNCERTAIN"]
    go_id: GoID | None = None
    reason_code: Literal["same_scope", "different_scope", "contextual", "outside_go", "no_suitable_hit", "ambiguous", "insufficient"]
    detail_query_id: Annotated[str, StringConstraints(pattern=r"^q[1-9][0-9]*$")] | None = None
    input_quote: Annotated[str, StringConstraints(max_length=350)] | None = None
    go_quote: Annotated[str, StringConstraints(max_length=350)] | None = None


class EvidenceError(ValueError):
    """A model decision cannot be established from the current transcript."""


class Relation(StrictModel):
    subject: GoID
    predicate: Predicate
    object: GoID
    source: str
    query_id: str


class ScopeComparison(StrictModel):
    local_entails_go: Literal["yes", "no", "unknown"]
    go_entails_local: Literal["yes", "no", "unknown"]
    link: Literal["comparable", "explains_aspect", "topic_only", "unclear"]
    difference: Literal["none", "referent_kind", "mechanism", "participants", "setting",
                        "granularity", "composite", "unclear"]


class ScopeReview(StrictModel):
    candidate_go_id: GoID
    proposed_status: Literal["MATCHED", "REFERENCE_ONLY"]
    term_query_id: str
    outcome: Literal["confirmed", "retained_reference", "different_scope", "contextual", "unrelated", "inconclusive", "unavailable"]
    model_calls: Literal[0, 1]
    comparison: ScopeComparison | None = None
    issue: str | None = None


class ScopeScreen(StrictModel):
    outcome: Literal["outside", "lookup", "inconclusive", "unavailable"]
    model_calls: Literal[1]
    query: str | None = None
    input_quote: str | None = None
    issue: str | None = None


class ReferenceResult(StrictModel):
    run_id: str
    concept_id: str
    status: Status
    go_id: GoID | None
    go_name: str | None
    go_definition: str | None
    relation_type: Literal["equivalent", "related_reference"] | None
    source: str | None
    term_query_id: str | None
    ontology_relations: list[Relation]
    rationale: str
    query_records: list[dict]
    exit_reason: str
    tool_calls: int
    model_calls: int
    format_retry: bool = False  # The first pilot predates the format-retry node.
    scope_screen: ScopeScreen | None = None
    scope_review: ScopeReview | None = None
    elapsed_s: float

    @model_validator(mode="after")
    def consistent_fields(self):
        positive = self.status in ("MATCHED", "REFERENCE_ONLY")
        required = [self.go_id, self.go_name, self.source, self.term_query_id, self.relation_type]
        if (positive and not all(required)) or (not positive and any(required + [self.go_definition, self.ontology_relations])):
            raise ValueError("Inconsistent GO citation")
        if self.status == "MATCHED" and (self.relation_type != "equivalent" or not self.go_definition):
            raise ValueError("A match requires a definition and an equivalence judgment")
        if self.status == "REFERENCE_ONLY" and self.relation_type != "related_reference":
            raise ValueError("A contextual reference requires related_reference")
        if self.scope_review is not None:
            outcomes = {"confirmed": "MATCHED", "retained_reference": "REFERENCE_ONLY", "different_scope": "REFERENCE_ONLY", "contextual": "REFERENCE_ONLY",
                        "unrelated": "UNCERTAIN", "inconclusive": "UNCERTAIN", "unavailable": "LOOKUP_FAILED"}
            review = self.scope_review
            if self.status != outcomes[review.outcome]:
                raise ValueError("Scope review and status disagree")
            if positive and (self.go_id, self.term_query_id) != (review.candidate_go_id, review.term_query_id):
                raise ValueError("Scope review and final citation refer to different evidence")
        query_ids = [r.get("query_id") for r in self.query_records]
        if any(not isinstance(q, str) or not q for q in query_ids) or len(set(query_ids)) != len(query_ids):
            raise ValueError("Missing or duplicate query identifiers in the transcript")
        if positive or self.scope_review:
            query_id = self.term_query_id if positive else self.scope_review.term_query_id
            go_id = self.go_id if positive else self.scope_review.candidate_go_id
            detail, _ = verified_detail(self.query_records, go_id, query_id)
            if positive and (self.go_name, self.go_definition, self.source) != (
                    detail["name"], detail.get("definition"), detail["source"]):
                raise ValueError("Stored citation fields disagree with their detail record")
        if self.ontology_relations:
            observed = direct_edges(self.go_id, self.query_records)
            if any(edge.model_dump() not in observed for edge in self.ontology_relations):
                raise ValueError("Stored ontology edge has no directed query evidence")
        if self.tool_calls != len(self.query_records):
            raise ValueError("Tool count disagrees with transcript")
        return self


def empty_citation():
    return dict(go_id=None, go_name=None, go_definition=None, relation_type=None,
                source=None, term_query_id=None, ontology_relations=[])


def official_source(url):
    if not isinstance(url, str):
        return False
    parsed = urlsplit(url)
    return parsed.scheme == "https" and parsed.netloc == "www.ebi.ac.uk" and parsed.path.startswith("/ols4/")


def direct_edges(go_id, records):
    """Preserve directed, typed evidence incident on the chosen term only."""
    edges = {}
    for row in records:
        if not row.get("ok") or row.get("tool") not in {"children", "relations"}:
            continue
        observation = row.get("result")
        if not isinstance(observation, dict) or not isinstance(observation.get("edges"), list):
            raise EvidenceError("Malformed relationship observation")
        for item in observation["edges"]:
            if not isinstance(item, dict):
                raise EvidenceError("Malformed relationship evidence")
            triple = (item.get("subject"), item.get("predicate"), item.get("object"))
            if (go_id not in (triple[0], triple[2]) or triple[1] not in PREDICATES
                    or not official_source(item.get("source"))
                    or not all(isinstance(i, str) and re.fullmatch(r"GO:[0-9]{7}", i) for i in (triple[0], triple[2]))):
                continue
            edges.setdefault(triple, dict(subject=triple[0], predicate=triple[1], object=triple[2],
                                          source=item["source"], query_id=row["query_id"]))
    return list(edges.values())


def verified_detail(records, go_id, query_id=None):
    for row in reversed(records):
        if (not row.get("ok") or row.get("tool") != "detail"
                or (query_id is not None and row.get("query_id") != query_id)):
            continue
        args, observation = row.get("arguments"), row.get("result")
        if not isinstance(args, dict) or args.get("go_id") != go_id or not isinstance(observation, dict):
            continue
        for term in observation.get("terms") or []:
            if (isinstance(term, dict) and term.get("go_id") == go_id and term.get("name")
                    and official_source(term.get("source"))):
                return term, row["query_id"]
    raise EvidenceError("The selected GO ID lacks a verified detail lookup for this run.")


def verify(decision, concept, records, selected_detail=None):
    allowed = REASONS[decision.status]
    if decision.reason_code not in allowed:
        raise EvidenceError("Reason code conflicts with status.")
    reason = allowed[decision.reason_code]
    if decision.input_quote:
        if not any(decision.input_quote in concept[k] for k in ("name", "definition")):
            raise EvidenceError("The cited concept wording was absent from the input.")
        reason += f' Input wording: “{decision.input_quote}”'
    citation = empty_citation()
    if decision.status in ("MATCHED", "REFERENCE_ONLY"):
        if selected_detail and decision.detail_query_id and selected_detail != decision.detail_query_id:
            raise EvidenceError("Conflicting detail query references.")
        term, query_id = verified_detail(records, decision.go_id, selected_detail or decision.detail_query_id)
        if decision.status == "MATCHED" and not term.get("definition"):
            raise EvidenceError("The selected term lacks a verified definition.")
        if decision.go_quote:
            if decision.go_quote not in (term.get("definition") or ""):
                raise EvidenceError("The GO quotation was absent from the retrieved definition.")
            reason += f' GO definition: “{decision.go_quote}”'
        citation.update(go_id=term["go_id"], go_name=term["name"], go_definition=term.get("definition"),
                        source=term["source"], term_query_id=query_id,
                        relation_type="equivalent" if decision.status == "MATCHED" else "related_reference",
                        ontology_relations=direct_edges(decision.go_id, records))
        return dict(status=decision.status, rationale=reason, **citation)
    if decision.go_id is not None or decision.go_quote or decision.detail_query_id:
        raise EvidenceError("A non-positive decision carried a GO ID.")
    if decision.status == "NO_MATCH":
        if decision.reason_code == "no_suitable_hit" and not any(r["tool"] == "search" and r["ok"] for r in records):
            raise EvidenceError("A negative search conclusion requires a successful search.")
    return dict(status=decision.status, rationale=reason, **citation)
