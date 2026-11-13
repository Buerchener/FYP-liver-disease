"""Explicit instance/candidate/unique-concept contracts; no extraction imports."""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import json

SCHEMA_VERSION = "event-taxonomy/1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_id(kind: str, *parts: Any) -> str:
    return kind + ":" + hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()[:32]


@dataclass
class SourceRef:
    source_profile: str
    source_run_id: Optional[str]
    file_sha256: str
    record_locator: str
    original_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION


@dataclass
class Document:
    document_id: str
    source_pmid: Optional[str]
    title: str
    text: str
    text_sha256: str
    data_origin: str
    source_refs: List[SourceRef]
    text_scope: str = "source_passage"
    schema_version: str = SCHEMA_VERSION


@dataclass
class Evidence:
    evidence_id: str
    document_id: str
    text: Optional[str]
    spans: List[Dict[str, int]]
    granularity: str
    alignment_status: str
    source_ref: SourceRef
    upstream_alignment_status: Optional[str] = None
    offset_unit: str = "unicode_code_point"
    locator_version: str = "literal_then_whitespace/1"
    schema_version: str = SCHEMA_VERSION


@dataclass
class EventInstance:
    event_id: str
    event_text: str
    source_pmid: Optional[str]
    document_id: str
    evidence_sentence: Optional[str]
    evidence_ids: List[str]
    local_event_id: Optional[str]
    source_run_id: str
    source_profile: str
    source_ref: SourceRef
    prediction_index: int
    trigger: Optional[str] = None
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    assertion: str = "unknown"
    context: Optional[str] = None
    review_status: str = "pending"
    validation_issues: List[Dict[str, Any]] = field(default_factory=list)
    data_origin: str = "real_literature"
    upstream_event_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION


@dataclass
class ConceptCandidate:
    candidate_id: str
    candidate_label: str
    candidate_description: Optional[str]
    supporting_event_ids: List[str]
    evidence_ids: List[str]
    candidate_role: str
    source_ref: SourceRef
    source_namespace: str
    data_origin: str
    candidate_origin: str = "existing_model_output"
    description_status: str = "missing"
    granularity_status: str = "unresolved"
    validation_status: str = "proposed"
    validation_reasons: List[str] = field(default_factory=list)
    provisional_parent_ids: List[str] = field(default_factory=list)
    generation: Dict[str, Any] = field(default_factory=dict)
    upstream_concept_id: Optional[str] = None
    upstream_normalized_label: Optional[str] = None
    disambiguation_key: Optional[str] = None
    stale: bool = False
    schema_version: str = SCHEMA_VERSION


@dataclass
class UniqueConcept:
    concept_id: str
    concept_label: str
    concept_description: Optional[str]
    candidate_ids: List[str]
    aliases: List[str]
    normalization_key: str
    normalization_version: str
    supporting_event_ids: List[str]
    evidence: List[str]
    concept_role: str
    upstream_concept_ids: List[str] = field(default_factory=list)
    provisional_parent_ids: List[str] = field(default_factory=list)
    origin_kind: str = "input_concept"
    validation_status: str = "proposed"
    granularity_status: str = "unresolved"
    description_provenance: Optional[str] = None
    review_history: List[Dict[str, Any]] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


@dataclass
class ParentChildEdge:
    edge_id: str
    parent_id: str
    child_id: str
    status: str
    decision_ref: str
    reason: str
    validator: str
    relation: str = "SUBSUMES"
    schema_version: str = SCHEMA_VERSION


@dataclass
class Taxonomy:
    taxonomy_id: str
    nodes: List[UniqueConcept]
    parent_child_edges: List[ParentChildEdge]
    predefined_roots: List[str]
    concept_evidence_mapping: List[Dict[str, Any]]
    instance_concept_links: List[Dict[str, Any]]
    unplaced_concept_ids: List[str]
    frozen_input_concept_ids: List[str]
    status: str
    version: int = 0
    input_freeze_status: str = "prepared_snapshot_pending_semantic_review"
    manifest_ref: str = "manifest.json"
    schema_version: str = SCHEMA_VERSION


@dataclass
class ImportBundle:
    documents: List[Document] = field(default_factory=list)
    events: List[EventInstance] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    candidates: List[ConceptCandidate] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    concept_generation_records: List[Dict[str, Any]] = field(default_factory=list)


def to_dict(value: Any) -> Dict[str, Any]:
    return asdict(value)
