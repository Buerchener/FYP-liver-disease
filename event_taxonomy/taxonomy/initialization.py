from copy import deepcopy
from ..models import ParentChildEdge, Taxonomy, UniqueConcept, stable_id, to_dict
from .dag import validate_dag


def read_roots(payload):
    ids = payload.get("predefined_roots")
    rows = payload.get("nodes")
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or not isinstance(rows, list):
        raise ValueError("Invalid root definition file")
    if len(rows) != len(ids) or {r.get("concept_id") for r in rows} != set(ids):
        raise ValueError("Root nodes and predefined_roots differ")
    if payload.get("parent_child_edges"):
        raise ValueError("Task 1 seed must contain roots only")
    roots = []
    for row in rows:
        if not isinstance(row.get("concept_description"), str) or not row["concept_description"].strip():
            raise ValueError("Predefined root definition is required")
        if not isinstance(row.get("concept_label"), str) or not row["concept_label"].strip():
            raise ValueError("Predefined root label is required")
        roots.append(UniqueConcept(row["concept_id"], row["concept_label"], row["concept_description"],
            [], [], "predefined:" + row["concept_id"], "predefined/1", [], [], "event_type",
            origin_kind="predefined_root", validation_status="predefined",
            granularity_status="broad", description_provenance="frozen_root_configuration"))
    return roots, ids


def build_initial_taxonomy(concepts, candidates, roots_payload, assignments=None):
    roots, root_ids = read_roots(roots_payload)
    inputs = [deepcopy(c) for c in concepts if c.concept_role == "event_type"]
    by_id = {c.concept_id: c for c in inputs}
    edges = []
    for row in assignments or []:
        cid = row.get("concept_id")
        parent = row.get("parent_id")
        if cid not in by_id or parent not in root_ids:
            raise ValueError("Root assignment references an unknown concept/root")
        status = row.get("status")
        if status not in {"provisional", "accepted"}:
            raise ValueError("Root assignment status must be provisional or accepted")
        for field in ("decision_id", "reason", "validator"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError("Root assignment needs " + field)
        if status == "accepted" and not by_id[cid].concept_description:
            raise ValueError("Accepted assignment requires a defined concept")
        edges.append(ParentChildEdge(stable_id("edge", parent, cid), parent, cid, status,
                                      row["decision_id"], row["reason"], row["validator"]))
        by_id[cid].provisional_parent_ids.append(parent)
    candidate_map = {c.candidate_id: c for c in candidates}
    links = []
    for concept in inputs:
        for event_id in concept.supporting_event_ids:
            matching = [candidate_map[cid] for cid in concept.candidate_ids
                        if event_id in candidate_map[cid].supporting_event_ids]
            links.append({"schema_version": "event-taxonomy/1", "event_id": event_id,
                "concept_id": concept.concept_id, "relation": "IS_INSTANCE_OF", "status": "proposed",
                "evidence_ids": sorted({eid for c in matching for eid in c.evidence_ids}),
                "candidate_ids": sorted(c.candidate_id for c in matching),
                "decision_ref": "imported_type_assignment_pending_semantic_review"})
    mapping = [{"concept_id": c.concept_id, "event_ids": list(c.supporting_event_ids),
                "evidence_ids": list(c.evidence), "mapping_kind": "direct", "via_child_ids": []}
               for c in inputs]
    placed = {e.child_id for e in edges}
    taxonomy = Taxonomy(stable_id("taxonomy", [to_dict(c) for c in roots + inputs],
                                 [to_dict(e) for e in edges]),
                        roots + inputs, edges, root_ids, mapping, links,
                        sorted(set(by_id) - placed), sorted(by_id), "partial")
    validate_dag(taxonomy)
    return taxonomy


def prepared_input_snapshot(taxonomy):
    wanted = set(taxonomy.frozen_input_concept_ids)
    return {"schema_version": "event-taxonomy-input-snapshot/1",
            "status": "prepared_snapshot_pending_semantic_review",
            "eligible_for_induction": False,
            "reason": "Input definitions, type validity and granularity are not semantically reviewed.",
            "concept_ids": sorted(wanted),
            "concepts": [to_dict(c) for c in taxonomy.nodes if c.concept_id in wanted],
            "concept_evidence_mapping": deepcopy(taxonomy.concept_evidence_mapping),
            "instance_concept_links": deepcopy(taxonomy.instance_concept_links)}
