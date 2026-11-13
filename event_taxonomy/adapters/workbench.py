"""Read saved workbench files without importing or running its web server."""
from copy import deepcopy
from ..models import ConceptCandidate, stable_id
from .common import add_document, add_event, require_text, source_ref


def _candidate(bundle, event, label, role, ref, namespace, ordinal, **kwargs):
    label = require_text(label, "Concept label")
    candidate = ConceptCandidate(
        stable_id("candidate", namespace, ref.source_run_id, event.event_id,
                  ref.original_id or ref.record_locator, role, ordinal, label),
        label, kwargs.pop("description", None), [event.event_id], list(event.evidence_ids),
        role, ref, namespace, event.data_origin, **kwargs)
    if candidate.candidate_description is not None:
        candidate.description_status = "provided_unverified"
    bundle.candidates.append(candidate)
    return candidate


def import_graph(source, bundle, namespace, origin):
    graph = source.json()
    if graph.get("schema_version") != "workbench-candidate-graph/1":
        raise ValueError("Unsupported workbench graph schema")
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Graph requires nodes and edges")
    by_id = {}
    for i, node in enumerate(nodes):
        nid = require_text(node.get("id"), "Graph node ID")
        if nid in by_id:
            raise ValueError("Duplicate graph node ID: " + nid)
        by_id[nid] = (i, node)
    documents = {}
    for i, node in enumerate(nodes):
        if node.get("kind") == "Source":
            ref = source_ref(source, "workbench_graph", None, "/nodes/" + str(i), node["id"])
            documents[node["id"]] = add_document(bundle, namespace, node.get("text"), node.get("pmid"),
                                                 node.get("label"), origin, ref)
    event_map = {}
    for i, node in enumerate(nodes):
        if node.get("kind") != "Event":
            continue
        run_ids = node.get("source_run_ids", [])
        if len(run_ids) != 1 or not isinstance(run_ids[0], str):
            raise ValueError("Event node must have exactly one source run ID")
        run_id = run_ids[0]
        doc = documents.get(node.get("source_node_id"))
        if doc is None or str(node.get("source_pmid")) != str(doc.source_pmid):
            raise ValueError("Event source/PMID does not resolve to its Document")
        index = node.get("prediction_index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("Event prediction_index must be a nonnegative integer")
        ref = source_ref(source, "workbench_graph", run_id, "/nodes/" + str(i), node["id"])
        event = add_event(bundle, doc, namespace, "workbench_graph", run_id,
                          {"Event": node.get("label"), "Event_ID": node.get("local_id"),
                           "_evidence": node.get("evidence"), "Trigger": node.get("trigger"),
                           "Arguments": node.get("arguments", []), "context": node.get("context"),
                           "assertion": node.get("assertion"),
                           "review_status": node.get("review", {}).get("status", "pending")},
                          index, ref, node["id"])
        event_map[node["id"]] = event
        bundle.concept_generation_records.append({"event_id": event.event_id,
                                                   "upstream_status": node.get("concept_status", "missing"),
                                                   "abstention_reason": node.get("abstention_reason")})
    seen_edges = set()
    for i, edge in enumerate(edges):
        edge_id = require_text(edge.get("id"), "Graph edge ID")
        if edge_id in seen_edges:
            raise ValueError("Duplicate graph edge ID: " + edge_id)
        seen_edges.add(edge_id)
        if edge.get("head") not in by_id or edge.get("tail") not in by_id:
            raise ValueError("Dangling source graph edge: " + edge_id)
        if edge.get("head") not in event_map or edge.get("kind") not in {"HAS_TYPE", "RELATED_TOPIC"}:
            continue
        event = event_map[edge["head"]]
        concept = by_id[edge["tail"]][1]
        upstream_role = "types" if edge["kind"] == "HAS_TYPE" else "topics"
        if concept.get("kind") != "Concept" or concept.get("concept_domain") != "Event" or concept.get("concept_role") != upstream_role:
            raise ValueError("Event concept role/domain mismatch")
        if str(edge.get("source_pmid")) != str(event.source_pmid):
            raise ValueError("Concept assignment PMID mismatch")
        ref = source_ref(source, "workbench_graph", event.source_run_id, "/edges/" + str(i), edge_id)
        _candidate(bundle, event, edge.get("original_label") or concept.get("label"),
                   "event_type" if upstream_role == "types" else "topic", ref, namespace, 0,
                   upstream_concept_id=concept["id"],
                   upstream_normalized_label=concept.get("normalized_label"),
                   description=concept.get("concept_description"),
                   generation={"provider": "saved_workbench_graph", "model_requested": None,
                               "model_returned": None, "prompt_hash": None},
                   validation_reasons=["upstream_assignment_not_semantically_validated"])
    for i, rejected in enumerate(graph.get("rejected", [])):
        bundle.issues.append({"kind": "upstream_graph_rejection", "raw": deepcopy(rejected),
                              "source_ref": source_ref(source, "workbench_graph", None,
                                                       "/rejected/" + str(i)).__dict__})


def import_workspace(source, bundle, namespace, origin, annotation_export=False):
    payload = source.json()
    if annotation_export:
        run = dict(payload["run"], events=payload["events"], articleSnapshot=payload["article"])
        run["reviews"] = {e["Event_ID"]: e.get("Review", {}) for e in run["events"]}
        runs = [run]
    else:
        runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Workbench workspace requires runs")
    seen_runs = set()
    for ri, run in enumerate(runs):
        rid = require_text(run.get("id"), "Workbench run ID")
        if rid in seen_runs:
            raise ValueError("Duplicate workbench run ID")
        seen_runs.add(rid)
        article = run.get("articleSnapshot", {})
        base = "/" if annotation_export else "/runs/" + str(ri)
        profile = "workbench_export" if annotation_export else "workbench_workspace"
        doc = add_document(bundle, namespace, article.get("abstract"), article.get("pmid"),
                           article.get("title"), origin,
                           source_ref(source, profile, rid, base + "/articleSnapshot", article.get("id")))
        rows = run.get("events")
        if not isinstance(rows, list):
            raise ValueError("Workbench run requires events")
        local = {}
        for i, row in enumerate(rows):
            eid = require_text(row.get("Event_ID"), "Workbench Event_ID")
            if eid in local:
                raise ValueError("Ambiguous workbench Event_ID: " + eid)
            record = dict(row, review_status=run.get("reviews", {}).get(eid, {}).get("status", "pending"))
            ref = source_ref(source, profile, rid, base + "/events/" + str(i), eid)
            local[eid] = add_event(bundle, doc, namespace, profile, rid, record, i, ref)
        concepts = run.get("eventConcepts")
        if concepts is None:
            for e in local.values():
                bundle.concept_generation_records.append({"event_id": e.event_id, "upstream_status": "not_present"})
            continue
        results = concepts.get("results")
        if not isinstance(results, list):
            raise ValueError("eventConcepts.results must be an array")
        seen = set()
        for i, row in enumerate(results):
            eid = row.get("eventId")
            if eid in seen or eid not in local:
                raise ValueError("Unknown or repeated conceptualized Event_ID")
            seen.add(eid)
            event = local[eid]
            labels = row.get("labels")
            if not isinstance(labels, list):
                raise ValueError("Concept labels must be an array")
            stale = row.get("event") != event.event_text
            bundle.concept_generation_records.append({"event_id": event.event_id,
                "upstream_status": "stale" if stale else ("generated" if labels else "explicit_empty"),
                "source_event_text": row.get("event")})
            for j, label in enumerate(labels):
                ref = source_ref(source, profile, rid, base + "/eventConcepts/results/" + str(i) + "/labels/" + str(j), eid)
                _candidate(bundle, event, label, "unknown", ref, namespace, j, stale=stale,
                    validation_status="deferred" if stale else "proposed",
                    validation_reasons=["event_text_changed"] if stale else ["legacy_labels_are_not_typed"],
                    generation={"provider": "saved_workbench_button", "model_requested": concepts.get("model"),
                                "model_returned": None, "prompt_version": concepts.get("promptVersion"),
                                "created_at": concepts.get("createdAt")})
        for eid, event in local.items():
            if eid not in seen:
                bundle.concept_generation_records.append({"event_id": event.event_id, "upstream_status": "not_present"})
