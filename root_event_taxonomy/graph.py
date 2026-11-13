"""Transactions, ancestry context, and final conservative DAG cleanup."""
from collections import defaultdict
from copy import deepcopy

from event_taxonomy.induction.contracts import CandidateError
from event_taxonomy.induction.graph import assert_preserved, norm
from event_taxonomy.models import ParentChildEdge, UniqueConcept, stable_id
from event_taxonomy.taxonomy.dag import adjacency, reachable, validate_dag


class GraphView:
    def __init__(self, taxonomy):
        self.nodes = {n.concept_id: n for n in taxonomy.nodes}
        self.roots = set(taxonomy.predefined_roots)
        self.children, _ = adjacency(self.nodes, taxonomy.parent_child_edges)
        self.parents = {n: set() for n in self.nodes}
        for edge in taxonomy.parent_child_edges:
            self.parents[edge.child_id].add(edge.parent_id)
        self.rooted = set().union(*(reachable(self.children, root) for root in self.roots)) | self.roots

    def frontier(self, scope):
        return sorted(n for n in scope if not self.parents[n] - self.roots)

    def paths(self, node_id, cap):
        paths, pending = [], [(node_id,)]
        while pending and len(paths) < cap:
            path = pending.pop()
            parents = self.parents[path[0]]
            if parents:
                pending.extend((p, *path) for p in sorted(parents, reverse=True))
            else:
                paths.append(path)
        return paths, bool(pending)


def refresh(taxonomy):
    view = GraphView(taxonomy)
    taxonomy.unplaced_concept_ids = sorted(set(taxonomy.frozen_input_concept_ids) - view.rooted)
    taxonomy.status = "partial"
    validate_dag(taxonomy)
    return view


def edge(parent, child, reason, decision, model):
    return ParentChildEdge(
        edge_id=stable_id("root-induction-edge", parent, child),
        parent_id=parent,
        child_id=child,
        status="provisional",
        decision_ref=decision,
        reason=reason,
        validator=model,
    )


def update_graph(before, node=None, edges=()):
    updated = deepcopy(before)
    if node is not None:
        updated.nodes.append(node)
        updated.concept_evidence_mapping.append({
            "concept_id": node.concept_id,
            "event_ids": node.supporting_event_ids,
            "evidence_ids": node.evidence,
            "mapping_kind": "derived_from_descendants",
            "via_child_ids": sorted(e.child_id for e in edges if e.parent_id == node.concept_id),
        })
    updated.parent_child_edges.extend(edges)
    updated.version += 1
    try:
        refresh(updated)
    except ValueError as error:
        raise CandidateError(str(error)) from error
    return updated


def intermediate(taxonomy, data, proposal, child_nodes, decision):
    name, definition = proposal["name"].strip(), proposal["definition"].strip()
    for existing in taxonomy.nodes:
        if norm(name) in {norm(s) for s in [existing.concept_label, *existing.aliases]}:
            raise CandidateError("Concept label or alias already exists: " + existing.concept_id)
        if norm(definition) == norm(data.definition(existing)):
            raise CandidateError("Exact concept definition already exists: " + existing.concept_id)
    return UniqueConcept(
        concept_id=stable_id("intermediate", taxonomy.taxonomy_id, norm(name)),
        concept_label=name,
        concept_description=definition,
        candidate_ids=[],
        aliases=[],
        normalization_key=norm(name),
        normalization_version="root-induction/1",
        supporting_event_ids=sorted({e for n in child_nodes for e in n.supporting_event_ids}),
        evidence=sorted({e for n in child_nodes for e in n.evidence}),
        concept_role="event_type",
        origin_kind="induced_intermediate",
        validation_status="llm_review_pending",
        granularity_status="intermediate",
        description_provenance=decision,
    )


def duplicate_keys(taxonomy, data):
    names, definitions = defaultdict(set), defaultdict(set)
    for node in taxonomy.nodes:
        for value in [node.concept_label, *node.aliases]:
            names[norm(value)].add(node.concept_id)
        definition = data.definition(node)
        if definition:
            definitions[norm(definition)].add(node.concept_id)
    return {
        field: [{"value": key, "concept_ids": sorted(ids)} for key, ids in sorted(index.items()) if len(ids) > 1]
        for field, index in (("names_or_aliases", names), ("definitions", definitions))
    }


def finalize(initial, working, valid_edges, distinct_nodes):
    """Keep independently checked changes; leave all original records intact."""
    original_nodes = {n.concept_id for n in initial.nodes}
    original_edges = {e.edge_id for e in initial.parent_child_edges}
    induced_ids = {n.concept_id for n in working.nodes} - original_nodes
    quarantined = induced_ids - distinct_nodes
    final = deepcopy(working)
    edges = [e for e in final.parent_child_edges if e.edge_id in original_edges | valid_edges]
    while True:
        edges = [e for e in edges if e.parent_id not in quarantined and e.child_id not in quarantined]
        children = defaultdict(set)
        for e in edges:
            children[e.parent_id].add(e.child_id)
        unsupported = {n for n in induced_ids - quarantined if len(children[n]) < 2}
        if not unsupported:
            break
        quarantined.update(unsupported)

    final.nodes = [n for n in final.nodes if n.concept_id not in quarantined]
    final.concept_evidence_mapping = [m for m in final.concept_evidence_mapping if m["concept_id"] not in quarantined]
    final.parent_child_edges = edges
    view = refresh(final)
    redundant = set()
    for e in final.parent_child_edges:
        if e.edge_id in original_edges:
            continue
        view.children[e.parent_id].remove(e.child_id)
        if e.child_id in reachable(view.children, e.parent_id):
            redundant.add(e.edge_id)
        else:
            view.children[e.parent_id].add(e.child_id)
    final.parent_child_edges = [e for e in final.parent_child_edges if e.edge_id not in redundant]
    for row in final.concept_evidence_mapping:
        node_id = row["concept_id"]
        if node_id not in induced_ids:
            continue
        node = view.nodes[node_id]
        sources = [view.nodes[n] for n in reachable(view.children, node_id) & original_nodes]
        node.supporting_event_ids = sorted({e for n in sources for e in n.supporting_event_ids})
        node.evidence = sorted({e for n in sources for e in n.evidence})
        row.update(event_ids=node.supporting_event_ids, evidence_ids=node.evidence,
                   via_child_ids=sorted(view.children[node_id]))

    final.version += 1
    refresh(final)
    assert_preserved(initial, final, accepted=True)
    removed = {e.edge_id for e in working.parent_child_edges} - {e.edge_id for e in final.parent_child_edges}
    return final, {"quarantined_nodes": sorted(quarantined), "removed_edges": sorted(removed),
                   "redundant_edges": sorted(redundant)}
