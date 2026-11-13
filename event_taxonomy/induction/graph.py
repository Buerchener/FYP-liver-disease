from copy import deepcopy
import hashlib
import unicodedata
from ..models import Taxonomy, UniqueConcept, ParentChildEdge, canonical_json, stable_id, to_dict
from ..taxonomy.dag import adjacency, reachable, validate_dag
from .contracts import CandidateError


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def norm(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_taxonomy(value):
    value = deepcopy(value)
    value["nodes"] = [UniqueConcept(**n) for n in value["nodes"]]
    value["parent_child_edges"] = [ParentChildEdge(**e) for e in value["parent_child_edges"]]
    result = Taxonomy(**value)
    validate_dag(result)
    return result


def children(taxonomy, parent):
    return sorted(e.child_id for e in taxonomy.parent_child_edges if e.parent_id == parent)


def shortest_depths(taxonomy):
    graph, _ = adjacency([n.concept_id for n in taxonomy.nodes], taxonomy.parent_child_edges)
    depths = {r: 0 for r in taxonomy.predefined_roots}
    pending = list(taxonomy.predefined_roots)
    for parent in pending:
        for child in sorted(graph[parent]):
            if child not in depths or depths[child] > depths[parent] + 1:
                depths[child] = depths[parent] + 1; pending.append(child)
    return depths


def structure_signature(taxonomy):
    return digest({"nodes": sorted((n.concept_id, n.concept_label, n.concept_description) for n in taxonomy.nodes),
                   "edges": sorted((e.parent_id, e.child_id) for e in taxonomy.parent_child_edges)})


def context_parts(taxonomy, parent):
    by_id = {n.concept_id: n for n in taxonomy.nodes}
    if parent not in by_id:
        raise ValueError("Unknown parent")
    reverse = {n: set() for n in by_id}
    for e in taxonomy.parent_child_edges:
        reverse[e.child_id].add(e.parent_id)
    ancestors, queue = set(), list(reverse[parent])
    while queue:
        node = queue.pop()
        if node not in ancestors:
            ancestors.add(node); queue.extend(reverse[node])
    direct = set(children(taxonomy, parent))
    incident = [e for e in taxonomy.parent_child_edges
                if e.child_id in ancestors | {parent} | direct or e.parent_id == parent]
    relevant = ancestors | direct | {parent} | {e.parent_id for e in incident}
    return by_id, ancestors, direct, incident, relevant


def local_signature(taxonomy, parent):
    by_id, _, _, edges, relevant = context_parts(taxonomy, parent)
    return digest({"nodes": sorted((i, by_id[i].concept_label, by_id[i].concept_description) for i in relevant),
                   "edges": sorted((e.parent_id, e.child_id, e.status) for e in edges)})


def assert_preserved(before, after, accepted=False):
    validate_dag(after)
    old = {n.concept_id: to_dict(n) for n in before.nodes}
    new = {n.concept_id: to_dict(n) for n in after.nodes}
    if any(new.get(n) != data for n, data in old.items()):
        raise CandidateError("Original concept/root identity, definition or evidence changed")
    for field in ["predefined_roots", "frozen_input_concept_ids", "instance_concept_links"]:
        if getattr(before, field) != getattr(after, field):
            raise CandidateError("Original instance links or input/root set changed")
    mappings = {row["concept_id"]: row for row in after.concept_evidence_mapping}
    if len(mappings) != len(after.concept_evidence_mapping):
        raise CandidateError("Duplicated concept evidence mapping")
    if any(mappings.get(row["concept_id"]) != row for row in before.concept_evidence_mapping):
        raise CandidateError("Original concept evidence mapping changed")
    old_graph, _ = adjacency(old, before.parent_child_edges)
    new_graph, _ = adjacency(new, after.parent_child_edges)
    for nid in old:
        if not reachable(old_graph, nid).issubset(reachable(new_graph, nid)):
            raise CandidateError("Existing ancestor paths were lost")
    if accepted:
        old_graph, _ = adjacency(old, [e for e in before.parent_child_edges if e.status == "accepted"])
        new_graph, _ = adjacency(new, [e for e in after.parent_child_edges if e.status == "accepted"])
        for nid in old:
            if not reachable(old_graph, nid).issubset(reachable(new_graph, nid)):
                raise CandidateError("Confirmed ancestor paths were downgraded")


def build_candidate(before, parent, response, max_new_nodes):
    """Pure transaction: add abstractions, rewire selected edges only, retain outliers."""
    proposals = response["intermediates"]
    if not proposals:
        raise CandidateError("no_intermediates_proposed")
    induced_count = sum(n.origin_kind == "induced_intermediate" for n in before.nodes)
    if induced_count + len(proposals) > max_new_nodes:
        raise CandidateError("maximum_new_nodes")
    by_id = {n.concept_id: n for n in before.nodes}
    direct = set(children(before, parent))
    existing_labels = {norm(s) for n in before.nodes for s in [n.concept_label] + n.aliases}
    existing_defs = {norm(n.concept_description) for n in before.nodes if n.concept_description}
    group_sets = set()
    after = deepcopy(before)
    added_nodes, added_edges, moved = [], [], set()
    for proposal in proposals:
        group = set(proposal["child_ids"])
        if len(group) < 2 or len(group) != len(proposal["child_ids"]) or not group <= direct:
            raise CandidateError("group_is_not_multiple_current_direct_children")
        if frozenset(group) in group_sets:
            raise CandidateError("duplicate_group")
        group_sets.add(frozenset(group))
        label, definition = norm(proposal["label"]), norm(proposal["description"])
        if not label or not definition or label in existing_labels or definition in existing_defs:
            raise CandidateError("duplicate_or_empty_intermediate_concept")
        existing_labels.add(label); existing_defs.add(definition)
        nid = stable_id("intermediate", parent, label, definition, sorted(group))
        event_ids = sorted({x for child in group for x in by_id[child].supporting_event_ids})
        evidence_ids = sorted({x for child in group for x in by_id[child].evidence})
        node = UniqueConcept(nid, proposal["label"], proposal["description"], [], [],
            "induced:" + digest([parent, label, definition, sorted(group)]), "induced/1",
            event_ids, evidence_ids, "event_type", origin_kind="induced_intermediate",
            validation_status="proposed", description_provenance="llm_induction_proposal")
        after.nodes.append(node); added_nodes.append(to_dict(node))
        after.concept_evidence_mapping.append({"concept_id": nid, "event_ids": event_ids,
            "evidence_ids": evidence_ids, "mapping_kind": "derived_from_descendants", "via_child_ids": sorted(group)})
        for p, c in [(parent, nid)] + [(nid, x) for x in sorted(group)]:
            edge = ParentChildEdge(stable_id("edge", p, c), p, c, "provisional", "candidate_pending_validation",
                                   response["reasoning"], "llm_proposal")
            after.parent_child_edges.append(edge); added_edges.append(to_dict(edge))
        moved.update(group)
    removed = [to_dict(e) for e in before.parent_child_edges if e.parent_id == parent and e.child_id in moved]
    after.parent_child_edges = [e for e in after.parent_child_edges if not (e.parent_id == parent and e.child_id in moved)]
    after.version += 1; after.status = "partial"
    assert_preserved(before, after)
    if structure_signature(before) == structure_signature(after):
        raise CandidateError("no_structural_change")
    return after, {"added_nodes": added_nodes, "added_edges": added_edges, "removed_edges": removed,
                   "unchanged_direct_children": sorted(direct - moved)}


def accept_candidate(before, candidate, changes, validation, decision_ref):
    after = deepcopy(candidate)
    nodes = {n["concept_id"] for n in changes["added_nodes"]}
    edges = {e["edge_id"] for e in changes["added_edges"]}
    for node in after.nodes:
        if node.concept_id in nodes:
            node.validation_status = "llm_validated_pending_independent_review"
            node.review_history.append({"decision_ref": decision_ref, "decision": validation["decision"],
                                       "reasoning": validation["reasoning"], "confidence": validation["confidence"]})
    for edge in after.parent_child_edges:
        if edge.edge_id in edges:
            edge.status = "accepted"; edge.validator = "LLM_structure_validation"
            edge.decision_ref = decision_ref; edge.reason = validation["reasoning"]
    assert_preserved(before, after, accepted=True)
    return after
