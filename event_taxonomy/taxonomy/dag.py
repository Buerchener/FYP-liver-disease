from collections import deque
from copy import deepcopy
from ..models import to_dict


def adjacency(node_ids, edges):
    graph = {n: set() for n in node_ids}
    indegree = dict.fromkeys(node_ids, 0)
    pairs = set()
    for edge in edges:
        if edge.relation != "SUBSUMES" or edge.status not in {"accepted", "provisional"}:
            raise ValueError("Invalid taxonomy edge semantics/status")
        if any(not isinstance(value, str) or not value.strip()
               for value in (edge.decision_ref, edge.reason, edge.validator)):
            raise ValueError("Taxonomy edge requires an auditable decision")
        pair = (edge.parent_id, edge.child_id)
        if pair in pairs:
            raise ValueError("Duplicate taxonomy edge")
        pairs.add(pair)
        if edge.parent_id not in graph or edge.child_id not in graph:
            raise ValueError("Dangling taxonomy edge")
        if edge.parent_id == edge.child_id:
            raise ValueError("Taxonomy self-loop")
        graph[edge.parent_id].add(edge.child_id)
        indegree[edge.child_id] += 1
    return graph, indegree


def reachable(graph, start):
    visited, pending = set(), list(graph[start])
    while pending:
        node = pending.pop()
        if node not in visited:
            visited.add(node)
            pending.extend(graph[node])
    return visited


def validate_dag(taxonomy):
    ids = [n.concept_id for n in taxonomy.nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate taxonomy node ID")
    if len(set(taxonomy.predefined_roots)) != len(taxonomy.predefined_roots):
        raise ValueError("Duplicate predefined roots")
    if not set(taxonomy.predefined_roots).issubset(ids):
        raise ValueError("Missing predefined root")
    edge_ids = [e.edge_id for e in taxonomy.parent_child_edges]
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Duplicate taxonomy edge ID")
    graph, indegree = adjacency(ids, taxonomy.parent_child_edges)
    if any(indegree[root] for root in taxonomy.predefined_roots):
        raise ValueError("Predefined root has a parent")
    pending = deque(n for n, degree in indegree.items() if degree == 0)
    visited = []
    while pending:
        node = pending.popleft()
        visited.append(node)
        for child in sorted(graph[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
    if len(visited) != len(ids):
        raise ValueError("Taxonomy cycle")
    inputs = set(taxonomy.frozen_input_concept_ids)
    if not inputs.issubset(ids):
        raise ValueError("Frozen input concept deleted")
    all_reachable = set().union(*(reachable(graph, r) for r in taxonomy.predefined_roots))
    accepted_graph, _ = adjacency(ids, [e for e in taxonomy.parent_child_edges if e.status == "accepted"])
    accepted_reachable = set().union(*(reachable(accepted_graph, r) for r in taxonomy.predefined_roots))
    expected_unplaced = inputs - all_reachable
    if set(taxonomy.unplaced_concept_ids) != expected_unplaced:
        raise ValueError("unplaced_concept_ids does not match root reachability")
    if taxonomy.status == "complete" and (inputs - accepted_reachable or
            taxonomy.input_freeze_status != "semantic_input_frozen"):
        raise ValueError("Unreviewed or disconnected taxonomy cannot be complete")
    return {"acyclic": True, "dangling_references": 0, "nodes": len(ids),
            "edges": len(edge_ids), "root_reachable_inputs": len(inputs & all_reachable),
            "accepted_root_reachable_inputs": len(inputs & accepted_reachable),
            "unplaced_inputs": len(expected_unplaced)}


def apply_edge_transaction(taxonomy, additions, remove_edge_ids=()):
    """Return a validated copy; failure leaves the caller's graph untouched."""
    validate_dag(taxonomy)
    next_state = deepcopy(taxonomy)
    known_edges = {e.edge_id for e in taxonomy.parent_child_edges}
    if not set(remove_edge_ids).issubset(known_edges):
        raise ValueError("Cannot remove unknown edge")
    next_state.parent_child_edges = [e for e in next_state.parent_child_edges if e.edge_id not in remove_edge_ids]
    next_state.parent_child_edges.extend(deepcopy(additions))
    ids = [n.concept_id for n in next_state.nodes]
    graph, _ = adjacency(ids, next_state.parent_child_edges)
    rooted = set().union(*(reachable(graph, r) for r in next_state.predefined_roots))
    next_state.unplaced_concept_ids = sorted(set(next_state.frozen_input_concept_ids) - rooted)
    next_state.status = "partial"
    validate_dag(next_state)
    old_accepted, _ = adjacency(ids, [e for e in taxonomy.parent_child_edges if e.status == "accepted"])
    new_accepted, _ = adjacency(ids, [e for e in next_state.parent_child_edges if e.status == "accepted"])
    for node in ids:
        if not reachable(old_accepted, node).issubset(reachable(new_accepted, node)):
            raise ValueError("Transaction loses an accepted ancestor relation")
    # Input nodes and provenance are immutable in this edge-only operation.
    if [to_dict(n) for n in next_state.nodes] != [to_dict(n) for n in taxonomy.nodes]:
        raise ValueError("Node mutated during edge transaction")
    next_state.version += 1
    return next_state
