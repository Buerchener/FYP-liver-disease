"""Incremental placement of independently reviewed entity relations."""

from copy import deepcopy
from graphlib import CycleError, TopologicalSorter


def _check_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string.")


def _check_node(node):
    for field in ("id", "kind", "name", "definition"):
        _check_text(node.get(field), field)


def _check_ids(ids, label):
    for ident in ids:
        _check_text(ident, label)
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate candidate in {label}.")


def _edge_rows(edges):
    return [{"child_id": child, "parent_id": parent} for child, parent in sorted(edges)]


def _reduce_edges(nodes, roots, edges):
    """Validate same-type rootedness and remove edges implied by another path."""
    parents = {ident: set() for ident in nodes}
    for child, parent in edges:
        if child not in nodes or parent not in nodes:
            raise ValueError("An edge refers to an unknown node.")
        if child in roots:
            raise ValueError("A type root cannot have a parent.")
        if nodes[child]["kind"] != nodes[parent]["kind"]:
            raise ValueError("An is_a edge must stay within its entity type.")
        parents[child].add(parent)
    if any(not upstream and ident not in roots for ident, upstream in parents.items()):
        raise ValueError("A nonroot concept must reach its type root.")

    ancestors = {}
    for ident in TopologicalSorter(parents).static_order():
        direct = parents[ident]
        ancestors[ident] = direct.union(*(ancestors[parent] for parent in direct))
    return {
        (child, parent)
        for child, parent in edges
        if not any(parent in ancestors[other] for other in parents[child] - {parent})
    }


class EntityDAG:
    """Keep one rooted is_a DAG per entity type, with child-to-parent edges."""

    def __init__(self, kinds):
        kinds = tuple(kinds)
        _check_ids(kinds, "entity type")
        self.roots = {kind: f"type:{kind}" for kind in kinds}
        self.nodes = {
            ident: {
                "id": ident,
                "name": kind,
                "kind": kind,
                "definition": f"Coarse entity type {kind}.",
                "aliases": [],
                "source_mentions": [],
                "origin": "schema_root",
            }
            for kind, ident in self.roots.items()
        }
        self.edges = set()

    def place(self, concept, *, parent_ids=(), child_ids=(), equivalent_ids=(), generated_parent=None):
        """Stage a reviewed placement and commit only after graph validation."""
        _check_node(concept)
        ident, kind = concept["id"], concept["kind"]
        root = self.roots.get(kind)
        if root is None or ident.startswith("type:") or ident in self.nodes:
            raise ValueError(f"Invalid or already placed concept ID {ident!r} for {kind!r}.")
        parents, children, equivalents = map(tuple, (parent_ids, child_ids, equivalent_ids))
        record = {
            "concept_id": ident,
            "source_mentions": list(concept.get("source_mentions", ())),
            "parent_ids": [],
            "child_ids": [],
            "equivalent_ids": [],
            "generated_parent_id": None,
            "added_edges": [],
            "removed_edges": [],
        }

        try:
            self._check_candidates(kind, parents, children, equivalents)
            if equivalents:
                return {
                    **record,
                    "action": "equivalent_pending",
                    "equivalent_ids": list(equivalents),
                    "reason": "Equivalent candidate held for an explicit identity decision.",
                }

            staged_nodes = self.nodes | {ident: deepcopy(concept)}
            staged_edges = self.edges | {(child, ident) for child in children}
            generated_id = None
            if generated_parent is not None:
                generated = deepcopy(generated_parent)
                _check_node(generated)
                generated_id = generated["id"]
                if generated["kind"] != kind or generated_id.startswith("type:") or generated_id in staged_nodes:
                    raise ValueError("The proposed parent needs a new ID in the same entity type.")
                upstream = tuple(generated.pop("parent_ids", ()))
                self._check_candidates(kind, upstream, (), ())
                generated.setdefault("aliases", [])
                generated.setdefault("source_mentions", [])
                generated.setdefault("origin", "proposed_broader_class")
                staged_nodes[generated_id] = generated
                staged_edges.update((generated_id, parent) for parent in upstream or (root,))
                parents = (*parents, generated_id)

            staged_edges.update((ident, parent) for parent in parents or (root,))
            reduced = _reduce_edges(staged_nodes, set(self.roots.values()), staged_edges)
        except CycleError:
            return {**record, "action": "rejected", "reason": "The proposed relations would form a cycle."}
        except ValueError as error:
            return {**record, "action": "rejected", "reason": str(error)}

        direct_parents = sorted(parent for child, parent in reduced if child == ident)
        direct_children = sorted(child for child, parent in reduced if parent == ident)
        if direct_children:
            action = "as_parent" if direct_parents == [root] else "insert_between"
        else:
            action = "new_under_root" if direct_parents == [root] else "as_child"
        record.update(
            action=action,
            reason="Validated placement in the entity type hierarchy.",
            parent_ids=direct_parents,
            child_ids=direct_children,
            generated_parent_id=generated_id,
            added_edges=_edge_rows(reduced - self.edges),
            removed_edges=_edge_rows(self.edges - reduced),
        )
        self.nodes, self.edges = staged_nodes, reduced
        return record

    def _check_candidates(self, kind, parents, children, equivalents):
        for ids, label in ((parents, "parent"), (children, "child"), (equivalents, "equivalent")):
            _check_ids(ids, label)
            for ident in ids:
                if ident not in self.nodes or self.nodes[ident]["kind"] != kind:
                    raise ValueError(f"{label.title()} candidate {ident!r} is outside the current entity type.")
                if ident == self.roots[kind] and label != "parent":
                    raise ValueError("A type root is eligible only as a parent.")
        if len(set((*parents, *children, *equivalents))) != len(parents) + len(children) + len(equivalents):
            raise ValueError("A candidate cannot play more than one role in a placement.")
