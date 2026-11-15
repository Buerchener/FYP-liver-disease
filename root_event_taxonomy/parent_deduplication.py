"""Find equivalent parents and merge only independently validated graph edits."""

from copy import deepcopy

import numpy as np

from event_taxonomy.induction.contracts import CandidateError, ContractError, obj, require, text
from event_taxonomy.induction.graph import norm
from event_taxonomy.models import stable_id, to_dict
from event_taxonomy.taxonomy.dag import reachable, validate_dag

from .graph import GraphView, edge
from .maintenance import assert_frozen_preserved, refresh_derived_support
from .upward_review import TRUTH_VALUES, by_review_id, checked_call, describe, truth_value


INACTIVE_STATUSES = {"admission_held", "maintenance_held", "merged", "merged_duplicate"}


def check_equivalence(response, expected):
    """Both directions must be supported; a name match is never sufficient."""
    obj(response, ("checks",))
    checks = by_review_id(response["checks"], "pair_id", set(expected))
    for check in checks.values():
        obj(check, ("pair_id", "left_entails_right", "right_entails_left"))
        for direction in ("left_entails_right", "right_entails_left"):
            result = check[direction]
            obj(result, ("verdict", "reason", "counterexample"))
            result["verdict"] = truth_value(result["verdict"])
            require(result["verdict"] in TRUTH_VALUES, "Invalid equivalence direction verdict")
            text(result["reason"])
            if result["verdict"] == "false":
                text(result["counterexample"])
            elif result["verdict"] == "true":
                require(result["counterexample"] is None, "A true direction cannot include a counterexample")
            elif result["counterexample"] is not None:
                text(result["counterexample"])
        check["equivalent"] = all(check[direction]["verdict"] == "true"
                                  for direction in ("left_entails_right", "right_entails_left"))
    return checks


def candidate_pairs(state, inducer, neighbors):
    """Exact matches and bounded nearest neighbors, with explicit coverage."""
    if not isinstance(neighbors, int) or isinstance(neighbors, bool) or neighbors < 0:
        raise ValueError("neighbors must be a nonnegative integer")
    view = GraphView(state)
    excluded = set(getattr(inducer, "excluded_ids", ()))
    active = {nid: node for nid, node in view.nodes.items()
              if nid not in excluded and node.validation_status not in INACTIVE_STATUSES}
    parents = {nid for nid, node in active.items()
               if node.origin_kind == "induced_intermediate" and view.children[nid]}
    coverage = {"strategy": "exact_name_or_definition_plus_nearest_neighbors", "neighbors": neighbors,
                "active_nodes": len(active), "induced_parents": len(parents), "exhaustive": False}
    if not parents or len(active) < 2:
        return [], coverage
    ids = sorted(active)
    ordered = {nid: active[nid] for nid in ids}
    vectors = np.asarray(inducer.vectors(ordered), dtype=float)
    if vectors.ndim != 2 or len(vectors) != len(ids) or not np.isfinite(vectors).all():
        raise ValueError("Invalid parent-deduplication embeddings")
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Invalid parent-deduplication embedding norms")
    unit = vectors / norms
    # Avoid spurious floating-point warnings from Accelerate-backed matmul.
    distances = 1 - np.einsum("ij,kj->ik", unit, unit, optimize=False)
    if not np.isfinite(distances).all():
        raise ValueError("Invalid parent-deduplication cosine distances")
    distances = np.clip(distances, 0, 2)
    positions = {nid: index for index, nid in enumerate(ids)}
    names = {nid: {norm(value) for value in [node.concept_label, *node.aliases]}
             for nid, node in active.items()}
    descendants = {nid: reachable(view.children, nid) for nid in ids}
    selected = {}
    for parent in sorted(parents):
        nearest = []
        for other in ids:
            if parent == other:
                continue
            pair = tuple(sorted((parent, other)))
            exact = bool(names[parent] & names[other]) or (
                bool(active[parent].concept_description) and
                norm(active[parent].concept_description) == norm(active[other].concept_description or ""))
            if exact:
                selected.setdefault(pair, set()).add("exact_name_or_definition")
            if other not in descendants[parent] and parent not in descendants[other]:
                nearest.append((float(distances[positions[parent], positions[other]]), other))
        for distance, other in sorted(nearest)[:neighbors]:
            selected.setdefault(tuple(sorted((parent, other))), set()).add("nearest_neighbor")
    rows = [{"pair_id": stable_id("parent-equivalence", *pair), "left": describe(active[pair[0]]),
             "right": describe(active[pair[1]]), "candidate_reasons": sorted(reasons)}
            for pair, reasons in sorted(selected.items())]
    coverage["candidate_pairs"] = len(rows)
    return rows, coverage


def _parent_proposal(state, parent_id, decision_ref):
    view = GraphView(state)
    parent = view.nodes[parent_id]
    children = sorted(view.children[parent_id])
    leaves = (reachable(view.children, parent_id) & set(state.frozen_input_concept_ids)) - set(children)
    return {"proposal_id": stable_id("dedup-merge-review", parent_id, parent.concept_label,
                                     parent.concept_description, children,
                                     [describe(view.nodes[nid]) for nid in sorted(leaves)]),
            "name": parent.concept_label, "definition": parent.concept_description,
            "reuse_id": parent_id, "children": [describe(view.nodes[nid]) for nid in children],
            "descendant_leaves": [describe(view.nodes[nid]) for nid in sorted(leaves)],
            "proposal_ref": decision_ref}


def _merge_candidate(state, canonical, removed, decision_ref, model):
    before = GraphView(state)
    frozen = set(state.frozen_input_concept_ids)
    if removed in frozen or removed in state.predefined_roots:
        raise CandidateError("A frozen input or predefined root cannot be removed")
    if (removed in reachable(before.children, canonical)
            or canonical in reachable(before.children, removed)):
        raise CandidateError("Equivalent ancestor candidates require separate structure repair")
    candidate = deepcopy(state)
    old_node = before.nodes[removed]
    candidate.nodes = [node for node in candidate.nodes if node.concept_id != removed]
    if canonical not in frozen:
        keep = next(node for node in candidate.nodes if node.concept_id == canonical)
        keep.aliases = sorted((set(keep.aliases) | {old_node.concept_label, *old_node.aliases}) - {keep.concept_label})
    candidate.concept_evidence_mapping = [row for row in candidate.concept_evidence_mapping
                                          if row["concept_id"] != removed]
    incident = [to_dict(link) for link in state.parent_child_edges
                if removed in (link.parent_id, link.child_id)]
    # Preserve unaffected edge provenance when a rewired edge has the same endpoints.
    links = {(link.parent_id, link.child_id): deepcopy(link) for link in state.parent_child_edges
             if removed not in (link.parent_id, link.child_id)}
    for link in state.parent_child_edges:
        if removed not in (link.parent_id, link.child_id):
            continue
        parent = canonical if link.parent_id == removed else link.parent_id
        child = canonical if link.child_id == removed else link.child_id
        if parent == child:
            continue
        links.setdefault((parent, child), edge(parent, child, "Equivalent parent consolidation pending review.",
                                               decision_ref, model))
    candidate.parent_child_edges = list(links.values())
    refresh_derived_support(candidate)
    validate_dag(candidate)
    view = GraphView(candidate)
    affected = {canonical} | reachable(view.parents, canonical)
    for nid in affected:
        if nid not in frozen and len(view.children[nid]) < 2:
            raise CandidateError("Merging would leave an induced ancestor with fewer than two children")
    return candidate, affected, {"removed_node": to_dict(old_node),
                                 "removed_evidence_mapping": deepcopy(next((row for row in state.concept_evidence_mapping
                                                                           if row["concept_id"] == removed), None)),
                                 "removed_edges": incident,
                                 "edge_replacements": [{"old_edge_id": link["edge_id"],
                                                        "parent_id": canonical if link["parent_id"] == removed else link["parent_id"],
                                                        "child_id": canonical if link["child_id"] == removed else link["child_id"]}
                                                       for link in incident]}


def deduplicate_parents(state, inducer, *, neighbors=5):
    """Merge equivalent parents only after validating the resulting relationships."""
    pairs, coverage = candidate_pairs(state, inducer, neighbors)
    report = {"candidate_coverage": coverage, "candidate_pairs": pairs,
              "equivalence_reviews": [], "merges": [], "holds": []}
    if not pairs:
        return state, report
    model = inducer.journal.settings["parent_equivalence"]["model"]
    checks = {}
    def review_batch(batch):
        # Candidate scores and retrieval reasons are deliberately not semantic evidence.
        payload = {"pairs": [{k: row[k] for k in ("pair_id", "left", "right")} for row in batch]}
        try:
            result, ref = checked_call(inducer.journal, "parent_equivalence", payload,
                                       lambda value: check_equivalence(value, {row["pair_id"] for row in batch}))
        except ContractError:
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            review_batch(batch[:middle])
            review_batch(batch[middle:])
            return
        for pair_id, check in result.items():
            checks[pair_id] = {**check, "review_ref": ref}
            report["equivalence_reviews"].append(checks[pair_id])
    for start in range(0, len(pairs), 6):
        review_batch(pairs[start:start + 6])
    current = state
    aliases = {}
    for pair in pairs:
        check = checks[pair["pair_id"]]
        if not check["equivalent"]:
            continue
        # A prior merge can alter the proposed identity. Do not infer transitive equivalence
        # from two separate model judgments; reconsider any new pair in the next round.
        left, right = pair["left"]["id"], pair["right"]["id"]
        if left in aliases or right in aliases:
            report["holds"].append({"pair_id": pair["pair_id"], "reason": "Endpoint merged earlier; re-evaluate the new pair in a later round."})
            continue
        frozen = set(current.frozen_input_concept_ids)
        canonical, removed = sorted((left, right), key=lambda nid: (nid not in frozen, nid))
        if removed in frozen:
            report["holds"].append({"pair_id": pair["pair_id"], "reason": "Two frozen inputs require leaf normalization, not parent merging."})
            continue
        try:
            candidate, affected, history = _merge_candidate(current, canonical, removed, check["review_ref"], model)
        except (CandidateError, ValueError) as error:
            report["holds"].append({"pair_id": pair["pair_id"], "reason": str(error)})
            continue
        proposals = [_parent_proposal(candidate, nid, check["review_ref"]) for nid in sorted(affected)]
        reviews = inducer.review(proposals)
        if not all(reviews.get(row["proposal_id"]) and
                   all(review["passes"] for review in reviews[row["proposal_id"]]) for row in proposals):
            report["holds"].append({"pair_id": pair["pair_id"], "reason": "Merged structure did not pass every affected parent review.",
                                    "reviews": reviews})
            continue
        cross = inducer.cross_review(proposals, GraphView(candidate))
        if not all(cross.get(row["proposal_id"], {}).get("verdict") == "pass" for row in proposals):
            report["holds"].append({"pair_id": pair["pair_id"], "reason": "Merged structure did not pass cross-parent review.",
                                    "reviews": reviews, "cross_review": cross})
            continue
        by_parent = {row["reuse_id"]: row for row in proposals}
        for link in candidate.parent_child_edges:
            if link.parent_id not in by_parent:
                continue
            proposal = by_parent[link.parent_id]
            review = reviews[proposal["proposal_id"]][0]
            checked_edge = next(row for row in review["edge_reviews"] if row["child_id"] == link.child_id)
            link.reason = checked_edge["reason"]
            link.decision_ref = review["review_ref"]
            link.validator = inducer.journal.settings["review"]["model"]
        candidate.version += 1
        assert_frozen_preserved(current, candidate)
        report["merges"].append({"pair_id": pair["pair_id"], "canonical_id": canonical,
                                 "removed_id": removed, "equivalence_ref": check["review_ref"],
                                 "reviews": reviews, "cross_review": cross, **history})
        aliases[removed] = canonical
        current = candidate
    return current, report
