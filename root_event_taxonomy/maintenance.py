"""Recheck and repair existing taxonomy relations without changing source concepts."""

from copy import deepcopy

from event_taxonomy.induction.contracts import CandidateError, ids, obj, require, text
from event_taxonomy.induction.graph import norm, structure_signature
from event_taxonomy.models import stable_id, to_dict
from event_taxonomy.taxonomy.dag import reachable, validate_dag

from .graph import GraphView, edge
from .upward_review import checked_call, describe


HELD_STATUS = "maintenance_held"


def assert_frozen_preserved(before, after):
    """Allow auditable derived-graph edits while protecting original input records."""
    validate_dag(after)
    old = {node.concept_id: to_dict(node) for node in before.nodes}
    new = {node.concept_id: to_dict(node) for node in after.nodes}
    frozen = set(before.frozen_input_concept_ids)
    if any(new.get(node_id) != old[node_id] for node_id in frozen):
        raise CandidateError("Maintenance cannot change frozen input concepts or their evidence")
    for field in ("frozen_input_concept_ids", "predefined_roots", "instance_concept_links"):
        if getattr(before, field) != getattr(after, field):
            raise CandidateError("Maintenance cannot change source instance links or input/root sets")
    old_mappings = {row["concept_id"]: row for row in before.concept_evidence_mapping}
    new_mappings = {row["concept_id"]: row for row in after.concept_evidence_mapping}
    if len(new_mappings) != len(after.concept_evidence_mapping):
        raise CandidateError("Maintenance produced duplicate evidence mappings")
    if any(new_mappings.get(node_id) != old_mappings.get(node_id) for node_id in frozen):
        raise CandidateError("Maintenance cannot change frozen source evidence mappings")


def refresh_derived_support(state):
    """Refresh mutable derived support in place; return the same state."""
    view = GraphView(state)
    frozen = set(state.frozen_input_concept_ids)
    mappings = {row["concept_id"]: row for row in state.concept_evidence_mapping}
    for node_id, node in view.nodes.items():
        if node_id in frozen:
            continue
        descendants = reachable(view.children, node_id) & frozen
        events = sorted({event_id for child_id in descendants
                         for event_id in view.nodes[child_id].supporting_event_ids})
        evidence = sorted({evidence_id for child_id in descendants
                           for evidence_id in view.nodes[child_id].evidence})
        node.supporting_event_ids = events
        node.evidence = evidence
        mapping = mappings.get(node_id)
        if mapping is None:
            mapping = {"concept_id": node_id}
            state.concept_evidence_mapping.append(mapping)
        mapping.update(event_ids=events, evidence_ids=evidence, mapping_kind="derived_from_descendants",
                       via_child_ids=sorted(view.children[node_id]))
    state.unplaced_concept_ids = sorted(frozen - view.rooted)
    state.status = "partial"
    validate_dag(state)
    return state


def parent_proposal(state, parent_id, child_ids=None, *, name=None, definition=None, kind="recheck"):
    view = GraphView(state)
    parent = view.nodes[parent_id]
    children = sorted(view.children[parent_id] if child_ids is None else child_ids)
    frozen = set(state.frozen_input_concept_ids)
    descendants = set().union(*(reachable(view.children, child_id) | {child_id} for child_id in children))
    leaf_ids = (descendants & frozen) - set(children)
    name = parent.concept_label if name is None else name
    definition = parent.concept_description if definition is None else definition
    return {
        "proposal_id": stable_id("maintenance-" + kind, parent_id, name, definition, children),
        "name": name, "definition": definition, "reuse_id": parent_id,
        "children": [describe(view.nodes[child_id]) for child_id in children],
        "descendant_leaves": [describe(view.nodes[leaf_id]) for leaf_id in sorted(leaf_ids)],
    }


def check_repair(response, proposal, frozen):
    obj(response, ("action", "name", "definition", "child_ids", "reason"))
    require(response["action"] in ("revise", "abstain"), "Unknown repair action")
    text(response["reason"])
    if response["action"] == "abstain":
        require(response["name"] is None and response["definition"] is None and response["child_ids"] == [],
                "An abstention must not contain a repair")
        return response
    text(response["name"])
    text(response["definition"])
    ids(response["child_ids"], {row["id"] for row in proposal["children"]}, minimum=1 if frozen else 2)
    if frozen:
        require((response["name"], response["definition"]) == (proposal["name"], proposal["definition"]),
                "Repair cannot rename or redefine an original input concept")
    return response


def _passes(reviews, proposal):
    rows = reviews.get(proposal["proposal_id"], [])
    return bool(rows) and all(row["passes"] for row in rows)


def _passing_children(state, proposal, reviews):
    view = GraphView(state)
    good = {row["id"] for row in proposal["children"]}
    for review in reviews[proposal["proposal_id"]]:
        good -= {row["child_id"] for row in review["edge_reviews"] if row["verdict"] != "pass"}
        failed_leaves = {row["leaf_id"] for row in review["leaf_reviews"] if row["verdict"] != "pass"}
        good = {child_id for child_id in good
                if not (reachable(view.children, child_id) | {child_id}) & failed_leaves}
    return sorted(good)


def _checked_revision(state, inducer, proposal, reviews, *, repair_ref=None):
    """Validate a candidate transaction; never expose a partially repaired graph."""
    if not _passes(reviews, proposal):
        raise CandidateError("Revised parent did not pass every independent review")
    parent_id = proposal["reuse_id"]
    frozen = set(state.frozen_input_concept_ids)
    candidate = deepcopy(state)
    view = GraphView(candidate)
    node = view.nodes[parent_id]
    old_node = to_dict(node)
    if parent_id not in frozen:
        for other_id, other in view.nodes.items():
            if other_id == parent_id:
                continue
            if norm(proposal["name"]) in {norm(value) for value in [other.concept_label, *other.aliases]}:
                raise CandidateError("Repaired parent duplicates another concept name")
            if norm(proposal["definition"]) == norm(other.concept_description or ""):
                raise CandidateError("Repaired parent duplicates another concept definition")
        node.concept_label = proposal["name"]
        node.concept_description = proposal["definition"]
        node.normalization_key = norm(node.concept_label)
        node.normalization_version = "upward-maintenance/1"
    elif (node.concept_label, node.concept_description) != (proposal["name"], proposal["definition"]):
        raise CandidateError("Repair cannot modify an original input definition")
    old_edges = [to_dict(link) for link in candidate.parent_child_edges if link.parent_id == parent_id]
    candidate.parent_child_edges = [link for link in candidate.parent_child_edges if link.parent_id != parent_id]
    review = reviews[proposal["proposal_id"]][0]
    model = inducer.journal.settings["review"]["model"]
    candidate.parent_child_edges.extend(edge(parent_id, row["child_id"], row["reason"], review["review_ref"], model)
                                        for row in review["edge_reviews"])
    refresh_derived_support(candidate)
    cross = inducer.cross_review([proposal], GraphView(candidate))
    check = cross.get(proposal["proposal_id"])
    if check is None or check["verdict"] != "pass":
        return None, {"status": "cross_review_held", "proposal": proposal, "reviews": reviews, "cross_review": cross}
    if parent_id not in frozen:
        node.validation_status = "llm_independently_reviewed"
        node.description_provenance = repair_ref or review["review_ref"]
        node.review_history.append({"kind": "maintenance_revision", "previous_name": old_node["concept_label"],
                                    "previous_definition": old_node["concept_description"],
                                    "decision_ref": node.description_provenance,
                                    "review_refs": [row["review_ref"] for row in reviews[proposal["proposal_id"]]],
                                    "cross_review_ref": check.get("review_ref")})
    candidate.version += 1
    assert_frozen_preserved(state, candidate)
    return candidate, {"status": "applied", "parent_id": parent_id, "before_node": old_node,
                       "replaced_edges": old_edges, "proposal": proposal, "reviews": reviews,
                       "cross_review": cross, "repair_ref": repair_ref}


def maintain_graph(state, inducer, *, max_repairs=1):
    """Recheck bottom-up; prune or revise once, otherwise quarantine invalid relations.

    max_repairs bounds proposal attempts per failing parent, not retries until acceptance.
    The journal caches identical reviews, including failed decisions. All edits occur on a copy.
    """
    if max_repairs not in (0, 1):
        raise ValueError("max_repairs must be 0 or 1 per parent")
    validate_dag(state)
    working = deepcopy(state)
    frozen = set(state.frozen_input_concept_ids)
    start = GraphView(state)
    parents = sorted((node_id for node_id in start.nodes if start.children[node_id]),
                     key=lambda node_id: (len(reachable(start.children, node_id)), node_id))
    report = {"reviewed_parent_ids": [], "reviews": {}, "repairs": [], "quarantined_edges": [],
              "held_parent_ids": [], "node_snapshots": [], "before_edge_count": len(state.parent_child_edges)}
    snapshotted = set()
    initial_mappings = {row["concept_id"]: row for row in state.concept_evidence_mapping}

    def snapshot(parent_id):
        if parent_id in snapshotted:
            return
        node = start.nodes[parent_id]
        mapping = initial_mappings.get(parent_id)
        report["node_snapshots"].append({"before_node": to_dict(node), "before_mapping": deepcopy(mapping)})
        snapshotted.add(parent_id)

    def quarantine(parent_id, reason):
        nonlocal working
        snapshot(parent_id)
        generated = parent_id not in frozen
        removed = [link for link in working.parent_child_edges
                   if link.parent_id == parent_id or (generated and link.child_id == parent_id)]
        removed_ids = {link.edge_id for link in removed}
        report["quarantined_edges"].extend({"edge": to_dict(link), "reason": reason,
                                             "reviewed_parent_id": parent_id} for link in removed)
        working.parent_child_edges = [link for link in working.parent_child_edges if link.edge_id not in removed_ids]
        if generated:
            node = GraphView(working).nodes[parent_id]
            node.validation_status = HELD_STATUS
            node.review_history.append({"kind": "maintenance_hold", "reason": reason,
                                        "removed_edge_ids": sorted(removed_ids)})
            report["held_parent_ids"].append(parent_id)
        working.version += 1
        refresh_derived_support(working)

    excluded = set(getattr(inducer, "excluded_ids", ()))
    rejected_edges = [link for link in working.parent_child_edges
                      if link.parent_id in excluded or link.child_id in excluded]
    for link in rejected_edges:
        report["quarantined_edges"].append({"edge": to_dict(link), "reason": "Concept admission did not pass",
                                            "reviewed_parent_id": link.parent_id})
    if rejected_edges:
        for parent_id in {link.parent_id for link in rejected_edges} - frozen:
            snapshot(parent_id)
        removed_ids = {link.edge_id for link in rejected_edges}
        working.parent_child_edges = [link for link in working.parent_child_edges if link.edge_id not in removed_ids]
        working.version += 1
        refresh_derived_support(working)

    for parent_id in parents:
        view = GraphView(working)
        minimum = 1 if parent_id in frozen else 2
        if len(view.children[parent_id]) < minimum:
            if parent_id not in frozen:
                quarantine(parent_id, "Generated parent has fewer than two supported direct children")
            continue
        proposal = parent_proposal(working, parent_id)
        reviews = inducer.review([proposal])
        report["reviewed_parent_ids"].append(parent_id)
        report["reviews"][parent_id] = reviews
        if _passes(reviews, proposal):
            continue
        snapshot(parent_id)
        passing = _passing_children(working, proposal, reviews)
        if minimum <= len(passing) < len(proposal["children"]):
            pruned = parent_proposal(working, parent_id, passing, kind="prune")
            pruning_reviews = inducer.review([pruned])
            if _passes(pruning_reviews, pruned):
                try:
                    candidate, record = _checked_revision(working, inducer, pruned, pruning_reviews)
                except (ValueError, CandidateError) as error:
                    candidate, record = None, {"status": "structural_hold", "proposal": pruned,
                                               "reason": str(error), "reviews": pruning_reviews}
                report["repairs"].append(record)
                if candidate is not None:
                    kept = set(passing)
                    report["quarantined_edges"].extend({"edge": to_dict(link), "reason": "Excluded after re-audited pruning",
                                                        "reviewed_parent_id": parent_id}
                                                       for link in working.parent_child_edges
                                                       if link.parent_id == parent_id and link.child_id not in kept)
                    working = candidate
                    continue
            else:
                report["repairs"].append({"status": "pruning_review_held", "proposal": pruned, "reviews": pruning_reviews})
        if max_repairs:
            payload = {"parent": proposal, "definition_is_frozen": parent_id in frozen,
                       "review_findings": reviews[proposal["proposal_id"]]}
            repair, ref = checked_call(inducer.journal, "repair", payload,
                                       lambda result: check_repair(result, proposal, parent_id in frozen))
            if repair["action"] == "revise":
                if (repair["name"], repair["definition"], set(repair["child_ids"])) == (
                        proposal["name"], proposal["definition"], {row["id"] for row in proposal["children"]}):
                    report["repairs"].append({"status": "unchanged_repair", "parent_id": parent_id,
                                               "reason": repair["reason"], "repair_ref": ref})
                    quarantine(parent_id, "Repair made no change; failed review must not be retried for a different verdict")
                    continue
                revision = parent_proposal(working, parent_id, repair["child_ids"],
                                           name=repair["name"], definition=repair["definition"], kind="repair")
                revised_reviews = inducer.review([revision])
                if _passes(revised_reviews, revision):
                    try:
                        candidate, record = _checked_revision(working, inducer, revision, revised_reviews, repair_ref=ref)
                    except (ValueError, CandidateError) as error:
                        candidate, record = None, {"status": "structural_hold", "proposal": revision, "reason": str(error),
                                                   "reviews": revised_reviews, "repair_ref": ref}
                    report["repairs"].append(record)
                    if candidate is not None:
                        kept = set(repair["child_ids"])
                        report["quarantined_edges"].extend({"edge": to_dict(link), "reason": "Excluded by independently reviewed repair",
                                                            "reviewed_parent_id": parent_id}
                                                           for link in working.parent_child_edges
                                                           if link.parent_id == parent_id and link.child_id not in kept)
                        working = candidate
                        continue
                else:
                    report["repairs"].append({"status": "repair_review_held", "proposal": revision,
                                               "reviews": revised_reviews, "repair_ref": ref})
            else:
                report["repairs"].append({"status": "abstained", "parent_id": parent_id,
                                           "reason": repair["reason"], "repair_ref": ref})
        quarantine(parent_id, "Existing parent or relations did not pass independent recheck and bounded repair")

    refresh_derived_support(working)
    assert_frozen_preserved(state, working)
    for node in working.nodes:
        if node.concept_id not in frozen and to_dict(node) != to_dict(start.nodes[node.concept_id]):
            snapshot(node.concept_id)
    held = {node.concept_id for node in working.nodes if node.validation_status == HELD_STATUS}
    if any(link.parent_id in held or link.child_id in held for link in working.parent_child_edges):
        raise CandidateError("A quarantined parent remains connected to the active graph")
    report["changed"] = structure_signature(state) != structure_signature(working)
    report["after_edge_count"] = len(working.parent_child_edges)
    return working, report
