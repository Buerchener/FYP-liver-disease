"""Maintenance must not turn a repair suggestion into an accepted relation."""

from copy import deepcopy
import unittest

from event_taxonomy.induction.contracts import CandidateError, ContractError
from event_taxonomy.models import Taxonomy, UniqueConcept, to_dict
from root_event_taxonomy.graph import GraphView, edge
from root_event_taxonomy.maintenance import (
    HELD_STATUS, assert_frozen_preserved, check_repair, maintain_graph, refresh_derived_support,
)


def fixture(*, ancestor=False, original_parent=False):
    leaves = [UniqueConcept(f"leaf:{letter}", name, definition, [], [], name, "fixture", [f"event:{letter}"],
                            [f"evidence:{letter}"], "event_type")
              for letter, name, definition in (
                  ("a", "Manual measurement", "Measure a quantity using a manual instrument."),
                  ("b", "Automated measurement", "Measure a quantity using an automated instrument."),
                  ("c", "Record deletion", "Delete an existing stored record."))]
    parent_id = "leaf:parent" if original_parent else "parent:measurement"
    parent = UniqueConcept(parent_id, "Measurement", "Measure a quantity.", [], [], "measurement", "fixture",
                           [], [], "event_type", origin_kind="input_concept" if original_parent else "induced_intermediate")
    nodes = [*leaves, parent]
    frozen = [node.concept_id for node in leaves] + ([parent_id] if original_parent else [])
    links = [{"event_id": f"event:{letter}", "concept_id": f"leaf:{letter}", "evidence_ids": [f"evidence:{letter}"]}
             for letter in "abc"]
    edges = [edge(parent_id, node.concept_id, "Fixture old decision", "fixture/old.json", "fixture") for node in leaves]
    if ancestor:
        nodes.append(UniqueConcept("parent:activity", "Activity", "Perform a data operation.", [], [], "activity",
                                   "fixture", [], [], "event_type", origin_kind="induced_intermediate"))
        edges.extend([edge("parent:activity", parent_id, "Fixture old decision", "fixture/old.json", "fixture"),
                      edge("parent:activity", "leaf:c", "Fixture old decision", "fixture/old.json", "fixture")])
    mappings = [{"concept_id": node.concept_id, "event_ids": node.supporting_event_ids[:],
                 "evidence_ids": node.evidence[:], "mapping_kind": "direct", "via_child_ids": []} for node in nodes]
    state = Taxonomy("fixture", nodes, edges, [], mappings, links, sorted(frozen), sorted(frozen), "partial")
    return refresh_derived_support(state), parent_id


class FakeJournal:
    def __init__(self, repair):
        self.settings = {"review": {"model": "fixture"}}
        self.last_ref = "fixture/repair.json"
        self.repair = repair
        self.calls = []

    def call(self, stage, payload):
        self.calls.append((stage, deepcopy(payload)))
        return deepcopy(self.repair)

    def mark_invalid(self, reason):
        pass


class FakeInducer:
    def __init__(self, *, reject_all=False, reject_repair=False, cross_fails=False, repair=None):
        self.reject_all = reject_all
        self.reject_repair = reject_repair
        self.cross_fails = cross_fails
        self.journal = FakeJournal(repair or {"action": "abstain", "name": None, "definition": None,
                                             "child_ids": [], "reason": "No supported repair."})
        self.reviewed = []
        self.crossed = []

    def review(self, proposals):
        decisions = {}
        for proposal in proposals:
            self.reviewed.append(deepcopy(proposal))
            repair = proposal["proposal_id"].startswith("maintenance-repair:")
            failing = self.reject_repair if repair else self.reject_all
            bad = {"leaf:c"} if proposal["name"] == "Measurement" else set()
            if failing:
                bad = {row["id"] for row in proposal["children"]}
            edges = [{"child_id": row["id"], "verdict": "fail" if row["id"] in bad else "pass",
                      "reason": "Fixture semantic decision"} for row in proposal["children"]]
            okay = not any(row["verdict"] == "fail" for row in edges)
            decisions[proposal["proposal_id"]] = [{
                "passes": okay, "verdict": "accept" if okay else "revise", "edge_reviews": edges,
                "leaf_reviews": [{"leaf_id": row["id"], "verdict": "pass"} for row in proposal["descendant_leaves"]],
                "review_ref": "fixture/review.json",
            }]
        return decisions

    def cross_review(self, proposals, view):
        self.crossed.extend(deepcopy(proposals))
        return {proposal["proposal_id"]: {"verdict": "fail" if self.cross_fails else "pass",
                                            "reason": "Fixture catalogue check", "review_ref": "fixture/cross.json"}
                for proposal in proposals}


class MaintenanceTests(unittest.TestCase):
    def test_invalid_child_is_pruned_only_after_final_subset_is_reviewed(self):
        state, parent = fixture()
        original = to_dict(state)
        inducer = FakeInducer()
        result, report = maintain_graph(state, inducer, max_repairs=0)
        self.assertEqual(GraphView(result).children[parent], {"leaf:a", "leaf:b"})
        self.assertEqual([row["edge"]["child_id"] for row in report["quarantined_edges"]], ["leaf:c"])
        self.assertEqual(len(inducer.reviewed), 2)
        self.assertEqual(len(inducer.crossed), 1)
        self.assertEqual(to_dict(state), original)
        self.assertEqual(result.instance_concept_links, state.instance_concept_links)
        assert_frozen_preserved(state, result)

    def test_rejected_parent_quarantine_propagates_to_unsupported_ancestor(self):
        state, parent = fixture(ancestor=True)
        result, report = maintain_graph(state, FakeInducer(reject_all=True), max_repairs=0)
        self.assertEqual(result.parent_child_edges, [])
        self.assertEqual(set(report["held_parent_ids"]), {parent, "parent:activity"})
        self.assertEqual(len(report["quarantined_edges"]), 5)
        self.assertTrue(all(node.validation_status == HELD_STATUS for node in result.nodes
                            if node.concept_id.startswith("parent:")))
        self.assertEqual(len(result.nodes), len(state.nodes))
        self.assertEqual(result.instance_concept_links, state.instance_concept_links)

    def test_cross_failure_does_not_apply_pruning(self):
        state, parent = fixture()
        result, report = maintain_graph(state, FakeInducer(cross_fails=True), max_repairs=0)
        self.assertEqual(result.parent_child_edges, [])
        self.assertEqual(report["repairs"][0]["status"], "cross_review_held")
        self.assertIn(parent, report["held_parent_ids"])

    def test_pruning_name_conflict_is_held_and_can_proceed_to_reviewed_repair(self):
        state, parent = fixture()
        # A legacy alias collision makes the otherwise passing pruning transaction invalid.
        GraphView(state).nodes["leaf:c"].aliases.append("Measurement")
        repair = {"action": "revise", "name": "Instrument measurement",
                  "definition": "Measure a quantity using an instrument.",
                  "child_ids": ["leaf:a", "leaf:b"], "reason": "Clarify the label and retain supported children"}
        inducer = FakeInducer(repair=repair)
        result, report = maintain_graph(state, inducer)
        self.assertEqual([row["status"] for row in report["repairs"]], ["structural_hold", "applied"])
        self.assertEqual(GraphView(result).children[parent], {"leaf:a", "leaf:b"})
        self.assertEqual(GraphView(result).nodes[parent].concept_label, "Instrument measurement")
        self.assertEqual(len(inducer.journal.calls), 1)
        self.assertEqual(len(inducer.crossed), 1)
        assert_frozen_preserved(state, result)

    def test_repair_must_pass_independent_review_before_it_can_change_definition(self):
        state, parent = fixture()
        repair = {"action": "revise", "name": "Data operation", "definition": "Perform a data operation.",
                  "child_ids": ["leaf:a", "leaf:b", "leaf:c"], "reason": "Proposed generalization"}
        inducer = FakeInducer(reject_all=True, reject_repair=True, repair=repair)
        result, report = maintain_graph(state, inducer)
        self.assertEqual(GraphView(result).nodes[parent].concept_description, "Measure a quantity.")
        self.assertEqual(result.parent_child_edges, [])
        self.assertEqual(len(inducer.journal.calls), 1)
        self.assertEqual(inducer.crossed, [])
        self.assertEqual(report["repairs"][0]["status"], "repair_review_held")

    def test_unchanged_repair_cannot_retry_a_failed_semantic_judgment(self):
        state, _ = fixture()
        repair = {"action": "revise", "name": "Measurement", "definition": "Measure a quantity.",
                  "child_ids": ["leaf:a", "leaf:b", "leaf:c"], "reason": "No actual modification"}
        inducer = FakeInducer(reject_all=True, repair=repair)
        result, report = maintain_graph(state, inducer)
        self.assertEqual(result.parent_child_edges, [])
        self.assertEqual(len(inducer.reviewed), 1)
        self.assertEqual(report["repairs"][0]["status"], "unchanged_repair")

    def test_approved_repair_rechecks_ancestor_using_changed_child_definition(self):
        state, parent = fixture(ancestor=True)
        repair = {"action": "revise", "name": "Data handling", "definition": "Perform an operation on data.",
                  "child_ids": ["leaf:a", "leaf:b"], "reason": "Fixture repair"}
        inducer = FakeInducer(reject_all=True, repair=repair)
        # The initial lower parent fails, its repair passes; the ancestor must see the revised child.
        original_review = inducer.review
        def review(proposals):
            if proposals[0]["reuse_id"] == "parent:activity":
                inducer.reject_all = False
            return original_review(proposals)
        inducer.review = review
        result, report = maintain_graph(state, inducer)
        self.assertEqual(GraphView(result).nodes[parent].concept_description, "Perform an operation on data.")
        ancestor_review = next(row for row in inducer.reviewed if row["reuse_id"] == "parent:activity")
        changed = next(row for row in ancestor_review["children"] if row["id"] == parent)
        self.assertEqual(changed["definition"], "Perform an operation on data.")
        self.assertEqual(report["repairs"][0]["status"], "applied")
        assert_frozen_preserved(state, result)

    def test_original_parent_is_never_renamed_or_removed(self):
        state, parent = fixture(original_parent=True)
        result, _ = maintain_graph(state, FakeInducer(reject_all=True), max_repairs=0)
        self.assertEqual(to_dict(GraphView(result).nodes[parent]), to_dict(GraphView(state).nodes[parent]))
        assert_frozen_preserved(state, result)
        proposal = {"name": "Original", "definition": "Fixed definition.", "children": [{"id": "leaf:a"}]}
        with self.assertRaises(ContractError):
            check_repair({"action": "revise", "name": "Changed", "definition": "Fixed definition.",
                          "child_ids": ["leaf:a"], "reason": "Unsupported rename"}, proposal, True)

    def test_protection_rejects_source_change_and_cycle(self):
        state, parent = fixture()
        changed = deepcopy(state)
        changed.nodes[0].concept_description = "Changed source"
        with self.assertRaises(CandidateError):
            assert_frozen_preserved(state, changed)
        changed = deepcopy(state)
        changed.parent_child_edges.append(edge("leaf:a", parent, "Bad cycle", "fixture", "fixture"))
        with self.assertRaises(ValueError):
            refresh_derived_support(changed)


if __name__ == "__main__":
    unittest.main()
