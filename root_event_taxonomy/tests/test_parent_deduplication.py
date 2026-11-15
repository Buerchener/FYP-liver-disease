"""Parent consolidation must be transactional and preserve original evidence."""

from copy import deepcopy
import unittest

import numpy as np

from event_taxonomy.induction.contracts import ContractError
from event_taxonomy.models import Taxonomy, UniqueConcept, to_dict
from event_taxonomy.taxonomy.dag import validate_dag
from root_event_taxonomy.graph import GraphView, edge
from root_event_taxonomy.maintenance import refresh_derived_support
from root_event_taxonomy.parent_deduplication import candidate_pairs, check_equivalence, deduplicate_parents
from root_event_taxonomy.upward_review import PARENT_CHECKS, check_parent_reviews


def fixture():
    leaves = [UniqueConcept(nid, "Input " + nid, "Original definition " + nid, [], [], nid,
                            "fixture", ["event:" + nid], ["evidence:" + nid], "event_type")
              for nid in "abcde"]
    parents = [UniqueConcept(nid, "Parent " + nid, "Parent definition " + nid, [], [], nid,
                             "fixture", [], [], "event_type", origin_kind="induced_intermediate")
               for nid in ("p1", "p2", "q1", "q2")]
    links = [("p1", "a"), ("p1", "b"), ("p2", "b"), ("p2", "c"),
             ("q1", "p1"), ("q1", "d"), ("q2", "p2"), ("q2", "e")]
    mappings = [{"concept_id": leaf.concept_id, "event_ids": leaf.supporting_event_ids,
                 "evidence_ids": leaf.evidence, "mapping_kind": "direct", "via_child_ids": []}
                for leaf in leaves]
    instances = [{"event_id": "event:" + leaf.concept_id, "concept_id": leaf.concept_id}
                 for leaf in leaves]
    state = Taxonomy("fixture", leaves + parents,
                     [edge(parent, child, "Original relation.", "fixture/original", "fixture")
                      for parent, child in links], [], mappings, instances, list("abcde"), list("abcde"), "partial")
    return refresh_derived_support(state)


class Journal:
    def __init__(self, equivalent=("p1", "p2"), uncertain=False):
        self.equivalent = set(equivalent)
        self.uncertain = uncertain
        self.calls = []
        self.last_ref = "llm/equivalence.json"
        self.settings = {stage: {"model": "fixture"} for stage in ("parent_equivalence", "review")}

    def call(self, stage, payload):
        assert stage == "parent_equivalence"
        self.calls.append(payload)
        checks = []
        for pair in payload["pairs"]:
            match = {pair[side]["id"] for side in ("left", "right")} == self.equivalent
            verdict = "uncertain" if match and self.uncertain else "true" if match else "false"
            direction = {"verdict": verdict, "reason": "Fixture equivalence decision.",
                         "counterexample": "Fixture counterexample." if verdict == "false" else None}
            checks.append({"pair_id": pair["pair_id"], "left_entails_right": deepcopy(direction),
                           "right_entails_left": deepcopy(direction)})
        return {"checks": checks}

    def mark_invalid(self, reason):
        pass


class SmallBatchJournal(Journal):
    def call(self, stage, payload):
        if len(payload["pairs"]) > 1:
            self.calls.append(payload)
            raise ContractError("Incomplete or non-stop LLM response: length")
        return super().call(stage, payload)


class Inducer:
    def __init__(self, *, equivalent=("p1", "p2"), uncertain=False, reject_parent=None, reject_cross=False):
        self.journal = Journal(equivalent, uncertain)
        self.reject_parent = reject_parent
        self.reject_cross = reject_cross
        self.reviewed_parents = []
        self.excluded_ids = set()

    def vectors(self, nodes):
        ids = list(nodes)
        return np.array([[1.0, 0.01 if nid in self.journal.equivalent else 1 + i]
                         for i, nid in enumerate(ids)])

    def review(self, proposals):
        self.reviewed_parents.extend(row["reuse_id"] for row in proposals)
        result = []
        for row in proposals:
            reject = row["reuse_id"] == self.reject_parent
            checks = {name: {"verdict": "pass", "reason": "Fixture quality."} for name in PARENT_CHECKS}
            if reject:
                checks["group_coherence"]["verdict"] = "fail"
            result.append({"proposal_id": row["proposal_id"], "parent_checks": checks,
                           "verdict": "revise" if reject else "accept", "definition_review": "Fixture definition.",
                           "suggested_action": "Fixture action.", "suggested_definition": None,
                           "edge_reviews": [{"child_id": child["id"], "child_entails_parent": "true",
                                             "broader_scope": "has_extra_cases", "counterexample": None,
                                             "reason": "Fixture reviewed relation.",
                                             "parent_only_example": "Fixture broader case.", "scope_reason": None}
                                            for child in row["children"]],
                           "leaf_reviews": [{"leaf_id": leaf["id"], "entails_parent": "true",
                                             "reason": "Fixture leaf inclusion.", "counterexample": None}
                                            for leaf in row["descendant_leaves"]]})
        checked = check_parent_reviews({"parent_reviews": result}, {row["proposal_id"]: row for row in proposals})
        return {pid: [{**review, "review_ref": "llm/review.json"}] for pid, review in checked.items()}

    def cross_review(self, proposals, view):
        return {row["proposal_id"]: {"verdict": "fail" if self.reject_cross else "pass",
                                      "reason": "Fixture cross check.", "review_ref": "llm/cross.json"}
                for row in proposals}


class ParentDeduplicationTest(unittest.TestCase):
    def test_boolean_directions_keep_their_truth_and_counterexample_requirements(self):
        def response(left, right, counterexample=None):
            return {"checks": [{"pair_id": "pair",
                                "left_entails_right": {"verdict": left, "reason": "Left inclusion.",
                                                       "counterexample": counterexample},
                                "right_entails_left": {"verdict": right, "reason": "Right inclusion.",
                                                       "counterexample": None}}]}
        accepted = check_equivalence(response(True, True), {"pair"})["pair"]
        self.assertTrue(accepted["equivalent"])
        self.assertEqual(accepted["left_entails_right"]["verdict"], "true")
        rejected = check_equivalence(response(False, True, "A valid counterexample."), {"pair"})["pair"]
        self.assertFalse(rejected["equivalent"])
        self.assertEqual(rejected["left_entails_right"]["verdict"], "false")
        for bad in (response(False, True), response(True, True, "Contradictory counterexample.")):
            with self.subTest(response=bad), self.assertRaises(ContractError):
                check_equivalence(bad, {"pair"})

    def test_numeric_equivalence_verdicts_are_not_booleans(self):
        for value in (0, 1, 0.0, 1.0):
            for side in ("left_entails_right", "right_entails_left"):
                result = {"pair_id": "pair",
                          "left_entails_right": {"verdict": "true", "reason": "Inclusion.", "counterexample": None},
                          "right_entails_left": {"verdict": "true", "reason": "Inclusion.", "counterexample": None}}
                result[side]["verdict"] = value
                with self.subTest(value=value, side=side), self.assertRaises(ContractError):
                    check_equivalence({"checks": [result]}, {"pair"})

    def test_finite_embeddings_with_overflowing_norms_are_rejected(self):
        inducer = Inducer()
        inducer.vectors = lambda nodes: np.full((len(nodes), 2), np.finfo(float).max)
        with self.assertRaisesRegex(ValueError, "embedding norms"):
            candidate_pairs(fixture(), inducer, neighbors=5)
        self.assertEqual(inducer.journal.calls, [])

    def test_merge_rewires_children_and_all_ancestors_preserving_sources(self):
        before = fixture()
        snapshot = to_dict(before)
        inducer = Inducer()
        after, report = deduplicate_parents(before, inducer, neighbors=8)
        self.assertEqual(to_dict(before), snapshot)
        view = GraphView(after)
        self.assertNotIn("p2", view.nodes)
        self.assertEqual(view.children["p1"], set("abc"))
        self.assertEqual(view.children["q2"], {"p1", "e"})
        self.assertEqual(set(inducer.reviewed_parents), {"p1", "q1", "q2"})
        self.assertEqual(after.instance_concept_links, before.instance_concept_links)
        for leaf in "abcde":
            self.assertEqual(to_dict(view.nodes[leaf]), next(row for row in snapshot["nodes"] if row["concept_id"] == leaf))
        self.assertEqual(len(report["merges"]), 1)
        self.assertEqual(report["merges"][0]["removed_node"]["concept_id"], "p2")
        self.assertEqual(len(report["merges"][0]["removed_edges"]), 3)
        self.assertFalse(report["candidate_coverage"]["exhaustive"])
        self.assertEqual(len(after.parent_child_edges), len({(e.parent_id, e.child_id) for e in after.parent_child_edges}))
        validate_dag(after)

    def test_uncertain_equivalence_does_not_rewrite(self):
        before = fixture()
        inducer = Inducer(uncertain=True)
        after, report = deduplicate_parents(before, inducer, neighbors=8)
        self.assertEqual(to_dict(after), to_dict(before))
        self.assertEqual(report["merges"], [])
        self.assertEqual(len(inducer.journal.calls), (len(report["candidate_pairs"]) + 5) // 6)

    def test_truncated_batches_split_without_retrying_semantic_decisions(self):
        before = fixture()
        inducer = Inducer()
        inducer.journal = SmallBatchJournal()
        after, report = deduplicate_parents(before, inducer, neighbors=8)
        self.assertNotIn("p2", GraphView(after).nodes)
        self.assertEqual(len(report["equivalence_reviews"]), len(report["candidate_pairs"]))
        singleton_calls = [call for call in inducer.journal.calls if len(call["pairs"]) == 1]
        self.assertEqual(len(singleton_calls), len(report["candidate_pairs"]))

    def test_failed_ancestor_or_cross_review_rolls_back(self):
        for parameters in ({"reject_parent": "q2"}, {"reject_cross": True}):
            with self.subTest(parameters=parameters):
                before = fixture()
                after, report = deduplicate_parents(before, Inducer(**parameters), neighbors=8)
                self.assertEqual(to_dict(after), to_dict(before))
                self.assertEqual(report["merges"], [])
                self.assertTrue(report["holds"])

    def test_equivalent_ancestor_is_held_instead_of_creating_cycle(self):
        before = fixture()
        # An exact lexical match enters candidates even though an ancestor pair is
        # excluded from nearest-neighbor retrieval.
        nodes = {node.concept_id: node for node in before.nodes}
        nodes["q1"].concept_description = nodes["p1"].concept_description
        after, report = deduplicate_parents(before, Inducer(equivalent=("q1", "p1")), neighbors=8)
        self.assertEqual(to_dict(after), to_dict(before))
        self.assertTrue(any("ancestor" in row["reason"] for row in report["holds"]))

    def test_frozen_canonical_and_its_original_mapping_are_unchanged(self):
        before = fixture()
        inducer = Inducer(equivalent=("p1", "e"))
        after, report = deduplicate_parents(before, inducer, neighbors=8)
        self.assertEqual(len(report["merges"]), 1)
        self.assertEqual(report["merges"][0]["canonical_id"], "e")
        self.assertNotIn("p1", GraphView(after).nodes)
        self.assertEqual(to_dict(GraphView(after).nodes["e"]), to_dict(GraphView(before).nodes["e"]))
        self.assertEqual(next(row for row in after.concept_evidence_mapping if row["concept_id"] == "e"),
                         next(row for row in before.concept_evidence_mapping if row["concept_id"] == "e"))

    def test_missing_direction_or_counterexample_is_invalid(self):
        for result in ({"pair_id": "pair", "left_entails_right": {}},
                       {"pair_id": "pair", "left_entails_right": {"verdict": "false", "reason": "Not contained.", "counterexample": None},
                        "right_entails_left": {"verdict": "true", "reason": "Contained.", "counterexample": None}}):
            with self.subTest(result=result), self.assertRaises(ContractError):
                check_equivalence({"checks": [result]}, {"pair"})


if __name__ == "__main__":
    unittest.main()
