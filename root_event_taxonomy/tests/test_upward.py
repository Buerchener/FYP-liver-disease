"""Checks that only reviewed parents become inputs to the next round."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from event_taxonomy.induction.contracts import ContractError, ProviderError
from event_taxonomy.induction.journal import LLMJournal
from event_taxonomy.induction.storage import read_json
from event_taxonomy.models import to_dict
from root_event_taxonomy.upward import UpwardInducer, initial_taxonomy
from root_event_taxonomy.upward_review import PROMPTS, check_parent_reviews, checked_call


class FixtureProvider:
    def __init__(self, *, fail_descendant=False, fail_cross=False, reject_requests=False):
        self.fail_descendant = fail_descendant
        self.fail_cross = fail_cross
        self.reject_requests = reject_requests
        self.calls = []

    def generate(self, stage, prompt, payload, parameters):
        self.calls.append((stage, payload))
        if self.reject_requests:
            raise ProviderError("fixture provider unavailable")
        if stage == "proposal":
            value = {"groups": [self.propose(group) for group in payload["groups"]]}
        elif stage == "review":
            value = {"parent_reviews": [self.review(row) for row in payload["proposals"]]}
        else:
            value = {"checks": [{
                "proposal_id": proposal_id, "verdict": "fail" if self.fail_cross else "pass",
                "related_id": None, "reason": "Fixture cross-group decision.",
            } for proposal_id in payload["target_ids"]]}
        return {"content": json.dumps(value), "model": "fixture", "finish_reason": "stop"}

    @staticmethod
    def propose(group):
        members = {row["name"]: row["id"] for row in group["members"]}
        pair = {
            frozenset({"MRI quality grading", "Ultrasound quality grading"}):
                ("Image quality grading", "Assign a grade to the quality of an image."),
            frozenset({"Image quality grading", "Image noise quantification"}):
                ("Image quality assessment", "Evaluate the quality of an image."),
        }.get(frozenset(members))
        proposals = [{"name": pair[0], "definition": pair[1], "reuse_id": None,
                      "child_ids": list(members.values())}] if pair else []
        return {"group_id": group["group_id"], "reason": "Fixture proposal.", "proposals": proposals}

    def review(self, proposal):
        descendant_failed = self.fail_descendant and bool(proposal["descendant_leaves"])
        return {
            "proposal_id": proposal["proposal_id"], "verdict": "revise" if descendant_failed else "accept",
            "definition_review": "Fixture parent decision.", "suggested_action": "Keep the parent.",
            "suggested_definition": None,
            "edge_reviews": [{
                "child_id": child["id"], "child_entails_parent": "true", "broader_scope": "has_extra_cases",
                "parent_only_example": "A fixture example outside the child.", "scope_reason": None,
                "reason": "Fixture direct containment.", "counterexample": None,
            } for child in proposal["children"]],
            "leaf_reviews": [{
                "leaf_id": leaf["id"], "entails_parent": "false" if self.fail_descendant else "true",
                "reason": "Fixture descendant containment.",
                "counterexample": "A fixture leaf outside the parent." if descendant_failed else None,
            } for leaf in proposal["descendant_leaves"]],
        }


def fixture():
    names = ("MRI quality grading", "Ultrasound quality grading", "Image noise quantification", "Study recruitment")
    return {
        "leaf_concepts": [{
            "concept_id": f"leaf:{index}", "concept_name": name, "description": name + " definition.",
            "source_concepts": [f"concept:{index}"], "source_events": [f"event:{index}"],
        } for index, name in enumerate(names)],
        "source_concepts": [{"source_concept_id": f"concept:{index}", "source_event": f"event:{index}"}
                            for index in range(len(names))],
        "source_events": [{"source_id": f"event:{index}", "Evidence": name + " evidence."}
                          for index, name in enumerate(names)],
    }


def vectors(nodes):
    angles = {
        "MRI quality grading": 0, "Ultrasound quality grading": .1, "Image noise quantification": 1.3,
        "Study recruitment": 3.1, "Image quality grading": .6, "Image quality assessment": .7,
    }
    return np.array([[np.cos(angles[row.concept_label]), np.sin(angles[row.concept_label])]
                     for row in nodes.values()])


class UpwardTest(unittest.TestCase):
    def test_claimed_broader_scope_requires_a_witness(self):
        parent = {"proposal_id": "parent", "children": [{"id": "child"}], "descendant_leaves": []}
        reply = FixtureProvider().review(parent)
        edge = reply["edge_reviews"][0]
        edge["parent_only_example"] = None
        with self.assertRaisesRegex(ContractError, "parent-only example"):
            check_parent_reviews({"parent_reviews": [reply]}, {"parent": parent})
        edge["broader_scope"] = "no_extra_cases"
        edge["scope_reason"] = "The parent and child have identical scope."
        with self.assertRaisesRegex(ContractError, "must pass every child"):
            check_parent_reviews({"parent_reviews": [reply]}, {"parent": parent})
        del edge["verdict"]
        reply["verdict"] = "revise"
        reviewed = check_parent_reviews({"parent_reviews": [reply]}, {"parent": parent})["parent"]
        self.assertFalse(reviewed["passes"])
        self.assertEqual(reviewed["edge_reviews"][0]["verdict"], "fail")

    def run_fixture(self, directory, provider, *, rounds=4, budget=100, result=None):
        taxonomy, _ = initial_taxonomy(fixture())
        journal = LLMJournal(directory / "llm", provider, PROMPTS, dict.fromkeys(PROMPTS, {"model": "fixture"}), budget)
        inducer = UpwardInducer(journal, vectors, (.2, .4, .55), 8, rounds)
        start = {"taxonomy": to_dict(taxonomy), "rounds": []} if result is None else result
        return inducer.run(directory / "hierarchy.json", start, .2, None)

    def test_descendant_rejection_preserves_first_layer_and_unassigned_leaves(self):
        provider = FixtureProvider(fail_descendant=True)
        with TemporaryDirectory() as temp:
            result = self.run_fixture(Path(temp), provider)
        first, second = result["rounds"]
        parent = first["accepted"][0]["parent_id"]
        self.assertIn(parent, second["input_ids"])
        self.assertIn("leaf:3", second["next_input_ids"])
        self.assertIn("leaf:2", second["unassigned_input_ids"])
        self.assertEqual({row["id"] for row in second["proposals"][0]["descendant_leaves"]}, {"leaf:0", "leaf:1"})
        self.assertEqual(second["accepted"], [])
        self.assertEqual(result["stop_reason"], "no_approved_new_parent")
        self.assertEqual(len(result["taxonomy"]["parent_child_edges"]), 2)
        self.assertEqual(len(result["taxonomy"]["instance_concept_links"]), 4)
        proposal_calls = [payload for stage, payload in provider.calls if stage == "proposal"]
        existing_parent = next(member for group in proposal_calls[-1]["groups"] for member in group["members"]
                               if member["direct_children"])
        self.assertNotEqual(existing_parent["id"], parent)
        self.assertTrue(all("id" not in child for child in existing_parent["direct_children"]))
        self.assertIn(parent, {child["id"] for child in second["proposals"][0]["children"]})

    def test_round_limit_records_two_layers_and_evidence(self):
        with TemporaryDirectory() as temp:
            result = self.run_fixture(Path(temp), FixtureProvider(), rounds=2)
        self.assertEqual(result["stop_reason"], "maximum_rounds")
        parent_id = result["rounds"][1]["accepted"][0]["parent_id"]
        parent = next(row for row in result["taxonomy"]["nodes"] if row["concept_id"] == parent_id)
        self.assertEqual(parent["supporting_event_ids"], ["event:0", "event:1", "event:2"])
        self.assertEqual(len(parent["evidence"]), 3)
        self.assertEqual(len(result["taxonomy"]["parent_child_edges"]), 4)

    def test_cross_group_failure_prevents_insertion(self):
        with TemporaryDirectory() as temp:
            result = self.run_fixture(Path(temp), FixtureProvider(fail_cross=True))
        self.assertEqual(result["stop_reason"], "no_approved_new_parent")
        self.assertEqual(result["rounds"][0]["accepted"], [])
        self.assertEqual(result["taxonomy"]["parent_child_edges"], [])

    def test_budget_pause_reuses_saved_responses(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            provider = FixtureProvider()
            result = self.run_fixture(directory, provider, budget=1)
            self.assertEqual(result["stop_reason"], "maximum_llm_calls")
            self.assertEqual(result["rounds"], [])
            provider.calls.clear()
            result = self.run_fixture(directory, provider, result=read_json(directory / "hierarchy.json"))
        self.assertEqual(provider.calls[0][0], "review")
        self.assertEqual(result["status"], "finished")
        self.assertEqual(len(result["rounds"][1]["accepted"]), 1)

    def test_invalid_retry_is_logged_and_can_be_retried(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            provider = FixtureProvider()
            journal = LLMJournal(directory, provider, PROMPTS, dict.fromkeys(PROMPTS, {"model": "fixture"}), 5)
            payload = {"target_ids": ["one"]}

            def reject(_value):
                raise ContractError("fixture missing field")

            with self.assertRaises(ContractError):
                checked_call(journal, "cross_review", payload, reject)
            records = sorted(directory.glob("*.json"))
            self.assertEqual([read_json(path)["status"] for path in records], ["response_invalid"] * 2)
            checked_call(journal, "cross_review", payload, lambda value: value)
        self.assertEqual(len(provider.calls), 3)


if __name__ == "__main__":
    unittest.main()
