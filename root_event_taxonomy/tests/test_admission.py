"""Admission must protect both concept scope and each original event link."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree

from event_taxonomy.induction.contracts import ContractError
from event_taxonomy.induction.journal import LLMJournal
from root_event_taxonomy.admission import (
    CONCEPT_CHECKS, INSTANCE_CHECKS, PROMPT_PATH, admission_inputs,
    check_admission_reviews, combined_verdict, review_admission,
)


def fixture():
    return {
        "leaf_concepts": [
            {"concept_id": "leaf:a", "concept_name": "Equipment inspection",
             "description": "Inspect the condition of equipment.",
             "source_concepts": ["source:a", "source:b"], "source_events": ["event:a", "event:b"]},
            {"concept_id": "leaf:b", "concept_name": "Equipment inspection",
             "description": "A proposal to inspect equipment in the future.",
             "source_concepts": ["source:c"], "source_events": ["event:c"]},
        ],
        "source_concepts": [
            {"source_concept_id": f"source:{letter}", "source_event": f"event:{letter}",
             "concept_name": "Equipment inspection", "description": "Original source type."}
            for letter in "abc"
        ],
        "source_events": [
            {"source_id": "event:a", "Event": "An inspector checked a pump.",
             "Evidence": "The inspector checked the pump condition.", "Trigger": "checked"},
            {"source_id": "event:b", "Event": "An inspector checked a valve.",
             "Evidence": "The valve condition was not recorded.", "Trigger": "recorded"},
            {"source_id": "event:c", "Event": "Inspection was proposed.",
             "Evidence": "The team proposed a future equipment inspection.", "Trigger": "proposed"},
        ],
        "unassigned_events": [{"source_event": "event:unassigned", "status": "no_input_concept"}],
    }


def decision(concept, *, concept_issue=None, event_issue=None):
    concept_checks = {key: {"verdict": "pass", "reason": "Fixture concept support."}
                      for key in CONCEPT_CHECKS}
    if concept_issue:
        key, verdict = concept_issue
        concept_checks[key] = {"verdict": verdict, "reason": "Fixture concept issue."}
    instances = []
    for event in concept["source_events"]:
        checks = {key: {"verdict": "pass", "reason": "Fixture event support."} for key in INSTANCE_CHECKS}
        if event_issue and event["event_id"] == event_issue[0]:
            checks[event_issue[1]] = {"verdict": event_issue[2], "reason": "Fixture event issue."}
        instances.append({"event_id": event["event_id"], **checks})
    all_verdicts = [check["verdict"] for check in concept_checks.values()]
    all_verdicts.extend(instance[key]["verdict"] for instance in instances for key in INSTANCE_CHECKS)
    return {"leaf_id": concept["leaf_id"], "concept_checks": concept_checks, "instance_reviews": instances,
            "verdict": combined_verdict(all_verdicts), "reason": "Fixture combined admission decision."}


class Provider:
    def __init__(self, issue=None):
        self.issue = issue
        self.calls = []

    def generate(self, stage, prompt, payload, parameters):
        self.calls.append((stage, deepcopy(payload)))
        value = {"reviews": [decision(concept, event_issue=self.issue) for concept in payload["concepts"]]}
        return {"content": json.dumps(value), "model": "fixture", "finish_reason": "stop"}


class AdmissionTest(unittest.TestCase):
    def test_truncated_batch_splits_and_resume_does_not_repeat_oversized_calls(self):
        class TruncatingProvider(Provider):
            def generate(self, stage, prompt, payload, parameters):
                response = super().generate(stage, prompt, payload, parameters)
                if len(payload["concepts"]) > 1:
                    response["finish_reason"] = "length"
                return response

        provider = TruncatingProvider()
        with TemporaryDirectory() as temp:
            journal = LLMJournal(Path(temp) / "llm", provider, {"admission": PROMPT_PATH.read_text()},
                                 {"admission": {"model": "fixture"}}, 10)
            result = review_admission(fixture(), journal)
            self.assertEqual([len(payload["concepts"]) for _, payload in provider.calls], [2, 2, 1, 1])
            resumed = review_admission(fixture(), journal)
        self.assertEqual(result, resumed)
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(result["accepted_leaf_ids"], ["leaf:a", "leaf:b"])

    def test_single_item_truncation_still_fails_and_semantic_failure_is_not_retried(self):
        class TruncatingProvider(Provider):
            def generate(self, stage, prompt, payload, parameters):
                response = super().generate(stage, prompt, payload, parameters)
                response["finish_reason"] = "length"
                return response

        with TemporaryDirectory() as temp:
            provider = TruncatingProvider()
            journal = LLMJournal(Path(temp) / "truncated", provider, {"admission": PROMPT_PATH.read_text()},
                                 {"admission": {"model": "fixture"}}, 10)
            with self.assertRaisesRegex(ContractError, "length"):
                review_admission(fixture(), journal, batch_size=1)
            self.assertEqual(len(provider.calls), 2)
            provider = Provider(("event:b", "instance_of", "fail"))
            journal = LLMJournal(Path(temp) / "semantic", provider, {"admission": PROMPT_PATH.read_text()},
                                 {"admission": {"model": "fixture"}}, 10)
            result = review_admission(fixture(), journal)
            self.assertEqual(result["held_leaf_ids"], ["leaf:a"])
            self.assertEqual(len(provider.calls), 1)

    def test_mixed_support_holds_whole_leaf_and_preserves_all_source_records(self):
        dataset = fixture()
        before = deepcopy(dataset)
        provider = Provider(("event:b", "evidence_support", "fail"))
        with TemporaryDirectory() as temp:
            journal = LLMJournal(Path(temp) / "llm", provider, {"admission": PROMPT_PATH.read_text()},
                                 {"admission": {"model": "fixture"}}, 10)
            result = review_admission(dataset, journal, batch_size=1)
            resumed = review_admission(dataset, journal, batch_size=1)
        self.assertEqual(dataset, before)
        self.assertEqual(result, resumed)
        self.assertEqual(result["held_leaf_ids"], ["leaf:a"])
        self.assertEqual(result["accepted_leaf_ids"], ["leaf:b"])
        self.assertEqual(len(result["reviews"][0]["instance_reviews"]), 2)
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(all(row["review_ref"].startswith("llm/") for row in result["reviews"]))

    def test_uncertainty_and_concept_failure_cannot_be_overridden_by_pass(self):
        concept = admission_inputs(fixture())[0]
        expected = {concept["leaf_id"]: concept}
        for issue in (("definition_clarity", "uncertain"), ("within_scope", "fail")):
            with self.subTest(issue=issue):
                row = decision(concept, concept_issue=issue)
                check_admission_reviews({"reviews": [row]}, expected)
                row["verdict"] = "pass"
                with self.assertRaisesRegex(ContractError, "every concept and source-event"):
                    check_admission_reviews({"reviews": [row]}, expected)
        row = decision(concept, event_issue=("event:b", "instance_of", "uncertain"))
        self.assertEqual(row["verdict"], "uncertain")
        row["verdict"] = "pass"
        with self.assertRaises(ContractError):
            check_admission_reviews({"reviews": [row]}, expected)

    def test_missing_duplicate_or_unknown_instances_are_rejected(self):
        concept = admission_inputs(fixture())[0]
        expected = {concept["leaf_id"]: concept}
        for modification in ("missing", "duplicate", "unknown"):
            with self.subTest(modification=modification):
                row = decision(concept)
                if modification == "missing":
                    row["instance_reviews"].pop()
                elif modification == "duplicate":
                    row["instance_reviews"].append(deepcopy(row["instance_reviews"][0]))
                else:
                    row["instance_reviews"][0]["event_id"] = "event:foreign"
                with self.assertRaises(ContractError):
                    check_admission_reviews({"reviews": [row]}, expected)
        with self.assertRaises(ContractError):
            check_admission_reviews({"reviews": []}, expected)

    def test_scope_policy_is_explicit_and_same_names_keep_their_own_definitions(self):
        provider = Provider()
        with TemporaryDirectory() as temp:
            journal = LLMJournal(Path(temp) / "llm", provider, {"admission": PROMPT_PATH.read_text()},
                                 {"admission": {"model": "fixture"}}, 10)
            result = review_admission(fixture(), journal, scope_policy="research_statements")
            with self.assertRaises(ValueError):
                review_admission(fixture(), journal, scope_policy="anything")
        payload = provider.calls[0][1]
        self.assertEqual(result["policy"], "research_statements")
        self.assertEqual(payload["scope_policy"], "research_statements")
        self.assertEqual(len(payload["concepts"]), 2)
        self.assertEqual(payload["concepts"][0]["name"], payload["concepts"][1]["name"])
        self.assertNotEqual(payload["concepts"][0]["definition"], payload["concepts"][1]["definition"])

    def test_prompt_schema_matches_contract(self):
        root = ElementTree.parse(PROMPT_PATH).getroot()
        output = root.find("output").text
        example = json.loads(output[output.index("{"):])
        row = example["reviews"][0]
        self.assertEqual(set(row["concept_checks"]), set(CONCEPT_CHECKS))
        self.assertEqual(set(row["instance_reviews"][0]) - {"event_id"}, set(INSTANCE_CHECKS))


if __name__ == "__main__":
    unittest.main()
