import unittest

from cognitive_agent.evidence_contrast import find_evidence_conflict
from cognitive_agent.evidence_pack import EvidencePackBuilder
from cognitive_agent.evidence_scope import compare_scope, extract_scope_witnesses
from cognitive_agent.schema.schema_profile import schema_profile_from_dict


PROFILE = schema_profile_from_dict({
    "artifact_type": "dataset_schema_profile",
    "name": "ScopeFixture",
    "contract_version": "v1",
    "semantic_target": "RELATION_TRUTH",
    "entity_types": ["Gene", "Disease"],
    "predicates": {
        "ASSOCIATED_WITH": {
            "allowed_signatures": [["Gene", "Disease"]],
            "explicit_patterns": [r"\bassociat(?:e|es|ed|ing)\s+with\b"],
        },
    },
})


RELATION = {
    "subject": "TP53", "subject_type": "Gene",
    "predicate": "ASSOCIATED_WITH", "object": "HCC", "object_type": "Disease",
}


def detect(text, *, compare_scope=True):
    return find_evidence_conflict(
        RELATION,
        source_index=EvidencePackBuilder().index(text),
        profile=PROFILE,
        subject_aliases=["TP53"],
        object_aliases=["HCC"],
        compare_scope=compare_scope,
    )


class EvidenceScopeTests(unittest.TestCase):
    def test_extracts_exact_cohort_sex_model_and_adjustment_witnesses(self):
        text = "Discovery cohort A included, among women, model 2 adjusted."
        witnesses = extract_scope_witnesses(text, char_offset=50)
        as_map = {(item["dimension"], item["value"]): item for item in witnesses}
        self.assertEqual(as_map[("cohort_role", "discovery")]["text"], "Discovery cohort")
        self.assertEqual(as_map[("cohort_id", "A")]["text"], "cohort A")
        self.assertEqual(as_map[("sex", "female")]["text"], "women")
        self.assertEqual(as_map[("model_id", "2")]["text"], "model 2")
        self.assertEqual(as_map[("adjustment", "adjusted")]["text"], "adjusted")
        for item in witnesses:
            self.assertEqual(text[item["char_start"] - 50:item["char_end"] - 50], item["text"])

    def test_scope_comparison_requires_equal_dimension_sets_and_no_overlap_for_different(self):
        discovery_a = extract_scope_witnesses("discovery cohort A among men")
        same = extract_scope_witnesses("discovery cohort A in male participants")
        validation_b = extract_scope_witnesses("validation cohort B among women")
        partial = extract_scope_witnesses("discovery cohort B among men")
        unknown = extract_scope_witnesses("the study participants")
        self.assertEqual(compare_scope(discovery_a, same), "SHARED_SCOPE")
        self.assertEqual(compare_scope(discovery_a, validation_b), "UNRESOLVED_SCOPE")
        self.assertEqual(compare_scope(discovery_a, partial), "DIFFERENT_SCOPE")
        self.assertEqual(compare_scope(discovery_a, unknown), "UNRESOLVED_SCOPE")

    def test_multiple_values_per_dimension_remain_ambiguous(self):
        multiple = extract_scope_witnesses("cohort A and cohort B among men")
        enumerated_cohorts = extract_scope_witnesses("cohorts A and B")
        compact_cohorts = extract_scope_witnesses("cohort A/B")
        enumerated_sexes = extract_scope_witnesses("in men and women")
        compact_sexes = extract_scope_witnesses("among men/women")
        enumerated_models = extract_scope_witnesses("models 1 and 2")
        compact_models = extract_scope_witnesses("models 1 & 2")
        single = extract_scope_witnesses("cohort C among women")
        self.assertEqual(compare_scope(multiple, single), "UNRESOLVED_SCOPE")
        self.assertEqual(compare_scope(enumerated_cohorts, extract_scope_witnesses("cohort A")), "UNRESOLVED_SCOPE")
        self.assertEqual(compare_scope(compact_cohorts, extract_scope_witnesses("cohort A")), "UNRESOLVED_SCOPE")
        self.assertEqual(compare_scope(enumerated_sexes, extract_scope_witnesses("among men")), "UNRESOLVED_SCOPE")
        self.assertEqual(compare_scope(compact_sexes, extract_scope_witnesses("among men")), "UNRESOLVED_SCOPE")
        self.assertEqual(compare_scope(enumerated_models, extract_scope_witnesses("model 1")), "UNRESOLVED_SCOPE")
        self.assertEqual(compare_scope(compact_models, extract_scope_witnesses("model 1")), "UNRESOLVED_SCOPE")

    def test_same_model_with_different_adjustment_is_partial_and_unresolved(self):
        adjusted = extract_scope_witnesses("model 1 adjusted")
        unadjusted = extract_scope_witnesses("model 1 unadjusted")
        other_model = extract_scope_witnesses("model 2 adjusted")
        self.assertEqual(compare_scope(adjusted, unadjusted), "DIFFERENT_SCOPE")
        self.assertEqual(compare_scope(adjusted, other_model), "DIFFERENT_SCOPE")
        self.assertEqual(extract_scope_witnesses("the finding was adjusted"), [])

    def test_patient_criterion_needs_case_sensitive_named_token(self):
        self.assertEqual(extract_scope_witnesses("In MASH patients with t2dm, result."), [])
        explicit = extract_scope_witnesses("In MASH patients with T2DM, result.")
        self.assertEqual(explicit[0]["dimension"], "patient_criterion:t2dm")
        participant_only = extract_scope_witnesses("In patients without T2DM, result.")
        self.assertEqual(participant_only[0]["value"], "without")

    def test_same_scope_pair_outranks_closer_different_scope_and_returns_groups(self):
        text = (
            "ABSTRACT: RESULTS: In cohort A TP53 was associated with HCC; "
            "In cohort B TP53 was not associated with HCC; "
            "In cohort A TP53 was not associated with HCC."
        )
        result = detect(text)
        self.assertEqual(result["scope_status"], "SHARED_SCOPE")
        self.assertEqual(result["scope_pair_counts"]["SHARED_SCOPE"], 1)
        self.assertEqual(result["scope_pair_counts"]["DIFFERENT_SCOPE"], 1)
        self.assertIn("cohort A", result["counterevidence"]["text"])
        self.assertEqual(
            text[result["counterevidence"]["char_start"]:result["counterevidence"]["char_end"]],
            result["counterevidence"]["text"],
        )
        self.assertTrue(any(
            group["polarity"] == "COUNTEREVIDENCE"
            and group["scope"].get("cohort_id") == ["A"]
            for group in result["scoped_assertions"]
        ))

    def test_different_scope_still_contests_and_unresolved_is_reported(self):
        text = (
            "ABSTRACT: RESULTS: Among men TP53 was associated with HCC; "
            "Among women TP53 was not associated with HCC."
        )
        result = detect(text)
        self.assertEqual(result["status"], "CONTESTED")
        self.assertEqual(result["scope_status"], "DIFFERENT_SCOPE")

        unknown = detect(
            "ABSTRACT: RESULTS: TP53 was associated with HCC; "
            "TP53 was not associated with HCC."
        )
        self.assertEqual(unknown["scope_status"], "UNRESOLVED_SCOPE")

    def test_named_membership_qualifier_distinguishes_t2dm_strata(self):
        text = (
            "ABSTRACT: FINDINGS: In MASH patients with T2DM, TP53 was associated with HCC; "
            "In MASH patients without T2DM, TP53 was not associated with HCC."
        )
        result = detect(text)
        self.assertEqual(result["scope_status"], "DIFFERENT_SCOPE")
        for trace in (result["support"], result["counterevidence"]):
            self.assertTrue(any(
                item["dimension"] == "patient_criterion:t2dm"
                and text[item["char_start"]:item["char_end"]] == item["text"]
                for item in trace["scope_witnesses"]
            ))

    def test_compare_scope_false_preserves_legacy_result_shape(self):
        result = detect(
            "ABSTRACT: RESULTS: TP53 was associated with HCC; "
            "TP53 was not associated with HCC.",
            compare_scope=False,
        )
        self.assertEqual(set(result), {"status", "support", "counterevidence"})


if __name__ == "__main__":
    unittest.main()
