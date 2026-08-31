import unittest

from scripts.evaluate_gold200_unified import (
    aliases,
    canonical_endpoint,
    cell_subset_signature,
    direction_kind,
    prf,
    score_funnel,
    score_view,
    semantic_risk_coverage,
    wilson_interval,
)
from collections import Counter


class Gold200UnifiedEvaluationTests(unittest.TestCase):
    def test_numbered_cell_subset_paraphrases_share_one_signature(self):
        self.assertEqual(
            cell_subset_signature("subpopulation 11 mononuclear macrophages"),
            cell_subset_signature("macrophage subset 11"),
        )

    def test_numbered_cell_subset_alias_resolves_to_gold_canonical(self):
        alias_map = {
            "macrophage subset 11": {("macrophage subset 11", "CellType")},
        }
        self.assertEqual(
            canonical_endpoint(
                "subpopulation 11 mononuclear macrophages", "CellType", alias_map,
            ),
            "macrophage subset 11",
        )

    def test_non_numbered_cell_phrases_are_not_loosely_merged(self):
        self.assertIsNone(cell_subset_signature("mononuclear macrophages"))

    def test_direction_kind_preserves_sign_vs_change_semantics(self):
        self.assertEqual(direction_kind({"direction": "positive"}), "ASSOCIATION_SIGN")
        self.assertEqual(direction_kind({"direction": "increase"}), "CHANGE_DIRECTION")
        self.assertEqual(
            direction_kind({"direction": "increase", "direction_semantics": "CUSTOM"}),
            "CUSTOM",
        )

    def test_wilson_interval_exposes_small_sample_uncertainty(self):
        low, high = wilson_interval(15, 15)
        self.assertLess(low, 0.85)
        self.assertEqual(high, 1.0)
        self.assertIsNone(wilson_interval(0, 0))

    def test_empty_prediction_precision_is_undefined(self):
        metric = prf(Counter(tp=0, fp=0, fn=3))
        self.assertIsNone(metric["precision"])
        self.assertEqual(metric["recall"], 0.0)
        self.assertEqual(metric["f1"], 0.0)

    def test_semantic_risk_coverage_reports_bounded_aurc(self):
        result = semantic_risk_coverage([(0.9, True), (0.8, True), (0.2, False)])
        self.assertEqual(result["population"], 3)
        self.assertGreaterEqual(result["aurc"], 0.0)
        self.assertLessEqual(result["aurc"], 1.0)

    def test_tp_survival_cannot_exceed_one_when_recovery_adds_tp(self):
        first = {
            "subject": "TP53", "subject_type": "Gene", "predicate": "ASSOCIATED_WITH",
            "object": "HCC", "object_type": "Disease",
        }
        recovered = {
            "subject": "EGFR", "subject_type": "Gene", "predicate": "ASSOCIATED_WITH",
            "object": "HCC", "object_type": "Disease",
        }
        record = {"pmid": "1", "phases": {
            "extraction": {"relations": [first]},
            "relation_candidate_projection": {"relations": [
                {**first, "candidate_id": "hint-1", "candidate_lane": "extracted_hint"},
            ]},
            "relation_core_selection": {"relations": [first, recovered]},
            "verification": {"relations": [
                {**first, "candidate_id": "hint-1", "candidate_lane": "extracted_hint",
                 "factual_status": "VALID", "semantic_status": "ACCEPTED"},
                {**recovered, "candidate_id": "recovery-1", "candidate_lane": "recovery",
                 "factual_status": "VALID", "semantic_status": "ACCEPTED"},
            ]},
        }}
        gold = {"1": {"entities": [], "relations": [first, recovered]}}
        source = {"1": {"title": "", "abstract": ""}}
        result = score_funnel([record], gold, source)
        self.assertEqual(result["derived"]["raw_canonical_tp_survival"], 1.0)
        self.assertEqual(result["derived"]["recovery_tp_gain"], 1)
        self.assertEqual(result["derived"]["net_tp_gain"], 1)

    def test_tigit_long_form_parenthetical_alias_is_typed(self):
        gold = {"entities": [{"mention": "TIGIT", "canonical": "TIGIT", "type": "Protein"}]}
        text = (
            "interactions between cluster of differentiation 155 (CD155) and "
            "T cell immunoreceptor with Ig and ITIM domains (TIGIT)."
        )
        audit = {}
        mapping = aliases(gold, text, audit)
        self.assertIn(
            ("tigit", "Protein"),
            mapping["t cell immunoreceptor with ig and itim domains"],
        )
        self.assertTrue(audit["accepted_pairs"])

    def test_hcc_parenthetical_endpoint_resolves_to_typed_gold_alias(self):
        gold = {
            "entities": [{
                "mention": "HCC", "canonical": "HCC", "type": "Disease",
            }],
        }
        mapping = aliases(
            gold, "Hepatocellular carcinoma (HCC) was investigated.", {}
        )
        self.assertEqual(
            canonical_endpoint(
                "Hepatocellular carcinoma (HCC)", "Disease", mapping,
            ),
            "hcc",
        )

    def test_untyped_diagnostic_separates_endpoint_type_error(self):
        gold_edge = {
            "subject": "OTUD5", "subject_type": "Gene",
            "predicate": "INTERACTS_WITH", "object": "MAVS",
            "object_type": "Protein",
        }
        predicted = {
            **gold_edge, "object_type": "Gene",
            "factual_status": "VALID", "semantic_status": "ACCEPTED",
        }
        metric = score_view(
            [{"pmid": "1", "phases": {"verification": {"relations": [predicted]}}}],
            {"1": {"entities": [], "relations": [gold_edge]}},
            {"1": {"title": "", "abstract": ""}},
            "candidate_semantic",
        )
        self.assertEqual(metric["typed_directed_triple_micro"]["tp"], 0)
        self.assertEqual(metric["untyped_endpoint_diagnostic"]["tp"], 1)

    def test_recovery_gain_reads_claim_source_lanes_not_representative_lane(self):
        edge = {
            "subject": "TP53", "subject_type": "Gene",
            "predicate": "ASSOCIATED_WITH", "object": "HCC",
            "object_type": "Disease",
        }
        record = {"pmid": "1", "phases": {
            "extraction": {"relations": []},
            "relation_candidate_projection": {"relations": []},
            "relation_core_selection": {"relations": [edge]},
            "verification": {"relations": [{
                **edge, "candidate_id": "c-representative",
                "candidate_lane": "extracted_hint", "source_lanes": ["recovery"],
                "claim_instances": [{
                    "candidate_id": "r-source", "candidate_lane": "recovery",
                }],
                "factual_status": "VALID", "semantic_status": "ACCEPTED",
            }]},
        }}
        result = score_funnel(
            [record], {"1": {"entities": [], "relations": [edge]}},
            {"1": {"title": "", "abstract": ""}},
        )
        self.assertEqual(result["derived"]["recovery_tp_gain"], 1)

    def test_lineage_survival_uses_merged_source_ids(self):
        edge = {
            "subject": "TP53", "subject_type": "Gene", "predicate": "ASSOCIATED_WITH",
            "object": "HCC", "object_type": "Disease",
        }
        record = {"pmid": "1", "phases": {
            "extraction": {"relations": [edge]},
            "relation_candidate_projection": {"relations": [
                {**edge, "candidate_id": "c-hint", "candidate_lane": "extracted_hint"},
            ]},
            "relation_core_selection": {"relations": [edge]},
            "verification": {"relations": [{
                **edge, "candidate_id": "c-representative",
                "merged_candidate_ids": ["c-hint"],
                "factual_status": "VALID", "semantic_status": "ACCEPTED",
            }]},
        }}
        result = score_funnel(
            [record], {"1": {"entities": [], "relations": [edge]}},
            {"1": {"title": "", "abstract": ""}},
        )
        self.assertEqual(result["derived"]["candidate_id_lineage_tp_survival"], 1.0)


if __name__ == "__main__":
    unittest.main()
