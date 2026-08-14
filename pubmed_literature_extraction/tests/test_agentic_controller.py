import unittest

from cognitive_agent.abbreviation_detector import AbbreviationDetector
from cognitive_agent.agentic_controller import (
    AgenticArticleController,
    partition_recovery_relations,
)
from cognitive_agent.collaborative_extractor import CollaborationResult, CollaborativeExtractor
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.verifier import KGVerifier


class OfflineKG:
    is_connected = False


def entity(mention, entity_type):
    return {"mention": mention, "type": entity_type, "attributes": {}, "grounded": True}


class AgenticControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = AgenticArticleController(max_recovery_candidates=12)
        self.reader = ArticleEvidenceReader()
        self.abbreviations = AbbreviationDetector()
        self.verifier = KGVerifier(OfflineKG())

    def test_repairs_only_with_an_exact_source_span_containing_both_endpoints(self):
        text = (
            "TITLE: OTUD5 study\nABSTRACT: RESULTS: OTUD5 interacted with MAVS "
            "in macrophages and the result was experimentally validated."
        )
        relations = [{
            "subject": "OTUD5", "subject_type": "Gene",
            "predicate": "INTERACTS_WITH",
            "object": "MAVS", "object_type": "Gene",
            "evidence": "its interaction with MAVS", "direction": "unknown",
            "negated": False, "uncertain": False,
        }]
        plan = self.controller.plan(
            text, [entity("OTUD5", "Gene"), entity("MAVS", "Gene")], relations,
            self.abbreviations.detect(text), self.reader.read(text),
        )
        self.assertEqual(len(plan.evidence_repairs), 1)
        repaired = plan.repaired_relations[0]["evidence"]
        self.assertIn("OTUD5 interacted with MAVS", repaired)
        self.assertIn(repaired, text)

    def test_builds_bounded_relation_choices_from_existing_entities(self):
        text = (
            "TITLE: CILP2 in MASLD\nABSTRACT: RESULTS: Our findings indicate that "
            "CILP2 contributes to MASLD via the IRE1α/XBP1 pathway."
        )
        entities = [
            entity("CILP2", "Gene"), entity("MASLD", "Disease"),
            entity("IRE1α/XBP1 pathway", "Pathway"),
        ]
        units = self.reader.read(text)
        abbreviation_map = self.abbreviations.detect(text)
        verified = self.verifier.verify(entities, [], text=text)
        plan = self.controller.plan(text, entities, [], abbreviation_map, units)
        self.controller.add_recovery_observation(
            plan, text, [item.to_dict() for item in verified.entities], [],
            abbreviation_map, units,
        )
        pairs = {
            (item["subject"], item["object"]): set(item["allowed_predicates"])
            for item in plan.recovery_candidates
        }
        self.assertIn("ASSOCIATED_WITH", pairs[("CILP2", "MASLD")])
        self.assertIn("PARTICIPATES_IN", pairs[("CILP2", "IRE1α/XBP1 pathway")])
        self.assertLessEqual(len(plan.recovery_candidates), 12)

    def test_does_not_recover_from_explicit_background_or_methods(self):
        text = (
            "TITLE: Study\nABSTRACT: BACKGROUND: TP53 is associated with HCC. "
            "METHODS: TP53 and HCC were measured."
        )
        entities = [entity("TP53", "Gene"), entity("HCC", "Disease")]
        units = self.reader.read(text)
        abbreviation_map = self.abbreviations.detect(text)
        verified = self.verifier.verify(entities, [], text=text)
        plan = self.controller.plan(text, entities, [], abbreviation_map, units)
        self.controller.add_recovery_observation(
            plan, text, [item.to_dict() for item in verified.entities], [],
            abbreviation_map, units,
        )
        self.assertEqual(plan.recovery_candidates, [])

    def test_precision_mode_disables_repair_and_recovery(self):
        controller = AgenticArticleController(
            enable_evidence_repair=False, enable_recovery=False
        )
        text = "ABSTRACT: RESULTS: TP53 was associated with HCC."
        relations = [{
            "subject": "TP53", "subject_type": "Gene",
            "predicate": "ASSOCIATED_WITH", "object": "HCC",
            "object_type": "Disease", "evidence": "truncated",
        }]
        units = self.reader.read(text)
        abbreviation_map = self.abbreviations.detect(text)
        plan = controller.plan(
            text, [entity("TP53", "Gene"), entity("HCC", "Disease")],
            relations, abbreviation_map, units,
        )
        controller.add_recovery_observation(
            plan, text, [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [], abbreviation_map, units,
        )
        self.assertEqual(plan.repaired_relations[0]["evidence"], "truncated")
        self.assertEqual(plan.evidence_repairs, [])
        self.assertEqual(plan.recovery_candidates, [])
        self.assertEqual(plan.actions[-1].reason, "disabled by precision mode")

    def test_recovery_ranking_filters_nonhuman_only_evidence(self):
        controller = AgenticArticleController(min_recovery_score=0.55)
        text = (
            "ABSTRACT: RESULTS: In mouse cell lines, TP53 was associated with HCC."
        )
        entities = [entity("TP53", "Gene"), entity("HCC", "Disease")]
        verified = self.verifier.verify(entities, [], text=text)
        candidates = controller.build_recovery_candidates(
            text, [item.to_dict() for item in verified.entities], [],
            self.abbreviations.detect(text), self.reader.read(text),
        )
        self.assertEqual(candidates, [])

    def test_shadow_partition_structurally_removes_agent_additions(self):
        base = {
            "subject": "TP53", "subject_type": "Gene",
            "predicate": "ASSOCIATED_WITH", "object": "HCC",
            "object_type": "Disease", "quality_flags": [],
        }
        recovered = {
            **base, "object": "MASLD",
            "quality_flags": ["agent_recovered_relation"],
        }
        production, shadow = partition_recovery_relations(
            [base, recovered], [base, recovered], "shadow-agent"
        )
        self.assertEqual(production, [base])
        self.assertEqual(shadow, [recovered])
        recall_production, _ = partition_recovery_relations(
            [base, recovered], [base, recovered], "recall"
        )
        self.assertEqual(len(recall_production), 2)

    def test_merge_adds_only_the_fixed_recovery_pair_and_evidence(self):
        candidate = {
            "candidate_id": "p000", "candidate_kind": "recovery", "raw_index": -1,
            "subject": "hepatic fibrosis", "subject_type": "Disease",
            "predicate": "NONE", "object": "right ventricular dysfunction",
            "object_type": "Disease", "direction": "unknown",
            "evidence": "Hepatic fibrosis was associated with right ventricular dysfunction.",
            "allowed_predicates": [{"predicate": "ASSOCIATED_WITH", "meaning": "association"}],
        }
        collaboration = CollaborationResult(
            status="OK", recovery_candidates=[candidate], review_decisions=[{
                "candidate_id": "p000", "candidate_kind": "recovery", "raw_index": -1,
                "action": "ADD_RELATION", "new_predicate": "ASSOCIATED_WITH",
                "new_direction": "unknown", "evidence_unit_id": "p001",
                "swap_endpoints": True, "reason_code": "EXPLICIT_DIRECT_RELATION",
                "reason": "explicit association", "confidence": 0.9,
            }],
        )
        merged = CollaborativeExtractor().merge([], [], {"relations": []}, collaboration)
        self.assertEqual(merged.relation_additions, 1)
        self.assertEqual(merged.relations[0]["subject"], "right ventricular dysfunction")
        self.assertEqual(merged.relations[0]["object"], "hepatic fibrosis")
        self.assertEqual(merged.relations[0]["evidence"], candidate["evidence"])

    def test_post_action_feedback_rolls_back_unsupported_recovery(self):
        text = "TP53 and HCC were included in the same analysis."
        relations = [{
            "subject": "TP53", "subject_type": "Gene",
            "predicate": "ASSOCIATED_WITH", "object": "HCC", "object_type": "Disease",
            "evidence": text, "direction": "unknown", "negated": False,
            "uncertain": False, "quality_flags": ["agent_recovered_relation"],
        }]
        verified = self.verifier.verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")], relations, text=text
        )
        final, audit = CollaborativeExtractor().finalize_after_reverification(
            relations, verified.to_dict()
        )
        self.assertEqual(final, [])
        self.assertEqual(audit["rolled_back_count"], 1)

    def test_post_action_feedback_deduplicates_by_typed_triple(self):
        text = "TP53 was associated with HCC."
        relation = {
            "subject": "TP53", "subject_type": "Gene",
            "predicate": "ASSOCIATED_WITH", "object": "HCC", "object_type": "Disease",
            "evidence": text, "direction": "unknown", "negated": False, "uncertain": False,
        }
        relations = [relation, dict(relation)]
        verified = self.verifier.verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")], relations, text=text
        )
        final, audit = CollaborativeExtractor().finalize_after_reverification(
            relations, verified.to_dict()
        )
        self.assertEqual(len(final), 1)
        self.assertEqual(audit["duplicate_relations_removed"], 1)


if __name__ == "__main__":
    unittest.main()
