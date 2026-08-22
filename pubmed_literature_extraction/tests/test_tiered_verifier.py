import unittest

from cognitive_agent.decision_engine import DecisionEngine
from cognitive_agent.verifier import KGVerifier


class OfflineKG:
    is_connected = False


def entity(mention, entity_type):
    return {"mention": mention, "type": entity_type, "attributes": {}}


def relation(evidence, **extra):
    return {
        "subject": extra.pop("subject", "TP53"),
        "subject_type": extra.pop("subject_type", "Gene"),
        "predicate": extra.pop("predicate", "ASSOCIATED_WITH"),
        "object": extra.pop("object", "HCC"),
        "object_type": extra.pop("object_type", "Disease"),
        "evidence": evidence,
        "direction": extra.pop("direction", "positive"),
        "negated": False,
        "uncertain": False,
        **extra,
    }


class TieredVerifierTests(unittest.TestCase):
    def verifier(self):
        return KGVerifier(OfflineKG(), verification_policy="tiered-v2")

    def test_background_method_prediction_are_review_not_semantic_reject(self):
        cases = [
            ("BACKGROUND: TP53 is associated with HCC.", "BACKGROUND"),
            ("METHODS: TP53 was associated with HCC in the screening model.", "METHOD"),
            ("RESULTS: TP53 may be associated with HCC.", "PREDICTION"),
        ]
        for text, expected_role in cases:
            with self.subTest(expected_role=expected_role):
                verified = self.verifier().verify(
                    [entity("TP53", "Gene"), entity("HCC", "Disease")],
                    [relation(text, claim_role=expected_role)],
                    text=f"TITLE: Study\nABSTRACT: {text}",
                )
                rel = verified.relations[0]
                self.assertNotEqual(rel.semantic_status, "REJECTED")
                self.assertIn(rel.write_status, {"HUMAN_REVIEW", "SEMANTIC_ONLY"})
                self.assertFalse(rel.import_ready)

    def test_attached_negation_outside_endpoint_scope_does_not_reject(self):
        text = (
            "TITLE: Study\nABSTRACT: RESULTS: No adverse events were reported. "
            "TP53 was associated with HCC."
        )
        verified = self.verifier().verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation("No adverse events were reported. TP53 was associated with HCC.")],
            text=text,
        )
        rel = verified.relations[0]
        self.assertNotIn("scoped_negation", rel.quality_flags)
        self.assertNotEqual(rel.factual_status, "REJECTED")

    def test_scoped_negation_remains_non_overridable(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 was not associated with HCC."
        verified = self.verifier().verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation(
                "TP53 was not associated with HCC.",
                quality_flags=["adjudicator_entailed", "qwen_critic_approved", "dual_model_entailed"],
            )],
            text=text,
        )
        rel = verified.relations[0]
        self.assertEqual(rel.factual_status, "REJECTED")
        self.assertEqual(rel.write_status, "BLOCKED")
        self.assertFalse(rel.import_ready)

    def test_dual_model_endorsement_can_unlock_non_current_write(self):
        text = "TITLE: Study\nABSTRACT: BACKGROUND: TP53 is associated with HCC."
        verified = self.verifier().verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation(
                "TP53 is associated with HCC.",
                claim_role="BACKGROUND",
                classifier_source="pairwise_judge_v1",
                evidence_entailment="ENTAILED",
                quality_flags=[
                    "non_current_finding_role", "adjudicator_entailed",
                    "qwen_critic_approved", "dual_model_entailed",
                ],
            )],
            text=text,
        )
        rel = verified.relations[0]
        self.assertEqual(rel.write_status, "IMPORT_READY")
        self.assertTrue(rel.import_ready)
        self.assertEqual(rel.claim_role, "BACKGROUND")

    def test_judge_source_and_trigger_risk_block_write(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 was measured in HCC."
        verified = self.verifier().verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation(
                "TP53 was measured in HCC.",
                classifier_source="pairwise_judge_v1",
                evidence_entailment="ENTAILED",
                quality_flags=[
                    "adjudicator_entailed",
                    "judge_quote_not_in_source",
                    "judge_entailment_without_trigger_support",
                ],
            )],
            text=text,
        )
        rel = verified.relations[0]
        self.assertFalse(rel.import_ready)
        self.assertEqual(rel.write_status, "HUMAN_REVIEW")
        self.assertIn("judge_quote_not_in_source", rel.write_reasons)

    def test_semantic_only_and_human_review_are_not_discarded(self):
        rel = self.verifier().verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation("TP53 expression changed in HCC.", quality_flags=["trigger_missing"])],
            text="TITLE: Study\nABSTRACT: RESULTS: TP53 expression changed in HCC.",
        ).relations[0]
        action = DecisionEngine(OfflineKG(), skip_neo4j_write=True)._decide_relation(
            rel, pmid="1"
        )
        self.assertEqual(action.type, "NO_ACTION")

    def test_semantic_review_cannot_be_import_ready_after_dual_endorsement(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 was measured in HCC."
        rel = self.verifier().verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation(
                "TP53 was measured in HCC.",
                classifier_source="pairwise_judge_v1",
                evidence_entailment="ENTAILED",
                quality_flags=[
                    "evidence_not_entailed", "adjudicator_entailed",
                    "qwen_critic_approved", "dual_model_entailed",
                ],
            )],
            text=text,
        ).relations[0]
        self.assertEqual(rel.semantic_status, "REVIEW")
        self.assertEqual(rel.write_status, "HUMAN_REVIEW")
        self.assertFalse(rel.import_ready)


if __name__ == "__main__":
    unittest.main()
