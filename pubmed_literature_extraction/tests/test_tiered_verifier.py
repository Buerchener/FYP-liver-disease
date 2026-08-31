import unittest

from cognitive_agent.decision_engine import DecisionEngine
from cognitive_agent.evidence_pack import EvidencePackBuilder
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

    def test_dual_model_endorsement_cannot_unlock_non_current_write(self):
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
        self.assertEqual(rel.semantic_status, "REVIEW")
        self.assertEqual(rel.write_status, "HUMAN_REVIEW")
        self.assertFalse(rel.import_ready)
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

    def test_cross_sentence_closed_pack_is_semantically_accepted_but_not_written(self):
        text = (
            "TITLE: Study\nABSTRACT: RESULTS: TP53 was measured in the cohort. "
            "It was associated with HCC."
        )
        rel = self.verifier().verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation(
                "TP53 was measured in the cohort. It was associated with HCC.",
                quality_flags=["coreference_only_support"],
            )],
            text=text,
        ).relations[0]
        self.assertNotEqual(rel.factual_status, "REJECTED")
        # v4 requires SELF_CONTAINED + EXPLICIT for deterministic promotion;
        # a resolved pronoun remains reviewable until structured adjudication.
        self.assertEqual(rel.semantic_status, "REVIEW")
        self.assertNotIn("cross_sentence", rel.semantic_reasons)
        self.assertEqual(rel.write_status, "HUMAN_REVIEW")

    def test_grounded_structured_adjudication_promotes_pair_dissent(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 expression increased in HCC."
        base = relation("TP53 expression increased in HCC.")
        span_id = EvidencePackBuilder().build(base, text=text).spans[0].span_id
        checked = self.verifier().verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation(
                "TP53 expression increased in HCC.",
                quality_flags=["pair_no_relation_dissent", "manual_review"],
                adjudication_verdict="SUPPORTED",
                adjudication_reason_code="EXPLICIT_DIRECT_RELATION",
                adjudication_confidence=0.95,
                supporting_span_ids=[span_id],
                adjudication={
                    "verdict": "SUPPORTED",
                    "reason_code": "EXPLICIT_DIRECT_RELATION",
                    "confidence": 0.95,
                    "supporting_span_ids": [span_id],
                    "model_id": "deepseek-test",
                },
            )],
            text=text,
        ).relations[0]
        self.assertEqual(checked.semantic_status, "ACCEPTED")
        self.assertEqual(checked.promotion_path, "ADJUDICATED")
        self.assertEqual(checked.adjudication_model_id, "deepseek-test")

    def test_supported_adjudication_without_valid_span_stays_review(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 expression increased in HCC."
        checked = self.verifier().verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation(
                "TP53 expression increased in HCC.",
                quality_flags=["pair_no_relation_dissent", "manual_review"],
                adjudication_verdict="SUPPORTED",
                adjudication_reason_code="EXPLICIT_DIRECT_RELATION",
                adjudication_confidence=0.95,
                supporting_span_ids=["missing-span"],
            )],
            text=text,
        ).relations[0]
        self.assertEqual(checked.semantic_status, "REVIEW")
        self.assertIn("adjudication_span_mismatch", checked.semantic_reasons)

    def test_qwen_agreement_can_clear_legacy_trigger_direction_mismatch(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: Obesity is strongly associated with HCC."
        base = relation(
            "Obesity is strongly associated with HCC.",
            subject="Obesity", subject_type="Disease",
        )
        span_id = EvidencePackBuilder().build(base, text=text).spans[0].span_id
        checked = self.verifier().verify(
            [entity("Obesity", "Disease"), entity("HCC", "Disease")],
            [{
                **base,
                "quality_flags": [
                    "trigger_direction_mismatch", "adjudicator_entailed",
                    "qwen_critic_approved", "dual_model_entailed",
                ],
                "adjudication_verdict": "SUPPORTED",
                "adjudication_reason_code": "EXPLICIT_DIRECT_RELATION",
                "adjudication_confidence": 0.95,
                "supporting_span_ids": [span_id],
            }],
            text=text,
        ).relations[0]
        self.assertEqual(checked.semantic_status, "ACCEPTED")
        self.assertEqual(checked.promotion_path, "ADJUDICATED")

    def test_trigger_direction_mismatch_without_qwen_stays_review(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: Obesity is strongly associated with HCC."
        base = relation(
            "Obesity is strongly associated with HCC.",
            subject="Obesity", subject_type="Disease",
        )
        span_id = EvidencePackBuilder().build(base, text=text).spans[0].span_id
        checked = self.verifier().verify(
            [entity("Obesity", "Disease"), entity("HCC", "Disease")],
            [{
                **base,
                "quality_flags": ["trigger_direction_mismatch", "adjudicator_entailed"],
                "adjudication_verdict": "SUPPORTED",
                "adjudication_reason_code": "EXPLICIT_DIRECT_RELATION",
                "adjudication_confidence": 0.95,
                "supporting_span_ids": [span_id],
            }],
            text=text,
        ).relations[0]
        self.assertEqual(checked.semantic_status, "REVIEW")
        self.assertIn("trigger_direction_mismatch", checked.semantic_reasons)

    def test_legacy_direction_diagnostics_do_not_block_semantic_acceptance(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 encodes p53."
        checked = self.verifier().verify(
            [entity("TP53", "Gene"), entity("p53", "Protein")],
            [relation(
                "TP53 encodes p53.", predicate="ENCODES", object="p53",
                object_type="Protein", direction="positive",
            )],
            text=text,
        ).relations[0]
        self.assertEqual(checked.association_sign, "UNKNOWN")
        self.assertEqual(checked.semantic_status, "ACCEPTED")
        self.assertNotIn("association_sign_not_applicable", checked.semantic_reasons)


if __name__ == "__main__":
    unittest.main()
