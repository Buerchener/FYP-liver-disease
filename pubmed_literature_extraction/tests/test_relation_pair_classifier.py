import unittest
import json
import tempfile
from pathlib import Path

from cognitive_agent.collaborative_extractor import CollaborativeConfig, CollaborativeExtractor
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.relation_pair_classifier import (
    BioREDPairClassifier,
    NO_RELATION,
    PairClassifierConfig,
    PairPrediction,
)
from cognitive_agent.verifier import KGVerifier
from cognitive_agent.rule_memory import RuleBundle, RuleMemory, RuleValidator, SoftRule


class OfflineKG:
    is_connected = False


def entity(mention, entity_type):
    return {"mention": mention, "type": entity_type, "attributes": {}}


class RelationPairClassifierTests(unittest.TestCase):
    def setUp(self):
        self.reader = ArticleEvidenceReader()
        self.entities = [entity("TP53", "Gene"), entity("HCC", "Disease")]

    def classify(self, text, relations=None, **overrides):
        config = PairClassifierConfig(mode="active", **overrides)
        return BioREDPairClassifier(config).classify(
            self.entities, relations or [], self.reader.read(text)
        )

    def test_builds_schema_constrained_evidence_local_pairs(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 was associated with HCC."
        result = self.classify(text)
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual((candidate.subject_type, candidate.object_type), ("Gene", "Disease"))
        self.assertIn("ASSOCIATED_WITH", candidate.allowed_predicates)
        self.assertEqual(text[candidate.evidence_char_start:candidate.evidence_char_end], candidate.evidence)

    def test_article_local_abbreviation_can_pair_long_form_entity_in_later_sentence(self):
        text = (
            "TITLE: Study\nABSTRACT: Icaritin (ICT) was administered. "
            "RESULTS: ICT directly interacted with GSTA1."
        )
        entities = [entity("Icaritin", "Metabolite"), entity("GSTA1", "Protein")]
        result = BioREDPairClassifier(PairClassifierConfig(mode="active")).classify(
            entities, [], self.reader.read(text), source_text=text,
        )
        pairs = {(item.subject, item.object) for item in result.candidates}
        self.assertIn(("Icaritin", "GSTA1"), pairs)

    def test_abbreviation_family_is_deduplicated_before_pairing(self):
        text = (
            "TITLE: Study\nABSTRACT: Hepatocellular carcinoma (HCC) was studied. "
            "RESULTS: TP53 was associated with HCC."
        )
        entities = [
            entity("Hepatocellular carcinoma", "Disease"),
            entity("HCC", "Disease"),
            entity("TP53", "Gene"),
        ]
        result = BioREDPairClassifier(PairClassifierConfig(mode="active")).classify(
            entities, [], self.reader.read(text), source_text=text,
        )
        gene_disease = [
            item for item in result.candidates
            if item.subject_type == "Gene" and item.object_type == "Disease"
        ]
        self.assertEqual(len(gene_disease), 1)
        self.assertEqual(gene_disease[0].object, "Hepatocellular carcinoma")

    def test_celltype_descriptor_anchor_recovers_crosstalk_window(self):
        text = (
            "TITLE: Study\nABSTRACT: We identify a subset of liver endothelial cells "
            "termed Endo4 liver endothelial cells as the source of Wnt9b. "
            "Immunostaining for the Endo4 marker reveals VWF+ vasculature juxtaposing "
            "activated hepatic stellate cells."
        )
        entities = [
            entity("Endo4 liver endothelial cells", "CellType"),
            entity("hepatic stellate cells", "CellType"),
        ]
        result = BioREDPairClassifier(PairClassifierConfig(mode="active")).classify(
            entities, [], self.reader.read(text), source_text=text,
        )
        pairs = {(item.subject, item.object): item for item in result.candidates}
        candidate = pairs[("Endo4 liver endothelial cells", "hepatic stellate cells")]
        self.assertIn("INTERACTS_WITH", candidate.allowed_predicates)
        self.assertEqual(candidate.evidence_trigger_predicate, "INTERACTS_WITH")
        self.assertEqual(
            text[candidate.evidence_char_start:candidate.evidence_char_end],
            candidate.evidence,
        )

    def test_incomplete_exact_hint_becomes_routed_review_candidate(self):
        text = (
            "TITLE: Study\nABSTRACT: RESULTS: TP53 expression increased. "
            "Controls remained stable. HCC cases were enrolled."
        )
        hint = [{
            "subject": "TP53", "subject_type": "Gene",
            "predicate": "ASSOCIATED_WITH",
            "object": "HCC", "object_type": "Disease",
            "evidence": "TP53 expression increased.",
        }]
        result = BioREDPairClassifier(PairClassifierConfig(mode="active")).classify(
            self.entities, hint, self.reader.read(text), source_text=text,
        )
        candidate = result.candidates[0]
        self.assertIn("incomplete_evidence_boundary", candidate.quality_flags)
        self.assertTrue(result.predictions[0].routed_to_llm)
        self.assertEqual(result.accepted_relations, [])
        self.assertEqual(len(result.low_confidence_relations), 1)
        flags = set(result.low_confidence_relations[0]["quality_flags"])
        self.assertIn("endpoint_not_in_evidence", flags)
        self.assertIn("manual_review", flags)
        verified = KGVerifier(
            OfflineKG(), verification_policy="tiered-v2"
        ).verify(
            self.entities, result.low_confidence_relations, text=text,
        ).relations[0]
        self.assertEqual(verified.factual_status, "REVIEW")
        self.assertNotEqual(verified.semantic_status, "REJECTED")
        self.assertEqual(verified.write_status, "HUMAN_REVIEW")
        self.assertFalse(verified.import_ready)

    def test_no_relation_is_an_explicit_class(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 and HCC samples were measured."
        result = self.classify(text)
        self.assertEqual(result.predictions[0].label, NO_RELATION)
        self.assertEqual(result.accepted_relations, [])

    def test_langextract_is_a_hint_not_a_forced_label(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 and HCC samples were measured."
        hint = [{
            "subject": "TP53", "subject_type": "Gene", "predicate": "ASSOCIATED_WITH",
            "object": "HCC", "object_type": "Disease", "evidence": "unsupported",
        }]
        result = self.classify(text, hint)
        prediction = result.predictions[0]
        self.assertLess(prediction.relation_probability, 0.5)
        self.assertNotIn("explicit_predicate_trigger", prediction.reason_codes)

    def test_only_uncertainty_band_candidate_routes_to_deepseek(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 was associated with HCC."
        result = self.classify(text, high_confidence_threshold=0.9)
        self.assertEqual(len(result.low_confidence_relations), 1)
        relation = result.low_confidence_relations[0]
        self.assertIn("pair_low_confidence", relation["quality_flags"])
        self.assertIn("manual_review", relation["quality_flags"])

    def test_classifier_metadata_survives_verification_and_blocks_write(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 was associated with HCC."
        result = self.classify(text, high_confidence_threshold=0.9)
        verified = KGVerifier(OfflineKG()).verify(
            self.entities, result.accepted_relations, text=text
        )
        relation = verified.relations[0]
        self.assertTrue(relation.candidate_id.startswith("p-"))
        self.assertGreater(relation.classifier_confidence, 0)
        self.assertFalse(relation.import_ready)

    def test_core_policy_does_not_send_high_confidence_pair_to_llm(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 was associated with HCC."
        result = self.classify(text, high_confidence_threshold=0.6)
        verified = KGVerifier(OfflineKG()).verify(
            self.entities, result.accepted_relations, text=text
        ).to_dict()
        extractor = CollaborativeExtractor(CollaborativeConfig(
            enabled=True, model_id="deepseek"), generate=lambda _: self.fail()
        )
        collaboration = extractor.collaborate(
            text, {"entities": self.entities}, verified
        )
        self.assertEqual(collaboration.status, "NOT_TRIGGERED")

    def test_batch_backend_is_used_once_for_all_candidates(self):
        class BatchBackend:
            name = "batch-test"

            def __init__(self):
                self.calls = 0

            def predict_many(self, candidates):
                self.calls += 1
                return [PairPrediction(
                    candidate_id=item.candidate_id, label=NO_RELATION,
                    confidence=0.9, relation_probability=0.1,
                    no_relation_probability=0.9, margin=0.8,
                    backend=self.name,
                ) for item in candidates]

        backend = BatchBackend()
        classifier = BioREDPairClassifier(
            PairClassifierConfig(mode="shadow"), backend=backend
        )
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 and HCC were measured."
        result = classifier.classify(self.entities, [], self.reader.read(text))
        self.assertGreater(len(result.candidates), 0)
        self.assertEqual(backend.calls, 1)

    def test_uncertain_no_relation_routes_as_write_blocked_abstention(self):
        class AbstainingBackend:
            name = "abstaining-test"

            def predict(self, candidate):
                return PairPrediction(
                    candidate_id=candidate.candidate_id, label=NO_RELATION,
                    confidence=0.51, relation_probability=0.49,
                    no_relation_probability=0.51, margin=0.02,
                    backend=self.name,
                    predicate_scores={"ASSOCIATED_WITH": 0.49},
                )

        classifier = BioREDPairClassifier(
            PairClassifierConfig(mode="active"), backend=AbstainingBackend()
        )
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 and HCC were jointly evaluated."
        result = classifier.classify(self.entities, [], self.reader.read(text))
        self.assertEqual(result.accepted_relations, [])
        self.assertEqual(len(result.low_confidence_relations), 1)
        flags = set(result.low_confidence_relations[0]["quality_flags"])
        self.assertIn("pair_low_confidence", flags)
        self.assertIn("pair_no_relation_abstention", flags)

    def test_supported_high_confidence_no_relation_routes_to_judge(self):
        class OverconfidentBackend:
            name = "overconfident-test"

            def predict(self, candidate):
                return PairPrediction(
                    candidate_id=candidate.candidate_id, label=NO_RELATION,
                    confidence=0.92, relation_probability=0.08,
                    no_relation_probability=0.92, margin=0.84,
                    backend=self.name, predicate_scores={"ASSOCIATED_WITH": 0.08},
                )

        classifier = BioREDPairClassifier(
            PairClassifierConfig(mode="active"), backend=OverconfidentBackend()
        )
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 and HCC were jointly evaluated."
        result = classifier.classify(self.entities, [], self.reader.read(text))
        self.assertEqual(result.predictions[0].label, NO_RELATION)
        self.assertTrue(result.predictions[0].routed_to_llm)
        self.assertIn("supported_no_relation_routed", result.predictions[0].reason_codes)
        self.assertEqual(len(result.low_confidence_relations), 1)

    def test_active_rule_prior_is_bounded_and_audited(self):
        rule_payload = {
            "kind": "pair_prior", "conditions": {"predicates": ["ASSOCIATED_WITH"]},
            "action": "ADJUST_PAIR_SCORE", "value": -0.2, "guidance": "",
        }
        rule = SoftRule(
            rule_id=RuleValidator.content_id(rule_payload), version=1, status="active",
            support_pmids=["10000001", "10000002", "10000003"],
            critic_approved=True, **rule_payload,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rules.json"
            path.write_text(json.dumps(RuleBundle(rules=[rule]).to_dict()), encoding="utf-8")
            memory = RuleMemory(mode="active", bundle_path=path)
            classifier = BioREDPairClassifier(
                PairClassifierConfig(mode="active"), rule_memory=memory,
            )
            text = "TITLE: Study\nABSTRACT: RESULTS: TP53 was associated with HCC."
            result = classifier.classify(
                self.entities, [], self.reader.read(text), source_text=text,
            )
        prediction = result.predictions[0]
        self.assertEqual(prediction.rule_score_delta, -0.2)
        self.assertEqual(prediction.rule_matches[0]["rule_id"], rule.rule_id)
        self.assertEqual(prediction.evidence_confidence, 0.92)


if __name__ == "__main__":
    unittest.main()
