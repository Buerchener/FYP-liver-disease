import json
import unittest

from cognitive_agent.aux_model_registry import AuxModelRegistry, AuxModelSpec
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.relation_pair_classifier import (
    BioREDPairClassifier,
    NO_RELATION,
    PairClassifierConfig,
    PairClassificationResult,
)
from cognitive_agent.pairwise_judge import (
    ENTAILED,
    NOT_ENOUGH_INFORMATION,
    PairwiseJudge,
    PairwiseJudgeConfig,
)
from cognitive_agent.verifier import KGVerifier


class OfflineKG:
    is_connected = False


class FakeRegistry(AuxModelRegistry):
    def __init__(self, payload):
        self.payload = payload
        self.roles_called = []
        super().__init__([AuxModelSpec(
            role="judge", provider="openai", model_id="fake-judge",
            api_base="http://localhost", api_key="test",
        )])

    def configured(self, role):
        return role == "judge"

    def call_json(self, role, *, system_prompt, user_prompt, schema_hint=None):
        self.roles_called.append(role)
        from cognitive_agent.aux_model_registry import StructuredModelResult
        return StructuredModelResult(
            role, "fake-judge", "OK", payload=self.payload,
            latency_s=0.0, prompt_tokens=1, output_tokens=1, attempts=1,
        )


def entity(mention, entity_type):
    return {"mention": mention, "type": entity_type, "attributes": {}}


class PairwiseJudgeTests(unittest.TestCase):
    TEXT = "TITLE: Study\nABSTRACT: RESULTS: TP53 expression was associated with HCC progression."

    def judge_with(self, payload, **overrides):
        config = PairwiseJudgeConfig(mode="active", **overrides)
        return PairwiseJudge(config, FakeRegistry(payload))

    def pair_result(self, relations=None):
        entities = [entity("TP53", "Gene"), entity("HCC", "Disease")]
        return BioREDPairClassifier(PairClassifierConfig(mode="active")).classify(
            entities, relations or [], ArticleEvidenceReader().read(self.TEXT),
            source_text=self.TEXT,
        )

    def test_predicate_outside_shortlist_is_rejected_to_no_relation(self):
        payload = {"decisions": [{
            "candidate_id": "p-x", "predicate": "EXPRESSED_IN",
            "evidence_quote": "TP53 expression was associated with HCC progression.",
            "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
        }]}
        judge = self.judge_with(payload)
        candidate = self.pair_result().candidates[0]
        decision = judge._validate(
            payload["decisions"][0], candidate, text=self.TEXT, alias_index={},
        )
        self.assertEqual(decision.label, NO_RELATION)
        self.assertIn("predicate_not_in_shortlist", decision.reason_codes)

    def test_quote_outside_source_falls_back_to_window_and_nei(self):
        payload = {"decisions": [{
            "candidate_id": "p-x", "predicate": "ASSOCIATED_WITH",
            "evidence_quote": "a hallucinated sentence that is not in the source",
            "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
        }]}
        judge = self.judge_with(payload)
        candidate = self.pair_result().candidates[0]
        decision = judge._validate(
            payload["decisions"][0], candidate, text=self.TEXT, alias_index={},
        )
        self.assertIn("judge_quote_not_in_source", decision.reason_codes)

    def test_refine_splits_accepted_and_uncertain(self):
        candidates = self.pair_result().candidates
        payload = {"decisions": [
            {
                "candidate_id": candidates[0].candidate_id,
                "predicate": "ASSOCIATED_WITH",
                "evidence_quote": "TP53 expression was associated with HCC progression.",
                "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
            },
        ]}
        judge = self.judge_with(payload)
        refined = judge.refine_pair_result(
            self.pair_result(), text=self.TEXT,
            entities=[entity("TP53", "Gene"), entity("HCC", "Disease")], pmid="1",
        )
        self.assertEqual(len(refined.accepted_relations), 1)
        relation = refined.accepted_relations[0]
        self.assertEqual(relation["classifier_source"], "pairwise_judge_v1")
        self.assertEqual(relation["evidence_entailment"], ENTAILED)
        self.assertIn("pairwise_judge", relation["quality_flags"])
        self.assertEqual(
            self.TEXT[relation["evidence_char_start"]:relation["evidence_char_end"]],
            relation["evidence"],
        )

    def test_refine_routes_low_confidence_to_uncertain(self):
        candidates = self.pair_result().candidates
        payload = {"decisions": [{
            "candidate_id": candidates[0].candidate_id,
            "predicate": "ASSOCIATED_WITH",
            "evidence_quote": "TP53 expression was associated with HCC progression.",
            "direction": "A_TO_B", "decision": NOT_ENOUGH_INFORMATION,
            "confidence": 0.4,
        }]}
        judge = self.judge_with(payload)
        refined = judge.refine_pair_result(
            self.pair_result(), text=self.TEXT,
            entities=[entity("TP53", "Gene"), entity("HCC", "Disease")], pmid="1",
        )
        self.assertEqual(refined.accepted_relations, [])
        self.assertEqual(len(refined.low_confidence_relations), 1)
        flags = set(refined.low_confidence_relations[0]["quality_flags"])
        self.assertIn("judge_uncertain", flags)
        self.assertIn("pair_low_confidence", flags)

    def test_source_direction_survives_judge_when_endpoints_and_predicate_stay_same(self):
        hint = [{
            "subject": "TP53", "subject_type": "Gene",
            "predicate": "ASSOCIATED_WITH", "object": "HCC",
            "object_type": "Disease",
            "evidence": "TP53 expression was associated with HCC progression.",
            "direction": "decrease",
        }]
        pair_result = self.pair_result(hint)
        candidate = pair_result.candidates[0]
        payload = {"decisions": [{
            "candidate_id": candidate.candidate_id,
            "predicate": "ASSOCIATED_WITH",
            "evidence_quote": "TP53 expression was associated with HCC progression.",
            "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
        }]}
        refined = self.judge_with(payload).refine_pair_result(
            pair_result, text=self.TEXT,
            entities=[entity("TP53", "Gene"), entity("HCC", "Disease")], pmid="1",
        )
        self.assertEqual(refined.accepted_relations[0]["direction"], "decrease")

    def test_judge_failure_keeps_local_predictions(self):
        judge = PairwiseJudge(
            PairwiseJudgeConfig(mode="active"),
            AuxModelRegistry.from_environment(judge_model="unconfigured-model"),
        )
        refined = judge.refine_pair_result(
            self.pair_result(), text=self.TEXT,
            entities=[entity("TP53", "Gene"), entity("HCC", "Disease")], pmid="1",
        )
        self.assertTrue(refined.fallback_reason)

    def test_judge_entailed_still_requires_deterministic_reverification(self):
        candidates = self.pair_result().candidates
        payload = {"decisions": [{
            "candidate_id": candidates[0].candidate_id,
            "predicate": "ASSOCIATED_WITH",
            "evidence_quote": "TP53 expression was associated with HCC progression.",
            "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
        }]}
        judge = self.judge_with(payload)
        refined = judge.refine_pair_result(
            self.pair_result(), text=self.TEXT,
            entities=[entity("TP53", "Gene"), entity("HCC", "Disease")], pmid="1",
        )
        verified = KGVerifier(OfflineKG()).verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            refined.accepted_relations, pmid="1", text=self.TEXT,
        )
        relation = verified.relations[0]
        # This text has a trigger linking both endpoints, so there is nothing
        # to override: the judge proposal is accepted on the merits.
        self.assertEqual(relation.semantic_status, "ACCEPTED")
        # The quote then independently satisfies every deterministic write
        # requirement.  IMPORT_READY is only a status; Neo4j remains disabled.
        self.assertTrue(relation.import_ready)

    def test_judge_entailed_is_a_proposal_not_an_override(self):
        # The deterministic trigger check fails to link the endpoints (the
        # trigger attaches to a third entity), so the relation carries
        # trigger_not_linking_endpoints.  A single-model judge-ENTAILED
        # proposal may NOT clear it: the conflict must go to REVIEW with
        # judge_verifier_conflict, which routes to independent adjudication.
        conflict_text = (
            "TITLE: Study\nABSTRACT: RESULTS: TP53 was studied. "
            "HCC was associated with progression."
        )
        entities = [entity("TP53", "Gene"), entity("HCC", "Disease")]
        pair_result = BioREDPairClassifier(PairClassifierConfig(mode="active")).classify(
            entities, [], ArticleEvidenceReader().read(conflict_text),
            source_text=conflict_text,
        )
        candidates = pair_result.candidates
        payload = {"decisions": [{
            "candidate_id": candidates[0].candidate_id,
            "predicate": "ASSOCIATED_WITH",
            "evidence_quote": "TP53 was studied. HCC was associated with progression.",
            "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
        }]}
        judge = self.judge_with(payload)
        refined = judge.refine_pair_result(
            pair_result, text=conflict_text, entities=entities, pmid="1",
        )
        self.assertTrue(refined.accepted_relations)
        verified = KGVerifier(OfflineKG()).verify(
            entities, refined.accepted_relations, pmid="1", text=conflict_text,
        )
        relation = verified.relations[0]
        self.assertEqual(relation.semantic_status, "REVIEW")
        self.assertIn("judge_verifier_conflict", relation.quality_flags)
        self.assertFalse(relation.import_ready)

    def test_adjudicator_entailed_clears_semantic_flags_only(self):
        # The independent second-model endorsement (adjudicator_entailed) may
        # clear the overridable semantic flags; the write gate still stands.
        conflict_text = (
            "TITLE: Study\nABSTRACT: RESULTS: TP53 was studied. "
            "HCC was associated with progression."
        )
        entities = [entity("TP53", "Gene"), entity("HCC", "Disease")]
        pair_result = BioREDPairClassifier(PairClassifierConfig(mode="active")).classify(
            entities, [], ArticleEvidenceReader().read(conflict_text),
            source_text=conflict_text,
        )
        payload = {"decisions": [{
            "candidate_id": pair_result.candidates[0].candidate_id,
            "predicate": "ASSOCIATED_WITH",
            "evidence_quote": "TP53 was studied. HCC was associated with progression.",
            "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
        }]}
        judge = self.judge_with(payload)
        refined = judge.refine_pair_result(
            pair_result, text=conflict_text, entities=entities, pmid="1",
        )
        relation = refined.accepted_relations[0]
        relation["quality_flags"] = sorted(
            set(relation["quality_flags"]) | {"adjudicator_entailed"}
        )
        verified = KGVerifier(OfflineKG()).verify(
            entities, [relation], pmid="1", text=conflict_text,
        )
        checked = verified.relations[0]
        self.assertEqual(checked.semantic_status, "ACCEPTED")
        self.assertFalse(checked.import_ready)

    def test_direction_b_to_a_swaps_directional_predicates(self):
        candidate = self.pair_result().candidates[0]
        judge = self.judge_with({})
        relation = judge._as_judge_relation(
            candidate,
            judge._validate({
                "candidate_id": candidate.candidate_id,
                "predicate": "PROGNOSTIC_IN",
                "evidence_quote": "TP53 expression was associated with HCC progression.",
                "direction": "B_TO_A", "decision": ENTAILED, "confidence": 0.8,
            }, candidate, text=self.TEXT, alias_index={}),
            quote_start=0, quote_end=10, section="RESULTS", uncertain=False,
        )
        # PROGNOSTIC_IN is directional: B_TO_A swaps the endpoints.
        self.assertEqual((relation["subject"], relation["object"]), ("HCC", "TP53"))

    def test_directional_verb_alone_does_not_entail_association(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 expression increased in HCC patients."
        entities = [entity("TP53", "Gene"), entity("HCC", "Disease")]
        result = BioREDPairClassifier(PairClassifierConfig(mode="active")).classify(
            entities, [], ArticleEvidenceReader().read(text), source_text=text,
        )
        candidate = result.candidates[0]
        judge = self.judge_with({})
        decision = judge._validate({
            "candidate_id": candidate.candidate_id,
            "predicate": "ASSOCIATED_WITH",
            "evidence_quote": "TP53 expression increased in HCC patients.",
            "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
        }, candidate, text=text, alias_index={})
        # "increased in" is directional, not an association trigger: the
        # decision must be downgraded instead of accepted.
        self.assertIn("judge_entailment_without_trigger_support", decision.reason_codes)
        self.assertEqual(decision.decision, NOT_ENOUGH_INFORMATION)

    def test_low_confidence_judge_relation_blocks_write(self):
        candidate = self.pair_result().candidates[0]
        judge = self.judge_with({})
        decision = judge._validate({
            "candidate_id": candidate.candidate_id,
            "predicate": "ASSOCIATED_WITH",
            "evidence_quote": "TP53 expression was associated with HCC progression.",
            "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.75,
        }, candidate, text=self.TEXT, alias_index={})
        relation = judge._as_judge_relation(
            candidate, decision, quote_start=0, quote_end=10,
            section="RESULTS", uncertain=False,
        )
        self.assertIn("judge_no_write_endorsement", relation["quality_flags"])


class AdjudicatorEntailmentUpgradeTests(unittest.TestCase):
    class FakeCollaboration:
        triggered = True
        review_decisions = [{
            "candidate_id": "r000", "pair_candidate_id": "p-abc",
            "action": "KEEP", "raw_index": 0,
        }]

    def test_keep_upgrades_judge_uncertain_relation(self):
        from cognitive_agent.agent import CognitiveAgent
        relations = [{
            "candidate_id": "p-abc",
            "quality_flags": ["judge_uncertain", "pair_low_confidence", "weak_evidence"],
            "uncertain": True,
            "evidence_entailment": "NOT_ENOUGH_INFORMATION",
        }]
        upgrades = CognitiveAgent._apply_adjudicator_entailment(
            relations, self.FakeCollaboration()
        )
        self.assertEqual(upgrades, 1)
        relation = relations[0]
        self.assertEqual(relation["evidence_entailment"], "ENTAILED")
        self.assertFalse(relation["uncertain"])
        flags = set(relation["quality_flags"])
        self.assertIn("adjudicator_entailed", flags)
        self.assertNotIn("judge_uncertain", flags)
        self.assertNotIn("pair_low_confidence", flags)

    def test_reject_does_not_upgrade(self):
        from cognitive_agent.agent import CognitiveAgent
        collaboration = self.FakeCollaboration()
        collaboration.review_decisions = [{
            "candidate_id": "r000", "pair_candidate_id": "p-abc",
            "action": "REJECT", "raw_index": 0,
        }]
        relations = [{
            "candidate_id": "p-abc",
            "quality_flags": ["judge_uncertain"],
            "uncertain": True,
            "evidence_entailment": "NOT_ENOUGH_INFORMATION",
        }]
        upgrades = CognitiveAgent._apply_adjudicator_entailment(relations, collaboration)
        self.assertEqual(upgrades, 0)
        self.assertEqual(relations[0]["evidence_entailment"], "NOT_ENOUGH_INFORMATION")


class JudgeLatticeIntegrationTests(unittest.TestCase):
    def test_adjacent_sentence_window_recovers_cross_sentence_pair(self):
        text = (
            "TITLE: Study\nABSTRACT: RESULTS: Serum PTX2 decreased in patients. "
            "Fibrotic liver tissue showed higher collagen deposition."
        )
        entities = [entity("PTX2", "Protein"), entity("fibrotic liver tissue", "Tissue")]
        classifier = BioREDPairClassifier(PairClassifierConfig(
            mode="active", include_parent_sentences=True,
            include_adjacent_windows=True,
        ))
        result = classifier.classify(
            entities, [], ArticleEvidenceReader().read(text), source_text=text,
        )
        pairs = {(item.subject, item.object) for item in result.candidates}
        self.assertIn(("PTX2", "fibrotic liver tissue"), pairs)


class ScriptedTwoStageRegistry(AuxModelRegistry):
    """Returns scripted payloads in call order: gate payloads, then predicate
    payloads; repeats the last payload for surplus calls."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0
        super().__init__([AuxModelSpec(
            role="judge", provider="openai", model_id="fake-judge",
            api_base="http://localhost", api_key="test",
        )])

    def configured(self, role):
        return role == "judge"

    def call_json(self, role, *, system_prompt, user_prompt, schema_hint=None):
        from cognitive_agent.aux_model_registry import StructuredModelResult
        index = min(self.calls, len(self.payloads) - 1)
        payload = self.payloads[index]
        self.calls += 1
        return StructuredModelResult(
            role, "fake-judge", "OK", payload=payload,
            latency_s=0.0, prompt_tokens=1, output_tokens=1, attempts=1,
        )


class ClaimGateTests(unittest.TestCase):
    TEXT = (
        "TITLE: Study\nABSTRACT: BACKGROUND: NAFLD is known to involve lipotoxicity. "
        "METHODS: Serum ALT was measured in patients with NAFLD. "
        "RESULTS: We found that TP53 expression was associated with HCC progression."
    )

    def pair_result(self):
        entities = [
            entity("TP53", "Gene"), entity("HCC", "Disease"),
            entity("ALT", "Metabolite"), entity("NAFLD", "Disease"),
            entity("lipotoxicity", "Pathway"),
        ]
        classifier = BioREDPairClassifier(PairClassifierConfig(
            mode="active", include_parent_sentences=True,
        ))
        return classifier.classify(
            entities, [], ArticleEvidenceReader().read(self.TEXT),
            source_text=self.TEXT,
        )

    def candidate_id(self, pair_result, subject, object_):
        for candidate in pair_result.candidates:
            if candidate.subject == subject and candidate.object == object_:
                return candidate.candidate_id
        return ""

    def two_stage_judge(self, gate_payload, predicate_payload):
        return PairwiseJudge(
            PairwiseJudgeConfig(mode="active", claim_gate_enabled=True),
            ScriptedTwoStageRegistry([gate_payload, predicate_payload]),
        )

    def test_gate_blocks_cohort_and_background_pairs(self):
        pair_result = self.pair_result()
        cohort_id = self.candidate_id(pair_result, "ALT", "NAFLD")
        direct_id = self.candidate_id(pair_result, "TP53", "HCC")
        self.assertTrue(cohort_id and direct_id)
        gate_payload = {"decisions": [
            {"candidate_id": cohort_id,
             "relation_asserted": "NOT_ASSERTED", "claim_role": "BACKGROUND",
             "rationale": "measured in patients with"},
            {"candidate_id": direct_id,
             "relation_asserted": "ASSERTED", "claim_role": "CURRENT_FINDING",
             "rationale": "we found"},
        ]}
        predicate_payload = {"decisions": [{
            "candidate_id": direct_id,
            "subject": "TP53", "object": "HCC",
            "predicate": "ASSOCIATED_WITH",
            "evidence_quote": "TP53 expression was associated with HCC progression.",
            "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
        }]}
        judge = self.two_stage_judge(gate_payload, predicate_payload)
        refined = judge.refine_pair_result(
            pair_result, text=self.TEXT,
            entities=[
                entity("TP53", "Gene"), entity("HCC", "Disease"),
                entity("ALT", "Metabolite"), entity("NAFLD", "Disease"),
                entity("lipotoxicity", "Pathway"),
            ],
            pmid="1",
        )
        by_id = {pred.candidate_id: pred for pred in refined.predictions}
        self.assertEqual(by_id[direct_id].label, "ASSOCIATED_WITH")
        self.assertEqual(by_id[cohort_id].label, NO_RELATION)
        self.assertIn("claim_gate_not_asserted", by_id[cohort_id].reason_codes)
        payload = judge.phase_payload(refined)
        self.assertTrue(payload["claim_gate_enabled"])
        self.assertEqual(payload["claim_gate"]["direct_finding_pass"], 1)

    def test_stage_b_preserves_prior_work_claim_role(self):
        pair_result = self.pair_result()
        direct_id = self.candidate_id(pair_result, "TP53", "HCC")
        gate_payload = {"decisions": [{
            "candidate_id": direct_id,
            "relation_asserted": "ASSERTED", "claim_role": "PRIOR_WORK",
            "rationale": "reported by previous studies",
        }]}
        predicate_payload = {"decisions": [{
            "candidate_id": direct_id,
            "subject": "TP53", "object": "HCC",
            "predicate": "ASSOCIATED_WITH",
            "evidence_quote": "TP53 expression was associated with HCC progression.",
            "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
        }]}
        judge = self.two_stage_judge(gate_payload, predicate_payload)
        refined = judge.refine_pair_result(
            pair_result, text=self.TEXT,
            entities=[
                entity("TP53", "Gene"), entity("HCC", "Disease"),
                entity("ALT", "Metabolite"), entity("NAFLD", "Disease"),
                entity("lipotoxicity", "Pathway"),
            ],
            pmid="1",
        )
        relation = next(
            item for item in refined.accepted_relations
            if item["candidate_id"] == direct_id
        )
        self.assertEqual(relation["claim_role"], "PRIOR_WORK")
        self.assertIn("non_current_finding_role", relation["quality_flags"])
        self.assertNotIn("manual_review", relation["quality_flags"])

    def test_model_cannot_promote_deterministic_background_role_to_current(self):
        text = (
            "TITLE: A narrative review\nABSTRACT: BACKGROUND: "
            "TP53 expression was associated with HCC progression."
        )
        entities = [entity("TP53", "Gene"), entity("HCC", "Disease")]
        pair_result = BioREDPairClassifier(PairClassifierConfig(mode="active")).classify(
            entities, [], ArticleEvidenceReader().read(text), source_text=text,
        )
        candidate = pair_result.candidates[0]
        self.assertEqual(candidate.claim_role, "BACKGROUND")
        gate_payload = {"decisions": [{
            "candidate_id": candidate.candidate_id,
            "relation_asserted": "ASSERTED", "claim_role": "CURRENT_FINDING",
        }]}
        predicate_payload = {"decisions": [{
            "candidate_id": candidate.candidate_id,
            "subject": "TP53", "object": "HCC", "predicate": "ASSOCIATED_WITH",
            "evidence_quote": "TP53 expression was associated with HCC progression.",
            "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
        }]}
        refined = self.two_stage_judge(gate_payload, predicate_payload).refine_pair_result(
            pair_result, text=text, entities=entities, pmid="1",
        )
        relation = refined.accepted_relations[0]
        self.assertEqual(relation["claim_role"], "BACKGROUND")
        self.assertIn("claim_role_promotion_blocked", relation["quality_flags"])
        self.assertNotIn("manual_review", relation["quality_flags"])

    def test_gate_only_direct_finding_reaches_predicate_stage(self):
        pair_result = self.pair_result()
        direct_id = self.candidate_id(pair_result, "TP53", "HCC")
        cohort_id = self.candidate_id(pair_result, "ALT", "NAFLD")
        # Predicate payload only covers the ASSERTED pair; the registry
        # verifies the predicate stage only ever sees gated-through pairs.
        gate_payload = {"decisions": [
            {"candidate_id": direct_id,
             "relation_asserted": "ASSERTED", "claim_role": "CURRENT_FINDING",
             "rationale": "we found"},
            {"candidate_id": cohort_id,
             "relation_asserted": "NOT_ASSERTED", "claim_role": "BACKGROUND",
             "rationale": "measured in patients"},
        ]}
        registry = ScriptedTwoStageRegistry([gate_payload, {"decisions": []}])
        judge = PairwiseJudge(
            PairwiseJudgeConfig(mode="active", claim_gate_enabled=True), registry,
        )
        refined = judge.refine_pair_result(
            pair_result, text=self.TEXT,
            entities=[
                entity("TP53", "Gene"), entity("HCC", "Disease"),
                entity("ALT", "Metabolite"), entity("NAFLD", "Disease"),
                entity("lipotoxicity", "Pathway"),
            ],
            pmid="1",
        )
        gate_calls = [
            audit for audit in refined.judge_audits if audit.stage == "gate"
        ]
        predicate_calls = [
            audit for audit in refined.judge_audits if audit.stage == "predicate"
        ]
        self.assertEqual(len(gate_calls), 1)
        self.assertEqual(len(predicate_calls), 1)
        self.assertEqual(predicate_calls[0].pairs_requested, 1)

    def test_gate_only_mode_keeps_local_prediction_for_direct_finding(self):
        pair_result = self.pair_result()
        direct_id = self.candidate_id(pair_result, "TP53", "HCC")
        cohort_id = self.candidate_id(pair_result, "ALT", "NAFLD")
        gate_payload = {"decisions": [
            {"candidate_id": direct_id,
             "relation_asserted": "ASSERTED", "claim_role": "CURRENT_FINDING",
             "rationale": "we found"},
            {"candidate_id": cohort_id,
             "relation_asserted": "NOT_ASSERTED", "claim_role": "BACKGROUND",
             "rationale": "measured in patients"},
        ]}
        # Single-element registry: any predicate-stage call would reuse the
        # gate payload, so the stage audit assertion catches it either way.
        registry = ScriptedTwoStageRegistry([gate_payload])
        judge = PairwiseJudge(
            PairwiseJudgeConfig(
                mode="active", claim_gate_enabled=True,
                predicate_stage_enabled=False,
            ), registry,
        )
        refined = judge.refine_pair_result(
            pair_result, text=self.TEXT,
            entities=[
                entity("TP53", "Gene"), entity("HCC", "Disease"),
                entity("ALT", "Metabolite"), entity("NAFLD", "Disease"),
                entity("lipotoxicity", "Pathway"),
            ],
            pmid="1",
        )
        by_id = {pred.candidate_id: pred for pred in refined.predictions}
        # Gate survivor keeps the local backend prediction (baseline
        # treatment); the cohort pair is still gated out.
        self.assertEqual(by_id[direct_id].label, "ASSOCIATED_WITH")
        self.assertEqual(by_id[cohort_id].label, NO_RELATION)
        self.assertIn("claim_gate_not_asserted", by_id[cohort_id].reason_codes)
        predicate_calls = [
            audit for audit in refined.judge_audits if audit.stage == "predicate"
        ]
        self.assertEqual(predicate_calls, [])

    def test_invalid_gate_status_fails_closed(self):
        candidate = self.pair_result().candidates[0]
        judge = self.two_stage_judge({}, {})
        decision = judge._validate_gate(
            {"candidate_id": candidate.candidate_id, "claim_status": "WHATEVER"},
            candidate,
        )
        self.assertEqual(decision.relation_asserted, "NOT_ASSERTED")
        self.assertEqual(decision.claim_status, "NO_EXPLICIT_RELATION")
        self.assertEqual(decision.label, NO_RELATION)
        self.assertIn("claim_gate_not_asserted", decision.reason_codes)

    def test_v2_gate_fields_derive_consistent_legacy_audit_status(self):
        candidate = self.pair_result().candidates[0]
        judge = self.two_stage_judge({}, {})
        current = judge._validate_gate({
            "candidate_id": candidate.candidate_id,
            "relation_asserted": "ASSERTED", "claim_role": "CURRENT_FINDING",
        }, candidate)
        prior = judge._validate_gate({
            "candidate_id": candidate.candidate_id,
            "relation_asserted": "ASSERTED", "claim_role": "PRIOR_WORK",
        }, candidate)
        self.assertEqual(current.claim_status, "DIRECT_FINDING")
        self.assertEqual(prior.claim_status, "PRIOR_WORK")

    def test_gate_abstention_cannot_erase_specific_grounded_interaction(self):
        text = "TITLE: Liver study\nABSTRACT: RESULTS: TP53 interacted with EGFR."
        entities = [entity("TP53", "Protein"), entity("EGFR", "Protein")]
        pair_result = BioREDPairClassifier(PairClassifierConfig(mode="active")).classify(
            entities, [], ArticleEvidenceReader().read(text), source_text=text,
        )
        candidate = pair_result.candidates[0]
        self.assertEqual(candidate.evidence_entailment, ENTAILED)
        decision = self.two_stage_judge({}, {})._validate_gate({
            "candidate_id": candidate.candidate_id,
            "relation_asserted": "UNCERTAIN", "claim_role": "OTHER",
        }, candidate)
        self.assertEqual(decision.relation_asserted, "ASSERTED")
        self.assertIn("deterministic_explicit_assertion_preserved", decision.reason_codes)

    def test_gate_cannot_demote_explicit_result_interaction_to_method(self):
        text = "TITLE: Liver study\nABSTRACT: CONCLUSION: TP53 interacted with EGFR."
        entities = [entity("TP53", "Protein"), entity("EGFR", "Protein")]
        pair_result = BioREDPairClassifier(PairClassifierConfig(mode="active")).classify(
            entities, [], ArticleEvidenceReader().read(text), source_text=text,
        )
        candidate = pair_result.candidates[0]
        self.assertEqual(candidate.claim_role, "CURRENT_FINDING")
        decision = self.two_stage_judge({}, {})._validate_gate({
            "candidate_id": candidate.candidate_id,
            "relation_asserted": "ASSERTED", "claim_role": "METHOD",
        }, candidate)
        self.assertEqual(decision.claim_role, "CURRENT_FINDING")
        self.assertIn("deterministic_current_role_preserved", decision.reason_codes)

    def test_judge_quote_missing_endpoint_falls_back_and_routes_to_review(self):
        candidate = self.pair_result().candidates[0]
        judge = self.two_stage_judge({}, {})
        decision = judge._validate({
            "candidate_id": candidate.candidate_id,
            "predicate": "ASSOCIATED_WITH",
            "evidence_quote": "associated with HCC progression",
            "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
        }, candidate, text=self.TEXT, alias_index={})
        self.assertEqual(decision.decision, NOT_ENOUGH_INFORMATION)
        self.assertEqual(decision.evidence_quote, candidate.evidence)
        self.assertIn("judge_quote_missing_endpoint", decision.reason_codes)

    def test_final_same_type_association_dedup_prefers_accepted(self):
        accepted = [{
            "subject": "Disease A", "subject_type": "Disease",
            "predicate": "ASSOCIATED_WITH", "object": "Disease B",
            "object_type": "Disease", "relation_probability": 0.8,
        }]
        review = [{
            "subject": "Disease B", "subject_type": "Disease",
            "predicate": "ASSOCIATED_WITH", "object": "Disease A",
            "object_type": "Disease", "relation_probability": 0.9,
        }]
        kept, abstained = PairwiseJudge._deduplicate_selected_relations(accepted, review)
        self.assertEqual(kept, accepted)
        self.assertEqual(abstained, [])

    def test_gate_failure_degrades_to_single_stage(self):
        pair_result = self.pair_result()
        direct_id = self.candidate_id(pair_result, "TP53", "HCC")
        from cognitive_agent.aux_model_registry import StructuredModelResult

        class FailingGateRegistry(ScriptedTwoStageRegistry):
            def call_json(self, role, *, system_prompt, user_prompt, schema_hint=None):
                payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
                self.calls += 1
                if self.calls == 1:
                    return StructuredModelResult(
                        role, "fake-judge", "FALLBACK", payload={},
                        latency_s=0.0, error="gate down",
                    )
                return StructuredModelResult(
                    role, "fake-judge", "OK", payload=payload,
                    latency_s=0.0, prompt_tokens=1, output_tokens=1, attempts=1,
                )

        predicate_payload = {"decisions": [{
            "candidate_id": direct_id,
            "relation_asserted": "ASSERTED", "claim_role": "CURRENT_FINDING",
            "subject": "TP53", "object": "HCC",
            "predicate": "ASSOCIATED_WITH",
            "evidence_quote": "TP53 expression was associated with HCC progression.",
            "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
        }]}
        judge = PairwiseJudge(
            PairwiseJudgeConfig(mode="active", claim_gate_enabled=True),
            FailingGateRegistry([{}, predicate_payload]),
        )
        refined = judge.refine_pair_result(
            pair_result, text=self.TEXT,
            entities=[
                entity("TP53", "Gene"), entity("HCC", "Disease"),
                entity("ALT", "Metabolite"), entity("NAFLD", "Disease"),
                entity("lipotoxicity", "Pathway"),
            ],
            pmid="1",
        )
        by_id = {pred.candidate_id: pred for pred in refined.predictions}
        self.assertEqual(by_id[direct_id].label, "ASSOCIATED_WITH")

    def test_stage_b_endpoint_disagreement_recorded_but_not_applied(self):
        candidate = self.pair_result().candidates[0]
        judge = self.two_stage_judge({}, {})
        decision = judge._validate(
            {
                "candidate_id": candidate.candidate_id,
                "subject": "SOMETHING ELSE", "object": "HCC",
                "predicate": "ASSOCIATED_WITH",
                "evidence_quote": "TP53 expression was associated with HCC progression.",
                "direction": "A_TO_B", "decision": ENTAILED, "confidence": 0.9,
            },
            candidate, text=self.TEXT, alias_index={},
        )
        self.assertIn("judge_endpoint_disagreement_subject", decision.reason_codes)
        self.assertEqual(decision.label, "ASSOCIATED_WITH")


if __name__ == "__main__":
    unittest.main()
