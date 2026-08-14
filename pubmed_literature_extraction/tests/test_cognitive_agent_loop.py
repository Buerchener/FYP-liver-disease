import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from cognitive_agent.agent import AgentConfig, CognitiveAgent
from cognitive_agent.context_activator import ContextCard
from cognitive_agent.memory.working_memory import WorkingMemory
from cognitive_agent.reviewer import ExtractionReviewer, ReviewerConfig
from cognitive_agent.conflict_resolver import ResolutionItem, ResolutionResult
from cognitive_agent.decision_engine import Action, DecisionEngine, ExecutionLog
from cognitive_agent.schema.examples import ALL_EXAMPLES, DEFAULT_EXAMPLES, KG_EXTRACTION_PROMPT
from cognitive_agent.strategy_manager import StrategyManager
from cognitive_agent.verifier import KGVerifier, VerifiedEntity, VerifiedRelation


class FakeKGMemory:
    GENERIC_TERM_BLACKLIST = set()

    def __init__(self):
        self.updated = []
        self.disputed = []

    def check_relation_exists(self, subject_name, predicate, object_name):
        return {"rel_element_id": "rel-1"}

    def update_relation(self, rel_element_id, new_evidence, new_confidence, pmid=""):
        self.updated.append((rel_element_id, new_evidence, new_confidence, pmid))
        return True

    def mark_disputed(self, rel_element_id, dispute_reason, conflicting_evidence, pmid=""):
        self.disputed.append((rel_element_id, dispute_reason, conflicting_evidence, pmid))
        return True


class CognitiveAgentLoopTests(unittest.TestCase):
    def test_low_kg_coverage_does_not_lower_medical_evidence_thresholds(self):
        manager = StrategyManager()
        card = ContextCard(
            coverage_score=0.1,
            extraction_goals=["exploratory_extraction"],
        )

        strategy = manager.get_strategy(card)

        self.assertEqual(strategy["extraction_mode"], "balanced")
        self.assertEqual(strategy["entity_confidence_threshold"], 0.7)

    def test_agent_strategy_selects_extended_examples_and_prompt(self):
        agent = object.__new__(CognitiveAgent)
        agent.current_examples = list(DEFAULT_EXAMPLES)
        strategy = {
            "extraction_mode": "exploratory",
            "use_extended_examples": False,
        }
        card = ContextCard(
            coverage_score=0.1,
            extraction_goals=["novel_entity_discovery"],
        )

        examples = agent._select_examples(strategy)
        prompt = agent._build_strategy_prompt(KG_EXTRACTION_PROMPT, card, strategy)

        self.assertEqual(len(examples), len(ALL_EXAMPLES))
        self.assertIn("extraction_mode: exploratory", prompt)
        self.assertIn("novel_entity_discovery", prompt)

    def test_strategy_threshold_changes_entity_decision(self):
        engine = DecisionEngine(FakeKGMemory(), skip_neo4j_write=True)
        entity = VerifiedEntity(
            mention="rare liver syndrome",
            entity_type="Disease",
            neo4j_status="NOVEL",
            confidence=0.55,
        )

        default_log = engine.decide([entity], [], pmid="1")
        exploratory_log = engine.decide(
            [entity],
            [],
            pmid="1",
            strategy={
                "entity_confidence_threshold": 0.6,
                "extraction_mode": "exploratory",
            },
        )

        self.assertEqual(default_log.actions[0].type, "NO_ACTION")
        self.assertEqual(exploratory_log.actions[0].type, "CREATE_ENTITY")

    def test_conflict_resolution_create_drives_relation_decision(self):
        engine = DecisionEngine(FakeKGMemory(), skip_neo4j_write=True)
        relation = VerifiedRelation(
            subject="TP53",
            predicate="ASSOCIATED_WITH",
            object="HCC",
            subject_type="Gene",
            object_type="Disease",
            neo4j_status="NOVEL",
            schema_valid=True,
            import_ready=True,
            evidence="TP53 mutations are associated with HCC.",
        )
        resolution = ResolutionResult(items=[
            ResolutionItem(
                subject="TP53",
                predicate="ASSOCIATED_WITH",
                object="HCC",
                decision="CREATE",
                adjusted_confidence=0.8,
                reasoning_trace="Novel relation, high confidence",
            )
        ])

        log = engine.decide(
            [],
            [relation],
            pmid="1",
            strategy={"relation_confidence_threshold": 0.7},
            conflict_resolution=resolution,
        )

        self.assertEqual(log.actions[0].type, "CREATE_RELATION")
        self.assertIn("Novel relation", log.actions[0].reason)

    def test_create_with_flag_cannot_bypass_import_ready_gate(self):
        engine = DecisionEngine(FakeKGMemory(), skip_neo4j_write=True)
        relation = VerifiedRelation(
            subject="TP53",
            predicate="ASSOCIATED_WITH",
            object="HCC",
            subject_type="Gene",
            object_type="Disease",
            neo4j_status="NOVEL",
            schema_valid=True,
            import_ready=False,
            evidence="TP53 may be associated with HCC.",
        )
        resolution = ResolutionResult(items=[
            ResolutionItem(
                subject="TP53",
                predicate="ASSOCIATED_WITH",
                object="HCC",
                decision="CREATE_WITH_FLAG",
                adjusted_confidence=0.5,
                reasoning_trace="Moderate confidence",
            )
        ])

        balanced = engine.decide(
            [],
            [relation],
            strategy={"conflict_resolution_mode": "balanced"},
            conflict_resolution=resolution,
        )
        aggressive = engine.decide(
            [],
            [relation],
            strategy={"conflict_resolution_mode": "aggressive"},
            conflict_resolution=resolution,
        )

        self.assertEqual(balanced.actions[0].type, "DISCARD")
        self.assertEqual(aggressive.actions[0].type, "DISCARD")

    def test_entity_creation_can_be_restricted_to_import_ready_endpoints(self):
        kg = FakeKGMemory()
        engine = DecisionEngine(kg, skip_neo4j_write=True)
        standalone = VerifiedEntity(
            mention="novel tissue", entity_type="Tissue",
            neo4j_status="NOVEL", confidence=0.9,
        )
        without_relation = engine.decide(
            [standalone], [],
            restrict_entities_to_import_ready_endpoints=True,
        )
        self.assertEqual(without_relation.actions, [])

        endpoint = VerifiedEntity(
            mention="liver", entity_type="Tissue",
            neo4j_status="NOVEL", confidence=0.9,
        )
        disease = VerifiedEntity(
            mention="HCC", entity_type="Disease",
            neo4j_status="NOVEL", confidence=0.9,
        )
        verified_relation = VerifiedRelation(
            subject="HCC", subject_type="Disease",
            predicate="ASSOCIATED_WITH", object="liver", object_type="Tissue",
            neo4j_status="NOVEL", schema_valid=True, import_ready=True,
            evidence_level=1,
        )
        with_relation = engine.decide(
            [endpoint, disease], [verified_relation],
            restrict_entities_to_import_ready_endpoints=True,
        )
        entity_actions = [action for action in with_relation.actions if action.entity]
        self.assertEqual(len(entity_actions), 2)
        self.assertTrue(all(action.type == "CREATE_ENTITY" for action in entity_actions))

        moderate = type(verified_relation)(**{
            **verified_relation.__dict__,
            "quality_flags": [],
        })
        resolution = {
            "items": [{
                "subject": "HCC",
                "predicate": "ASSOCIATED_WITH",
                "object": "liver",
                "decision": "CREATE_WITH_FLAG",
                "adjusted_confidence": 0.5,
            }]
        }
        no_relation_write = engine.decide(
            [endpoint, disease], [moderate],
            conflict_resolution=resolution,
            strategy={"conflict_resolution_mode": "balanced"},
            restrict_entities_to_import_ready_endpoints=True,
        )
        self.assertFalse(any(
            action.type == "CREATE_ENTITY" for action in no_relation_write.actions
        ))

    def test_neutral_entity_threshold_does_not_reject_exact_point_seven(self):
        kg = FakeKGMemory()
        engine = DecisionEngine(kg, skip_neo4j_write=True)
        candidate = VerifiedEntity(
            mention="liver", entity_type="Tissue",
            neo4j_status="NOVEL", confidence=0.7,
        )
        action = engine._decide_entity(
            candidate,
            pmid="1",
            strategy={"entity_confidence_threshold": 0.7},
        )
        self.assertEqual(action.type, "CREATE_ENTITY")

    def test_duplicate_relation_triples_are_decided_once(self):
        kg = FakeKGMemory()
        engine = DecisionEngine(kg, skip_neo4j_write=True)
        relation_a = VerifiedRelation(
            subject="CCND1", subject_type="Gene",
            predicate="ASSOCIATED_WITH",
            object="HBV-related liver fibrosis", object_type="Disease",
            neo4j_status="NOVEL", schema_valid=True, import_ready=True,
            evidence="CCND1 is associated with HBV-related liver fibrosis.",
            evidence_level=1,
        )
        relation_b = VerifiedRelation(**{
            **relation_a.__dict__,
            "evidence": "CCND1 was associated with HBV-related liver fibrosis.",
        })
        log = engine.decide([], [relation_a, relation_b])
        self.assertEqual(
            sum(action.type == "CREATE_RELATION" for action in log.actions),
            1,
        )

    def test_execute_applies_update_and_dispute_actions(self):
        kg = FakeKGMemory()
        engine = DecisionEngine(kg, skip_neo4j_write=False)
        log = ExecutionLog(
            pmid="123",
            actions=[
                Action(
                    type="UPDATE_RELATION",
                    relation={
                        "subject": "TP53",
                        "predicate": "ASSOCIATED_WITH",
                        "object": "HCC",
                        "evidence": "new evidence",
                    },
                    confidence=0.9,
                ),
                Action(
                    type="MARK_DISPUTED",
                    relation={
                        "subject": "TP53",
                        "predicate": "ASSOCIATED_WITH",
                        "object": "HCC",
                        "evidence": "conflicting evidence",
                    },
                    reason="conflict",
                    confidence=0.5,
                ),
            ],
        )

        engine.execute(log)

        self.assertEqual(len(kg.updated), 1)
        self.assertEqual(len(kg.disputed), 1)
        self.assertEqual(log.relations_updated, 1)
        self.assertEqual(log.disputed, 1)
    def test_verifier_reads_and_clamps_entity_confidence(self):
        kg = type("DisconnectedKG", (), {"is_connected": False})()
        verifier = KGVerifier(kg)

        high = verifier.verify(
            [{"mention": "TP53", "type": "Gene", "confidence": 1.8}], [], pmid="1"
        )
        low = verifier.verify(
            [{"mention": "HCC", "type": "Disease", "confidence": -0.2}], [], pmid="2"
        )

        self.assertEqual(high.entities[0].confidence, 1.0)
        self.assertEqual(low.entities[0].confidence, 0.0)

    def test_working_memory_instances_are_article_local(self):
        memories = {}

        def worker(pmid):
            memory = WorkingMemory(current_article_pmid=pmid)
            memory.extraction_targets.append(pmid)
            memories[pmid] = memory

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, ["pmid-a", "pmid-b"]))

        self.assertIsNot(memories["pmid-a"], memories["pmid-b"])
        self.assertEqual(memories["pmid-a"].extraction_targets, ["pmid-a"])
        self.assertEqual(memories["pmid-b"].extraction_targets, ["pmid-b"])
    def test_entity_candidate_scoring_rejects_short_substring(self):
        from cognitive_agent.memory.kg_memory import KGMemory

        self.assertEqual(KGMemory.score_entity_candidate("p53", "TP53")[0], 0.0)
        score, kind = KGMemory.score_entity_candidate(
            "hepatocellular carcinoma", "Hepatocellular-Carcinoma"
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(kind, "exact_normalized")
        self.assertEqual(
            KGMemory.score_entity_candidate("HBV-related liver fibrosis", "HBV")[0],
            0.0,
        )
        self.assertEqual(
            KGMemory.score_entity_candidate("HBV-related liver fibrosis", "fibrosis")[0],
            0.0,
        )

    def test_ambiguous_entity_is_not_created(self):
        kg = type("AmbiguousKG", (), {"GENERIC_TERM_BLACKLIST": set(), "is_connected": True, "find_entity": lambda *a, **k: None,
            "find_entity_fuzzy": lambda *a, **k: {
                "ambiguous": True,
                "fuzzy_matches": [
                    {"name": "alpha disease", "node_id": "a", "score": 0.8, "match_kind": "contains"},
                    {"name": "alpha disorder", "node_id": "b", "score": 0.77, "match_kind": "contains"},
                ],
            }})()
        verified = KGVerifier(kg).verify(
            [{"mention": "alpha", "type": "Disease", "confidence": 0.95}], [], pmid="1"
        )
        self.assertEqual(verified.entities[0].neo4j_status, "AMBIGUOUS")
        log = DecisionEngine(kg, skip_neo4j_write=True).decide(verified.entities, [])
        self.assertEqual(log.actions[0].type, "NO_ACTION")

    def test_conflicting_relation_directions_are_detected(self):
        class RelationKG:
            GENERIC_TERM_BLACKLIST = set()
            is_connected = True
            def check_relation_exists(self, subject, predicate, object_name, **kwargs):
                if subject == "TP53" and object_name == "HCC":
                    return {"rel_element_id": "r1", "confidence": 0.9, "direction": "increase"}
                return None
        relation = {
            "subject": "TP53", "predicate": "ASSOCIATED_WITH", "object": "HCC",
            "subject_type": "Gene", "object_type": "Disease",
            "direction": "decrease", "evidence": "opposite result",
        }
        result = KGVerifier(RelationKG()).verify([], [relation], pmid="1")
        self.assertEqual(result.relations[0].neo4j_status, "CONTRADICTING")
        self.assertIn("opposite_direction", result.relations[0].quality_flags)

    def test_inverted_relation_is_not_auto_created(self):
        class InverseKG:
            GENERIC_TERM_BLACKLIST = set()
            is_connected = True
            def check_relation_exists(self, subject, predicate, object_name, **kwargs):
                if subject == "HCC" and object_name == "TP53":
                    return {"rel_element_id": "r2", "confidence": 0.8, "direction": "positive"}
                return None
        relation = VerifiedRelation(
            subject="TP53", predicate="ASSOCIATED_WITH", object="HCC",
            subject_type="Gene", object_type="Disease", schema_valid=True,
            import_ready=True,
        )
        relation.direction = "positive"
        relation.neo4j_status = "INVERTED"
        log = DecisionEngine(InverseKG(), skip_neo4j_write=True).decide([], [relation])
        self.assertEqual(log.actions[0].type, "NO_ACTION")

        config = AgentConfig(api_key="", reviewer_enabled=False)
        with patch("cognitive_agent.agent.KGMemory"):
            agent = CognitiveAgent(config)
        self.assertFalse(agent.config.reviewer_enabled)
        self.assertFalse(agent.reviewer.enabled)

    def test_openai_compatible_gemini_endpoint_selects_openai_langextract_provider(self):
        config = AgentConfig(
            api_key="unused",
            api_base="https://api-666.cc/v1",
            model_id="count.gmcli-gemini-3-flash-preview",
        )
        with patch("cognitive_agent.agent.KGMemory"):
            agent = CognitiveAgent(config)
        self.assertEqual(agent.extraction_kernel.model_config.provider, "openai")
        self.assertEqual(
            agent.extraction_kernel.model_config.provider_kwargs["base_url"],
            "https://api-666.cc/v1",
        )

    def test_extraction_inner_worker_setting_reaches_kernel(self):
        config = AgentConfig(api_key="unused", extraction_inner_max_workers=1)
        with patch("cognitive_agent.agent.KGMemory"):
            agent = CognitiveAgent(config)
        self.assertEqual(agent.extraction_kernel.inner_max_workers, 1)

    def test_reviewer_flag_is_recorded_only_when_enabled(self):
        config = AgentConfig(api_key="", reviewer_enabled=True, reviewer_model_id="review-model")
        with patch("cognitive_agent.agent.KGMemory"):
            agent = CognitiveAgent(config)
        self.assertTrue(agent.config.reviewer_enabled)
        self.assertEqual(agent.reviewer.config.model_id, "review-model")

        reviewer = ExtractionReviewer(
            ReviewerConfig(model_id="review-model"),
            generate=lambda prompt: '''```json
{"overall_score": 1.4, "decision": "accept", "hallucination_flags": ["none"]}
```''',
        )

        result = reviewer.review("TP53 is associated with HCC.", {"relations": []})

        self.assertEqual(result.status, "OK")
        self.assertEqual(result.decision, "ACCEPT")
        self.assertEqual(result.overall_score, 1.0)
        self.assertEqual(result.hallucination_flags, ["none"])

    def test_reviewer_failure_falls_back_to_manual_review(self):
        reviewer = ExtractionReviewer(
            ReviewerConfig(model_id="review-model"),
            generate=lambda prompt: "not-json",
        )

        result = reviewer.review("text", {})

        self.assertEqual(result.status, "FALLBACK")
        self.assertEqual(result.decision, "REVIEW")
        self.assertTrue(result.error)

    def test_reviewer_is_disabled_without_model_configuration(self):
        result = ExtractionReviewer().review("text", {})

        self.assertEqual(result.status, "DISABLED")
        self.assertEqual(result.decision, "REVIEW")


if __name__ == "__main__":
    unittest.main()
