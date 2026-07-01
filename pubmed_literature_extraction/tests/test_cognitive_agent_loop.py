import unittest

from cognitive_agent.agent import CognitiveAgent
from cognitive_agent.context_activator import ContextCard
from cognitive_agent.conflict_resolver import ResolutionItem, ResolutionResult
from cognitive_agent.decision_engine import Action, DecisionEngine, ExecutionLog
from cognitive_agent.schema.examples import ALL_EXAMPLES, DEFAULT_EXAMPLES, KG_EXTRACTION_PROMPT
from cognitive_agent.strategy_manager import StrategyManager
from cognitive_agent.verifier import VerifiedEntity, VerifiedRelation


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
    def test_strategy_manager_switches_low_coverage_to_exploratory(self):
        manager = StrategyManager()
        card = ContextCard(
            coverage_score=0.1,
            extraction_goals=["exploratory_extraction"],
        )

        strategy = manager.get_strategy(card)

        self.assertEqual(strategy["extraction_mode"], "exploratory")
        self.assertLess(strategy["entity_confidence_threshold"], 0.7)

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

    def test_create_with_flag_requires_aggressive_mode(self):
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

        self.assertEqual(balanced.actions[0].type, "NO_ACTION")
        self.assertEqual(aggressive.actions[0].type, "CREATE_RELATION")

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


if __name__ == "__main__":
    unittest.main()
