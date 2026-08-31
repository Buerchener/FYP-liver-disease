import unittest

from cognitive_agent.central_agent_v2 import CentralAgentV2


class CentralAgentV2Tests(unittest.TestCase):
    def test_quality_budgets_are_route_adaptive_and_wide(self):
        controller = CentralAgentV2(
            execution_mode="agent-v2-shadow", budget_profile="quality"
        )
        fast = controller.start("1", "FAST")
        standard = controller.start("2", "STANDARD")
        deep = controller.start("3", "DEEP")

        self.assertEqual(fast.budget.max_aux_remote_calls, 2)
        self.assertEqual(standard.budget.max_aux_remote_calls, 4)
        self.assertEqual(deep.budget.max_aux_remote_calls, 6)
        self.assertEqual(deep.budget.hard_max_aux_remote_calls, 8)

    def test_budget_escalates_only_for_unresolved_work(self):
        controller = CentralAgentV2(execution_mode="agent-v2")
        state = controller.start("1", "FAST")
        self.assertFalse(controller.maybe_escalate(state, "nothing_to_fix"))
        state.unresolved_issues = ["reviewable_relations"]
        self.assertTrue(controller.maybe_escalate(state, "high_value_relation"))
        self.assertEqual(state.route, "STANDARD")
        self.assertEqual(len(state.budget_escalations), 1)

    def test_no_change_outcomes_do_not_change_logical_route(self):
        controller = CentralAgentV2(execution_mode="agent-v2")
        state = controller.start("1", "DEEP")
        state.review_relations = [{"subject": "TP53"}]
        for _ in range(2):
            controller.record_action(
                state, tool="second_llm_refiner", decision="CALL",
                reason="test", before="same", after="same", remote=True,
                result_status="OK",
            )
        allowed, reason = controller.can_call_remote(state, "second_llm_refiner")
        self.assertTrue(allowed)
        self.assertIn("within_remote_budget", reason)

    def test_cache_hit_does_not_consume_remote_budget(self):
        controller = CentralAgentV2(execution_mode="agent-v2")
        state = controller.start("1", "FAST")
        controller.record_action(
            state, tool="second_llm_refiner", decision="CALL",
            reason="warm replay", before="a", after="b", remote=True,
            cache_status="persistent_hit", result_status="OK",
        )
        self.assertEqual(state.aux_remote_calls, 0)
        self.assertEqual(state.cache["remote_calls_avoided"], 1)

    def test_cache_key_excludes_secret_fields(self):
        first = CentralAgentV2.tool_cache_key(
            "review", {
                "api_key": "secret-a", "model_id": "m", "text": "x",
                "provider": {"access_key": "nested-a"},
            }
        )
        second = CentralAgentV2.tool_cache_key(
            "review", {
                "api_key": "secret-b", "model_id": "m", "text": "x",
                "provider": {"access_key": "nested-b"},
            }
        )
        self.assertEqual(first, second)

    def test_neo4j_respects_action_and_hard_time_budgets(self):
        controller = CentralAgentV2(
            execution_mode="agent-v2", max_actions=1, hard_timeout_s=180,
        )
        state = controller.start("1", "FAST")
        controller.record_action(
            state, tool="article_preprocessing", decision="CALL", reason="test",
        )
        self.assertEqual(controller.can_call_neo4j(state)[1], "action_budget_exhausted")
        state.action_trace.clear()
        state.started_at -= 181
        self.assertEqual(controller.can_call_neo4j(state)[1], "hard_timeout_reached")

    def test_observation_partitions_hard_blocks_from_reviewable_relations(self):
        controller = CentralAgentV2(execution_mode="agent-v2-shadow")
        state = controller.start("1", "STANDARD")
        verification = {
            "entities": [],
            "relations": [
                {"subject": "A", "import_ready": True, "quality_flags": []},
                {"subject": "B", "import_ready": False, "quality_flags": ["schema_mismatch"]},
                {"subject": "C", "import_ready": False, "quality_flags": ["weak_evidence"]},
            ],
        }
        controller.observe(
            state, entities=[], candidate_pairs=[], verification=verification
        )
        self.assertEqual(len(state.accepted_relations), 1)
        self.assertEqual(len(state.rejected_relations), 1)
        self.assertEqual(len(state.review_relations), 1)

    def test_causal_requires_two_import_ready_relations(self):
        controller = CentralAgentV2(execution_mode="agent-v2")
        state = controller.start("1", "DEEP")
        state.accepted_relations = [{"subject": "A"}]
        self.assertFalse(controller.should_run_causal(state)[0])
        state.accepted_relations.append({"subject": "B"})
        self.assertTrue(controller.should_run_causal(state)[0])


if __name__ == "__main__":
    unittest.main()
