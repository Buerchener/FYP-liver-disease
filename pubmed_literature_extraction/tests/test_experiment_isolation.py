import tempfile
import unittest
from pathlib import Path

from cognitive_agent.agent import bind_pairwise_gate_to_hints, resolve_feature_toggle
from scripts.run_tiered_v2_sentinel10 import (
    aggregate_remote_usage,
    arm_flags,
    neo4j_mutation_count,
    provider_failure_count,
)
from cognitive_agent.agent import build_candidate_audit_ledger


class ExperimentIsolationTests(unittest.TestCase):
    def test_explicit_disable_overrides_enabled_environment(self):
        self.assertFalse(
            resolve_feature_toggle(
                cli_enable=False,
                cli_disable=True,
                env_value="true",
            )
        )

    def test_cli_enable_and_environment_enable_remain_supported(self):
        self.assertTrue(
            resolve_feature_toggle(
                cli_enable=True,
                cli_disable=False,
                env_value="",
            )
        )
        self.assertTrue(
            resolve_feature_toggle(
                cli_enable=False,
                cli_disable=False,
                env_value="yes",
            )
        )

    def test_sentinel_legacy_arm_isolates_second_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flags = arm_flags(
                "legacy",
                frozen=root / "candidates.jsonl",
                cache_path=root / "cache.sqlite3",
            )
        self.assertIn("--disable-second-llm", flags)
        self.assertNotIn("--second-llm-enabled", flags)

    def test_sentinel_counts_provider_failures_and_observed_mutations(self):
        records = [{"phases": {
            "agent_v2": {"remote_usage": {"failed": 2}},
            "execution": {
                "dry_run": False,
                "entities_written": 1,
                "relations_written": 2,
                "relations_updated_in_neo4j": 3,
            },
        }}]
        self.assertEqual(provider_failure_count(records), 2)
        self.assertEqual(neo4j_mutation_count(records), 6)

    def test_sentinel_does_not_count_dry_run_proposals_as_writes(self):
        records = [{"phases": {"execution": {
            "dry_run": True,
            "entities_created": 2,
            "relations_created": 2,
            "entities_written": 0,
            "relations_written": 0,
            "relations_updated_in_neo4j": 0,
        }}}]
        self.assertEqual(neo4j_mutation_count(records), 0)

    def test_candidate_audit_accounts_for_kept_and_factual_discarded_hints(self):
        projected = [
            {"candidate_id": "c-kept", "candidate_lane": "extracted_hint"},
            {"candidate_id": "c-bad", "candidate_lane": "extracted_hint"},
        ]
        ledger = build_candidate_audit_ledger(
            projected_relations=projected,
            core_relations=projected,
            final_relations=[{
                "candidate_id": "c-kept", "candidate_version": 1,
                "factual_status": "VALID", "semantic_status": "REVIEW",
                "write_status": "HUMAN_REVIEW",
            }],
            finalization_audit={"passes": [{"factual_discards": [{
                "candidate_id": "c-bad", "reason_codes": ["schema_invalid"],
            }]}]},
        )
        by_id = {row["candidate_id"]: row for row in ledger["rows"]}
        self.assertEqual(by_id["c-bad"]["disposition"], "FACTUAL_DISCARD")
        self.assertEqual(ledger["summary"]["audit_lineage_accounting"], 1.0)
        self.assertEqual(ledger["summary"]["eligible_lineage_survival"], 1.0)

    def test_pairwise_roles_bind_by_candidate_id_after_array_reorder(self):
        hints = [
            {"candidate_id": "c-current", "claim_role": "CURRENT_FINDING"},
            {"candidate_id": "c-background", "claim_role": "CURRENT_FINDING"},
        ]
        bound = bind_pairwise_gate_to_hints(
            list(reversed(hints)),
            pair_candidates=[
                {"candidate_id": "c-current", "pair_candidate_id": "p-1"},
                {"candidate_id": "c-background", "pair_candidate_id": "p-2"},
            ],
            pair_predictions=[
                {"candidate_id": "c-current", "label": "ASSOCIATED_WITH"},
                {"candidate_id": "c-background", "label": "NO_RELATION"},
            ],
            gate_table=[{
                "candidate_id": "c-background",
                "source_candidate_ids": ["c-background"],
                "relation_asserted": "NOT_ASSERTED",
                "claim_role": "BACKGROUND",
            }],
        )
        by_id = {item["candidate_id"]: item for item in bound}
        self.assertEqual(by_id["c-current"]["claim_role"], "CURRENT_FINDING")
        self.assertEqual(by_id["c-background"]["claim_role"], "BACKGROUND")
        self.assertIn(
            "pair_no_relation_dissent", by_id["c-background"]["quality_flags"]
        )

    def test_remote_usage_uses_agent_v2_stages_without_double_counting(self):
        records = [{"phases": {"agent_v2": {"remote_usage": {
            "pairwise_judge": {
                "attempted": 2, "successful": 2, "prompt_tokens": 20,
                "output_tokens": 4, "latency_s": 2.0,
            },
            "second_llm_refiner": {
                "attempted": 1, "successful": 1, "prompt_tokens": 30,
                "output_tokens": 5, "latency_s": 3.0,
            },
        }}}}]
        usage = aggregate_remote_usage(records)
        self.assertEqual(usage["pairwise_judge"]["attempted"], 2)
        self.assertEqual(usage["deepseek_adjudicator"]["attempted"], 1)
        self.assertEqual(usage["qwen_critic"]["attempted"], 0)


if __name__ == "__main__":
    unittest.main()
