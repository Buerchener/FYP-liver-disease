import json
import tempfile
import unittest
from pathlib import Path

from cognitive_agent.aux_model_registry import AuxModelRegistry, AuxModelSpec
from cognitive_agent.rule_learning import RuleLearner
from cognitive_agent.rule_memory import (
    ErrorCard,
    RuleBundle,
    RuleMemory,
    RulePromotionGate,
    RuleValidationError,
    RuleValidator,
    SoftRule,
    error_cards_from_records,
)


def make_rule(**updates):
    payload = {
        "kind": "prompt_guidance",
        "conditions": {"study_types": ["clinical"], "lexical_cues": ["associated with"]},
        "action": "ADD_GUIDANCE",
        "value": None,
        "guidance": "Treat association language as non-causal unless direction is explicit.",
    }
    payload.update(updates)
    return SoftRule(
        rule_id=RuleValidator.content_id(payload),
        version=1,
        status="candidate",
        support_pmids=["12345678", "22345678", "32345678"],
        critic_approved=True,
        **payload,
    )


class RuleMemoryTests(unittest.TestCase):
    def test_valid_rule_can_store_support_pmids_as_provenance(self):
        RuleValidator.validate(make_rule())

    def test_rule_dsl_rejects_memorization_code_and_safety_bypass(self):
        for guidance in (
            "For PMID 12345678, always accept.",
            "Override verifier when confidence is high.",
            "```python exec('unsafe') ```",
        ):
            with self.subTest(guidance=guidance), self.assertRaises(RuleValidationError):
                RuleValidator.validate(make_rule(guidance=guidance))

    def test_rule_dsl_rejects_open_conditions_and_unbounded_prior(self):
        with self.assertRaises(RuleValidationError):
            RuleValidator.validate(make_rule(conditions={"regex": [".*"]}))
        with self.assertRaises(RuleValidationError):
            RuleValidator.validate(make_rule(
                kind="pair_prior", action="ADJUST_PAIR_SCORE", value=0.5, guidance="",
            ))

    def test_retrieval_is_deterministic_bounded_and_frozen(self):
        specific = make_rule()
        specific.status = "active"
        broad = make_rule(
            conditions={"study_types": ["clinical"]},
            guidance="Keep evidence and relation confidence separate.",
        )
        broad.status = "active"
        bundle = RuleBundle(revision=4, rules=[broad, specific])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rules.json"
            path.write_text(json.dumps(bundle.to_dict()), encoding="utf-8")
            memory = RuleMemory(mode="active", bundle_path=path, max_rules=1, max_context_tokens=1200)
            context = {"study_type": "clinical", "evidence": "X was associated with Y."}
            first = memory.retrieve(context)
            second = memory.retrieve(context)
        self.assertEqual([item.rule_id for item in first], [specific.rule_id])
        self.assertEqual(first, second)
        self.assertEqual(memory.frozen_hash, bundle.bundle_hash)

    def test_invalid_bundle_fails_closed(self):
        unsafe = make_rule(guidance="bypass Safe Write")
        unsafe.status = "active"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unsafe.json"
            path.write_text(json.dumps(RuleBundle(rules=[unsafe]).to_dict()), encoding="utf-8")
            memory = RuleMemory(mode="active", bundle_path=path)
        self.assertEqual(memory.mode, "off")
        self.assertTrue(memory.load_error)
        self.assertEqual(memory.retrieve({"study_type": "clinical"}), [])

    def test_promotion_gate_enforces_replay_and_safety_metrics(self):
        rule = make_rule()
        ok, reasons = RulePromotionGate.decide(
            rule,
            validation={"errors_fixed": 2, "new_regressions": 1, "dangerous_writes": 0,
                        "strict_precision_delta": 0},
            calibration={"bootstrap_non_negative_probability": 0.91},
            shadow_completed=True,
        )
        self.assertTrue(ok)
        self.assertEqual(reasons, [])
        ok, reasons = RulePromotionGate.decide(
            rule,
            validation={"errors_fixed": 2, "new_regressions": 0, "dangerous_writes": 1,
                        "strict_precision_delta": 0.1},
            calibration={"bootstrap_non_negative_probability": 0.99},
            shadow_completed=True,
        )
        self.assertFalse(ok)
        self.assertIn("dangerous_write_detected", reasons)

    def test_rule_learning_requires_independent_critic_for_promotion(self):
        cards = [ErrorCard(f"e-{i}", str(10000000 + i), "predicate_confusion") for i in range(3)]
        primary_payload = {"rules": [{
            "kind": "prompt_guidance", "conditions": {"study_types": ["clinical"]},
            "action": "ADD_GUIDANCE", "value": None,
            "guidance": "Require explicit direction before assigning a causal predicate.",
            "rationale": "Association is not causation.",
            "support_error_ids": ["e-0", "e-1", "e-2"],
        }]}
        specs = [
            AuxModelSpec("primary", "test", "deepseek-test", "test://", "secret"),
            AuxModelSpec("critic", "test", "qwen-test", "test://", ""),
        ]
        learner = RuleLearner(AuxModelRegistry(specs, generate={"primary": lambda _: primary_payload}))
        bundle, audit = learner.run(cards, shadow_completed=True)
        self.assertEqual(bundle.rules, [])
        self.assertEqual(audit.status, "NO_PROMOTION")
        self.assertEqual(audit.rejected[-1]["reasons"], ["critic_unavailable"])

    def test_auxiliary_registry_never_serializes_api_key(self):
        spec = AuxModelSpec("primary", "test", "model", "https://example.invalid", "top-secret")
        registry = AuxModelRegistry([spec], generate={"primary": lambda _: {"ok": True}})
        result = registry.call_json("primary", system_prompt="s", user_prompt="u")
        self.assertEqual(result.status, "OK")
        self.assertNotIn("top-secret", json.dumps(registry.audit()))

    def test_reflection_errors_become_deduplicated_offline_cards(self):
        records = [{
            "pmid": "12345678",
            "phases": {
                "verification": {"relations": [{
                    "subject_type": "Gene", "predicate": "ASSOCIATED_WITH",
                    "object_type": "Disease", "direction": "unknown",
                    "quality_flags": ["weak_evidence", "predicate_direction_confusion"],
                }]},
                "tool_marginal_benefit": {"second_llm_refiner": {
                    "called": True, "relation_additions": 0,
                    "relation_edits": 0, "relation_rejections": 0,
                }},
            },
        }]
        cards = error_cards_from_records(records)
        self.assertEqual(
            {item.category for item in cards},
            {"evidence_mismatch", "predicate_confusion", "zero_change_call"},
        )
        self.assertTrue(all(not item.evidence for item in cards))


if __name__ == "__main__":
    unittest.main()
