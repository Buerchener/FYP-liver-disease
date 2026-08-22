#!/usr/bin/env python3
"""Unit tests for the Entity Coverage Critic → Missing Entity Recovery pass."""

from __future__ import annotations

import unittest

from cognitive_agent.abbreviation_detector import AbbreviationDetector, AbbreviationMap
from cognitive_agent.entity_recovery import (
    EntityRecovery,
    EntityRecoveryConfig,
    GENERIC_RECOVERY_GUIDANCE,
)
from cognitive_agent.aux_model_registry import AuxModelRegistry, AuxModelSpec


class FakeRecoveryRegistry:
    """Registry stub whose 'recovery' role returns a canned payload."""

    def __init__(self, payload: dict | None = None, status: str = "OK"):
        self.payload = payload
        self.status = status
        self.last_system_prompt = ""
        self.last_user_prompt = ""

    def call_json(self, role, *, system_prompt, user_prompt, schema_hint=None):
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        if self.status != "OK":
            return type("R", (), {
                "status": self.status, "model_id": "fake-recovery",
                "latency_s": 0.1, "prompt_tokens": 0, "output_tokens": 0,
                "error": "boom",
            })()
        return type("R", (), {
            "status": "OK", "model_id": "fake-recovery",
            "payload": self.payload or {},
            "latency_s": 0.1, "prompt_tokens": 10, "output_tokens": 5,
            "error": "",
        })()


TEXT = (
    "Primary biliary cholangitis patients presented increased NK cell activation. "
    "Inhibiting hepatic ferroptosis slowed disease progression. "
    "Deacetylasperulosidic acid methyl ester (DAM) binds Wnt1 in fibrotic liver tissue."
)


def make_abbr_map() -> AbbreviationMap:
    return AbbreviationDetector().detect(TEXT)


class EntityRecoveryValidationTests(unittest.TestCase):
    def setUp(self):
        self.registry = FakeRecoveryRegistry(payload={
            "coverage_gaps": [{"gap": "no pathway entities", "why_relevant": "pathways co-occur"}],
            "entities": [
                {"name": "NK cell activation", "type": "Pathway", "evidence_quote": "increased NK cell activation"},
                {"name": "hepatic ferroptosis", "type": "Pathway", "evidence_quote": "Inhibiting hepatic ferroptosis"},
                {"name": "DAM", "type": "Metabolite", "evidence_quote": "methyl ester (DAM)"},
                {"name": "fibrotic liver", "type": "Tissue", "evidence_quote": "fibrotic liver tissue"},
            ],
        })
        self.recovery = EntityRecovery(
            EntityRecoveryConfig(mode="active"), registry=self.registry,
        )

    def test_accepts_contiguous_schema_entities(self):
        result = self.recovery.run(
            pmid="1", text=TEXT,
            entities=[{"mention": "Wnt1", "type": "Protein"}],
            abbr_map=make_abbr_map(),
        )
        self.assertEqual(result.status, "OK")
        self.assertEqual(len(result.recovered), 4)
        self.assertEqual(len(result.rejected), 0)
        for entity in result.recovered:
            self.assertTrue(entity["grounded"])
            self.assertEqual(entity["alignment_status"], "RECOVERED")
            self.assertEqual(entity["extraction_class"], "entity_recovery")
            self.assertEqual(entity["source_span"], entity["mention"])

    def test_rejects_span_not_in_source(self):
        self.registry.payload["entities"].append(
            {"name": "Liver Fibrosis", "type": "Disease", "evidence_quote": ""}
        )
        result = self.recovery.run(
            pmid="1", text=TEXT, entities=[], abbr_map=make_abbr_map(),
        )
        reasons = [r["reason"] for r in result.rejected]
        self.assertIn("span_not_in_source", reasons)

    def test_rejects_bad_type_and_generic_term(self):
        self.registry.payload["entities"] = [
            {"name": "NK cell activation", "type": "Cell", "evidence_quote": ""},
            {"name": "patients", "type": "Disease", "evidence_quote": ""},
        ]
        result = self.recovery.run(
            pmid="1", text=TEXT, entities=[], abbr_map=make_abbr_map(),
        )
        reasons = sorted(r["reason"] for r in result.rejected)
        self.assertEqual(reasons, ["bad_type", "generic_term"])

    def test_rejects_what_downstream_filters_would_drop(self):
        # prepare_extraction rejects non-anatomical Tissue mentions and
        # method terms; the recovery pass mirrors those filters so its slots
        # are never wasted on entities the verification chain would drop.
        self.registry.payload["entities"] = [
            {"name": "Hydrogels", "type": "Tissue", "evidence_quote": ""},
            {"name": "tumor immune microenvironment", "type": "Tissue",
             "evidence_quote": ""},
        ]
        result = self.recovery.run(
            pmid="1", text=TEXT, entities=[], abbr_map=make_abbr_map(),
        )
        self.assertEqual(result.recovered, [])
        reasons = sorted(r["reason"] for r in result.rejected)
        # The context term is caught by the generic-term block first; the
        # non-anatomical tissue by the downstream mirror.
        self.assertEqual(reasons, [
            "downstream_tissue_not_anatomical",
            "generic_term",
        ])

    def test_dedup_alias_aware(self):
        # Existing inventory already has the abbreviation long form.
        self.registry.payload["entities"] = [
            {"name": "deacetylasperulosidic acid methyl ester", "type": "Metabolite",
             "evidence_quote": ""},
        ]
        result = self.recovery.run(
            pmid="1", text=TEXT,
            entities=[{"mention": "deacetylasperulosidic acid methyl ester",
                       "type": "Metabolite"}],
            abbr_map=make_abbr_map(),
        )
        self.assertEqual(result.recovered, [])
        reasons = {r["reason"] for r in result.rejected}
        self.assertTrue(reasons & {"duplicate_existing", "duplicate_existing_alias"})

    def test_dedup_exact_normalized(self):
        self.registry.payload["entities"] = [
            {"name": "NK cell activation", "type": "Pathway", "evidence_quote": ""},
        ]
        result = self.recovery.run(
            pmid="1", text=TEXT,
            entities=[{"mention": "NK cell activation", "type": "Pathway"}],
            abbr_map=make_abbr_map(),
        )
        self.assertEqual(result.recovered, [])
        self.assertEqual({r["reason"] for r in result.rejected}, {"duplicate_existing"})

    def test_cap_max_accepted(self):
        proposals = [
            {"name": f"NK cell activation {i}" if i else "NK cell activation",
             "type": "Pathway", "evidence_quote": ""}
            for i in range(6)
        ]
        self.registry.payload["entities"] = proposals
        recovery = EntityRecovery(
            EntityRecoveryConfig(mode="active", max_accepted=3), registry=self.registry,
        )
        result = recovery.run(pmid="1", text=TEXT, entities=[], abbr_map=None)
        self.assertEqual(len(result.recovered), 1)  # only the real span passes
        self.assertIn("span_not_in_source", {r["reason"] for r in result.rejected})

    def test_fail_closed_on_model_error(self):
        registry = FakeRecoveryRegistry(status="FALLBACK")
        recovery = EntityRecovery(EntityRecoveryConfig(mode="active"), registry=registry)
        result = recovery.run(pmid="1", text=TEXT, entities=[], abbr_map=None)
        self.assertEqual(result.status, "FALLBACK")
        self.assertEqual(result.recovered, [])

    def test_off_mode_never_calls(self):
        registry = FakeRecoveryRegistry()
        recovery = EntityRecovery(EntityRecoveryConfig(mode="off"), registry=registry)
        result = recovery.run(pmid="1", text=TEXT, entities=[], abbr_map=None)
        self.assertEqual(result.status, "OFF")
        self.assertFalse(hasattr(registry, "last_user_prompt") and registry.last_user_prompt)

    def test_prompt_uses_only_non_gold_guidance(self):
        self.recovery.run(pmid="1", text=TEXT, entities=[], abbr_map=None)
        prompt = self.registry.last_system_prompt
        self.assertIn("HARD RULES", prompt)
        self.assertNotIn("NK cell activation", prompt)
        self.assertNotIn("hepatic ferroptosis", prompt)
        guided = EntityRecovery(
            EntityRecoveryConfig(mode="active", golden_shot=True), registry=self.registry,
        )
        guided.run(pmid="1", text=TEXT, entities=[], abbr_map=None)
        self.assertIn(GENERIC_RECOVERY_GUIDANCE, self.registry.last_system_prompt)
        user = self.registry.last_user_prompt
        self.assertIn("PMID: 1", user)
        self.assertIn(TEXT[:40], user)


class RegistryRecoveryRoleTests(unittest.TestCase):
    def test_recovery_role_in_valid_roles(self):
        self.assertIn("recovery", AuxModelRegistry.VALID_ROLES)

    def test_from_environment_builds_recovery_spec(self):
        import os
        os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
        registry = AuxModelRegistry.from_environment(
            primary_model="m1", critic_model="", judge_model="m2",
            recovery_model="m3",
        )
        self.assertIn("recovery", registry.specs)
        self.assertEqual(registry.specs["recovery"].model_id, "m3")


if __name__ == "__main__":
    unittest.main()
