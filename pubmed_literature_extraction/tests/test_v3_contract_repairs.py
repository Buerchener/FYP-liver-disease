"""Regression gates for the post-Round-4 safety repairs.

These tests are intentionally offline.  They guard compatibility, evidence
boundaries and request budgeting without invoking an LLM provider.
"""

from __future__ import annotations

import unittest

from cognitive_agent.agent import AgentConfig
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.pairwise_judge import PairwiseJudgeConfig


class RelationAuthorityContractTests(unittest.TestCase):
    def test_legacy_is_the_default_production_authority(self):
        config = AgentConfig()
        self.assertEqual(config.relation_authority, "legacy")

    def test_entity_recovery_guidance_is_off_by_default(self):
        config = AgentConfig()
        self.assertFalse(config.entity_recovery_golden_shot)


class EvidenceBoundaryContractTests(unittest.TestCase):
    def test_adjacent_window_never_crosses_structured_sections(self):
        text = (
            "TITLE: Example\nABSTRACT: METHODS: TP53 was measured. "
            "RESULTS: HCC progression was evaluated."
        )
        units = ArticleEvidenceReader().read(text)
        parents = ArticleEvidenceReader.parent_units(text, units)
        windows = ArticleEvidenceReader.adjacent_sentence_windows(text, parents)
        self.assertEqual(windows, [])


class JudgeBudgetContractTests(unittest.TestCase):
    def test_default_budget_is_article_global(self):
        config = PairwiseJudgeConfig()
        self.assertEqual(config.max_calls_per_article, 2)
        # refine_pair_result allocates this shared budget across Claim Gate and
        # predicate stages; neither stage receives an independent allowance.
        self.assertGreaterEqual(config.max_pairs_per_call, 1)


if __name__ == "__main__":
    unittest.main()
