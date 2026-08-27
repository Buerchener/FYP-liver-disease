import unittest

from cognitive_agent.aux_model_registry import AuxModelRegistry, AuxModelSpec
from cognitive_agent.evidence_selector import (
    CONTRADICTED,
    ENTAILED,
    NOT_ENOUGH_INFORMATION,
    EvidenceEntailmentEngine,
    EvidenceSelector,
)
from cognitive_agent.evidence_units import ArticleEvidenceReader, EvidenceUnit


class EvidenceSelectorTests(unittest.TestCase):
    def setUp(self):
        self.selector = EvidenceSelector()

    def test_selects_shortest_contiguous_assertive_span_with_offsets(self):
        text = "TITLE: T\nABSTRACT: RESULTS: TP53 was associated with hepatocellular carcinoma in patients."
        units = ArticleEvidenceReader().read(text)
        selected = self.selector.select(
            candidate_id="p1", subject_mentions=["TP53"],
            object_mentions=["hepatocellular carcinoma"], predicate="ASSOCIATED_WITH",
            units=units, source_text=text,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.local_label, ENTAILED)
        self.assertEqual(text[selected.char_start:selected.char_end], selected.text)
        self.assertIn("TP53", selected.text)
        self.assertIn("associated", selected.text)
        self.assertIn("hepatocellular carcinoma", selected.text)
        self.assertLess(len(selected.text), len(units[0].text))

    def test_negation_is_preserved_in_minimal_span_and_contradicts(self):
        text = "TITLE: T\nABSTRACT: RESULTS: No association between TP53 and cirrhosis was detected."
        selected = self.selector.select(
            candidate_id="p1", subject_mentions=["TP53"], object_mentions=["cirrhosis"],
            predicate="ASSOCIATED_WITH", units=ArticleEvidenceReader().read(text), source_text=text,
        )
        self.assertEqual(selected.local_label, CONTRADICTED)
        self.assertTrue(selected.text.casefold().startswith("no association"))

    def test_background_assertion_abstains_and_cross_unit_pair_is_forbidden(self):
        text = "TITLE: T\nABSTRACT: BACKGROUND: TP53 is associated with cirrhosis. RESULTS: TP53 increased."
        units = ArticleEvidenceReader().read(text)
        selected = self.selector.select(
            candidate_id="p1", subject_mentions=["TP53"], object_mentions=["cirrhosis"],
            predicate="ASSOCIATED_WITH", units=units, source_text=text,
        )
        self.assertEqual(selected.local_label, NOT_ENOUGH_INFORMATION)
        missing = self.selector.select(
            candidate_id="p2", subject_mentions=["increased"], object_mentions=["cirrhosis"],
            predicate="ASSOCIATED_WITH", units=units, source_text=text,
        )
        self.assertIsNone(missing)

    def test_trigger_attached_to_third_entity_is_routed_to_nei(self):
        text = (
            "RESULTS: Acromegaly was less frequent in steatosis, and GH was "
            "inversely associated with steatosis."
        )
        unit = EvidenceUnit("u1", "RESULTS", text, 0, len(text), "s1")
        selected = self.selector.select(
            candidate_id="p1", subject_mentions=["Acromegaly"],
            object_mentions=["steatosis"], predicate="ASSOCIATED_WITH",
            units=[unit], source_text=text, other_mentions=["GH"],
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.local_label, NOT_ENOUGH_INFORMATION)
        self.assertIsNone(selected.trigger_span)
        self.assertIn("trigger_attachment_ambiguous", selected.reason_codes)

    def test_remote_quote_must_realign_and_qwen_only_reviews_conflict(self):
        text = "TP53 and cirrhosis were measured in this cohort."
        unit = EvidenceUnit("u1", "RESULTS", text, 0, len(text), "s1")
        selection = self.selector.select(
            candidate_id="p1", subject_mentions=["TP53"], object_mentions=["cirrhosis"],
            predicate="ASSOCIATED_WITH", units=[unit], source_text=text,
        )
        calls = {"primary": 0, "critic": 0}

        def primary(_):
            calls["primary"] += 1
            return {"decisions": [{
                "candidate_id": "p1", "label": "CONTRADICTED", "quote": selection.text,
            }]}

        def critic(_):
            calls["critic"] += 1
            return {"decisions": [{
                "candidate_id": "p1", "label": "ENTAILED", "quote": selection.text,
            }]}

        specs = [
            AuxModelSpec("primary", "test", "deepseek", "test://", "x"),
            AuxModelSpec("critic", "test", "qwen", "test://", "y"),
        ]
        engine = EvidenceEntailmentEngine(AuxModelRegistry(
            specs, generate={"primary": primary, "critic": critic},
        ))
        decisions, _ = engine.assess(
            [selection], source_text=text,
            candidate_payloads={"p1": {"subject": "TP53", "predicate": "ASSOCIATED_WITH", "object": "cirrhosis"}},
            allow_remote=True,
        )
        self.assertEqual(calls, {"primary": 1, "critic": 1})
        self.assertEqual(decisions[0].label, NOT_ENOUGH_INFORMATION)
        self.assertEqual(decisions[0].source, "qwen_conflict_critic")


if __name__ == "__main__":
    unittest.main()
