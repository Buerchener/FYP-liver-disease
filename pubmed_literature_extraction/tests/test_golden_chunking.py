import unittest

from cognitive_agent.article_chunker import ArticleChunker
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.golden_examples import GoldenExampleSelector


class GoldenExampleSelectorTests(unittest.TestCase):
    def test_selects_three_positive_and_one_boundary_example(self):
        selected = GoldenExampleSelector().select(
            "RESULTS: OTUD5 interacted with MAVS and was expressed in macrophages.",
            "human_omics",
        )
        self.assertEqual(len(selected.examples), 4)
        self.assertEqual(len(set(selected.names)), 4)
        self.assertEqual(
            sum(name.startswith("negative_") for name in selected.names), 1
        )
        self.assertEqual(selected.names[0], "gold_otud5_interaction_expression")

    def test_excludes_example_derived_from_current_gold_document(self):
        selected = GoldenExampleSelector().select(
            "OTUD5 interacted with MAVS in patients.", "clinical",
            document_id="41650163",
        )
        self.assertNotIn("gold_otud5_interaction_expression", selected.names)

    def test_computational_article_gets_no_relation_boundary(self):
        selected = GoldenExampleSelector().select(
            "Network pharmacology and molecular docking screened TP53.",
            "computational",
        )
        self.assertIn("negative_computational_no_relation", selected.names)
        self.assertEqual(
            sum(name.startswith("negative_") for name in selected.names), 2
        )

    def test_single_example_uses_the_most_relevant_positive(self):
        selected = GoldenExampleSelector().select(
            "RESULTS: OTUD5 interacted with MAVS in macrophages.",
            "human_omics", max_examples=1,
        )
        self.assertEqual(selected.names, ["gold_otud5_interaction_expression"])

    def test_single_review_example_preserves_the_negative_boundary(self):
        selected = GoldenExampleSelector().select(
            "This review summarizes therapeutic strategies for HCC.",
            "review", max_examples=1,
        )
        self.assertEqual(selected.names, ["negative_review_no_asserted_relation"])


class ArticleChunkerTests(unittest.TestCase):
    def setUp(self):
        self.reader = ArticleEvidenceReader()

    def test_short_article_remains_one_exact_window(self):
        text = "TITLE: Test\nABSTRACT: TP53 was associated with HCC."
        chunks = ArticleChunker(max_chars=800).build(text, self.reader.read(text))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)

    def test_long_article_uses_contiguous_overlapping_windows(self):
        sentences = [
            f"RESULTS: Gene{i} was associated with liver disease in patients and was validated."
            for i in range(30)
        ]
        text = "TITLE: Long study\nABSTRACT: " + " ".join(sentences)
        chunks = ArticleChunker(max_chars=900, max_chunks=3).build(
            text, self.reader.read(text), high_complexity=True
        )
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(text[chunk.char_start:chunk.char_end], chunk.text)
        self.assertGreaterEqual(chunks[1].overlapped_parent_sentences, 1)
        self.assertEqual(chunks[-1].char_end, len(text))
        owners = [sentence for chunk in chunks for sentence in chunk.owner_sentence_ids]
        self.assertEqual(len(owners), len(set(owners)))
        self.assertTrue(chunks[1].context_sentence_ids)
        for chunk in chunks:
            self.assertFalse(
                set(chunk.owner_sentence_ids) & set(chunk.context_sentence_ids)
            )


if __name__ == "__main__":
    unittest.main()
