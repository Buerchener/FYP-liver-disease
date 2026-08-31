import unittest

from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.semantic_chunker import (
    SemanticChunkPlanError,
    chunks_from_manifest,
    parent_sentences,
    plan_from_payload,
)


class SemanticChunkerTests(unittest.TestCase):
    def setUp(self):
        self.text = (
            "TITLE: A liver study\nABSTRACT: Background establishes chronic liver disease. "
            "Methods enrolled a prospective patient cohort. Biomarker A increased in fibrosis. "
            "Biomarker A correlated with disease severity. Treatment reduced Biomarker A. "
            "The findings support a mechanistic association."
        )
        self.reader = ArticleEvidenceReader()
        self.units = self.reader.read(self.text)
        self.sentences = parent_sentences(self.text, self.units)

    def test_stable_sentence_ids_and_exact_overlapped_chunks(self):
        self.assertEqual(
            [item.sentence_id for item in self.sentences],
            [f"s{index:03d}" for index in range(1, 7)],
        )
        plan = plan_from_payload(self.text, self.units, {"groups": [
            {"start_id": "s001", "end_id": "s003", "topic": "setup"},
            {"start_id": "s004", "end_id": "s006", "topic": "findings"},
        ]})
        self.assertEqual(len(plan.chunks), 2)
        first, second = plan.chunks
        self.assertEqual(first.char_start, 0)
        self.assertEqual(first.text, self.text[first.char_start:first.char_end])
        self.assertEqual(second.text, self.text[second.char_start:second.char_end])
        self.assertEqual(second.char_start, self.sentences[2].char_start)
        self.assertEqual(second.overlapped_parent_sentences, 1)
        self.assertEqual(second.char_end, len(self.text))
        self.assertEqual(second.topic, "findings")

    def test_manifest_rebuild_rejects_source_hash_mismatch(self):
        plan = plan_from_payload(self.text, self.units, {"groups": [
            {"start_id": "s001", "end_id": "s003", "topic": "setup"},
            {"start_id": "s004", "end_id": "s006", "topic": "findings"},
        ]})
        record = plan.to_dict()
        self.assertEqual(len(chunks_from_manifest(self.text, self.units, record)), 2)
        record["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(SemanticChunkPlanError, "hash mismatch"):
            chunks_from_manifest(self.text, self.units, record)

    def test_rejects_missing_duplicate_reversed_and_unknown_ranges(self):
        cases = [
            [
                {"start_id": "s001", "end_id": "s002", "topic": "a"},
                {"start_id": "s004", "end_id": "s006", "topic": "b"},
            ],
            [
                {"start_id": "s001", "end_id": "s003", "topic": "a"},
                {"start_id": "s003", "end_id": "s006", "topic": "b"},
            ],
            [
                {"start_id": "s001", "end_id": "s004", "topic": "a"},
                {"start_id": "s005", "end_id": "s003", "topic": "b"},
            ],
            [
                {"start_id": "s001", "end_id": "s003", "topic": "a"},
                {"start_id": "s004", "end_id": "s999", "topic": "b"},
            ],
        ]
        for groups in cases:
            with self.subTest(groups=groups):
                with self.assertRaises(SemanticChunkPlanError):
                    plan_from_payload(self.text, self.units, {"groups": groups})

    def test_rejects_wrong_group_count_and_avoidable_oversize(self):
        with self.assertRaisesRegex(SemanticChunkPlanError, "2 or 3"):
            plan_from_payload(self.text, self.units, {"groups": [
                {"start_id": "s001", "end_id": "s006", "topic": "all"},
            ]})
        with self.assertRaisesRegex(SemanticChunkPlanError, "maximum"):
            plan_from_payload(self.text, self.units, {"groups": [
                {"start_id": "s001", "end_id": "s003", "topic": "a"},
                {"start_id": "s004", "end_id": "s006", "topic": "b"},
            ]}, max_group_chars=100)


if __name__ == "__main__":
    unittest.main()
