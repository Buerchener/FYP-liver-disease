import json
import unittest

from scripts.run_agent_v3_ablation100 import load_jsonl
from scripts.run_gold20_semantic_chunk_ab import (
    GOLD_PATH,
    QUOTAS,
    SOURCE_PATH,
    SPLIT_MANIFEST_PATH,
    bootstrap_f1_difference,
    arm_usage,
    select_gold20,
)


class Gold20SemanticChunkABTests(unittest.TestCase):
    def test_selection_is_reproducible_length_stratified_and_balanced(self):
        first_articles, first = select_gold20(load_jsonl(GOLD_PATH), load_jsonl(SOURCE_PATH))
        second_articles, second = select_gold20(load_jsonl(GOLD_PATH), load_jsonl(SOURCE_PATH))
        self.assertEqual(first["manifest_hash"], second["manifest_hash"])
        self.assertEqual(
            [str(item["pmid"]) for item in first_articles],
            [str(item["pmid"]) for item in second_articles],
        )
        self.assertEqual(len(first["selection"]), 20)
        self.assertEqual(sum(item["positive"] for item in first["selection"]), 10)
        for index, (positive, zero) in enumerate(QUOTAS, 1):
            rows = [item for item in first["selection"] if item["length_stratum"] == f"Q{index}"]
            self.assertEqual(sum(item["positive"] for item in rows), positive)
            self.assertEqual(sum(not item["positive"] for item in rows), zero)

    def test_paired_bootstrap_uses_only_paired_rows(self):
        a = [{"tp": 1, "fp": 1, "fn": 1}, {"tp": 0, "fp": 0, "fn": 1}]
        b = [{"tp": 2, "fp": 0, "fn": 0}, {"tp": 1, "fp": 0, "fn": 0}]
        result = bootstrap_f1_difference(a, b, samples=100)
        self.assertEqual(result["samples"], 100)
        self.assertGreater(result["observed_b_minus_a"], 0)
        self.assertEqual(len(result["percentile_95"]), 2)

    def test_gold20_explicitly_contains_six_calibration_seen_documents(self):
        _, sample = select_gold20(load_jsonl(GOLD_PATH), load_jsonl(SOURCE_PATH))
        split = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
        calibration = {
            str(item["pmid"]) for item in split["records"]
            if item.get("split") == "calibration"
        }
        selected = {str(item["pmid"]) for item in sample["selection"]}
        self.assertEqual(len(selected & calibration), 6)

    def test_usage_separates_effective_reused_and_incremental_execution(self):
        executed = {
            "pmid": "1",
            "phases": {"extraction": {"chunk_count": 2, "retry_count": 0}},
        }
        reused = {
            "pmid": "2",
            "phases": {
                "extraction": {"chunk_count": 1, "retry_count": 0},
                "paired_design": {"extraction_reused": True, "reused_from_arm": "A_one_shot"},
            },
        }
        usage = arm_usage([executed, reused])
        self.assertEqual(usage["effective_arm"]["gemini_requests_estimated"], 3)
        self.assertEqual(usage["incremental_execution"]["gemini_requests_estimated"], 2)
        self.assertEqual(usage["reused"]["document_count"], 1)
        self.assertEqual(usage["executed_document_count"], 1)


if __name__ == "__main__":
    unittest.main()
