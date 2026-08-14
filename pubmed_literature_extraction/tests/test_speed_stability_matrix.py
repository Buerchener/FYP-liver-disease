import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_speed_stability_matrix import (
    DEFAULT_PMIDS,
    STRATA,
    build_plan,
    classify_failure,
    selected_rows,
    summarize,
)


class SpeedStabilityMatrixTests(unittest.TestCase):
    def test_staged_plan_has_inner_one_and_changes_one_dimension(self):
        plan = build_plan(("model-a", "model-b"), full_factorial=False)
        self.assertEqual(len(plan), 6)
        self.assertTrue(all(cell.inner_workers == 1 for cell in plan))
        self.assertEqual({cell.outer_workers for cell in plan[:3]}, {1, 2, 4})
        self.assertIn("model-b", {cell.model_id for cell in plan})

    def test_selected_rows_require_all_four_strata(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.jsonl"
            gold = Path(temp) / "gold.jsonl"
            source.write_text("\n".join(json.dumps({"pmid": p}) for p in DEFAULT_PMIDS), encoding="utf-8")
            gold.write_text("\n".join(json.dumps({"pmid": p}) for p in DEFAULT_PMIDS), encoding="utf-8")
            articles, gold_rows = selected_rows(source, gold, DEFAULT_PMIDS)
        self.assertEqual([row["pmid"] for row in articles], list(DEFAULT_PMIDS))
        self.assertEqual(len(gold_rows), 4)
        self.assertEqual(len({STRATA[p][1] for p in DEFAULT_PMIDS}), 4)

    def test_transport_failure_invalidates_quality_aggregates(self):
        rows = [{
            "success": False, "error_category": "transport", "latency_s": 1.0, "entity_count": 2,
            "relation_count": 1, "verified_relation_count": 1,
            "evidence_contiguous_rate": 1.0,
        }]
        transport = {
            "connect_error_count": 1, "requests": 1, "invalid_json_count": 0,
            "ttfb_s": [0.2], "prompt_tokens": None, "output_tokens": None,
            "reasoning_tokens": None,
        }
        summary = summarize(rows, transport, {"status": "ok"})
        self.assertEqual(summary["quality_status"], "invalid_transport")
        self.assertIsNone(summary["mean_entities_per_article"])

    def test_authentication_failure_is_not_labeled_as_transport(self):
        category = classify_failure(RuntimeError("Error code: 401 - Invalid token"))
        self.assertEqual(category, "provider_auth")


if __name__ == "__main__":
    unittest.main()
