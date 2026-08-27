import unittest

from scripts.evaluate_gold200_unified import canonical_endpoint, cell_subset_signature


class Gold200UnifiedEvaluationTests(unittest.TestCase):
    def test_numbered_cell_subset_paraphrases_share_one_signature(self):
        self.assertEqual(
            cell_subset_signature("subpopulation 11 mononuclear macrophages"),
            cell_subset_signature("macrophage subset 11"),
        )

    def test_numbered_cell_subset_alias_resolves_to_gold_canonical(self):
        alias_map = {
            "macrophage subset 11": {("macrophage subset 11", "CellType")},
        }
        self.assertEqual(
            canonical_endpoint(
                "subpopulation 11 mononuclear macrophages", "CellType", alias_map,
            ),
            "macrophage subset 11",
        )

    def test_non_numbered_cell_phrases_are_not_loosely_merged(self):
        self.assertIsNone(cell_subset_signature("mononuclear macrophages"))


if __name__ == "__main__":
    unittest.main()
