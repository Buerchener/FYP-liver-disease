import unittest

from scripts.prepare_agent_v3_splits import TARGETS, split_rows


class AgentV3SplitTests(unittest.TestCase):
    @staticmethod
    def rows():
        output = []
        contexts = ["human_cohort", "mouse_mechanistic", "in_vitro", "computational_omics", "review"]
        for index in range(200):
            output.append({
                "pmid": str(10_000_000 + index),
                "study_context": contexts[index % len(contexts)],
                "in_scope": index % 4 != 0,
                "relations": ([{"import_ready": index % 7 == 0}] if index % 3 == 0 else []),
            })
        return output

    def test_split_is_exact_disjoint_and_deterministic(self):
        rows = self.rows()
        first = split_rows(rows, 20260814)
        second = split_rows(rows, 20260814)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {str(item["pmid"]) for item in rows})
        counts = {name: sum(split == name for split in first.values()) for name in TARGETS}
        self.assertEqual(counts, TARGETS)

    def test_split_refuses_non_200_input(self):
        with self.assertRaises(ValueError):
            split_rows(self.rows()[:-1], 1)


if __name__ == "__main__":
    unittest.main()
