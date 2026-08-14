import unittest
from pathlib import Path

from scripts.check_agent_v3_leakage import find_leaks
from scripts.evaluate_agent_v3_experiments import metrics, paired_delta
from scripts.preregister_blind50 import STRATA, classify


class AgentV3ResearchProtocolTests(unittest.TestCase):
    def test_blind_strata_are_mutually_exclusive_and_cover_expected_examples(self):
        examples = {
            "clinical": {"abstract": "Patients in a prospective clinical cohort were enrolled."},
            "mechanistic_in_vitro": {"abstract": "HepG2 cell line experiments tested the mechanism."},
            "human_omics_computational": {"abstract": "Human transcriptomics and bioinformatics were integrated."},
            "animal": {"abstract": "Mice received treatment in vivo."},
            "review_other": {"abstract": "This narrative review summarizes the field."},
        }
        self.assertEqual({classify(value) for value in examples.values()}, set(STRATA))
        for expected, article in examples.items():
            self.assertEqual(classify(article), expected)

    def test_leakage_guard_detects_pmid_and_hash(self):
        manifest = {"records": [{"pmid": "123", "abstract_sha256": "abc"}]}
        leaks = find_leaks(manifest, [(Path("rules.json"), {"support_pmids": ["123"]})])
        self.assertEqual(leaks[0]["kind"], "blind_pmid")
        self.assertEqual(find_leaks(manifest, [(Path("safe.json"), {"pmid": "999"})]), [])

    def test_article_metrics_and_paired_bootstrap_are_deterministic(self):
        baseline = [
            {"pmid": str(i), "tp": 1, "fp": 1, "fn": 1} for i in range(10)
        ]
        improved = [
            {"pmid": str(i), "tp": 2, "fp": 0, "fn": 0} for i in range(10)
        ]
        self.assertGreater(metrics(improved)["relation_f1"], metrics(baseline)["relation_f1"])
        first = paired_delta(baseline, improved, iterations=200, seed=7)
        second = paired_delta(baseline, improved, iterations=200, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_non_negative_probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
