import unittest

from cognitive_agent.evidence_statistics import classify_opposition


def classify(clause, *, offset=100, estimate_arg="fibrosis stage"):
    negation = (clause.index("not"), clause.index("not") + len("not"))
    trigger_start = clause.index("improve")
    right_end = clause.index(estimate_arg) + len(estimate_arg)
    return classify_opposition(
        clause,
        char_offset=offset,
        negation=negation,
        trigger=(trigger_start, trigger_start + len("improve")),
        right_argument_end=right_end,
    )


class EvidenceStatisticsTests(unittest.TestCase):
    def test_rr_interval_crossing_null_is_inconclusive_with_exact_offsets(self):
        clause = "semaglutide did not improve fibrosis stage (RR = 1.21, 95% CI 0.94-1.56, p = 0.13)"
        result = classify(clause)
        self.assertEqual(result["polarity"], "INCONCLUSIVE")
        estimate = result["estimate"]
        self.assertEqual(estimate["measure"], "RR")
        self.assertEqual((estimate["point"], estimate["lower"], estimate["upper"], estimate["null"]), (1.21, .94, 1.56, 1.0))
        self.assertEqual(clause[estimate["char_start"] - 100:estimate["char_end"] - 100], estimate["text"])
        self.assertIn("confidence_interval_crosses_null", result["reasons"])

    def test_ratio_interval_excluding_null_does_not_override_negation(self):
        clause = "Drug did not improve fibrosis stage (OR 1.40, 95% CI 1.10 to 1.80)"
        result = classify(clause)
        self.assertEqual(result["polarity"], "COUNTEREVIDENCE")
        self.assertEqual(result["estimate"]["measure"], "OR")
        self.assertIn("confidence_interval_excludes_null", result["reasons"])

    def test_parenthesized_ci_format_is_attached_to_same_measure(self):
        clause = "Drug did not improve fibrosis stage (RR 1.2 (95% CI: 0.9 to 1.6))"
        result = classify(clause)
        self.assertEqual(result["polarity"], "INCONCLUSIVE")
        self.assertEqual(result["estimate"]["text"], "RR 1.2 (95% CI: 0.9 to 1.6")

    def test_hr_supports_unicode_minus_and_scientific_notation(self):
        clause = "Drug did not improve fibrosis stage (HR = 1.2e0; 95% CI 0.8–1.4)"
        result = classify(clause)
        self.assertEqual(result["polarity"], "INCONCLUSIVE")
        self.assertEqual(result["estimate"]["measure"], "HR")

    def test_difference_measures_use_zero_as_null(self):
        crossing = "Drug did not improve fibrosis stage (MD = -0.2, 95% CI −0.5 to 0.1)"
        excluding = "Drug did not improve fibrosis stage (SMD = 0.4, 95% CI 0.1–0.7)"
        self.assertEqual(classify(crossing)["estimate"]["null"], 0.0)
        self.assertEqual(classify(crossing)["polarity"], "INCONCLUSIVE")
        self.assertEqual(classify(excluding)["polarity"], "COUNTEREVIDENCE")

    def test_explicit_nonsignificance_language_is_inconclusive_without_ci(self):
        clause = "Drug did not statistically significantly improve fibrosis stage"
        result = classify(clause)
        self.assertEqual(result["polarity"], "INCONCLUSIVE")
        self.assertEqual(result["reasons"], ["explicit_nonsignificance_language"])

    def test_ordinary_negation_defaults_to_counterevidence(self):
        result = classify("Drug did not improve fibrosis stage")
        self.assertEqual(result, {
            "polarity": "COUNTEREVIDENCE", "reasons": ["explicit_negation"],
        })

    def test_malformed_or_ambiguous_ci_is_not_parsed(self):
        reversed_interval = "Drug did not improve fibrosis stage (RR 1.2, 95% CI 1.8-0.9)"
        point_outside = "Drug did not improve fibrosis stage (RR 2.0, 95% CI 0.8-1.5)"
        multiple = "Drug did not improve fibrosis stage (RR 1.2; OR 1.4, 95% CI 1.1-1.8)"
        for clause in (reversed_interval, point_outside, multiple):
            with self.subTest(clause=clause):
                result = classify(clause)
                self.assertEqual(result["polarity"], "COUNTEREVIDENCE")
                self.assertNotIn("estimate", result)

    def test_real_41610839_quote_is_nonsignificant_not_logical_refutation(self):
        clause = (
            "semaglutide did not improve fibrosis stage "
            "(RR = 1.21, 95% CI 0.94-1.56, p = 0.13; I2 = 0%)"
        )
        result = classify(clause)
        self.assertEqual(result["polarity"], "INCONCLUSIVE")
        self.assertEqual(result["estimate"]["upper"], 1.56)


if __name__ == "__main__":
    unittest.main()
