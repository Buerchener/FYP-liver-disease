import unittest

from cognitive_agent.conformal_router import (
    ABSTAIN,
    ACCEPT_LOCAL,
    CALL_DEEPSEEK,
    CALL_QWEN_CRITIC,
    HUMAN_REVIEW,
    CalibrationExample,
    ConformalCalibration,
    ConformalRiskRouter,
    RiskFeatures,
)


def features(**updates):
    payload = {
        "candidate_id": "p1", "relation_score": 0.99, "evidence_score": 0.99,
        "verifier_passed": True, "local_label": "ASSOCIATED_WITH",
        "study_type": "clinical", "predicate": "ASSOCIATED_WITH", "section": "RESULTS",
    }
    payload.update(updates)
    return RiskFeatures(**payload)


class ConformalRiskRouterTests(unittest.TestCase):
    def calibration(self, size=40, *, group="global", error_indexes=()):
        return ConformalCalibration(examples=[
            CalibrationExample(f"c-{index}", 0.02, int(index in error_indexes), group)
            for index in range(size)
        ])

    def test_certified_low_risk_accepts_only_after_verifier(self):
        router = ConformalRiskRouter(self.calibration())
        accepted = router.route(features())
        self.assertEqual(accepted.decision, ACCEPT_LOCAL)
        blocked = router.route(features(verifier_passed=False, verifier_flags=["schema_mismatch"]))
        self.assertEqual(blocked.decision, ABSTAIN)

    def test_uncalibrated_or_high_risk_routes_models_then_review(self):
        router = ConformalRiskRouter(ConformalCalibration())
        first = router.route(features(relation_score=0.5, evidence_score=0.5))
        self.assertEqual(first.decision, CALL_DEEPSEEK)
        second = router.route(features(
            relation_score=0.5, evidence_score=0.5,
            deepseek_label="NO_RELATION",
        ))
        self.assertEqual(second.decision, CALL_QWEN_CRITIC)
        third = router.route(features(
            relation_score=0.5, evidence_score=0.5,
            deepseek_label="NO_RELATION", qwen_label="ASSOCIATED_WITH",
        ))
        self.assertEqual(third.decision, HUMAN_REVIEW)

    def test_mondrian_group_requires_twenty_then_falls_back_global(self):
        target = features().mondrian_group
        calibration = ConformalCalibration(examples=[
            *[CalibrationExample(f"g-{i}", 0.02, 0, target) for i in range(19)],
            *[CalibrationExample(f"o-{i}", 0.02, 0, "other") for i in range(21)],
        ])
        router = ConformalRiskRouter(calibration, min_group_size=20)
        route = router.route(features())
        self.assertEqual(route.calibration_group, "global_fallback")
        calibration.examples.append(CalibrationExample("g-20", 0.02, 0, target))
        route = router.route(features())
        self.assertEqual(route.calibration_group, target)
        self.assertEqual(route.calibration_size, 20)

    def test_calibration_errors_make_acceptance_conservative(self):
        router = ConformalRiskRouter(self.calibration(error_indexes=(0, 1)))
        route = router.route(features())
        self.assertNotEqual(route.decision, ACCEPT_LOCAL)
        self.assertGreater(route.conformal_upper_risk, route.alpha)

    def test_conformal_layer_never_upgrades_failed_verifier(self):
        router = ConformalRiskRouter(self.calibration())
        route = router.route(features())
        downgraded = router.enforce_no_upgrade(route, verifier_passed=False)
        self.assertEqual(downgraded.decision, ABSTAIN)
        self.assertIn("accept_downgraded_by_verifier", downgraded.reason_codes)


if __name__ == "__main__":
    unittest.main()
