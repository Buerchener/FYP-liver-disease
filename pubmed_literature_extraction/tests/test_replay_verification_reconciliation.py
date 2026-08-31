import unittest

from scripts.replay_verification_reconciliation import (
    _bind_legacy_adjudications,
    _legacy_replay_candidates,
)


class ReplayVerificationReconciliationTests(unittest.TestCase):
    def test_legacy_review_candidate_overlays_matching_extracted_hint(self):
        projected = [{
            "candidate_id": "c-projected",
            "subject": "TP53",
            "subject_type": "Gene",
            "predicate": "ASSOCIATED_WITH",
            "object": "HCC",
            "object_type": "Disease",
            "evidence": "TP53 was associated with HCC.",
            "provenance": ["top_level"],
        }]
        reviewed = {**projected[0], "candidate_id": "r000", "quality_flags": ["manual_review"]}
        record = {"phases": {"collaboration": {
            "review_candidates": [reviewed],
            "review_decisions": [{
                "candidate_id": "r000",
                "action": "KEEP",
                "reason_code": "EXPLICIT_DIRECT_RELATION",
                "confidence": 0.95,
            }],
        }}}
        candidates, decisions = _legacy_replay_candidates(record, projected)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["candidate_id"], "r000")
        self.assertIn("legacy_collaboration_replay", candidates[0]["provenance"])
        self.assertIn("r000", decisions)

    def test_supported_legacy_decision_is_bound_only_to_real_pack_spans(self):
        relations = [{
            "candidate_id": "r000",
            "evidence_pack": {
                "source_traceable": True,
                "spans": [{"span_id": "span-1"}, {"span_id": "span-2"}],
            },
        }]
        decisions = {"r000": {
            "candidate_id": "r000",
            "action": "KEEP",
            "reason_code": "EXPLICIT_DIRECT_RELATION",
            "confidence": 0.95,
        }}
        bound, audit = _bind_legacy_adjudications(
            relations, decisions, model_id="deepseek-cache",
        )
        self.assertEqual(bound[0]["adjudication_verdict"], "SUPPORTED")
        self.assertEqual(bound[0]["supporting_span_ids"], ["span-1", "span-2"])
        self.assertEqual(bound[0]["adjudication_model_id"], "deepseek-cache")
        self.assertTrue(audit[0]["source_traceable"])

    def test_rejected_legacy_decision_cannot_be_migrated_as_supported(self):
        bound, _ = _bind_legacy_adjudications(
            [{
                "candidate_id": "r001",
                "evidence_pack": {"source_traceable": True, "spans": [{"span_id": "span-1"}]},
            }],
            {"r001": {
                "candidate_id": "r001",
                "action": "REJECT",
                "reason_code": "BACKGROUND_ONLY",
                "confidence": 0.99,
            }},
            model_id="deepseek-cache",
        )
        self.assertEqual(bound[0]["adjudication_verdict"], "NOT_SUPPORTED")
        self.assertEqual(bound[0]["supporting_span_ids"], [])


if __name__ == "__main__":
    unittest.main()
