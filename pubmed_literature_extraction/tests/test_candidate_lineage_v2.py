import unittest

from cognitive_agent.candidate_lineage import (
    bind_by_lineage, edited_version, normalize_lineage,
)


class CandidateLineageV2Tests(unittest.TestCase):
    def test_recovery_lane_survives_normalization(self):
        row = normalize_lineage({"candidate_id": "r-1"}, lane="recovery")
        self.assertEqual(row["candidate_lane"], "recovery")
        self.assertEqual(row["source_lanes"], ["recovery"])

    def test_edit_increments_version_and_keeps_parent(self):
        row = edited_version(
            normalize_lineage({"candidate_id": "c-1"}, lane="extracted_hint"),
            reason_code="predicate_edit",
        )
        self.assertEqual(row["candidate_version"], 2)
        self.assertEqual(row["parent_candidate_id"], "c-1")
        self.assertEqual(row["parent_version"], 1)

    def test_v1_decision_cannot_bind_v2_candidate(self):
        bound, unbound = bind_by_lineage(
            [{"candidate_id": "c-1", "candidate_version": 2}],
            [{"candidate_id": "c-1", "candidate_version": 1}],
        )
        self.assertEqual(bound, {})
        self.assertIn("lineage_binding_missing", unbound[0]["quality_flags"])

    def test_live_rows_never_fall_back_to_array_position(self):
        bound, unbound = bind_by_lineage(
            [{"candidate_id": "c-1", "candidate_version": 1}],
            [{"candidate_id": "c-other", "candidate_version": 1}],
            legacy_index_compatibility=True,
        )
        self.assertEqual(bound, {})
        self.assertEqual(len(unbound), 1)
