import unittest

from liverkg_cli.review_data import build_review_data


class ReviewDataTests(unittest.TestCase):
    def test_reads_verified_relations_and_limits_output_fields(self):
        payload = {
            "config": {"run_id": "fixture", "api_key": "secret"},
            "report": {"run_id": "demo-7", "schema_profile": {
                "name": "Locations", "manifest_sha256": "abc", "extra": "unused",
            }},
            "records": [{
                "pmid": "17", "title": "Locations", "source_text": "private",
                "phases": {
                    "extraction": {"relations": [{"predicate": "raw-only"}]},
                    "verification": {"relations": [{
                        "subject": "museum", "subject_type": "Facility",
                        "predicate": "LocatedIn", "object": "London", "object_type": "City",
                        "evidence": "The museum is in London.", "semantic_status": "ACCEPTED",
                        "factual_status": "VALID", "write_status": "SEMANTIC_ONLY",
                        "schema_valid": True, "import_ready": False,
                        "evidence_pack": {"spans": [{
                            "text": "The museum is in London.", "role": "OWNER",
                            "alignment_status": "MATCH_EXACT", "char_start": 0, "char_end": 24,
                            "unused": "filtered",
                        }]},
                    }]},
                },
            }],
        }
        data = build_review_data(payload)
        relation = data["articles"][0]["relations"][0]
        self.assertEqual(data["run_id"], "demo-7")
        self.assertEqual(data["schema_hash"], "abc")
        self.assertEqual(relation["predicate"], "LocatedIn")
        self.assertEqual(relation["candidate_schema_valid"], True)
        self.assertEqual(relation["import_ready"], False)
        self.assertEqual(relation["spans"][0]["char_end"], 24)
        self.assertNotIn("raw-only", str(data))
        self.assertNotIn("private", str(data))
        self.assertNotIn("secret", str(data))

    def test_archive_does_not_gain_invented_statuses(self):
        data = build_review_data({"records": [{
            "pmid": "old", "phases": {"verification": {"relations": [{
                "subject": "A", "predicate": "RELATED", "object": "B",
                "evidence_spans": [{"text": "A related to B."}],
            }]}}
        }]}, demo=True)
        relation = data["articles"][0]["relations"][0]
        self.assertEqual(data["mode"], "demo")
        self.assertEqual(relation["semantic_status"], "UNKNOWN")
        self.assertEqual(relation["factual_status"], "UNKNOWN")
        self.assertIsNone(relation["import_ready"])
        self.assertEqual(relation["spans"][0]["alignment_status"], "UNKNOWN")

    def test_empty_records_and_article_errors_are_preserved(self):
        self.assertEqual(build_review_data({"records": []})["articles"], [])
        data = build_review_data({"records": [{"pmid": "x", "error": "failed"}]})
        self.assertEqual(data["articles"][0]["error"], "failed")
        self.assertEqual(data["articles"][0]["relations"], [])

    def test_rejects_payload_without_records_array(self):
        with self.assertRaisesRegex(ValueError, "records.*array"):
            build_review_data({"records": {"items": []}})


if __name__ == "__main__":
    unittest.main()
