import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cognitive_agent.candidate_store import CandidateRelationStore
from cognitive_agent.evidence_pack import EvidencePackBuilder
from cognitive_agent.verifier import KGVerifier


class OfflineKG:
    is_connected = False


def entity(mention, entity_type):
    return {"mention": mention, "type": entity_type, "attributes": {}}


def relation(evidence, **extra):
    return {
        "subject": extra.pop("subject", "TP53"),
        "subject_type": extra.pop("subject_type", "Gene"),
        "predicate": extra.pop("predicate", "ASSOCIATED_WITH"),
        "object": extra.pop("object", "HCC"),
        "object_type": extra.pop("object_type", "Disease"),
        "evidence": evidence,
        "direction": extra.pop("direction", "positive"),
        "negated": False,
        "uncertain": False,
        **extra,
    }


class CandidateStoreTests(unittest.TestCase):
    def test_sqlite_store_records_all_final_relation_statuses(self):
        text = (
            "TITLE: Study\nABSTRACT: RESULTS: TP53 was associated with HCC. "
            "BACKGROUND: EGFR is associated with HCC. "
            "PTEN expression changed in HCC. "
            "MDM2 was not associated with HCC."
        )
        verified = KGVerifier(OfflineKG(), verification_policy="tiered-v2").verify(
            [
                entity("TP53", "Gene"), entity("EGFR", "Gene"),
                entity("PTEN", "Gene"), entity("MDM2", "Gene"),
                entity("HCC", "Disease"),
            ],
            [
                relation("TP53 was associated with HCC.", candidate_id="c-ready"),
                relation(
                    "EGFR is associated with HCC.",
                    subject="EGFR", claim_role="BACKGROUND",
                    candidate_id="c-review",
                ),
                relation(
                    "PTEN expression changed in HCC.",
                    subject="PTEN", quality_flags=["trigger_missing"],
                    candidate_id="c-semantic",
                ),
                relation(
                    "MDM2 was not associated with HCC.",
                    subject="MDM2", candidate_id="c-blocked",
                ),
            ],
            pmid="123",
            text=text,
        )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "candidates.sqlite3"
            store = CandidateRelationStore(mode="sqlite", path=db_path)
            summary = store.record_verified(
                pmid="123", title="Study", run_id="run-a", verified=verified
            )
            store.close()

            self.assertTrue(summary.written)
            self.assertEqual(summary.relation_count, 4)
            con = sqlite3.connect(db_path)
            rows = con.execute(
                "SELECT candidate_id, write_status, relation_json "
                "FROM candidate_relations ORDER BY candidate_id"
            ).fetchall()
            con.close()

        self.assertEqual(len(rows), 4)
        statuses = {candidate_id: status for candidate_id, status, _ in rows}
        self.assertIn(statuses["c-ready"], {"IMPORT_READY", "HUMAN_REVIEW", "SEMANTIC_ONLY"})
        self.assertIn(statuses["c-review"], {"HUMAN_REVIEW", "SEMANTIC_ONLY"})
        self.assertIn(statuses["c-semantic"], {"HUMAN_REVIEW", "SEMANTIC_ONLY"})
        self.assertEqual(statuses["c-blocked"], "BLOCKED")
        payload = json.loads(rows[0][2])
        self.assertIn("factual_status", payload)
        self.assertIn("evidence_spans", payload)

    def test_repeated_candidate_key_updates_in_place(self):
        verified = KGVerifier(OfflineKG()).verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation("TP53 was associated with HCC.", candidate_id="c-1")],
            pmid="123",
            text="TITLE: Study\nABSTRACT: TP53 was associated with HCC.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "candidates.sqlite3"
            store = CandidateRelationStore(mode="sqlite", path=db_path)
            store.record_verified(pmid="123", title="v1", run_id="run-a", verified=verified)
            store.record_verified(pmid="123", title="v2", run_id="run-b", verified=verified)
            stats = store.stats()
            store.close()

        self.assertEqual(stats["entries"], 1)

    def test_candidate_versions_are_retained_as_separate_audit_rows(self):
        verified = KGVerifier(OfflineKG(), verification_policy="tiered-v2").verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation("TP53 was associated with HCC.", candidate_id="lineage-1")],
            pmid="123", text="TP53 was associated with HCC.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "candidates.sqlite3"
            store = CandidateRelationStore(mode="sqlite", path=db_path)
            store.record_verified(pmid="123", verified=verified)
            verified.relations[0].parent_version = 1
            verified.relations[0].candidate_version = 2
            store.record_verified(pmid="123", verified=verified)
            stats = store.stats()
            store.close()
        self.assertEqual(stats["entries"], 2)

    def test_out_of_scope_relations_are_counted_outside_review_queue(self):
        text = (
            "TITLE: TP53 in pulmonary fibrosis\nABSTRACT: RESULTS: "
            "TP53 was associated with HCC."
        )
        verified = KGVerifier(OfflineKG(), verification_policy="tiered-v2").verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation("TP53 was associated with HCC.", candidate_id="scope-1")],
            pmid="scope", text=text,
        )
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "candidates.sqlite3"
            store = CandidateRelationStore(mode="sqlite", path=db_path)
            summary = store.record_verified(pmid="scope", verified=verified)
            stats = store.stats()
            store.close()
            con = sqlite3.connect(db_path)
            scope_status = con.execute(
                "SELECT scope_status FROM candidate_relations"
            ).fetchone()[0]
            con.close()
        self.assertEqual(summary.out_of_scope_count, 1)
        self.assertEqual(summary.human_review_count, 0)
        self.assertEqual(scope_status, "OUT_OF_SCOPE")
        self.assertEqual(stats["by_scope_status"]["OUT_OF_SCOPE"], 1)

    def test_adjudication_lineage_survives_sqlite_relation_json(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: TP53 expression increased in HCC."
        base = relation("TP53 expression increased in HCC.")
        span_id = EvidencePackBuilder().build(base, text=text).spans[0].span_id
        verified = KGVerifier(OfflineKG(), verification_policy="tiered-v2").verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation(
                "TP53 expression increased in HCC.", candidate_id="audit-1",
                adjudication_verdict="SUPPORTED",
                adjudication_reason_code="EXPLICIT_DIRECT_RELATION",
                adjudication_confidence=0.95,
                supporting_span_ids=[span_id],
                adjudication={
                    "verdict": "SUPPORTED", "reason_code": "EXPLICIT_DIRECT_RELATION",
                    "confidence": 0.95, "supporting_span_ids": [span_id],
                    "model_id": "deepseek-test",
                },
            )],
            pmid="audit", text=text,
        )
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "candidates.sqlite3"
            store = CandidateRelationStore(mode="sqlite", path=db_path)
            store.record_verified(pmid="audit", verified=verified)
            store.close()
            con = sqlite3.connect(db_path)
            payload = json.loads(con.execute(
                "SELECT relation_json FROM candidate_relations"
            ).fetchone()[0])
            con.close()
        self.assertEqual(payload["adjudication_verdict"], "SUPPORTED")
        self.assertEqual(payload["supporting_span_ids"], [span_id])
        self.assertEqual(payload["adjudication"]["model_id"], "deepseek-test")


if __name__ == "__main__":
    unittest.main()
