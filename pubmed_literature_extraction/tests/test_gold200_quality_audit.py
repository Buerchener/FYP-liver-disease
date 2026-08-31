import tempfile
import unittest
from pathlib import Path

from scripts.audit_gold200_quality import (
    CLAIM_ROLES,
    ReviewCache,
    build_review_ledgers,
    direction_semantics,
    locate_evidence,
    reject_invalid_missing_proposals,
    select_documents,
    static_audit,
    validate_review_payload,
)


def source(pmid="1", abstract="TP53 was associated with HCC."):
    return {"pmid": pmid, "title": "Study", "abstract": abstract}


def relation(**overrides):
    value = {
        "subject": "TP53", "subject_type": "Gene",
        "predicate": "ASSOCIATED_WITH",
        "object": "HCC", "object_type": "Disease",
        "direction": "positive",
        "evidence": "TP53 was associated with HCC.",
        "import_ready": False, "exclusion_reason": "",
    }
    value.update(overrides)
    return value


def gold(pmid="1", relations=None):
    return {
        "pmid": pmid, "title": "Study", "in_scope": True,
        "study_context": "human_observational",
        "entities": [
            {"mention": "TP53", "canonical": "TP53", "type": "Gene"},
            {"mention": "HCC", "canonical": "HCC", "type": "Disease"},
        ],
        "relations": list(relations if relations is not None else [relation()]),
        "negative_notes": [], "review_status": "test",
    }


class Gold200QualityAuditTests(unittest.TestCase):
    def test_direction_semantics_does_not_merge_sign_and_change(self):
        self.assertEqual(direction_semantics("positive"), "ASSOCIATION_SIGN")
        self.assertEqual(direction_semantics("increase"), "CHANGE_DIRECTION")
        self.assertEqual(direction_semantics("none"), "NON_DIRECTIONAL")

    def test_evidence_span_uses_combined_source_offsets(self):
        text = "TITLE: Study\nABSTRACT: TP53 was associated with HCC."
        spans = locate_evidence("TP53 was associated with HCC.", text)
        self.assertEqual(len(spans), 1)
        self.assertEqual(text[spans[0]["start"]:spans[0]["end"]], spans[0]["text"])

    def test_static_audit_augments_without_changing_frozen_relation(self):
        rows = [gold()]
        _, ledger, draft, summary = static_audit(rows, [source()])
        self.assertEqual(rows[0]["relations"][0], relation())
        self.assertEqual(ledger[0]["suggested_interface_fields"]["claim_role"], "CURRENT_FINDING")
        self.assertEqual(draft[0]["relations"][0]["gold_write_status"], "WRITE_CONTRACT")
        self.assertEqual(draft[0]["relations"][0]["direction_semantics"], "ASSOCIATION_SIGN")
        self.assertFalse(draft[0]["adjudication_complete"])
        self.assertFalse(summary["structural_valid"])
        self.assertIn("document_count_mismatch", {item["code"] for item in summary["structural_errors"]})

    def test_zero_relation_strong_cue_is_l1(self):
        rows = [gold(relations=[])]
        docs, _, _, _ = static_audit(rows, [source()])
        self.assertEqual(docs[0]["audit_tier"], "L1_HIGH_RISK")
        self.assertIn("zero_relation_with_strong_cue", docs[0]["risk_flags"])
        self.assertEqual(select_documents("l1", rows, docs), rows)

    def test_review_payload_requires_full_ids_and_exact_quotes(self):
        payload = {
            "document_decision": {"in_scope": True, "zero_relation_annotation": "CONFIRMED"},
            "relation_reviews": [{
                "relation_id": "1:r1", "decision": "CONFIRM",
                "claim_role": "CURRENT_FINDING", "direction": "positive",
                "import_ready": "NO", "source_quote": "TP53 was associated with HCC.",
                "reason": "supported",
            }],
            "missing_relations": [], "expert_review_required": False,
        }
        self.assertTrue(CLAIM_ROLES)
        self.assertEqual(
            validate_review_payload(payload, {"1:r1"}, "TP53 was associated with HCC."),
            [],
        )
        payload["relation_reviews"][0]["source_quote"] = "not in source"
        self.assertIn(
            "source_quote_not_exact:1:r1",
            validate_review_payload(payload, {"1:r1"}, "TP53 was associated with HCC."),
        )

    def test_invalid_missing_proposal_does_not_discard_relation_reviews(self):
        payload = {
            "relation_reviews": [{"relation_id": "1:r1"}],
            "missing_relations": [{"predicate": "BAD"}, {"predicate": "OK"}],
        }
        sanitized, critical, rejected = reject_invalid_missing_proposals(
            payload,
            ["missing_relation_signature_invalid:0", "source_quote_not_exact:1:r1"],
        )
        self.assertEqual(rejected, 1)
        self.assertEqual(sanitized["missing_relations"], [{"predicate": "OK"}])
        self.assertEqual(critical, ["source_quote_not_exact:1:r1"])

    def test_review_cache_only_replays_ok_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ReviewCache(Path(tmp) / "reviews.sqlite3")
            reviewer = type("R", (), {
                "name": "gemini",
                "spec": type("S", (), {"model_id": "m"})(),
            })()
            cache.put("ok", reviewer, {"status": "OK", "review": {"x": 1}})
            cache.put("bad", reviewer, {"status": "INVALID_RESPONSE", "review": {}})
            self.assertEqual(cache.get("ok")["review"], {"x": 1})
            self.assertEqual(cache.get("bad")["status"], "INVALID_RESPONSE")
            self.assertIsNone(cache.get("bad", retry_invalid=True))
            cache.close()

    def test_review_ledgers_never_apply_automatic_gold_changes(self):
        documents = [{
            "pmid": "1", "zero_relation": False, "audit_tier": "L1_HIGH_RISK",
            "risk_flags": ["historical_import_ready"],
        }]
        relations = [{
            "pmid": "1", "relation_id": "1:r1", "audit_tier": "L1_HIGH_RISK",
            "risk_flags": ["historical_import_ready"],
            "current_annotation": relation(import_ready=True),
        }]
        decision = {
            "relation_id": "1:r1", "decision": "CONFIRM",
            "claim_role": "CURRENT_FINDING", "direction": "positive",
            "import_ready": "YES", "source_quote": "TP53 was associated with HCC.",
        }
        results = [
            {"pmid": "1", "reviewer": reviewer, "status": "OK", "review": {
                "document_decision": {"zero_relation_annotation": "CONFIRMED"},
                "relation_reviews": [decision], "missing_relations": [],
            }}
            for reviewer in ("gemini", "deepseek")
        ]
        _, enriched, queue, summary = build_review_ledgers(documents, relations, results)
        self.assertEqual(enriched[0]["model_review_status"], "CONSENSUS")
        self.assertEqual(queue[0]["required_action"], "EXPERT_REVIEW_REQUIRED")
        self.assertEqual(summary["automatic_gold_changes"], 0)

    def test_missing_relation_direction_disagreement_is_not_consensus(self):
        documents = [{
            "pmid": "1", "zero_relation": True, "audit_tier": "L1_HIGH_RISK",
            "risk_flags": ["zero_relation_with_strong_cue"],
        }]
        base = {
            "subject": "iodine", "subject_type": "Metabolite",
            "predicate": "ASSOCIATED_WITH", "object": "fibrosis",
            "object_type": "Disease", "claim_role": "CURRENT_FINDING",
            "source_quote": "iodine was associated with fibrosis",
        }
        results = []
        for reviewer, direction in (("gemini", "positive"), ("deepseek", "negative")):
            results.append({
                "pmid": "1", "reviewer": reviewer, "status": "OK", "review": {
                    "document_decision": {"zero_relation_annotation": "LIKELY_MISSING"},
                    "relation_reviews": [],
                    "missing_relations": [{**base, "direction": direction}],
                },
            })
        _, _, queue, summary = build_review_ledgers(documents, [], results)
        proposal = next(item for item in queue if item["item_type"] == "MISSING_RELATION_PROPOSAL")
        self.assertEqual(proposal["model_review_status"], "DISAGREEMENT")
        self.assertEqual(summary["missing_relation_consensus"], 0)


if __name__ == "__main__":
    unittest.main()
