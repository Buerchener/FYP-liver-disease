import json
import unittest
from pathlib import Path

from cognitive_agent.evidence_contrast import find_evidence_conflict
from cognitive_agent.evidence_pack import EvidencePackBuilder
from cognitive_agent.schema.schema_profile import schema_profile_from_dict
from cognitive_agent.verifier import KGVerifier


class OfflineGraph:
    is_connected = False


def profile():
    return schema_profile_from_dict({
        "artifact_type": "dataset_schema_profile",
        "name": "ContrastFixture",
        "contract_version": "v1",
        "semantic_target": "RELATION_TRUTH",
        "entity_types": ["Gene", "Disease"],
        "predicates": {
            "ASSOCIATED_WITH": {
                "allowed_signatures": [["Gene", "Disease"]],
                "explicit_patterns": [r"\bassociat(?:e|es|ed|ing)\s+with\b"],
                "weak_patterns": [],
                # This is intentionally not treated as logical refutation by itself.
                "exclusion_patterns": [r"\bnot associated with\b"],
            },
        },
    })


def relation(**changes):
    value = {
        "subject": "TP53", "subject_type": "Gene",
        "predicate": "ASSOCIATED_WITH",
        "object": "HCC", "object_type": "Disease",
    }
    value.update(changes)
    return value


class EvidenceContrastTests(unittest.TestCase):
    def detect(self, text, rel=None):
        index = EvidencePackBuilder().index(text)
        return find_evidence_conflict(
            rel or relation(), source_index=index, profile=profile(),
            subject_aliases=["TP53"], object_aliases=["HCC"],
        )

    def test_pairs_explicit_support_and_scoped_negation_in_results(self):
        text = (
            "ABSTRACT: RESULTS: TP53 was associated with HCC; "
            "TP53 was not associated with HCC."
        )
        result = self.detect(text)
        self.assertEqual(result["status"], "CONTESTED")
        self.assertEqual(result["support"]["text"], "TP53 was associated with HCC")
        self.assertEqual(result["counterevidence"]["text"], "TP53 was not associated with HCC.")
        self.assertEqual(result["support"]["section"], "RESULTS")
        self.assertEqual(
            text[result["counterevidence"]["char_start"]:result["counterevidence"]["char_end"]],
            result["counterevidence"]["text"],
        )

    def test_negation_may_have_bounded_adverb_between_it_and_trigger(self):
        result = self.detect(
            "ABSTRACT: CONCLUSIONS: TP53 is associated with HCC. "
            "TP53 was not statistically significantly associated with HCC."
        )
        self.assertEqual(result["status"], "CONTESTED")

    def test_other_pair_and_unrelated_measured_endpoint_do_not_conflict(self):
        self.assertEqual(self.detect(
            "ABSTRACT: RESULTS: TP53 was associated with HCC; "
            "EGFR was not associated with ALT."
        ), {})
        self.assertEqual(self.detect(
            "ABSTRACT: RESULTS: TP53 was not associated with ALT, "
            "and HCC was measured."
        ), {})
        self.assertEqual(self.detect(
            "ABSTRACT: RESULTS: TP53 was associated with HCC. "
            "TP53 was not associated with ALT and HCC was measured."
        ), {})
        self.assertEqual(self.detect(
            "ABSTRACT: RESULTS: TP53 was associated with HCC. "
            "TP53 was quantified and ALT was not associated with HCC."
        ), {})

    def test_not_only_and_background_or_methods_are_not_counterevidence(self):
        self.assertEqual(self.detect(
            "ABSTRACT: RESULTS: TP53 was associated with HCC; "
            "TP53 was not only associated with HCC."
        ), {})
        self.assertEqual(self.detect(
            "ABSTRACT: BACKGROUND: TP53 was associated with HCC. "
            "METHODS: TP53 was not associated with HCC."
        ), {})

    def test_signature_mismatch_is_not_compared(self):
        self.assertEqual(self.detect(
            "ABSTRACT: RESULTS: TP53 was associated with HCC; "
            "TP53 was not associated with HCC.",
            relation(object_type="ChemicalEntity"),
        ), {})

    def test_detected_conflict_is_a_verification_and_write_gate(self):
        supporting_quote = "TP53 was associated with HCC."
        text = "ABSTRACT: RESULTS: " + supporting_quote + " TP53 was not associated with HCC."
        candidate = relation(evidence=supporting_quote, claim_role="CURRENT_FINDING")
        entities = [{"mention": "TP53", "type": "Gene"}, {"mention": "HCC", "type": "Disease"}]
        baseline = KGVerifier(
            OfflineGraph(), schema_profile=profile(), dataset_only=True, evidence_contrast=False,
        )
        proposed = KGVerifier(OfflineGraph(), schema_profile=profile(), dataset_only=True)
        self.assertEqual(baseline.verify(entities, [candidate], text=text).relations[0].semantic_status, "ACCEPTED")
        checked = proposed.verify(entities, [candidate], text=text)
        relation_result = checked.relations[0]
        self.assertEqual(relation_result.semantic_status, "REVIEW")
        self.assertIn("relation_counterevidence", relation_result.semantic_reasons)
        self.assertFalse(relation_result.import_ready)
        self.assertEqual(checked.summary["contested_relations"], 1)
        trace = relation_result.contrastive_evidence["counterevidence"]
        self.assertEqual(text[trace["negation_start"]:trace["negation_end"]], "not")

    def test_real_abstract_denial_about_a_different_outcome_is_not_a_conflict(self):
        sources = Path(__file__).resolve().parents[1] / "extraction_output" / "pubmed_converted_500.jsonl"
        with sources.open(encoding="utf-8") as rows:
            article = next(item for item in map(json.loads, rows) if str(item["pmid"]) == "41846463")
        text = f"TITLE: {article['title']}\nABSTRACT: {article['abstract']}"
        candidate = relation(
            subject="RRT on days 1 to 3", subject_type="Exposure",
            object="higher hazard of LT", object_type="Outcome",
            evidence="RRT on days 1 to 3 was associated with higher hazard of LT",
            claim_role="CURRENT_FINDING",
        )
        payload = profile().manifest()
        payload["entity_types"] = ["Exposure", "Outcome"]
        payload["predicates"]["ASSOCIATED_WITH"]["allowed_signatures"] = [["Exposure", "Outcome"]]
        verifier = KGVerifier(OfflineGraph(), schema_profile=schema_profile_from_dict(payload), dataset_only=True)
        entities = [
            {"mention": candidate[side], "type": candidate[f"{side}_type"]}
            for side in ("subject", "object")
        ]
        checked = verifier.verify(entities, [candidate], text=text).relations[0]
        self.assertEqual(checked.semantic_status, "ACCEPTED")
        self.assertFalse(checked.contrastive_evidence)


if __name__ == "__main__":
    unittest.main()
