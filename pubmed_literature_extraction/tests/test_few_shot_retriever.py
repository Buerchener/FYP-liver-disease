import json
import tempfile
import unittest
from pathlib import Path

from cognitive_agent.few_shot_retriever import (
    FewShotRetriever,
    detect_error_signatures,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def gold_row(pmid, title, relations, entities, study="experimental"):
    return {
        "pmid": pmid, "title": title, "in_scope": "True",
        "study_context": study, "review_status": "test",
        "entities": entities,
        "relations": relations,
        "negative_notes": [],
    }


class FewShotRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.gold = self.dir / "gold.jsonl"
        self.source = self.dir / "source.jsonl"
        text = (
            "TITLE: TP53 and liver fibrosis\nABSTRACT: "
            "RESULTS: TP53 expression was associated with liver fibrosis progression "
            "and serum ALT levels were measured."
        )
        rows = [
            gold_row("1001", "TP53 study", [{
                "subject": "TP53", "subject_type": "Gene",
                "predicate": "ASSOCIATED_WITH",
                "object": "liver fibrosis", "object_type": "Disease",
                "evidence": "TP53 expression was associated with liver fibrosis progression",
                "import_ready": True, "exclusion_reason": "",
            }], [
                {"mention": "TP53", "canonical": "TP53", "type": "Gene"},
                {"mention": "liver fibrosis", "canonical": "liver fibrosis", "type": "Disease"},
                {"mention": "ALT", "canonical": "ALT", "type": "Metabolite"},
                {"mention": "serum", "canonical": "serum", "type": "Tissue"},
            ]),
        ]
        write_jsonl(self.gold, rows)
        write_jsonl(self.source, [{
            "pmid": "1001", "title": "TP53 study",
            "abstract": (
                "RESULTS: TP53 expression was associated with liver fibrosis progression "
                "and serum ALT levels were measured."
            ),
        }])

    def tearDown(self):
        self.temp.cleanup()

    def make_retriever(self, **kwargs):
        return FewShotRetriever(
            pool_path=str(self.gold), source_path=str(self.source),
            max_examples=4, **kwargs,
        )

    def test_loads_positive_and_hard_negative_examples(self):
        retriever = self.make_retriever()
        audit = retriever.audit()
        self.assertGreaterEqual(audit["positive_count"], 1)
        # ALT and serum co-occur with schema-compatible types but no gold
        # relation: a hard NO_RELATION demonstration.
        self.assertGreaterEqual(audit["hard_negative_count"], 1)

    def test_retrieval_returns_positive_for_matching_pair(self):
        retriever = self.make_retriever()
        examples = retriever.retrieve(
            subject="STAT3", subject_type="Gene",
            object_="liver fibrosis", object_type="Disease",
            allowed_predicates=["ASSOCIATED_WITH", "PROGNOSTIC_IN"],
            source_predicates=["ASSOCIATED_WITH"],
            focus_sentence="STAT3 expression was associated with liver fibrosis.",
            exclude_pmid="9999",
        )
        labels = [example.predicate for example in examples]
        self.assertIn("ASSOCIATED_WITH", labels)

    def test_current_pmid_is_never_used_as_its_own_example(self):
        retriever = self.make_retriever()
        examples = retriever.retrieve(
            subject="TP53", subject_type="Gene",
            object_="liver fibrosis", object_type="Disease",
            allowed_predicates=["ASSOCIATED_WITH"],
            source_predicates=["ASSOCIATED_WITH"],
            focus_sentence="TP53 expression was associated with liver fibrosis progression.",
            exclude_pmid="1001",
        )
        self.assertEqual(examples, [])

    def test_exclude_pmids_set_is_honoured(self):
        retriever = self.make_retriever(exclude_pmids={"1001"})
        audit = retriever.audit()
        self.assertEqual(audit["example_count"], 0)


class ErrorSignatureTests(unittest.TestCase):
    def test_cohort_context_detected(self):
        signatures = detect_error_signatures(
            "Serum ALT levels were measured in patients with NAFLD."
        )
        self.assertIn("cohort_context_false_association", signatures)

    def test_measurement_without_claim(self):
        signatures = detect_error_signatures(
            "Expression levels of TNF were assessed in all samples."
        )
        self.assertIn("measurement_without_claim", signatures)

    def test_background_relation_by_section_and_phrasing(self):
        signatures = detect_error_signatures(
            "It is known that TNF plays an important role in inflammation.",
        )
        self.assertIn("background_relation", signatures)
        signatures = detect_error_signatures(
            "TNF is associated with inflammation.", section="INTRODUCTION",
        )
        self.assertIn("background_relation", signatures)

    def test_prediction_only(self):
        signatures = detect_error_signatures(
            "Molecular docking predicted TNF binding to the receptor."
        )
        self.assertIn("prediction_not_observation", signatures)

    def test_causality_is_not_association(self):
        signatures = detect_error_signatures(
            "TNF signaling leads to hepatocyte apoptosis."
        )
        self.assertIn("association_vs_causality", signatures)

    def test_direction_confusion_passive(self):
        signatures = detect_error_signatures(
            "TNF expression is regulated by NF-kB."
        )
        self.assertIn("direction_confusion", signatures)

    def test_method_only(self):
        signatures = detect_error_signatures(
            "Cells were stained with anti-TNF antibody for analysis."
        )
        self.assertIn("method_only_relation", signatures)

    def test_mere_cooccurrence_fallback(self):
        signatures = detect_error_signatures(
            "TNF and IL-6 levels were measured."
        )
        self.assertEqual(signatures, ["mere_cooccurrence"])
        # A relational verb suppresses the fallback category.
        signatures = detect_error_signatures(
            "TNF was associated with IL-6 levels.",
        )
        self.assertNotIn("mere_cooccurrence", signatures)


class ErrorPatternRetrievalTests(unittest.TestCase):
    """Same-class hard negative retrieval via deterministic signatures."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.gold = self.dir / "gold.jsonl"
        self.source = self.dir / "source.jsonl"
        # Article A: clean direct finding (positive pool entry).
        # Article B: cohort-flavoured co-occurrence with no gold relation
        # (same-class hard negative pool entry).
        write_jsonl(self.gold, [
            gold_row("2001", "A study", [{
                "subject": "TP53", "subject_type": "Gene",
                "predicate": "ASSOCIATED_WITH",
                "object": "liver fibrosis", "object_type": "Disease",
                "evidence": "TP53 was associated with liver fibrosis.",
                "import_ready": True, "exclusion_reason": "",
            }], [
                {"mention": "TP53", "canonical": "TP53", "type": "Gene"},
                {"mention": "liver fibrosis", "canonical": "liver fibrosis", "type": "Disease"},
            ]),
            gold_row("2002", "B study", [], [
                {"mention": "ALT", "canonical": "ALT", "type": "Metabolite"},
                {"mention": "NAFLD", "canonical": "NAFLD", "type": "Disease"},
            ]),
        ])
        write_jsonl(self.source, [
            {"pmid": "2001", "title": "A study",
             "abstract": "RESULTS: TP53 was associated with liver fibrosis."},
            {"pmid": "2002", "title": "B study",
             "abstract": "RESULTS: Serum ALT levels were measured in patients with NAFLD."},
        ])

    def tearDown(self):
        self.temp.cleanup()

    def make_retriever(self, **kwargs):
        return FewShotRetriever(
            str(self.gold), source_path=str(self.source),
            max_examples=3, **kwargs,
        )

    def test_same_class_hard_negative_fills_slot(self):
        retriever = self.make_retriever()
        examples = retriever.retrieve(
            subject="ALT", subject_type="Metabolite",
            object_="NAFLD", object_type="Disease",
            allowed_predicates=["ASSOCIATED_WITH"],
            source_predicates=["ASSOCIATED_WITH"],
            focus_sentence="Serum ALT levels were measured in patients with NAFLD.",
            exclude_pmid="9999", mode="error_pattern",
        )
        hard_negatives = [
            example for example in examples if example.predicate == "NO_RELATION"
        ]
        self.assertTrue(hard_negatives)
        self.assertEqual(
            hard_negatives[0].error_category,
            "cohort_context_false_association",
        )

    def test_positive_example_prefers_clean_direct_finding(self):
        retriever = self.make_retriever()
        examples = retriever.retrieve(
            subject="TP53", subject_type="Gene",
            object_="liver fibrosis", object_type="Disease",
            allowed_predicates=["ASSOCIATED_WITH"],
            source_predicates=["ASSOCIATED_WITH"],
            focus_sentence="TP53 was associated with liver fibrosis.",
            exclude_pmid="9999", mode="error_pattern",
        )
        positives = [
            example for example in examples if example.predicate != "NO_RELATION"
        ]
        self.assertTrue(positives)
        self.assertEqual(positives[0].sentence_error_signatures, "")

    def test_render_labels_error_category(self):
        retriever = self.make_retriever()
        examples = retriever.retrieve(
            subject="ALT", subject_type="Metabolite",
            object_="NAFLD", object_type="Disease",
            allowed_predicates=["ASSOCIATED_WITH"],
            source_predicates=["ASSOCIATED_WITH"],
            focus_sentence="Serum ALT levels were measured in patients with NAFLD.",
            exclude_pmid="9999", mode="error_pattern",
        )
        negative = next(
            example for example in examples if example.predicate == "NO_RELATION"
        )
        self.assertIn("cohort_context_false_association", negative.render())


if __name__ == "__main__":
    unittest.main()
