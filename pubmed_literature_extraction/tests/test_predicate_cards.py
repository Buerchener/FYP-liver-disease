import unittest

from cognitive_agent.schema.predicate_cards import (
    RELATION_CARDS,
    match_relation_card,
)


class PredicateCardTests(unittest.TestCase):
    def test_all_eight_predicates_have_complete_cards(self):
        self.assertEqual(len(RELATION_CARDS), 8)
        for predicate, card in RELATION_CARDS.items():
            with self.subTest(predicate=predicate):
                self.assertTrue(card.description)
                self.assertTrue(card.allowed_signatures)
                self.assertTrue(card.trigger_patterns)
                self.assertTrue(card.high_precision_patterns)
                self.assertIn(card.relation_direction, {"NON_DIRECTIONAL", "SUBJECT_TO_OBJECT"})

    def test_explicit_examples_match_their_predicate_cards(self):
        cases = [
            ("ASSOCIATED_WITH", "Gene", "Disease", "TP53 was associated with HCC."),
            ("ASSOCIATED_WITH_METABOLITE", "Gene", "Metabolite", "GENE1 was associated with lactate."),
            ("INTERACTS_WITH", "Protein", "Protein", "Protein A binds to protein B."),
            ("ENCODES", "Gene", "Protein", "TP53 encodes p53."),
            ("PARTICIPATES_IN", "Protein", "Pathway", "p53 participates in apoptosis."),
            ("EXPRESSED_IN", "Gene", "Tissue", "ALB is expressed in liver tissue."),
            ("PROGNOSTIC_IN", "Gene", "Disease", "TP53 is prognostic in HCC."),
            ("PROGRESSES_TO", "Disease", "Disease", "MASLD progresses to MASH."),
        ]
        for predicate, subject_type, object_type, evidence in cases:
            with self.subTest(predicate=predicate):
                result = match_relation_card(
                    predicate, subject_type, object_type, evidence,
                    endpoints_linked=True,
                )
                self.assertEqual(result["match"], "EXPLICIT")

    def test_type_mismatch_and_boundary_conflict_are_explicit(self):
        mismatch = match_relation_card(
            "ENCODES", "Disease", "Protein", "HCC encodes p53."
        )
        conflict = match_relation_card(
            "PROGNOSTIC_IN", "Gene", "Disease",
            "TP53 is a potential target for diagnosis and treatment of HCC.",
        )
        self.assertEqual(mismatch["match"], "NOT_APPLICABLE")
        self.assertEqual(conflict["match"], "CONFLICT")

    def test_biomedical_noun_phrase_triggers_are_explicit(self):
        cases = [
            (
                "INTERACTS_WITH", "Protein", "Protein",
                "interactions between CD155 and TIGIT",
            ),
            (
                "EXPRESSED_IN", "Protein", "CellType",
                "cell-type-specific expression patterns of CCND1 in hepatocytes",
            ),
            (
                "PARTICIPATES_IN", "Protein", "Pathway",
                "CCND1 and IL7R are core JAK-STAT pathway genes",
            ),
        ]
        for predicate, subject_type, object_type, evidence in cases:
            with self.subTest(predicate=predicate):
                matched = match_relation_card(
                    predicate, subject_type, object_type, evidence,
                    endpoints_linked=True,
                )
                self.assertEqual(matched["match"], "EXPLICIT")


if __name__ == "__main__":
    unittest.main()
