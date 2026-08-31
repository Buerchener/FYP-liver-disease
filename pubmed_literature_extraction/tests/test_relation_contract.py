import unittest

from cognitive_agent.relation_contract import (
    RelationCandidateProjector,
    normalize_relation_semantics,
    stable_candidate_id,
)
from cognitive_agent.verifier import KGVerifier


class OfflineKG:
    is_connected = False


class RelationContractTests(unittest.TestCase):
    def test_predicate_specific_direction_contract_and_stable_lineage(self):
        associated = normalize_relation_semantics({
            "subject": "TP53", "subject_type": "Gene",
            "predicate": "ASSOCIATED_WITH", "object": "HCC",
            "object_type": "Disease", "direction": "negative",
            "evidence": "TP53 was negatively associated with HCC.",
        })
        encoded = normalize_relation_semantics({
            "subject": "TP53", "subject_type": "Gene", "predicate": "ENCODES",
            "object": "p53", "object_type": "Protein", "direction": "unknown",
            "evidence": "TP53 encodes p53.",
        })
        self.assertEqual(associated["relation_direction"], "NON_DIRECTIONAL")
        self.assertEqual(associated["association_sign"], "NEGATIVE")
        self.assertEqual(encoded["relation_direction"], "SUBJECT_TO_OBJECT")
        self.assertEqual(encoded["association_sign"], "UNKNOWN")
        changed_evidence = {**associated, "evidence": "A second exact source quote."}
        self.assertEqual(
            stable_candidate_id(associated), stable_candidate_id(changed_evidence)
        )

    def test_expression_and_activity_changes_do_not_reverse_relation(self):
        expression = normalize_relation_semantics({
            "predicate": "EXPRESSED_IN", "direction": "decrease",
            "evidence": "Protein expression decreased in liver tissue.",
        })
        activity = normalize_relation_semantics({
            "predicate": "PARTICIPATES_IN", "direction": "increase",
            "evidence": "The pathway activity was activated.",
        })
        self.assertEqual(expression["relation_direction"], "SUBJECT_TO_OBJECT")
        self.assertEqual(expression["expression_change"], "DOWN")
        self.assertEqual(activity["relation_direction"], "SUBJECT_TO_OBJECT")
        self.assertEqual(activity["activity_change"], "ACTIVATED")

    def test_article_abbreviation_family_deduplicates_relation_core(self):
        text = (
            "Metabolic dysfunction-associated fatty liver disease (MAFLD) was studied. "
            "Testosterone was associated with MAFLD."
        )
        entities = [
            {"mention": "Testosterone", "type": "Metabolite", "attributes": {}},
            {"mention": "MAFLD", "type": "Disease", "attributes": {}},
            {"mention": "Metabolic dysfunction-associated fatty liver disease", "type": "Disease", "attributes": {}},
        ]
        relations = [
            {"subject": "Testosterone", "subject_type": "Metabolite", "predicate": "ASSOCIATED_WITH",
             "object": "MAFLD", "object_type": "Disease", "evidence": "Testosterone was associated with MAFLD."},
            {"subject": "Testosterone", "subject_type": "Metabolite", "predicate": "ASSOCIATED_WITH",
             "object": "Metabolic dysfunction-associated fatty liver disease", "object_type": "Disease",
             "evidence": "Metabolic dysfunction-associated fatty liver disease (MAFLD) was studied."},
        ]
        result = RelationCandidateProjector().consolidate(relations, text=text, entities=entities)
        self.assertEqual(len(result), 1)

    def test_attribute_and_top_level_relations_share_one_candidate_ledger(self):
        text = (
            "Bile salt export pump (BSEP) is associated with hepatocellular "
            "carcinoma (HCC)."
        )
        entities = [
            {
                "mention": "BSEP", "type": "Protein",
                "attributes": {"protein_name": "BSEP", "associated_with": [{
                    "target_entity": "HCC", "target_type": "disease",
                    "direction": "positive", "negated": False, "uncertain": False,
                    "evidence": text,
                }]},
            },
            {"mention": "HCC", "type": "Disease", "attributes": {}},
        ]
        top = [{
            "subject": "BSEP", "subject_type": "Protein",
            "predicate": "ASSOCIATED_WITH", "object": "HCC",
            "object_type": "Disease", "direction": "positive",
            "evidence": text,
        }]
        result = RelationCandidateProjector().project(entities, top, text)

        self.assertEqual(len(result.relations), 1)
        self.assertEqual(result.audit["attribute_relation_count"], 1)
        self.assertEqual(result.audit["duplicates_merged"], 1)
        self.assertEqual(
            result.relations[0]["provenance"],
            ["entity_attribute:associated_with", "top_level"],
        )
        self.assertTrue(result.relations[0]["evidence_contiguous"])

    def test_non_human_relation_is_semantic_only_not_semantically_deleted(self):
        text = (
            "TITLE: Liver fibrosis in mice\n"
            "ABSTRACT: MFN2 is associated with liver fibrosis."
        )
        entities = [
            {"mention": "MFN2", "type": "Protein", "attributes": {}},
            {"mention": "liver fibrosis", "type": "Disease", "attributes": {}},
        ]
        relations = [{
            "subject": "MFN2", "subject_type": "Protein",
            "predicate": "ASSOCIATED_WITH", "object": "liver fibrosis",
            "object_type": "Disease", "direction": "positive",
            "evidence": "MFN2 is associated with liver fibrosis.",
            "species": "Mus musculus", "negated": False, "uncertain": False,
        }]
        result = KGVerifier(OfflineKG()).verify(entities, relations, text=text)
        checked = result.relations[0]

        self.assertEqual(checked.semantic_status, "ACCEPTED")
        self.assertEqual(checked.write_status, "SEMANTIC_ONLY")
        self.assertFalse(checked.import_ready)
        self.assertIn("non_human", checked.write_reasons)
        self.assertIn("non_human_article", checked.write_reasons)


if __name__ == "__main__":
    unittest.main()
