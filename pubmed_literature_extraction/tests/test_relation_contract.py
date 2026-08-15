import unittest

from cognitive_agent.relation_contract import RelationCandidateProjector
from cognitive_agent.verifier import KGVerifier


class OfflineKG:
    is_connected = False


class RelationContractTests(unittest.TestCase):
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
