import json
import tempfile
import unittest
from pathlib import Path

from cognitive_agent.biored_adapter import BioREDPredicateRegistry
from cognitive_agent.schema.schema_profile import (
    liverkg_schema_profile,
    load_schema_profile,
    schema_profile_from_dict,
)
from cognitive_agent.verifier import KGVerifier


class OfflineKG:
    is_connected = False


def custom_payload():
    return {
        "artifact_type": "dataset_schema_profile",
        "name": "ToyDrugDataset",
        "contract_version": "toy-drug-v1",
        "semantic_target": "RELATION_TRUTH",
        "entity_validation": "source_grounded",
        "article_quality_mode": "none",
        "entity_types": ["ChemicalEntity", "DiseaseOrPhenotypicFeature"],
        "predicates": {
            "CAUSES": {
                "allowed_signatures": [
                    ["ChemicalEntity", "DiseaseOrPhenotypicFeature"],
                ],
                "symmetric": False,
                "relation_direction": "SUBJECT_TO_OBJECT",
                "explicit_patterns": [r"\bcaus(?:e|es|ed|ing)\b"],
                "weak_patterns": [r"\binduc(?:e|es|ed|ing)\b"],
                "exclusion_patterns": [r"\bdid not cause\b"],
            }
        },
    }


class SchemaProfileTests(unittest.TestCase):
    def test_default_profile_preserves_liverkg_contract(self):
        profile = liverkg_schema_profile()
        self.assertTrue(profile.valid("ASSOCIATED_WITH", "Gene", "Disease"))
        self.assertFalse(profile.valid("CAUSES", "ChemicalEntity", "Disease"))
        self.assertEqual(profile.semantic_target, "CURRENT_FINDING")
        self.assertFalse(profile.manifest()["authorizes_neo4j_write"])

    def test_json_profile_changes_candidate_validation_and_matcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy-schema.json"
            path.write_text(json.dumps(custom_payload()), encoding="utf-8")
            profile = load_schema_profile(path)

        support = profile.support_match(
            "CAUSES",
            "ChemicalEntity",
            "DiseaseOrPhenotypicFeature",
            "Aspirin causes gastritis.",
            subject_aliases=["Aspirin"],
            object_aliases=["gastritis"],
        )
        self.assertTrue(profile.valid(
            "CAUSES", "ChemicalEntity", "DiseaseOrPhenotypicFeature",
        ))
        self.assertEqual(support["match"], "EXPLICIT")

    def test_verifier_uses_custom_profile_but_not_as_write_permission(self):
        profile = schema_profile_from_dict(custom_payload())
        verifier = KGVerifier(
            OfflineKG(), verification_policy="tiered-v2", schema_profile=profile,
        )
        text = "Aspirin causes gastritis."
        result = verifier.verify(
            [
                {"mention": "Aspirin", "type": "ChemicalEntity", "attributes": {}},
                {
                    "mention": "gastritis",
                    "type": "DiseaseOrPhenotypicFeature",
                    "attributes": {},
                },
            ],
            [{
                "subject": "Aspirin",
                "subject_type": "ChemicalEntity",
                "predicate": "CAUSES",
                "object": "gastritis",
                "object_type": "DiseaseOrPhenotypicFeature",
                "evidence": text,
                "claim_role": "CURRENT_FINDING",
            }],
            text=text,
        )
        relation = result.relations[0]
        self.assertTrue(relation.candidate_schema_valid)
        self.assertEqual(relation.relation_card_match, "EXPLICIT")
        self.assertEqual(relation.schema_profile_name, "ToyDrugDataset")
        self.assertFalse(relation.write_contract_valid)
        self.assertFalse(relation.import_ready)
        self.assertFalse(result.summary["schema_profile"]["authorizes_neo4j_write"])

    def test_wrong_dataset_signature_remains_schema_mismatch(self):
        profile = schema_profile_from_dict(custom_payload())
        self.assertFalse(profile.valid(
            "CAUSES", "DiseaseOrPhenotypicFeature", "ChemicalEntity",
        ))
        support = profile.support_match(
            "CAUSES",
            "DiseaseOrPhenotypicFeature",
            "ChemicalEntity",
            "Gastritis causes aspirin use.",
            subject_aliases=["Gastritis"],
            object_aliases=["aspirin"],
        )
        self.assertEqual(support["match"], "CONFLICT")

    def test_profile_rejects_invalid_regex_and_unknown_entity_type(self):
        payload = custom_payload()
        payload["predicates"]["CAUSES"]["explicit_patterns"] = ["("]
        with self.assertRaises(ValueError):
            schema_profile_from_dict(payload)

        payload = custom_payload()
        payload["predicates"]["CAUSES"]["allowed_signatures"] = [
            ["ChemicalEntity", "UnknownType"],
        ]
        with self.assertRaises(ValueError):
            schema_profile_from_dict(payload)

    def test_biored_registry_exposes_same_profile_interface(self):
        registry = BioREDPredicateRegistry({
            "Bind": {("GeneOrGeneProduct", "ChemicalEntity")},
        })
        profile = registry.to_schema_profile(source="train:fixture")
        self.assertTrue(profile.valid("Bind", "ChemicalEntity", "GeneOrGeneProduct"))
        self.assertEqual(profile.semantic_target, "RELATION_TRUTH")
        self.assertEqual(profile.entity_validation, "given_entity")
        support = profile.support_match(
            "Bind",
            "GeneOrGeneProduct",
            "ChemicalEntity",
            "EGFR binds gefitinib.",
            subject_aliases=["EGFR"],
            object_aliases=["gefitinib"],
        )
        self.assertEqual(support["match"], "EXPLICIT")
        self.assertFalse(profile.manifest()["authorizes_neo4j_write"])


if __name__ == "__main__":
    unittest.main()
