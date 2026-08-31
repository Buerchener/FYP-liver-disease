import sqlite3
import tempfile
import unittest
from pathlib import Path

from cognitive_agent.candidate_store import CandidateRelationStore
from cognitive_agent.decision_engine import DecisionEngine
from cognitive_agent.memory.kg_memory import KGMemory
from cognitive_agent.schema.write_contract import write_contract_assessment
from cognitive_agent.verifier import KGVerifier


class OfflineKG:
    is_connected = False


def entity(mention, entity_type):
    return {"mention": mention, "type": entity_type, "attributes": {}}


def relation(**overrides):
    payload = {
        "subject": "TP53",
        "subject_type": "Gene",
        "predicate": "ASSOCIATED_WITH",
        "object": "HCC",
        "object_type": "Disease",
        "evidence": "TP53 was associated with HCC.",
        "direction": "positive",
        "negated": False,
        "uncertain": False,
    }
    payload.update(overrides)
    return payload


class MainKGWriteContractTests(unittest.TestCase):
    def verifier(self):
        return KGVerifier(OfflineKG(), verification_policy="tiered-v2")

    def test_broad_literature_relation_is_preserved_but_never_write_eligible(self):
        text = "TITLE: Study\nABSTRACT: RESULTS: BSEP was associated with HCC."
        verified = self.verifier().verify(
            [entity("BSEP", "Protein"), entity("HCC", "Disease")],
            [relation(
                subject="BSEP",
                subject_type="Protein",
                candidate_id="protein-disease",
                evidence="BSEP was associated with HCC.",
            )],
            pmid="1",
            text=text,
        )
        checked = verified.relations[0]

        self.assertTrue(checked.schema_valid)
        self.assertTrue(checked.candidate_schema_valid)
        self.assertEqual(checked.factual_status, "VALID")
        self.assertEqual(checked.semantic_status, "ACCEPTED")
        self.assertFalse(checked.write_contract_valid)
        self.assertEqual(checked.write_status, "SEMANTIC_ONLY")
        self.assertFalse(checked.import_ready)
        self.assertIn("unsupported_main_kg_signature", checked.schema_gap_reasons)
        action = DecisionEngine(OfflineKG(), skip_neo4j_write=True)._decide_relation(
            checked, pmid="1"
        )
        self.assertEqual(action.type, "NO_ACTION")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate_relations.sqlite3"
            store = CandidateRelationStore(mode="sqlite", path=path)
            store.record_verified(pmid="1", verified=verified)
            stats = store.stats()
            store.close()
            con = sqlite3.connect(path)
            row = con.execute(
                "SELECT candidate_schema_valid, write_contract_valid, "
                "schema_gap_reasons_json FROM candidate_relations"
            ).fetchone()
            con.close()

        self.assertEqual(row[:2], (1, 0))
        self.assertIn("unsupported_main_kg_signature", row[2])
        self.assertEqual(stats["by_schema_gap_reason"]["unsupported_main_kg_signature"], 1)

    def test_main_contract_relation_can_reach_create_decision(self):
        text = "TITLE: Liver HCC study\nABSTRACT: RESULTS: TP53 was associated with HCC."
        verified = self.verifier().verify(
            [entity("TP53", "Gene"), entity("HCC", "Disease")],
            [relation(candidate_id="gene-disease")],
            pmid="2",
            text=text,
        )
        checked = verified.relations[0]

        self.assertTrue(checked.candidate_schema_valid)
        self.assertTrue(checked.write_contract_valid)
        self.assertEqual(checked.write_status, "IMPORT_READY")
        self.assertTrue(checked.import_ready)
        action = DecisionEngine(OfflineKG(), skip_neo4j_write=True)._decide_relation(
            checked, pmid="2"
        )
        self.assertEqual(action.type, "CREATE_RELATION")

    def test_unresolved_gene_protein_endpoint_ambiguity_requires_review(self):
        text = (
            "TITLE: OTUD5 and MAVS\n"
            "ABSTRACT: RESULTS: OTUD5 interacted with MAVS in patient macrophages."
        )
        verified = self.verifier().verify(
            [
                entity("OTUD5", "Gene"),
                entity("OTUD5", "Protein"),
                entity("MAVS", "Protein"),
            ],
            [relation(
                subject="OTUD5",
                subject_type="Protein",
                predicate="INTERACTS_WITH",
                object="MAVS",
                object_type="Protein",
                evidence="OTUD5 interacted with MAVS in patient macrophages.",
                direction="none",
            )],
            pmid="type-ambiguity",
            text=text,
        )
        checked = verified.relations[0]

        self.assertTrue(checked.candidate_schema_valid)
        self.assertTrue(checked.write_contract_valid)
        self.assertEqual(checked.factual_status, "REVIEW")
        self.assertEqual(checked.semantic_status, "REVIEW")
        self.assertEqual(checked.write_status, "HUMAN_REVIEW")
        self.assertFalse(checked.import_ready)
        self.assertIn("endpoint_type_ambiguous", checked.quality_flags)

    def test_normalized_protein_endpoint_resolves_gene_protein_surface_overlap(self):
        text = (
            "TITLE: OTUD5 and MAVS\n"
            "ABSTRACT: RESULTS: OTUD5 interacted with MAVS in patient macrophages."
        )
        gene = entity("OTUD5", "Gene")
        gene["attributes"]["normalized_id"] = "NCBIGene:29952"
        protein = entity("OTUD5", "Protein")
        protein["attributes"]["normalized_id"] = "UniProt:Q8N6M9"
        mavs = entity("MAVS", "Protein")
        mavs["attributes"]["normalized_id"] = "UniProt:Q7Z434"
        verified = self.verifier().verify(
            [gene, protein, mavs],
            [relation(
                subject="OTUD5",
                subject_type="Protein",
                predicate="INTERACTS_WITH",
                object="MAVS",
                object_type="Protein",
                evidence="OTUD5 interacted with MAVS in patient macrophages.",
                direction="none",
            )],
            pmid="type-resolved",
            text=text,
        )
        checked = verified.relations[0]

        self.assertEqual(checked.factual_status, "VALID")
        self.assertEqual(checked.write_status, "IMPORT_READY")
        self.assertTrue(checked.import_ready)
        self.assertNotIn("endpoint_type_ambiguous", checked.quality_flags)

    def test_low_level_writer_rejects_candidate_only_typed_edge(self):
        memory = object.__new__(KGMemory)
        memory._driver = object()
        memory._write_target_allowed = True
        candidate = relation(
            subject="BSEP",
            subject_type="Protein",
            object_type="Disease",
        )

        self.assertFalse(write_contract_assessment(candidate).valid)
        self.assertIsNone(memory.create_relation(
            "protein-node", "ASSOCIATED_WITH", "disease-node", {},
            subject_type="Protein", object_type="Disease",
        ))
        self.assertEqual(memory.create_relations_batch([{
            **candidate,
            "subject_element_id": "protein-node",
            "object_element_id": "disease-node",
        }]), [])

    def test_contract_uses_the_main_graphs_typed_relation_id_properties(self):
        progression = write_contract_assessment(relation(
            subject="fibrosis",
            subject_type="Disease",
            predicate="PROGRESSES_TO",
            object="cirrhosis",
            object_type="Disease",
        ))
        metabolite = write_contract_assessment(relation(
            subject="glucose",
            subject_type="Metabolite",
            object_type="Disease",
        ))

        self.assertTrue(progression.valid)
        self.assertEqual(progression.relation_id_property, "progression_id")
        self.assertTrue(metabolite.valid)
        self.assertEqual(metabolite.relation_id_property, "relationship_id")


if __name__ == "__main__":
    unittest.main()
