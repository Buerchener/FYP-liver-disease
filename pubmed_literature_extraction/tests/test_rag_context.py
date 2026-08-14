import unittest

from cognitive_agent.rag_context import ControlledNeo4jRAG, RAGConfig


class FakeRAGKG:
    def __init__(self, connected=True, fail=False):
        self.is_connected = connected
        self.fail = fail
        self.calls = []
        self.write_calls = 0

    def get_schema_profile(self):
        return {
            "labels": {"Gene": {"count": 2, "properties": ["name", "gene_id"]}},
            "relationship_types": {"ASSOCIATED_WITH": 1},
            "compatible_entity_types": ["Gene"],
        }

    def find_entity(self, mention, entity_type="", normalized_id=""):
        self.calls.append(("exact_or_id", mention, entity_type, normalized_id))
        if self.fail:
            raise RuntimeError("query failed")
        if mention:
            return {
                "element_id": f"exact:{mention}", "labels": [entity_type],
                "node_id": f"ID:{mention}", "name": mention,
            }
        if normalized_id:
            return {
                "element_id": f"id:{normalized_id}", "labels": [entity_type],
                "node_id": normalized_id, "name": f"normalized-{normalized_id}",
            }
        return None

    def find_entity_fuzzy(self, mention, entity_type=""):
        self.calls.append(("fuzzy", mention, entity_type))
        if self.fail:
            raise RuntimeError("fuzzy failed")
        if entity_type:
            return {"fuzzy_matches": []}
        return {"fuzzy_matches": [{
            "element_id": f"fuzzy:{mention}", "labels": ["Protein"],
            "node_id": f"FUZZY:{mention}", "name": f"{mention} protein",
            "entity_type": "Protein", "score": 0.61,
        }]}

    def get_rag_entity_context(self, element_id, relation_limit=5, evidence_limit=500):
        self.calls.append(("neighbors", element_id, relation_limit, evidence_limit))
        if self.fail:
            raise RuntimeError("neighbor failed")
        return {
            "synonyms": [f"alias-{index}" for index in range(20)],
            "relations": [
                {
                    "predicate": "ASSOCIATED_WITH",
                    "target_name": f"target-{index}",
                    "target_type": "Disease",
                    "edge_orientation": "outgoing",
                    "direction": "positive",
                    "source_pmid": str(1000 + index),
                    "source_evidence": "x" * (evidence_limit + 100),
                }
                for index in range(20)
            ][:relation_limit],
        }

    def create_entity(self, *args, **kwargs):
        self.write_calls += 1
        raise AssertionError("RAG must never write")

    def create_relation(self, *args, **kwargs):
        self.write_calls += 1
        raise AssertionError("RAG must never write")


class ControlledNeo4jRAGTests(unittest.TestCase):
    def test_disabled_context_performs_no_queries(self):
        kg = FakeRAGKG()
        result = ControlledNeo4jRAG(kg).build([
            {"mention": "TP53", "type": "Gene"}
        ])

        self.assertEqual(result.status, "DISABLED")
        self.assertEqual(result.entity_contexts, [])
        self.assertEqual(kg.calls, [])

    def test_offline_context_returns_empty_without_failure(self):
        kg = FakeRAGKG(connected=False)
        result = ControlledNeo4jRAG(kg, RAGConfig(enabled=True)).build([
            {"mention": "TP53", "type": "Gene"}
        ])

        self.assertEqual(result.status, "OFFLINE")
        self.assertEqual(result.entity_contexts, [])
        self.assertEqual(kg.calls, [])

    def test_retrieval_order_and_required_context_fields(self):
        kg = FakeRAGKG()
        rag = ControlledNeo4jRAG(kg, RAGConfig(
            enabled=True,
            max_candidates_per_entity=3,
            max_total_candidates=3,
            max_neighbors_per_candidate=2,
            max_synonyms_per_candidate=3,
            max_evidence_chars=25,
            max_total_chars=6000,
        ))
        result = rag.build([{
            "mention": "TP53", "type": "Gene",
            "attributes": {"normalized_id": "NCBIGene:7157"},
        }])

        self.assertEqual(result.status, "OK")
        self.assertEqual([call[0] for call in kg.calls[:4]], [
            "exact_or_id", "exact_or_id", "fuzzy", "fuzzy",
        ])
        methods = [
            item["match_method"]
            for item in result.entity_contexts[0]["candidates"]
        ]
        self.assertEqual(methods, ["exact_name", "normalized_id", "fuzzy_candidate"])
        card = result.entity_contexts[0]["candidates"][0]
        self.assertTrue({
            "entity_name", "entity_type", "normalized_id", "synonyms",
            "candidate_relations", "match_method", "score",
        }.issubset(card))
        self.assertLessEqual(len(card["synonyms"]), 3)
        self.assertLessEqual(len(card["candidate_relations"]), 2)
        relation = card["candidate_relations"][0]
        self.assertTrue({
            "direction", "source_pmid", "source_evidence", "predicate",
        }.issubset(relation))
        self.assertLessEqual(len(relation["source_evidence"]), 25)
        self.assertTrue(relation["memory_only"])
        self.assertFalse(result.usage_policy["current_article_evidence"])
        self.assertFalse(result.usage_policy["may_auto_accept_relation"])
        self.assertFalse(result.usage_policy["may_write_neo4j"])
        self.assertEqual(result.schema_profile["compatible_entity_types"], ["Gene"])
        self.assertEqual(kg.write_calls, 0)

    def test_entity_candidate_neighbor_and_character_limits_are_enforced(self):
        kg = FakeRAGKG()
        config = RAGConfig(
            enabled=True,
            max_entities=2,
            max_candidates_per_entity=1,
            max_total_candidates=2,
            max_neighbors_per_candidate=1,
            max_synonyms_per_candidate=1,
            max_evidence_chars=10,
            max_total_chars=1800,
        )
        entities = [
            {"mention": name, "type": "Gene"}
            for name in ("TP53", "EGFR", "NFE2L2", "AKT1")
        ]
        result = ControlledNeo4jRAG(kg, config).build(entities)

        self.assertEqual(result.candidate_count, 2)
        self.assertLessEqual(len(result.entity_contexts), 2)
        self.assertLessEqual(result.total_chars, config.max_total_chars)
        for entity_context in result.entity_contexts:
            self.assertLessEqual(len(entity_context["candidates"]), 1)
            card = entity_context["candidates"][0]
            self.assertLessEqual(len(card["synonyms"]), 1)
            self.assertLessEqual(len(card["candidate_relations"]), 1)
            self.assertLessEqual(
                len(card["candidate_relations"][0]["source_evidence"]), 10
            )

    def test_total_character_limit_can_drop_oversized_cards(self):
        kg = FakeRAGKG()
        result = ControlledNeo4jRAG(kg, RAGConfig(
            enabled=True,
            max_total_chars=100,
            max_candidates_per_entity=1,
        )).build([{"mention": "TP53", "type": "Gene"}])

        self.assertEqual(result.status, "EMPTY")
        self.assertTrue(result.truncated)
        self.assertEqual(result.candidate_count, 0)
        self.assertLessEqual(result.total_chars, 100)

    def test_duplicate_input_entities_are_queried_once(self):
        kg = FakeRAGKG()
        result = ControlledNeo4jRAG(kg, RAGConfig(
            enabled=True, max_candidates_per_entity=1,
        )).build([
            {"mention": "TP53", "type": "Gene"},
            {"mention": "tp53", "type": "Gene"},
        ])

        self.assertEqual(result.candidate_count, 1)
        exact_calls = [call for call in kg.calls if call[0] == "exact_or_id"]
        self.assertEqual(len(exact_calls), 1)

    def test_query_exception_returns_empty_context_and_does_not_raise(self):
        kg = FakeRAGKG(fail=True)
        result = ControlledNeo4jRAG(kg, RAGConfig(enabled=True)).build([
            {"mention": "TP53", "type": "Gene"}
        ])

        self.assertEqual(result.status, "ERROR")
        self.assertEqual(result.entity_contexts, [])
        self.assertEqual(result.candidate_count, 0)
        self.assertTrue(result.error)
        self.assertEqual(kg.write_calls, 0)


if __name__ == "__main__":
    unittest.main()
