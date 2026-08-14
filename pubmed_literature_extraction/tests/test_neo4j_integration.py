import os
import unittest
import uuid

from cognitive_agent.memory.kg_memory import KGMemory


@unittest.skipUnless(
    os.environ.get("NEO4J_TEST_URI")
    and os.environ.get("NEO4J_TEST_USER")
    and os.environ.get("NEO4J_TEST_PASSWORD")
    and os.environ.get("NEO4J_TEST_DATABASE"),
    "set NEO4J_TEST_URI/USER/PASSWORD/DATABASE for isolated Neo4j integration tests",
)
class Neo4jIntegrationTests(unittest.TestCase):
    """Opt-in tests; never fall back to the production AgentConfig endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.marker = f"cognitive-agent-test-{uuid.uuid4().hex}"
        cls.kg = KGMemory(
            uri=os.environ["NEO4J_TEST_URI"],
            user=os.environ["NEO4J_TEST_USER"],
            password=os.environ["NEO4J_TEST_PASSWORD"],
            database=os.environ["NEO4J_TEST_DATABASE"],
            allow_isolated_test_writes=True,
        )
        try:
            with cls.kg._driver.session(database=cls.kg.database) as session:
                session.run("RETURN 1").consume()
        except Exception as exc:
            cls.kg.close()
            raise unittest.SkipTest(f"isolated Neo4j unavailable: {exc}")

    @classmethod
    def tearDownClass(cls):
        try:
            with cls.kg._driver.session(database=cls.kg.database) as session:
                session.run(
                    "MATCH (n) WHERE n.test_marker = $marker DETACH DELETE n",
                    marker=cls.marker,
                ).consume()
        finally:
            cls.kg.close()

    def test_entity_and_relation_merge_are_idempotent(self):
        gene = self.kg.create_entity(
            "Gene", self.marker, {"test_marker": self.marker},
            evidence="evidence-a", pmid="test", confidence=0.8,
        )
        disease = self.kg.create_entity(
            "Disease", f"Disease {self.marker}", {"test_marker": self.marker},
            evidence="evidence-b", pmid="test", confidence=0.8,
        )
        self.assertTrue(gene)
        self.assertTrue(disease)

        first = self.kg.create_relation(
            gene, "ASSOCIATED_WITH", disease, {"test_marker": self.marker},
            evidence="relation-evidence", pmid="test", confidence=0.8,
        )
        second = self.kg.create_relation(
            gene, "ASSOCIATED_WITH", disease, {"test_marker": self.marker},
            evidence="relation-evidence", pmid="test", confidence=0.6,
        )
        self.assertEqual(first, second)

        with self.kg._driver.session(database=self.kg.database) as session:
            node_count = session.run(
                "MATCH (n) WHERE n.test_marker = $marker RETURN count(n) AS count",
                marker=self.marker,
            ).single()["count"]
            rel = session.run(
                "MATCH (s)-[r:ASSOCIATED_WITH]->(o) "
                "WHERE s.test_marker = $marker AND o.test_marker = $marker "
                "RETURN r.confidence AS confidence, r.evidence AS evidence",
                marker=self.marker,
            ).single()

        self.assertEqual(node_count, 2)
        self.assertEqual(rel["confidence"], 0.8)
        self.assertEqual(rel["evidence"], "relation-evidence")

    def test_exact_lookup_is_cached_read_only(self):
        name = f"Lookup {self.marker}"
        self.kg.create_entity(
            "Disease", name, {"test_marker": self.marker},
            evidence="lookup-evidence", pmid="test", confidence=0.8,
        )
        before = self.kg.entity_lookup_cache_stats()
        first = self.kg.find_entity(name, entity_type="Disease")
        second = self.kg.find_entity(name, entity_type="Disease")
        after = self.kg.entity_lookup_cache_stats()
        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        self.assertEqual(after["misses"], before["misses"] + 1)
        self.assertEqual(after["hits"], before["hits"] + 1)


if __name__ == "__main__":
    unittest.main()
