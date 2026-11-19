import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langextract import data

from cognitive_agent.agent import AgentConfig, CognitiveAgent


class DatasetExtractionTests(unittest.TestCase):
    def test_profile_controls_model_prompt_parsing_and_batch_export(self):
        profile = {
            "name": "Facilities",
            "contract_version": "facilities-v1",
            "semantic_target": "RELATION_TRUTH",
            "entity_types": ["facility_attributes", "city"],
            "predicates": {
                "LocatedIn": {
                    "allowed_signatures": [["facility_attributes", "city"]],
                    "explicit_patterns": [r"\bis in\b"],
                },
            },
        }
        quote = "The museum is in London."

        def extract(**kwargs):
            prompt = kwargs["prompt_description"]
            self.assertIn("LocatedIn", prompt)
            self.assertIn("facility_attributes", prompt)
            self.assertNotIn("ASSOCIATED_WITH", prompt)
            document = kwargs["text_or_documents"][0]
            return [data.AnnotatedDocument(document_id=document.document_id, extractions=[
                data.Extraction(
                    extraction_class="entity", extraction_text="museum",
                    attributes={"entity_type": "facility_attributes", "relations": [{
                        "predicate": "LocatedIn", "target_entity": "London", "target_type": "city",
                        "claim_role": "BACKGROUND", "evidence": quote,
                    }]},
                ),
                data.Extraction(extraction_class="entity", extraction_text="London",
                                attributes={"entity_type": "city"}),
            ])]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            profile_path = output / "schema.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            agent = CognitiveAgent(AgentConfig(
                schema_profile=profile_path, skip_neo4j_write=True, max_workers=1,
            ))
            try:
                with patch("cognitive_agent.extraction_kernel.lx.extract", side_effect=extract):
                    report = agent.run_batch(
                        [{"pmid": "fixture", "title": "Facilities", "abstract": quote}],
                        run_id="schema-test", output_dir=output,
                    )
                results = json.loads((output / "agent_results_schema-test.json").read_text())
            finally:
                agent.article_preprocessor.close()
                agent.extraction_cache.close()
                agent.auxiliary_cache.close()
                agent.candidate_store.close()
                agent.kg_memory.close()

        record = results["records"][0]
        self.assertNotIn("error", record)
        relation = record["phases"]["verification"]["relations"][0]
        self.assertEqual(relation["predicate"], "LocatedIn")
        self.assertEqual(relation["subject_type"], "facility_attributes")
        self.assertTrue(relation["candidate_schema_valid"])
        self.assertEqual(relation["semantic_status"], "ACCEPTED")
        self.assertFalse(relation["import_ready"])
        self.assertEqual(report["schema_profile"]["name"], "Facilities")

    def test_profile_rejects_graph_writes_at_python_entrypoint(self):
        with self.assertRaisesRegex(ValueError, "candidate export only"):
            CognitiveAgent(AgentConfig(schema_profile="schema.json", skip_neo4j_write=False))


if __name__ == "__main__":
    unittest.main()
