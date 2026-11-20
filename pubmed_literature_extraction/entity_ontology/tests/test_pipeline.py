"""Exercise the production pipeline with synthetic provider and extractor responses."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from entity_ontology import model, pipeline
from entity_ontology.concepts.embedding import EmbeddingAPI
from entity_ontology.hierarchy.entity_placement_dag import EntityDAG
from entity_ontology.model import endpoint, read_settings
from entity_ontology.storage import atomic_json, read_json


def fake_extract(articles, output, *, config):
    article = json.loads(articles.read_text())
    text = "TITLE: " + article["title"] + "\nABSTRACT: " + article["abstract"]
    units = []
    for index, fragment in enumerate(article["abstract"].split(". "), 1):
        sentence = fragment if fragment.endswith(".") else fragment + "."
        start = text.index(sentence)
        units.append({
            "unit_id": f"u{index}", "parent_sentence_id": f"s{index}", "section": "ABSTRACT",
            "text": sentence, "char_start": start, "char_end": start + len(sentence),
        })
    entities = []
    for name in ("Beta protein", "BP", "Dual protein"):
        start = text.index(name)
        entities.append({
            "mention": name, "type": "Protein", "source_span": name, "char_start": start,
            "char_end": start + len(name), "grounded": True, "alignment_status": "PARSED",
        })
    artifact = output / "agent_results.json"
    atomic_json(artifact, {
        "report": {"dry_run": True, "neo4j_connected": False, "total_articles": 1, "model": "fixture"},
        "records": [{"pmid": article["pmid"], "phases": {
            "reader": {"units": units}, "extraction": {"entities": entities},
        }}],
    })
    return artifact


def fake_embeddings(texts):
    vectors = {"Alpha protein": [1, 0, 0], "Beta protein": [0, 1, 0], "Dual protein": [0, 0, 1]}
    return {"vectors": [vectors[text.splitlines()[0]] for text in texts], "model": "fixture"}


class FixtureProvider:
    def generate(self, stage, prompt, payload, parameters):
        if stage == "global_concept_rank":
            concepts = sorted(payload["concepts"], key=lambda row: row["name"])
            result = {"order": [row["id"] for row in concepts]}
        else:
            result = {"rows": [self.answer(stage, item) for item in payload["items"]]}
        return {
            "content": json.dumps(result), "finish_reason": "stop",
            "model": parameters["model"], "usage": {},
        }

    def answer(self, stage, item):
        row = {"id": item["id"]}
        if stage == "entity_coverage_v1":
            return {**row, "additions": [
                {"mention": "Alpha protein", "kind": "Protein"},
                {"mention": "Invented protein", "kind": "Protein"},
            ]}
        if stage == "normalize_referent_v2":
            name = "Beta protein" if item["mention"] == "BP" else item["mention"]
            return {**row, "name": name, "kind": "Protein", "definition": f"Class {name}.",
                    "aliases": ["BP"] if name == "Beta protein" else [], "reason": "fixture"}
        if stage == "normalize_type_check_v2":
            return {**row, "kind": "Protein"}
        if stage == "equivalence":
            same = item["left"]["name"] == item["right"]["name"]
            return {**row, "verdict": "equivalent" if same else "distinct", "reason": "fixture"}
        if stage == "incremental_propose":
            return {**row, "new_parent": None, "comparisons": [
                {"id": candidate["id"], "a_in_b": item["name"] == "Dual protein",
                 "b_in_a": False, "reason": "fixture"}
                for candidate in item["candidates"]
            ]}
        if stage == "incremental_edge_review":
            return {**row, "supported": item["child"]["name"] == "Dual protein", "reason": "fixture"}
        raise AssertionError(f"Unexpected fixture model stage {stage}.")


class PipelineTest(unittest.TestCase):
    def test_grounded_coverage_through_multi_parent_placement_and_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            articles = root / "articles.jsonl"
            articles.write_text(json.dumps({
                "pmid": "100000", "title": "Synthetic article",
                "abstract": "Alpha protein level rose. Beta protein and BP were stable. Dual protein remained.",
            }) + "\n")
            config = root / "config.json"
            example = Path(__file__).resolve().parents[1] / "config.example.json"
            settings = read_json(example)
            settings["embedding"]["dimensions"] = 3
            atomic_json(config, settings)
            runs = root / "runs"
            run = runs / "example"
            provider = FixtureProvider()
            with (
                patch.dict("os.environ", {"LIVERKG_API_BASE": "https://provider.example/v1"}),
                patch.object(pipeline, "extract_articles", side_effect=fake_extract) as extract,
                patch.object(model, "live_provider", return_value=provider) as language_model,
                patch("entity_ontology.concepts.retrieve_pairs.live_provider", return_value=provider),
                patch.object(EmbeddingAPI, "embed", staticmethod(fake_embeddings)),
            ):
                connection, _ = read_settings(config, "audit_edges")
                self.assertEqual(endpoint(connection), endpoint(settings["embedding"]))
                options = {"config": config, "runs_dir": runs, "go_reference": False}
                result = pipeline.run_batch(articles, "example", **options)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["counts"]["coverage_mentions"], 1)
                self.assertEqual(result["counts"]["concepts"], 3)
                self.assertEqual(result["counts"]["edges"], 4)

                mentions = read_json(run / "input_mentions.json")["mentions"]
                self.assertEqual([row["mention"] for row in mentions],
                                 ["Beta protein", "BP", "Dual protein", "Alpha protein"])
                added = mentions[-1]
                self.assertEqual(added["offset_status"], "exact")
                self.assertEqual(added["upstream_status"], "ontology_coverage")
                supplement = read_json(run / "entity_coverage.json")["records"][0]
                self.assertEqual([row["reason"] for row in supplement["rejected"]], ["not_an_exact_unique_span"])

                placement_path = run / "incremental-placement/placement.json"
                placement = read_json(placement_path)
                ids = {node["name"]: node["id"] for node in placement["nodes"]}
                beta = next(node for node in placement["nodes"] if node["id"] == ids["Beta protein"])
                self.assertEqual({"100000:M01", "100000:M02"}, set(beta["source_mentions"]))
                direct_parents = {edge["parent_id"] for edge in placement["edges"]
                                  if edge["child_id"] == ids["Dual protein"]}
                self.assertEqual(direct_parents, {ids["Alpha protein"], ids["Beta protein"]})
                before = language_model.call_count
                resumed = pipeline.run_batch(articles, "example", **options)
                self.assertEqual(resumed, result)
                self.assertEqual(language_model.call_count, before)
                extract.assert_called_once()

                placement_path.write_text("{}")
                with self.assertRaisesRegex(ValueError, "completed batch files"):
                    pipeline.run_batch(articles, "example", **options)

    def test_cycle_does_not_change_placed_graph(self):
        dag = EntityDAG(["Protein"])
        for ident, parents in (("a", []), ("b", ["a"])):
            dag.place({"id": ident, "kind": "Protein", "name": ident, "definition": ident}, parent_ids=parents)
        original_edges = dag.edges.copy()
        rejected = dag.place({"id": "c", "kind": "Protein", "name": "c", "definition": "c"},
                             parent_ids=["b"], child_ids=["a"])
        self.assertEqual(rejected["action"], "rejected")
        self.assertIn("cycle", rejected["reason"])
        self.assertEqual(dag.edges, original_edges)
        self.assertNotIn("c", dag.nodes)
