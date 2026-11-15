import json
from pathlib import Path
import tempfile
import unittest

from event_taxonomy.induction.storage import atomic_json
from event_taxonomy.tests.synthetic import dataset
from taxonomy_workbench.pipeline import pipeline_view, safe_file, upward_view
from taxonomy_workbench.server import Router
from taxonomy_workbench.service import Workbench


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.run = self.runs / "xml-event-only" / "test-induction"
        self.source = dataset()
        atomic_json(self.run / "input_dataset.json", self.source)
        atomic_json(self.run / "final_taxonomy.json", self.source["taxonomy"])
        atomic_json(self.run / "report.json", {
            "status": "partial", "stop_reason": "no_semantic_progress",
            "rounds": [], "llm_calls": 0,
        })
        atomic_json(self.run / "manifest.json", {
            "config": {"dataset": str(self.run / "input_dataset.json"),
                       "llm": {"api_key": "do-not-send-to-browser"}},
        })

    def test_saved_graph_without_review_keeps_edges_unreviewed(self):
        view = pipeline_view(self.runs)
        self.assertEqual(len(view["edges"]), len(self.source["taxonomy"]["parent_child_edges"]))
        self.assertTrue(all(edge["review"] == "unreviewed" for edge in view["edges"]))
        self.assertNotIn("do-not-send-to-browser", json.dumps(view))
        self.assertEqual(view["stats"]["leaves"], 8)

    def test_review_for_different_endpoints_is_not_attached(self):
        edge = self.source["taxonomy"]["parent_child_edges"][0]
        atomic_json(self.run / "edge_definition_review.json", {"edges": [{
            "edge_id": edge["edge_id"], "parent_id": "different",
            "child_id": edge["child_id"], "definition_review": "supported",
        }]})
        view = pipeline_view(self.runs)
        self.assertEqual(next(row for row in view["edges"] if row["id"] == edge["edge_id"])["review"], "unreviewed")

    def test_upstream_path_cannot_escape_results_or_follow_symlinks(self):
        outside = self.root / "outside"
        atomic_json(outside / "manifest.json", {"secret": "private"})
        alias = self.runs / "link"
        alias.symlink_to(outside, target_is_directory=True)
        self.assertFalse(safe_file(alias / "manifest.json", self.runs))
        self.assertFalse(safe_file(self.runs / ".." / "outside" / "manifest.json", self.runs))
        manifest = {"config": {"dataset": str(alias / "induction_input.json")}}
        atomic_json(self.run / "manifest.json", manifest)
        self.assertEqual(pipeline_view(self.runs)["concepts"], [])

    def test_empty_api_and_export_are_read_only(self):
        service = Workbench(self.root / "empty-runs", self.root / "state")
        router = Router(service)
        headers = {"Host": "127.0.0.1:4318"}
        status, _, body = router.dispatch("GET", "/api/pipeline", headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"available": False})
        self.assertEqual(router.dispatch("GET", "/api/pipeline/export", headers)[0], 404)
        self.assertIsNone(service.process)

    def test_export_matches_original_saved_graph(self):
        service = Workbench(self.runs, self.root / "state")
        status, headers, body = Router(service).dispatch("GET", "/api/pipeline/export", {"Host": "127.0.0.1:4318"})
        self.assertEqual(status, 200)
        self.assertEqual(body, (self.run / "final_taxonomy.json").read_bytes())
        self.assertIn("attachment", headers["Content-Disposition"])


class UpwardViewTests(unittest.TestCase):
    def test_held_inputs_and_retired_parents_do_not_enter_effective_graph(self):
        def node(id, status="original"):
            return {"concept_id": id, "concept_label": id, "concept_description": id,
                    "supporting_event_ids": [], "validation_status": status}
        result = {"taxonomy": {
            "nodes": [node("a"), node("b"), node("held"), node("solo"),
                      node("parent"), node("retired", "maintenance_held")],
            "parent_child_edges": [
                {"edge_id": "e" + child, "parent_id": parent, "child_id": child,
                 "status": "provisional", "reason": "saved review"}
                for parent, child in [("parent", "a"), ("parent", "b"),
                                      ("parent", "held"), ("retired", "solo")]],
            "frozen_input_concept_ids": ["a", "b", "held", "solo"], "predefined_roots": []},
            "admission": {"held_leaf_ids": ["held"], "reviews": [
                {"leaf_id": "held", "reason": "Needs review"}]},
            "source_events": [], "source_concepts": [], "leaf_concepts": [],
            "rounds": [], "unassigned_events": [], "status": "finished",
            "stop_reason": "maximum_rounds"}
        before = json.dumps(result)
        view = upward_view(result, "fixture")
        self.assertEqual({n["id"] for n in view["nodes"]}, {"a", "b", "solo", "parent"})
        self.assertEqual(view["unplaced_ids"], ["solo"])
        self.assertEqual(view["stats"]["intermediates"], 1)
        self.assertEqual(view["stats"]["branches"], 1)
        self.assertEqual(len(view["edges"]), 2)
        self.assertEqual({n["id"] for n in view["excluded_nodes"]}, {"held", "retired"})
        self.assertEqual(json.dumps(result), before)
