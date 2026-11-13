import json
from pathlib import Path
import tempfile
import unittest

from event_taxonomy.induction.storage import atomic_json
from event_taxonomy.tests.synthetic import dataset
from taxonomy_workbench.pipeline import pipeline_view, safe_file
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
