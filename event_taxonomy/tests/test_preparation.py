from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

from event_taxonomy.adapters import load_sources
from event_taxonomy.concepts.registry import normalize_candidates
from event_taxonomy.models import (ConceptCandidate, Document, ParentChildEdge, SourceRef,
                                   UniqueConcept, to_dict)
from event_taxonomy.pipeline import prepare
from event_taxonomy.preparation.evidence import locate_evidence
from event_taxonomy.preparation.sources import SourceFile
from event_taxonomy.taxonomy.dag import apply_edge_transaction, validate_dag
from event_taxonomy.taxonomy.initialization import build_initial_taxonomy, prepared_input_snapshot
from event_taxonomy.validation import validate_bundle, validate_preparation


def synthetic_roots():
    return {"predefined_roots": ["SYN-BIO", "SYN-RESEARCH"], "parent_child_edges": [], "nodes": [
        {"concept_id": "SYN-BIO", "concept_label": "Synthetic biological root", "concept_description": "Synthetic biological processes."},
        {"concept_id": "SYN-RESEARCH", "concept_label": "Synthetic research root", "concept_description": "Synthetic research actions."}]}


def synthetic_graph():
    text = "😀 Drug A reduced inflammation."
    quote = "Drug A reduced inflammation."
    evidence = [{"text": quote, "status": "verified_exact", "char_start": 2,
                 "char_end": len(text), "offset_unit": "unicode_code_point"}]
    nodes = [{"id": "SYN-S1", "kind": "Source", "pmid": None, "label": "Synthetic Data", "text": text}]
    for i in range(2):
        nodes.append({"id": "SYN-E" + str(i), "kind": "Event", "label": quote,
            "source_node_id": "SYN-S1", "source_pmid": None, "source_run_ids": ["SYN-run"],
            "local_id": "E" + str(i + 1), "prediction_index": i, "evidence": deepcopy(evidence),
            "concept_status": "assigned", "arguments": [], "review": {"status": "pending"}})
    nodes.extend([{"id": "SYN-C1", "kind": "Concept", "label": "inflammation reduction",
                   "normalized_label": "inflammation reduction", "concept_domain": "Event", "concept_role": "types"},
                  {"id": "SYN-C2", "kind": "Concept", "label": "Drug A",
                   "normalized_label": "drug a", "concept_domain": "Event", "concept_role": "topics"}])
    edges = [{"id": "SYN-L" + str(i), "head": "SYN-E" + str(i), "tail": "SYN-C1",
              "kind": "HAS_TYPE", "source_pmid": None, "original_label": "inflammation reduction"} for i in range(2)]
    edges.append({"id": "SYN-topic", "head": "SYN-E0", "tail": "SYN-C2", "kind": "RELATED_TOPIC",
                  "source_pmid": None, "original_label": "Drug A"})
    return {"schema_version": "workbench-candidate-graph/1", "nodes": nodes, "edges": edges,
            "rejected": [], "data_origin": "synthetic"}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def source(self, obj, name="synthetic.json", raw=False):
        path = self.folder / name
        path.write_text(obj if raw else json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        return SourceFile.read(path)

    def graph_bundle(self, graph=None):
        source = self.source(graph or synthetic_graph())
        bundle = load_sources([source], "workbench_graph", "SYN-dataset", "synthetic")
        validate_bundle(bundle, [source])
        return source, bundle

    def prepared(self):
        source, bundle = self.graph_bundle()
        concepts, mappings, _ = normalize_candidates(bundle.candidates)
        taxonomy = build_initial_taxonomy(concepts, bundle.candidates, synthetic_roots())
        return bundle, concepts, mappings, taxonomy, prepared_input_snapshot(taxonomy)

    def test_realistic_graph_preserves_events_and_all_concept_roles(self):
        _, bundle = self.graph_bundle()
        self.assertEqual(len(bundle.events), 2)  # Identical sentences are distinct instances.
        self.assertEqual(len({e.event_id for e in bundle.events}), 2)
        concepts, mappings, _ = normalize_candidates(bundle.candidates)
        self.assertEqual(len(bundle.candidates), 3)
        self.assertEqual(len(concepts), 2)
        self.assertEqual({c.concept_role for c in concepts}, {"event_type", "topic"})
        typed = next(c for c in concepts if c.concept_role == "event_type")
        self.assertEqual(len(typed.supporting_event_ids), 2)
        self.assertEqual(len(typed.candidate_ids), 2)
        self.assertEqual(len(mappings), 3)

    def test_source_location_changes_do_not_change_identity(self):
        _, one = self.graph_bundle()
        source = self.source(synthetic_graph(), "copied.json")
        two = load_sources([source], "workbench_graph", "SYN-dataset", "synthetic")
        self.assertEqual([e.event_id for e in one.events], [e.event_id for e in two.events])

    def test_runs_do_not_share_event_ids(self):
        _, one = self.graph_bundle()
        graph = synthetic_graph()
        for node in graph["nodes"]:
            if node["kind"] == "Event":
                node["source_run_ids"] = ["SYN-run-2"]
        _, two = self.graph_bundle(graph)
        self.assertFalse({e.event_id for e in one.events} & {e.event_id for e in two.events})

    def test_unicode_offsets_are_code_points(self):
        _, bundle = self.graph_bundle()
        self.assertEqual(bundle.evidence[0].spans[0]["start"], 2)
        self.assertEqual(bundle.evidence[0].alignment_status, "verified_exact")

    def test_repeated_sentence_retains_both_spans(self):
        ref = SourceRef("synthetic", "SYN-run", "0" * 64, "/")
        doc = Document("SYN-doc", None, "Synthetic Data", "A changes. A changes.", "", "synthetic", [ref])
        ev = locate_evidence(doc, "A changes.", ref)
        self.assertEqual(ev.alignment_status, "ambiguous")
        self.assertEqual(ev.spans, [{"start": 0, "end": 10}, {"start": 11, "end": 21}])

    def test_whitespace_match_is_distinguished_from_exact(self):
        ref = SourceRef("synthetic", "SYN-run", "0" * 64, "/")
        doc = Document("SYN-doc", None, "Synthetic Data", "A\u00a0changes.", "", "synthetic", [ref])
        ev = locate_evidence(doc, "A changes.", ref)
        self.assertEqual(ev.alignment_status, "verified_whitespace")

    def test_ungrounded_quote_is_not_rewritten_or_dropped(self):
        graph = synthetic_graph()
        graph["nodes"][1]["evidence"][0]["text"] = "This sentence is not in the source."
        _, bundle = self.graph_bundle(graph)
        self.assertEqual(len(bundle.events), 2)
        self.assertEqual(bundle.evidence[0].alignment_status, "unverified")
        self.assertIsNone(bundle.events[0].evidence_sentence)
        self.assertIn("upstream_offset_mismatch", [x["kind"] for x in bundle.events[0].validation_issues])

    def test_synthetic_data_cannot_claim_a_pmid(self):
        graph = synthetic_graph()
        graph["nodes"][0]["pmid"] = "123456"
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            self.graph_bundle(graph)

    def test_synthetic_source_cannot_be_relabelled_as_real(self):
        source = self.source(synthetic_graph())
        with self.assertRaisesRegex(ValueError, "data_origin"):
            load_sources([source], "workbench_graph", "SYN-dataset", "real_literature")

    def test_single_run_export_import_keeps_labels_untyped(self):
        row = {"Event_ID": "E1", "Event": "A changes.", "Evidence": "A changes.",
               "Arguments": [], "Review": {"status": "excluded"}}
        payload = {"data_origin": "synthetic", "article": {"pmid": None, "abstract": "A changes."},
            "run": {"id": "SYN-run", "eventConcepts": {"model": "synthetic-model", "promptVersion": "synthetic/1",
                "results": [{"eventId": "E1", "event": "A changes.", "labels": ["change"]}]}}, "events": [row]}
        source = self.source(payload)
        bundle = load_sources([source], "workbench_export", "SYN-export", "synthetic")
        validate_bundle(bundle, [source])
        self.assertEqual(bundle.events[0].review_status, "excluded")
        self.assertEqual(bundle.candidates[0].candidate_role, "unknown")
        self.assertFalse(bundle.candidates[0].stale)

    def test_graph_dangling_edge_fails(self):
        graph = synthetic_graph()
        graph["edges"][0]["tail"] = "MISSING"
        with self.assertRaisesRegex(ValueError, "Dangling"):
            self.graph_bundle(graph)

    def test_topic_cannot_be_imported_as_type(self):
        graph = synthetic_graph()
        graph["edges"][2]["kind"] = "HAS_TYPE"
        with self.assertRaisesRegex(ValueError, "role/domain"):
            self.graph_bundle(graph)

    def test_official_json_extension_can_contain_jsonl(self):
        row = {"id": "not-a-pmid", "metadata": {}, "original_text": "A changes.",
               "event_entity_dict": [{"Event": "A changed.", "Entity": ["A"]}]}
        source = self.source(json.dumps(row) + "\n", "official.json", raw=True)
        bundle = load_sources([source], "official_jsonl", "SYN-official", "synthetic", "SYN-run")
        validate_bundle(bundle, [source])
        self.assertIsNone(bundle.events[0].source_pmid)
        self.assertIsNone(bundle.events[0].evidence_sentence)
        self.assertEqual(bundle.evidence[0].alignment_status, "passage_only")
        self.assertEqual(bundle.events[0].arguments[0]["role_provenance"], "untyped_original_participation")

    def test_role_nested_errors_and_excluded_events_remain(self):
        row = {"pmid": None, "text": "A changes.", "events": [{"Event_ID": "E1", "Event": "A changes.",
            "Evidence": "No such quote.", "Arguments": [{"Role": "Theme", "Type": "Event", "Value": "E9"}]}],
            "structural_errors": [{"event": "E1", "kind": "evidence_not_grounded"}]}
        source = self.source(row)
        bundle = load_sources([source], "role_nested", "SYN-nested", "synthetic", "SYN-run")
        validate_bundle(bundle, [source])
        kinds = {x["kind"] for x in bundle.events[0].validation_issues}
        self.assertIn("evidence_not_grounded", kinds)
        self.assertIn("unresolved_nested_event_reference", kinds)

    def test_workbench_stale_and_empty_and_missing_are_different(self):
        rows = [{"Event_ID": "E" + str(i), "Event": "A changes.", "Evidence": "A changes.", "Arguments": []}
                for i in range(3)]
        run = {"id": "SYN-run", "articleSnapshot": {"pmid": None, "abstract": "A changes."},
               "events": rows, "reviews": {"E2": {"status": "excluded"}},
               "eventConcepts": {"model": "synthetic-model", "promptVersion": "synthetic/1",
                   "results": [{"eventId": "E0", "event": "The old event.", "labels": ["change"]},
                               {"eventId": "E1", "event": "A changes.", "labels": []}]}}
        source = self.source({"runs": [run]})
        bundle = load_sources([source], "workbench_workspace", "SYN-workspace", "synthetic")
        self.assertEqual(len(bundle.events), 3)
        self.assertEqual(bundle.events[2].review_status, "excluded")
        self.assertEqual({r["upstream_status"] for r in bundle.concept_generation_records},
                         {"stale", "explicit_empty", "not_present"})
        concepts, mappings, _ = normalize_candidates(bundle.candidates)
        self.assertFalse(concepts)
        self.assertIsNone(mappings[0]["concept_id"])
        self.assertTrue(bundle.candidates[0].stale)

    def test_new_candidates_do_not_merge_case_or_synonyms(self):
        _, bundle = self.graph_bundle()
        base = bundle.candidates[0]
        candidates = [replace(base, candidate_id="SYN-" + str(i), candidate_label=label,
                              upstream_concept_id=None, upstream_normalized_label=None)
                      for i, label in enumerate(["ABC activation", "abc activation", "activation of ABC"])]
        concepts, _, _ = normalize_candidates(candidates)
        self.assertEqual(len(concepts), 3)

    def test_same_label_different_definition_stays_distinct(self):
        _, bundle = self.graph_bundle()
        base = bundle.candidates[0]
        candidates = [replace(base, candidate_id="SYN-" + str(i), upstream_concept_id=None,
                              candidate_description=definition)
                      for i, definition in enumerate(["Synthetic definition A", "Synthetic definition B"])]
        concepts, _, _ = normalize_candidates(candidates)
        self.assertEqual(len(concepts), 2)

    def test_upstream_concept_identity_survives_label_edit(self):
        _, bundle = self.graph_bundle()
        first, _, _ = normalize_candidates([bundle.candidates[0]])
        renamed = replace(bundle.candidates[0], candidate_label="A corrected label")
        second, _, _ = normalize_candidates([renamed])
        self.assertEqual(first[0].concept_id, second[0].concept_id)

    def test_initial_graph_keeps_unplaced_types_but_not_topics(self):
        bundle, concepts, mappings, taxonomy, snapshot = self.prepared()
        report = validate_preparation(bundle, concepts, mappings, taxonomy, snapshot)
        self.assertEqual(len(taxonomy.nodes), 3)
        self.assertEqual(len(taxonomy.unplaced_concept_ids), 1)
        self.assertEqual(len(taxonomy.instance_concept_links), 2)
        self.assertFalse(taxonomy.parent_child_edges)
        self.assertFalse(report["induction_ready"])

    def test_multi_parent_assignment_is_supported(self):
        bundle, concepts, _, taxonomy, _ = self.prepared()
        cid = taxonomy.frozen_input_concept_ids[0]
        decisions = [{"concept_id": cid, "parent_id": root, "status": "provisional",
                      "decision_id": "SYN-decision-" + root, "reason": "Synthetic Data test only", "validator": "synthetic_fixture"}
                     for root in taxonomy.predefined_roots]
        assigned = build_initial_taxonomy(concepts, bundle.candidates, synthetic_roots(), decisions)
        self.assertEqual(len(assigned.parent_child_edges), 2)
        self.assertFalse(assigned.unplaced_concept_ids)
        self.assertEqual(validate_dag(assigned)["accepted_root_reachable_inputs"], 0)

    def test_accepted_assignment_requires_definition(self):
        bundle, concepts, _, taxonomy, _ = self.prepared()
        decision = {"concept_id": taxonomy.frozen_input_concept_ids[0], "parent_id": "SYN-BIO", "status": "accepted",
                    "decision_id": "SYN-decision", "reason": "Synthetic", "validator": "synthetic_fixture"}
        with self.assertRaisesRegex(ValueError, "defined concept"):
            build_initial_taxonomy(concepts, bundle.candidates, synthetic_roots(), [decision])

    def test_cycle_transaction_rolls_back(self):
        _, _, _, taxonomy, _ = self.prepared()
        cid = taxonomy.frozen_input_concept_ids[0]
        taxonomy.nodes.append(replace(taxonomy.nodes[-1], concept_id="SYN-second"))
        before = to_dict(taxonomy)
        edges = [ParentChildEdge("SYN-edge1", cid, "SYN-second", "provisional", "SYN-d", "synthetic", "fixture"),
                 ParentChildEdge("SYN-edge2", "SYN-second", cid, "provisional", "SYN-d", "synthetic", "fixture")]
        with self.assertRaisesRegex(ValueError, "cycle"):
            apply_edge_transaction(taxonomy, edges)
        self.assertEqual(before, to_dict(taxonomy))

    def test_self_loop_and_root_parent_are_rejected(self):
        _, _, _, taxonomy, _ = self.prepared()
        cid = taxonomy.frozen_input_concept_ids[0]
        for parent, child in [(cid, cid), (cid, "SYN-BIO")]:
            with self.subTest(child=child), self.assertRaises(ValueError):
                apply_edge_transaction(taxonomy, [ParentChildEdge("SYN-e", parent, child,
                    "provisional", "SYN-d", "synthetic", "fixture")])

    def test_accepted_ancestor_cannot_be_removed(self):
        _, _, _, taxonomy, _ = self.prepared()
        cid = taxonomy.frozen_input_concept_ids[0]
        edge = ParentChildEdge("SYN-e", "SYN-BIO", cid, "accepted", "SYN-d", "synthetic", "fixture")
        assigned = apply_edge_transaction(taxonomy, [edge])
        with self.assertRaisesRegex(ValueError, "ancestor"):
            apply_edge_transaction(assigned, [], ["SYN-e"])

    def test_frozen_concept_or_mapping_tamper_is_rejected(self):
        bundle, concepts, mappings, taxonomy, snapshot = self.prepared()
        altered = deepcopy(taxonomy)
        altered.nodes[-1].concept_label = "Changed meaning"
        with self.assertRaisesRegex(ValueError, "Frozen input"):
            validate_preparation(bundle, concepts, mappings, altered, snapshot)
        altered = deepcopy(taxonomy)
        altered.concept_evidence_mapping[0]["event_ids"] = []
        with self.assertRaisesRegex(ValueError, "Frozen input"):
            validate_preparation(bundle, concepts, mappings, altered, snapshot)

    def test_complete_cannot_hide_unreviewed_or_unplaced_inputs(self):
        _, _, _, taxonomy, _ = self.prepared()
        taxonomy.status = "complete"
        with self.assertRaisesRegex(ValueError, "complete"):
            validate_dag(taxonomy)

    def test_pipeline_snapshots_replay_and_never_overwrites(self):
        source = self.source(synthetic_graph())
        roots = self.source(synthetic_roots(), "roots.json")
        # Output lives outside the input directory; source directories are protected.
        source_dir = self.folder / "input"
        source_dir.mkdir()
        source.path.rename(source_dir / source.path.name)
        config = {"schema_version": "event-taxonomy-preparation-config/1", "profile": "workbench_graph",
                  "namespace": "SYN-pipeline", "source_root": str(source_dir), "input_files": ["synthetic.json"],
                  "roots_file": str(roots.path), "data_origin": "synthetic", "source_run_id": None,
                  "assignments_file": None, "output_root": str(self.folder / "runs"), "allow_model_calls": False,
                  "run_induction": False, "semantic_synonym_merging": False, "split": "synthetic_only"}
        config_file = self.source(config, "config.json")
        first = prepare(config_file.path, run_id="SYN-one")
        second = prepare(config_file.path, run_id="SYN-two")
        for filename in ["events.jsonl", "unique_concepts.jsonl", "initial_taxonomy.json", "prepared_input_snapshot.json"]:
            self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
        checksums = json.loads((first / "checksums.json").read_text())
        self.assertTrue(all(hashlib.sha256((first / name).read_bytes()).hexdigest() == h for name, h in checksums.items()))
        self.assertEqual(json.loads((first / "manifest.json").read_text())["new_api_calls"], 0)
        with self.assertRaises(FileExistsError):
            prepare(config_file.path, run_id="SYN-one")
        (source_dir / "synthetic.json").unlink()
        roots.path.unlink()
        replay = prepare(first / "replay_config.json", run_id="SYN-replay")
        self.assertEqual((first / "events.jsonl").read_bytes(), (replay / "events.jsonl").read_bytes())
        self.assertTrue((first / "code_snapshot/event_taxonomy/models.py").is_file())

    def test_source_change_is_detected(self):
        source = self.source({"data_origin": "synthetic"})
        source.path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            source.assert_unchanged()


if __name__ == "__main__":
    unittest.main()
