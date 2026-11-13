"""Small regression cases for graph integrity and per-round bottom-up updates."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from event_taxonomy.induction.contracts import CandidateError
from event_taxonomy.induction.graph import assert_preserved, load_taxonomy
from event_taxonomy.induction.journal import LLMJournal
from event_taxonomy.induction.storage import read_json
from event_taxonomy.tests.synthetic import dataset

from root_event_taxonomy.clustering import candidates
from root_event_taxonomy.data import InputData, validate_source
from root_event_taxonomy.decisions import PROMPTS, group_decision
from root_event_taxonomy.embedding import Embeddings
from root_event_taxonomy.engine import Inducer
from root_event_taxonomy.graph import GraphView, edge, finalize, intermediate, update_graph


def links(*ids):
    return [{"id": n, "reason": "Synthetic judgment"} for n in ids]


def without_edges():
    data = dataset()
    data["taxonomy"]["parent_child_edges"] = []
    data["taxonomy"]["unplaced_concept_ids"] = data["taxonomy"]["frozen_input_concept_ids"]
    return data


class TestInduction(unittest.TestCase):
    def test_soft_root_penalty_does_not_preassign_classes(self):
        vectors = np.array([[1, 0.8], [0.8, 1]], dtype=float)
        roots = np.eye(2)
        base, root_scores = candidates(["A", "B"], vectors, roots, 0.1, 0, 4)
        guided, _ = candidates(["A", "B"], vectors, roots, 0.1, 0.2, 4)
        separated, _ = candidates(["A", "B"], vectors, roots, 0.1, 1, 4)
        self.assertEqual(list(root_scores.argmax(axis=1)), [0, 1])
        self.assertEqual(guided[0].members, ("A", "B"))
        self.assertGreater(guided[0].height, base[0].height)
        self.assertEqual(len(separated), 2)

    def test_split_and_reuse_do_not_rewrite_existing_definition(self):
        judgment = {"reason": "Split by process", "proposals": [
            {"parent": "broad", "definition": "A different definition", "children": links("a", "b")},
            {"parent": "New type", "definition": "A strict upper type", "children": links("c", "d")},
        ]}
        result = group_decision(judgment, {"a", "b", "c", "d", "broad", "outlier"}, {"broad"})
        self.assertEqual(result["action"], "split")
        self.assertEqual(result["proposals"][0]["reuse_id"], "broad")
        self.assertIsNone(result["proposals"][0]["definition"])

    def test_graph_rolls_back_cycle_and_quarantines_unsupported_nodes(self):
        data = InputData(without_edges(), None, {}, 1)
        initial = data.initial
        view = GraphView(initial)
        a, b = "SYN-IMG-A", "SYN-IMG-B"
        node = intermediate(initial, data, {"name": "Trial imaging", "definition": "Imaging for trial diagnosis"},
                            [view.nodes[a], view.nodes[b]], "group.json")
        e1, e2 = [edge(node.concept_id, n, "strict", "group.json", "test") for n in (a, b)]
        parent = edge("SYN-RESEARCH", node.concept_id, "strict", "place.json", "test")
        working = update_graph(initial, node, [e1, e2, parent])
        with self.assertRaisesRegex(CandidateError, "cycle"):
            update_graph(working, edges=[edge(a, node.concept_id, "bad", "group.json", "test")])
        final, audit = finalize(initial, working, {e1.edge_id, parent.edge_id}, {node.concept_id})
        self.assertEqual(audit["quarantined_nodes"], [node.concept_id])
        self.assertFalse(final.parent_child_edges)
        self.assertEqual(final.instance_concept_links, initial.instance_concept_links)
        validate_source(data.dataset, final)

    def test_recursion_and_rejected_cluster_retry(self):
        fixture = without_edges()
        selected = ["SYN-ACT-A", "SYN-ACT-B", "SYN-INH-A", "SYN-INH-B", "SYN-IMG-A", "SYN-IMG-B"]
        data = InputData(fixture, selected, {}, 1)
        vectors = {
            "Activation of gene A": [1, 0, 0], "Activation of gene B": [1, 0.06, 0],
            "Inhibition of gene A": [0.83, 0.56, 0], "Inhibition of gene B": [0.8, 0.6, 0],
            "Trial diagnostic imaging A": [0, 0, 1], "Trial diagnostic imaging B": [0, 0.02, 1],
            "Biological Event": [1, 0, 0], "Clinical Care Event": [0, 0, 1],
            "Biomedical Research Event": [0, 1, 1],
            "Activation": [0.99, 0.04, 0], "Inhibition": [0.82, 0.57, 0],
            "Regulation": [0.94, 0.33, 0], "Trial imaging": [0, 0, 1],
        }

        class FixtureEmbeddings:
            def embed(self, texts):
                return {"vectors": [vectors[text.split("\n")[0]] for text in texts],
                        "model": "fixture", "usage": None}

        class FixtureLLM:
            def generate(self, stage, prompt, payload, parameters):
                if stage == "group":
                    names = {r["name"] for r in payload["members"]}
                    prefixes = {n.split()[0] for n in names}
                    rules = {
                        frozenset({"Activation"}): "Activation",
                        frozenset({"Inhibition"}): "Inhibition",
                        frozenset({"Trial"}): "Trial imaging",
                    }
                    parent = "Regulation" if names == {"Activation", "Inhibition"} else rules.get(frozenset(prefixes))
                    value = {"reason": "Synthetic group", "proposals": [
                        {"parent": parent, "definition": "A distinct synthetic type called " + parent,
                         "children": links(*(r["id"] for r in payload["members"]))},
                    ] if parent else []}
                elif stage == "placement":
                    root_names = ({"Clinical Care Event", "Biomedical Research Event"}
                                  if payload["target"]["name"] == "Trial imaging" else {"Biological Event"})
                    value = {"parents": links(*(r["id"] for r in payload["roots"] if r["name"] in root_names)),
                             "seek_broader": True, "reason": "Synthetic placement"}
                elif stage == "audit_edges":
                    value = {"checks": [{"edge_id": r["edge_id"], "verdict": "valid", "reason": "Synthetic audit"}
                                        for r in payload["edges"]]}
                else:
                    value = {"checks": [{"id": r["concept"]["id"], "verdict": "distinct",
                                         "duplicate_of": None, "reason": "Synthetic audit"}
                                        for r in payload["comparisons"]]}
                return {"content": json.dumps(value), "model": "fixture", "usage": None, "finish_reason": "stop"}

        config = {
            "clustering": {"distance_cutoff": 0.3, "root_weight": 0.1}, "max_rounds": 4,
            "max_group_size": 4, "max_new_nodes": 8, "parent_candidates": 3, "reuse_candidates": 4,
            "max_ancestor_paths": 4, "audit_batch_size": 5, "rejected_group_split_depth": 1,
            "response_retries": 1, "llm": {"stages": {stage: {"model": "fixture"} for stage in PROMPTS}},
        }
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            embeddings = Embeddings(FixtureEmbeddings(), {"model": "fixture", "batch_size": 6},
                                    data, directory / "embeddings")
            journal = LLMJournal(directory / "llm", FixtureLLM(), PROMPTS, config["llm"]["stages"], 40)
            result = Inducer(directory, config, data, embeddings, journal).run()
            final = load_taxonomy(read_json(directory / "final_taxonomy.json"))
        assert_preserved(data.initial, final, accepted=True)
        self.assertEqual(result["rounds"][0]["actions"], {"accept": 3, "reject": 1})
        self.assertEqual(result["rounds"][1]["committed_new"], 1)
        self.assertEqual(result["selected_root_reachable"], 6)
        self.assertEqual(len(result["multiple_parents_in_scope"]), 1)
        self.assertEqual(len(result["induced_concepts_retained"]), 4)
        graph = GraphView(final)
        node_id = next(n.concept_id for n in final.nodes if n.concept_label == "Regulation")
        self.assertEqual({graph.nodes[n].concept_label for n in graph.children[node_id]}, {"Activation", "Inhibition"})


if __name__ == "__main__":
    unittest.main()
