import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from event_taxonomy.induction.contracts import CandidateError, ContractError
from event_taxonomy.induction.graph import assert_preserved
from event_taxonomy.induction.journal import LLMJournal
from event_taxonomy.tests.synthetic import dataset
from root_event_taxonomy.data import InputData, validate_source
from root_event_taxonomy.discovery import knn_candidates
from root_event_taxonomy.embedding import Embeddings
from root_event_taxonomy.graph import GraphView, edge, intermediate, update_graph
from root_event_taxonomy.refinement import (FiveStepInducer, REFINEMENT_PROMPTS,
                                          containment_checks, insert_transaction)


class FixtureAPI:
    def __init__(self, bad_child=None, bad_placement=False):
        self.bad_child, self.bad_placement = bad_child, bad_placement

    def embed(self, texts):
        return {"vectors": [[1, 0.01 * (i + 1), 0.02] for i, _ in enumerate(texts)],
                "model": "fixture", "usage": None}

    def generate(self, stage, prompt, payload, parameters):
        if stage == "group":
            members = payload["members"]
            # Refuse groups larger than two so the test exercises smaller retries.
            value = {"reason": "fixture", "proposals": [] if len(members) > 2 else [{
                "parent": "Fixture abstraction " + " / ".join(sorted(m["name"] for m in members)),
                "definition": "Fixture union of " + " / ".join(sorted(m["name"] for m in members)),
                "children": [{"id": m["id"], "reason": "hypothesis"} for m in members]}]}
        elif stage == "validate_edges":
            value = {"checks": [{"edge_id": e["edge_id"],
                     "child_entails_parent": "false" if e["child"]["name"] == self.bad_child or (
                         self.bad_placement and e["parent"]["name"] == "Biological Event") else "true",
                     "parent_entails_child": "false", "reason": "fixture independent check"}
                    for e in payload["edges"]]}
        elif stage == "placement":
            value = {"parents": [{"id": r["id"], "reason": "hypothesis"} for r in payload["roots"]
                                  if r["name"] == "Biological Event"], "seek_broader": False, "reason": "fixture"}
        else:
            value = {"checks": [{"id": r["concept"]["id"], "verdict": "distinct", "duplicate_of": None,
                                  "reason": "fixture"} for r in payload["comparisons"]]}
        return {"content": json.dumps(value), "model": "fixture", "finish_reason": "stop", "usage": None}


class RefinementTests(unittest.TestCase):
    def engine(self, directory, provider=None, selected=None):
        fixture = dataset()
        fixture["taxonomy"]["parent_child_edges"] = []
        fixture["taxonomy"]["unplaced_concept_ids"] = fixture["taxonomy"]["frozen_input_concept_ids"]
        data = InputData(fixture, selected or ["SYN-ACT-A", "SYN-ACT-B", "SYN-INH-A"], {}, 1)
        provider = provider or FixtureAPI()
        config = {"clustering": {"distance_cutoff": .43, "root_weight": .15},
                  "discovery": {"k": 3, "block_size": 2, "max_candidates_per_round": 5},
                  "max_rounds": 2, "max_group_size": 3, "max_new_nodes": 5,
                  "parent_candidates": 3, "reuse_candidates": 4, "max_ancestor_paths": 4,
                  "audit_batch_size": 5, "rejected_group_split_depth": 1, "response_retries": 0,
                  "llm": {"stages": {s: {"model": "fixture"} for s in REFINEMENT_PROMPTS}}}
        embeddings = Embeddings(provider, {"model": "fixture", "batch_size": 6}, data, directory / "embeddings")
        journal = LLMJournal(directory / "llm", provider, REFINEMENT_PROMPTS, config["llm"]["stages"], 100)
        return FiveStepInducer(directory, config, data, embeddings, journal)

    def test_independent_containment_rejects_equivalence_reverse_and_uncertainty(self):
        for pair, expected in [(('true', 'false'), 'valid'), (('true', 'true'), 'invalid'),
                               (('false', 'true'), 'invalid'), (('true', 'uncertain'), 'uncertain')]:
            response = {"checks": [{"edge_id": "e", "child_entails_parent": pair[0],
                                     "parent_entails_child": pair[1], "reason": "fixture"}]}
            self.assertEqual(containment_checks(response, {"e"})["checks"][0]["verdict"], expected)
        with self.assertRaises(ContractError):
            containment_checks({"checks": []}, {"e"})

    def test_knn_overlaps_and_blocking_do_not_make_edges(self):
        vectors = np.array([[1, 0], [.95, .05], [.85, .15], [.7, .3]])
        args = dict(k=2, cutoff=.43, root_weight=.15, max_group_size=3)
        a, _ = knn_candidates(list('ABCD'), vectors, np.eye(2), block_size=1, **args)
        b, _ = knn_candidates(list('ABCD'), vectors, np.eye(2), block_size=4, **args)
        self.assertEqual([x.members for x in a], [x.members for x in b])
        self.assertTrue(any(set(x.members) & set(y.members) and x.members != y.members for x in a for y in a))
        c, _ = knn_candidates(list('ABCD'), vectors, np.eye(2), forbidden={('A', 'B')}, **args)
        self.assertFalse(any({'A', 'B'} <= set(x.members) for x in c))

    def test_invalid_child_is_excluded_before_insertion_and_provenance_is_filtered(self):
        with TemporaryDirectory() as temp:
            engine = self.engine(Path(temp), FixtureAPI(bad_child="Inhibition of gene A", bad_placement=True))
            proposal = {"reuse_id": None, "name": "Fixture parent", "definition": "Fixture abstraction",
                        "children": [{"id": n, "reason": "proposal"} for n in sorted(engine.scope)]}
            parent = engine.apply_proposal(proposal, "fixture.json")
            view = GraphView(engine.taxonomy)
            self.assertEqual(view.children[parent], {"SYN-ACT-A", "SYN-ACT-B"})
            self.assertFalse(view.parents[parent])
            self.assertEqual(set(view.nodes[parent].supporting_event_ids),
                             set(view.nodes['SYN-ACT-A'].supporting_event_ids + view.nodes['SYN-ACT-B'].supporting_event_ids))
            self.assertTrue(all(engine.validations[e.edge_id]['verdict'] == 'valid'
                                for e in engine.taxonomy.parent_child_edges))
            validate_source(engine.data.dataset, engine.taxonomy)

    def test_failed_validation_has_no_graph_side_effect(self):
        with TemporaryDirectory() as temp:
            engine = self.engine(Path(temp), FixtureAPI(bad_child="Activation of gene B"))
            before = engine.taxonomy
            with self.assertRaises(CandidateError):
                engine.apply_proposal({"reuse_id": None, "name": "Fixture parent", "definition": "Fixture abstraction",
                    "children": [{"id": n, "reason": "proposal"} for n in ['SYN-ACT-A', 'SYN-ACT-B']]}, "fixture.json")
            self.assertIs(engine.taxonomy, before)

    def test_insertion_rewires_shortcuts_retains_other_parent_and_rolls_back_cycle(self):
        with TemporaryDirectory() as temp:
            engine = self.engine(Path(temp))
            p, other, a, b = 'SYN-BIO', 'SYN-RESEARCH', 'SYN-ACT-A', 'SYN-ACT-B'
            initial = update_graph(engine.taxonomy, edges=[edge(p, n, 'fixture', 'fixture', 'fixture') for n in (a,b)] +
                                   [edge(other, a, 'fixture', 'fixture', 'fixture')])
            view = GraphView(initial)
            node = intermediate(initial, engine.data, {"name": "Fixture parent", "definition": "Fixture abstraction"},
                                [view.nodes[a], view.nodes[b]], 'fixture')
            additions = [edge(x,y,'fixture','fixture','fixture') for x,y in [(p,node.concept_id),(node.concept_id,a),(node.concept_id,b)]]
            after, removed = insert_transaction(initial, node, additions)
            self.assertEqual(len(removed), 2)
            self.assertIn(other, GraphView(after).parents[a])
            assert_preserved(initial, after, accepted=True)
            with self.assertRaises(CandidateError):
                insert_transaction(after, None, [edge(a,p,'fixture','fixture','fixture')])
            self.assertEqual(len(initial.parent_child_edges), 3)

    def test_end_to_end_retries_and_local_recursion_preserve_inputs(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            engine = self.engine(directory)
            result = engine.run()
            self.assertGreater(result['rounds'][0]['actions'].get('reject', 0), 0)
            self.assertGreater(result['rounds'][0]['committed_new'], 0)
            self.assertEqual(len(result['rounds']), 2)
            assert_preserved(engine.data.initial, engine.taxonomy, accepted=True)
            saved = json.loads((directory / 'final_taxonomy.json').read_text())
            self.assertEqual(len(saved['frozen_input_concept_ids']), 8)
            records = [json.loads(p.read_text()) for p in (directory/'trace').glob('*.json')]
            self.assertTrue(any(r.get('retry') == 1 for r in records))

    def test_smaller_retry_survives_round_quota(self):
        with TemporaryDirectory() as temp:
            engine = self.engine(Path(temp))
            engine.config['discovery']['max_candidates_per_round'] = 1
            result = engine.run()
            self.assertEqual(result['rounds'][0]['committed_new'], 0)
            self.assertGreater(result['rounds'][0]['pending_candidates'], 0)
            self.assertEqual(result['rounds'][1]['committed_new'], 1)

    def test_confirmed_shortcut_is_not_downgraded_to_provisional_path(self):
        with TemporaryDirectory() as temp:
            engine = self.engine(Path(temp))
            parent, a, b = 'SYN-BIO', 'SYN-ACT-A', 'SYN-ACT-B'
            confirmed = edge(parent, a, 'fixture', 'fixture', 'fixture')
            confirmed.status = 'accepted'
            before = update_graph(engine.taxonomy, edges=[confirmed])
            view = GraphView(before)
            node = intermediate(before, engine.data, {'name': 'Fixture parent', 'definition': 'Fixture abstraction'},
                                [view.nodes[a], view.nodes[b]], 'fixture')
            additions = [edge(x,y,'fixture','fixture','fixture') for x,y in
                         [(parent,node.concept_id),(node.concept_id,a),(node.concept_id,b)]]
            after, removed = insert_transaction(before, node, additions)
            self.assertFalse(removed)
            self.assertIn(confirmed, after.parent_child_edges)
            assert_preserved(before, after, accepted=True)

    def test_reused_parent_removes_redundant_edge_from_its_ancestor(self):
        with TemporaryDirectory() as temp:
            engine = self.engine(Path(temp))
            p, a, b = 'SYN-BIO', 'SYN-ACT-A', 'SYN-ACT-B'
            before = update_graph(engine.taxonomy, edges=[edge(p,n,'fixture','fixture','fixture') for n in (a,b)])
            after, removed = insert_transaction(before, None, [edge(a,b,'fixture','fixture','fixture')])
            self.assertEqual([(e['parent_id'],e['child_id']) for e in removed], [(p,b)])
            self.assertEqual(GraphView(after).parents[b], {a})
            assert_preserved(before, after, accepted=True)


if __name__ == '__main__':
    unittest.main()
