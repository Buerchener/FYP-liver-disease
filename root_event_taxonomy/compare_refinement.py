"""Blind, identical definition-based API review of two saved induction results.

The same LLM is a fallible consistency check, not a human gold standard. The
evaluation never changes either graph and does not feed judgments into induction.
"""
import argparse
from collections import Counter
from pathlib import Path
import random

from event_taxonomy.induction.graph import digest, load_taxonomy
from event_taxonomy.induction.journal import LLMJournal
from event_taxonomy.induction.providers import OpenAICompatibleProvider
from event_taxonomy.induction.storage import atomic_json, read_json
from event_taxonomy.taxonomy.dag import reachable, validate_dag
from .__main__ import connection
from .graph import GraphView
from .refinement import REFINEMENT_PROMPTS, containment_checks


def edge_key(view, edge):
    p, c = view.nodes[edge.parent_id], view.nodes[edge.child_id]
    return digest([p.concept_label, p.concept_description, c.concept_label, c.concept_description])


def comparison(old_directory, new_directory, output, provider_config=None):
    output.mkdir(parents=True, exist_ok=False)
    inputs = [read_json(d / 'input_dataset.json') for d in (old_directory, new_directory)]
    if digest(inputs[0]) != digest(inputs[1]):
        raise ValueError('Comparison requires identical input datasets')
    graphs = [load_taxonomy(read_json(d / 'final_taxonomy.json')) for d in (old_directory, new_directory)]
    manifest = read_json(new_directory / 'manifest.json')
    config = connection(provider_config or manifest['config']['llm']['provider'])
    settings = manifest['config']['llm']['stages']['validate_edges']
    journal = LLMJournal(output / 'llm', OpenAICompatibleProvider(config),
                         {'comparison': REFINEMENT_PROMPTS['validate_edges']}, {'comparison': settings}, 80)
    # Deduplicate identical definition pairs so a shared edge gets exactly the
    # same judgment in both runs. Randomize without revealing method or run ID.
    pool = {}
    for graph in graphs:
        view = GraphView(graph)
        for e in graph.parent_child_edges:
            key = edge_key(view, e)
            pool[key] = {'edge_id': key,
                         'parent': {'name': view.nodes[e.parent_id].concept_label,
                                    'definition': view.nodes[e.parent_id].concept_description},
                         'child': {'name': view.nodes[e.child_id].concept_label,
                                   'definition': view.nodes[e.child_id].concept_description}}
    keys = sorted(pool)
    random.Random(20261113).shuffle(keys)
    checks = {}
    atomic_json(output / 'manifest.json', {'input_sha256': digest(inputs[0]),
        'old_run': str(old_directory), 'new_run': str(new_directory), 'unique_edges': len(keys),
        'prompt': REFINEMENT_PROMPTS['validate_edges'], 'settings': settings,
        'note': 'Same-model blind recheck; not an expert gold standard; graphs stay unchanged.'})
    for start in range(0, len(keys), 10):
        batch = keys[start:start + 10]
        aliases = {f'e{i + 1}': key for i, key in enumerate(batch)}
        payload = {'edges': [{**pool[key], 'edge_id': alias} for alias, key in aliases.items()]}
        try:
            value = containment_checks(journal.call('comparison', payload), set(aliases))
        except ValueError:
            value = journal.call('comparison', {**payload, 'correction': 'Return exactly one complete check per supplied edge ID.'})
            value = containment_checks(value, set(aliases))
        for row in value['checks']:
            key = aliases[row['edge_id']]
            checks[key] = {**row, 'edge_id': key, 'ref': journal.last_ref}
        atomic_json(output / 'checks.json', checks)
        print(f'Compared {len(checks)}/{len(keys)} distinct definition pairs', flush=True)
    results = []
    for directory, graph in zip((old_directory, new_directory), graphs):
        view = GraphView(graph)
        verified = {n: set() for n in view.nodes}
        counts = Counter()
        rows = []
        for e in graph.parent_child_edges:
            row = checks[edge_key(view, e)]
            counts[row['verdict']] += 1
            forward, reverse = row['child_entails_parent'], row['parent_entails_child']
            counts['reversed'] += (forward, reverse) == ('false', 'true')
            counts['equivalent'] += (forward, reverse) == ('true', 'true')
            if row['verdict'] == 'valid':
                verified[e.parent_id].add(e.child_id)
            rows.append({**row, 'edge_id': e.edge_id, 'parent_id': e.parent_id, 'child_id': e.child_id,
                         'parent_name': view.nodes[e.parent_id].concept_label,
                         'child_name': view.nodes[e.child_id].concept_label})
        rooted = set().union(*(reachable(verified, r) for r in view.roots))
        leaves = set(graph.frozen_input_concept_ids)
        intermediates = [n for n in graph.nodes if n.origin_kind == 'induced_intermediate']
        result = {'run': directory.name, 'structure': validate_dag(graph),
                  'intermediate_count': len(intermediates), 'edge_review': dict(counts),
                  'supported_root_reachable_leaves': len(leaves & rooted),
                  'parentless_leaves': len([n for n in leaves if not view.parents[n]]),
                  'edges': rows,
                  'intermediates': [{'id': n.concept_id, 'name': n.concept_label,
                      'definition': n.concept_description,
                      'parents': [view.nodes[p].concept_label for p in sorted(view.parents[n.concept_id])],
                      'children': [view.nodes[c].concept_label for c in sorted(view.children[n.concept_id])],
                      'supported_children': len(verified[n.concept_id])} for n in intermediates]}
        results.append(result)
    old_config = read_json(old_directory / 'manifest.json')['config']
    report = {'same_input': True,
              'same_embedding_model': old_config['embedding']['model'] == manifest['config']['embedding']['model'],
              'same_generation_model': old_config['llm']['stages']['group']['model'] == manifest['config']['llm']['stages']['group']['model'],
              'limitations': 'One run per method; same LLM generates and reviews; no expert gold standard.',
              'baseline': results[0], 'five_step': results[1]}
    atomic_json(output / 'comparison.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    comparison(args.baseline.resolve(), args.new.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
