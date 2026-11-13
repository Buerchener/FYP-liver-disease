"""Evidence-backed local DAG views; context limits never silently drop siblings."""
from ..models import canonical_json, to_dict
from .contracts import ContextLimit
from .graph import context_parts


class ContextBuilder:
    def __init__(self, dataset, settings):
        self.settings = settings
        self.events = {row['event_id']: row for row in dataset.get('events', [])}
        self.evidence = {row['evidence_id']: row for row in dataset.get('evidence', [])}

    def node(self, node):
        result = to_dict(node)
        limit = self.settings['evidence_per_concept']
        event_ids, evidence_ids = sorted(node.supporting_event_ids), sorted(node.evidence)
        result['supporting_events_sample'] = [self.events[e] for e in event_ids[:limit] if e in self.events]
        result['evidence_sample'] = [self.evidence[e] for e in evidence_ids[:limit] if e in self.evidence]
        result['sample_metadata'] = {'selection': 'sorted ID prefix, deterministic',
            'event_total': len(event_ids), 'evidence_total': len(evidence_ids),
            'events_omitted': max(0, len(event_ids) - limit),
            'evidence_omitted': max(0, len(evidence_ids) - limit)}
        return result

    def build(self, taxonomy, parent):
        nodes, ancestors, direct, edges, relevant = context_parts(taxonomy, parent)
        reverse = {i: [] for i in nodes}
        for edge in taxonomy.parent_child_edges:
            reverse[edge.child_id].append(edge.parent_id)
        cap = self.settings['max_ancestor_paths']
        paths, pending = [], [[parent]]
        while pending and len(paths) <= cap:
            path = pending.pop()
            parents = sorted(reverse[path[0]], reverse=True)
            if not parents:
                paths.append(path)
            else:
                pending.extend([[p] + path for p in parents])
        result = {'parent': self.node(nodes[parent]),
            'children': [self.node(nodes[c]) for c in sorted(direct)],
            'ancestor_nodes': [self.node(nodes[a]) for a in sorted(ancestors)],
            'ancestor_paths': paths[:cap], 'ancestor_paths_truncated': bool(pending) or len(paths) > cap,
            'ancestor_edges_complete': True,
            'ancestor_edges': [to_dict(e) for e in edges if e.child_id in ancestors | {parent}],
            'local_edges': [to_dict(e) for e in edges],
            'other_parent_nodes': [self.node(nodes[n]) for n in sorted(relevant - ancestors - direct - {parent})],
            'data_origin': 'synthetic' if any(e.get('data_origin') == 'synthetic' for e in self.events.values()) else 'imported',
            'warning': 'Evidence samples are bounded; IDs and ancestor edges are complete. Missing evidence is not negative evidence. Node provisional_parent_ids are immutable input metadata; current parentage is defined only by graph edges.'}
        self.check(result)
        return result

    def check(self, value):
        if len(canonical_json(value)) > self.settings['max_characters']:
            raise ContextLimit('context_character_limit')


def concept_catalog(taxonomy):
    return [{'concept_id': n.concept_id, 'label': n.concept_label, 'description': n.concept_description,
             'aliases': n.aliases} for n in sorted(taxonomy.nodes, key=lambda n: n.concept_id)]
