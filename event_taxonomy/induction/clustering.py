"""Replaceable sibling grouping. Similarity proposes; LLM validation decides."""
import math
from .contracts import CandidateError, ids


class LLMGuidedClusterer:
    name = 'llm_guided'

    def cluster(self, context, diagnosis, taxonomy):
        known = {c['concept_id'] for c in context['children']}
        groups, seen = [], set()
        for group in diagnosis['candidate_subgroups']:
            ids(group['child_ids'], known)
            key = frozenset(group['child_ids'])
            if key not in seen:
                groups.append(dict(group)); seen.add(key)
        return groups, {'method': self.name, 'unassigned_child_ids': sorted(known - set().union(*(set(g['child_ids']) for g in groups)))}


class EmbeddingClusterer:
    """Deterministic complete-link cosine clustering of CURRENT siblings only.

    Input vectors must be supplied, never downloaded. An induced node lacking a
    vector gets the normalized mean of its unique vector-bearing descendants.
    Missing vectors remain outliers. This is not a TaxoAdapt reproduction.
    """
    name = 'embedding_complete_link'

    def __init__(self, vectors, threshold=0.82, max_cluster_size=20):
        if not -1 <= threshold <= 1 or max_cluster_size < 2:
            raise ValueError('Invalid embedding clustering parameters')
        self.threshold, self.max_size = threshold, max_cluster_size
        self.vectors, dimension = {}, None
        for key, value in vectors.items():
            if not isinstance(value, list) or not value or any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
                raise ValueError('Embedding must be a finite numeric vector')
            dimension = dimension or len(value)
            if len(value) != dimension:
                raise ValueError('Embedding dimensions differ')
            length = math.sqrt(sum(v * v for v in value))
            if not length:
                raise ValueError('Zero embedding vector')
            self.vectors[key] = [v / length for v in value]

    def _representations(self, taxonomy, wanted):
        from ..taxonomy.dag import adjacency, reachable
        graph, _ = adjacency([n.concept_id for n in taxonomy.nodes], taxonomy.parent_child_edges)
        induced = {n.concept_id for n in taxonomy.nodes if n.origin_kind == 'induced_intermediate'}
        result, derived = {}, {}
        for nid in wanted:
            if nid in self.vectors:
                result[nid] = self.vectors[nid]
            elif nid in induced:
                sources = sorted(reachable(graph, nid) & set(self.vectors))
                if sources:
                    avg = [sum(self.vectors[s][i] for s in sources) / len(sources) for i in range(len(self.vectors[sources[0]]))]
                    length = math.sqrt(sum(v*v for v in avg))
                    if length:
                        result[nid] = [v / length for v in avg]; derived[nid] = sources
        return result, derived

    def cluster(self, context, diagnosis, taxonomy):
        wanted = {c['concept_id'] for c in context['children']}
        vectors, derived = self._representations(taxonomy, wanted)
        clusters = [(i,) for i in sorted(vectors)]
        while True:
            best = None
            for i, left in enumerate(clusters):
                for j in range(i + 1, len(clusters)):
                    right = clusters[j]
                    if len(left) + len(right) > self.max_size:
                        continue
                    score = min(sum(a*b for a, b in zip(vectors[x], vectors[y])) for x in left for y in right)
                    # Stable traversal order breaks exact ties deterministically.
                    if score >= self.threshold and (best is None or score > best[0]):
                        best = (score, i, j)
            if best is None:
                break
            _, i, j = best
            merged = tuple(sorted(clusters[i] + clusters[j]))
            clusters = sorted([g for k, g in enumerate(clusters) if k not in (i, j)] + [merged])
        groups = [{'group_id': 'embedding-%d' % i, 'child_ids': list(group),
                   'reasoning': 'Complete-link cosine candidate among current siblings; requires semantic validation.'}
                  for i, group in enumerate(clusters) if len(group) >= 2]
        assigned = set().union(*(set(g['child_ids']) for g in groups))
        return groups, {'method': self.name, 'threshold': self.threshold,
            'derived_vectors_from_unique_descendants': derived,
            'missing_vector_ids': sorted(wanted - set(vectors)), 'unassigned_child_ids': sorted(wanted - assigned)}


def make_clusterer(config, vectors=None):
    if config['method'] == 'llm_guided':
        return LLMGuidedClusterer()
    if config['method'] == 'embedding':
        return EmbeddingClusterer(vectors or {}, config['cosine_threshold'], config['max_cluster_size'])
    raise CandidateError('Unknown clustering method')
