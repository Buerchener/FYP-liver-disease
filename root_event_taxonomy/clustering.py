"""Soft root guidance for agglomerative cosine/average candidate generation."""
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist


@dataclass(frozen=True)
class Candidate:
    members: tuple
    height: float = 0.0
    parts: tuple = ()


def candidates(ids, vectors, root_vectors, cutoff, root_weight, max_group_size):
    """Build a flat partition and retain subclusters for a bounded retry."""
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    roots = root_vectors / np.linalg.norm(root_vectors, axis=1, keepdims=True)
    root_scores = np.clip(unit @ roots.T, -1, 1)
    trees = [Candidate((node_id,)) for node_id in ids]
    if len(ids) < 2:
        return trees, root_scores

    cosine = np.clip(pdist(unit, "cosine"), 0, 2)
    root_delta = pdist(root_scores, "cityblock") / len(roots)
    distances = (1 - root_weight) * cosine + root_weight * root_delta
    for left, right, height, _ in linkage(distances, method="average"):
        parts = (trees[int(left)], trees[int(right)])
        members = tuple(sorted(parts[0].members + parts[1].members))
        trees.append(Candidate(members, float(height), parts))

    partition, pending = [], [trees[-1]]
    while pending:
        candidate = pending.pop()
        if candidate.height <= cutoff and len(candidate.members) <= max_group_size:
            partition.append(candidate)
        else:
            pending.extend(reversed(candidate.parts))
    return sorted(partition, key=lambda c: c.members), root_scores
