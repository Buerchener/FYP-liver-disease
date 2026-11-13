"""Overlapping cosine KNN candidates; bounded-memory exact search for small pilots."""
import numpy as np

from .clustering import Candidate


def knn_candidates(ids, vectors, root_vectors, *, k, cutoff, root_weight,
                   max_group_size, block_size=64, anchors=None, forbidden=()):
    """Return neighborhoods, not a partition. Never infer an is-a edge here.

    Exact search with per-row sorting costs O(n²(d + log n)) time and
    O(block_size*n + n*k) working space (apart from input vectors);
    an approximate index can replace this routine for larger datasets.
    """
    unit = np.asarray(vectors, dtype=float)
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    roots = np.asarray(root_vectors, dtype=float)
    roots = roots / np.linalg.norm(roots, axis=1, keepdims=True)
    profiles = np.clip(unit @ roots.T, -1, 1)
    anchors = set(ids) if anchors is None else set(anchors)
    forbidden = set(forbidden)
    groups = {}
    for start in range(0, len(ids), block_size):
        distances = 1 - np.clip(unit[start:start + block_size] @ unit.T, -1, 1)
        delta = np.mean(np.abs(profiles[start:start + block_size, None, :] - profiles[None, :, :]), axis=2)
        distances = (1 - root_weight) * distances + root_weight * delta
        for local, row in enumerate(distances):
            i = start + local
            if ids[i] not in anchors:
                continue
            ranked = sorted((float(row[j]), ids[j]) for j in range(len(ids))
                            if i != j and row[j] <= cutoff and
                            (ids[i], ids[j]) not in forbidden and (ids[j], ids[i]) not in forbidden)[:k]
            neighbors = []
            for distance, node in ranked:
                if any((node, other) in forbidden or (other, node) in forbidden for other in neighbors):
                    continue
                neighbors.append(node)
                if len(neighbors) >= max_group_size - 1:
                    break
            if not neighbors:
                continue
            members = tuple(sorted([ids[i], *neighbors]))
            # A rejected neighborhood retries anchor-neighbor pairs, never by
            # weakening the validator or repeatedly submitting the same group.
            pairs = tuple(Candidate(tuple(sorted([ids[i], other])), distance)
                          for distance, other in ranked if other in neighbors)
            candidate = Candidate(members, max(p.height for p in pairs), pairs if len(members) > 2 else ())
            if members not in groups or candidate.height < groups[members].height:
                groups[members] = candidate
    return sorted(groups.values(), key=lambda row: (row.height, row.members)), profiles
