"""Retrieve possible synonyms and prepare the workbench's embedding views."""

from collections import defaultdict
from heapq import nsmallest
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

from .embedding import LOCAL_KIND, EmbeddingAPI, Embeddings, LocalCharNgramEmbeddings
from ..model import endpoint, live_provider, read_settings
from ..storage import digest, norm, read_json, run_lock


DEFAULT_KNN_K = 2
CLUSTER_SETTINGS = {
    "min_cluster_size": 2,
    "min_samples": 1,
    "cluster_selection_method": "eom",
    "allow_single_cluster": True,
    "cluster_selection_epsilon": 0.25,
}


def _entity_vectors(rows, directory, config_path):
    cfg = read_json(config_path)["embedding"]
    connection, parameters = read_settings(config_path, "audit_edges")
    local = cfg.get("kind") == LOCAL_KIND
    if local:
        if type(cfg.get("dimensions")) is not int or cfg["dimensions"] < 64:
            raise ValueError("Local embeddings need integer dimensions of at least 64.")
        base, provider = LOCAL_KIND, LocalCharNgramEmbeddings(cfg)
    else:
        base = endpoint(cfg).rstrip("/")
        if (not base or base != endpoint(connection).rstrip("/")
                or cfg["api_key_env"] != connection["api_key_env"]):
            raise ValueError("Embedding and language models must use the same configured credential and endpoint.")
        cfg["base_url"] = base
        provider = EmbeddingAPI(cfg)
    model_spec = {key: cfg[key] for key in ("model", "dimensions", "batch_size")}
    vector_dir = directory / "vectors" / digest([base, cfg["model"], cfg["dimensions"]])[:20]
    vector_dir.mkdir(parents=True, exist_ok=True)
    texts = {
        row["id"]: "\n".join([row["name"], row["kind"], row["definition"], *row["aliases"]])
        for row in rows
    }
    embeddings = Embeddings(provider, cfg, vector_dir)
    embeddings.restore()
    cached = sum(digest([entity_id, text]) in embeddings.vectors for entity_id, text in texts.items())
    if cached < len(texts) and not local:
        live_provider(connection, parameters)
    vectors = embeddings.for_texts(texts)
    return vectors, {
        **model_spec, "returned_model": embeddings.returned_model,
        "cache_dir": str(vector_dir.relative_to(directory)), "restored_vectors": cached,
        "input_fields": ["name", "kind", "definition", "aliases"],
    }


def _cluster_view(rows, vectors, distances):
    count, dimension = vectors.shape
    coordinates = np.zeros((count, 2))
    ratios = np.zeros(2)
    if count > 1 and not np.all(vectors == vectors[0]):
        pca = PCA(n_components=min(count, dimension, 2), svd_solver="full")
        coordinates[:, :pca.n_components] = pca.fit_transform(vectors)
        ratios[:pca.n_components] = pca.explained_variance_ratio_
    projection = {
        "method": "PCA", "axis_labels": ["PC1", "PC2"], "explained_variance_ratio": ratios.tolist(),
    }
    labels, probabilities = [-1] * count, [0.0] * count
    if count >= CLUSTER_SETTINGS["min_cluster_size"]:
        estimator = HDBSCAN(metric="precomputed", copy=True, **CLUSTER_SETTINGS)
        labels = estimator.fit_predict(distances)
        probabilities = estimator.probabilities_
    memberships = defaultdict(list)
    points = []
    for row, (x, y), label, probability in zip(rows, coordinates, labels, probabilities):
        label = int(label)
        points.append({
            "source_id": row["id"], "concept_name": row["name"], "description": row["definition"],
            "x": float(x), "y": float(y), "cluster_id": label, "probability": float(probability),
        })
        if label != -1:
            memberships[label].append(row["id"])
    clusters = [
        {"cluster_id": label, "member_ids": members, "pair_count": len(members) * (len(members) - 1) // 2}
        for label, members in sorted(memberships.items())
    ]
    return points, clusters, projection


def _validate_entities(rows):
    seen = set()
    for row in rows:
        for key in ("id", "name", "definition", "kind"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Every entity needs a nonempty {key}.")
        ident = row["id"].strip()
        if ident in seen:
            raise ValueError(f"Duplicate entity ID {ident}.")
        seen.add(ident)
        if (not isinstance(row["aliases"], list)
                or not all(isinstance(alias, str) and alias.strip() for alias in row["aliases"])):
            raise ValueError(f"Invalid aliases for {ident}.")


def retrieve_pairs(rows, directory, config_path, *, knn_k=DEFAULT_KNN_K):
    """Select candidates with global cosine neighbors and shared names or aliases."""
    if knn_k < 1:
        raise ValueError("KNN requires at least one neighbor.")
    search = {"knn_k": knn_k, "metric": "cosine", "shared_names_and_aliases": True}
    clustering = {
        "method": "HDBSCAN", "distance": "full-dimensional cosine", "settings": dict(CLUSTER_SETTINGS),
    }
    if not rows:
        model = read_json(config_path)["embedding"]
        return {
            "input_count": 0, "possible_pairs": 0, "candidate_count": 0,
            "embedding": {key: model[key] for key in ("model", "dimensions")},
            "candidate_search": search,
            "clustering": {**clustering, "clusters": [], "noise_ids": []},
            "projection": {"method": "PCA", "explained_variance_ratio": []},
            "points": [], "pairs": [],
        }
    _validate_entities(rows)
    directory = Path(directory) / "retrieval"
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        vectors, embedding = _entity_vectors(rows, directory, Path(config_path))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            distances = cosine_distances(np.asarray(vectors, dtype=np.float64))
        if not np.isfinite(distances).all():
            raise ValueError("Cosine distance matrix contains non-finite values.")
        np.fill_diagonal(distances, 0.0)
        ids = [row["id"] for row in rows]
        indexes = {entity_id: i for i, entity_id in enumerate(ids)}
        reasons = defaultdict(set)
        by_name = defaultdict(set)
        for row in rows:
            for alias in [row["name"], *row["aliases"]]:
                by_name[norm(alias)].add(row["id"])
        for group in by_name.values():
            for pair in combinations(sorted(group), 2):
                reasons[pair].add("shared_name_or_alias")
        for i, entity_id in enumerate(ids):
            neighbors = nsmallest(
                knn_k, (j for j in range(len(ids)) if j != i),
                key=lambda j: (distances[i, j], ids[j]),
            )
            for j in neighbors:
                reasons[tuple(sorted((entity_id, ids[j])))].add("global_knn")
        pairs = [
            {"left": left, "right": right, "reasons": sorted(pair_reasons),
             "cosine_similarity": round(1.0 - float(distances[indexes[left], indexes[right]]), 6)}
            for (left, right), pair_reasons in reasons.items()
        ]
        pairs.sort(key=lambda p: ("shared_name_or_alias" not in p["reasons"],
                                  -p["cosine_similarity"], p["left"], p["right"]))
        points, clusters, projection = _cluster_view(rows, vectors, distances)
        return {
            "input_count": len(rows), "possible_pairs": len(rows) * (len(rows) - 1) // 2,
            "candidate_count": len(pairs), "embedding": embedding, "candidate_search": search,
            "clustering": {
                **clustering, "clusters": clusters,
                "noise_ids": [p["source_id"] for p in points if p["cluster_id"] == -1],
            },
            "projection": projection, "points": points, "pairs": pairs,
        }
