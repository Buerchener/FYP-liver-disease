"""Batched OpenAI-compatible text embeddings with evidence-aware inputs."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from event_taxonomy.induction.contracts import ProviderError
from event_taxonomy.induction.graph import digest
from event_taxonomy.induction.storage import atomic_json


class EmbeddingAPI:
    def __init__(self, config):
        self.config = config

    def embed(self, texts):
        cfg = self.config
        key = os.environ.get(cfg["api_key_env"])
        if not key:
            raise ProviderError("Embedding API key environment variable is not set")
        url = cfg["base_url"].rstrip("/") + "/embeddings"
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ProviderError("Invalid HTTPS embedding base URL")
        body = {"model": cfg["model"], "input": texts}
        if cfg.get("dimensions") is not None:
            body["dimensions"] = cfg["dimensions"]
        request = urllib.request.Request(
            url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=cfg["timeout_seconds"]) as response:
                envelope = json.loads(response.read())
            rows = sorted(envelope["data"], key=lambda row: row["index"])
            if [row["index"] for row in rows] != list(range(len(texts))):
                raise ValueError("Incomplete embedding indices")
            return {
                "vectors": [row["embedding"] for row in rows],
                "model": str(envelope.get("model") or cfg["model"]).replace(key, "[REDACTED]"),
                "usage": envelope.get("usage"),
            }
        except urllib.error.HTTPError as error:
            raise ProviderError("Embedding HTTP status %d" % error.code) from None
        except (OSError, ValueError, KeyError, TypeError):
            raise ProviderError("Embedding transport or response-envelope error") from None


class Embeddings:
    def __init__(self, provider, config, data, directory):
        self.provider = provider
        self.config = config
        self.data = data
        self.directory = directory
        self.vectors = {}
        self.batches = 0
        self.dimension = config.get("dimensions")
        self.returned_model = None

    def for_nodes(self, nodes):
        by_id = {n.concept_id: n for n in nodes}
        texts = {}
        for node_id, node in by_id.items():
            described = self.data.describe(
                node, evidence=True, excerpt_chars=self.config.get("evidence_char_limit", 900),
            )
            lines = [described["name"], described["definition"]]
            lines.extend(ev["excerpt"] for ev in described["evidence"])
            texts[node_id] = "\n".join(lines)
        keys = {node_id: digest([node_id, value]) for node_id, value in texts.items()}
        missing = [node_id for node_id in by_id if keys[node_id] not in self.vectors]
        batch_size = self.config["batch_size"]
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            self.batches += 1
            path = self.directory / ("%06d.json" % self.batches)
            record = {"status": "request_started", "model": self.config["model"],
                      "input": [{"concept_id": n, "text": texts[n]} for n in batch]}
            atomic_json(path, record)
            try:
                response = self.provider.embed([texts[n] for n in batch])
                vectors = self._check(response, len(batch))
            except ProviderError as error:
                atomic_json(path, {**record, "status": "provider_error", "error": str(error)})
                raise
            atomic_json(path, {**record, "status": "response_received", **response})
            for node_id, vector in zip(batch, vectors):
                self.vectors[keys[node_id]] = vector
        return np.vstack([self.vectors[keys[n]] for n in by_id])

    def _check(self, response, count):
        try:
            vectors = np.asarray(response["vectors"], dtype=float)
            model = response["model"]
        except (ValueError, TypeError, KeyError):
            raise ProviderError("Invalid embedding vectors or model") from None
        if (vectors.ndim != 2 or vectors.shape[0] != count or vectors.shape[1] < 1
                or not np.isfinite(vectors).all() or not isinstance(model, str) or not model.strip()):
            raise ProviderError("Invalid embedding matrix or model")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ProviderError("Invalid embedding norm")
        if self.dimension is not None and vectors.shape[1] != self.dimension:
            raise ProviderError("Embedding dimension changed")
        if self.returned_model is not None and self.returned_model != model:
            raise ProviderError("Embedding model changed during the run")
        self.dimension, self.returned_model = vectors.shape[1], model
        return vectors
