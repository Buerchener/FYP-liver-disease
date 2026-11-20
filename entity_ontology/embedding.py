"""Batched OpenAI-compatible text embeddings with evidence-aware inputs."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from .model import ProviderError
from .storage import atomic_json, digest, read_json


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
        local = (cfg.get("allow_loopback_http") and parsed.scheme == "http"
                 and parsed.hostname in {"127.0.0.1", "localhost", "::1"})
        if (not (parsed.scheme == "https" or local) or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment):
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
                raw = response.read()
        except urllib.error.HTTPError as error:
            raise ProviderError("Embedding HTTP status %d" % error.code) from None
        except (OSError, HTTPException, ValueError, TypeError):
            raise ProviderError("Embedding transport or response-envelope error") from None

        try:
            envelope = json.loads(raw)
            rows = sorted(envelope["data"], key=lambda row: row["index"])
            if [row["index"] for row in rows] != list(range(len(texts))):
                raise ValueError("Incomplete embedding indices")
            result = {
                "vectors": [row["embedding"] for row in rows],
                "model": str(envelope.get("model") or cfg["model"]).replace(key, "[REDACTED]"),
                "usage": envelope.get("usage"),
            }
        except (ValueError, KeyError, TypeError):
            raise ProviderError("Embedding transport or response-envelope error") from None
        return result


class Embeddings:
    def __init__(self, provider, config, directory):
        self.provider = provider
        self.config = config
        self.directory = directory
        self.vectors = {}
        self.batches = 0
        self.dimension = config.get("dimensions")
        self.returned_model = None

    def for_texts(self, texts, *, workers=None):
        """Embed exact strings concurrently, committing vectors in input order.

        Each response is durable before its worker finishes. If one request
        fails, already submitted requests are drained and preserved for resume.
        """
        if not texts:
            raise ValueError("At least one embedding input is required.")
        workers = self.config.get("workers", 1) if workers is None else workers
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("Embedding workers must be an integer from 1 to 8")
        keys = {node_id: digest([node_id, value]) for node_id, value in texts.items()}
        missing = [node_id for node_id in texts if keys[node_id] not in self.vectors]
        batch_size = self.config["batch_size"]
        batches = iter(missing[start:start + batch_size] for start in range(0, len(missing), batch_size))

        def request(batch, path, record):
            try:
                response = self.provider.embed([texts[n] for n in batch])
            except Exception as error:
                atomic_json(path, {**record, "status": "provider_error",
                                   "error": str(error) if isinstance(error, ProviderError) else type(error).__name__})
                raise
            atomic_json(path, {**record, **response, "status": "response_received"})
            return response

        def submit(executor, batch):
            self.batches += 1
            path = self.directory / ("%06d.json" % self.batches)
            record = {"status": "request_started", "model": self.config["model"],
                      "input": [{"concept_id": n, "text": texts[n]} for n in batch]}
            atomic_json(path, record)
            return batch, path, record, executor.submit(request, batch, path, record)

        error = None
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = deque()
            for _ in range(workers):
                batch = next(batches, None)
                if batch is not None:
                    pending.append(submit(executor, batch))
            while pending:
                batch, path, record, future = pending.popleft()
                try:
                    response = future.result()
                    try:
                        vectors = self._check(response, len(batch))
                    except ProviderError as invalid:
                        atomic_json(path, {**record, **response, "status": "response_invalid", "error": str(invalid)})
                        raise
                    for node_id, vector in zip(batch, vectors):
                        self.vectors[keys[node_id]] = vector
                except Exception as failure:
                    if error is None:
                        error = failure
                if error is None:
                    batch = next(batches, None)
                    if batch is not None:
                        pending.append(submit(executor, batch))
        if error is not None:
            raise error
        return np.vstack([self.vectors[keys[n]] for n in texts])

    def restore(self):
        """Restore completed embedding requests from this run's directory."""
        for path in sorted(self.directory.glob("*.json")):
            self.batches = max(self.batches, int(path.stem))
            record = read_json(path)
            if record["status"] != "response_received":
                continue
            inputs = record["input"]
            vectors = self._check(record, len(inputs))
            for item, vector in zip(inputs, vectors):
                key = digest([item["concept_id"], item["text"]])
                self.vectors[key] = vector

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
