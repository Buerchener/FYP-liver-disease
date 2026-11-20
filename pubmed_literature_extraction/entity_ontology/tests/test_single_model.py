"""Running with a single OpenAI-compatible model and no embedding service."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from entity_ontology.concepts.embedding import LocalCharNgramEmbeddings
from entity_ontology.concepts.retrieve_pairs import retrieve_pairs
from entity_ontology.extraction.native import extractor_env

LLM = {"provider": {"kind": "openai_compatible", "base_url_env": "LIVERKG_API_BASE",
                    "api_key_env": "LIVERKG_API_KEY", "timeout_seconds": 60},
       "stages": {"audit_edges": {"model": "chat-model", "temperature": 0, "max_tokens": 6000}}}
LOCAL = {"kind": "local_char_ngram", "model": "local-char-ngram-hash", "dimensions": 256, "batch_size": 2}


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class LocalEmbeddingTest(unittest.TestCase):
    def test_vectors_are_deterministic_normalised_and_lexically_close(self):
        embedder = LocalCharNgramEmbeddings(LOCAL)
        texts = ["type 2 diabetes", "type 2 diabetes mellitus", "liver fibrosis"]
        vectors = np.array(embedder.embed(texts)["vectors"])
        self.assertEqual(vectors.shape, (3, 256))
        self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 1.0))
        self.assertGreater(cosine(vectors[0], vectors[1]), cosine(vectors[0], vectors[2]))
        alone = np.array(embedder.embed(["liver fibrosis"])["vectors"][0])
        self.assertTrue(np.array_equal(alone, vectors[2]))

    def test_retrieval_runs_without_any_api_key(self):
        rows = [{"id": f"m{i}", "name": name, "kind": "Disease", "definition": f"{name} definition",
                 "aliases": []} for i, name in enumerate(["Type 2 diabetes", "Type 2 diabetes mellitus",
                                                           "Liver fibrosis"])]
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"llm": LLM, "embedding": LOCAL}))
            result = retrieve_pairs(rows, directory, config, knn_k=1)
        self.assertEqual(result["embedding"]["model"], "local-char-ngram-hash")
        best = max(result["pairs"], key=lambda pair: pair["cosine_similarity"])
        self.assertEqual({best["left"], best["right"]}, {"m0", "m1"})


class SingleModelExtractorTest(unittest.TestCase):
    def environment(self, directory, extra=None, base="https://relay.example/v1"):
        config = Path(directory) / "config.json"
        config.write_text(json.dumps({"llm": LLM, "embedding": LOCAL}))
        runtime = Path(directory) / "runtime"
        runtime.mkdir()
        variables = {"LIVERKG_API_BASE": base, "LIVERKG_API_KEY": "test-key", **(extra or {})}
        with patch.dict(os.environ, variables, clear=True):
            return extractor_env(runtime, Path(directory), config, runtime / "out", None)

    def test_one_key_drives_every_auxiliary_role(self):
        with TemporaryDirectory() as directory:
            env = self.environment(directory)
        for name, value in (("SECOND_LLM_API_KEY", "test-key"), ("SECOND_LLM_MODEL_ID", "chat-model"),
                            ("SECOND_LLM_API_BASE", "https://relay.example/v1"),
                            ("DEEPSEEK_API_BASE", "https://relay.example/v1"), ("AUX_PRIMARY_MODEL", "chat-model"),
                            ("GEMINI_MODEL", "chat-model")):
            self.assertEqual(env[name], value, name)

    def test_separate_second_model_is_kept(self):
        with TemporaryDirectory() as directory:
            env = self.environment(directory, {"DEEPSEEK_API_KEY": "deepseek-key"})
        self.assertEqual(env["DEEPSEEK_API_KEY"], "deepseek-key")
        self.assertNotIn("SECOND_LLM_MODEL_ID", env)
        self.assertNotIn("AUX_PRIMARY_MODEL", env)

    def test_endpoint_must_be_openai_compatible(self):
        with TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "/v1"):
            self.environment(directory, base="https://relay.example")


if __name__ == "__main__":
    unittest.main()
