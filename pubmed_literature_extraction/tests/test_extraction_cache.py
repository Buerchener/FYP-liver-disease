import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cognitive_agent.extraction_cache import LightweightExtractionCache
from cognitive_agent.extraction_kernel import ExtractionKernel, RawExtraction


class LightweightExtractionCacheTests(unittest.TestCase):
    def test_memory_mode_creates_no_disk_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "must-not-exist.sqlite3"
            cache = LightweightExtractionCache(mode="memory", path=path)
            payload, status = cache.get_or_compute(
                "key", lambda: ({"entities": [{"mention": "TP53"}]}, 1.5),
                cacheable=lambda value: bool(value["entities"]),
            )
            self.assertEqual(status, "miss")
            self.assertFalse(path.exists())
            cache.close()

    def test_memory_lru_is_bounded(self):
        cache = LightweightExtractionCache(mode="memory", memory_max_entries=2)
        for index in range(3):
            cache.get_or_compute(
                str(index), lambda index=index: ({"entities": [index]}, 0.1),
                cacheable=lambda value: True,
            )
        stats = cache.stats()
        self.assertEqual(stats["memory_entries"], 2)
        self.assertEqual(stats["evictions"], 1)

    def test_singleflight_executes_factory_once(self):
        cache = LightweightExtractionCache(mode="memory")
        calls = 0
        lock = threading.Lock()
        def factory():
            nonlocal calls
            with lock:
                calls += 1
            time.sleep(0.03)
            return {"entities": [{"mention": "TP53"}]}, 0.03
        with ThreadPoolExecutor(max_workers=12) as pool:
            rows = list(pool.map(
                lambda _: cache.get_or_compute(
                    "same", factory, cacheable=lambda value: True,
                ), range(12),
            ))
        self.assertEqual(calls, 1)
        self.assertEqual(len(rows), 12)
        self.assertGreaterEqual(cache.stats()["singleflight_waits"], 1)

    def test_noncacheable_failure_payload_is_not_reused(self):
        cache = LightweightExtractionCache(mode="memory")
        calls = 0
        def factory():
            nonlocal calls
            calls += 1
            return {"entities": [], "error": "timeout"}, 0.1
        for _ in range(2):
            cache.get_or_compute(
                "failed", factory,
                cacheable=lambda value: bool(value["entities"]) and not value["error"],
            )
        self.assertEqual(calls, 2)
        self.assertEqual(cache.stats()["writes"], 0)

    def test_persistent_cache_survives_process_object_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.sqlite3"
            first = LightweightExtractionCache(mode="persistent", path=path)
            first.get_or_compute(
                "key", lambda: ({"entities": [{"mention": "TP53"}]}, 2.0),
                cacheable=lambda value: True,
            )
            first.close()
            second = LightweightExtractionCache(mode="persistent", path=path)
            payload, status = second.get_or_compute(
                "key", lambda: self.fail("persistent hit should avoid factory"),
                cacheable=lambda value: True,
            )
            self.assertEqual(status, "persistent_hit")
            self.assertEqual(payload["entities"][0]["mention"], "TP53")
            second.close()

    def test_persistent_cache_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = LightweightExtractionCache(
                mode="persistent", path=Path(temp) / "cache.sqlite3",
                persistent_max_entries=2,
            )
            for index in range(3):
                cache.get_or_compute(
                    str(index), lambda index=index: ({"entities": [index]}, 0.1),
                    cacheable=lambda value: True,
                )
            self.assertEqual(cache.stats()["persistent_entries"], 2)
            cache.close()


class ExtractionKernelCacheTests(unittest.TestCase):
    def make_kernel(self):
        cache = LightweightExtractionCache(mode="memory")
        return ExtractionKernel(SimpleNamespace(model_id="test-model"), cache=cache)

    def test_hit_rebinds_current_document_id(self):
        kernel = self.make_kernel()
        fake = RawExtraction(
            pmid="old", entities=[{"mention": "TP53", "type": "Gene"}],
            relations=[],
        )
        with patch.object(kernel, "_extract_uncached", return_value=fake) as remote:
            first = kernel.extract("same text", document_id="PMID-1", examples=[], prompt="v1")
            second = kernel.extract("same text", document_id="PMID-2", examples=[], prompt="v1")
        self.assertEqual(remote.call_count, 1)
        self.assertEqual(first.pmid, "PMID-1")
        self.assertEqual(second.pmid, "PMID-2")
        self.assertIn("extraction_cache:memory_hit", second.warnings)

    def test_valid_empty_extraction_is_cached_for_exact_warm_replay(self):
        kernel = self.make_kernel()
        empty = RawExtraction(
            pmid="old", entities=[], relations=[],
            warnings=["All 1 attempts produced 0 entities"],
        )
        with patch.object(kernel, "_extract_uncached", return_value=empty) as remote:
            first = kernel.extract("same empty text", document_id="PMID-1", examples=[], prompt="v1")
            second = kernel.extract("same empty text", document_id="PMID-1", examples=[], prompt="v1")
        self.assertEqual(remote.call_count, 1)
        self.assertIn("extraction_cache:miss", first.warnings)
        self.assertIn("extraction_cache:memory_hit", second.warnings)

    def test_parse_failure_is_not_cached_even_without_error_field(self):
        kernel = self.make_kernel()
        malformed = RawExtraction(
            pmid="old", entities=[], relations=[],
            warnings=["Parse error: invalid structured response"],
        )
        with patch.object(kernel, "_extract_uncached", return_value=malformed) as remote:
            kernel.extract("malformed", document_id="PMID-1", examples=[], prompt="v1")
            kernel.extract("malformed", document_id="PMID-1", examples=[], prompt="v1")
        self.assertEqual(remote.call_count, 2)

    def test_prompt_or_model_change_invalidates_key(self):
        kernel = self.make_kernel()
        fake = RawExtraction(entities=[{"mention": "TP53"}])
        with patch.object(kernel, "_extract_uncached", return_value=fake) as remote:
            kernel.extract("text", examples=[], prompt="prompt-v1")
            kernel.extract("text", examples=[], prompt="prompt-v2")
            kernel.model_config.model_id = "other-model"
            kernel.extract("text", examples=[], prompt="prompt-v2")
        self.assertEqual(remote.call_count, 3)

    def test_api_key_is_excluded_but_endpoint_is_part_of_key(self):
        def kernel(secret, endpoint):
            return ExtractionKernel(SimpleNamespace(
                provider="gemini", model_id="same-model",
                provider_kwargs={
                    "api_key": secret,
                    "http_options": {"base_url": endpoint},
                },
            ))
        first = kernel("secret-a", "https://endpoint-a.example")
        second = kernel("secret-b", "https://endpoint-a.example")
        third = kernel("secret-b", "https://endpoint-b.example")
        args = {"text": "text", "examples": [], "prompt": "prompt", "retry_on_empty": True}
        self.assertEqual(first._cache_key(**args), second._cache_key(**args))
        self.assertNotEqual(first._cache_key(**args), third._cache_key(**args))

    def test_openai_base_url_is_part_of_key_but_api_key_is_not(self):
        def kernel(secret, endpoint):
            return ExtractionKernel(SimpleNamespace(
                provider="openai", model_id="same-model",
                provider_kwargs={"api_key": secret, "base_url": endpoint},
            ))
        first = kernel("secret-a", "https://endpoint-a.example/v1")
        second = kernel("secret-b", "https://endpoint-a.example/v1/")
        third = kernel("secret-b", "https://endpoint-b.example/v1")
        args = {"text": "text", "examples": [], "prompt": "prompt", "retry_on_empty": True}
        self.assertEqual(first._cache_key(**args), second._cache_key(**args))
        self.assertNotEqual(first._cache_key(**args), third._cache_key(**args))

    def test_inner_worker_count_invalidates_cache_key(self):
        config = SimpleNamespace(provider="openai", model_id="model", provider_kwargs={})
        args = {"text": "text", "examples": [], "prompt": "prompt", "retry_on_empty": True}
        self.assertNotEqual(
            ExtractionKernel(config, inner_max_workers=1)._cache_key(**args),
            ExtractionKernel(config, inner_max_workers=2)._cache_key(**args),
        )

    def test_timeout_has_a_separate_small_retry_budget(self):
        kernel = self.make_kernel()
        with (
            patch.dict(
                "os.environ",
                {
                    "PRIMARY_LLM_MAX_RETRIES": "5",
                    "PRIMARY_LLM_TIMEOUT_MAX_RETRIES": "1",
                    "PRIMARY_LLM_RETRY_BASE_DELAY_S": "0",
                },
            ),
            patch(
                "cognitive_agent.extraction_kernel.lx.extract",
                side_effect=TimeoutError("Request timed out."),
            ) as remote,
            patch("cognitive_agent.extraction_kernel.time.sleep"),
        ):
            result = kernel._extract_uncached(
                text="text", document_id="timeout", examples=[],
                prompt="prompt", retry_on_empty=False,
            )
        self.assertEqual(remote.call_count, 2)
        self.assertEqual(result.retry_count, 1)
        self.assertIn("timed out", result.error)


if __name__ == "__main__":
    unittest.main()
