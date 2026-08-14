import unittest
import threading
from concurrent.futures import ThreadPoolExecutor

from cognitive_agent.abbreviation_detector import AbbreviationDetector
from cognitive_agent.article_preprocessing import ParallelArticlePreprocessor
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.golden_examples import GoldenExampleSelector


class ParallelArticlePreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ParallelArticlePreprocessor(
            ArticleEvidenceReader(), AbbreviationDetector(), GoldenExampleSelector(),
            max_workers=4,
        )

    def tearDown(self):
        self.preprocessor.close()

    def prepare(self):
        title = "TP53 in hepatocellular carcinoma"
        abstract = "RESULTS: Tumor protein p53 (TP53) was associated with HCC."
        return self.preprocessor.prepare(
            title=title, abstract=abstract,
            text=f"TITLE: {title}\nABSTRACT: {abstract}", study_type="clinical",
            max_examples=4, document_id="cache-test",
        )

    def test_second_preparation_hits_all_component_caches(self):
        first = self.prepare()
        second = self.prepare()
        self.assertFalse(any(first.cache_hits.values()))
        self.assertTrue(all(second.cache_hits.values()))
        self.assertEqual(first.content_hash, second.content_hash)

    def test_cache_is_safe_under_concurrency(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.prepare(), range(16)))
        self.assertEqual(len({item.content_hash for item in results}), 1)
        self.assertGreater(self.preprocessor.cache.stats()["hits"], 0)

    def test_preprocessing_cache_is_bounded(self):
        self.preprocessor.close()
        self.preprocessor = ParallelArticlePreprocessor(
            ArticleEvidenceReader(), AbbreviationDetector(), GoldenExampleSelector(),
            max_workers=4, cache_max_entries=4,
        )
        for index in range(3):
            title = f"TP53 study {index}"
            abstract = f"RESULTS: TP53 was associated with HCC in cohort {index}."
            self.preprocessor.prepare(
                title=title, abstract=abstract,
                text=f"TITLE: {title}\nABSTRACT: {abstract}", study_type="clinical",
                max_examples=4, document_id=str(index),
            )
        stats = self.preprocessor.cache.stats()
        self.assertLessEqual(stats["entries"], 4)
        self.assertGreater(stats["evictions"], 0)

    def test_eviction_does_not_break_waiting_singleflight_consumers(self):
        from cognitive_agent.article_preprocessing import ContentAddressedCache
        cache = ContentAddressedCache(max_entries=1)
        started = threading.Event()
        release = threading.Event()
        calls = 0
        def slow():
            nonlocal calls
            calls += 1
            started.set()
            release.wait(timeout=2)
            return "shared"
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(cache.get_or_compute, "a", "x", slow)
            started.wait(timeout=1)
            waiter = pool.submit(cache.get_or_compute, "a", "x", slow)
            cache.get_or_compute("b", "x", lambda: "other")
            release.set()
            self.assertEqual(first.result()[0], "shared")
            self.assertEqual(waiter.result()[0], "shared")
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
