import unittest
from concurrent.futures import ThreadPoolExecutor

from cognitive_agent.article_preprocessing import ContentAddressedCache
from cognitive_agent.tool_router import ArticleToolRouter


class RouterFaultTests(unittest.TestCase):
    def test_corrupted_cache_entry_is_recomputed(self):
        cache = ContentAddressedCache(version="test")
        key = cache.content_hash("title", "abstract")
        cache._values[(key, "profile")] = None
        value, hit = cache.get_or_compute(key, "profile", lambda: {"safe": True})
        self.assertFalse(hit)
        self.assertEqual(value, {"safe": True})

    def test_router_concurrency_has_no_cross_article_state(self):
        router = ArticleToolRouter()
        cases = [
            ("A review", "This review summarizes published studies."),
            ("Mouse study", "TP53 inhibited fibrosis in mice."),
            ("Computational model", "A mathematical model predicted HBV transmission."),
        ]
        def route(case):
            legacy = router.plan_before_extraction(
                *case, memory_available=False, rag_enabled=False,
                second_llm_enabled=False, reviewer_enabled=False,
            )
            return router.shadow_plan_before_extraction(
                *case, legacy_plan=legacy, memory_available=False,
                rag_enabled=False, second_llm_enabled=False,
            ).route
        with ThreadPoolExecutor(max_workers=12) as pool:
            values = list(pool.map(route, cases * 40))
        self.assertEqual(values[0::3], ["FAST"] * 40)
        self.assertEqual(values[2::3], ["FAST"] * 40)
