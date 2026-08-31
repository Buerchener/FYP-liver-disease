import unittest

from cognitive_agent.aux_model_registry import AuxModelRegistry, AuxModelSpec, StructuredModelResult
from cognitive_agent.extraction_cache import LightweightExtractionCache
from cognitive_agent.remote_call_broker import ArticleRemoteCallBroker


class CacheAwareHarnessTests(unittest.TestCase):
    def test_aux_registry_retries_429_then_records_total_attempts(self):
        spec = AuxModelSpec(
            role="judge", provider="openai", model_id="deepseek-test",
            api_base="https://api.deepseek.com", api_key="test-key",
            max_retries=2, retry_base_delay_s=0.0, retry_max_delay_s=0.0,
        )
        registry = AuxModelRegistry([spec])
        calls = {"count": 0}

        class RateLimited(Exception):
            status_code = 429

        def fake_call(*_args, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RateLimited("429 too many requests")
            return {"ok": True}, registry._parse_usage({"prompt_tokens": 7, "completion_tokens": 2})

        registry._openai_call = fake_call
        result = registry.call_json("judge", system_prompt="stable", user_prompt="article")
        self.assertEqual(result.status, "OK")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(calls["count"], 2)
        self.assertEqual(registry.usage["judge"]["attempted"], 2)

    def test_aux_registry_stops_after_retryable_502_budget(self):
        spec = AuxModelSpec(
            role="judge", provider="openai", model_id="deepseek-test",
            api_base="https://api.deepseek.com", api_key="test-key",
            max_retries=2, retry_base_delay_s=0.0, retry_max_delay_s=0.0,
        )
        registry = AuxModelRegistry([spec])

        class BadGateway(Exception):
            status_code = 502

        registry._openai_call = lambda *_args, **_kwargs: (_ for _ in ()).throw(BadGateway("502 bad gateway"))
        result = registry.call_json("judge", system_prompt="stable", user_prompt="article")
        self.assertEqual(result.status, "FALLBACK")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(registry.usage["judge"]["attempted"], 3)

    def test_deepseek_usage_fields_are_normalized(self):
        usage = AuxModelRegistry._parse_usage({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_cache_hit_tokens": 70,
            "prompt_cache_miss_tokens": 30,
        })
        self.assertEqual(usage["provider_cache_read_tokens"], 70)
        self.assertEqual(usage["provider_cache_miss_tokens"], 30)
        self.assertTrue(usage["provider_prompt_hit"])
        self.assertAlmostEqual(usage["provider_cache_hit_rate"], 0.7)

    def test_qwen_usage_fields_are_normalized(self):
        usage = AuxModelRegistry._parse_usage({
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 40},
            "cache_creation_input_tokens": 60,
        })
        self.assertEqual(usage["provider_cache_read_tokens"], 40)
        self.assertEqual(usage["provider_cache_write_tokens"], 60)
        self.assertEqual(usage["uncached_input_tokens"], 60)

    def test_structured_decoder_accepts_fenced_json_and_rejects_empty_content(self):
        self.assertEqual(
            AuxModelRegistry._decode_json_object("```json\n{\"ok\": true}\n```"),
            {"ok": True},
        )
        self.assertEqual(
            AuxModelRegistry._decode_json_object("Result:\n{\"ok\": true}\nDone"),
            {"ok": True},
        )
        with self.assertRaisesRegex(ValueError, "empty structured response"):
            AuxModelRegistry._decode_json_object("")

    def test_empty_structured_response_is_retryable(self):
        self.assertTrue(
            AuxModelRegistry._is_retryable_error(
                ValueError("empty structured response"), "empty structured response",
            )
        )

    def test_local_warm_replay_marks_local_result_hit_and_skips_invoke(self):
        cache = LightweightExtractionCache(mode="memory")
        broker = ArticleRemoteCallBroker(cache=cache)
        calls = {"count": 0}

        def invoke(role, *, system_prompt, user_prompt, schema_hint):
            calls["count"] += 1
            return StructuredModelResult(
                role=role,
                model_id="deepseek-test",
                status="OK",
                payload={"ok": True},
                prompt_tokens=10,
                output_tokens=1,
                provider_cache_read_tokens=5,
                provider_prompt_hit=True,
            )

        kwargs = {
            "role": "judge",
            "system_prompt": "stable policy",
            "user_prompt": "dynamic article tail",
            "schema_hint": {"ok": True},
            "invoke": invoke,
        }
        cold = broker.intercept(**kwargs)
        warm = broker.intercept(**kwargs)
        self.assertEqual(calls["count"], 1)
        self.assertFalse(cold.local_result_hit)
        self.assertTrue(warm.local_result_hit)
        self.assertTrue(warm.provider_prompt_hit)


if __name__ == "__main__":
    unittest.main()
