import json
import unittest
from unittest.mock import patch

from cognitive_agent.hybrid_article_profiler import (
    _validated_llm, parse_features, profile_article, rule_profile,
)


def test_inline_structured_headings_are_detected():
    p = rule_profile("A cohort study", "METHODS: We enrolled patients. RESULTS: Outcomes improved. CONCLUSION: Done.")
    assert p.has_structured_results is True


def test_explicit_abstract_review_overrides_topic_cues():
    p = rule_profile("Spheroids in cancer", "This review highlights in vitro cell lines and future research.")
    assert p.primary_study_type == "review"
    assert p.recommended_route == "FAST"
    assert p.llm_trigger_reasons == []


def test_llm_not_called_for_unambiguous_profile():
    p = profile_article("Retrospective cohort", "METHODS: 200 patients were enrolled. RESULTS: Associations were measured.")
    assert p.source == "rules"
    assert p.llm_status == "not_called"


def test_feature_parser_separates_prediction_and_validation():
    f = parse_features("Machine learning omics", "GEO and WGCNA were used. Findings were validated by Western blot in human liver tissues.")
    assert f.computational_cues
    assert f.validation_cues
    assert f.human_cues


class StrictProfilerFailureTests(unittest.TestCase):
    def test_llm_cannot_forge_evidence_quote(self):
        with self.assertRaises(ValueError):
            _validated_llm({
                "primary_study_type": "clinical",
                "secondary_modalities": ["human_observational"],
                "species_scope": "human",
                "evidence_design": "human_observational",
                "causal_strength": "association_only",
                "validation_level": "direct_human_observation",
                "high_extraction_complexity": False,
                "confidence": 0.9,
                "rationale": "test",
                "evidence_quotes": ["This quote was invented."],
            }, "A title", "Patients were observed.")

    @patch("cognitive_agent.hybrid_article_profiler.urllib.request.urlopen", side_effect=TimeoutError("slow"))
    def test_llm_timeout_safely_falls_back_to_rules(self, _urlopen):
        p = profile_article(
            "Machine learning with cell validation",
            "GEO machine learning predictions were tested in cultured cell lines.",
            api_key="test-only", force_llm=True,
        )
        self.assertEqual(p.source, "rules")
        self.assertTrue(p.llm_status.startswith("fallback:TimeoutError"))

    @patch("cognitive_agent.hybrid_article_profiler.urllib.request.urlopen")
    def test_invalid_json_retries_once_then_falls_back(self, urlopen):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self):
                return json.dumps({"choices": [{"message": {"content": "not json"}}]}).encode()
        urlopen.return_value = Response()
        p = profile_article("Mixed study", "Patients and mice were studied.", api_key="test-only", force_llm=True)
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(p.source, "rules")
        self.assertTrue(p.llm_status.startswith("fallback:JSONDecodeError"))
