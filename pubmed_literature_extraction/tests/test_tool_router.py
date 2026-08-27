import json
import unittest
from concurrent.futures import ThreadPoolExecutor

from cognitive_agent.tool_router import ArticleToolRouter
from cognitive_agent.agent import AgentConfig, CognitiveAgent


class ArticleToolRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = ArticleToolRouter()

    def pre(self, title, abstract, **overrides):
        options = {
            "memory_available": True,
            "rag_enabled": True,
            "second_llm_enabled": True,
            "reviewer_enabled": False,
        }
        options.update(overrides)
        return self.router.plan_before_extraction(title, abstract, **options)

    def post(self, pre, extraction, verification, **overrides):
        options = {
            "memory_available": True,
            "rag_enabled": True,
            "second_llm_enabled": True,
            "second_llm_mode": "conditional",
            "reviewer_enabled": False,
        }
        options.update(overrides)
        return self.router.plan_after_verification(
            pre, extraction, verification, **options
        )

    def test_review_without_reported_findings_uses_fast_path(self):
        pre = self.pre(
            "A systematic review of biomarkers in liver disease",
            "This review summarizes published biomarker research.",
        )
        post = self.post(
            pre,
            {"entities": [], "relations": [], "warnings": [], "error": ""},
            {"entities": [], "relations": [], "summary": {}},
        )

        self.assertEqual(pre.route, "FAST")
        self.assertFalse(pre.should_call("context_memory"))
        self.assertFalse(post.should_call("neo4j_rag"))
        self.assertFalse(post.should_call("second_llm_refiner"))

    def test_narrative_review_cue_in_abstract_is_not_misclassified_as_mechanistic(self):
        profile = self.router.profile(
            "Gut-liver cellular crosstalk",
            "This narrative review summarizes studies in patients and cultured cells.",
        )
        self.assertEqual(profile.study_type, "review")
        self.assertGreater(profile.profile_confidence, 0)
        self.assertEqual(profile.evidence_design, "evidence_synthesis")

    def test_computational_discovery_with_human_validation_is_mixed_human_omics(self):
        profile = self.router.profile(
            "Transcriptomic discovery in liver disease",
            "Bioinformatics predictions in patients were experimentally validated by qPCR in human tissue.",
        )
        self.assertEqual(profile.study_type, "human_omics")
        self.assertEqual(profile.validation_level, "human_wet_lab_validated")

    def test_complex_mechanistic_article_preloads_memory(self):
        abstract = (
            "RESULTS: TP53 activates AKT and regulates MTOR, whereas NFE2L2 inhibits "
            "oxidative signaling and suppresses fibrosis; knockdown of TP53 promoted "
            "HCC, while overexpression of NFE2L2 inhibited HCC in cultured cells. "
            + "This long mechanistic sentence contains multiple linked experimental clauses " * 4
        )
        pre = self.pre("Mechanisms controlling HCC", abstract)

        self.assertEqual(pre.route, "DEEP")
        self.assertTrue(pre.should_call("context_memory"))
        self.assertTrue(pre.should_call("golden_example_selector"))
        self.assertEqual(pre.profile.study_type, "in_vitro")

    def test_long_complex_article_calls_chunker(self):
        abstract = (
            "RESULTS: TP53 activates AKT and regulates MTOR in cultured cells. "
            * 35
        )
        pre = self.pre("Long mechanistic HCC study", abstract)
        self.assertTrue(pre.should_call("article_chunker"))

    def test_clean_standard_result_avoids_both_expensive_post_tools(self):
        pre = self.pre(
            "TP53 and hepatocellular carcinoma",
            "TP53 was associated with hepatocellular carcinoma.",
        )
        extraction = {"entities": [{"mention": "TP53"}], "warnings": [], "error": ""}
        verification = {
            "entities": [{"mention": "TP53", "type": "Gene"}],
            "relations": [{
                "subject": "TP53", "predicate": "ASSOCIATED_WITH", "object": "HCC",
                "import_ready": True, "quality_flags": [],
            }],
            "summary": {},
        }
        post = self.post(pre, extraction, verification)

        self.assertFalse(post.should_call("neo4j_rag"))
        self.assertFalse(post.should_call("second_llm_refiner"))
        self.assertTrue(post.should_call("causal_reasoner"))
        self.assertTrue(post.should_call("conflict_resolver"))

    def test_endpoint_or_schema_problem_is_rejected_without_llm_rescue(self):
        pre = self.pre("A gene relation in HCC", "TP53 activates HCC.")
        extraction = {"entities": [{"mention": "TP53"}], "warnings": [], "error": ""}
        verification = {
            "entities": [{"mention": "TP53", "type": "Gene"}],
            "relations": [{
                "subject": "TP53", "predicate": "ACTIVATES", "object": "HCC",
                "import_ready": False,
                "quality_flags": ["unresolved_endpoint", "schema_mismatch"],
            }],
            "summary": {},
        }
        post = self.post(pre, extraction, verification)

        self.assertEqual(post.route, "STANDARD")
        self.assertFalse(post.should_call("neo4j_rag"))
        self.assertFalse(post.should_call("second_llm_refiner"))

    def test_grounded_high_risk_predicate_calls_edit_only_refiner(self):
        pre = self.pre(
            "A prognostic marker in HCC",
            "TP53 is a potential prognostic marker in HCC.",
        )
        extraction = {"entities": [{"mention": "TP53"}], "warnings": [], "error": ""}
        verification = {
            "entities": [{"mention": "TP53", "type": "Gene"}],
            "relations": [{
                "subject": "TP53", "subject_type": "Gene",
                "predicate": "PROGNOSTIC_IN", "object": "HCC", "object_type": "Disease",
                "evidence": "TP53 is a potential prognostic marker in HCC.",
                "schema_valid": True, "import_ready": True, "quality_flags": [],
            }],
            "summary": {},
        }
        post = self.post(pre, extraction, verification)
        self.assertTrue(post.should_call("second_llm_refiner"))

    def test_zero_entity_mechanistic_failure_uses_recovery_without_rag(self):
        pre = self.pre(
            "NFE2L2 mechanism in HCC",
            "We found that NFE2L2 inhibits HCC progression.",
        )
        post = self.post(
            pre,
            {"entities": [], "relations": [], "warnings": [], "error": "timeout"},
            {"entities": [], "relations": [], "summary": {}},
        )

        self.assertEqual(post.route, "RECOVERY")
        self.assertFalse(post.should_call("second_llm_refiner"))
        self.assertFalse(post.should_call("neo4j_rag"))

    def test_router_is_stateless_and_deterministic_under_concurrency(self):
        args = (
            "TP53 regulation in liver cancer",
            "RESULTS: TP53 regulates AKT signaling and promotes liver cancer in mice.",
        )

        def route(_):
            return json.dumps(self.pre(*args).to_dict(), sort_keys=True)

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(route, range(200)))

        self.assertEqual(len(set(results)), 1)

    def test_disabled_router_preserves_legacy_fixed_chain(self):
        router = ArticleToolRouter(enabled=False)
        pre = router.plan_before_extraction(
            "A title", "An abstract.", memory_available=False,
            rag_enabled=True, second_llm_enabled=True, reviewer_enabled=True,
        )
        post = router.plan_after_verification(
            pre,
            {"entities": [], "relations": [], "warnings": [], "error": ""},
            {"entities": [], "relations": [], "summary": {}},
            memory_available=False, rag_enabled=True, second_llm_enabled=True,
            reviewer_enabled=True,
        )

        self.assertEqual(pre.route, "LEGACY")
        self.assertTrue(pre.should_call("context_memory"))
        self.assertTrue(post.should_call("neo4j_rag"))
        self.assertTrue(post.should_call("second_llm_refiner"))
        self.assertTrue(post.should_call("debug_reviewer"))
        self.assertTrue(post.should_call("causal_reasoner"))
        self.assertTrue(post.should_call("conflict_resolver"))

    def test_shadow_complex_but_prediction_only_stays_fast(self):
        title = "Network pharmacology and molecular docking of a multi-target treatment"
        abstract = (
            "METHODS: Five databases and enrichment analyses identified many targets. "
            "RESULTS: Molecular docking predicted AKT1, EGFR, MAPK1 and MTOR interactions. "
            "CONCLUSION: These predictions require subsequent experimental validation."
        )
        legacy = self.pre(title, abstract)
        shadow = self.router.shadow_plan_before_extraction(
            title, abstract, legacy_plan=legacy, memory_available=True,
            rag_enabled=True, second_llm_enabled=True,
        )
        self.assertEqual(shadow.route, "FAST")
        self.assertEqual(shadow.profile.evidence_design, "computational_prediction")
        self.assertIn("prediction_only_without_experimental_validation", shadow.early_stop_reasons)

    def test_shadow_post_routes_only_linking_ambiguity_to_neo4j(self):
        legacy_pre = self.pre("Gene association", "TP53 was associated with HCC.")
        shadow_pre = self.router.shadow_plan_before_extraction(
            "Gene association", "TP53 was associated with HCC.", legacy_plan=legacy_pre,
            memory_available=True, rag_enabled=True, second_llm_enabled=True,
        )
        extraction = {"entities": [{"mention": "TP53"}], "warnings": [], "error": ""}
        verification = {"relations": [{
            "subject": "TP53", "predicate": "ASSOCIATED_WITH", "object": "HCC",
            "schema_valid": True, "import_ready": False,
            "quality_flags": ["ambiguous_endpoint"],
        }]}
        legacy_post = self.post(legacy_pre, extraction, verification)
        shadow_post = self.router.shadow_plan_after_verification(
            shadow_pre, legacy_post, extraction, verification,
            memory_available=True, rag_enabled=True, second_llm_enabled=True,
        )
        self.assertTrue(shadow_post.should_call("neo4j_rag"))
        self.assertFalse(shadow_post.should_call("second_llm_refiner"))
        self.assertFalse(shadow_post.should_call("relation_recovery"))
        self.assertNotEqual(shadow_post.route, "DEEP")

    def test_missing_endpoint_is_not_repairable_by_neo4j_or_llm(self):
        title, abstract = "Gene association", "TP53 was associated with HCC."
        legacy_pre = self.pre(title, abstract)
        shadow_pre = self.router.shadow_plan_before_extraction(
            title, abstract, legacy_plan=legacy_pre, memory_available=True,
            rag_enabled=True, second_llm_enabled=True,
        )
        extraction = {"entities": [{"mention": "TP53"}], "warnings": [], "error": ""}
        verification = {"relations": [{
            "subject": "TP53", "predicate": "PROGNOSTIC_IN", "object": "HCC",
            "schema_valid": True, "import_ready": False,
            "quality_flags": ["object_endpoint_missing", "weak_evidence"],
        }]}
        legacy_post = self.post(legacy_pre, extraction, verification)
        shadow_post = self.router.shadow_plan_after_verification(
            shadow_pre, legacy_post, extraction, verification,
            memory_available=True, rag_enabled=True, second_llm_enabled=True,
        )
        self.assertFalse(shadow_post.should_call("neo4j_rag"))
        self.assertFalse(shadow_post.should_call("second_llm_refiner"))

    def test_shadow_simple_runtime_failure_upgrades_to_recovery(self):
        title = "A simple association study"
        abstract = "TP53 was associated with HCC."
        legacy_pre = self.pre(title, abstract)
        shadow_pre = self.router.shadow_plan_before_extraction(
            title, abstract, legacy_plan=legacy_pre, memory_available=False,
            rag_enabled=False, second_llm_enabled=True,
        )
        extraction = {"entities": [], "relations": [], "warnings": [], "error": "timeout"}
        verification = {"relations": []}
        legacy_post = self.post(legacy_pre, extraction, verification)
        shadow_post = self.router.shadow_plan_after_verification(
            shadow_pre, legacy_post, extraction, verification,
            memory_available=False, rag_enabled=False, second_llm_enabled=True,
        )
        self.assertEqual(shadow_post.route, "RECOVERY")
        self.assertFalse(shadow_post.should_call("second_llm_refiner"))

    def test_shadow_semantic_uncertainty_reaches_second_llm(self):
        # Semantic uncertainty (weak trigger heuristics, hedging) is exactly
        # what the bounded second model is for.  Starving it here costs far
        # more precision than the calls it saves; deterministic hard blockers
        # alone stay local.
        title, abstract = "Prognostic study", "TP53 may be prognostic in HCC."
        legacy_pre = self.pre(title, abstract)
        shadow_pre = self.router.shadow_plan_before_extraction(
            title, abstract, legacy_plan=legacy_pre, memory_available=False,
            rag_enabled=False, second_llm_enabled=True,
        )
        extraction = {"entities": [{"mention": "TP53"}], "warnings": [], "error": ""}
        verification = {"relations": [{
            "subject": "TP53", "predicate": "PROGNOSTIC_IN", "object": "HCC",
            "schema_valid": True, "import_ready": False,
            "quality_flags": ["weak_evidence"],
        }]}
        legacy_post = self.post(legacy_pre, extraction, verification)
        shadow_post = self.router.shadow_plan_after_verification(
            shadow_pre, legacy_post, extraction, verification,
            memory_available=False, rag_enabled=False, second_llm_enabled=True,
        )
        self.assertTrue(shadow_post.should_call("second_llm_refiner"))
        self.assertEqual(shadow_post.route, "DEEP")
        self.assertTrue(
            shadow_post.layer_trace["layer_3_sufficiency_judgment"]["deep_audit_required"]
        )

    def test_four_layer_trace_exposes_masks_pool_sufficiency_and_utility(self):
        title = "Network pharmacology prediction in HCC"
        abstract = (
            "Molecular docking predicted that AKT1 interacts with TP53. "
            "These predictions require subsequent experimental validation."
        )
        legacy_pre = self.pre(title, abstract)
        shadow_pre = self.router.shadow_plan_before_extraction(
            title, abstract, legacy_plan=legacy_pre, memory_available=True,
            rag_enabled=True, second_llm_enabled=True,
        )
        self.assertEqual(shadow_pre.routing_version, "four-layer-shadow-v1")
        self.assertTrue(shadow_pre.decisions["causal_reasoner"].hard_masked)
        self.assertFalse(shadow_pre.decisions["causal_reasoner"].candidate_pool)
        self.assertEqual(set(shadow_pre.layer_trace), {
            "layer_1_safety_mask", "layer_2_candidate_pool",
            "layer_3_sufficiency_judgment", "layer_4_net_utility",
        })

        extraction = {"entities": [{"mention": "AKT1"}], "warnings": [], "error": ""}
        verification = {"entities": [], "relations": [{
            "subject": "AKT1", "predicate": "INTERACTS_WITH", "object": "TP53",
            "schema_valid": True, "import_ready": True, "quality_flags": [],
            "neo4j_status": "NOVEL",
        }]}
        legacy_post = self.post(legacy_pre, extraction, verification)
        shadow_post = self.router.shadow_plan_after_verification(
            shadow_pre, legacy_post, extraction, verification,
            memory_available=True, rag_enabled=True, second_llm_enabled=True,
        )
        self.assertFalse(shadow_post.should_call("causal_reasoner"))
        self.assertEqual(
            shadow_post.layer_trace["layer_1_safety_mask"]["masked_tools"]["causal_reasoner"],
            "prediction_only_cannot_support_causal_inference",
        )
        utility = shadow_post.layer_trace["layer_4_net_utility"]["utilities"]["causal_reasoner"]
        self.assertIn("net_utility", utility)
        self.assertEqual(utility["estimator"], "auditable_rule_prior_not_learned_probability")

    def test_conflict_resolver_only_enters_pool_for_existing_graph_relation(self):
        title, abstract = "TP53 in HCC", "TP53 was associated with HCC in patients."
        legacy_pre = self.pre(title, abstract)
        shadow_pre = self.router.shadow_plan_before_extraction(
            title, abstract, legacy_plan=legacy_pre, memory_available=True,
            rag_enabled=True, second_llm_enabled=True,
        )
        extraction = {"entities": [{"mention": "TP53"}], "warnings": [], "error": ""}
        base_relation = {
            "subject": "TP53", "predicate": "ASSOCIATED_WITH", "object": "HCC",
            "schema_valid": True, "import_ready": True, "quality_flags": [],
        }
        novel_verification = {"entities": [], "relations": [
            {**base_relation, "neo4j_status": "NOVEL"},
        ]}
        novel_post = self.router.shadow_plan_after_verification(
            shadow_pre, self.post(legacy_pre, extraction, novel_verification),
            extraction, novel_verification, memory_available=True,
            rag_enabled=True, second_llm_enabled=True,
        )
        self.assertFalse(novel_post.should_call("conflict_resolver"))

        known_verification = {"entities": [], "relations": [
            {**base_relation, "neo4j_status": "KNOWN"},
        ]}
        known_post = self.router.shadow_plan_after_verification(
            shadow_pre, self.post(legacy_pre, extraction, known_verification),
            extraction, known_verification, memory_available=True,
            rag_enabled=True, second_llm_enabled=True,
        )
        self.assertTrue(known_post.should_call("conflict_resolver"))

    def test_linking_candidate_score_distribution_can_trigger_rag(self):
        title, abstract = "Marker in HCC", "ABC was associated with HCC in patients."
        legacy_pre = self.pre(title, abstract)
        shadow_pre = self.router.shadow_plan_before_extraction(
            title, abstract, legacy_plan=legacy_pre, memory_available=True,
            rag_enabled=True, second_llm_enabled=False,
        )
        extraction = {"entities": [{"mention": "ABC"}], "warnings": [], "error": ""}
        verification = {"entities": [{
            "mention": "ABC", "neo4j_status": "AMBIGUOUS",
            "ambiguity_reason": "multiple close fuzzy candidates",
            "candidates": [{"score": 0.81}, {"score": 0.78}],
        }], "relations": []}
        legacy_post = self.post(legacy_pre, extraction, verification)
        shadow_post = self.router.shadow_plan_after_verification(
            shadow_pre, legacy_post, extraction, verification,
            memory_available=True, rag_enabled=True, second_llm_enabled=False,
        )
        self.assertTrue(shadow_post.should_call("neo4j_rag"))
        features = shadow_post.layer_trace["layer_3_sufficiency_judgment"]["linking_features"]
        self.assertEqual(features["close_top2_margin_count"], 1)

    def test_shadow_plan_cannot_mutate_production_plan(self):
        legacy = self.pre(
            "A review of cell models", "This review discusses in vitro cell lines."
        )
        before = json.dumps(legacy.to_dict(), sort_keys=True)
        shadow = self.router.shadow_plan_before_extraction(
            "A review of cell models", "This review discusses in vitro cell lines.",
            legacy_plan=legacy, memory_available=True, rag_enabled=True,
            second_llm_enabled=True,
        )
        after = json.dumps(legacy.to_dict(), sort_keys=True)
        self.assertEqual(before, after)
        self.assertEqual(shadow.plan_status, "shadow")
        self.assertTrue(self.router.compare_plans(legacy, shadow)["production_plan_unchanged"])

    def test_active_shadow_mode_refuses_write_authority(self):
        with self.assertRaisesRegex(ValueError, "restricted to dry-run"):
            CognitiveAgent(AgentConfig(
                api_key="unused", router_execution_mode="active_shadow",
                skip_neo4j_write=False,
            ))

    def test_active_shadow_mode_is_allowed_only_for_dry_run(self):
        # Validate the guard before expensive component construction.
        config = AgentConfig(api_key="unused", router_execution_mode="active_shadow", skip_neo4j_write=True)
        self.assertEqual(config.router_execution_mode, "active_shadow")


if __name__ == "__main__":
    unittest.main()
