import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cognitive_agent.collaborative_extractor import (
    CollaborationResult,
    CollaborativeConfig,
    CollaborativeExtractor,
)
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.verifier import KGVerifier


class OfflineKG:
    is_connected = False


def entity(mention, entity_type):
    return {"mention": mention, "type": entity_type, "attributes": {}}


def relation(predicate="ASSOCIATED_WITH", evidence="TP53 is associated with HCC."):
    return {
        "subject": "TP53", "subject_type": "Gene", "predicate": predicate,
        "object": "HCC", "object_type": "Disease", "evidence": evidence,
        "direction": "positive", "negated": False, "uncertain": False,
    }


def decision(candidate_id="r000", action="KEEP", **overrides):
    value = {
        "candidate_id": candidate_id, "action": action,
        "new_predicate": "", "new_direction": "", "evidence_unit_id": "",
        "swap_endpoints": False,
        "reason_code": "EXPLICIT_DIRECT_RELATION", "reason": "direct support",
        "confidence": 0.9,
    }
    value.update(overrides)
    return value


class CollaborativeExtractorTests(unittest.TestCase):
    def setUp(self):
        self.verifier = KGVerifier(OfflineKG())
        self.entities = [entity("TP53", "Gene"), entity("HCC", "Disease")]

    def verified(self, text, rel):
        return self.verifier.verify(self.entities, [rel], text=text).to_dict()

    @staticmethod
    def explicit_prognostic_text():
        return "TP53 expression was prognostic for overall survival in HCC patients."

    def test_disabled_never_calls_model(self):
        called = []
        text = "TP53 may be prognostic in HCC."
        rel = relation("PROGNOSTIC_IN", text)
        result = CollaborativeExtractor(generate=lambda prompt: called.append(prompt)).collaborate(
            text, {"entities": self.entities, "relations": [rel]}, self.verified(text, rel)
        )
        self.assertEqual(result.status, "DISABLED")
        self.assertEqual(called, [])

    def test_clean_direct_relation_does_not_call_second_model(self):
        called = []
        text = "TP53 is associated with HCC."
        rel = relation(evidence=text)
        extractor = CollaborativeExtractor(
            CollaborativeConfig(enabled=True, model_id="second"),
            generate=lambda prompt: called.append(prompt),
        )
        result = extractor.collaborate(
            text, {"entities": self.entities, "relations": [rel]}, self.verified(text, rel)
        )
        self.assertEqual(result.status, "NOT_TRIGGERED")
        self.assertEqual(called, [])

    def test_openai_call_disables_thinking_and_omits_output_cap(self):
        text = self.explicit_prognostic_text()
        rel = relation("PROGNOSTIC_IN", text)
        payload = {"review_decisions": [decision()], "review_reason": "reviewed"}
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload)), finish_reason="stop"
            )],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )
        with patch("openai.OpenAI") as client_class:
            client_class.return_value.chat.completions.create.return_value = response
            extractor = CollaborativeExtractor(CollaborativeConfig(
                enabled=True, provider="openai", api_key="test-key",
                api_base="https://api.deepseek.com", model_id="deepseek-v4-flash",
            ))
            result = extractor.collaborate(
                text, {"entities": self.entities, "relations": [rel]}, self.verified(text, rel)
            )
        request = client_class.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("max_tokens", request)
        self.assertEqual(result.total_tokens, 18)
        self.assertEqual(result.finish_reason, "stop")

    def test_aliyun_openai_compatibility_uses_qwen_thinking_switch(self):
        text = self.explicit_prognostic_text()
        rel = relation("PROGNOSTIC_IN", text)
        payload = {"review_decisions": [decision()], "review_reason": "reviewed"}
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload)), finish_reason="stop"
            )],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )
        with patch("openai.OpenAI") as client_class:
            client_class.return_value.chat.completions.create.return_value = response
            extractor = CollaborativeExtractor(CollaborativeConfig(
                enabled=True, provider="openai", api_key="test-key",
                api_base=(
                    "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
                ),
                model_id="qwen3.5-flash",
            ))
            extractor.collaborate(
                text, {"entities": self.entities, "relations": [rel]}, self.verified(text, rel)
            )
        request = client_class.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["extra_body"], {"enable_thinking": False})
        self.assertNotIn("max_tokens", request)

    def test_invalid_json_falls_back_without_open_generation(self):
        text = self.explicit_prognostic_text()
        rel = relation("PROGNOSTIC_IN", text)
        extractor = CollaborativeExtractor(
            CollaborativeConfig(enabled=True, model_id="second"), generate=lambda prompt: "bad-json"
        )
        verified = self.verified(text, rel)
        result = extractor.collaborate(text, {"entities": self.entities}, verified)
        merged = extractor.merge(self.entities, [rel], verified, result)
        self.assertEqual(result.status, "FALLBACK")
        self.assertEqual(merged.relation_additions, 0)

    def test_reject_decision_removes_relation_but_keeps_audit(self):
        text = self.explicit_prognostic_text()
        rel = relation("PROGNOSTIC_IN", text)
        verified = self.verified(text, rel)
        collaboration = CollaborationResult(
            status="OK", review_decisions=[{
                **decision(action="REJECT", reason_code="INSUFFICIENT_SUPPORT"),
                "raw_index": 0,
            }],
        )
        merged = CollaborativeExtractor().merge(self.entities, [rel], verified, collaboration)
        self.assertEqual(merged.relations, [])
        self.assertEqual(merged.relation_rejections, 1)
        self.assertEqual(len(merged.second_model_rejections), 1)

    def test_confident_keep_clears_only_pair_uncertainty_barrier(self):
        text = "TP53 is associated with HCC."
        rel = {
            **relation(evidence=text),
            "quality_flags": ["pair_classifier_candidate", "pair_low_confidence", "manual_review"],
        }
        verified = self.verified(text, rel)
        collaboration = CollaborationResult(
            status="OK", review_decisions=[{
                **decision(action="KEEP", confidence=0.9),
                "raw_index": 0,
            }],
        )
        merged = CollaborativeExtractor().merge(self.entities, [rel], verified, collaboration)
        flags = set(merged.relations[0]["quality_flags"])
        self.assertNotIn("pair_low_confidence", flags)
        self.assertNotIn("manual_review", flags)
        self.assertIn("second_llm_confirmed", flags)

    def test_predicate_edit_is_reverified_and_never_adds_relation(self):
        text = "TP53 is associated with HCC."
        rel = relation("PROGNOSTIC_IN", text)
        verified = self.verified(text, rel)
        collaboration = CollaborationResult(
            status="OK", review_decisions=[{
                **decision(action="CHANGE_PREDICATE", new_predicate="ASSOCIATED_WITH"),
                "raw_index": 0,
            }],
        )
        merged = CollaborativeExtractor().merge(self.entities, [rel], verified, collaboration)
        final = self.verifier.verify(merged.entities, merged.relations, text=text)
        self.assertEqual(merged.relation_edits, 1)
        self.assertEqual(merged.relation_additions, 0)
        self.assertEqual(final.relations[0].predicate, "ASSOCIATED_WITH")

    def test_hard_invalid_relation_is_deterministically_pruned(self):
        text = "TP53 was measured in HCC samples."
        rel = relation(evidence="not a source quote")
        verified = self.verified(text, rel)
        merged = CollaborativeExtractor().merge(
            self.entities, [rel], verified, CollaborationResult(status="NOT_TRIGGERED")
        )
        self.assertEqual(merged.relations, [])
        self.assertEqual(len(merged.deterministic_rejections), 1)

    def test_prompt_contains_only_candidates_units_and_non_evidence_rag(self):
        captured = []
        text = self.explicit_prognostic_text()
        rel = relation("PROGNOSTIC_IN", text)
        extractor = CollaborativeExtractor(
            CollaborativeConfig(enabled=True, model_id="second"),
            generate=lambda prompt: captured.append(prompt) or {
                "review_decisions": [decision()], "review_reason": "ok"
            },
        )
        extractor.collaborate(
            text, {"entities": self.entities}, self.verified(text, rel),
            rag_context={"entity_contexts": [{"mention": "TP53-memory"}]},
        )
        self.assertIn("TP53-memory", captured[0])
        self.assertIn("绝不能作为当前文章证据", captured[0])
        self.assertIn("candidate_id", captured[0])
        self.assertIn("evidence_unit", captured[0])

    def test_invalid_or_duplicate_decisions_are_ignored(self):
        text = self.explicit_prognostic_text()
        rel = relation("PROGNOSTIC_IN", text)
        payload = {
            "review_decisions": [
                decision(action="CHANGE_PREDICATE", new_predicate="NOT_ALLOWED"),
                decision(candidate_id="unknown", action="REJECT"),
            ],
            "review_reason": "invalid",
        }
        extractor = CollaborativeExtractor(
            CollaborativeConfig(enabled=True, model_id="second"), generate=lambda prompt: payload
        )
        result = extractor.collaborate(text, {"entities": self.entities}, self.verified(text, rel))
        self.assertEqual(result.review_decisions, [])
        self.assertTrue(result.parse_warnings)

    def test_primary_failure_is_quarantined_not_regenerated(self):
        extractor = CollaborativeExtractor(
            CollaborativeConfig(enabled=True, model_id="second"), generate=lambda prompt: self.fail()
        )
        result = extractor.collaborate(
            "No usable extraction.", {"entities": [], "relations": [], "error": "timeout"},
            {"entities": [], "relations": [], "summary": {}},
        )
        self.assertEqual(result.status, "RECOVERY_QUARANTINED")
        self.assertFalse(result.triggered)

    def test_evidence_reader_preserves_exact_clause_spans(self):
        text = (
            "TITLE: Study\nABSTRACT: RESULTS: TP53 increased in HCC; whereas AKT decreased "
            "in cirrhosis while the experiment remained controlled and carefully validated."
        )
        units = ArticleEvidenceReader().read(text)
        self.assertGreaterEqual(len(units), 2)
        for unit in units:
            self.assertEqual(text[unit.char_start:unit.char_end], unit.text)
        self.assertTrue(all(unit.section == "RESULTS" for unit in units))

    def test_finalizer_consolidates_alias_and_gene_protein_views(self):
        extractor = CollaborativeExtractor()
        raw = [
            {
                **relation(evidence="OTUD5 was associated with PBC."),
                "subject": "OTUD5", "subject_type": "Protein", "object": "PBC",
            },
            {
                **relation(evidence="OTUD5 was associated with PBC."),
                "subject": "OTU deubiquitinase 5", "subject_type": "Gene",
                "object": "Primary Biliary Cholangitis",
            },
        ]
        checked = {
            "relations": [
                {
                    **raw[0], "subject": "OTU deubiquitinase 5",
                    "object": "Primary Biliary Cholangitis", "import_ready": True,
                    "evidence_level": 1, "quality_flags": [],
                },
                {
                    **raw[1], "import_ready": True, "evidence_level": 1,
                    "quality_flags": [],
                },
            ],
            "entities": [
                {
                    "mention": "OTU deubiquitinase 5", "type": "Protein",
                    "neo4j_status": "NOVEL", "confidence": 0.8,
                },
                {
                    "mention": "OTU deubiquitinase 5", "type": "Gene",
                    "neo4j_status": "EXACT_MATCH", "neo4j_node_id": "NCBIGene:1",
                    "confidence": 0.7,
                },
            ],
            "review": {
                "mention_to_canonical": {
                    "OTUD5": "OTU deubiquitinase 5",
                    "PBC": "Primary Biliary Cholangitis",
                }
            },
        }
        final, audit = extractor.finalize_after_reverification(raw, checked)
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["subject_type"], "Gene")
        self.assertEqual(audit["duplicate_relations_removed"], 1)

    def test_finalizer_orients_undirected_edge_by_first_evidence_mention(self):
        extractor = CollaborativeExtractor()
        evidence = "OTUD5 directly interacted with MAVS in macrophages."
        raw = [{
            "subject": "MAVS", "subject_type": "Gene",
            "predicate": "INTERACTS_WITH",
            "object": "OTUD5", "object_type": "Gene",
            "evidence": evidence,
        }]
        checked = {
            "relations": [{
                **raw[0], "subject": "mitochondrial antiviral signalling",
                "object": "OTU deubiquitinase 5", "import_ready": True,
                "evidence_level": 1, "quality_flags": [],
            }],
            "entities": [],
            "review": {
                "mention_to_canonical": {
                    "MAVS": "mitochondrial antiviral signalling",
                    "OTUD5": "OTU deubiquitinase 5",
                }
            },
        }
        final, audit = extractor.finalize_after_reverification(raw, checked)
        self.assertEqual(final[0]["subject"], "OTUD5")
        self.assertEqual(final[0]["object"], "MAVS")
        self.assertEqual(audit["symmetric_orientation_changes"], 1)

    def test_finalizer_prefers_explicit_protein_definition_over_ambiguous_gene_view(self):
        extractor = CollaborativeExtractor()
        evidence = "OTUD5 interacted with MAVS."
        raw = [
            {
                "subject": "OTUD5", "subject_type": "Gene",
                "predicate": "INTERACTS_WITH", "object": "MAVS",
                "object_type": "Gene", "evidence": evidence,
            },
            {
                "subject": "OTUD5", "subject_type": "Gene",
                "predicate": "INTERACTS_WITH", "object": "MAVS",
                "object_type": "Protein", "evidence": evidence,
            },
        ]
        checked = {
            "relations": [
                {**item, "import_ready": True, "evidence_level": 1, "quality_flags": []}
                for item in raw
            ],
            "entities": [
                {"mention": "MAVS", "type": "Gene", "neo4j_status": "NOVEL"},
                {"mention": "MAVS", "type": "Protein", "neo4j_status": "NOVEL"},
            ],
            "review": {"mention_to_canonical": {"MAVS": "MAVS", "OTUD5": "OTUD5"}},
        }
        source = "Mitochondrial antiviral signalling protein (MAVS) was studied. " + evidence
        final, _ = extractor.finalize_after_reverification(raw, checked, source_text=source)
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["object_type"], "Protein")

    def test_broad_association_is_not_offered_as_one_model_recovery(self):
        text = "TP53 was higher in HCC."
        candidate = {
            "candidate_id": "p000", "raw_index": -1, "candidate_kind": "recovery",
            "subject": "TP53", "subject_type": "Gene", "object": "HCC",
            "object_type": "Disease", "evidence": text,
            "allowed_predicates": ["ASSOCIATED_WITH"],
        }
        called = []
        extractor = CollaborativeExtractor(
            CollaborativeConfig(enabled=True, model_id="second"),
            generate=lambda prompt: called.append(prompt) or {},
        )
        result = extractor.collaborate(
            text=text, extraction={"entities": self.entities, "relations": []},
            verification={"entities": self.entities, "relations": [], "summary": {}},
            recovery_candidates=[candidate],
        )
        self.assertEqual(result.status, "NOT_TRIGGERED")
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
