import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cognitive_agent.biored_adapter import (
    BIORED_RELATION_TYPES, apply_native_label_adjudication,
    apply_train_signature_prior_guard,
    audit_false_negatives, build_native_label_review_items,
    build_native_pair_candidates, enrich_native_relation, guard_biored_test,
    parse_biored_pubtator, registry_from_training, stable_dev_partition,
    validate_native_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".cache/research/biored/dataset/BioRED"


class BioREDAdapterTests(unittest.TestCase):
    def test_dev_counts_and_fixed_partition(self):
        documents = parse_biored_pubtator(DATA / "Dev.PubTator")
        self.assertEqual(len(documents), 100)
        self.assertEqual(sum(len(item.relations) for item in documents), 1162)
        calibration, evaluation = stable_dev_partition(documents)
        self.assertEqual((len(calibration), len(evaluation)), (20, 80))
        again, _ = stable_dev_partition(reversed(documents))
        self.assertEqual(
            [item.pmid for item in calibration], [item.pmid for item in again]
        )

    def test_native_registry_has_all_official_labels(self):
        registry, manifest = registry_from_training(DATA / "Train.PubTator")
        self.assertEqual(set(registry.cards), set(BIORED_RELATION_TYPES))
        self.assertEqual(manifest["semantic_target"], "RELATION_TRUTH")
        self.assertEqual(len(manifest["manifest_sha256"]), 64)
        self.assertTrue(all(card.description for card in registry.cards.values()))
        self.assertTrue(all(card.boundary_note for card in registry.cards.values()))
        self.assertEqual(len(registry.prompt_cards()), 8)
        self.assertEqual(
            registry.dominant_label(
                "DiseaseOrPhenotypicFeature", "GeneOrGeneProduct",
            ),
            "Association",
        )
        self.assertGreater(
            registry.label_priors(
                "DiseaseOrPhenotypicFeature", "GeneOrGeneProduct",
            )["Association"],
            0.90,
        )

    def test_native_cards_cover_reported_mutations_and_treatment_adverse_events(self):
        registry, _ = registry_from_training(DATA / "Train.PubTator")
        profile = registry.to_schema_profile(source="train:test")
        mutation_support = profile.support_match(
            "Association",
            "GeneOrGeneProduct",
            "DiseaseOrPhenotypicFeature",
            (
                "Mutations in the SOX2 and CHX10 genes have been reported in "
                "patients with anophthalmia and/or microphthalmia."
            ),
            subject_aliases=["SOX2"],
            object_aliases=["anophthalmia"],
        )
        adverse_event_support = profile.support_match(
            "Positive_Correlation",
            "ChemicalEntity",
            "DiseaseOrPhenotypicFeature",
            (
                "Fewer subjects reported adverse events following treatment with "
                "desipramine alone; nausea was reported for patients treated with "
                "desipramine."
            ),
            subject_aliases=["desipramine"],
            object_aliases=["nausea"],
        )
        self.assertEqual(mutation_support["match"], "EXPLICIT")
        self.assertEqual(adverse_event_support["match"], "EXPLICIT")

    def test_invalid_id_label_and_unknown_novelty_are_audited(self):
        document = parse_biored_pubtator(DATA / "Dev.PubTator")[0]
        registry, _ = registry_from_training(DATA / "Train.PubTator")
        ids = list(document.concepts)
        values = [
            {"arg1_id": "invented", "arg2_id": ids[0], "relation_type": "Association"},
            {"arg1_id": ids[0], "arg2_id": ids[1], "relation_type": "LiverKGLabel"},
        ]
        clean, audit = validate_native_predictions(values, document, registry)
        self.assertEqual(clean, [])
        self.assertEqual(len(audit), 2)

    def test_official_test_is_guarded(self):
        with self.assertRaises(PermissionError):
            guard_biored_test(DATA / "Test.PubTator")
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "Test.PubTator"
            fake.write_text("x", encoding="utf-8")
            manifest = Path(directory) / "release.json"
            manifest.write_text('{"allow_official_test": false, "test_sha256": "x"}')
            with self.assertRaises(PermissionError):
                guard_biored_test(fake, manifest)

    def test_concept_pair_candidates_do_not_read_dev_relations(self):
        document = parse_biored_pubtator(DATA / "Dev.PubTator")[0]
        registry, _ = registry_from_training(DATA / "Train.PubTator")
        with_gold, _ = build_native_pair_candidates(document, registry)
        without_gold, _ = build_native_pair_candidates(
            replace(document, relations=[]), registry,
        )
        self.assertEqual(with_gold, without_gold)
        self.assertTrue(with_gold)
        self.assertTrue(all(item["candidate_id"].startswith("r-br-") for item in with_gold))
        self.assertTrue(all(item["pair_candidate_id"].startswith("p-br-") for item in with_gold))

    def test_native_evidence_pack_repairs_owner_and_multispan_support(self):
        document = next(
            item for item in parse_biored_pubtator(DATA / "Dev.PubTator")
            if item.pmid == "28518143"
        )
        registry, _ = registry_from_training(DATA / "Train.PubTator")
        relation = enrich_native_relation({
            "arg1_id": "D005947",
            "arg2_id": "D009202",
            "relation_type": "Positive_Correlation",
            "novelty": "Novel",
            "confidence": 0.95,
            "evidence_quote": (
                "HG induces myocardial energy metabolism disorder via decrease of "
                "CaSR expression, and activation of gp78-ubiquitin proteasome system."
            ),
            "claim_role": "CURRENT_FINDING",
        }, document, registry, lane="extracted_hint")
        self.assertEqual(relation["semantic_status"], "ACCEPTED")
        self.assertNotEqual(relation["support_mode"], "UNRESOLVED")
        self.assertEqual(relation["lineage_contract_version"], "lineage-v2")
        self.assertEqual(relation["candidate_lane"], "extracted_hint")
        self.assertTrue(relation["evidence_pack"]["minimal_support_span_ids"])

    def test_fn_audit_categories_are_mutually_exclusive(self):
        report_path = ROOT / "benchmark_output/biored_given_entity_dev_v5_20260831_141424/report.json"
        if not report_path.exists():
            self.skipTest("frozen BioRED Dev primary output is unavailable")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        documents = parse_biored_pubtator(DATA / "Dev.PubTator")
        ledger, summary = audit_false_negatives(
            report["records"], documents, relation_field="raw_relations",
        )
        self.assertEqual(len(ledger), 856)
        self.assertEqual(summary["category_counts"], {
            "GEMINI_COMPLETE_MISS": 531,
            "RELATION_LABEL_WRONG": 177,
            "ENDPOINTS_FOUND_PAIR_NOT_GENERATED": 148,
        })
        self.assertIn("Gold exact mismatch", summary["audit_scope"])

    def test_dual_label_edit_is_versioned_and_span_grounded(self):
        document = next(
            item for item in parse_biored_pubtator(DATA / "Dev.PubTator")
            if item.pmid == "28518143"
        )
        registry, _ = registry_from_training(DATA / "Train.PubTator")
        relation = enrich_native_relation({
            "arg1_id": "D005947", "arg2_id": "D009202",
            "relation_type": "Association", "confidence": 0.96,
            "evidence_quote": (
                "HG induces myocardial energy metabolism disorder via decrease of "
                "CaSR expression, and activation of gp78-ubiquitin proteasome system."
            ),
        }, document, registry, lane="extracted_hint")
        review = build_native_label_review_items([relation], document, registry)[0]
        span_ids = [item["span_id"] for item in review["minimal_support_spans"]]
        decision = {
            "candidate_id": relation["candidate_id"], "candidate_version": 1,
            "verdict": "EDIT", "recommended_relation_type": "Positive_Correlation",
            "relation_asserted": "YES", "confidence": 0.97,
            "supporting_span_ids": span_ids,
        }
        edited, audit = apply_native_label_adjudication(
            [relation], document, registry, [decision], [decision],
        )
        self.assertEqual(edited[0]["relation_type"], "Positive_Correlation")
        self.assertEqual(edited[0]["candidate_version"], 2)
        self.assertEqual(edited[0]["parent_version"], 1)
        self.assertEqual(audit[0]["decision"], "EDIT")
        self.assertIn("native_relation_label_edited", edited[0]["quality_flags"])

    def test_label_disagreement_keeps_original_in_review(self):
        document = next(
            item for item in parse_biored_pubtator(DATA / "Dev.PubTator")
            if item.pmid == "28518143"
        )
        registry, _ = registry_from_training(DATA / "Train.PubTator")
        relation = enrich_native_relation({
            "arg1_id": "D005947", "arg2_id": "D009202",
            "relation_type": "Association", "confidence": 0.96,
            "evidence_quote": (
                "HG induces myocardial energy metabolism disorder via decrease of "
                "CaSR expression, and activation of gp78-ubiquitin proteasome system."
            ),
        }, document, registry, lane="extracted_hint")
        span_ids = relation["evidence_pack"]["minimal_support_span_ids"]
        judge = {
            "candidate_id": relation["candidate_id"], "candidate_version": 1,
            "verdict": "EDIT", "recommended_relation_type": "Positive_Correlation",
            "relation_asserted": "YES", "confidence": 0.97,
            "supporting_span_ids": span_ids,
        }
        critic = {
            **judge, "recommended_relation_type": "Negative_Correlation",
        }
        kept, audit = apply_native_label_adjudication(
            [relation], document, registry, [judge], [critic],
        )
        self.assertEqual(kept[0]["relation_type"], "Association")
        self.assertEqual(kept[0]["candidate_version"], 1)
        self.assertEqual(kept[0]["semantic_status"], "REVIEW")
        self.assertIn("label_adjudicators_disagree", kept[0]["quality_flags"])
        self.assertEqual(audit[0]["decision"], "REVIEW")

    def test_train_signature_prior_versions_only_non_explicit_label(self):
        documents = parse_biored_pubtator(DATA / "Dev.PubTator")
        document = next(
            doc for doc in documents
            if any(value.entity_type == "DiseaseOrPhenotypicFeature" for value in doc.concepts.values())
            and any(value.entity_type == "GeneOrGeneProduct" for value in doc.concepts.values())
        )
        disease = next(
            key for key, value in document.concepts.items()
            if value.entity_type == "DiseaseOrPhenotypicFeature"
        )
        gene = next(
            key for key, value in document.concepts.items()
            if value.entity_type == "GeneOrGeneProduct"
        )
        registry, _ = registry_from_training(DATA / "Train.PubTator")
        candidate = {
            "candidate_id": "r-br-prior", "candidate_version": 1,
            "candidate_lane": "recovery", "arg1_id": disease, "arg2_id": gene,
            "relation_type": "Positive_Correlation", "relation_card_match": "NONE",
            "confidence": 0.9, "evidence_quote": "",
        }
        edited, audit = apply_train_signature_prior_guard(
            [candidate], document, registry,
        )
        self.assertEqual(edited[0]["relation_type"], "Association")
        self.assertEqual(edited[0]["candidate_version"], 2)
        self.assertEqual(audit[0]["reason_code"], "train_signature_dominant_label")
        explicit, audit = apply_train_signature_prior_guard(
            [{**candidate, "relation_card_match": "EXPLICIT"}], document, registry,
        )
        self.assertEqual(explicit[0]["relation_type"], "Positive_Correlation")
        self.assertEqual(audit, [])

    def test_type_only_review_fuses_pair_local_spans_with_full_document_context(self):
        document = next(
            item for item in parse_biored_pubtator(DATA / "Dev.PubTator")
            if item.pmid == "15069170"
        )
        registry, _ = registry_from_training(DATA / "Train.PubTator")
        relation = enrich_native_relation({
            "arg1_id": "D053579", "arg2_id": "c|INS||96",
            "relation_type": "Positive_Correlation", "confidence": 0.92,
            "candidate_lane": "recovery",
            "owner_sentence_ids": ["s013", "s014", "s015"],
            "evidence_quote": (
                "In cDNA derived from patients with IVS16+1G>A, an additional 96 bp "
                "insertion between exons 16 and 17 was observed. Six out of seven "
                "patients were compound heterozygotes, and the remaining one carried "
                "a single heterozygous mutation. CONCLUSIONS: We found four novel "
                "mutations in the NCCT gene in seven Japanese patients with GS."
            ),
        }, document, registry, lane="recovery")
        self.assertEqual(relation["endpoint_support_mode"], "MULTI_OWNER_ENDPOINT_CLOSED")
        self.assertEqual(relation["support_mode"], "UNRESOLVED")
        context = relation["evidence_pack"]["adjudication_spans"]
        self.assertTrue(any("autosomal recessive disorder" in item["text"] for item in context))
        review = build_native_label_review_items([relation], document, registry)[0]
        self.assertGreater(len(review["minimal_support_spans"]), 3)
        self.assertIn("signature_boundaries", review["signature_contract"])
        self.assertIn("Positive_Correlation", review["allowed_relation_types"])

    def test_train_dominant_type_only_label_survives_shared_model_polarity_bias(self):
        document = next(
            item for item in parse_biored_pubtator(DATA / "Dev.PubTator")
            if item.pmid == "17397547"
        )
        registry, _ = registry_from_training(DATA / "Train.PubTator")
        gene = next(
            key for key, value in document.concepts.items()
            if any(mention.text.casefold() == "ptprcap" for mention in value.mentions)
        )
        disease = next(
            key for key, value in document.concepts.items()
            if any(mention.text.casefold() == "inflammation" for mention in value.mentions)
        )
        relation = enrich_native_relation({
            "arg1_id": gene, "arg2_id": disease,
            "relation_type": "Association", "confidence": 0.95,
            "evidence_quote": (
                "Among PAR1-dependent transcripts, the following have been implicated "
                "in the inflammatory process: b2m, ccl7, cd200, cd63, cdbpd, cfl1, "
                "dusp1, fkbp1a, fth1, hspb1, marcksl1, mmp2, myo5a, nfkbia, pax1, "
                "plaur, ppia, ptpn1, ptprcap, s100a10, sim2, and tnfaip2."
            ),
        }, document, registry, lane="recovery")
        self.assertIn("native_relation_card_type_only", relation["quality_flags"])
        span_ids = relation["endpoint_support_span_ids"]
        decision = {
            "candidate_id": relation["candidate_id"], "candidate_version": 1,
            "verdict": "EDIT", "relation_asserted": "YES",
            "recommended_relation_type": "Positive_Correlation",
            "relation_confidence": 0.97, "label_confidence": 0.97,
            "confidence": 0.97, "supporting_span_ids": span_ids,
        }
        kept, audit = apply_native_label_adjudication(
            [relation], document, registry, [decision], [decision],
        )
        self.assertEqual(kept[0]["relation_type"], "Association")
        self.assertEqual(kept[0]["semantic_status"], "ACCEPTED")
        self.assertEqual(kept[0]["candidate_version"], 1)
        self.assertTrue(kept[0]["label_adjudication"]["train_dominant_keep"])
        self.assertEqual(audit[0]["decision"], "KEEP")

    def test_relation_and_label_confidence_are_independent_gates(self):
        document = next(
            item for item in parse_biored_pubtator(DATA / "Dev.PubTator")
            if item.pmid == "17397547"
        )
        registry, _ = registry_from_training(DATA / "Train.PubTator")
        gene = next(
            key for key, value in document.concepts.items()
            if any(mention.text.casefold() == "ptprcap" for mention in value.mentions)
        )
        disease = next(
            key for key, value in document.concepts.items()
            if any(mention.text.casefold() == "inflammation" for mention in value.mentions)
        )
        relation = enrich_native_relation({
            "arg1_id": gene, "arg2_id": disease, "relation_type": "Association",
            "confidence": 0.95,
            "evidence_quote": "ptprcap has been implicated in the inflammatory process.",
        }, document, registry, lane="recovery")
        decision = {
            "candidate_id": relation["candidate_id"], "candidate_version": 1,
            "verdict": "KEEP", "relation_asserted": "YES",
            "recommended_relation_type": "Association",
            "relation_confidence": 0.50, "label_confidence": 0.99,
            "confidence": 0.99,
            "supporting_span_ids": relation["endpoint_support_span_ids"],
        }
        kept, audit = apply_native_label_adjudication(
            [relation], document, registry, [decision], [], require_critic=False,
        )
        self.assertEqual(kept[0]["semantic_status"], "REVIEW")
        self.assertIn("adjudicator_relation_low_confidence", audit[0]["reason_codes"])

    def test_dual_adjudication_can_correct_signed_label_to_association(self):
        document = next(
            item for item in parse_biored_pubtator(DATA / "Dev.PubTator")
            if item.pmid == "15086325"
        )
        registry, _ = registry_from_training(DATA / "Train.PubTator")
        disease = next(
            key for key, value in document.concepts.items()
            if any("factor V deficiency" in mention.text for mention in value.mentions)
        )
        variant = next(
            key for key, value in document.concepts.items()
            if any(mention.text == "IVS8 -2A>G" for mention in value.mentions)
        )
        relation = enrich_native_relation({
            "arg1_id": disease, "arg2_id": variant,
            "relation_type": "Positive_Correlation", "confidence": 0.95,
            "evidence_quote": (
                "Three F5 gene mutations, IVS8 -2A>G, 2238-9del AG and G6410T, "
                "have been identified in two Chinese pedigree with congenital FV deficiency."
            ),
        }, document, registry, lane="recovery")
        decision = {
            "candidate_id": relation["candidate_id"], "candidate_version": 1,
            "verdict": "EDIT", "relation_asserted": "YES",
            "recommended_relation_type": "Association",
            "relation_confidence": 0.97, "label_confidence": 0.97,
            "confidence": 0.97,
            "supporting_span_ids": relation["endpoint_support_span_ids"],
        }
        edited, audit = apply_native_label_adjudication(
            [relation], document, registry, [decision], [decision],
        )
        self.assertEqual(edited[0]["relation_type"], "Association")
        self.assertEqual(edited[0]["candidate_version"], 2)
        self.assertEqual(audit[0]["decision"], "EDIT")


if __name__ == "__main__":
    unittest.main()
