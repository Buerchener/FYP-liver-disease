import unittest

from cognitive_agent.extraction_quality import locate_contiguous, prepare_extraction
from cognitive_agent.verifier import KGVerifier


class OfflineKG:
    is_connected = False


def entity(mention, entity_type, **extra):
    return {"mention": mention, "type": entity_type, "attributes": {}, **extra}


def relation(
    subject="TP53",
    subject_type="Gene",
    predicate="ASSOCIATED_WITH",
    obj="HCC",
    object_type="Disease",
    evidence="TP53 is associated with HCC.",
    direction="positive",
    **extra,
):
    return {
        "subject": subject,
        "subject_type": subject_type,
        "predicate": predicate,
        "object": obj,
        "object_type": object_type,
        "evidence": evidence,
        "direction": direction,
        "negated": False,
        "uncertain": False,
        **extra,
    }


class ExtractionQualityTests(unittest.TestCase):
    def setUp(self):
        self.verifier = KGVerifier(OfflineKG())

    def verify(self, text, entities=None, relations=None):
        return self.verifier.verify(
            entities or [entity("TP53", "Gene"), entity("HCC", "Disease")],
            relations or [relation()],
            pmid="test",
            text=text,
        )

    def test_generic_entities_are_rejected_but_preserved_for_review(self):
        text = "Tumor immune microenvironment and cancer were discussed."
        entities = [
            entity("tumor immune microenvironment", "Tissue"),
            entity("cancer", "Disease"),
            entity("infections", "Disease"),
            entity("cells", "CellType"),
            entity("pathway", "Pathway"),
            entity("response", "Pathway"),
            entity("replication", "Pathway"),
        ]
        result = self.verifier.verify(entities, [], text=text)

        self.assertEqual(result.entities, [])
        self.assertEqual(len(result.filtered_entities), len(entities))
        for rejected in result.filtered_entities:
            self.assertEqual(rejected["filter_status"], "rejected")
            self.assertIn("filter_reason", rejected)
            self.assertIn("char_start", rejected)
            self.assertIn("was_relation_endpoint", rejected)

    def test_specific_mechanisms_are_retained_with_grounded_spans(self):
        text = (
            "VEGFA participates in angiogenesis. NFE2L2 participates in oxidative stress. "
            "Ferroptosis and TNF signaling pathway were measured. AREG-EGFR signaling was active."
        )
        entities = [
            entity("VEGFA", "Gene"), entity("angiogenesis", "Pathway"),
            entity("NFE2L2", "Gene"), entity("oxidative stress", "Pathway"),
            entity("Ferroptosis", "Pathway"),
            entity("TNF signaling pathway", "Pathway"),
            entity("AREG-EGFR signaling", "Pathway"),
        ]
        relations = [
            relation(
                "VEGFA", "Gene", "PARTICIPATES_IN", "angiogenesis", "Pathway",
                "VEGFA participates in angiogenesis.", "none",
            ),
            relation(
                "NFE2L2", "Gene", "PARTICIPATES_IN", "oxidative stress", "Pathway",
                "NFE2L2 participates in oxidative stress.", "none",
            ),
        ]
        result = self.verifier.verify(entities, relations, text=text)

        mentions = {item.mention for item in result.entities}
        self.assertIn("angiogenesis", mentions)
        self.assertIn("oxidative stress", mentions)
        self.assertIn("Ferroptosis", mentions)
        self.assertIn("TNF signaling pathway", mentions)
        self.assertIn("AREG-EGFR signaling", mentions)
        self.assertTrue(all(item.grounded for item in result.entities))
        self.assertTrue(all(item.import_ready for item in result.relations))

    def test_bare_conditional_process_without_relation_is_rejected(self):
        result = self.verifier.verify(
            [entity("inflammation", "Pathway")], [], text="Inflammation was mentioned in the background."
        )
        self.assertFalse(result.entities)
        self.assertEqual(result.filtered_entities[0]["filter_reason"], "bare_process_without_direct_evidence")

    def test_type_boundaries_reject_context_and_generic_cells(self):
        entities = [
            entity("tumor microenvironment", "Tissue"),
            entity("immune cells", "CellType"),
            entity("inflammation", "Disease"),
            entity("pathway", "Pathway"),
            entity("liver", "Tissue"),
            entity("Kupffer cells", "CellType"),
        ]
        text = "The tumor microenvironment contains immune cells. Inflammation occurs in liver Kupffer cells."
        result = self.verifier.verify(entities, [], text=text)
        mentions = {item.mention for item in result.entities}

        self.assertEqual(mentions, {"liver", "Kupffer cells"})
        reasons = {item["filter_reason"] for item in result.filtered_entities}
        self.assertIn("generic_or_context_term", reasons)
        self.assertIn("pathological_process_not_disease", reasons)

    def test_anatomical_tissues_and_specific_cell_subsets_are_retained(self):
        text = (
            "Aorta, pulmonary artery, colon, and peripheral blood were sampled. "
            "Macrophage subset 11 was also profiled."
        )
        entities = [
            entity("Aorta", "Tissue"),
            entity("pulmonary artery", "Tissue"),
            entity("colon", "Tissue"),
            entity("peripheral blood", "Tissue"),
            entity("Macrophage subset 11", "CellType"),
        ]
        result = self.verifier.verify(entities, [], text=text)

        self.assertEqual({item.mention for item in result.entities}, {
            "Aorta", "pulmonary artery", "colon", "peripheral blood",
            "Macrophage subset 11",
        })

    def test_entity_absent_from_source_is_rejected(self):
        result = self.verifier.verify(
            [entity("TP53", "Gene")], [], text="EGFR was measured."
        )
        self.assertFalse(result.entities)
        self.assertEqual(
            result.filtered_entities[0]["filter_reason"],
            "mention_not_in_source",
        )

    def test_gene_and_protein_with_same_surface_are_not_merged(self):
        text = "EGFR gene and EGFR protein were quantified."
        result = self.verifier.verify(
            [entity("EGFR", "Gene"), entity("EGFR", "Protein")], [], text=text
        )
        self.assertEqual({item.entity_type for item in result.entities}, {"Gene", "Protein"})
        self.assertEqual(len(result.entities), 2)

    def test_explicit_source_type_word_blocks_gene_protein_swap(self):
        text = (
            "Mitochondrial antiviral signalling protein (MAVS) was measured. "
            "The TP53 gene was measured."
        )
        result = self.verifier.verify(
            [
                entity("Mitochondrial antiviral signalling", "Gene"),
                entity("TP53", "Protein"),
            ],
            [],
            text=text,
        )
        self.assertFalse(result.entities)
        self.assertEqual(
            {item["filter_reason"] for item in result.filtered_entities},
            {"explicit_protein_as_gene", "explicit_gene_as_protein"},
        )

    def test_cross_type_surface_collision_does_not_erase_abbreviation_alias(self):
        text = (
            "OTU deubiquitinase 5 (OTUD5) protein was measured. "
            "The OTUD5 gene is associated with primary biliary cholangitis (PBC)."
        )
        entities = [
            entity("OTU deubiquitinase 5", "Gene"),
            entity("OTUD5", "Gene"),
            entity("OTU deubiquitinase 5", "Protein"),
            entity("primary biliary cholangitis", "Disease"),
            entity("PBC", "Disease"),
        ]
        rel = relation(
            "OTUD5", "Gene", "ASSOCIATED_WITH", "PBC", "Disease",
            "The OTUD5 gene is associated with primary biliary cholangitis (PBC).", "positive",
        )
        result = self.verifier.verify(entities, [rel], text=text)

        self.assertTrue(result.relations[0].subject_grounded_in_evidence)
        self.assertTrue(result.relations[0].object_grounded_in_evidence)
        self.assertTrue(result.relations[0].import_ready)

    def test_explicit_article_abbreviation_merges_and_remaps_endpoint(self):
        text = (
            "Hepatocellular carcinoma (HCC) was studied. "
            "TP53 is associated with HCC."
        )
        entities = [
            entity("Hepatocellular carcinoma", "Disease"),
            entity("HCC", "Disease"),
            entity("TP53", "Gene"),
        ]
        result = self.verifier.verify(entities, [relation()], text=text)

        disease_mentions = [item.mention for item in result.entities if item.entity_type == "Disease"]
        self.assertEqual(disease_mentions, ["Hepatocellular carcinoma"])
        self.assertEqual(result.mention_to_canonical["HCC"], "Hepatocellular carcinoma")
        self.assertEqual(result.relations[0].object, "Hepatocellular carcinoma")
        self.assertTrue(result.relations[0].endpoint_remapped)
        self.assertTrue(result.relations[0].import_ready)
        self.assertTrue(result.merged_entities)

    def test_undefined_abbreviation_is_not_automatically_merged(self):
        text = "Alpha beta complex and ABC were independently measured."
        prepared = prepare_extraction(
            [entity("Alpha beta complex", "Protein"), entity("ABC", "Protein")], [], text
        )
        self.assertEqual(len(prepared.entities), 2)

    def test_same_normalized_id_merges_only_within_same_type(self):
        left = entity("HCC", "Disease")
        left["attributes"]["normalized_id"] = "UMLS:C2239176"
        right = entity("hepatocellular carcinoma", "Disease")
        right["attributes"]["normalized_id"] = "UMLS:C2239176"
        text = "HCC, also termed hepatocellular carcinoma, was studied."
        result = self.verifier.verify([left, right], [], text=text)
        self.assertEqual(len(result.entities), 1)
        self.assertEqual(len(result.merged_entities), 1)

    def test_same_surface_merges_when_only_one_candidate_has_normalized_id(self):
        with_id = entity("CCND1", "Gene")
        with_id["attributes"]["normalized_id"] = "HGNC:1582"
        without_id = entity("CCND1", "Gene")
        result = self.verifier.verify(
            [with_id, without_id], [], text="CCND1 was measured twice."
        )
        self.assertEqual(len(result.entities), 1)
        self.assertEqual(result.entities[0].mention, "CCND1")
        self.assertEqual(len(result.merged_entities), 1)

    def test_empty_evidence_is_blocked(self):
        result = self.verify("TP53 is associated with HCC.", relations=[relation(evidence="")])
        self.assertFalse(result.relations[0].import_ready)
        self.assertIn("empty_evidence", result.relations[0].quality_flags)

    def test_non_contiguous_paraphrase_is_blocked(self):
        text = "TP53 is strongly associated with HCC."
        rel = relation(evidence="TP53 associated with HCC")
        result = self.verify(text, relations=[rel])
        self.assertFalse(result.relations[0].import_ready)
        self.assertIn("evidence_not_contiguous", result.relations[0].quality_flags)

    def test_evidence_without_both_endpoints_is_blocked(self):
        text = "TP53 expression increased. HCC samples were collected."
        rel = relation(evidence="TP53 expression increased.", direction="increase")
        result = self.verify(text, relations=[rel])
        self.assertFalse(result.relations[0].import_ready)
        self.assertIn("object_not_grounded", result.relations[0].quality_flags)

    def test_endpoint_absent_from_article_is_blocked(self):
        text = "TP53 expression increased in liver tissue."
        rel = relation(evidence="TP53 expression increased in HCC.", direction="increase")
        result = self.verify(text, relations=[rel])
        self.assertFalse(result.relations[0].import_ready)
        self.assertIn("evidence_not_contiguous", result.relations[0].quality_flags)

    def test_method_and_prediction_only_relations_are_blocked(self):
        cases = [
            (
                "Network pharmacology selected TP53 as associated with HCC.",
                "method_only",
            ),
            ("TP53 may be associated with HCC.", "prediction_only"),
        ]
        for text, expected_flag in cases:
            with self.subTest(expected_flag=expected_flag):
                result = self.verify(text, relations=[relation(evidence=text)])
                self.assertFalse(result.relations[0].import_ready)
                self.assertIn(expected_flag, result.relations[0].quality_flags)

    def test_negated_and_uncertain_relations_are_blocked(self):
        cases = [
            ("TP53 is not associated with HCC.", {"negated"}),
            ("TP53 could be associated with HCC.", {"uncertain", "prediction_only"}),
        ]
        for text, flags in cases:
            with self.subTest(text=text):
                result = self.verify(text, relations=[relation(evidence=text)])
                self.assertFalse(result.relations[0].import_ready)
                self.assertTrue(flags.issubset(set(result.relations[0].quality_flags)))

    def test_valid_contiguous_evidence_with_direction_passes(self):
        text = "TP53 expression increased in HCC."
        result = self.verify(
            text,
            relations=[relation(evidence=text, direction="increase")],
        )
        verified = result.relations[0]
        self.assertTrue(verified.import_ready)
        self.assertEqual(verified.evidence_level, 1)
        self.assertTrue(verified.direction_trigger_consistent)

    def test_direction_trigger_mismatch_is_blocked(self):
        text = "TP53 expression decreased in HCC."
        result = self.verify(
            text,
            relations=[relation(evidence=text, direction="increase")],
        )
        self.assertFalse(result.relations[0].import_ready)
        self.assertIn("trigger_direction_mismatch", result.relations[0].quality_flags)

    def test_trigger_must_link_the_proposed_endpoints(self):
        text = (
            "In HCC associated with HBV/HCV, the microbiome modulates "
            "immune surveillance and viral persistence."
        )
        entities = [entity("immune surveillance", "Pathway"), entity("HBV/HCV", "Disease")]
        rel = relation(
            "immune surveillance", "Pathway", "ASSOCIATED_WITH",
            "HBV/HCV", "Disease", text, "unknown",
        )
        result = self.verifier.verify(entities, [rel], text=text)
        self.assertFalse(result.relations[0].import_ready)
        self.assertIn("trigger_not_linking_endpoints", result.relations[0].quality_flags)

    def test_title_background_and_method_quotes_are_blocked(self):
        cases = [
            (
                "TITLE: TP53 is associated with HCC.\nABSTRACT: RESULTS: Other findings.",
                "TP53 is associated with HCC.",
                "title_only",
            ),
            (
                "TITLE: HCC study.\nABSTRACT: BACKGROUND: TP53 is associated with HCC. "
                "RESULTS: Other findings.",
                "TP53 is associated with HCC.",
                "background_only",
            ),
            (
                "TITLE: HCC study.\nABSTRACT: METHODS: TP53 is associated with HCC. "
                "RESULTS: Other findings.",
                "TP53 is associated with HCC.",
                "method_section_only",
            ),
        ]
        for text, evidence, expected_flag in cases:
            with self.subTest(expected_flag=expected_flag):
                result = self.verify(text, relations=[relation(evidence=evidence)])
                self.assertFalse(result.relations[0].import_ready)
                self.assertIn(expected_flag, result.relations[0].quality_flags)

    def test_result_quote_in_liver_article_remains_import_ready(self):
        text = (
            "TITLE: TP53 in liver disease.\nABSTRACT: RESULTS: "
            "TP53 is associated with HCC."
        )
        result = self.verify(text, relations=[relation(evidence="TP53 is associated with HCC.")])
        self.assertTrue(result.relations[0].import_ready)

    def test_agent_envelope_blocks_out_of_scope_and_non_human_titles(self):
        cases = [
            (
                "TITLE: TP53 in cardiac fibrosis.\nABSTRACT: RESULTS: "
                "TP53 is associated with HCC.",
                "article_out_of_scope",
            ),
            (
                "TITLE: TP53 in hepatic injury in mice.\nABSTRACT: RESULTS: "
                "TP53 is associated with HCC.",
                "non_human_article",
            ),
        ]
        for text, expected_flag in cases:
            with self.subTest(expected_flag=expected_flag):
                result = self.verify(text, relations=[relation(evidence="TP53 is associated with HCC.")])
                self.assertFalse(result.relations[0].import_ready)
                self.assertIn(expected_flag, result.relations[0].quality_flags)

    def test_screening_and_management_review_is_semantic_review_not_import_ready(self):
        text = (
            "TITLE: Screening and management of metabolic liver disease.\n"
            "ABSTRACT: MASLD is associated with HCC."
        )
        result = self.verify(text, relations=[relation(
            subject="MASLD", subject_type="Disease", evidence="MASLD is associated with HCC."
        )],
                             entities=[entity("MASLD", "Disease"), entity("HCC", "Disease")])
        self.assertEqual(result.relations[0].semantic_status, "REVIEW")
        self.assertFalse(result.relations[0].import_ready)
        self.assertIn("review_article", result.relations[0].quality_flags)

    def test_composite_and_generic_disease_entities_are_rejected(self):
        text = "HBV/HCV and liver diseases were discussed."
        result = self.verifier.verify(
            [entity("HBV/HCV", "Disease"), entity("liver diseases", "Disease")],
            [],
            text=text,
        )
        self.assertFalse(result.entities)
        self.assertEqual(
            {item["filter_reason"] for item in result.filtered_entities},
            {"composite_entity", "generic_disease_category"},
        )

    def test_blocking_subject_and_reducing_object_implies_positive_direction(self):
        text = "Blocking AREG-EGFR signaling attenuates pan-arterial fibrosis."
        entities = [
            entity("AREG-EGFR signaling", "Pathway"),
            entity("pan-arterial fibrosis", "Disease"),
        ]
        positive = relation(
            "AREG-EGFR signaling", "Pathway", "ASSOCIATED_WITH",
            "pan-arterial fibrosis", "Disease", text, "increase",
        )
        negative = {**positive, "direction": "decrease"}

        accepted = self.verifier.verify(entities, [positive], text=text).relations[0]
        rejected = self.verifier.verify(entities, [negative], text=text).relations[0]
        self.assertTrue(accepted.import_ready)
        self.assertFalse(rejected.import_ready)
        self.assertIn("trigger_direction_mismatch", rejected.quality_flags)

    def test_characterized_by_is_valid_association_trigger(self):
        text = "Primary biliary cholangitis is characterized by cholestasis."
        entities = [
            entity("Primary biliary cholangitis", "Disease"),
            entity("cholestasis", "Disease"),
        ]
        rel = relation(
            "Primary biliary cholangitis", "Disease", "ASSOCIATED_WITH",
            "cholestasis", "Disease", text, "none",
        )
        result = self.verifier.verify(entities, [rel], text=text)
        self.assertTrue(result.relations[0].import_ready)

    def test_strict_adjacent_sentence_evidence_can_be_level_two(self):
        text = "TP53 was measured in HCC patients. TP53 is associated with HCC."
        result = self.verify(text, relations=[relation(evidence=text)])
        self.assertTrue(result.relations[0].import_ready)
        self.assertEqual(result.relations[0].evidence_level, 2)
        self.assertIn("cross_sentence", result.relations[0].quality_flags)

    def test_filtered_entity_cannot_be_relation_endpoint(self):
        text = "TP53 is associated with cancer."
        entities = [entity("TP53", "Gene"), entity("cancer", "Disease")]
        rel = relation(obj="cancer", evidence=text)
        result = self.verifier.verify(entities, [rel], text=text)
        self.assertFalse(result.relations[0].import_ready)
        self.assertIn("filtered_endpoint", result.relations[0].quality_flags)
        self.assertTrue(result.unresolved_relations)
        self.assertTrue(result.filtered_entities[0]["was_relation_endpoint"])

    def test_quality_metrics_have_counts_and_denominators(self):
        text = "TP53 is associated with HCC."
        result = self.verify(text)
        summary = result.summary
        self.assertIn("structural_score", summary)
        self.assertIn("semantic_score", summary)
        self.assertIn("evidence_score", summary)
        self.assertIn("linking_stats", summary)
        for metric in summary["quality_metrics"].values():
            self.assertIn("count", metric)
            self.assertIn("denominator", metric)
            self.assertIn("value", metric)

    def test_whitespace_and_punctuation_normalization_still_requires_continuity(self):
        text = "TP53 is associated\nwith HCC."
        ok, start, end = locate_contiguous("TP53 is associated with HCC.", text)
        self.assertTrue(ok)
        self.assertEqual(text[start:end], "TP53 is associated\nwith HCC")


if __name__ == "__main__":
    unittest.main()
