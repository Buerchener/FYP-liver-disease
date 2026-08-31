import json
import unittest
from pathlib import Path

from cognitive_agent.evidence_pack import EvidencePackBuilder
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.verifier import KGVerifier


class OfflineKG:
    is_connected = False


def relation(evidence: str, **updates):
    value = {
        "candidate_id": "c-test",
        "subject": "TP53", "subject_type": "Gene",
        "predicate": "ASSOCIATED_WITH",
        "object": "HCC", "object_type": "Disease",
        "evidence": evidence,
        "direction": "positive",
        "negated": False, "uncertain": False,
    }
    value.update(updates)
    return value


class EvidencePackSupportTests(unittest.TestCase):
    def setUp(self):
        self.builder = EvidencePackBuilder()
        self.reader = ArticleEvidenceReader()

    def test_owner_self_contained_ignores_attached_context_spans(self):
        text = (
            "TITLE: Study\nABSTRACT: RESULTS: TP53 and HCC were measured. "
            "TP53 was associated with HCC."
        )
        parents = self.reader.parent_units(text, self.reader.read(text))
        context, owner = parents[-2], parents[-1]
        item = relation(
            owner.text,
            evidence_candidates=[context.text],
            owner_sentence_ids=[owner.parent_sentence_id],
            context_sentence_ids=[context.parent_sentence_id],
        )
        pack = self.builder.build(item, text=text)
        self.assertEqual(pack.support_mode, "SELF_CONTAINED")
        self.assertEqual(pack.support_sentence_ids, [owner.parent_sentence_id])
        self.assertTrue(pack.context_span_ids)
        self.assertEqual(len(pack.minimal_support_span_ids), 1)

    def test_multi_span_and_coreference_are_distinct(self):
        multi_text = "TP53 was measured. Association with HCC was significant."
        multi = self.builder.build(relation(multi_text), text=multi_text)
        self.assertEqual(multi.support_mode, "MULTI_SPAN")
        self.assertGreaterEqual(len(multi.minimal_support_span_ids), 2)
        self.assertNotIn("COREFERENCE", multi.resolution_steps)

        coref_text = "TP53 was measured. These genes were associated with HCC."
        coref = self.builder.build(relation(coref_text), text=coref_text)
        self.assertEqual(coref.support_mode, "COREFERENCE")
        self.assertIn("COREFERENCE", coref.resolution_steps)

    def test_partial_trace_is_unresolved_not_untraceable(self):
        text = "TP53 was measured in the study. No disease endpoint was reported."
        pack = self.builder.build(
            relation("TP53 was measured in the study."), text=text,
        )
        self.assertTrue(pack.source_traceable)
        self.assertEqual(pack.support_mode, "UNRESOLVED")
        self.assertEqual(pack.minimal_support_span_ids, [])

    def test_pmid_41948344_ccnd1_relations_close_on_s018(self):
        root = Path(__file__).resolve().parents[1]
        source_path = root / "benchmark_output/gold20_strict_monotonic_v2_20260830_100322/input_gold20.jsonl"
        source = next(
            json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("pmid") == "41948344"
        )
        text = f"TITLE: {source['title']}\nABSTRACT: {source['abstract']}"
        evidence = (
            "scRNA-seq validated cell-type-specific expression patterns of CCND1 "
            "(epithelial cells, endothelial cells, hepatocytes, macrophages) and "
            "IL7R (T/NK cells)."
        )
        entities = [{"mention": "CCND1", "type": "Gene", "attributes": {}}]
        objects = (
            "endothelial cells", "epithelial cells", "hepatocytes", "macrophages",
        )
        entities.extend(
            {"mention": value, "type": "CellType", "attributes": {}}
            for value in objects
        )
        candidates = [relation(
            evidence,
            candidate_id=f"c-ccnd1-{index}",
            subject="CCND1", predicate="EXPRESSED_IN",
            object=value, object_type="CellType", direction="unknown",
        ) for index, value in enumerate(objects)]
        checked = KGVerifier(
            OfflineKG(), verification_policy="tiered-v2",
        ).verify(entities, candidates, pmid="41948344", text=text)
        self.assertEqual(len(checked.relations), 4)
        for item in checked.relations:
            with self.subTest(object=item.object):
                self.assertEqual(item.support_mode, "SELF_CONTAINED")
                self.assertEqual(item.support_sentence_ids, ["s018"])
                self.assertEqual(item.minimal_support_span_ids, ["ep-s018-2037-2191"])
                self.assertEqual(item.relation_card_match, "EXPLICIT")
                self.assertNotIn("cross_sentence", item.quality_flags)
                self.assertNotIn("trigger_not_linking_endpoints", item.quality_flags)
                if item.object == "epithelial cells":
                    self.assertEqual(item.semantic_status, "REVIEW")
                    self.assertIn("filtered_endpoint", item.semantic_reasons)
                else:
                    self.assertEqual(item.semantic_status, "ACCEPTED")


if __name__ == "__main__":
    unittest.main()
