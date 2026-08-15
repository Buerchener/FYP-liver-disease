from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.evaluate_agent_v3_experiments import metrics
from scripts.run_agent_v3_ablation100 import (
    ARMS,
    atomic_json,
    arm_args,
    build_manifest,
    load_jsonl,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


class Ablation100RunnerTests(unittest.TestCase):
    def test_atomic_json_supports_concurrent_status_writers(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "status.json"
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [
                    pool.submit(atomic_json, path, {"writer": index})
                    for index in range(100)
                ]
                for future in futures:
                    future.result()
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn(payload["writer"], range(100))
            self.assertFalse(list(path.parent.glob(".*.tmp")))

    def test_manifest_is_deterministic_disjoint_and_cross_fitted(self):
        gold = load_jsonl(ROOT / "gold_annotations/pubmed_200_gold_v2_strict.jsonl")
        source = load_jsonl(ROOT / "extraction_output/pubmed_converted_500.jsonl")
        first = build_manifest(gold, source)
        second = build_manifest(gold, source)
        self.assertEqual(first["manifest_hash"], second["manifest_hash"])
        validate_manifest(
            first,
            ROOT / "gold_annotations/blind50/blind50_preregistered_seed20260814.json",
        )
        self.assertEqual(len(first["evaluation_pmids"]), 100)
        self.assertEqual([len(item["test_pmids"]) for item in first["folds"]], [20] * 5)
        for fold in first["folds"]:
            self.assertEqual(
                (len(fold["induction_pmids"]), len(fold["validation_pmids"]), len(fold["calibration_pmids"])),
                (120, 30, 30),
            )
            self.assertFalse(set(fold["test_pmids"]) & set(
                fold["induction_pmids"] + fold["validation_pmids"] + fold["calibration_pmids"]
            ))

    def test_arm_matrix_contains_real_component_switches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle.json"
            calibration = root / "calibration.json"
            snapshot = root / "snapshot.json"
            for arm in ARMS:
                common, variant = arm_args(
                    arm, snapshot_path=snapshot, bundle=bundle,
                    calibration=calibration, cache_path=root / f"{arm}.sqlite3",
                )
                joined = " ".join([*common, *variant])
                if arm != "v3_no_cache":
                    self.assertIn("--frozen-candidates", joined)
                else:
                    self.assertIn("--extraction-cache-mode off", joined)
                    self.assertNotIn("--frozen-candidates", joined)
            self.assertIn("--disable-qwen-critic", " ".join(arm_args(
                "v3_no_qwen_critic", snapshot_path=snapshot, bundle=bundle,
                calibration=calibration, cache_path=root / "x",
            )[1]))
            self.assertIn("--disable-evidence-selector", " ".join(arm_args(
                "v3_no_evidence_selector", snapshot_path=snapshot, bundle=bundle,
                calibration=calibration, cache_path=root / "x",
            )[1]))
            self.assertIn("--disable-causal-conflict", " ".join(arm_args(
                "v3_no_causal_conflict", snapshot_path=snapshot, bundle=bundle,
                calibration=calibration, cache_path=root / "x",
            )[1]))

    def test_comprehensive_metrics_include_denominators_and_latency_quantiles(self):
        row = {
            "pmid": "1", "entity_tp": 1, "entity_fp": 1, "entity_fn": 0,
            "tp": 1, "fp": 1, "fn": 1, "strict_tp": 1, "strict_fp": 0,
            "strict_fn": 0, "evidence_exact_tp": 1, "evidence_iou_tp": 1,
            "evidence_pred": 2, "evidence_gold": 2, "evidence_contiguous": 2,
            "endpoint_coverage": 1, "trigger_coverage": 1, "zero_relation_gold": 0,
            "zero_relation_correct": 0, "latency_s": 2.0, "aux_calls": 1,
            "remote_attempted": 1, "remote_successful": 1, "state_changes": 1,
            "confidence_correct": [[0.8, 1], [0.4, 0]],
            "predicate_counts": {"ASSOCIATED_WITH": {"tp": 1, "fp": 1, "fn": 1, "support": 2}},
        }
        result = metrics([row])
        self.assertIn("raw_denominators", result)
        self.assertIn("p99", result["latency"])
        self.assertIn("ASSOCIATED_WITH", result["predicate_metrics"])
        self.assertIn("brier", result["calibration"])
        self.assertEqual(result["remote_usage"]["attempted"], 1)


if __name__ == "__main__":
    unittest.main()
