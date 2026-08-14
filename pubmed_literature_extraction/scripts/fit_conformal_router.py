#!/usr/bin/env python3
"""Fit a frozen non-parametric conformal calibration from the 40-article split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cognitive_agent.conformal_router import CalibrationExample, ConformalCalibration, RiskFeatures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    allowed_pmids = {
        str(item["pmid"]) for item in manifest.get("records", [])
        if item.get("split") == "calibration"
    }
    rows = [json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
    unexpected = sorted({str(item.get("pmid", "")) for item in rows} - allowed_pmids)
    if unexpected:
        raise ValueError(f"non-calibration PMID detected: {unexpected[:5]}")
    examples = []
    feature_fields = RiskFeatures.__dataclass_fields__
    for row in rows:
        features = RiskFeatures(**{key: row[key] for key in feature_fields if key in row})
        examples.append(CalibrationExample.from_features(features, error=bool(row.get("error"))))
    calibration = ConformalCalibration(
        examples=examples,
        source_manifest_hash=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "examples": len(examples),
        "calibration_pmids": len({str(row.get('pmid')) for row in rows}),
    }))


if __name__ == "__main__":
    main()
