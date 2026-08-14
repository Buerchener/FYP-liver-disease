#!/usr/bin/env python3
"""Fail closed if blind documents leak into rules, calibration, or examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def find_leaks(blind_manifest: dict, artifacts: list[tuple[Path, object]]) -> list[dict]:
    blind_pmids = {str(item["pmid"]) for item in blind_manifest.get("records", [])}
    blind_hashes = {str(item["abstract_sha256"]) for item in blind_manifest.get("records", [])}
    leaks = []
    for path, payload in artifacts:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        for pmid in sorted(blind_pmids):
            if pmid and pmid in serialized:
                leaks.append({"artifact": str(path), "kind": "blind_pmid", "value": pmid})
        for digest in sorted(blind_hashes):
            if digest and digest in serialized:
                leaks.append({"artifact": str(path), "kind": "blind_abstract_hash", "value": digest})
    return leaks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    args = parser.parse_args()
    blind = json_or_jsonl(args.blind_manifest)
    artifacts = [(path, json_or_jsonl(path)) for path in args.artifact]
    leaks = find_leaks(blind, artifacts)
    print(json.dumps({"status": "FAILED" if leaks else "OK", "leaks": leaks}, ensure_ascii=False, indent=2))
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
