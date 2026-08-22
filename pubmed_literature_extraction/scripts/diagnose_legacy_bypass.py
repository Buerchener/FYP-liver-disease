#!/usr/bin/env python3
"""Diagnose the legacy relation bypass.

Reproduces the merge in `cognitive_agent/agent.py:1252-1324`: the
frozen candidate snapshot stores **post-extraction artifacts** only
(`entities` + LangExtract `raw_extraction.relations`), not the post-judge
relations.  So this script measures what *would* reach the verifier under the
two regimes:

* **Legacy bypass ON** (Round-3 A): every LangExtract relation in the snapshot
  flows straight into `relation_core_relations` → verifier.
* **Legacy bypass OFF** (Round-4): only relations that carry a
  `pair_candidate_id` (lattice/judge output) survive; LangExtract hints are
  down-graded to evidence/entity hints.

Input : `benchmark_output/pairwise_judge_frozen_candidates.json`
Output: per-PMID counts + roll-up.
"""
from __future__ import annotations

import json
from pathlib import Path

SNAPSHOT = Path("benchmark_output/pairwise_judge_frozen_candidates.json")

# Fields present on every LangExtract raw relation (per
# cognitive_agent/relation_contract.py schema) but NOT added by the lattice.
LEGACY_CORE = {"subject", "subject_type", "predicate",
               "object", "object_type"}


def is_legacy(relation: dict) -> bool:
    """True if a relation dict originates from LangExtract raw extraction."""
    keys = set(relation.keys())
    if relation.get("classifier_source"):
        return False                      # lattice / judge produced it
    if relation.get("pair_candidate_id") or relation.get("candidate_id", "").startswith("p-"):
        return False                      # has a lattice pair id
    return bool(keys & LEGACY_CORE) and not (keys - LEGACY_CORE - {
        "direction", "negated", "uncertain", "evidence", "confidence",
        "disease_stage", "species", "grounded", "candidate_id",
        "predicate",  # allowed
    }) and "classifier_source" not in keys


def count_legacy(relations: list[dict]) -> int:
    return sum(1 for r in relations if is_legacy(r))


def count_lattice(relations: list[dict]) -> int:
    return sum(1 for r in relations
               if r.get("classifier_source") or
                  (str(r.get("candidate_id", "")).startswith("p-")) or
                  str(r.get("pair_candidate_id", "")).startswith("p-"))


def main() -> None:
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    candidates: dict = data["candidates"]

    total_rels = legacy = lattice = 0
    zero_doc = 0
    rows = []
    for pmid, rec in sorted(candidates.items()):
        rels = rec.get("relations", [])
        n = len(rels)
        l = count_legacy(rels)
        lat = count_lattice(rels)
        total_rels += n
        legacy += l
        lattice += lat
        if n == 0:
            zero_doc += 1
        rows.append((str(pmid), n, l, lat))

    print(f"=== legacy bypass diagnostic ===")
    print(f"  PMIDs:            {len(candidates)}")
    print(f"  total relations:  {total_rels}")
    print(f"  legacy-bypass ON (Round-3 A): {legacy} relations reach verifier directly")
    print(f"  lattice/judge output:         {lattice} relations")
    print(f"  zero-relation docs:           {zero_doc}")
    print()
    pct = legacy / total_rels * 100 if total_rels else 0
    print(f"  legacy share of final relation stream: "
          f"{legacy}/{total_rels} ({pct:.1f}%)")
    print()
    print("Per-PMID (top 12 by legacy count):")
    rows.sort(key=lambda x: x[2], reverse=True)
    for pmid, n, l, lat in rows[:12]:
        print(f"  PMID {pmid}: total={n:<3} legacy={l:<3} lattice={lat:<3}")
    print()
    if total_rels:
        print(">>> Conclusion: with legacy bypass ON, all "
              f"{legacy} LangExtract relations go straight to the verifier.")
        print(">>> With legacy bypass OFF (Round-4 arm A/B), "
              "these are down-graded to entity/evidence hints; "
              f"only {lattice} lattice-produced relations survive.")


if __name__ == "__main__":
    main()
