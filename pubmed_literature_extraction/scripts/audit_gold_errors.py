#!/usr/bin/env python3
"""Gold Error Audit for Round-4.

Samples two error cohorts from a Round-3 arm output:

  * 20 false-positive ASSOCIATED_WITH where gold == NO_RELATION
  * 20 false-negative where gold == positive but Claim Gate rejected it

For each sample we emit: PMID, Entity A/B + types, evidence quote, gold label,
predicted label, Claim Gate claim_status (or relation_asserted/claim_role if
available), and a human-readable error-type slot.

Usage:
    python3 scripts/audit_gold_errors.py \
        --arm agent_results_pairwise_R3_A_baseline_round3_full50.json \
        --gate-arm agent_results_pairwise_R3_C_gate_only_round3_full50.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_agent.extraction_quality import normalize_surface  # noqa: E402
from scripts.run_pairwise_judge_experiments import load_jsonl  # noqa: E402

GOLD_PATH = ROOT / "gold_annotations" / "pubmed_200_gold_v2_strict.jsonl"


# ── triple keying (alias-aware, mirrors score_arm) ───────────────────────────

def _build_alias_index(gold_rows: list[dict]) -> dict:
    index: dict[str, set[tuple[str, str]]] = {}
    for g in gold_rows:
        for ent in g.get("entities", []):
            canonical = normalize_surface(ent.get("canonical", ent.get("mention", "")))
            typed = (canonical, str(ent.get("type", "")))
            for value in (ent.get("mention", ""), ent.get("canonical", "")):
                index.setdefault(normalize_surface(value), set()).add(typed)
    return index


def _canonical_endpoint(value: str, etype: str, aliases: dict) -> tuple[str, str]:
    surface = normalize_surface(value)
    matches = aliases.get(surface, set())
    if matches:
        canonical, ctype = next(iter(matches))
        return canonical, ctype
    return surface, normalize_surface(etype)


def _triple(relation: dict, aliases: dict) -> tuple:
    subj_c, subj_t = _canonical_endpoint(
        str(relation.get("subject", "")),
        str(relation.get("subject_type", "")),
        aliases,
    )
    obj_c, obj_t = _canonical_endpoint(
        str(relation.get("object", "")),
        str(relation.get("object_type", "")),
        aliases,
    )
    return (subj_c, subj_t,
            str(relation.get("predicate", "")).upper(),
            obj_c, obj_t)


# ── gold load ─────────────────────────────────────────────────────────────────

def load_gold() -> dict[str, dict]:
    rows = load_jsonl(GOLD_PATH)
    by_pmid: dict[str, dict] = {}
    for row in rows:
        by_pmid[str(row["pmid"])] = row
    return by_pmid


# ── prediction extraction from arm records ───────────────────────────────────

def _verified_relations(record: dict) -> list[dict]:
    ver = record.get("phases", {}).get("verification", {}) or {}
    rels = ver.get("relations", [])
    # Some arms store under "semantics" / nested dicts.
    if not rels:
        rels = ver.get("verified_relations", [])
    return rels


def _gate_table(record: dict) -> list[dict]:
    pj = record.get("phases", {}).get("pairwise_judge", {}) or {}
    return pj.get("gate_table", [])


# ── main audit ────────────────────────────────────────────────────────────────

def audit(arm_path: Path, gate_arm_path: Path | None, limit: int = 20):
    gold = load_gold()
    aliases = _build_alias_index(list(gold.values()))

    # Build predicted triples per PMID from the production relation stream.
    predicted: dict[str, list[tuple]] = defaultdict(list)
    predicted_relations: dict[str, list[dict]] = defaultdict(list)

    payload = json.loads(arm_path.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else payload

    for rec in records:
        pmid = str(rec["pmid"])
        rels = _verified_relations(rec)
        for rel in rels:
            predicted[pmid].append(_triple(rel, aliases))
            predicted_relations[pmid].append(rel)

    # Gold triples per PMID
    gold_triples: dict[str, set[tuple]] = defaultdict(set)
    gold_associated: set[tuple] = set()
    for pmid, g in gold.items():
        for rel in g.get("relations", []):
            t = _triple(rel, aliases)
            gold_triples[pmid].add(t)
            if str(rel.get("predicate", "")).upper() == "ASSOCIATED_WITH":
                gold_associated.add(t)

    fp_samples = []  # gold NO_RELATION but predict ASSOCIATED_WITH
    fn_samples = []  # gold positive but gate rejected (from gate_arm)

    for pmid, pred_triples in predicted.items():
        gset = gold_triples.get(pmid, set())
        for rel, t in zip(predicted_relations[pmid], pred_triples):
            if t[2] == "ASSOCIATED_WITH" and t not in gset:
                fp_samples.append((pmid, rel, t, gset))
            elif t not in gset and t[2] != "ASSOCIATED_WITH":
                pass  # other predicate FP

    # FN: gold-assoc_with not matched by any prediction
    for pmid, gset in gold_triples.items():
        pred_set = set(predicted.get(pmid, []))
        for t in gset:
            if t[2] == "ASSOCIATED_WITH" and t not in pred_set:
                fn_samples.append((pmid, t, None))

    # Augment FN with Claim Gate rejection info from gate_arm
    gate_rejections: dict[str, set[tuple]] = defaultdict(set)
    if gate_arm_path:
        gp = json.loads(gate_arm_path.read_text(encoding="utf-8"))
        grecords = gp.get("records", []) if isinstance(gp, dict) else gp
        for rec in grecords:
            pmid = str(rec["pmid"])
            for row in _gate_table(rec):
                # gate_table rows carry claim_status from Round-3 C arm
                if row.get("claim_status") in {"BACKGROUND", "PRIOR_WORK",
                                                 "COHORT_CONTEXT", "METHOD",
                                                 "NO_EXPLICIT_RELATION",
                                                 "SPECULATIVE", "PREDICTION_ONLY"}:
                    subj_c, subj_t = _canonical_endpoint(row["subject"], row.get("subject_type",""), aliases)
                    obj_c, obj_t = _canonical_endpoint(row["object"], row.get("object_type",""), aliases)
                    gate_rejections[pmid].add((subj_c, subj_t, "ASSOCIATED_WITH", obj_c, obj_t))

    print("=" * 72)
    print(f"Round-3 Gold Error Audit  (gold={GOLD_PATH.name})")
    print("=" * 72)

    print(f"\n## 1. False Positive: gold=NO_RELATION, prediction=ASSOCIATED_WITH")
    print(f"   (total AW FP: {len(fp_samples)})\n")
    for i, (pmid, rel, t, gset) in enumerate(fp_samples[:limit]):
        evidence = rel.get("evidence", "")[:180]
        print(f"  {i+1:2d}. PMID {pmid}")
        print(f"     A={rel.get('subject')} ({rel.get('subject_type')})  B={rel.get('object')} ({rel.get('object_type')})")
        print(f"     evidence: {evidence}")
        print(f"     gold:     NO_RELATION (triple not in gold set)")
        print(f"     model:    ASSOCIATED_WITH")
        print(f"     source:   {rel.get('classifier_source','legacy')}")
        print()

    print(f"\n## 2. False Negative: gold=positive, Claim Gate rejected")
    print(f"   (gold ASSOCIATED_WITH missing; total candidate-gold ASSOCIATED_WITH: {sum(len(g.get('relations',[])) for g in gold.values() if any(r.get('predicate')=='ASSOCIATED_WITH' for r in g.get('relations',[])))}, missing: {len(fn_samples)})\n")
    # cross-reference with gate rejections
    gate_rejected = 0
    for i, (pmid, t, _) in enumerate(fn_samples[:limit]):
        gr = gate_rejections.get(pmid, set())
        matched = any(t[0]==gt[0] and t[3]==gt[3] and t[2]==gt[2] for gt in gr)
        if matched:
            gate_rejected += 1
        print(f"  {i+1:2d}. PMID {pmid}")
        print(f"     A={t[0]} ({t[1]})  --ASSOCIATED_WITH-->  B={t[3]} ({t[4]})")
        print(f"     gold:     ASSOCIATED_WITH")
        print(f"     gate rejection: {'YES' if matched else 'NO (not in gate_table / DIRECT_FINDING but judge missed)'}")
        print()

    print("=" * 72)
    print(f"### Error-type tally")
    print(f"- FP AW (gold NO_RELATION, model predicts ASSOCIATED_WITH): {len(fp_samples)}")
    print(f"- FN  (gold positive, not predicted): {len(fn_samples)}")
    print(f"-   of which gate rejected the gold triple: ~{gate_rejected}")
    print(f"-   (note: 'is independently associated with' co-occurrence style is")
    print(f"  the dominant FP signature per Round-3 report)")
    print("=" * 72)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True,
                   help="agent_results JSON for the arm you want to audit (e.g. R3_A or R3_C)")
    p.add_argument("--gate-arm", default=None,
                   help="agent_results JSON for the gate arm (to inspect claim_status on FNs)")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()
    audit(Path(args.arm), Path(args.gate_arm) if args.gate_arm else None, args.limit)
