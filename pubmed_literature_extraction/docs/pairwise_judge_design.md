# Pairwise-judge design (2026-08-18)

This document describes the pairwise LLM judge and the high-recall candidate
lattice that replaced the weak local pair classifier as the semantic authority
for pair candidates.  The deterministic verifier keeps owning facts and the
Safe Write boundary.

## Motivation (measured on the frozen 100-doc dev set)

1. The old clause-only candidate lattice covered only 69.8% of gold relation
   endpoint pairs; widening pairing windows to full sentences and adjacent
   sentences raised candidate-pair recall to 82.5% (+12.7pp).
2. The deterministic/TF-IDF pair backends were shadow-only (target-domain F1
   0.02–0.06); local heuristics cannot be the semantic authority.
3. Second-model adjudication had been starved by document-level routing: with
   identical fixed candidates, always-on DeepSeek adjudication reached
   semantic F1 44.7% vs 15.6% for the starved live run.

## New data flow

```text
LangExtract entities (+alias expansion)
              │
              ▼
high-recall pair lattice ── schema type mask ── clause + sentence
              │                                + adjacent windows
              │                                trigger-optional
              ▼
pair-centric LLM judge (DeepSeek, bounded batches)
   per pair: predicate | NO_RELATION + evidence_quote
             + direction (A_TO_B/B_TO_A/NONE)
             + ENTAILED / CONTRADICTED / NOT_ENOUGH_INFORMATION
              │
              ├─ ENTAILED + trigger-supported quote + conf ≥ min
              │    → judge_entailed (semantic accept candidate)
              ├─ uncertain (NEI / low conf / quote not grounded)
              │    → judge_uncertain → bounded DeepSeek adjudication
              │      (KEEP upgrades entailment; REJECT prunes)
              └─ NO_RELATION → recorded negative
              ▼
EvidenceSelector realignment (quote must be a contiguous source span)
              ▼
deterministic verifier (facts + Safe Write, unchanged write gates)
              ▼
Decision Engine / dry-run
```

## Invariants

- The judge may not add endpoints, predicates outside the pair's schema
  shortlist, or evidence text.  Every quote is realigned with
  `locate_contiguous`; a positive decision whose quote cannot be found in the
  source is downgraded to NOT_ENOUGH_INFORMATION.
- A positive ENTAILED decision additionally requires an explicit predicate
  trigger inside the quoted span (`PREDICATE_TRIGGERS`); trigger-free
  co-occurrence is downgraded to adjudication.
- Judge-ENTAILED relations may override the deterministic *semantic* review
  flags `trigger_missing / trigger_not_linking_endpoints /
  trigger_direction_mismatch / weak_evidence` (relation_contract.py
  `JUDGE_SEMANTIC_OVERRIDABLE_FLAGS`).  Factual rejects (schema, endpoints,
  negation, method/background/prediction-only, evidence continuity) always
  stay in force, and the write gate is untouched: those flags still block
  `import_ready`.
- A DeepSeek KEEP on a judge-uncertain candidate sets
  `adjudicator_entailed` and upgrades the entailment label before
  re-verification; the verifier re-checks everything.
- Hard blockers (endpoint missing, illegal schema, evidence not in source,
  explicit negation, method-only) never reach an LLM.
- Judge mode `active` is restricted to dry-run, exactly like the active pair
  classifier.

## Dynamic few-shot

`FewShotRetriever` indexes the development gold (never the blind-50).  For
each candidate pair it retrieves at most one positive example (same predicate,
same endpoint-type pair), one hard NO_RELATION example (co-occurring pair with
no gold relation), and one confusable example (same type pair, different
predicate), ranked by type pair, predicate, study type and character-ngram
similarity.  The current PMID and explicitly excluded PMIDs are never used.

## Round-3 modules (2026-08-18, training-free, Verifier-governed)

### Entity recovery (`entity_recovery.py`)

One bounded LLM call per article (role `recovery`) proposes missing endpoint
entities from coverage gaps.  Every proposal is re-validated
deterministically: contiguous span in the source, schema type whitelist,
generic-term blocklist, alias-aware dedup against the existing inventory, and
a mirror of `prepare_extraction`'s downstream filters (`entity_filter_reason`)
so recovery never spends slots on entities the verification chain would drop
(e.g. non-anatomical Tissue).  Fail-closed: any validation failure rejects the
proposal.  Accepted entities are appended before projection, so the pair
lattice and the judge see them; the extraction record itself is unchanged.

### Claim Gate (two-stage judge, `pairwise_judge.py`)

Stage A classifies every candidate pair into one of eight claim statuses
(DIRECT_FINDING / BACKGROUND / PRIOR_WORK / COHORT_CONTEXT / METHOD /
PREDICTION_ONLY / SPECULATIVE / NO_EXPLICIT_RELATION).  Only DIRECT_FINDING
reaches Stage B (predicate decision).  Everything else is a confident
NO_RELATION with reason code `claim_gate_<status>`, which also vetoes the
legacy extractor relation for the same pair.  Invalid gate output fails
closed; gate failure degrades to the single-stage judge.  Ablation switch:
`predicate_stage_enabled=False` keeps the gate but lets DIRECT_FINDING
survivors keep their local backend prediction (isolates the gate's own
effect).  `phases.pairwise_judge.gate_table` records per-candidate
claim_status / predicate / reason codes for offline P/R analysis.

### Error-pattern few-shot (`few_shot_retriever.py`)

`--few-shot-mode error_pattern` replaces similarity retrieval with
deterministic error-signature detection (8 categories: cohort context,
measurement without claim, background relation, prediction vs observation,
association vs causality, direction confusion, method-only relation, bare
co-occurrence).  The judge is shown one same-class hard NO_RELATION example
plus a clean positive example; hard negatives are labelled in the prompt with
the error category they demonstrate.

### Single-model judge cannot override the verifier (`verifier.py`)

A judge-ENTAILED proposal is a semantic *proposal*, never an override.  If it
conflicts with the verifier's semantic review flags the relation goes to
REVIEW with `judge_verifier_conflict`; only an independent second-model
endorsement (`adjudicator_entailed`) clears overridable semantic flags.
Hard blockers are never excused.  Safe Write is unchanged.

## Measured behaviour (frozen 100-doc dev set, 2026-08-18)

- Lattice recall: 69.8% (clause-only, cap 64) → 82.5% (sentence + adjacent
  windows, cap 128).  Remaining misses: endpoint not extracted (12.7%),
  document-level (1.6%), truncation (1.6%), pairing-form (1.6%).
- Semantic recall on the frozen set: baseline 0.317 → 0.38–0.41 with the
  judge; per-predicate PROGRESSES_TO F1 0.25 → 0.40.
- Precision stays the binding constraint: judge and adjudicator both
  over-accept ASSOCIATED_WITH; with this candidate snapshot every arm
  plateaus at semantic F1 0.12–0.15 (all-verified scoring) / 0.12–0.13
  (accepted-only), vs 0.40–0.45 reported for the same pipeline family on
  better candidate snapshots (v9 run, aliyun fixed-candidate benchmark).
- Candidate quality of the primary extractor dominates configuration choice:
  the same arms scored 0.24–0.27 F1 on the first 50 docs of the frozen
  v10-style snapshot vs 0.40–0.45 for the v9 snapshot.
- Bounded adjudication reduces zero-relation false-positive documents
  (35/71 → 21–27/71) even where F1 is unchanged.

## Round-3 measured behaviour (frozen v9cand first-50, 2026-08-18)

Single evaluator + v2-strict gold, all arms dry-run.  Baseline A: semantic
0.237/0.511/0.324 (all-verified), AW FP 61, NO_REL FP docs 10/31.  Findings:

- **Claim Gate works on FP**: gate-only (C) cuts AW FP 61 → 28 (−54%) and
  zero-doc AW FP 19 → 6; NO_REL FP docs 10/31 → 4/31.  All residual AW FPs
  are legacy-provenance (no candidate pair to veto).  Gate P vs gold is only
  0.057 — the allow-set is far wider than gold, but downstream filtering
  keeps most of it out.
- **Gate recall is the binding loss**: only 20/40 gold candidate pairs get
  DIRECT_FINDING (NO_EXPLICIT_RELATION 7-8, BACKGROUND 5-6, PRIOR_WORK 1,
  unjudged-budget 5-6).  Semantic recall collapses 0.511 → 0.311; no judge
  arm beats baseline F1 (C 0.283, D 0.273).
- **Entity recovery**: candidate-pair recall 88.9% → 91.1% (+2.2pp, one of
  three endpoint misses recovered).  In production it is inert without the
  pair lattice (arm B output is bit-identical to A), and in D–G recovered
  endpoints produce FP relations on zero-relation docs.  Recovery entities
  survive verification at 99% after mirroring `entity_filter_reason`.
- **Error-pattern few-shot hurts** (D→E: P 0.231→0.203, AW FP 38→51, judge
  FPs 11→24).  Same-class hard negatives did not suppress cohort-phrasing
  mis-entailment.
- **Conditional DeepSeek hurts** (E→F: F1 0.258→0.199, AW FP 51→68;
  16 triggers, KEEPs re-admit co-occurrence noise).
- **Qwen critic never triggered** (0 calls in 50 docs); G ≈ F.  Marginal
  effect unmeasured by design.
- Best import-ready F1 remains baseline A (0.286); judge arms C/E/F reach 0.

## Files

- `cognitive_agent/pairwise_judge.py` — judge config, validation, refine.
- `cognitive_agent/few_shot_retriever.py` — demonstration index/retrieval.
- `cognitive_agent/relation_pair_classifier.py` — high-recall lattice
  (`_pairing_windows`, trigger-optional `build_candidates`, cap 128).
- `cognitive_agent/evidence_units.py` — `parent_units`,
  `adjacent_sentence_windows`.
- `cognitive_agent/tool_router.py` — semantic uncertainty (incl.
  `judge_uncertain`) is reviewable again; hard blockers stay local.
- `cognitive_agent/collaborative_extractor.py` — review-candidate gate
  removed; `pair_candidate_id` carried in normalized decisions.
- `cognitive_agent/verifier.py` + `relation_contract.py` — judge semantic
  override (semantic-only, write gate unchanged).
- `cognitive_agent/agent.py` — `pairwise_judge` phase, adjudicator entailment
  upgrade, CLI/env flags (`--pairwise-judge-mode`, `--few-shot-mode`, ...).
- `scripts/diagnose_candidate_misses.py` — offline lattice miss diagnostic.
- `scripts/run_pairwise_judge_experiments.py` — A–F frozen ablation harness.
