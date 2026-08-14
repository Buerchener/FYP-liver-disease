# Agent v3 experiment status (English)

## Reproducibility status

The three implementation batches are complete in code. All 179 deterministic
tests pass, with two Neo4j integration tests skipped unless an isolated
test database is explicitly configured. The 200-article development manifest
contains exactly 120 induction, 40 validation and 40 conformal-calibration
articles. The preregistered blind cohort contains exactly 10 articles in each
of five strata and its annotation template is still `UNLABELED`.

## Results that may be reported now

- Legacy compatibility remains the default.
- Rule bundles fail closed and cannot bypass schema, evidence or Safe Write.
- Evidence spans are contiguous and offset-realigned in deterministic tests.
- DeepSeek sees only uncertain bounded candidates; Qwen sees only conflicts.
- Mondrian calibration falls back globally below 20 samples.
- Blind PMID/hash leakage is machine-checked.
- The experiment runner requires all nine registered ablations and performs
  10,000 article-level bootstrap iterations by default.

## Five-article development smoke and replay

This is a systems smoke test, not a quality benchmark. It used the first five
development articles, active evidence-first pair classification, active
evidence entailment, DeepSeek/Qwen, shadow conformal routing, dry-run and no
active learned rule bundle. It did not touch the preregistered blind cohort.

| Measurement | Cold run | Fully warm replay |
| --- | ---: | ---: |
| Total latency | 82.5 s | 0.8 s |
| Article P95 | 26.41 s | 0.20 s |
| Primary extraction cache | 0% | 100% (8/8 windows; 9/9 shared lookups) |
| Primary remote requests | 8 | 0 |
| Evidence auxiliary cache | 0% | 100% (5/5 batches) |
| Evidence DeepSeek/Qwen requests | 5 / 1 | 0 / 0 |
| Separate relation adjudicator requests | 1 | 0 new request (cached) |
| Continuous source evidence | 100% | 100% |
| Zero-change call rate | 0% | 0% |
| Runtime errors / actual writes | 0 / 0 | 0 / 0 |

Cold and warm extraction entities, relation candidates and final verified
relations were identical. The cold run selected 157 bounded evidence spans and
produced 99 entailed, 54 NEI and 4 contradicted decisions. The seven cold
auxiliary calls equal 1.4 calls/article, within the internal ≤2 smoke target.
These counts do not establish semantic F1 because the smoke was not scored
against expert gold.

## Results not yet available

No final blind-test F1, precision, confidence interval or significance claim is
reported because expert annotation has not been completed. The blind set must
not be used for rule learning, prompt changes or threshold selection. Claiming
the requested publication targets before that step would contaminate the
evaluation.

The acceptance table must be populated only from frozen runs:

| Criterion | Target | Current status |
| --- | ---: | --- |
| Semantic relation F1 | ≥0.45 and ≥legacy+0.03 | Pending frozen validation/blind run |
| Strict import-ready precision | ≥0.65 | Pending expert labels |
| Evidence precision | ≥0.40 | Pending expert labels |
| Continuous source evidence | 1.00 | Enforced/tested; blind measurement pending |
| Dangerous writes | 0 | Enforced/tested; blind measurement pending |
| Mean auxiliary calls | ≤2/article | Pending routed experiment |
| Zero-change calls | ≤20% | Pending routed experiment |
| Routed P95 | ≤70% of always-call | Pending latency experiment |
| Warm primary cache | 100% | Must be rerun with frozen v3 bundle |

## Registered ablations

The final report must include legacy, Agent v2 without rules, DeepSeek
always-call, rule memory, no Qwen critic, no EvidenceSelector, no conformal
router, no Causal/Conflict and no cache. Causal/Conflict will be removed from
performance claims if its ablation shows no benefit.
