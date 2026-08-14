# Agent v3 experiment status (English)

## Reproducibility status

The three implementation batches are complete in code. The deterministic test
suite currently passes with Neo4j integration tests skipped unless an isolated
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

