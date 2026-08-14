# Four-layer router v10: fixed 50-paper offline replay

- Input state: saved extraction and deterministic-verification state from `router_v9_final_gold50_20260814`.
- Execution: offline plan replay only; no API call and no Neo4j write.
- Production authority: unchanged (`legacy`).

## Route and call summary

| Metric | Legacy plan | Four-layer shadow |
|---|---:|---:|
| FAST | 3 | 17 |
| STANDARD | 13 | 22 |
| DEEP | 34 | 11 |
| DEEP rate | 68.0% | 22.0% |

- `article_chunker`: 39 → 5
- `causal_reasoner`: 4 → 2
- `conflict_resolver`: 37 → 1
- `context_memory`: 13 → 11
- `neo4j_rag`: 2 → 4
- `second_llm_refiner`: 13 → 1

## Counterfactual reduction

- Optional tool plans: 108 → 24 (77.8% reduction).
- Remote or multi-window plans: 54 → 10 (81.5% reduction).
- These are planned-call reductions, not measured latency/cost savings. Quality is unchanged in this replay because saved candidates are reused and the new plan has no production authority.

## Safety interpretation

- Hard masks remain outside the utility calculation and cannot be overridden.
- FAST/STANDARD/DEEP are audit labels; DEEP does not imply a fixed expensive tool bundle.
- A live dry-run A/B is still required before claiming relation-F1, latency, token, or cost improvement.
