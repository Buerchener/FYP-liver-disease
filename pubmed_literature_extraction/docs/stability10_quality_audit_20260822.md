# Live 10-Article Stability And Quality Audit

## Scope

This development sentinel exercised live primary extraction rather than frozen
candidates. It is a diagnostic gate, not a publishable or blind-test result.

- Run ID: `stability10_gemini25_20260822_094124`
- Primary model: `[S]gemini-2.5-flash-c`
- Primary configuration: `reasoning_effort=minimal`, four dynamic extraction
  demonstrations, 60-second request deadline, one timeout retry maximum.
- Agent configuration: `tiered-v2`, pair classifier active, conditional
  DeepSeek and Qwen enabled, two article workers, Neo4j writes disabled.
- Documents: 10 preselected development PMIDs covering zero-relation,
  background/review, prediction, cross-sentence, scope and positive-relation
  cases.

The raw run artifacts are intentionally local and ignored by Git:
`.cache/liverkg_runs/stability10_gemini25_20260822_094124/`.

## Stability Result

The run completed all 10 documents without a provider error, timeout, quota
failure, or Neo4j write.

| Metric | Result |
|---|---:|
| Completed documents | 10 / 10 |
| Provider failures | 0 |
| Mean end-to-end latency | 69.0 s/document |
| P50 end-to-end latency | 63.6 s/document |
| P90 end-to-end latency | 92.2 s/document |
| Slowest document | 124.3 s |
| DeepSeek calls | 4 / 4 successful |
| Qwen critic calls | 4 / 4 successful |
| Neo4j writes | 0 |

The primary extraction path is no longer the earlier Gemini 3 preview timeout
failure. A simple canary completed extraction in 7.927 seconds; complex
abstracts still require materially more model reasoning and downstream review.

## Quality Result

Quality did not pass the gate for a Gold-200 expansion. The values below use
the candidate semantic gold view and the project relation matching convention.

| Metric | Result |
|---|---:|
| Relation TP / FP / FN | 1 / 4 / 9 |
| Relation precision / recall / F1 | 0.200 / 0.100 / 0.133 |
| Entity endpoint precision / recall / F1 | 0.152 / 0.680 / 0.248 |
| Zero-relation specificity | 1.000 |
| Dangerous writes | 0 |
| Strict import-ready recall / F1 | 0.000 / 0.000 |

The strict precision convention is vacuously 1.0 because the run produced no
`IMPORT_READY` relation. It must not be interpreted as evidence of quality.

The candidate funnel shows the problem is not a lack of raw proposals:

| Stage | Count |
|---|---:|
| Raw relation attributes | 91 |
| Projected relation candidates after deduplication | 81 |
| Entity-pair candidates | 509 |
| Relations reaching verification | 6 |
| Semantic accepted | 5 |
| Semantic review | 1 |
| Semantic rejected by verifier | 0 |

## Diagnosis

The dominant failure is before the final Verifier. The Verifier saw only six
relations and accepted five of them; it did not semantically reject a large
number of otherwise viable candidates. The main loss occurs in candidate core
selection, deterministic endpoint/evidence filters and pair generation.

1. PMID `41650163` has six candidate-view gold relations. Four were present
   at raw or projected stages, but all were removed before final verification.
   Deterministic filters such as `endpoint_not_in_evidence`,
   `object_not_grounded` and unresolved/filtered endpoint checks removed them
   without a DeepSeek recovery route.
2. PMID `41581151` missed the cross-sentence `Endo4 liver endothelial cells
   -> Hepatic stellate cells` relation. The candidate generator produced an
   Endo4-to-fibrosis relation instead of the two supported cell endpoints.
3. PMID `41475279` retained four HBV/HCV-to-HCC associations but missed the
   gold NAFLD-to-HCC association. This indicates both pair selection failure
   and review/background claims being labelled as `CURRENT_FINDING`.
4. The raw entity pool is over-broad (14.4 entities/document on average),
   producing 509 pair candidates. The resulting candidate volume increases
   latency and makes deterministic pruning overly consequential.

## Required Changes Before Gold-200

1. Generate cross-sentence candidates using sentence-window and article-local
   abbreviation/coreference links, especially CellType-to-CellType and
   Disease-to-Disease pairs.
2. Change incomplete evidence boundary and endpoint-grounding failures from
   unconditional deletion to `HUMAN_REVIEW` or conditional DeepSeek routing.
   Keep the four non-overridable factual gates unchanged: impossible schema,
   missing endpoint, no source trace and scoped target negation.
3. Improve claim-role classification for reviews, background and prior work.
   These claims can remain semantic candidates but must not be labelled as a
   current finding or promoted toward import readiness.
4. Reduce pair explosion before the classifier through entity deduplication,
   canonical abbreviation linking and evidence-window pair generation. Do not
   solve it by tightening the final Verifier further.
5. Re-run this exact 10-article sentinel and require a material candidate-F1
   improvement while preserving zero-relation specificity and zero dangerous
   writes. Only then run Gold-200.

## Reproducibility Notes

The live run used the project-local `.env`, which is ignored by Git and set to
owner-readable permissions. The model configuration was deduplicated so the
last-value environment parsing behavior cannot silently restore the previous
Gemini 3 preview model.
