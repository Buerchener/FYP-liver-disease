# Router v10 four-layer live 100-paper evaluation

Run ID: `router_v10_four_layer_live_gold100_20260814`

## Execution scope

- 100 real remote LangExtract calls were executed with `[按次]gemini-2.5-flash` as the candidate generator.
- Four-layer routing had execution authority through `active_shadow`.
- `deepseek-v4-flash` was enabled as the conditional second model.
- Neo4j RAG was enabled read-only; Neo4j writes were disabled.
- Concurrency: 12 workers. Cache mode: process-local memory.
- Gold evaluation used the matching first 100 records from `pubmed_200_gold_v2_strict.jsonl`.

## Gold quality metrics (100 documents)

| Metric | Precision | Recall | F1 / accuracy |
|---|---:|---:|---:|
| Relation-endpoint/core-disease entities | 20.36% | 88.79% | 33.13% |
| Gene/Protein type | — | — | 84.21% (32/38) |
| Semantic relation triples | 12.07% | 22.22% | 15.64% |
| Strict Import-Ready triples | 21.05% | 20.00% | 20.51% |
| Evidence span, correct triple and IoU >= 0.5 | 7.76% (9/116) | — | — |
| Evidence copied contiguously from source | 100% (116/116) | — | — |
| Duplicate candidate rate | 9.03% (200/2,215) | — | — |

Candidate duplicate detail: entities 149/1,324 (11.25%); relations 51/891 (5.72%).

The entity metric is relation-centric, not exhaustive NER: the gold entities contain relation endpoints and core disease context. The evidence IoU metric only credits a span when the predicted triple also matches the gold triple.

## Profile and routing metrics (frozen 40-document blind set)

| Metric | Result | Target | Status |
|---|---:|---:|---|
| Primary study type accuracy | 85.0% | >=90% | Fail |
| Structured-results accuracy | 100.0% | — | — |
| High-extraction-complexity accuracy | 70.0% | — | — |
| Pre-extraction route accuracy | 87.5% | >=85% | Pass |
| Post-verification route accuracy | 77.5% | >=85% | Fail |
| Species-scope accuracy | 67.5% | — | — |
| Evidence-design accuracy | 72.5% | — | — |
| Causal-strength accuracy | 87.5% | — | — |
| Validation-level accuracy | 75.0% | — | — |
| Secondary-modality micro P/R/F1 | 80.82% / 67.82% / 73.75% | — | — |

All 40 blind profiles came from rules. Profile-LLM call rate was therefore 0%, below the <=30% budget, but it did not reach the primary-type target.

Across all 100 documents, the legacy aggregate route was FAST 6 / STANDARD 25 / DEEP 69. The executed four-layer route was FAST 30 / STANDARD 16 / DEEP 54. DEEP fell by 15 percentage points but remained well above the 15%–30% target.

## Tool routing and call economy

| Tool | Legacy planned | Four-layer executed/planned |
|---|---:|---:|
| Chunking | 75 | 12 |
| Context memory | 25 | 23 |
| Neo4j RAG | 4 | 3 |
| Second LLM | 34 | 1 |
| Causal reasoner | 9 | 2 |
| Conflict resolver | 80 | 2 |

Optional tool calls fell from 227 to 43 (81.1%) against the legacy plan. However, compared with the immediately preceding v9 active run, remote calls per article only fell from 1.16 to 1.13 (2.6%), so the >=15% remote-call reduction is not demonstrated against the latest active baseline.

Four-layer early stops: 43 all-relations-hard-blocked, 21 evidence-synthesis-without-current-experiment, and 10 prediction-only-without-experimental-validation.

## Latency and reliability

| Metric | Result |
|---|---:|
| Wall-clock time | 235.3 s for 100 documents |
| Throughput | 0.425 documents/s |
| Per-document latency mean / p50 / p95 / max | 25.09 / 21.49 / 59.21 / 73.35 s |
| Extraction-phase mean | 24.79 s |
| Whole-document failures | 0/100 |
| DeepSeek second-model calls | 1 triggered, 1 success, 0 fallback |
| DeepSeek invalid JSON | 0/1 attempts |
| Neo4j RAG | 3 OK, 0 offline, 0 error, 15 candidates |

The 100-document wall time corresponds to 117.65 seconds per 50 documents at the observed throughput. This is 18.1% faster than the saved v9 50-document run (143.6 s), but only 8.1% faster than the plan's historical 128-second reference, not the requested 15%.

The recorded extraction warning list contains 112 cold-cache misses and 15 generic-entity review warnings, not fatal errors. Console logging additionally showed one LangExtract schema warning and one JSON parse warning; both windows completed through the SDK's recovery path, and final whole-document failure rate remained zero.

## Cache and cost observability

- Extraction memory cache: 0/112 hits on this cold, unique-document run; 112 writes.
- Local preprocessing cache: 0/400 hits on this cold, unique-document run.
- Neo4j entity lookup cache: 1,152 hits / 917 misses, 55.68% hit rate.
- The second LLM used 2,306 input and 801 output tokens. At the current official DeepSeek V4 Flash cache-miss reference prices ($0.14/M input, $0.28/M output), that one call is approximately $0.000547.
- The main `[按次]gemini-2.5-flash` proxy did not return token or billing metadata. It made 112 extraction-window requests, so exact total tokens and invoice cost are unavailable and must not be reported as zero.

## Internal quality signals versus gold

The internal report gave structure 1.000, semantic 0.833, and evidence 0.921. It also reported 116/116 evidence spans as exact source substrings. These are conformance/grounding checks, not correctness against gold: gold evidence precision was only 9/116 because most predicted triples did not match a gold triple.

Verifier output contained 116 relations, of which 19 were Import-Ready. The most common remaining risk flags were weak evidence (97), missing trigger (84), novel subject (78), novel object (66), and trigger/direction mismatch (41). Sixty-three documents had no verified relation, and 91 had no Import-Ready relation.

Semantic false positives were dominated by `ASSOCIATED_WITH` (62/102). The model also missed 27 gold `ASSOCIATED_WITH` relations, showing that the issue is not simply one permissive threshold: endpoint selection, evidence-policy classification, and relation normalization are inconsistent.

## Same-50 comparison with v9

| Metric | v9 saved run | v10 same first 50 | Change |
|---|---:|---:|---:|
| Entity F1 | 37.64% | 35.52% | -2.12 pp |
| Semantic relation F1 | 39.56% | 27.45% | -12.11 pp |
| Strict Import-Ready F1 | 28.57% | 25.81% | -2.77 pp |
| Evidence IoU precision | 21.74% | 15.79% | -5.95 pp |
| Duplicate candidate rate | 8.40% | 7.09% | -1.31 pp |
| Mean article latency | 25.08 s | 25.42 s | +1.34% |

Because the candidate model is stochastic, this is an observed run-to-run comparison rather than a perfectly paired deterministic ablation. Nevertheless, it fails the requirement that relation F1 decline by no more than one percentage point.

## Acceptance decision

Do not enable Neo4j production writes and do not replace the current production router with v10 yet.

- Passed: pre-extraction blind route accuracy, profile-LLM budget, final JSON reliability, whole-document reliability, read-only safety, optional-tool pruning.
- Failed: primary study-type accuracy, post-verification route accuracy, DEEP-rate target, relation-F1 non-regression, Safe Write Precision, and the 15% reduction against the latest active remote-call baseline.
- Actual dangerous writes were zero because the run was dry-run. Counterfactually, only 4 of 19 Import-Ready predictions matched the strict gold set, so the write gate is not safe enough for activation.

The next iteration should keep layer 1 safety masks and layer 2 candidate pooling, but recalibrate layers 3 and 4. In particular, a high count of hard-blocked candidates must lower sufficiency/utility rather than cause a DEEP upgrade, and `ASSOCIATED_WITH` needs a stricter study-posture and endpoint-pair gate before it can become Import-Ready.
