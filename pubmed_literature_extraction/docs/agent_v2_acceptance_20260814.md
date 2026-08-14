# Central Agent v2 calibration — 2026-08-14

## Scope

- Frozen set: first 50 PMIDs in `extraction_output/pubmed_converted_500.jsonl`,
  matching `gold_annotations/pubmed_50_gold_v1.jsonl`.
- Primary model: configured Gemini-compatible endpoint/model.
- Concurrency: outer worker 1, extraction inner worker 1.
- Modes exercised: legacy, `agent-v2-shadow`, and active `agent-v2` dry-run.
- Neo4j writes: disabled in every run.
- Auxiliary second LLM and Debug Reviewer: disabled because no independent
  auxiliary credential was configured; their real marginal benefit is not yet
  calibrated.

## Runtime and cache results

| Run | Total | Article P50 | Article P95 | Main extraction cache |
| --- | ---: | ---: | ---: | ---: |
| Shadow, new persistent cache | 255.2 s | 5.49 s | 8.16 s | 0% |
| Shadow, fully populated warm replay | 4.9 s | 0.09 s | 0.20 s | 100% (75/75 reads, 0 remote) |
| Legacy, same warm cache | 4.4 s | — | 0.19 s | 100% |
| Active v2 dry-run, earlier warm cache | 8.3 s | 0.07 s | 0.22 s | 98.65% |

The fully populated replay avoided every primary extraction request and saved
an estimated 246.4 seconds. Its extraction, verification and decision payloads
were identical to the preceding populated-cache replay. Shadow and legacy
verification/decision payloads were also identical when both consumed the same
cached primary candidates. Shadow P95 was 1.05x legacy, below the 2x ceiling.

The Neo4j exact-lookup cache hit rate was 49.72%. This is 2.28 percentage points
below the historical approximately 52% reference and should be watched, but is
not a material regression on a 50-article sample.

## Safety and quality

- Errors reported by the article pipeline: 0.
- Actual entity/relation/update/dispute writes: 0.
- Active v2 invoked Causal twice and produced 20 hypotheses; all remained in
  `hypothesis_relations`, had `import_ready=false`, and were excluded from the
  Decision Engine write input.
- Active v2 invoked Conflict four times. Its recommendations did not change the
  deterministic verifier result, and the final decisions matched legacy.
- 154 unit/regression tests passed; 2 isolated Neo4j integration tests were
  skipped because their dedicated test database variables were not supplied.

For the final stable warm candidate set, gold metrics were:

| Metric | Value |
| --- | ---: |
| Semantic relation F1 | 0.3188 |
| Strict import-ready precision | 0.5714 |
| Evidence span precision (IoU >= 0.5) | 0.2500 |
| Evidence source contiguous rate | 1.0000 |

These values are equal between legacy and shadow for the same cached primary
candidates. They do **not** establish the target `+0.03` F1 improvement, because
shadow intentionally cannot alter production output and the auxiliary model was
disabled.

## Acceptance status

Passed: legacy compatibility, dry-run/write safety, verifier authority, Causal
isolation, warm replay consistency, 100% populated-cache replay, zero-change
rate for executed auxiliary calls (no such calls), and latency ceiling.

Pending before production activation:

1. Configure an independent auxiliary model/reviewer credential.
2. Run the selected complex articles with up to six auxiliary calls and measure
   per-round state change, token use, latency and gold delta.
3. Demonstrate relation F1 non-regression and preferably `+0.03`, with strict
   import-ready and evidence precision non-regression.
4. Keep active `agent-v2` dry-run-only until those gates pass.
