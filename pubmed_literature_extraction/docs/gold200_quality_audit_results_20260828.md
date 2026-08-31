# Gold200 quality audit results (2026-08-28)

## Status

Gold200 is a development and error-analysis set. The frozen v2 file remains unchanged. The audit generated an additive v3 draft, ledgers, and an expert queue; no model decision was automatically applied to Gold.

- Frozen Gold SHA-256: `83eb7cc1380b433d33745da31aa6095d166a3478dfaa53f9abe6464c3d5043a9`
- Source corpus SHA-256: `b4f17b3e8e473db1c4348b4ac985787f28181aacca6f087446ab42a2dbd32b21`
- Structure: 200 documents, 172 relations, 123 zero-relation documents
- L0 result: structurally valid; no PMID, source, endpoint, duplicate, or evidence-offset errors
- Frozen metadata gaps: all 172 relations lack `claim_role`
- Write subsets: 70 write-contract relations, 29 historical import-ready relations, and 15 strict import-ready relations concentrated in 4 documents

## Tiered review

L1 selected 96 documents and 124 relations. Selection covered all historical import-ready and write-contract relations, strong-trigger zero-relation documents, uncertainty, non-current context, rare predicates, and same-pair multi-label cases.

L2 submitted all 200 documents to blinded Gemini and DeepSeek review. Reviewers saw only the article and annotation contract, never system predictions.

- 400 reviewer tasks: 397 valid responses, 3 invalid structured responses, 0 provider failures
- Relation review: 94 consensus, 69 disagreement, 5 single-review, 4 unreviewed
- Document review: 138 consensus, 60 disagreement, 1 single-review, 1 unreviewed
- 138 grouped missing-relation proposals, but only 3 matched on endpoints, predicate, direction, and claim role
- 45 malformed additive proposals were rejected without discarding otherwise valid relation reviews
- Warm replay: 400 local result hits and 0 remote requests
- Automatic Gold changes: 0

The three invalid responses were exact-quote or relation-ID contract failures for PMID 41430095 and PMID 41804293. They are review coverage gaps, not evidence that those Gold labels are wrong.

## Expert priorities

Four likely omissions require biomedical adjudication and are recorded in `gold_annotations/pubmed_200_gold_v3_adjudication_candidates.jsonl`:

1. PMID 41518251: `MASLD -> MASH / PROGRESSES_TO` as background.
2. PMID 41679350: `Galectin-1 -> Hepatitis / ASSOCIATED_WITH`; tuple likely valid, but the review article role should be background rather than current finding.
3. PMID 41801960: `Obesity -> HCC / ASSOCIATED_WITH` as an explicit background association.
4. PMID 41995526: `Iodine -> Advanced liver fibrosis / ASSOCIATED_WITH`; endpoint relation is likely present, but direction semantics remain disputed.

The same queue also records an endpoint-type dispute for PMID 41650163. The frozen Gold labels `OTUD5 (Gene) -> MAVS (Protein) / INTERACTS_WITH`, whereas a physical interaction normally refers to two proteins. Because the article contains both Gene and Protein readings without authoritative endpoint IDs, neither interpretation is silently promoted.

Among 29 historical import-ready labels, 23 received dual-model confirmation, 2 had reviewer disagreement, and 4 had only one valid review. No existing import-ready relation received dual-model consensus for removal. This supports retaining them for now, while keeping the disputed and single-review records in the expert queue.

## Consequences

The v3 draft is not an adjudicated Gold yet. `claim_role`, `direction_semantics`, evidence spans, and contract-derived write status are usable for auditing, but all semantic revisions remain `EXPERT_REVIEW_REQUIRED` until a qualified reviewer decides them.

The first post-audit Sentinel run was experimentally contaminated because its legacy arm inherited `SECOND_LLM_ENABLED=true` from `.env`; its A/B comparison must not be reported. The harness now explicitly disables the second model for legacy runs.

The clean final Sentinel is under `benchmark_output/tiered_v2_sentinel10_gold_audit_20260828_v3`:

- Legacy used 0 auxiliary calls; tiered-v2 used 2, both successful, with no retries or provider failures.
- Tiered cold run completed in 118 seconds; warm replay completed in about 1 second with 15 local cache hits and no remote calls.
- Provider prefix cache read 6,912 of 36,970 prompt tokens (18.7%).
- Zero-relation specificity improved from 0.833 to 1.000 and dangerous writes fell from 2 to 0.
- Exact relation F1 improved from 0.118 to 0.133 (`+0.016`), below the planned `+0.03` gate.
- Strict import-ready made no positive predictions in this 10-paper sample. The displayed empty-set precision is not evidence of quality and must not be reported as a substantive score.

The clean run therefore passes stability, cache, hard-negative, and dangerous-write gates, but fails the semantic-improvement gate. Gold200 batch evaluation and BioRED Test remain paused. The next optimization should target the existing pairwise judge/relation-core selection bottleneck after expert adjudication, not add another Agent module.
