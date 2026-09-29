# V0.1 evaluation status

Date: 2026-09-29. Development sample: PMIDs 41873055, 41546802, 41641927, selected from the existing converted PubMed corpus. Their abstracts are in `../data/abstracts.jsonl`; the source file hash is in `../data/provenance.json`. This is a purposive development sample, not a held-out benchmark.

## Successful live run

`outputs/pilot_ludies_configured_v01/` used the project's existing structured client with a locally configured Ludies OpenAI-compatible API endpoint and `gemini-3.7-flash-high`. All six logical calls completed; one event call received HTTP 429 before a successful SDK retry. The explicit `--env-file` precedence was fixed before this run because a stale inherited environment variable had overridden the newly configured key. The key is not committed.

| PMID | Events | Event relations | Concept rows |
|---|---:|---:|---:|
| 41873055 | 5 | 0 | 53 |
| 41546802 | 7 | 0 | 91 |
| 41641927 | 6 | 2 | 49 |
| **Total** | **18** | **2** | **193** |

The two accepted relations are `AS_A_RESULT`; the other four allowed labels did not appear. The concepts comprise 61 Event rows, 126 Entity rows and 6 Relation rows, each a separate candidate. `rejected.jsonl` and `failures.jsonl` are empty. The 213 rows in `manual_review.csv` all remain `pending`. Automated inspection confirmed every event/relation evidence offset and participant offset slices the original text exactly, and each event relation references accepted events. These are structural/source-grounding checks only.

Spot checks found one uncertain conclusion preserved as uncertain (PMID 41873055), and two causal directions on PMID 41641927 supported by explicit wording. No person has completed the line-by-line biomedical review; full semantic support, entity specificity, and concept abstraction quality are unverified. In particular, a composite event in PMID 41546802 includes MASH → HSC activation → matrix deposition → fibrosis within one sentence, so a zero relation count for that article is not evidence that the abstract has no causal sequence. Several concept candidates are broad or potentially disputable and need review before reuse.

The 15 offline contract tests passed, including an end-to-end no-network replay and a regression for explicit env-file precedence. Replaying the six published raw responses reproduced exactly the same 18 event IDs, 2 relation IDs and 193 concept IDs. Existing pipeline tests (article preprocessing 4, evidence selector 5, main KG write contract 6, extraction quality 41, collaborative extractor 19) also passed in the target branch checkout. Existing pipeline files were not modified.

## Earlier failed attempts

Before the key update, both configured auxiliary and primary endpoints returned HTTP 401 and generated zero accepted rows. After the key update, a standalone `urllib` probe was blocked with HTTP 403 / `error code: 1010`, while the pipeline's actual structured client completed calls. These failed local runs remain ignored by Git; the successful published sample is the directory above. Do not interpret the failed probes as model evaluation.

## Remaining manual gate

Review all 18 event sentences against the exact abstract, including species/model, negation and uncertainty. Review both event relations for head/tail causality; identify missing relations without adding unsupported ones. Review concept rows for genuine higher-level meaning rather than paraphrase or category error. Fill the CSV review columns and then report support/quality counts. No precision/recall or medical claim should be reported from this unreviewed development sample.
