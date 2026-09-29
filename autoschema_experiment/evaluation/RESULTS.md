# V0.1 evaluation status

Date: 2026-09-29. Development sample: PMIDs 41873055, 41546802, 41641927, selected from the existing converted PubMed corpus. Their abstracts are in `../data/abstracts.jsonl` with source file hash in `../data/provenance.json`. This is a purposive sample, not a frozen held-out evaluation set.

## Runs

Two live attempts against pre-existing project configuration returned HTTP 401 Unauthorized for all three articles: `deepseek-v4-flash` via the configured auxiliary endpoint, then `gemini-3.7-flash-high` via the configured primary endpoint. Neither generated events, relations or concepts. Local failed run directories retain manifests/logs; they are ignored by Git. No claim of successful semantic extraction or manual review is made. No credentials are stored in this repository.

The 14 offline contract tests passed, covering strict source offsets, participants, relation enum and endpoint integrity, concept types/coverage, replay hash binding, duplicate PMIDs and overwrite protection. Existing pipeline tests (article preprocessing 4, evidence selector 5, main KG write contract 6, extraction quality 41, collaborative extractor 19) also passed in the target branch's checkout. These checks support integration safety but cannot measure biomedical extraction quality.

## Required live gate

After a working project API is configured locally, run the documented 3-abstract command into a new directory. Require zero call failures and inspect every `rejected.jsonl` item. Check that accepted events are complete and faithful to species, experiment, negation and uncertainty; inspect exact source offsets and relation direction; review the concept candidates for genuine semantic abstraction. Populate `manual_review.csv`, then summarize counts by support and quality. Until this happens, no precision/recall or successful pilot result can be reported.
