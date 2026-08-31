# Gold200 quality audit v3

Gold200 is a development and error-analysis set. It has been used in repeated
system diagnosis, so it is not the final leakage-free benchmark. BioRED's
official test split is the external final test; code, prompts, Rule Memory and
thresholds must be frozen before that test is run.

## Immutable source and audit levels

`pubmed_200_gold_v2_strict.jsonl` remains immutable. Its SHA-256 is recorded in
every audit manifest. The audit CLI produces two local ledgers plus an optional
tracked draft without overwriting the frozen file.

- L0 validates source identity and order, exact evidence spans, typed endpoints,
  candidate signatures, direction vocabulary and view counts.
- L1 reviews all historical import-ready relations, all main-KG contract
  relations, uncertainty/negation and non-current contexts, same-pair
  multi-label cases, rare predicates and zero-relation documents with strong
  relation language.
- L2 reviews all 200 documents and all 172 relations.

Run the stages with at most two workers:

```bash
.venv-cognitive/bin/python scripts/audit_gold200_quality.py \
  --stage static --emit-tracked-draft

.venv-cognitive/bin/python scripts/audit_gold200_quality.py \
  --stage l1 --reviewers both --max-workers 2

.venv-cognitive/bin/python scripts/audit_gold200_quality.py \
  --stage l2 --reviewers both --max-workers 2
```

The Gemini and DeepSeek prompts include only the source article, frozen
annotation and annotation policy. They never include extractor predictions.
Responses are cached in SQLite, including terminal invalid responses, so an
identical warm replay makes zero remote calls. Pass `--retry-invalid` when the
operator intentionally wants to retry only cached invalid entries. Model
suggestions are advisory.

## Audit data contract

The tracked audit draft adds `relation_id`, `claim_role`, `evidence_spans`,
`direction_semantics`, `gold_write_status`, `adjudication_status` and explicit
reasons. `direction` remains compatible with v2; `direction_semantics`
distinguishes association sign from expression/change direction.

`EXPERT_REVIEW_REQUIRED` is mandatory for historical import-ready items and
other L1 risks. A model consensus cannot set `adjudication_complete=true` and
cannot add, remove or upgrade a relation automatically. Until every unresolved
item has an explicit human decision, the v3 artifact is an audit draft rather
than an adjudicated gold set.

## Evaluation policy

The unified evaluator defaults to frozen v2. Audit v3 requires an explicit
development-only override while adjudication is incomplete:

```bash
.venv-cognitive/bin/python scripts/evaluate_gold200_unified.py \
  --result RESULT.json \
  --gold-profile audit-v3 --allow-incomplete-audit-gold \
  --output-json metrics.json --output-md metrics.md
```

Candidate semantic, review-inclusive coverage, main-KG contract and strict
import-ready remain separate views. Strict import-ready contains only 15
relations in four documents, so the evaluator reports Wilson intervals and a
small-sample warning; this view is a safety regression gate, not standalone
evidence of production write quality.
