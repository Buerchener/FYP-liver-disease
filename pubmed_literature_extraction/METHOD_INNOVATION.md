# Agent v3 method innovations

Agent v3 is a verifier-governed extraction system that learns compact soft
rules without training a local BERT/Transformer. Its scientific hypothesis is
that deterministic evidence constraints and reusable rule memory can resolve
most biomedical relation candidates, while large language models should be
reserved for genuinely ambiguous cases.

## 1. Verifiable dual-model rule induction

DeepSeek converts development-set `ErrorCard` batches into a closed rule DSL.
Qwen independently checks safety, scope, conflicts and over-generalisation.
Rules can add bounded prompt guidance, adjust a pair score by at most ±0.20,
request review, or downgrade a decision. They cannot execute code, contain
article sentences, identify a test PMID, modify the ontology, or waive a hard
verification gate.

The lifecycle is:

```text
candidate → validated → shadow → active
                         └──────→ rejected / retired
```

Automatic promotion requires both models to approve, support from at least
three development PMIDs, two repaired errors, at most one new regression, no
strict-precision loss, zero dangerous writes, a 90% bootstrap probability of
non-negative calibration gain, and one completed shadow replay. The active
bundle is immutable within an evaluation run and participates in every cache
key.

This turns “self-reflection” from unverified online prompt mutation into an
offline, versioned and reversible learning process.

## 2. Evidence-closed relation construction

The relation core searches evidence units before constructing schema-valid
entity pairs. For each pair and predicate it selects the shortest contiguous
source span that includes both endpoints and an explicit trigger. Nearby
negation is retained in the minimal span; background, objective and method
sections cannot be treated as findings.

Evidence and relation confidence are separate variables. Local rules decide
clear entailment or contradiction. DeepSeek sees only bounded uncertain
candidates and must return `ENTAILED`, `CONTRADICTED`, or
`NOT_ENOUGH_INFORMATION`. Every returned quote is realigned to source offsets.
Qwen is called only when the evidence decision conflicts with the proposed
predicate. Neither auxiliary model may add an entity or an open-ended relation.

This creates a closed loop:

```text
exact span → endpoints → schema pair → predicate → entailment
           → verifier → decision or abstention
```

## 3. Replay-gated rule memory

The 200 available development articles are frozen into 120 induction, 40
validation and 40 calibration articles using a label-aware deterministic
manifest. Rule learning may run for at most three rounds and stops when two
successive validation gains are below 0.005. A bundle is evaluated on fixed
primary candidates, so model sampling cannot be mistaken for a rule gain.

Runtime failures become privacy-safe ErrorCards rather than direct strategy
changes. Exact article evidence is excluded from induction prompts, and a
separate leakage guard checks blind PMIDs and abstract hashes against rules,
calibration rows and examples.

## 4. Conformal risk routing

The non-parametric router combines relation score, evidence score, model
agreement, verifier flags, rule support/conflict, linking ambiguity, study type,
predicate and section. Split-conformal residuals produce a finite-sample upper
error-risk estimate. Mondrian groups are used only with at least 20 calibration
examples; otherwise the router explicitly falls back to global calibration.

The policy is monotone in safety:

```text
certified low risk  → ACCEPT_LOCAL
uncertain           → CALL_DEEPSEEK
model disagreement  → CALL_QWEN_CRITIC
unresolved risk     → ABSTAIN / HUMAN_REVIEW
Verifier failure    → ABSTAIN (never upgraded)
```

Targets are α=0.05 for import-ready acceptance and α=0.10 for semantic-only
acceptance. These are prospective error targets, not claimed achieved results;
coverage and realised error must be reported on the frozen expert blind set.

## Intended contribution and limits

The contribution is the combination of closed rule induction, independent
model criticism, source-offset evidence closure and calibrated selective
routing under a single immutable permission boundary. Individual components
are established ideas; the research claim must be supported by the registered
ablations rather than novelty language alone.

Agent v3 does not eliminate annotation requirements. The current 200 labels are
development resources, not final publication gold. The preregistered blind 50
must be independently expert-labelled, at least 20% double-reviewed, and run
exactly once after code, prompts, thresholds and the rule bundle are frozen.

