# PubMed relation gold sets

The current strict 200-document gold set is documented in
`README_pubmed_200_gold_v2_strict.md` and stored in
`pubmed_200_gold_v2_strict.jsonl`.

The remainder of this file describes the historical 50-document v1 set.

# PubMed 50 relation gold set v1

This directory contains a relation-centric manual annotation of the first 50
records in `extraction_output/pubmed_converted_500.jsonl`.

The annotations were produced by direct LLM reading and adjudication, without
using the project's extractor output as labels and without writing an automatic
annotation program. They are intended as a reproducible evaluation baseline,
not as a substitute for final review by a hepatologist or biomedical curator.

## Files

- `pubmed_50_gold_v1.jsonl`: one JSON object per PubMed record, in source order.

## Scope

- Entity types are restricted to: `Gene`, `Protein`, `Disease`, `Pathway`,
  `Metabolite`, `Tissue`, and `CellType`.
- Predicates are restricted to the eight predicates currently supported by the
  project.
- This is relation-centric rather than exhaustive NER annotation. `entities`
  contains relation endpoints plus the core disease entity needed to explain a
  zero-relation document. It should not be used alone to score exhaustive entity
  recall.
- `relations` contains semantically supported relations. `import_ready=true`
  is the stricter target for automatic Neo4j insertion.
- A semantically supported animal, in-vitro, review-only, background-only, or
  computationally predicted relation can remain in `relations` while carrying
  `import_ready=false` and an explicit `exclusion_reason`.
- `negative_notes` records why an article has no supported relation under the
  current ontology or why tempting candidates must not be treated as facts.

## Interpretation

- Evaluate semantic extraction against all entries in `relations`.
- Evaluate safe automatic ingestion against only relations where
  `import_ready=true`.
- Treat an empty `relations` array as a deliberate negative example, not a
  missing annotation.
- Evidence strings are copied as continuous spans from the supplied abstract.

## Version 1 counts

- Documents: 50
- In-scope documents: 42
- Deliberate zero-relation documents: 31
- Semantically annotated relations: 45
- Strict `import_ready=true` relations: 22
- Predicate coverage: `ASSOCIATED_WITH`, `EXPRESSED_IN`, `INTERACTS_WITH`,
  `PARTICIPATES_IN`, and `PROGRESSES_TO`

## Review status

Every row has `review_status="llm_manual_v1"`. Before calling the dataset a
publication-grade biomedical gold standard, a domain expert should independently
adjudicate Gene/Protein distinctions, broad `ASSOCIATED_WITH` edges, and clinical
phenotype boundaries.
# Article profile gold sets

- `article_profiles_10_gold_v1.json`: development set. It may be used to tune rules and prompts, so it is not an unbiased final estimate.
- `article_profiles_40_blind_v1.json`: frozen blind evaluation set, disjoint from the 10-item development set. Labels were assigned from titles and full abstracts before running the router evaluation.

The v2 profile schema separates `evidence_design`, `causal_strength`, and `validation_level`. Aggregate FAST/STANDARD/DEEP labels remain for compatibility, while tool-level gates are evaluated separately.
