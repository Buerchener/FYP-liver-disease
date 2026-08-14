# PubMed 200 relation gold set v2 — strict adjudication

`pubmed_200_gold_v2_strict.jsonl` is a relation-centric manual annotation of
the first 200 records in `extraction_output/pubmed_converted_500.jsonl`, kept
in exactly the same source order.

## Annotation boundary

- The records were read and annotated directly, without using the project's
  extractor output as labels and without asking another model to generate the
  annotations.
- Allowed entity types are `Gene`, `Protein`, `Disease`, `Pathway`,
  `Metabolite`, `Tissue`, and `CellType`.
- Allowed predicates are the eight predicates defined in
  `cognitive_agent/schema/relation_signatures.py`.
- The set is relation-centric, not an exhaustive NER corpus. `entities`
  contains relation endpoints and enough core disease context to make
  deliberate zero-relation records interpretable.
- Each evidence value is a continuous, verbatim span from the supplied title
  or abstract. No evidence text was repaired or paraphrased, including source
  typos.
- `relations` contains semantically supported statements. The narrower
  `import_ready=true` subset is intended for safe automatic Neo4j ingestion.
  Animal, cell-line, review/background, computational, uncertain, single-case,
  underpowered, or otherwise weakly generalizable evidence is retained only
  with `import_ready=false` and a written exclusion reason.
- Empty `relations` is a curated negative decision, not a missing label.

## Strict adjudication checks

The final 200-record file passed all of the following checks with zero errors:

1. exactly 200 records and 200 unique PMIDs;
2. PMID and title match the first 200 source records in exact order;
3. every entity mention occurs verbatim in the supplied title or abstract;
4. every evidence span occurs verbatim and continuously in the supplied title
   or abstract;
5. every relation endpoint maps to an entity canonical name and type in the
   same record;
6. every predicate and endpoint-type signature is allowed by the project
   ontology;
7. every direction is in the allowed direction vocabulary;
8. every non-importable relation has a non-empty exclusion reason, and every
   import-ready relation has no exclusion reason;
9. no duplicate relation/evidence tuple occurs within a record.

After structural validation, every `import_ready=true` relation was manually
re-reviewed. Computational correlations, case reports, four-patient discovery
results, tentative combined biomarkers, and Mendelian-randomization mediation
claims were downgraded where automatic ingestion would overstate the evidence.

## Final counts

- Documents: 200
- In-scope documents: 159
- Deliberate zero-relation documents: 123
- Semantic relations: 172
- Strict `import_ready=true` relations: 29
- Predicate counts:
  - `ASSOCIATED_WITH`: 117
  - `INTERACTS_WITH`: 15
  - `PROGRESSES_TO`: 13
  - `PARTICIPATES_IN`: 12
  - `EXPRESSED_IN`: 11
  - `PROGNOSTIC_IN`: 3
  - `ASSOCIATED_WITH_METABOLITE`: 1
  - `ENCODES`: 0

## Files

- `pubmed_200_gold_v2_strict.jsonl`: final combined, strictly adjudicated set.
- `pubmed_50_gold_v1.jsonl`: historical first-50 component, with exact-span
  corrections made during v2 review.
- `pubmed_51_200_gold_v2_strict.jsonl`: manually annotated records 51–200.

All rows in the final combined file carry
`review_status="llm_manual_v2_strict_adjudicated"`.

This is a rigorously curated LLM gold set for project development and internal
evaluation. A publication-grade clinical benchmark should still receive an
independent, blinded adjudication by at least one hepatologist or professional
biomedical curator; that limitation must not be concealed in reporting.
