# Golden-shot and chunked extraction

The article Agent selects demonstrations and extraction windows before the
LangExtract candidate generator runs.

- Clinical, mechanistic, omics, animal and in-vitro articles receive three
  profile/cue-matched positive demonstrations and one boundary/negative example.
- Computational and review articles receive two positives and two negative
  boundary examples to prevent screening, docking and background co-occurrence
  from becoming asserted relations.
- Demonstrations derived from the 50-document gold set carry a source PMID and
  are excluded when that same PMID is evaluated.
- Short/simple abstracts remain one request. Long abstracts (>1,800 characters)
  or high-complexity abstracts (>=1,400 characters) are split into at most three
  contiguous section/sentence-aligned windows with one parent-sentence overlap.
- Chunk-local character spans are rebased to the full article. Duplicate overlap
  candidates are merged before the normal verifier/collaborator/decision chain.
- Non-exact LangExtract alignment is retained in the audit trail but marked
  ungrounded, so an expanded phrase cannot enter an import-ready relation.

All thresholds are configurable and the report records example names, selection
reasons, chunk boundaries, number of extraction windows and whether chunking was
used for each article.
