# BioRED-style pair core + CoRE confidence routing

## Implemented data flow

```text
LangExtract entities + exact spans
              │
              ▼
evidence-local entity graph ── type-signature mask ── predicate shortlist
              │
              ▼
pair classifier: predicate | NO_RELATION + calibrated confidence + margin
       ┌──────┴──────────────┐
       │                     │
 high confidence       uncertainty band only
       │                     │
       ▼                     ▼
 deterministic          bounded DeepSeek judge
 verification           (no open endpoints/labels/evidence)
       └──────────┬──────────┘
                  ▼
       re-verification → safe-write gate
```

LangExtract is no longer treated as the final relation authority in active
mode. Its relation output is only a feature/hint. It remains responsible for
the part it currently does well: entity discovery and source-span alignment.

## Literature ideas used

- [BioRED](https://github.com/ncbi/BioRED): document-level entity-pair
  classification with an explicit relation label and novelty distinction. The
  official repository provides 600 abstracts, guidelines, source, and models.
- [CoRE, ACL 2026](https://aclanthology.org/2026.acl-long.1517.pdf): accept
  high-confidence PLM predictions locally and route only the low-confidence
  subset to an LLM. On BioRED, 12.63% of instances were routed in its reported
  split; this is a target to calibrate, not a threshold copied blindly.
- [GLiM, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.727/):
  reduce the quadratic entity-pair space with a graph model before using an
  LLM on difficult cases. Here this becomes an evidence-unit graph plus a hard
  ontology type mask and a candidate cap.
- [RELATE, 2025 preprint](https://arxiv.org/abs/2509.19057): retrieve a short
  ontology predicate list before contextual reranking and explicitly handle
  negation. Here the project relation signatures form the hard shortlist.
- [GRAFT, BioNLP 2026](https://aclanthology.org/2026.bionlp-1.74.pdf): gate
  retrieved context so it can help without overwhelming the local assertion.
  Neo4j/RAG remains linking context and can never become article evidence.
- [Bio-RFX, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.588/): relation
  classification and structural constraints can narrow entity ambiguity. Here
  impossible type-predicate combinations are removed before classification.

## Project-specific innovation

The combination is not a direct reproduction of any one paper:

1. The candidate graph is evidence-bearing. Every edge stores an exact source
   unit and character offsets before any LLM is called.
2. Confidence is not write authority. A high classifier score still has to
   pass schema, endpoint, negation, evidence-role, direction, and safe-write
   checks.
3. DeepSeek has field-level permissions. It can keep/reject or choose from a
   supplied predicate shortlist; it cannot invent an entity, predicate, or
   evidence span.
4. A confident DeepSeek `KEEP` clears only the pair-uncertainty barrier. It
   cannot clear deterministic evidence/schema blockers, and the result is
   re-verified.
5. Shadow and active outputs share one audit schema, allowing a paired
   counterfactual evaluation without changing production writes.

## Current validation status (2026-08-14)

All code tests pass: **137 passed, 2 integration tests skipped**.

The NCBI data, guideline, and baseline source were downloaded to the ignored
research cache and checksum-verified. The official model archive is about
868 MB and was deliberately not made a production dependency.

A small optional TF-IDF + calibrated linear model was trained on the official
BioRED train split and calibrated on dev:

| Official mapped test class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `ASSOCIATED_WITH` | 0.608 | 0.379 | 0.467 |
| `NO_RELATION` | 0.921 | 0.967 | 0.944 |
| `INTERACTS_WITH` | 0.000 | 0.000 | 0.000 |

Only three compatible `INTERACTS_WITH` examples remained in the mapped test
set, so that zero must not be generalized. More importantly, zero-shot transfer
to the existing 100-article liver run failed the project relation gate (F1
0.022). The deterministic fallback also failed the active-quality gate (all
positive predictions F1 0.061; high-confidence precision 0.273). These backends
therefore remain **shadow-only**. This prevents benchmark leakage and unsafe
Neo4j writes.

The candidate lattice covered 42 of 63 gold relation endpoint pairs (recall
0.667) in those 100 articles. This is better interpreted as the current upper
bound of the classifier than as final recall: the remaining 21 pairs cannot be
recovered by any downstream pair classifier without improving entity/evidence
candidate construction.

## Activation gate

Before `PAIR_CLASSIFIER_MODE=active` can be considered for production:

1. Train on project pair instances derived only from the development split,
   including hard `NO_RELATION` pairs and predicate-confusion negatives.
2. Calibrate confidence on a separate calibration split; select thresholds by
   safe-write precision and expected calibration error, not raw accuracy.
3. Evaluate candidate recall, pair-label F1, routed-subset F1, routing rate,
   evidence exactness, latency, token cost, and dangerous writes separately on
   the frozen blind split.
4. Require zero dangerous writes and no decrease in safe-write precision.
5. Keep the prior LangExtract route as a configuration-level rollback.
