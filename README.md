<div align="center">

# LiverKG

### An evidence-grounded liver disease knowledge graph and agentic PubMed extraction system

[简体中文](README.zh-CN.md) · [PubMed Agent](pubmed_literature_extraction/README.md) · [Human Review](HUMAN_REVIEW.md)

</div>

## Overview

LiverKG is a research-oriented knowledge graph centred on the progression of
chronic liver disease. It combines curated biomedical databases with a
verifier-guided literature extraction agent that turns PubMed abstracts into
traceable candidate entities and relations.

The project is designed around one principle: **a model may propose knowledge,
but only schema-valid, source-grounded and deterministically verified evidence
may reach the write path**.

```text
NAFLD → NASH → Fibrosis → Cirrhosis → HCC
```

## Highlights

- A five-stage liver disease backbone with auditable identifiers and provenance.
- Integrated gene, protein, pathway, tissue, cell-type and metabolite context.
- A legacy multi-stage extractor and a backward-compatible Central Agent v2.
- Evidence-first relation validation with exact source-span checks.
- Schema-constrained entity-pair classification with an explicit `NO_RELATION` class.
- Conditional DeepSeek/Qwen-compatible adjudication; every edit is re-verified.
- Cache-first execution, bounded remote budgets and per-action latency/token traces.
- Dry-run-by-default Neo4j integration with a deterministic Safe Write gate.
- Frozen gold annotations, reproducible benchmarks and ablation-ready reports.

## Knowledge graph scope

| Layer | Current scale | Primary source |
| --- | ---: | --- |
| Disease | 5 | UMLS-aligned backbone |
| Gene | 836 | DisGeNET |
| Protein | 793 | STRING |
| Pathway | 1,721 | KEGG, Reactome |
| Tissue | 1 | HPA |
| Cell type | 154 | HPA single-cell data |
| Metabolite | 36 | HMDB |
| Gene–disease associations | 1,036 | DisGeNET |
| Protein–protein interactions | 7,154 | STRING |
| Gene–pathway memberships | 9,947 | KEGG, Reactome |
| HPA expression relations | 17,745 | HPA |
| HPA LIHC prognostic relations | 1,384 | HPA |
| Gene–metabolite relations | 97 | HMDB |

These figures describe the curated v2 data package. PubMed-derived statements
remain candidates until they pass the evidence, schema and import-readiness
gates.

## System architecture

```text
Curated databases ────────────────┐
                                  ├─→ Normalised TSV/JSON ─→ Neo4j
PubMed abstracts                  │
  └─→ preprocessing               │
      └─→ primary extraction      │
          └─→ pair classification │
              └─→ Controller      │
                  ├─→ KG lookup   │
                  ├─→ adjudicator │
                  ├─→ reviewer    │
                  └─→ causal/conflict tools
                         ↓
                 deterministic verifier
                         ↓
                    Safe Write gate ──────┘
```

The Central Agent chooses tools from the current article state. Causal output
is isolated as a hypothesis, retrieved graph context cannot replace article
evidence, and no LLM can override a hard verifier failure.

## Repository layout

```text
.
├── data/                         # Curated v2 import data
├── scripts/                      # Data acquisition, normalisation and import
├── review/                       # Current schema, scope and caveats
├── archive/                      # Historical reports and migration artefacts
├── pubmed_literature_extraction/ # Agentic PubMed extraction workstream
├── HUMAN_REVIEW.md               # Recommended review entry point
└── MANIFEST.txt                  # Package inventory
```

## Quick start

The literature agent is the most actively developed component:

```bash
cd pubmed_literature_extraction
python3.12 -m venv .venv-cognitive
source .venv-cognitive/bin/activate
python -m pip install -r requirements-cognitive-agent.txt

cp .env.example .env
# Add credentials locally. Never commit .env.
set -a && source .env && set +a

# Backward-compatible, read-only run
./run_cognitive_agent.sh 5

# Auditable Central Agent v2 shadow run
AGENT_EXECUTION_MODE=agent-v2-shadow ./run_cognitive_agent.sh 5
```

See the [PubMed extraction guide](pubmed_literature_extraction/README.md) for
configuration, evaluation and safety details.

## Reproducibility and safety

- The default execution path is dry-run; Neo4j writes require explicit opt-in.
- Active Agent v2 and active relation-classifier experiments remain dry-run-only.
- API keys, passwords, local environments, caches and generated bulk outputs are ignored.
- External KG context is an entity-linking or conflict hint, never article evidence.
- Every model-produced modification is passed through deterministic verification.
- Isolated Neo4j integration tests never fall back to a production database.

## Evidence boundaries

- DisGeNET associations form the principal disease-evidence backbone.
- STRING interactions are molecular context, not direct disease causality.
- KEGG/Reactome membership does not imply disease-stage specificity.
- HPA cell-type observations are pan-tissue context unless explicitly liver-specific.
- HMDB relations are conservatively filtered disease context, not a complete HMDB import.

## Project status

The curated v2 package has passed backbone-scope checks. Central Agent v2 is
implemented with legacy compatibility, dynamic budgets, persistent replay
caching and Safe Write isolation. Current work focuses on evidence precision,
learned biomedical relation classification and statistically powered evaluation.

For review, start with [HUMAN_REVIEW.md](HUMAN_REVIEW.md) and the
[current caveats](review/04_caveats_and_next_cleanup.md).
