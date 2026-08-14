<div align="center">

# PubMed Literature Extraction Agent

### Evidence-grounded biomedical relation extraction for LiverKG

[简体中文](README.zh-CN.md) · [Back to LiverKG](../README.md) · [Agent v3 architecture](ARCHITECTURE_V3.md) · [Method innovations](METHOD_INNOVATION.md)

</div>

## What this project does

This workstream extracts liver-disease knowledge from PubMed abstracts and
turns it into auditable Neo4j candidates. It contains the original prompt-based
pipeline, a closed-loop cognitive agent, and Central Agent v2: a deterministic,
verifier-guided controller that calls expensive tools only when the current
article state justifies them.

Agent v3 adds verifier-safe rule memory, DeepSeek/Qwen dual-model criticism,
minimal-span evidence entailment and a non-parametric conformal risk router.
It does not train a local BERT; its research path uses frozen development
splits and a preregistered expert blind cohort.

The system deliberately separates **proposal** from **authority**:

- LLMs, retrieval and causal tools may propose or annotate candidates.
- The deterministic verifier owns evidence and schema validity.
- The Decision Engine and Safe Write gate own import readiness.
- The default runtime is read-only; active v2 remains dry-run-only.

## Pipeline

```text
PubMed JSONL
  → parallel preprocessing
  → section-aware chunking and primary extraction
  → evidence-local entity-pair lattice
  → predicate / NO_RELATION classification
  → deterministic verification
  → Central Agent observation
      ├─ cache-first Neo4j lookup
      ├─ bounded second-model adjudication
      ├─ targeted debug review
      ├─ evidence repair / relation recovery
      └─ causal and conflict analysis
  → re-verification after every modification
  → Decision Engine
  → dry-run or Safe Write
```

## Central Agent v2

Central Agent v2 maintains article-level state for entities, pairs, evidence,
accepted/rejected/reviewed relations, linking ambiguity, model disagreement,
hypotheses, budgets and termination reasons.

| Route | Action soft budget | Auxiliary remote calls | Neo4j calls | Soft timeout |
| --- | ---: | ---: | ---: | ---: |
| FAST | 12 | 2 | 2 | 30 s |
| STANDARD | 20 | 4 | 4 | 60 s |
| DEEP | 28 | 6 | 6 | 120 s |

The global guards are 40 actions, 8 auxiliary remote requests, 8 Neo4j calls
and 180 seconds per article. A cache hit consumes no remote budget. Two
consecutive remote calls without a candidate, verification or review-state
change terminate further remote work.

Execution modes:

| Mode | Behaviour |
| --- | --- |
| `legacy` | Backward-compatible default; existing production result wins. |
| `agent-v2-shadow` | Records counterfactual actions without changing production output. |
| `agent-v2` | Executes v2 routing, but is hard-restricted to dry-run. |

## Evidence and write safety

No LLM or tool can override:

- an illegal entity/relation schema;
- a missing or unresolved endpoint;
- evidence absent from the source text;
- negated, background-only, objective-only or method-only claims;
- non-human evidence outside the configured scope;
- an `import_ready=false` decision;
- the Safe Write restrictions.

Causal chains are stored separately as `hypothesis_relations`. Without direct
article evidence they can be reviewed as research hypotheses, but never become
write-ready facts. Conflict analysis may recommend create, keep, update,
dispute or review; the verifier and Decision Engine still decide authority.

## Installation

```bash
python3.12 -m venv .venv-cognitive
source .venv-cognitive/bin/activate
python -m pip install -r requirements-cognitive-agent.txt

cp .env.example .env
# Add secrets locally. .env is ignored by Git.
set -a && source .env && set +a
```

Important variables:

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY`, `GEMINI_API_BASE`, `GEMINI_MODEL` | Primary extraction model. |
| `SECOND_LLM_ENABLED` | Enables bounded auxiliary adjudication. |
| `SECOND_LLM_API_KEY`, `SECOND_LLM_API_BASE`, `SECOND_LLM_MODEL_ID` | DeepSeek/Qwen-compatible adjudicator. |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | Optional KG access. |
| `NEO4J_RAG_ENABLED` | Enables bounded read-only KG context. |
| `EXTRACTION_CACHE_MODE`, `EXTRACTION_CACHE_PATH` | Memory or persistent replay cache. |
| `AGENT_EXECUTION_MODE`, `AGENT_BUDGET_PROFILE` | Controller mode and quality/cost profile. |
| `RULE_MEMORY_MODE`, `RULE_BUNDLE` | Frozen Agent v3 soft-rule memory. |
| `EVIDENCE_ENTAILMENT_MODE` | Local-first evidence adjudication. |
| `RISK_ROUTER_MODE`, `CONFORMAL_CALIBRATION` | Selective conformal routing. |

## Running the system

Legacy-compatible dry run:

```bash
./run_cognitive_agent.sh 5
```

Agent v2 shadow run with persistent replay cache:

```bash
export AGENT_EXECUTION_MODE=agent-v2-shadow
export AGENT_BUDGET_PROFILE=quality
export EXTRACTION_CACHE_MODE=persistent
export EXTRACTION_CACHE_PATH=.cache/agent-v2.sqlite3
./run_cognitive_agent.sh 50
```

Conditional DeepSeek adjudication:

```bash
export SECOND_LLM_ENABLED=true
export SECOND_LLM_PROVIDER=openai
export SECOND_LLM_API_BASE=https://api.deepseek.com
export SECOND_LLM_MODEL_ID=deepseek-v4-flash
# SECOND_LLM_API_KEY may be set directly; the launcher can also use DEEPSEEK_API_KEY.
./run_cognitive_agent.sh 5 --max-workers=1
```

Original baseline:

```bash
python multi_stage_extraction_pipeline.py \
  --input extraction_output/pubmed_backup_liver_strict_50.jsonl \
  --limit 30 \
  --run-id v1_pipeline_pubmed30 \
  --skip-neo4j
```

## Cache and auditability

The extraction layer provides:

- bounded in-process L1 cache;
- optional bounded SQLite L2 cache;
- content/configuration/model-aware cache keys;
- concurrent single-flight deduplication;
- no API keys in cache keys or values;
- caching only for structurally replayable results;
- attempted/successful/retried/cached counts, token usage and latency percentiles.

With a fully populated 50-article cache, the acceptance run reached 100% main
extraction cache hits, made zero primary extraction requests and completed in
4.9 seconds. See the [calibration report](docs/agent_v2_acceptance_20260814.md).

## Evaluation

Gold annotations and research outputs are organised under:

```text
gold_annotations/  # frozen human-readable gold data
benchmark_output/  # fixed-candidate model/router comparisons
scripts/           # evaluation and benchmark runners
docs/              # architecture, evidence policy and acceptance reports
tests/              # deterministic and integration tests
```

Run the local suite:

```bash
python -m unittest discover -s tests -v
```

Neo4j integration tests require an explicitly isolated database:

```bash
export NEO4J_TEST_URI=bolt://localhost:7687
export NEO4J_TEST_USER=neo4j
export NEO4J_TEST_PASSWORD=local-test-password
export NEO4J_TEST_DATABASE=cognitive-agent-test
python -m unittest tests.test_neo4j_integration -v
```

If these variables are absent, the integration tests skip safely and never
fall back to a production database.

## Repository map

| Path | Responsibility |
| --- | --- |
| `cognitive_agent/agent.py` | End-to-end execution and compatibility layer. |
| `cognitive_agent/central_agent_v2.py` | State, budgets, routing and action audit. |
| `cognitive_agent/verifier.py` | Evidence, endpoint and schema validation. |
| `cognitive_agent/relation_pair_classifier.py` | BioRED-style pair lattice and classification. |
| `cognitive_agent/rule_memory.py` | Closed rule DSL, lifecycle and promotion gates. |
| `cognitive_agent/evidence_selector.py` | Minimal exact spans and entailment closure. |
| `cognitive_agent/conformal_router.py` | Global/Mondrian non-parametric risk routing. |
| `cognitive_agent/collaborative_extractor.py` | Bounded second-model adjudication. |
| `cognitive_agent/extraction_cache.py` | L1/L2 cache and single-flight execution. |
| `entity_linking_preflight.py` | Read-only endpoint-linking preflight. |
| `experiment_metrics.py` | Experiment metrics and comparisons. |
| `run_cognitive_agent.sh` | Safe launcher; dry-run by default. |

## Research status

The three Agent v3 implementation batches are available. See the honest
[English](RESULTS_V3.md) or [Chinese](RESULTS_V3.zh-CN.md) experiment status.
Final paper claims remain pending expert annotation and the single frozen blind
run; the preregistered cohort is deliberately not used for tuning.
