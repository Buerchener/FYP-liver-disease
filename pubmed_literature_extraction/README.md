# PubMed Literature Extraction Workstream

Owner: Shaopeng Chen, 2330026016

This folder contains the PubMed literature extraction component for the liver disease knowledge graph project. It includes the original multi-stage LLM baseline pipeline, the closed-loop cognitive agent, Neo4j schema/import gating, quality-defense logic, and the 30-paper benchmark outputs used to compare V1/V2/V3 extraction quality.

## What is included

| Path | Purpose |
| --- | --- |
| `cognitive_agent/` | V2/V3 closed-loop cognitive agent for PubMed extraction, KG context activation, evidence/schema verification, reasoning, decision ranking, abbreviation deduplication, and optional Neo4j write-back. |
| `multi_stage_extraction_pipeline.py` | V1 baseline PubMed extraction pipeline using a prompt-driven LLM workflow. |
| `entity_linking_preflight.py` | Read-only preflight gate for checking whether extracted relation endpoints can be matched to existing Neo4j KG nodes before import. |
| `convert_pubmed_xml_to_jsonl.py` | Utility for converting PubMed XML exports to JSONL input records. |
| `run_cognitive_agent.sh` | Main launcher for the cognitive agent. Defaults to dry-run mode. |
| `docs/` | Architecture notes, LangExtract redesign notes, and the PubMed extraction report. |
| `reports/` | Supporting migration and quality-fix reports. |
| `tests/` | Focused tests for the cognitive agent loop. |
| `extraction_output/` | Small input samples and selected 30-paper benchmark outputs. Large generated batches are intentionally excluded. |

## Environment

```bash
python3.12 -m venv .venv-cognitive
source .venv-cognitive/bin/activate
python -m pip install -r requirements-cognitive-agent.txt

cp .env.example .env
# Fill API keys and Neo4j password locally. Do not commit .env.
set -a
source .env
set +a
```

Required runtime variables:

| Variable | Meaning |
| --- | --- |
| `GEMINI_API_KEY` or `LLM_API_KEY` | API key for the Gemini-compatible extraction call used by the agent. |
| `GEMINI_API_BASE` | Optional custom OpenAI-compatible proxy base URL. |
| `GEMINI_MODEL` | Gemini-compatible model name. |
| `NEO4J_PASSWORD` | Required only when reading from or writing to Neo4j. |

## Run V1 baseline

```bash
python multi_stage_extraction_pipeline.py \
  --input extraction_output/pubmed_backup_liver_strict_50.jsonl \
  --limit 30 \
  --run-id v1_pipeline_pubmed30 \
  --skip-neo4j
```

## Run V2/V3 cognitive agent

Dry run, no Neo4j write:

```bash
./run_cognitive_agent.sh 30
```

Optional Neo4j write test after reviewing outputs:

```bash
./run_cognitive_agent.sh 5 --write-neo4j
```

## Benchmark files

The selected benchmark artifacts are kept under `extraction_output/`:

| File | Description |
| --- | --- |
| `abtest_compare_20260629_155656.md` | V1 vs V2 30-paper comparison summary. |
| `extraction_results_v1_pipeline_pubmed30_20260629_155656.json` | V1 extracted statements for 30 papers. |
| `quality_report_v1_pipeline_pubmed30_20260629_155656.json` | V1 quality report. |
| `agent_results_v2_agent_pubmed30_20260629_155656.json` | V2 agent results for 30 papers. |
| `agent_results_v3_final_quality_30.json` | V3 quality-defense run after noise filtering and abbreviation deduplication. |
| `agent_report_v3_final_quality_30.json` | V3 aggregate report. |

## Safety notes

- `.env`, API keys, passwords, virtual environments, caches, PPT files, Word thesis files, and large historical backups are intentionally excluded.
- The default agent path is dry-run first. Use `--write-neo4j` only after reviewing schema validity, import readiness, and entity-linking preflight results.
- Neo4j credentials are read from environment variables and must not be committed.
