# LiverKG CLI

LiverKG is the product entry point for the PubMed liver disease knowledge graph extractor. It wraps the existing Cognitive Agent without changing the extraction algorithm, verifier, router, or Safe Write boundary.

## Install

Use Python 3.11 or newer:

```bash
pipx install .
uv tool install .
```

The console command is:

```bash
liverkg --help
```

## Common Workflow

```bash
liverkg init
liverkg doctor
liverkg extract extraction_output/pubmed_converted_500.jsonl --profile quality --limit 50
liverkg extract --pmid 12345678 --detach
liverkg status
liverkg logs
liverkg report RUN_ID
```

`quality` is the default profile and runs in dry-run mode. Neo4j writes require `--write-neo4j` plus an explicit confirmation. In non-interactive jobs, use `--write-neo4j --yes`; the write guard still rejects non-localhost Neo4j targets and non-`neo4j` databases.

## Inputs

`liverkg extract` accepts:

- JSONL records with `pmid`, `title`, and `abstract`;
- single PubMed XML files;
- repeated `--pmid` values, fetched through NCBI E-utilities with raw XML snapshots.

Local input and `--pmid` are mutually exclusive.

Pass a dataset schema JSON to use its entity types, predicates, endpoint signatures, and evidence rules throughout extraction and validation.

```bash
liverkg extract articles.jsonl --schema-profile docs/schema_profile.example.json
```

The run saves a schema snapshot for background execution and resume. Custom schema runs export JSON candidates and require dry-run mode. See [Schema profile configuration](docs/schema_profiles.zh-CN.md).

## Evidence Review

```bash
liverkg review -o defense-review.html
liverkg review extraction_output/agent_results_RUN_ID.json -o audit.html
liverkg --lang zh review --run-id RUN_ID -o audit-zh.html
```

Open the generated HTML file in a browser. Search by endpoint, predicate or PMID and filter by semantic status. Each candidate shows its recorded evidence, schema checks, semantic decision and graph eligibility. Missing fields in older runs display as unrecorded. Import eligibility reflects a verifier decision about a candidate.

With no input, the command runs a fictional demonstration through the verifier. The HTML embeds the selected article and candidate data and works offline.

## Configuration

Non-sensitive configuration is stored in the platform user config directory and can be overridden by a project `.liverkg.toml`.

Resolution order:

```text
CLI options > environment variables > project config > user config > built-in defaults
```

Secrets are read from environment variables first and then from the system keychain. They are never passed as ordinary CLI arguments and are redacted in diagnostics.

## Run Management

Background runs are independent subprocesses, not a daemon. Each run stores:

- `runspec.json`
- `status.json`
- heartbeat/PID
- logs
- normalized input
- raw PMID snapshots when applicable
- article-level checkpoints

Use:

```bash
liverkg runs list
liverkg status RUN_ID
liverkg logs RUN_ID
liverkg runs stop RUN_ID
liverkg resume RUN_ID
```

## Research Namespace

Research helpers live under:

```bash
liverkg research evaluate
liverkg research ablate
liverkg research leakage-check
```

The previous five-fold ablation runner is intentionally not exposed in the public CLI.
