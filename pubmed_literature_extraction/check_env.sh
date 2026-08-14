#!/bin/bash
# Quick local environment check before running the Cognitive Agent.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -x "$PROJECT_ROOT/.venv-cognitive/bin/python" ]; then
        PYTHON_BIN="$PROJECT_ROOT/.venv-cognitive/bin/python"
    elif command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3.12)"
    else
        PYTHON_BIN="$(command -v python3)"
    fi
fi

if [ -f workstreams/literature_hmdb_kegg/.env ]; then
    set +u
    set -a
    source workstreams/literature_hmdb_kegg/.env
    set +a
    set -u
fi

echo "Cognitive Agent - Environment Check"
echo "==================================="
echo "Project root: $PROJECT_ROOT"
echo "Python bin:   $PYTHON_BIN"
echo ""

echo -n "Python 3.10+: "
"$PYTHON_BIN" -c "import sys; assert sys.version_info >= (3, 10); print('OK', sys.version.split()[0])" 2>&1 || echo "FAIL"

echo -n "LangExtract import: "
"$PYTHON_BIN" -c "import langextract; print('OK')" 2>&1 || echo "FAIL - install requirements-cognitive-agent.txt in this Python environment"

echo -n "Neo4j driver import: "
"$PYTHON_BIN" -c "from neo4j import GraphDatabase; print('OK')" 2>&1 || echo "FAIL - install requirements-cognitive-agent.txt in this Python environment"

echo -n "Gemini API key: "
if [ -n "${GEMINI_API_KEY:-${LLM_API_KEY:-}}" ]; then
    echo "OK"
else
    echo "MISSING - set GEMINI_API_KEY or LLM_API_KEY"
fi

echo -n "Neo4j password: "
if [ -n "${NEO4J_PASSWORD:-}" ]; then
    echo "SET"
else
    echo "MISSING - offline dry runs are still allowed"
fi

if [ -n "${NEO4J_PASSWORD:-}" ]; then
    echo -n "Neo4j connectivity: "
    "$PYTHON_BIN" - <<'PY'
import os
from urllib.parse import urlparse

try:
    from neo4j import GraphDatabase
except Exception as exc:
    print(f"SKIP - neo4j driver is not importable: {exc}")
    raise SystemExit(0)

uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
user = os.environ.get("NEO4J_USER", "neo4j")
password = os.environ["NEO4J_PASSWORD"]

try:
    if urlparse(uri).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Neo4j connections are restricted to localhost")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    driver.close()
    print("OK")
except Exception as exc:
    print(f"FAIL - {exc}")
PY
fi

echo -n "Default input file: "
if [ -f extraction_output/pubmed_converted_500.jsonl ]; then
    count="$(wc -l < extraction_output/pubmed_converted_500.jsonl | tr -d ' ')"
    echo "OK ($count records)"
else
    echo "MISSING - expected extraction_output/pubmed_converted_500.jsonl"
fi

echo ""
echo "Run a dry probe with:"
echo "  export PYTHON_BIN=\"$PROJECT_ROOT/.venv-cognitive/bin/python\""
echo "  ./run_cognitive_agent.sh 5"
