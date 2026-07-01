#!/bin/bash
# Cognitive Agent launcher for the liver disease KG project.
#
# Usage:
#   ./run_cognitive_agent.sh                    # dry run, 50 articles
#   ./run_cognitive_agent.sh 100                # dry run, 100 articles
#   ./run_cognitive_agent.sh 5 --write-neo4j    # small Neo4j write test

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

API_KEY="${GEMINI_API_KEY:-${LLM_API_KEY:-}}"
MODEL_ID="${GEMINI_MODEL:-[按次]gemini-2.5-flash}"
API_BASE="${GEMINI_API_BASE:-https://new.bitexingai.com}"
NEO4J_PASSWORD_VALUE="${NEO4J_PASSWORD:-}"

LIMIT="50"
WRITE_FLAG="--skip-neo4j-write"

for arg in "$@"; do
    case "$arg" in
        --write-neo4j)
            WRITE_FLAG="--write-neo4j"
            ;;
        --skip-neo4j-write)
            WRITE_FLAG="--skip-neo4j-write"
            ;;
        ''|*[!0-9]*)
            echo "[ERROR] Unknown argument: $arg"
            echo "        Usage: ./run_cognitive_agent.sh [limit] [--write-neo4j]"
            exit 1
            ;;
        *)
            LIMIT="$arg"
            ;;
    esac
done

INPUT="${COGNITIVE_AGENT_INPUT:-extraction_output/pubmed_converted_500.jsonl}"
RUN_ID="${COGNITIVE_AGENT_RUN_ID:-agent_${LIMIT}_$(date +%Y%m%d_%H%M%S)}"

if [ -z "$API_KEY" ]; then
    echo "[ERROR] GEMINI_API_KEY or LLM_API_KEY is required."
    echo "        Configure it in workstreams/literature_hmdb_kegg/.env or export it in this shell."
    exit 1
fi

"$PYTHON_BIN" - <<'PY' >/dev/null 2>&1 || {
import sys

if sys.version_info < (3, 10):
    raise SystemExit(1)

import langextract  # noqa: F401
import google.genai  # noqa: F401
from neo4j import GraphDatabase  # noqa: F401
PY
    echo "[ERROR] Python environment is missing required Cognitive Agent dependencies."
    echo "        Python: $PYTHON_BIN"
    echo "        Setup:"
    echo "          python3.12 -m venv .venv-cognitive"
    echo "          source .venv-cognitive/bin/activate"
    echo "          python -m pip install -r requirements-cognitive-agent.txt"
    echo "          export PYTHON_BIN=\"$PROJECT_ROOT/.venv-cognitive/bin/python\""
    exit 1
}

if [ "$WRITE_FLAG" = "--write-neo4j" ] && [ -z "$NEO4J_PASSWORD_VALUE" ]; then
    echo "[ERROR] --write-neo4j requires NEO4J_PASSWORD."
    exit 1
fi

echo "============================================================"
echo "  Cognitive Agent - Liver Disease KG"
echo "  Root:    $PROJECT_ROOT"
echo "  Input:   $INPUT"
echo "  Limit:   $LIMIT"
echo "  Run ID:  $RUN_ID"
echo "  Python:  $PYTHON_BIN"
echo "  Model:   $MODEL_ID"
echo "  Neo4j:   $([ -n "$NEO4J_PASSWORD_VALUE" ] && echo CONNECTED || echo OFFLINE)"
echo "  Write:   $WRITE_FLAG"
echo "============================================================"
echo ""

"$PYTHON_BIN" -m cognitive_agent.agent \
    --input "$INPUT" \
    --limit "$LIMIT" \
    --run-id "$RUN_ID" \
    --api-key "$API_KEY" \
    --api-base "$API_BASE" \
    --model-id "$MODEL_ID" \
    --neo4j-password "$NEO4J_PASSWORD_VALUE" \
    "$WRITE_FLAG" \
    --reflection-interval "${REFLECTION_INTERVAL:-10}"

echo ""
echo "Done. Results -> extraction_output/agent_results_${RUN_ID}.json"
echo "      Report  -> extraction_output/agent_report_${RUN_ID}.json"
