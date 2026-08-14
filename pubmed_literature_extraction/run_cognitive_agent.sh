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

# Optional ignored local override for the currently active extraction endpoint.
# It is sourced after .env so the last definition wins without modifying the
# shared example or exposing credentials in process arguments.
if [ -f workstreams/literature_hmdb_kegg/active-gemini.env ]; then
    set +u
    set -a
    source workstreams/literature_hmdb_kegg/active-gemini.env
    set +a
    set -u
fi

API_KEY="${GEMINI_API_KEY:-${LLM_API_KEY:-}}"
MODEL_ID="${GEMINI_MODEL:-[按次]gemini-2.5-flash}"
API_BASE="${GEMINI_API_BASE:-https://new.bitexingai.com}"
NEO4J_PASSWORD_VALUE="${NEO4J_PASSWORD:-}"
NEO4J_URI_VALUE="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER_VALUE="${NEO4J_USER:-neo4j}"
NEO4J_DATABASE_VALUE="${NEO4J_DATABASE:-neo4j}"
SECOND_LLM_ENABLED_VALUE="${SECOND_LLM_ENABLED:-false}"
SECOND_LLM_ACTIVE="DISABLED"
SECOND_LLM_ARGS=(
    --second-llm-provider "${SECOND_LLM_PROVIDER:-openai}"
    --second-llm-mode "${SECOND_LLM_MODE:-conditional}"
    --second-llm-timeout "${SECOND_LLM_TIMEOUT:-45}"
)
case "$SECOND_LLM_ENABLED_VALUE" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        SECOND_LLM_ACTIVE="ENABLED"
        SECOND_LLM_ARGS+=(--second-llm-enabled)
        SECOND_LLM_ARGS+=(--second-llm-model-id "${SECOND_LLM_MODEL_ID:-deepseek-v4-flash}")
        SECOND_LLM_ARGS+=(--second-llm-api-base "${SECOND_LLM_API_BASE:-https://api.deepseek.com}")
        ;;
esac
NEO4J_RAG_ENABLED_VALUE="${NEO4J_RAG_ENABLED:-false}"
NEO4J_RAG_ACTIVE="DISABLED"
RAG_ARGS=(
    --rag-max-entities "${RAG_MAX_ENTITIES:-12}"
    --rag-max-candidates-per-entity "${RAG_MAX_CANDIDATES_PER_ENTITY:-3}"
    --rag-max-total-candidates "${RAG_MAX_TOTAL_CANDIDATES:-20}"
    --rag-max-neighbors-per-candidate "${RAG_MAX_NEIGHBORS_PER_CANDIDATE:-4}"
    --rag-max-evidence-chars "${RAG_MAX_EVIDENCE_CHARS:-500}"
    --rag-max-total-chars "${RAG_MAX_TOTAL_CHARS:-6000}"
)
case "$NEO4J_RAG_ENABLED_VALUE" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        NEO4J_RAG_ACTIVE="ENABLED (read-only)"
        RAG_ARGS+=(--neo4j-rag-enabled)
        ;;
esac

LIMIT="50"
MAX_WORKERS_VALUE="${MAX_WORKERS:-5}"
WRITE_FLAG="--skip-neo4j-write"

for arg in "$@"; do
    case "$arg" in
        --write-neo4j)
            WRITE_FLAG="--write-neo4j"
            ;;
        --skip-neo4j-write)
            WRITE_FLAG="--skip-neo4j-write"
            ;;
        --max-workers)
            shift
            MAX_WORKERS_VALUE="${1:?--max-workers requires a value}"
            ;;
        --max-workers=*)
            MAX_WORKERS_VALUE="${arg#*=}"
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

if [ "$WRITE_FLAG" = "--write-neo4j" ]; then
    case "$NEO4J_URI_VALUE" in
        bolt://localhost:*|bolt://127.0.0.1:*|neo4j://localhost:*|neo4j://127.0.0.1:*) ;;
        *)
            echo "[ERROR] --write-neo4j is restricted to a localhost Neo4j URI."
            exit 1
            ;;
    esac
    if [ "$NEO4J_DATABASE_VALUE" != "neo4j" ]; then
        echo "[ERROR] --write-neo4j is restricted to database 'neo4j'."
        exit 1
    fi
fi

# Pass credentials through the inherited environment, not command-line
# arguments, so they are not exposed by process listings.
export GEMINI_API_KEY="$API_KEY"
export NEO4J_PASSWORD="$NEO4J_PASSWORD_VALUE"
export NEO4J_URI="$NEO4J_URI_VALUE"
export NEO4J_USER="$NEO4J_USER_VALUE"
export NEO4J_DATABASE="$NEO4J_DATABASE_VALUE"

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
echo "  Second:  $SECOND_LLM_ACTIVE"
echo "  RAG:     $NEO4J_RAG_ACTIVE"
echo "  Agent:   ${AGENT_MODE:-precision}"
echo "  Router:  ${ROUTER_EXECUTION_MODE:-legacy}"
echo "  Agent v2:${AGENT_EXECUTION_MODE:-legacy} (${AGENT_BUDGET_PROFILE:-quality})"
echo "  Cache:   ${EXTRACTION_CACHE_MODE:-memory}"
echo "============================================================"
echo ""

"$PYTHON_BIN" -m cognitive_agent.agent \
    --input "$INPUT" \
    --limit "$LIMIT" \
    --run-id "$RUN_ID" \
    --api-base "$API_BASE" \
    --model-id "$MODEL_ID" \
    "$WRITE_FLAG" \
    "${SECOND_LLM_ARGS[@]}" \
    "${RAG_ARGS[@]}" \
    --agent-mode "${AGENT_MODE:-precision}" \
    --router-execution-mode "${ROUTER_EXECUTION_MODE:-legacy}" \
    --extraction-cache-mode "${EXTRACTION_CACHE_MODE:-memory}" \
    --extraction-cache-path "${EXTRACTION_CACHE_PATH:-.cache/langextract_candidates.sqlite3}" \
    --extraction-cache-memory-entries "${EXTRACTION_CACHE_MEMORY_ENTRIES:-256}" \
    --extraction-cache-max-entries "${EXTRACTION_CACHE_MAX_ENTRIES:-2000}" \
    --extraction-cache-max-mb "${EXTRACTION_CACHE_MAX_MB:-200}" \
    --extraction-cache-ttl-days "${EXTRACTION_CACHE_TTL_DAYS:-30}" \
    --reflection-interval "${REFLECTION_INTERVAL:-10}" \
    --max-workers "$MAX_WORKERS_VALUE" \
    --extraction-inner-max-workers "${EXTRACTION_INNER_MAX_WORKERS:-2}"

echo ""
echo "Done. Results -> extraction_output/agent_results_${RUN_ID}.json"
echo "      Report  -> extraction_output/agent_report_${RUN_ID}.json"
