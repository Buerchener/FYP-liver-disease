#!/bin/zsh
set -eu

PROJECT_ROOT="/Users/a1234/Desktop/FYP/github_upload/FYP-liver-disease/pubmed_literature_extraction"
PYTHON_BIN="/Users/a1234/Desktop/FYP/.venv-ppt-master/bin/python"
RUN_ID="${1:?run id is required}"
RUN_DIR="$PROJECT_ROOT/benchmark_output/$RUN_ID"

mkdir -p "$RUN_DIR"
cd "$PROJECT_ROOT"
set -a
source "$PROJECT_ROOT/workstreams/literature_hmdb_kegg/.env"
set +a

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/benchmark_three_extractors.py" \
  --provider openai \
  --model-id deepseek-chat \
  --api-base https://api.deepseek.com \
  --run-id "$RUN_ID" \
  --max-workers 10 \
  --input-price-per-million 0.14 \
  --output-price-per-million 0.28 \
  > "$RUN_DIR/background.log" 2>&1
