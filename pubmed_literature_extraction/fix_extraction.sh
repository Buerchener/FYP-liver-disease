#!/bin/bash
# Deprecated one-off repair script.
#
# The Document -> [doc] LangExtract fix and KGMemory query cleanup have already
# been merged into the current codebase. This file is kept only as a historical
# pointer so old notes do not lead to source-rewriting shell snippets.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "[INFO] fix_extraction.sh is deprecated; no files were changed."
echo "[INFO] Current project root: $PROJECT_ROOT"
echo "[INFO] Run a dry probe with:"
echo "       ./run_cognitive_agent.sh 5"
