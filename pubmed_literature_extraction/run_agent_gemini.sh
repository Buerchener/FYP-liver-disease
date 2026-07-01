#!/bin/bash
# Backward-compatible Gemini launcher.
#
# Historical versions of this file rewrote source files in place. The current
# version is intentionally a thin wrapper around run_cognitive_agent.sh.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

export GEMINI_MODEL="${GEMINI_MODEL:-[按次]gemini-2.5-flash}"
export GEMINI_API_BASE="${GEMINI_API_BASE:-https://new.bitexingai.com}"

exec "$PROJECT_ROOT/run_cognitive_agent.sh" "$@"
