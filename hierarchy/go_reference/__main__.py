import argparse
import json
import logging
import sys
from pathlib import Path

from . import ReferenceAgent


def main():
    parser = argparse.ArgumentParser(description="Consult GO for a single LiverKG concept; read JSON from a file or stdin.")
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--model-config", type=Path, help="Project Qwen configuration (or GO_REF_MODEL_CONFIG)")
    parser.add_argument("--mode", choices=["auto", "mcp", "rest"], default="auto")
    parser.add_argument("--max-tool-calls", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=15, help="OLS request timeout in seconds")
    parser.add_argument("--model-timeout", type=float, help="Override the model configuration timeout")
    parser.add_argument("--state-dir", type=Path, default=Path("go_reference/runs"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        data = json.loads(args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read())
        agent = ReferenceAgent.live(model_config=args.model_config, mode=args.mode, timeout=args.timeout,
                                    model_timeout=args.model_timeout, max_tool_calls=args.max_tool_calls,
                                    state_dir=args.state_dir)
        print(json.dumps(agent.run(data), ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Input, configuration or local I/O error: {exc}\n")


if __name__ == "__main__":
    main()
