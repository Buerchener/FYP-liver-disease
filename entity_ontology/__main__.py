"""Run or inspect the LiverKG entity ontology pipeline."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import DEFAULT_RUNS_DIR
from .model import DEFAULT_CONFIG

PORT = 4320


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup-extractor", help="Install the pinned classmate extractor")
    run = commands.add_parser("run", help="Extract entities and build a candidate is_a hierarchy")
    run.add_argument("--articles", required=True, type=Path, help="PubMed title/abstract JSONL")
    run.add_argument("--run-id", required=True, help="Unique name for this batch and its outputs")
    run.add_argument("--config", type=Path, default=Path(os.environ.get("ENTITY_MODEL_CONFIG", str(DEFAULT_CONFIG))))
    run.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    run.add_argument("--title")
    run.add_argument("--knn-k", type=int, default=2)
    run.add_argument("--no-coverage", action="store_true", help="Skip the measurement-phrase check")
    run.add_argument("--no-go-reference", action="store_true", help="Skip GO lookup for unresolved eligible pairs")
    serve = commands.add_parser("serve", help="Open the read-only workbench for completed batches")
    serve.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    serve.add_argument("--port", type=int, default=PORT)
    serve.add_argument("--open", action="store_true", help="Open the workbench in a browser")
    args = parser.parse_args(argv)

    if args.command == "setup-extractor":
        from .extraction import EXTRACTOR_CACHE, REPOSITORY_ROOT, archive_source

        source = archive_source(REPOSITORY_ROOT / EXTRACTOR_CACHE, REPOSITORY_ROOT)
        return subprocess.call([sys.executable, "-m", "pip", "install", str(source)])

    if args.command == "serve":
        from .workbench import serve as serve_workbench

        serve_workbench(args.runs_dir, port=args.port, open_browser=args.open)
        return 0

    from .pipeline import run_batch

    try:
        result = run_batch(
            args.articles, args.run_id, config=args.config, runs_dir=args.runs_dir,
            title=args.title, knn_k=args.knn_k, go_reference=not args.no_go_reference,
            entity_coverage=not args.no_coverage,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Entity ontology run failed: {error}\n")
    print(json.dumps(result["counts"], ensure_ascii=False))
    print(f"Results: {args.runs_dir / args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
