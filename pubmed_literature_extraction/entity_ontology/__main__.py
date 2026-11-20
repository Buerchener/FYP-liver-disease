"""Run or inspect the LiverKG entity ontology and literature graph pipeline."""

import argparse
import json
import os
from pathlib import Path
import sys

from . import DEFAULT_RUNS_DIR
from .model import DEFAULT_CONFIG

PORT = 4320


def confirm_load(run_id, assume_yes):
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        raise ValueError("Pass --yes to load without an interactive confirmation.")
    answer = input(f"Type the run ID ({run_id}) to load it into Neo4j: ").strip()
    return answer == run_id


def main(argv=None):
    parser = argparse.ArgumentParser(prog="liverkg ontology", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup-extractor", help="Show how to install the in-repository extractor")
    run = commands.add_parser("run", help="Extract, build the ontology, project relations and export the graph")
    run.add_argument("--articles", required=True, type=Path, help="PubMed title/abstract JSONL")
    run.add_argument("--run-id", required=True, help="Unique name for this batch and its outputs")
    run.add_argument("--config", type=Path, default=Path(os.environ.get("ENTITY_MODEL_CONFIG", str(DEFAULT_CONFIG))))
    run.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    run.add_argument("--title")
    run.add_argument("--knn-k", type=int, default=2)
    run.add_argument("--no-coverage", action="store_true", help="Skip the measurement-phrase check")
    run.add_argument("--no-go-reference", action="store_true", help="Skip GO lookup for unresolved eligible pairs")
    run.add_argument("--v2-data", type=Path, help="v2 data package directory (default: the repository's data/)")
    serve = commands.add_parser("serve", help="Open the read-only workbench for completed batches")
    serve.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    serve.add_argument("--port", type=int, default=PORT)
    serve.add_argument("--open", action="store_true", help="Open the workbench in a browser")
    load = commands.add_parser("load", help="Load a completed run's AUTO relations into a local Neo4j database")
    load.add_argument("--run-id", required=True)
    load.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    load.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    load.add_argument("--database", required=True, help="Target database; liver-kg-core-v02 is refused")
    load.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    load.add_argument("--dry-run", action="store_true", help="Print what would be loaded without connecting")
    load.add_argument("--yes", action="store_true", help="Skip the interactive confirmation")
    args = parser.parse_args(argv)

    if args.command == "setup-extractor":
        print("The LiverKG extractor ships in this repository. From pubmed_literature_extraction/ run:\n"
              "  python -m pip install -e '.[ontology]'")
        return 0

    if args.command == "serve":
        from .workbench.server import serve as serve_workbench

        serve_workbench(args.runs_dir, port=args.port, open_browser=args.open)
        return 0

    if args.command == "load":
        from .export.neo4j_load import check_target, load_run
        from .workbench.server import batch_manifest

        try:
            check_target(args.uri, args.database)
            folder, manifest = batch_manifest(Path(args.runs_dir).resolve(), args.run_id)
            if manifest.get("status") != "complete":
                raise ValueError("Only completed runs can be loaded.")
            if not args.dry_run and not confirm_load(args.run_id, args.yes):
                parser.exit(1, "Load cancelled.\n")
            summary = load_run(folder, uri=args.uri, database=args.database, user=args.user,
                               password=os.environ.get("NEO4J_PASSWORD", ""), dry_run=args.dry_run)
        except Exception as error:  # Neo4j driver errors are reported, not raised.
            if not isinstance(error, (OSError, ValueError, KeyError)) and \
                    not type(error).__module__.startswith("neo4j"):
                raise
            parser.exit(1, f"Load failed: {error}\n")
        print(json.dumps(summary, ensure_ascii=False))
        return 0

    from .pipeline import run_batch

    try:
        result = run_batch(
            args.articles, args.run_id, config=args.config, runs_dir=args.runs_dir,
            title=args.title, knn_k=args.knn_k, go_reference=not args.no_go_reference,
            entity_coverage=not args.no_coverage, v2_data=args.v2_data,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Entity ontology run failed: {error}\n")
    print(json.dumps(result["counts"], ensure_ascii=False))
    print(f"Results: {args.runs_dir / args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
