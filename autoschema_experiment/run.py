"""python -m autoschema_experiment.run --help"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
from .common import digest, load_articles, read_jsonl, write_jsonl
from .event_extraction import extract
from .concept_induction import induce, objects_from_events
from .concept_induction.induce import validate_objects
from .evaluation.review import export_review
from .llm import Model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path(__file__).parent / "data/abstracts.jsonl")
    p.add_argument("--output", type=Path, required=True, help="New directory; never overwrite an existing run")
    p.add_argument("--stage", choices=["all", "events", "concepts"], default="all")
    p.add_argument("--objects", type=Path, help="For independent concepts stage: existing Entity/Event/Relation JSONL")
    p.add_argument("--replay", type=Path, help="Replay saved raw requests; no API calls")
    p.add_argument("--env-file", type=Path, help="Read existing pipeline .env without copying credentials")
    p.add_argument("--model-role", choices=["auxiliary", "extraction"], default="auxiliary")
    p.add_argument("--model", help="Override existing second_llm_model_id")
    args = p.parse_args(argv)
    if args.stage == "concepts" and not args.objects:
        p.error("--objects required for concepts stage")
    if args.stage != "concepts" and args.objects:
        p.error("--objects applies only to concepts stage")
    # Validate input before creating run artifacts or making any remote call.
    articles = load_articles(args.input) if args.stage != "concepts" else []
    objects = read_jsonl(args.objects) if args.objects else []
    validate_objects(objects)
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "run.log")])
    events, relations, concepts, rejected, failures = [], [], [], [], []
    model = None
    started = datetime.now(timezone.utc).isoformat()
    try:
        model = Model(args.output, replay=args.replay, env_file=args.env_file, model=args.model, model_role=args.model_role)
        for a in articles:
            try:
                es, rs, rejects = extract(a, model)
                events.extend(es); relations.extend(rs); rejected.extend(rejects)
                if args.stage == "all":
                    cs, rejects = induce(objects_from_events(es, rs), model, a["pmid"])
                    concepts.extend(cs); rejected.extend(rejects)
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                failures.append({"source_pmid": a["pmid"], "error_type": type(exc).__name__, "message": str(exc)})
                logging.error("PMID %s failed: %s", a["pmid"], type(exc).__name__)
        if args.stage == "concepts":
            for pmid in sorted({o["source_pmid"] for o in objects}):
                cs, rejects = induce([o for o in objects if o["source_pmid"] == pmid], model, pmid)
                concepts.extend(cs); rejected.extend(rejects)
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        failures.append({"error_type": type(exc).__name__, "message": str(exc)})
    finally:
        for filename, rows in (("events", events), ("event_relations", relations), ("concepts", concepts),
                               ("rejected", rejected), ("failures", failures)):
            write_jsonl(args.output / (filename + ".jsonl"), rows)
        export_review(args.output / "manual_review.csv", events, relations, concepts)
        write_jsonl(args.output / "concept_inputs.jsonl", objects or objects_from_events(events, relations))
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                             text=True, capture_output=True)
        manifest = {"schema_version": "0.1", "started_at": started, "git_base": git.stdout.strip(),
            "stage": args.stage, "mode": "replay" if args.replay else "live",
            "input_hash": digest(articles or objects), "pmids": [a["pmid"] for a in articles],
            "counts": {"events": len(events), "event_relations": len(relations), "concepts": len(concepts),
                       "rejected": len(rejected), "failures": len(failures)},
            "concept_types": dict(Counter(c["source_type"] for c in concepts)),
            "calls": model.calls if model else [], "status": "failed" if failures else "needs_review",
            "manual_review_completed": False, "neo4j_access": False}
        (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(json.dumps(manifest["counts"]))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
