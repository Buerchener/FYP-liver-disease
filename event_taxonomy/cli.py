import argparse
from pathlib import Path
from .pipeline import prepare


def main():
    parser = argparse.ArgumentParser(description="Independent Event Taxonomy preparation and recursive induction.")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare", help="Import saved results and prepare an initial DAG")
    command.add_argument("--config", type=Path,
                         default=Path(__file__).parent / "configs/task1.workbench.json")
    command.add_argument("--source-root", type=Path, help="Override data source location")
    command.add_argument("--output-parent", type=Path, help="Override output parent; never overwrites a run")
    command.add_argument("--run-id", help="Optional unique run directory name")
    induce = sub.add_parser("induce", help="Run recursive induction; defaults to Synthetic Data + Mock LLM")
    induce.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/task2.mock.json")
    induce.add_argument("--output-parent", type=Path)
    induce.add_argument("--run-id")
    induce.add_argument("--live", action="store_true", help="Explicitly enable real model API calls")
    induce.add_argument("--step-limit", type=int, help="Pause after this many durable transitions")
    resume = sub.add_parser("resume", help="Resume a frozen induction run")
    resume.add_argument("--run", type=Path, required=True)
    resume.add_argument("--live", action="store_true")
    resume.add_argument("--step-limit", type=int)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            output = prepare(args.config, args.source_root, args.output_parent, args.run_id)
        elif args.command == 'induce':
            from .induction.run import create_run
            output = create_run(args.config, args.output_parent, args.run_id, args.live, args.step_limit)
        else:
            from .induction.run import resume_run
            output = resume_run(args.run, args.live, args.step_limit)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, "Experiment failed: " + str(error) + "\n")
    print("Output: " + str(output))
    print("Report: " + str(output / ("report.md" if args.command == 'prepare' else 'preservation_report.json')))
    if args.command != 'prepare':
        print('Visualization: ' + str(output / 'visualization.html'))


if __name__ == "__main__":
    main()
