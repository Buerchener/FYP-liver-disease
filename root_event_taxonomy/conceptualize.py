"""Conceptualize XML extraction output using the configured API."""
import argparse
import os
from pathlib import Path

from event_taxonomy.adapters.common import require_text
from event_taxonomy.induction.contracts import ContractError, ProviderError
from event_taxonomy.induction.journal import LLMJournal
from event_taxonomy.induction.providers import OpenAICompatibleProvider
from event_taxonomy.induction.storage import atomic_json, read_json, run_lock
from event_taxonomy.preparation.sources import SourceFile

PROMPT = Path(__file__).parent / "prompts" / "event-conceptualization.txt"
INPUT_FIELDS = {"source_id", "source_label", "context", "trigger"}
OUTPUT_FIELDS = {"source_id", "concept_name", "description"}
PLACEHOLDERS = {"null", "none", "n/a", "not applicable", "undefined", "-"}


def load_objects(source, pmids=()):
    objects = source.json()["objects"]
    if not isinstance(objects, list) or not objects:
        raise ValueError("Expected a nonempty objects array.")
    seen, selected = set(), []
    prefixes = tuple(f"pmid:{pmid}:" for pmid in pmids)
    for row in objects:
        if not isinstance(row, dict) or set(row) != INPUT_FIELDS:
            raise ValueError("Expected source_id, source_label, context and trigger.")
        for key in INPUT_FIELDS:
            require_text(row[key], key)
        if row["source_id"] in seen:
            raise ValueError("Repeated source_id: " + row["source_id"])
        seen.add(row["source_id"])
        if not prefixes or row["source_id"].startswith(prefixes):
            selected.append({"source_id": row["source_id"], "event": row["source_label"],
                             "trigger": row["trigger"], "source_sentence": row["context"]})
    if not selected:
        raise ValueError("No events match the requested PMIDs.")
    if pmids and set(pmids) != {row["source_id"].split(":")[1] for row in selected}:
        raise ValueError("One or more requested PMIDs have no events in the input.")
    return selected


def check(result, batch):
    if not isinstance(result, dict) or set(result) != {"concepts"} or not isinstance(result["concepts"], list):
        raise ContractError("Expected one concepts array.")
    wanted = {o["source_id"] for o in batch}
    seen, checked, issues = set(), {}, {}
    for row in result["concepts"]:
        if (not isinstance(row, dict) or set(row) != OUTPUT_FIELDS
                or not isinstance(row["source_id"], str) or row["source_id"] not in wanted
                or row["source_id"] in seen):
            raise ContractError("Invalid or repeated source_id, or invalid concept record.")
        source_id = row["source_id"]
        seen.add(source_id)
        if row["concept_name"] is not None or row["description"] is not None:
            try:
                row = {"source_id": source_id,
                       "concept_name": require_text(row["concept_name"], "concept_name").strip(),
                       "description": require_text(row["description"], "description").strip()}
                if any(row[field].casefold() in PLACEHOLDERS for field in ("concept_name", "description")):
                    raise ValueError("A name and description must contain real text, or both be JSON null.")
            except ValueError as error:
                issues[source_id] = str(error)
                continue
        checked[source_id] = row
    issues.update({source_id: "Missing event record." for source_id in wanted - seen})
    return checked, issues


def conceptualize(objects, directory, provider, parameters, batch_size, source_sha256):
    batches = [objects[i:i + batch_size] for i in range(0, len(objects), batch_size)]
    journal = LLMJournal(directory / "llm", provider, {"conceptualize": PROMPT.read_text(encoding="utf-8")},
                         {"conceptualize": parameters}, 2 * len(batches))
    concepts, completed, failed = {}, [], []
    for batch in batches:
        pending = batch
        feedback = None
        for attempt in (1, 2):
            payload = {"objects": pending}
            if feedback:
                payload["validation_feedback"] = feedback
            try:
                result, issues = check(journal.call("conceptualize", payload), pending)
            except (ContractError, ProviderError) as error:
                feedback = str(error)
            else:
                concepts.update(result)
                if result:
                    completed.append({"source_ids": list(result), "decision_ref": journal.last_ref})
                if not issues:
                    break
                pending = [o for o in pending if o["source_id"] in issues]
                feedback = "; ".join(f"{source_id}: {reason}" for source_id, reason in issues.items())
            if attempt == 2:
                failed.append({"source_ids": [o["source_id"] for o in pending],
                               "reason": feedback, "decision_ref": journal.last_ref})
    concepts = [concepts[o["source_id"]] for o in objects if o["source_id"] in concepts]
    atomic_json(directory / "concepts.json", {"concepts": concepts})
    report = {"source_sha256": source_sha256, "input_events": len(objects),
              "labeled_events": sum(c["concept_name"] is not None for c in concepts),
              "abstained_events": sum(c["concept_name"] is None for c in concepts),
              "completed_batches": completed, "failed_batches": failed}
    atomic_json(directory / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--pmid", action="append", default=[], help="Only process this PMID; repeat for multiple papers.")
    args = parser.parse_args(argv)
    try:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive.")
        if any(not (pmid.isascii() and pmid.isdecimal()) for pmid in args.pmid):
            raise ValueError("--pmid requires a numeric PMID.")
        source = SourceFile.read(args.input)
        objects = load_objects(source, args.pmid)
        settings = read_json(args.model_config)["llm"]["provider"]
        if settings["kind"] != "openai_compatible":
            raise ValueError("Conceptualization expects an openai_compatible provider.")
        key_env = require_text(settings.get("api_key_env"), "API key environment variable name")
        if not os.environ.get(key_env):
            raise ValueError(f"Set {key_env} before running conceptualization.")
        require_text(settings.get("base_url"), "API base URL")
        parameters = {"model": require_text(settings.get("model"), "API model"),
                      "temperature": 0, "max_tokens": 12000}
        provider = OpenAICompatibleProvider({**settings, "timeout_seconds": settings.get("timeout_seconds", 150)})
        args.output.mkdir(parents=True, exist_ok=False)
        with run_lock(args.output):
            report = conceptualize(objects, args.output, provider, parameters, args.batch_size, source.sha256)
        print(args.output.resolve() / "report.json")
        return 1 if report["failed_batches"] else 0
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
