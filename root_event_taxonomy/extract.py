"""Extract evidence-bearing Event records and write the next model's input."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import re

from event_taxonomy.adapters.common import require_text
from event_taxonomy.induction.contracts import BudgetExceeded, ContractError, ProviderError
from event_taxonomy.induction.journal import LLMJournal
from event_taxonomy.induction.providers import OpenAICompatibleProvider
from event_taxonomy.induction.storage import atomic_json, read_json, run_lock
from event_taxonomy.preparation.sources import SourceFile
from pubmed_literature_extraction.cognitive_agent.evidence_units import ArticleEvidenceReader

from .local import LocalMLXProvider, local_path


PROMPT = Path(__file__).parent / "prompts" / "event-extraction.xml"


def load_abstracts(source):
    articles, seen = [], set()
    for line, row in source.jsonl():
        if not isinstance(row, dict):
            raise ValueError(f"Input line {line} must be a JSON object.")
        pmid = require_text(row.get("pmid"), f"PMID at line {line}")
        if not pmid.isascii() or not pmid.isdecimal() or pmid in seen:
            raise ValueError(f"PMID at line {line} must be numeric and unique.")
        seen.add(pmid)
        # Preserve exact source whitespace for evidence checks.
        passage = require_text(row.get("abstract"), f"Abstract at line {line}")
        articles.append((pmid, passage))
    if not articles:
        raise ValueError("Input JSONL contains no abstracts.")
    return articles


def check_events(result, pmid, passage, sentences):
    if not isinstance(result, dict) or set(result) != {"events"} or not isinstance(result["events"], list):
        raise ContractError("Expected an object with one events array.")
    positions = {row.parent_sentence_id: i for i, row in enumerate(sentences)}
    accepted, rejected = [], []
    for index, row in enumerate(result["events"], 1):
        record_id = f"pmid:{pmid}:E{index}"
        try:
            if not isinstance(row, dict) or set(row) != {"Event", "Sentence_IDs", "Trigger"}:
                raise ValueError("expected_event_sentence_ids_trigger")
            event = require_text(row["Event"], "Event").strip()
            trigger = require_text(row["Trigger"], "Trigger").strip()
            ids = row["Sentence_IDs"]
            if not isinstance(ids, list) or not ids or not all(isinstance(s, str) and s in positions for s in ids):
                raise ValueError("unknown_or_missing_sentence_ids")
            indices = [positions[s] for s in ids]
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError("evidence_sentences_not_consecutive")
            evidence = passage[sentences[indices[0]].char_start:sentences[indices[-1]].char_end]
            left = r"(?<!\w)" if re.match(r"\w", trigger[0]) else ""
            right = r"(?!\w)" if re.match(r"\w", trigger[-1]) else ""
            if not re.search(left + re.escape(trigger) + right, evidence):
                raise ValueError("trigger_not_verbatim_span_in_evidence")
            if trigger.rstrip(".!?") == evidence.rstrip(".!?") and len(trigger.split()) > 1:
                raise ValueError("trigger_copies_entire_evidence")
        except ValueError as error:
            reason = str(error)
        else:
            accepted.append({"source_id": record_id, "source_pmid": pmid, "Event": event,
                             "Evidence": evidence, "Trigger": trigger})
            continue
        rejected.append({"source_id": record_id, "reason": reason, "record": row})
    return accepted, rejected


def extract(articles, output, source, provider, prompt, parameters):
    journal = LLMJournal(output / "llm", provider, {"extract": prompt}, {"extract": parameters}, len(articles))
    events, rejected, articles_report = [], [], []
    reader = ArticleEvidenceReader()
    for pmid, passage in articles:
        try:
            sentences = reader.parent_units(passage, reader.read(passage))
            if not sentences:
                raise ContractError("No source sentences found.")
            result = journal.call("extract", {"pmid": pmid, "sentences": [
                {"id": s.parent_sentence_id, "section": s.section, "text": s.text} for s in sentences
            ]})
            accepted, invalid = check_events(result, pmid, passage, sentences)
            events.extend(accepted)
            rejected.extend(invalid)
            articles_report.append({
                "pmid": pmid, "accepted": len(accepted), "rejected": len(invalid),
                "decision_ref": journal.last_ref,
            })
        except (BudgetExceeded, ContractError, ProviderError) as error:
            articles_report.append({"pmid": pmid, "error": str(error), "decision_ref": journal.last_ref})

    atomic_json(output / "events.json", events)
    atomic_json(output / "concept_inputs.json", {"objects": [{
        "source_id": row["source_id"],
        "source_label": row["Event"],
        "context": row["Evidence"],
        "trigger": row["Trigger"],
    } for row in events]})
    atomic_json(output / "rejected.json", rejected)
    failures = sum("error" in row for row in articles_report)
    report = {
        "source_sha256": source.sha256,
        "articles": articles_report,
        "events_ready_for_conceptualization": len(events),
        "rejected_event_records": len(rejected),
        "failed_articles": failures,
    }
    atomic_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL with string pmid and abstract fields")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--model-config", type=Path, required=True, help="Extraction provider configuration")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        if args.max_tokens < 1:
            raise ValueError("--max-tokens must be positive.")
        source = SourceFile.read(args.input)
        articles = load_abstracts(source)
        settings = read_json(args.model_config)["llm"]["provider"]
        if settings["kind"] == "openai_compatible":
            model = require_text(settings.get("model"), "API model")
            key_env = require_text(settings.get("api_key_env"), "API key environment variable name")
            if not os.environ.get(key_env):
                raise ValueError(f"Set the API key environment variable {key_env} before running extraction.")
            base_url = require_text(settings.get("base_url"), "API base URL")
            provider_context = nullcontext(OpenAICompatibleProvider({
                **settings, "api_key_env": key_env, "base_url": base_url,
                "timeout_seconds": settings.get("timeout_seconds", 150),
            }))
        elif settings["kind"] == "local_mlx":
            config = {**settings, "model_path": local_path(settings, "model_path"),
                      "python": local_path(settings, "python")}
            model = Path(config["model_path"]).name
            provider_context = LocalMLXProvider(config, args.output)
        else:
            raise ValueError("The extraction command expects an openai_compatible or local_mlx provider.")
        parameters = {"model": model, "temperature": 0, "max_tokens": args.max_tokens}
        prompt = PROMPT.read_text(encoding="utf-8")
        args.output.mkdir(parents=True, exist_ok=False)
        with run_lock(args.output), provider_context as provider:
            report = extract(articles, args.output, source, provider, prompt, parameters)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Events: {report['events_ready_for_conceptualization']}; "
          f"rejected: {report['rejected_event_records']}; "
          f"failed abstracts: {report['failed_articles']}")
    print(args.output.resolve() / "concept_inputs.json")
    return 1 if report["failed_articles"] or report["rejected_event_records"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
