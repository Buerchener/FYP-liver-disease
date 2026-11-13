"""Review frozen first-pass decisions using source evidence in a separate run."""
import argparse
from collections import Counter
from itertools import combinations
import os
from pathlib import Path
from urllib.parse import urlsplit

from event_taxonomy.adapters.common import require_text
from event_taxonomy.induction.graph import digest
from event_taxonomy.induction.journal import LLMJournal, strict_json
from event_taxonomy.induction.providers import OpenAICompatibleProvider
from event_taxonomy.induction.storage import atomic_json, read_json, run_lock
from event_taxonomy.preparation.sources import SourceFile

from .leaf_concepts import (
    PROMPT_DIR, VALIDATION_STAGE, candidate_pairs, check_decisions, partition_validated,
    review_pairs, with_review_decisions,
)
from .leaf_containment import check_scope_decisions

STAGE = "second-validation"
PROMPT_FILE = PROMPT_DIR / "leaf-second-validation.xml"
def load_baseline(directory):
    sources = {name: SourceFile.read(directory / name) for name in
               ("manifest.json", "candidate_clusters.json", "validation.json", "leaf_concepts.json")}
    manifest = sources["manifest.json"].json()
    cluster_data = sources["candidate_clusters.json"].json()
    clusters, concepts = cluster_data["clusters"], cluster_data["source_concepts_by_id"]
    events = {row["source_id"]: row for row in sources["leaf_concepts.json"].json()["source_events"]}
    members = [member for group in clusters for member in group["members"]]
    if len(members) != len(concepts) or set(members) != concepts.keys():
        raise ValueError("Baseline clusters must partition the source concepts.")
    if len({group["cluster_id"] for group in clusters}) != len(clusters):
        raise ValueError("Duplicate baseline cluster ID.")
    for concept in concepts.values():
        event = events[concept["source_event"]]
        for field in ("Event", "Trigger", "Evidence"):
            require_text(event[field], field)

    requested = candidate_pairs(clusters)
    by_pair = {pair: cluster_id for cluster_id, pairs in requested.items() for pair in pairs}
    first_pass, holds, journals = {}, {}, {}
    prompt_hash = manifest["prompt_sha256"][VALIDATION_STAGE]
    parameters = manifest["config"]["llm"]["parameters"]
    validator = check_scope_decisions if manifest.get("equivalence_response_format") == "containment" else check_decisions
    for row in sources["validation.json"].json()["relations"]:
        first = row["initial"]
        pair = (first["left_id"], first["right_id"])
        if pair in first_pass or by_pair.get(pair) != row["cluster_id"]:
            raise ValueError("Baseline pair IDs do not match the frozen clusters.")
        source_ids = [concepts[node]["source_concept_id"] for node in pair]
        if source_ids != [row["left_source_concept_id"], row["right_source_concept_id"]]:
            raise ValueError("Baseline decision refers to a different source concept.")
        if "review" in row:
            holds[frozenset(source_ids)] = row["review"]
        if first["status"] != "judged":
            raise ValueError("Finish the first-pass API requests before comparing.")
        ref = first["decision_ref"]
        if ref not in journals:
            record = SourceFile.read(directory / ref)
            sources[ref] = record
            saved = record.json()
            request, response = saved["request"], saved["response"]
            if (saved["status"] != "response_received" or response["finish_reason"] != "stop"
                    or request["stage"] != VALIDATION_STAGE or request["parameters"] != parameters
                    or request["prompt_sha256"] != prompt_hash or digest(request["prompt"]) != prompt_hash):
                raise ValueError("Baseline request does not match its saved settings or prompt.")
            payload_clusters = request["input"]["clusters"]
            for group in payload_clusters:
                for concept in group["concepts"]:
                    expected = concepts[concept["id"]]
                    if any(concept[k] != expected[k] for k in ("concept_name", "description")):
                        raise ValueError("Baseline request has different concept text.")
            pairs = [(item["left_id"], item["right_id"])
                     for group in payload_clusters for item in group["pairs"]]
            journals[ref] = validator(strict_json(response["content"]), pairs)
        if any(first[key] != value for key, value in journals[ref][pair].items()):
            raise ValueError("Baseline decision differs from its API response.")
        first_pass[pair] = first
    if first_pass.keys() != by_pair.keys():
        raise ValueError("Baseline does not contain a first-pass decision for every candidate pair.")
    return sources, manifest, clusters, concepts, events, requested, first_pass, holds


def summarize(groups, conflicts, relations):
    return {"proposed_leaf_count": len(groups),
            "proposed_merge_groups": sum(len(g["members"]) > 1 for g in groups),
            "proposed_merge_pairs": sum(len(list(combinations(g["members"], 2))) for g in groups),
            "conflicting_components": len(conflicts),
            "relations": dict(Counter(row["final"]["relation"] for row in relations))}


def require_merge_agreement(first, second):
    """Keep any disputed or unconfirmed merge available for review."""
    if (first["relation"] == "equivalent") != (second["relation"] == "equivalent"):
        return {**second, "relation": "uncertain", "needs_evidence": False,
                "status": "merge_disagreement",
                "reason": "Only one round judged this pair equivalent."}
    return second


def compare(output, baseline, journal, pairs_per_request):
    sources, _, clusters, concepts, events, requested, first_pass, holds = baseline
    issues = []
    second = review_pairs(requested, concepts, events, journal, issues, pairs_per_request,
                          with_evidence=True, stage=STAGE, validate=check_scope_decisions)
    before, after, agreed = [], [], []
    for cluster_id, pairs in requested.items():
        for left, right in pairs:
            pair = (left, right)
            first, reviewed = first_pass[pair], second[pair]
            base = {"cluster_id": cluster_id,
                    "left_source_concept_id": concepts[left]["source_concept_id"],
                    "right_source_concept_id": concepts[right]["source_concept_id"],
                    "initial": first}
            before.append({**base, "final": first})
            final = reviewed if reviewed["status"] == "judged" else first
            after.append({**base, "second": reviewed, "final": final})
            agreed.append({**base, "second": reviewed, "final": require_merge_agreement(first, reviewed)})
    with_review_decisions(before, holds)
    with_review_decisions(after, holds)
    with_review_decisions(agreed, holds)
    before_groups, before_conflicts = partition_validated(clusters, before)
    after_groups, after_conflicts = partition_validated(clusters, after)
    agreed_groups, agreed_conflicts = partition_validated(clusters, agreed)
    changed = [{"cluster_id": a["cluster_id"], "left_id": a["initial"]["left_id"],
                "right_id": a["initial"]["right_id"], "baseline": b["final"]["relation"],
                "second": a["final"]["relation"]}
               for b, a in zip(before, after) if b["final"]["relation"] != a["final"]["relation"]]
    failures = sum(row["status"] != "judged" for row in second.values())
    for source in sources.values():
        source.assert_unchanged()
    atomic_json(output / "baseline_validation.json", {
        "relations": before, "groups": before_groups, "conflicts": before_conflicts})
    atomic_json(output / "validation.json", {
        "relations": after, "groups": after_groups, "conflicts": after_conflicts, "issues": issues})
    atomic_json(output / "consensus_validation.json", {
        "relations": agreed, "groups": agreed_groups, "conflicts": agreed_conflicts, "issues": issues})
    report = {
        "status": "partial" if failures else "completed",
        "pairs": len(first_pass),
        "pairs_without_second_decision": failures,
        "review_holds_in_both": len(holds),
        "baseline": summarize(before_groups, before_conflicts, before),
        "with_second_validation": summarize(after_groups, after_conflicts, after),
        "with_merge_agreement": summarize(agreed_groups, agreed_conflicts, agreed),
        "merge_disagreements": sum(row["final"]["status"] == "merge_disagreement" for row in agreed),
        "changed_pairs": changed,
    }
    atomic_json(output / "comparison.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Saved leaf_concepts run directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory for validation and comparison")
    args = parser.parse_args(argv)
    try:
        directory = args.baseline.resolve()
        output = args.output.resolve()
        if directory in output.parents or directory == output:
            raise ValueError("Choose an output directory outside the baseline run.")
        baseline = load_baseline(directory)
        sources, manifest, _, _, _, _, first_pass, _ = baseline
        config = manifest["config"]
        pairs_per_request = config["pairs_per_request"]
        if type(pairs_per_request) is not int or pairs_per_request < 1:
            raise ValueError("Invalid baseline pair batch size.")
        provider_config, parameters = (config["llm"][k] for k in ("provider", "parameters"))
        parsed = urlsplit(provider_config["base_url"])
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("The baseline API URL must be HTTPS without credentials or query parameters.")
        key_env = provider_config["api_key_env"]
        if first_pass and not os.environ.get(key_env):
            raise ValueError(f"Set {key_env} before running.")
        prompt = PROMPT_FILE.read_text(encoding="utf-8")
        output.mkdir(parents=True, exist_ok=False)
        with run_lock(output):
            atomic_json(output / "manifest.json", {
                "baseline_directory": str(directory),
                "baseline_files": {name: source.sha256 for name, source in sources.items()},
                "baseline_prompt_sha256": manifest["prompt_sha256"][VALIDATION_STAGE],
                "second_prompt_file": PROMPT_FILE.name,
                "second_prompt_sha256": digest(prompt),
                "parameters": parameters,
                "pairs_per_request": pairs_per_request,
            })
            journal = LLMJournal(output / "llm", OpenAICompatibleProvider(provider_config),
                                 {STAGE: prompt}, {STAGE: parameters}, 2 * len(first_pass))
            result = compare(output, baseline, journal, pairs_per_request)
        print(output / "comparison.json")
        return 1 if result["status"] == "partial" else 0
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
