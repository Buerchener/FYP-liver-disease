"""Normalize equivalent event types before intermediate concept induction."""
import argparse
from collections import Counter
from itertools import combinations
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from event_taxonomy.adapters.common import require_text
from event_taxonomy.induction.contracts import ContractError, ProviderError, obj, require, text
from event_taxonomy.induction.graph import digest
from event_taxonomy.induction.journal import LLMJournal
from event_taxonomy.induction.providers import OpenAICompatibleProvider
from event_taxonomy.induction.storage import atomic_json, read_json, run_lock
from event_taxonomy.models import stable_id
from event_taxonomy.preparation.sources import SourceFile

from .conceptualize import PLACEHOLDERS
from .embedding import EmbeddingAPI, Embeddings
from .leaf_containment import check_scope_decisions
from .leaf_dataset import import_events, induction_input
from .leaf_deduplication import deduplicate_leaves, read_decisions, read_review_holds

PROMPT_DIR = Path(__file__).parent / "prompts"
PAIR_RELATIONS = {"equivalent", "broader", "narrower", "related_but_different", "uncertain"}
VALIDATION_STAGE = "equivalence"
NAMING_STAGE = "canonicalization"
EQUIVALENCE_PROMPTS = {"baseline": "leaf-equivalence.txt", "xml": "leaf-equivalence.xml",
                       "xml-containment": "leaf-equivalence-containment.xml"}


def read_inputs(concept_source, event_source):
    events = event_source.json()
    if not isinstance(events, list):
        raise ValueError("Expected an array of extracted events.")
    event_by_id = {}
    for row in events:
        source_id = require_text(row["source_id"], "Event source_id")
        pmid = require_text(row["source_pmid"], "PMID")
        if source_id in event_by_id:
            raise ValueError(f"Duplicate event ID: {source_id}")
        if not pmid.isascii() or not pmid.isdecimal():
            raise ValueError(f"Invalid PMID for {source_id}.")
        for field in ("Event", "Trigger", "Evidence"):
            require_text(row[field], field)
        event_by_id[source_id] = row
    rows = concept_source.json()["concepts"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty concepts array.")
    concepts, unassigned, seen = [], [], set()
    for index, row in enumerate(rows):
        source_id = row["source_id"]
        if not isinstance(source_id, str) or source_id not in event_by_id or source_id in seen:
            raise ValueError(f"Unresolved or duplicate event ID at concept index {index}.")
        seen.add(source_id)
        name, description = row["concept_name"], row["description"]
        if name is None and description is None:
            unassigned.append({"source_event": source_id, "status": "no_input_concept"})
            continue
        if any(require_text(value, field).strip().casefold() in PLACEHOLDERS
               for field, value in (("concept_name", name), ("description", description))):
            raise ValueError(f"Invalid concept placeholder for {source_id}.")
        concepts.append({"source_concept_id": stable_id("source-concept", source_id, name, description),
                         "source_event": source_id, "concept_name": name, "description": description,
                         "record_locator": f"/concepts/{index}"})
    unassigned.extend({"source_event": source_id, "status": "no_conceptualization_record"}
                      for source_id in event_by_id if source_id not in seen)
    if not concepts:
        raise ValueError("No defined concepts to cluster.")
    return concepts, events, unassigned


def concept_texts(concepts):
    return {f"C{i}": f'{row["concept_name"]}\n{row["description"]}'
            for i, row in enumerate(concepts, 1)}


def cluster_candidates(ids, vectors, threshold):
    """Cut a complete-linkage dendrogram at the configured cosine distance."""
    if len(ids) == 1:
        return [(tuple(ids), 0.0)]
    distances = np.clip(pdist(vectors, metric="cosine"), 0.0, 2.0)
    tree = linkage(distances, method="complete")
    labels = fcluster(tree, t=threshold, criterion="distance")
    members = {}
    for node_id, label in zip(ids, labels):
        members.setdefault(int(label), []).append(node_id)
    # With complete linkage, the largest distance within a flat cluster equals
    # the height of its final merge. Track that height without a second matrix.
    heights = dict.fromkeys(members, 0.0)
    tree_labels = list(labels)
    for left, right, height, _ in tree:
        left_label, right_label = tree_labels[int(left)], tree_labels[int(right)]
        label = left_label if left_label == right_label else None
        tree_labels.append(label)
        if label is not None:
            heights[label] = float(height)
    return [(tuple(group), heights[label]) for label, group in members.items()]


def check_decisions(response, pairs):
    obj(response, {"decisions"})
    require(isinstance(response["decisions"], list), "decisions must be an array")
    expected, checked = set(pairs), {}
    for row in response["decisions"]:
        obj(row, {"left_id", "right_id", "relation", "reason", "needs_evidence"})
        text(row["left_id"])
        text(row["right_id"])
        text(row["relation"])
        text(row["reason"])
        pair = (row["left_id"], row["right_id"])
        require(pair in expected and pair not in checked, "Unknown, reversed or duplicate pair")
        require(row["relation"] in PAIR_RELATIONS, "Unknown relation")
        require(type(row["needs_evidence"]) is bool, "needs_evidence must be boolean")
        require(not row["needs_evidence"] or row["relation"] == "uncertain",
                "Only an uncertain decision can request evidence")
        checked[pair] = row
    require(checked.keys() == expected, "Return a decision for every requested pair")
    return checked


def check_names(response, group_ids):
    obj(response, {"concepts"})
    require(isinstance(response["concepts"], list), "concepts must be an array")
    expected, checked = set(group_ids), {}
    for row in response["concepts"]:
        obj(row, {"group_id", "concept_name", "description"})
        group_id = row["group_id"]
        text(group_id)
        require(group_id in expected and group_id not in checked, "Unknown or duplicate group_id")
        if row["concept_name"] is None and row["description"] is None:
            checked[group_id] = None
            continue
        for field in ("concept_name", "description"):
            text(row[field])
            require(row[field].strip().casefold() not in PLACEHOLDERS, "Name and description need real text")
        checked[group_id] = (row["concept_name"].strip(), row["description"].strip())
    require(checked.keys() == expected, "Return a name and description or null for every group")
    return checked


def call_checked(journal, stage, payload, validate, issues):
    for attempt in (1, 2):
        try:
            return validate(journal.call(stage, payload)), journal.last_ref
        except (ContractError, ProviderError) as error:
            issues.append({"stage": stage, "attempt": attempt, "reason": str(error),
                           "decision_ref": journal.last_ref})
            payload = {**payload, "validation_feedback": str(error)}
    return None, journal.last_ref


def equivalent_groups(members, decisions):
    """Build disjoint groups in input order; equivalence must hold for every pair."""
    # Conflicting votes (A=B, A=C, B!=C) are not used to pick an arbitrary merge.
    neighbors = {member: set() for member in members}
    for (left, right), decision in decisions.items():
        if decision["relation"] == "equivalent":
            neighbors[left].add(right)
            neighbors[right].add(left)
    groups, conflicts, seen = [], [], set()
    for member in members:
        if member in seen:
            continue
        connected, pending = {member}, [member]
        while pending:
            adjacent = neighbors[pending.pop()] - connected
            pending.extend(adjacent)
            connected.update(adjacent)
        component = tuple(node for node in members if node in connected)
        seen.update(component)
        if all(len(neighbors[node]) == len(component) - 1 for node in component):
            groups.append(component)
        else:
            conflicts.append(component)
            groups.extend((node,) for node in component)
    return groups, conflicts


def pair_batches(clusters, pairs_per_request):
    batch, count = [], 0
    for cluster_id, pairs in clusters.items():
        for start in range(0, len(pairs), pairs_per_request):
            chunk = pairs[start:start + pairs_per_request]
            if count + len(chunk) > pairs_per_request:
                yield batch
                batch, count = [], 0
            batch.append((cluster_id, chunk))
            count += len(chunk)
    if batch:
        yield batch


def review_pairs(requested, concept_by_id, event_by_id, journal, issues, pairs_per_request, *,
                 with_evidence=False, stage=VALIDATION_STAGE, validate=check_decisions):
    results = {}
    for batch in pair_batches(requested, pairs_per_request):
        entries = []
        for cluster_id, pairs in batch:
            member_ids = dict.fromkeys(node for pair in pairs for node in pair)
            members = []
            for node in member_ids:
                concept = concept_by_id[node]
                row = {"id": node, "concept_name": concept["concept_name"], "description": concept["description"]}
                if with_evidence:
                    source = event_by_id[concept["source_event"]]
                    row.update({field.lower(): source[field] for field in ("Event", "Trigger", "Evidence")})
                members.append(row)
            entries.append({"cluster_id": cluster_id, "concepts": members,
                            "pairs": [{"left_id": left, "right_id": right} for left, right in pairs]})
        all_pairs = [pair for _, pairs in batch for pair in pairs]
        response, ref = call_checked(journal, stage, {"clusters": entries},
                                     lambda value: validate(value, all_pairs), issues)
        for left, right in all_pairs:
            pair = (left, right)
            if response is None:
                decision = {"left_id": left, "right_id": right, "relation": "uncertain",
                            "reason": "No valid API decision.", "needs_evidence": False,
                            "status": "response_unavailable"}
            else:
                decision = {**response[pair], "status": "judged"}
            results[pair] = {**decision, "decision_ref": ref}
    return results


def candidate_pairs(clusters):
    return {c["cluster_id"]: list(combinations(c["members"], 2)) for c in clusters if len(c["members"]) > 1}


def validate_clusters(clusters, concept_by_id, event_by_id, journal, issues, pairs_per_request, *,
                      validate=check_decisions):
    requested = candidate_pairs(clusters)
    initial = review_pairs(requested, concept_by_id, event_by_id, journal, issues, pairs_per_request,
                           validate=validate)
    return [{"cluster_id": cluster_id,
             "left_source_concept_id": concept_by_id[left]["source_concept_id"],
             "right_source_concept_id": concept_by_id[right]["source_concept_id"],
             "initial": initial[left, right], "final": initial[left, right]}
            for cluster_id, pairs in requested.items() for left, right in pairs]


def partition_validated(clusters, relations):
    by_pair = {(row["final"]["left_id"], row["final"]["right_id"]): row["final"] for row in relations}
    groups, conflicts = [], []
    for cluster in clusters:
        cluster_id, members = cluster["cluster_id"], cluster["members"]
        decisions = {pair: by_pair[pair] for pair in combinations(members, 2)}
        equivalents, inconsistent = equivalent_groups(members, decisions)
        groups.extend({"cluster_id": cluster_id, "members": group} for group in equivalents)
        conflicts.extend({"cluster_id": cluster_id, "members": group} for group in inconsistent)
    for i, group in enumerate(groups, 1):
        group["group_id"] = f"G{i}"
    return groups, conflicts


def with_review_decisions(relations, reviewed_pairs):
    """Apply explicit review holds while preserving each model judgment."""
    for row in relations:
        key = frozenset((row["left_source_concept_id"], row["right_source_concept_id"]))
        if key not in reviewed_pairs:
            continue
        note = reviewed_pairs[key]
        row["review"] = note
        row["final"] = {**row["final"], "relation": "uncertain", "needs_evidence": False,
                        "reason": note["reason"], "status": "review_hold"}


def canonicalize(groups, concept_by_id, journal, issues):
    proposed_merges = [group for group in groups if len(group["members"]) > 1]
    names, refs = {}, {}
    for start in range(0, len(proposed_merges), 6):
        batch = proposed_merges[start:start + 6]
        payload = {"groups": [
            {"group_id": group["group_id"], "concepts": [
                {key: concept_by_id[member][key] for key in ("concept_name", "description")}
                for member in group["members"]]}
            for group in batch
        ]}
        ids = [group["group_id"] for group in batch]
        result, ref = call_checked(journal, NAMING_STAGE, payload,
                                   lambda value: check_names(value, ids), issues)
        names.update(result or dict.fromkeys(ids))
        refs.update(dict.fromkeys(ids, ref))

    return build_leaves(groups, concept_by_id, names, refs)


def build_leaves(groups, concept_by_id, names, refs):
    leaves, group_records = [], []
    for group in groups:
        group_id, members = group["group_id"], group["members"]
        canonical = names.get(group_id)
        if canonical:
            accepted = [(members, canonical)]
            status = "merged"
        else:
            accepted = [((member,), (concept_by_id[member]["concept_name"], concept_by_id[member]["description"]))
                        for member in members]
            status = "singleton" if len(members) == 1 else "canonicalization_unresolved"
        leaf_ids = []
        for current, (name, description) in accepted:
            source_concepts = [concept_by_id[member]["source_concept_id"] for member in current]
            source_events = [concept_by_id[member]["source_event"] for member in current]
            leaf_id = stable_id("LC", sorted(source_concepts))
            leaf_ids.append(leaf_id)
            leaves.append({"concept_id": leaf_id, "concept_name": name, "description": description,
                           "source_concepts": source_concepts, "source_events": source_events})
        group_records.append({**group, "status": status, "leaf_concept_ids": leaf_ids,
                              "canonicalization_ref": refs.get(group_id)})
    return leaves, group_records


def reuse_embeddings(directory, sources, config, texts):
    """Check provenance and exact input texts before reusing a previous matrix."""
    manifest = read_json(directory / "manifest.json")
    for key, source in zip(("concepts", "events"), sources):
        if manifest["inputs"][key]["sha256"] != source.sha256:
            raise ValueError("Embedding source checksums differ from the current inputs.")
    previous = manifest["config"]["embedding"]
    if any(previous[k] != config[k] for k in ("model", "dimensions")):
        raise ValueError("Embedding model or dimensions differ.")
    matrix = np.load(directory / "concept_embeddings.npy", allow_pickle=False)
    if matrix.shape != (len(texts), config["dimensions"]) or not np.isfinite(matrix).all():
        raise ValueError("Invalid reused embedding matrix.")
    by_id = {}
    for path in sorted((directory / "embeddings").glob("*.json")):
        batch = read_json(path)
        if batch["status"] != "response_received" or batch["model"] != config["model"]:
            raise ValueError("Reused embedding response is incomplete or from another model.")
        if len(batch["input"]) != len(batch["vectors"]):
            raise ValueError("Reused embedding batch is missing input or vector records.")
        for item, vector in zip(batch["input"], batch["vectors"]):
            if item["concept_id"] in by_id:
                raise ValueError("Duplicate ID in reused embedding records.")
            by_id[item["concept_id"]] = (item["text"], vector)
    if set(by_id) != set(texts):
        raise ValueError("Reused embeddings do not cover the current inputs.")
    for index, (node, expected) in enumerate(texts.items()):
        actual, vector = by_id[node]
        if actual != expected or not np.array_equal(matrix[index], vector):
            raise ValueError("Reused embedding does not match its input or response.")
    return matrix


def run(directory, inputs, sources, config, prompts, previous_embeddings=None, imported=None, roots=None, holds=None,
        *, validate=check_decisions, dedup_decisions=None):
    concepts, events, unassigned = inputs
    holds = holds or {}
    source_ids = {concept["source_concept_id"] for concept in concepts}
    if any(not pair <= source_ids for pair in holds):
        raise ValueError("Review holds refer to unknown source concept IDs.")
    embedding_config = config["embedding"]
    embeddings = Embeddings(EmbeddingAPI(embedding_config), embedding_config, None, directory / "embeddings")
    concept_by_id = {f"C{i}": concept for i, concept in enumerate(concepts, 1)}
    texts = concept_texts(concepts)
    if previous_embeddings is None:
        vectors = embeddings.for_texts(texts)
    else:
        vectors = reuse_embeddings(previous_embeddings, sources, embedding_config, texts)
    np.save(directory / "concept_embeddings.npy", vectors, allow_pickle=False)
    pairs = cluster_candidates(list(concept_by_id), vectors, config["distance_threshold"])
    clusters = [{"cluster_id": f"K{i}", "members": group, "max_cosine_distance": distance}
                for i, (group, distance) in enumerate(pairs, 1)]
    cluster_report = {"distance": "cosine", "linkage": "complete", "distance_threshold": config["distance_threshold"],
                      "source_concepts_by_id": concept_by_id, "clusters": clusters}
    atomic_json(directory / "candidate_clusters.json", cluster_report)
    sizes = Counter(len(cluster["members"]) for cluster in clusters)
    print(f"Found {len(clusters)} candidate clusters for {len(concepts)} concepts, {sizes[1]} singletons.",
          flush=True)

    max_pairs = sum(len(c["members"]) * (len(c["members"]) - 1) // 2 for c in clusters)
    provider = OpenAICompatibleProvider(config["llm"]["provider"])
    journal = LLMJournal(directory / "llm", provider, prompts,
                         dict.fromkeys(prompts, config["llm"]["parameters"]), 2 * max_pairs + 2 * len(concepts))
    issues = []
    event_by_id = {event["source_id"]: event for event in events}
    relations = validate_clusters(
        clusters, concept_by_id, event_by_id, journal, issues, config["pairs_per_request"], validate=validate)
    with_review_decisions(relations, holds)
    groups, conflicts = partition_validated(clusters, relations)
    atomic_json(directory / "validation.json", {"relations": relations, "conflicts": conflicts, "groups": groups,
                                                "issues": list(issues)})
    leaves, group_records = canonicalize(groups, concept_by_id, journal, issues)
    atomic_json(directory / "canonicalization.json", {"groups": group_records,
                                                     "issues": [issue for issue in issues if issue["stage"] == NAMING_STAGE]})

    leaf_data = deduplicate_leaves(
        {"leaf_concepts": leaves, "source_concepts": concepts,
         "source_events": events, "unassigned_events": unassigned},
        dedup_decisions, holds=holds,
    )
    leaves = leaf_data["leaf_concepts"]
    deduplication = leaf_data["deduplication"]
    for source in sources:
        source.assert_unchanged()
    atomic_json(directory / "leaf_concepts.json", leaf_data)
    if imported is not None:
        deduplicated_ids = {row["concept_id"] for action in ("merges", "renames")
                            for row in deduplication[action]}
        dataset = induction_input(leaves, concepts, *sources[:3], imported, roots,
                                  deduplicated_ids=deduplicated_ids)
        atomic_json(directory / "induction_input.json", dataset)
    relation_counts = Counter(row["final"]["relation"] for row in relations)
    api_failures = sum(row["final"]["status"] == "response_unavailable" for row in relations)
    unresolved = sum(row["status"] == "canonicalization_unresolved" for row in group_records)
    logs = [read_json(path) for path in (directory / "llm").glob("*.json")]
    unresolved_names = deduplication["unresolved_names"]
    report = {"status": "partial" if api_failures or unresolved or unresolved_names else "completed",
              "input_events": len(events), "source_concepts": len(concepts), "unassigned_events": len(unassigned),
              "candidate_clusters": len(clusters), "candidate_cluster_sizes": dict(sorted(sizes.items())),
              "pair_relations": dict(relation_counts), "pairs_without_valid_response": api_failures,
              "review_holds": sum(row["final"]["status"] == "review_hold" for row in relations),
              "pairs_reviewed_with_evidence": 0,
              "conflicting_equivalence_groups": len(conflicts), "canonicalization_unresolved": unresolved,
              "merged_leaf_concepts": sum(len(leaf["source_concepts"]) > 1 for leaf in leaves),
              "unresolved_names": unresolved_names,
              "leaf_concepts": len(leaves), "embedding_model": embeddings.returned_model or embedding_config["model"],
              "embedding_dimensions": int(vectors.shape[1]), "embedding_calls": embeddings.batches,
              "embeddings_reused_from": str(previous_embeddings) if previous_embeddings else None,
              "induction_input": "induction_input.json" if imported is not None else None,
              "llm_models": sorted({row["response"]["model"] for row in logs if "response" in row}),
              "llm_calls": dict(Counter(row["request"]["stage"] for row in logs)),
              "distance_threshold": config["distance_threshold"], "issues": issues}
    atomic_json(directory / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts", type=Path, required=True, help="Conceptualization concepts.json")
    parser.add_argument("--events", type=Path, required=True, help="Matching extraction events.json")
    parser.add_argument("--abstracts", type=Path, help="Original abstract JSONL, to also export induction_input.json")
    parser.add_argument("--model-config", type=Path, required=True, help="Existing API configuration")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--embedding-model", default="text-embedding-v4")
    parser.add_argument("--embedding-dimensions", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=10)
    parser.add_argument("--distance-threshold", type=float, default=0.35)
    parser.add_argument("--pairs-per-request", type=int, default=16)
    parser.add_argument("--equivalence-prompt", choices=tuple(EQUIVALENCE_PROMPTS), default="baseline",
                        help="First-round prompt version")
    parser.add_argument("--reuse-embeddings", type=Path, help="Run directory with matching embedding inputs and vectors")
    parser.add_argument("--review-holds", type=Path, help="Optional JSON with pairs that must remain separate for review")
    parser.add_argument("--dedup-decisions", type=Path, help="Reviewed merges and renames, or a prior leaf export")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.distance_threshold) or not 0 <= args.distance_threshold <= 2:
            raise ValueError("--distance-threshold must be between 0 and 2.")
        if min(args.embedding_dimensions, args.embedding_batch_size, args.pairs_per_request) < 1:
            raise ValueError("Dimensions and batch sizes must be positive.")
        sources = [SourceFile.read(args.concepts), SourceFile.read(args.events)]
        inputs = read_inputs(*sources)
        imported, roots = None, None
        if args.abstracts:
            abstracts = SourceFile.read(args.abstracts)
            root_source = SourceFile.read(PROMPT_DIR.parents[1] / "event_taxonomy/configs/roots.v0_1.json")
            imported = import_events(abstracts, sources[1], inputs[1])
            roots = root_source.json()
            sources.extend([abstracts, root_source])
        settings = read_json(args.model_config)["llm"]["provider"]
        if settings["kind"] != "openai_compatible":
            raise ValueError("Concept clustering requires an OpenAI-compatible API configuration.")
        base_url = require_text(settings.get("base_url"), "API base URL")
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Configure an HTTPS API URL without credentials or query parameters.")
        key_env = require_text(settings.get("api_key_env"), "API key environment variable name")
        if not os.environ.get(key_env):
            raise ValueError(f"Set the API key environment variable {key_env} before running.")
        model = require_text(settings.get("model"), "API model")
        connection = {"base_url": base_url, "api_key_env": key_env,
                      "timeout_seconds": settings.get("timeout_seconds", 150)}
        config = {"distance_threshold": args.distance_threshold, "pairs_per_request": args.pairs_per_request,
                  "embedding": {**connection, "model": args.embedding_model, "dimensions": args.embedding_dimensions,
                                "batch_size": args.embedding_batch_size},
                  "llm": {"provider": {**connection, "extra_body": settings.get("extra_body", {})},
                          "parameters": {"model": model, "temperature": 0, "max_tokens": 8000}}}
        prompt_files = {VALIDATION_STAGE: EQUIVALENCE_PROMPTS[args.equivalence_prompt],
                        NAMING_STAGE: "leaf-canonicalization.txt"}
        response_format = "containment" if args.equivalence_prompt == "xml-containment" else "relation"
        prompts = {name: (PROMPT_DIR / filename).read_text(encoding="utf-8")
                   for name, filename in prompt_files.items()}
        previous = args.reuse_embeddings.resolve() if args.reuse_embeddings else None
        holds = read_review_holds(args.review_holds) if args.review_holds else {}
        decisions = read_decisions(args.dedup_decisions) if args.dedup_decisions else None
        args.output.mkdir(parents=True, exist_ok=False)
        with run_lock(args.output):
            atomic_json(args.output / "manifest.json", {
                "inputs": {label: {"path": str(source.path), "sha256": source.sha256}
                           for label, source in zip(("concepts", "events", "abstracts", "roots"), sources)},
                "config": config,
                "embedding_source": str(previous) if previous else None,
                "review_holds": list(holds.values()),
                "dedup_decisions": decisions or {},
                "prompt_files": prompt_files,
                "equivalence_response_format": response_format,
                "prompt_sha256": {name: digest(prompt) for name, prompt in prompts.items()}})
            validator = check_scope_decisions if response_format == "containment" else check_decisions
            report = run(args.output, inputs, sources, config, prompts, previous, imported, roots, holds,
                         validate=validator, dedup_decisions=decisions)
        print(f"Leaf concepts: {report['leaf_concepts']}. Report: {args.output.resolve() / 'report.json'}")
        return 1 if report["status"] == "partial" else 0
    except (KeyError, OSError, ProviderError, TypeError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
