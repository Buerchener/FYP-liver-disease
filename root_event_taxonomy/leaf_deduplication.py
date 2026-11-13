"""Deduplicate leaf concepts and preserve the original concept and event records."""

import argparse
from collections import defaultdict
from pathlib import Path

from event_taxonomy.adapters.common import require_text
from event_taxonomy.concepts.registry import conservative_key
from event_taxonomy.induction.storage import atomic_json, read_json
from event_taxonomy.models import stable_id
from event_taxonomy.preparation.sources import SourceFile


def index_records(rows, id_field):
    if not isinstance(rows, list):
        raise ValueError(f"Expected an array of records with {id_field}.")
    indexed = {}
    for row in rows:
        record_id = require_text(row[id_field], id_field)
        if record_id in indexed:
            raise ValueError(f"Duplicate {id_field}: {record_id}")
        indexed[record_id] = row
    return indexed


def validate_leaf_links(dataset):
    leaves = index_records(dataset["leaf_concepts"], "concept_id")
    sources = index_records(dataset["source_concepts"], "source_concept_id")
    events = index_records(dataset["source_events"], "source_id")
    owners = {}
    for leaf_id, leaf in leaves.items():
        require_text(leaf["concept_name"], "Leaf name")
        require_text(leaf["description"], "Leaf definition")
        members, event_ids = leaf["source_concepts"], leaf["source_events"]
        if not isinstance(members, list) or not members or not isinstance(event_ids, list):
            raise ValueError(f"Invalid source links for {leaf_id}.")
        expected_events = set()
        for member in members:
            if member not in sources or member in owners:
                raise ValueError(f"Unknown or repeated source concept: {member}")
            owners[member] = leaf_id
            expected_events.add(sources[member]["source_event"])
        if (len(event_ids) != len(expected_events) or expected_events != set(event_ids)
                or not expected_events <= events.keys()):
            raise ValueError(f"Event links do not match the source concepts for {leaf_id}.")
    if owners.keys() != sources.keys():
        raise ValueError("Every source concept must be assigned to a leaf.")
    return leaves, owners


def review_holds_by_pair(rows):
    if not isinstance(rows, list):
        raise ValueError("Review holds must be an array.")
    holds = {}
    for row in rows:
        ids = row["source_concept_ids"]
        if not isinstance(ids, list) or len(ids) != 2 or any(not isinstance(value, str) for value in ids):
            raise ValueError("Each review hold needs two source concept IDs.")
        pair = frozenset(ids)
        if len(pair) != 2 or pair in holds:
            raise ValueError("Repeated ID or pair in review holds.")
        require_text(row["reason"], "Review hold reason")
        holds[pair] = row
    return holds


def read_review_holds(path):
    data = read_json(path)
    rows = data["deduplication"]["review_holds"] if "deduplication" in data else data["holds"]
    return review_holds_by_pair(rows)


def read_decisions(path):
    data = read_json(path)
    if "deduplication" not in data:
        return data
    steps = saved_review_steps(data)
    if len(steps) != 1:
        raise ValueError("Use leaf_deduplication --review-from to replay an export with multiple review steps.")
    return steps[0]["decisions"]


def saved_review_steps(dataset):
    return dataset["deduplication"]["decision_steps"]


def reviewed_groups(leaves, owners, decisions):
    if not isinstance(decisions, dict) or decisions.keys() - {"merges", "renames"}:
        raise ValueError("Deduplication decisions must contain merges or renames.")
    used_sources = set()
    for action, minimum in (("merges", 2), ("renames", 1)):
        rows = decisions.get(action, [])
        if not isinstance(rows, list):
            raise ValueError(f"{action} must be an array.")
        for row in rows:
            members = row["source_concept_ids"]
            if (not isinstance(members, list) or len(members) < minimum
                    or any(not isinstance(member, str) for member in members)
                    or len(set(members)) != len(members)):
                raise ValueError(f"Invalid source concept IDs in {action}.")
            members = set(members)
            if not members <= owners.keys() or members & used_sources:
                raise ValueError("Deduplication decisions contain unknown or overlapping source concept IDs.")
            used_sources.update(members)
            leaf_ids = list(dict.fromkeys(owners[member] for member in row["source_concept_ids"]))
            group = [leaves[leaf_id] for leaf_id in leaf_ids]
            if {member for leaf in group for member in leaf["source_concepts"]} != members:
                raise ValueError("A decision must cover every source concept in each affected leaf.")
            require_text(row["concept_name"], "Reviewed concept name")
            require_text(row["reason"], "Deduplication decision reason")
            if action == "merges":
                require_text(row["description"], "Reviewed concept definition")
            elif len(group) != 1:
                raise ValueError("A rename must refer to exactly one leaf.")
            yield action, row, group


def merge_leaves(group, name, description, reason):
    members = sorted(member for leaf in group for member in leaf["source_concepts"])
    merged = {
        "concept_id": stable_id("LC", members),
        "concept_name": name,
        "description": description,
        "source_concepts": members,
        "source_events": sorted({event for leaf in group for event in leaf["source_events"]}),
    }
    previous_ids = [leaf["concept_id"] for leaf in group]
    record = {"concept_id": merged["concept_id"], "previous_leaf_ids": previous_ids, "reason": reason}
    return merged, record


def apply_replacements(leaves, replacements):
    return list({leaf["concept_id"]: leaf for leaf in
                 (replacements.get(row["concept_id"], row) for row in leaves)}.values())


def deduplicate_leaves(dataset, decisions=None, *, holds=None):
    """Apply explicit decisions, consolidate exact duplicates and list unresolved names."""
    leaves, owners = validate_leaf_links(dataset)
    previous = dataset.get("deduplication", {})
    decisions = {} if decisions is None else decisions
    if holds is None:
        holds = review_holds_by_pair(previous.get("review_holds", []))
    if any(not pair <= owners.keys() for pair in holds):
        raise ValueError("Review holds refer to unknown source concept IDs.")
    if any(len({owners[member] for member in pair}) == 1 for pair in holds):
        raise ValueError("A review hold is already violated by the input leaves.")

    def is_held(group):
        members = {member for leaf in group for member in leaf["source_concepts"]}
        return any(pair <= members for pair in holds)

    replacements, merges, renames = {}, [], []
    for action, decision, group in reviewed_groups(leaves, owners, decisions):
        name, reason = decision["concept_name"], decision["reason"]
        if action == "merges":
            if is_held(group):
                raise ValueError(f"A review hold prevents merging {name}.")
            definition = decision["description"]
            if len(group) == 1 and (group[0]["concept_name"], group[0]["description"]) == (name, definition):
                continue
            result, record = merge_leaves(group, name, definition, reason)
            merges.append(record)
            replacements.update(dict.fromkeys(record["previous_leaf_ids"], result))
        else:
            leaf, = group
            if leaf["concept_name"] == name:
                continue
            replacements[leaf["concept_id"]] = {**leaf, "concept_name": name}
            renames.append({"concept_id": leaf["concept_id"], "previous_name": leaf["concept_name"],
                            "concept_name": name, "reason": reason})
    reviewed = apply_replacements(leaves.values(), replacements)
    exact_groups = defaultdict(list)
    for leaf in reviewed:
        key = (conservative_key(leaf["concept_name"]), conservative_key(leaf["description"]))
        exact_groups[key].append(leaf)
    replacements = {}
    for group in exact_groups.values():
        if len(group) == 1 or is_held(group):
            continue
        first = group[0]
        result, record = merge_leaves(
            group, first["concept_name"], first["description"],
            "Same name and definition after Unicode and whitespace normalization.",
        )
        merges.append(record)
        replacements.update(dict.fromkeys(record["previous_leaf_ids"], result))

    output = apply_replacements(reviewed, replacements)
    names = defaultdict(list)
    for leaf in output:
        names[conservative_key(leaf["concept_name"])].append(leaf["concept_id"])
    unresolved = [{"concept_name": name, "concept_ids": ids}
                  for name, ids in names.items() if len(ids) > 1]
    hold_records = list(holds.values())
    if (previous and not merges and not renames and hold_records == previous["review_holds"]
            and unresolved == previous["unresolved_names"]):
        return dataset
    steps = list(saved_review_steps(dataset)) if previous else []
    if not previous or decisions or hold_records != previous["review_holds"]:
        steps.append({"decisions": decisions, "review_holds": hold_records})
    result = {
        **dataset, "leaf_concepts": output,
        "deduplication": {
            "version": 1, "input_leaf_count": previous.get("input_leaf_count", len(leaves)),
            "leaf_count": len(output), "decision_steps": steps, "review_holds": hold_records,
            "merges": [*previous.get("merges", []), *merges],
            "renames": [*previous.get("renames", []), *renames],
            "unresolved_names": unresolved,
        },
    }
    validate_leaf_links(result)
    return result


def replay_deduplication(dataset, steps):
    if not isinstance(steps, list) or not steps:
        raise ValueError("Expected at least one saved deduplication step.")
    for step in steps:
        holds = review_holds_by_pair(step["review_holds"])
        dataset = deduplicate_leaves(dataset, step["decisions"], holds=holds)
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing leaf_concepts.json")
    parser.add_argument("--output", type=Path, required=True, help="Write a deduplicated leaf_concepts.json")
    decisions_group = parser.add_mutually_exclusive_group()
    decisions_group.add_argument("--decisions", type=Path, help="Reviewed merges and renames, or a single-step export")
    decisions_group.add_argument("--review-from", type=Path, help="Replay review steps from a previous output")
    parser.add_argument("--review-holds", type=Path, help="Pairs that must remain separate, or a previous output")
    args = parser.parse_args(argv)
    try:
        paths = (args.input, args.decisions, args.review_holds, args.review_from)
        if any(args.output.resolve() == path.resolve() for path in paths if path is not None):
            raise ValueError("--output must be separate from the input and review files.")
        source = SourceFile.read(args.input)
        dataset = source.json()
        if args.review_from:
            if args.review_holds:
                raise ValueError("Replay already uses the review holds recorded with each step.")
            steps = saved_review_steps(read_json(args.review_from))
            result = replay_deduplication(dataset, steps)
        else:
            decisions = read_decisions(args.decisions) if args.decisions else None
            holds = read_review_holds(args.review_holds) if args.review_holds else None
            result = deduplicate_leaves(dataset, decisions, holds=holds)
        source.assert_unchanged()
        atomic_json(args.output, result)
        audit = result["deduplication"]
        unresolved = audit["unresolved_names"]
        print(f'Leaf concepts: {len(dataset["leaf_concepts"])} -> {audit["leaf_count"]}. '
              f"Unresolved names: {len(unresolved)}.")
        for row in unresolved:
            print(f'{row["concept_name"]}: {", ".join(row["concept_ids"])}')
        return 1 if unresolved else 0
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
