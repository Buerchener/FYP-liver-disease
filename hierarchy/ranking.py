"""Rank entity concepts in bounded shards and merge them while preserving local order."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from time import sleep

from ..model import ProviderError
from .ranking_journal import ContractError, LLMJournal, strict_json
from ..storage import read_json


STAGE = "global_concept_rank"
REPAIR_STAGE = "global_concept_rank_repair"
MERGE_STAGE = "merge_concept_rank"
MERGE_SIDES_STAGE = "merge_concept_rank_sides"
RANK_SHARD_SIZE = 120
MERGE_WINDOW_SIZE = 60
MIN_MERGE_WINDOW_SIZE = 15


def rank_concepts(nodes, provider, parameters, journal_dir, *, prompts):
    """Rank Concepts by scope, merging bounded model requests for large input sets."""
    if len(nodes) <= RANK_SHARD_SIZE:
        return _rank_shard(nodes, provider, parameters, journal_dir, prompts=prompts)
    by_id = {
        node["concept_id"]: {"id": i, "name": node["concept_label"], "description": node["concept_description"]}
        for i, node in enumerate(nodes, start=1)
    }
    if len(by_id) != len(nodes):
        raise ValueError("Duplicate Concept ID in ranking input")
    by_alias = {row["id"]: concept_id for concept_id, row in by_id.items()}
    shards = [nodes[start:start + RANK_SHARD_SIZE] for start in range(0, len(nodes), RANK_SHARD_SIZE)]

    def rank_shard(item):
        index, shard = item
        ids = _rank_shard(
            shard, provider, parameters, journal_dir / f"shard-{index:06d}", prompts=prompts,
        )
        return [by_id[concept_id] for concept_id in ids]

    with ThreadPoolExecutor(max_workers=4) as pool:
        ranked = list(pool.map(rank_shard, enumerate(shards, start=1)))
        level = 0
        while len(ranked) > 1:
            level += 1

            def merge_pair(index):
                directory = journal_dir / f"merge-{level:04d}-{index:06d}"
                return _merge_rankings(
                    ranked[index], ranked[index + 1], provider, parameters, directory, prompts=prompts,
                )

            merged = list(pool.map(merge_pair, range(0, len(ranked) - 1, 2)))
            if len(ranked) % 2:
                merged.append(ranked[-1])
            ranked = merged
    return [by_alias[row["id"]] for row in ranked[0]]


def _merge_rankings(left, right, provider, parameters, directory, *, prompts):
    max_windows = (len(left) + len(right) + MIN_MERGE_WINDOW_SIZE - 1) // MIN_MERGE_WINDOW_SIZE
    prompts = {stage: prompts[stage] for stage in (MERGE_STAGE, MERGE_SIDES_STAGE)}
    journal = LLMJournal(directory, provider, prompts, dict.fromkeys(prompts, parameters),
                         maximum_calls=len(list(directory.glob("*.json"))) + 18 * max_windows)
    left_offset = right_offset = 0
    merged = []
    window_size = MERGE_WINDOW_SIZE
    while left_offset < len(left) and right_offset < len(right):
        while True:
            left_window = left[left_offset:left_offset + window_size]
            right_window = right[right_offset:right_offset + window_size]
            try:
                order = _rank_window(journal, left_window, right_window)
                break
            except ContractError:
                previous_size = max(len(left_window), len(right_window))
                if previous_size <= MIN_MERGE_WINDOW_SIZE:
                    order = _merge_fixed_sides(journal, left_window, right_window)
                    break
                next_size = max(MIN_MERGE_WINDOW_SIZE, previous_size // 2)
                print(f'Ranking merge {directory.name}: reducing invalid window from {previous_size} to {next_size} IDs per list',
                      flush=True)
                window_size = next_size
        left_ids = {row["id"] for row in left_window}
        window = {row["id"]: row for row in left_window + right_window}
        left_consumed = right_consumed = 0
        # Refill when the next element from either list is outside the model window.
        for alias in order:
            merged.append(window[alias])
            if alias in left_ids:
                left_consumed += 1
            else:
                right_consumed += 1
            if left_consumed == len(left_window) or right_consumed == len(right_window):
                break
        left_offset += left_consumed
        right_offset += right_consumed
    return merged + left[left_offset:] + right[right_offset:]


def _rank_window(journal, left, right):
    payload = {"left": left, "right": right}
    expected_left = [row["id"] for row in left]
    expected_right = [row["id"] for row in right]
    left_ids, right_ids = set(expected_left), set(expected_right)

    def validate(response):
        order = response.get("order") if isinstance(response, dict) and set(response) == {"order"} else None
        if (not isinstance(order, list) or len(order) != len(left) + len(right)
                or any(type(alias) is not int for alias in order)
                or [alias for alias in order if alias in left_ids] != expected_left
                or [alias for alias in order if alias in right_ids] != expected_right):
            raise ContractError("Merge output must interleave both lists without changing either internal order")
        return order

    saved = journal.revalidate_invalid(MERGE_STAGE, payload, validate)
    if saved is not None:
        return saved[0]
    correction = {
        "left_ids_in_order": expected_left,
        "right_ids_in_order": expected_right,
        "expected_length": len(left) + len(right),
        "instruction": "Interleave both complete ID lists. Check that every ID occurs exactly once "
                       "and both original orders are preserved before returning.",
    }
    error = None
    for request in (payload, {**payload, "output_format_correction": correction}):
        rejected = journal.invalid_reason(MERGE_STAGE, request)
        if rejected is not None:
            error = ContractError(rejected)
            continue
        for attempt in range(3):
            try:
                return validate(journal.call(MERGE_STAGE, request))
            except ContractError as failure:
                journal.mark_invalid(str(failure))
                error = failure
                # An exact reply saved just before interruption may have masked
                # a valid older correction until that exact reply was checked.
                saved = journal.revalidate_invalid(MERGE_STAGE, payload, validate)
                if saved is not None:
                    return saved[0]
                break
            except ProviderError as failure:
                if attempt == 2 or not re.search(r"HTTP status (?:429|5\d\d)\b|Transport or response-envelope", str(failure)):
                    raise
                sleep(2 ** (attempt + 1))
    raise error


def _merge_fixed_sides(journal, left, right):
    """Recover a window without letting the model reorder either input list."""
    legacy = journal.request_record(MERGE_STAGE, {"left": left, "right": right})
    left_ids = [row["id"] for row in left]
    right_ids = [row["id"] for row in right]
    expected = set(left_ids + right_ids)
    sides = dict.fromkeys(left_ids, "L") | dict.fromkeys(right_ids, "R")

    def interleave(picks):
        remaining = {"L": iter(left_ids), "R": iter(right_ids)}
        return [next(remaining[side]) for side in picks]

    # Replay complete saved replies if every cross-list relation can be kept:
    # only repair order inside uninterrupted blocks from the same list.
    for path in sorted(journal.directory.glob(f"*-{MERGE_STAGE}.json")):
        saved = read_json(path)
        request, reply = saved.get("request", {}), saved.get("response", {})
        payload = dict(request.get("input", {}))
        payload.pop("output_format_correction", None)
        if {**request, "input": payload} != legacy or reply.get("finish_reason") != "stop":
            continue
        try:
            response = strict_json(reply.get("content"))
        except ContractError:
            continue
        sequence = response.get("order") if isinstance(response, dict) and set(response) == {"order"} else None
        if (not isinstance(sequence, list) or len(sequence) != len(expected)
                or any(type(alias) is not int for alias in sequence) or set(sequence) != expected):
            continue
        block = 0
        previous_side = None
        blocks = {}
        for alias in sequence:
            side = sides[alias]
            if side != previous_side:
                block += 1
                previous_side = side
            blocks[alias] = block
        if any([blocks[alias] for alias in ids] != sorted(blocks[alias] for alias in ids)
               for ids in (left_ids, right_ids)):
            continue
        print(f"Replayed cross-list positions from {journal.directory.name}/{path.name}", flush=True)
        return interleave(sides[alias] for alias in sequence)

    # Each side contains at least this many unused IDs, so any valid sequence
    # can only consume existing items, exactly once and in their original order.
    steps = min(len(left), len(right))
    payload = {"left": left, "right": right, "steps": steps}

    def validate(response):
        picks = response.get("picks") if isinstance(response, dict) and set(response) == {"picks"} else None
        if not isinstance(picks, str):
            raise ContractError("Expected a picks string containing L and R")
        picks = "".join(picks.split()).upper()
        if len(picks) < steps or any(side not in "LR" for side in picks):
            raise ContractError(f"Expected at least {steps} L/R choices")
        return interleave(picks[:steps])

    saved = journal.revalidate_invalid(MERGE_SIDES_STAGE, payload, validate)
    if saved is not None:
        return saved[0]
    correction = {**payload, "output_format_correction": f"Return {steps} L/R letters in a JSON picks string."}
    error = None
    for request in (payload, correction):
        rejected = journal.invalid_reason(MERGE_SIDES_STAGE, request)
        if rejected is not None:
            error = ContractError(rejected)
            continue
        try:
            return validate(journal.call(MERGE_SIDES_STAGE, request))
        except ContractError as failure:
            journal.mark_invalid(str(failure))
            error = failure
    raise error


def _rank_shard(nodes, provider, parameters, journal_dir, *, prompts):
    indexed = {i: node for i, node in enumerate(nodes, 1)}
    payload = {'concepts': [{'id': i, 'name': node['concept_label'], 'description': node['concept_description']}
                            for i, node in indexed.items()]}
    journal = LLMJournal(journal_dir, provider, {stage: prompts[stage] for stage in (STAGE, REPAIR_STAGE)},
                         {STAGE: parameters, REPAIR_STAGE: parameters},
                         maximum_calls=len(list(journal_dir.glob('*.json'))) + 6 + 4 * len(nodes))

    def valid_sequence(response):
        order = response.get('order') if isinstance(response, dict) and set(response) == {'order'} else None
        if (not isinstance(order, list) or not order or any(type(i) is not int for i in order)
                or len(order) != len(set(order)) or not set(order) <= indexed.keys()):
            raise ContractError('Expected a JSON order array with unique supplied numeric IDs')
        return order

    def unambiguous_sequence(response):
        if not isinstance(response, dict) or set(response) != {'order'} or not isinstance(response['order'], list):
            return []
        supplied = [alias for alias in response['order'] if type(alias) is int and alias in indexed]
        counts = Counter(supplied)
        # Repeated aliases do not have a reliable position. Leave every
        # occurrence out so the explicit placement repair can rank it once.
        return [alias for alias in supplied if counts[alias] == 1]

    def repair_unambiguous(order, reference):
        print(f'Replaying {len(order)} unambiguous ranks from {reference}; repairing all other IDs', flush=True)
        return repair_order(order, reference, payload['concepts'], journal, validate)

    def validate(response):
        order = valid_sequence(response)
        if len(order) != len(indexed):
            raise ContractError('The ranked sequence leaves out supplied IDs')
        return [indexed[i]['concept_id'] for i in order]

    saved = journal.revalidate_invalid(STAGE, payload, validate)
    if saved is not None:
        return saved[0]

    best = []
    rank_reference = None
    salvage, salvage_reference = [], None
    for path in sorted(journal_dir.glob('*.json')):
        record = read_json(path)
        request, response = record.get('request', {}), record.get('response', {})
        if (request.get('stage') != STAGE or request.get('prompt') != prompts[STAGE]
                or request.get('parameters') != parameters or response.get('finish_reason') != 'stop'
                or request.get('input', {}).get('concepts') != payload['concepts']):
            continue
        try:
            decoded = strict_json(response.get('content'))
        except ContractError:
            continue
        try:
            order = valid_sequence(decoded)
        except ContractError:
            unique = unambiguous_sequence(decoded)
            if len(unique) > len(salvage):
                salvage, salvage_reference = unique, path.name
            continue
        if len(order) > len(best):
            best, rank_reference = order, path.name
    if best:
        if len(best) == len(indexed):
            return validate({'order': best})
        return repair_order(best, rank_reference, payload['concepts'], journal, validate)
    if salvage:
        return repair_unambiguous(salvage, salvage_reference)

    for attempt in range(3):
        response = None
        try:
            response = journal.call(STAGE, payload)
            order = valid_sequence(response)
        except ContractError as error:
            journal.mark_invalid(str(error))
            unique = unambiguous_sequence(response)
            if unique:
                return repair_unambiguous(unique, Path(journal.last_ref).name)
            if attempt == 2:
                raise
            payload = {**payload, 'output_format_correction': f'{error}. Return exactly {len(indexed)} integer IDs.'}
        except ProviderError as error:
            if attempt == 2 or not re.search(r'HTTP status (?:429|5\d\d)\b|Transport or response-envelope', str(error)):
                raise
            sleep(2 ** (attempt + 1))
        else:
            if len(order) == len(indexed):
                return validate({'order': order})
            ref = Path(journal.last_ref).name
            journal.mark_invalid('The ranked sequence leaves out supplied IDs')
            return repair_order(order, ref, payload['concepts'], journal, validate)


def repair_order(ranked, reference, concepts, journal, validate):
    by_alias = {item['id']: item for item in concepts}
    missing = [alias for alias in by_alias if alias not in ranked]
    print(f'Recovering ranking from {reference}: {len(ranked)}/{len(concepts)} IDs; placing {len(missing)} missing IDs',
          flush=True)
    completed = _repair_placements(ranked, missing, by_alias, journal)
    return validate({'order': completed})


def _apply_placements(response, ranked, missing, *, allow_chains=False):
    placements = response.get('placements') if isinstance(response, dict) and set(response) == {'placements'} else None
    if not isinstance(placements, list) or len(placements) != len(missing):
        raise ContractError('Expected one placement for each missing ID')
    missing_ids = set(missing)
    allowed = set(ranked) | missing_ids if allow_chains else set(ranked)
    at_anchor = defaultdict(list)
    seen = set()
    for row in placements:
        if not isinstance(row, dict) or set(row) != {'id', 'after_id'}:
            raise ContractError('Each placement needs an id and an after_id')
        alias, anchor = row['id'], row['after_id']
        if type(alias) is not int or alias not in missing_ids or alias in seen:
            raise ContractError('A placement used an invalid or duplicate missing ID')
        if anchor is not None and (type(anchor) is not int or anchor == alias or anchor not in allowed):
            raise ContractError('A placement used an invalid anchor ID')
        seen.add(alias)
        at_anchor[anchor].append(alias)

    def descendants(anchor):
        for child in at_anchor[anchor]:
            yield child
            yield from descendants(child)

    completed = list(descendants(None))
    for alias in ranked:
        completed.append(alias)
        completed.extend(descendants(alias))
    if len(completed) != len(ranked) + len(missing):
        raise ContractError('A placement chain has a cycle or no ranked anchor')
    return completed


def _repair_placements(ranked, missing, by_alias, journal):
    payload = {'ranked': [by_alias[alias] for alias in ranked],
               'missing': [by_alias[alias] for alias in missing]}
    expected_request = journal.request_record(REPAIR_STAGE, payload)
    responses = []
    for path in sorted(journal.directory.glob(f'*-{REPAIR_STAGE}.json')):
        record = read_json(path)
        request, response = record.get('request', {}), record.get('response', {})
        saved_input = dict(request.get('input', {}))
        saved_input.pop('output_format_correction', None)
        if ({**request, 'input': saved_input} != expected_request
                or record.get('status') not in ('response_received', 'response_invalid')
                or response.get('finish_reason') != 'stop'):
            continue
        try:
            responses.append((path.name, strict_json(response.get('content'))))
        except ContractError:
            continue

    # Prefer the original contract whenever a saved reply passes it.
    # A fully anchored, acyclic chain gives an explicit order in one rank gap.
    # Replay it without editing the original model reply or adding a request.
    for allow_chains in (False, True):
        for reference, response in responses:
            try:
                completed = _apply_placements(response, ranked, missing, allow_chains=allow_chains)
            except ContractError:
                continue
            if allow_chains:
                print(f'Replayed anchored placement chains from {reference}', flush=True)
            return completed

    error = None
    correction = {'placement_ids': missing, 'ranked_anchor_ids': ranked,
                  'instruction': 'Use only ranked_anchor_ids or null for after_id. '
                                 'Do not link missing IDs to each other.'}
    for request in (payload, {**payload, 'output_format_correction': correction}):
        rejected = journal.invalid_reason(REPAIR_STAGE, request)
        if rejected is not None:
            error = ContractError(rejected)
            continue
        for attempt in range(3):
            try:
                response = journal.call(REPAIR_STAGE, request)
                return _apply_placements(response, ranked, missing, allow_chains=True)
            except ContractError as failure:
                journal.mark_invalid(str(failure))
                error = failure
                break
            except ProviderError as failure:
                if attempt == 2 or not re.search(r'HTTP status (?:429|5\d\d)\b|Transport or response-envelope', str(failure)):
                    raise
                sleep(2 ** (attempt + 1))
    if len(missing) == 1:
        raise error
    middle = len(missing) // 2
    completed = _repair_placements(ranked, missing[:middle], by_alias, journal)
    return _repair_placements(completed, missing[middle:], by_alias, journal)
