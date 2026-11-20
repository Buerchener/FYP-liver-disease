"""Model-assisted conceptualization, equivalence review and ordering."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from types import SimpleNamespace

from .model import endpoint, live_provider, read_settings
from .normalization import (
    AMBIGUOUS, IN_SCOPE, PROMPT as REFERENT_PROMPT, TYPE_PROMPT,
    audited_kind, classify_output,
)
from .order_entities import apply_entity_order, rank_entity_concepts
from .retrieve_pairs import DEFAULT_KNN_K, retrieve_pairs
from .storage import atomic_json, read_json

JSON_RETRY = ' Reply exactly as {"rows":[...]} with every requested ID once.'
TYPES = IN_SCOPE
EQUIV = """Judge each enumerated pair for strict concept equivalence using the candidate names, types, definitions and observed context. Merge only for the same biomedical referent at the same granularity. Relatedness, anatomical containment, biological role, association, distinct severity, and broader/narrower are separate. An abbreviation needs contextual support. If doubtful use uncertain. Return JSON object with rows array, one per pair id: id, verdict (equivalent/distinct/uncertain), reason (brief Chinese)."""


def key(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:20]


def normalization_requests(concept_input):
    fields = ("id", "mention")
    fields = (*fields, "context_id", "same_span_conflicts", "supporting_sentence_ids")
    for article in concept_input["articles"]:
        items = [
            {field: mention[field] for field in fields}
            for mention in article["mentions"]
            if mention["offset_status"] == "exact"
        ]
        if items:
            yield (
                "normalize_referent_v2",
                REFERENT_PROMPT,
                {
                    "pmid": article["pmid"],
                    "types": TYPES,
                    "sentences": article["sentences"],
                    "items": items,
                },
            )


class Stages:
    """Keep model configuration and journals scoped to a single batch."""

    def __init__(self, config, output):
        self.config = config
        self.output = output

    def request_settings(self):
        conn, params = read_settings(self.config, "audit_edges")
        return conn, {**params, "max_tokens": 6000}


    def req(self, task):
        stage, prompt, payload = task
        conn, params = self.request_settings()
        journal = (
            self.output
            / "calls"
            / f"{stage}-{key([prompt, payload, params, endpoint(conn)])}.json"
        )
        ids = [r["id"] for r in payload["items"]]
        if journal.exists():
            result = read_json(journal)["response"]
        else:
            partials = [
                self.output / "invalid_responses" / f"{journal.stem}-{i}.json" for i in (1, 2)
            ]
            provider = live_provider(conn, params)
            for attempt in range(2):
                instruction = (
                    prompt
                    if attempt == 0
                    else prompt + JSON_RETRY
                )
                answer = provider.generate(stage, instruction, payload, params)
                try:
                    result = json.loads(answer["content"])
                    valid = answer["finish_reason"] == "stop" and sorted(ids) == sorted(
                        row["id"] for row in result["rows"]
                    )
                except (KeyError, ValueError, TypeError):
                    valid = False
                if valid:
                    atomic_json(
                        journal,
                        {
                            "stage": stage,
                            "model": answer["model"],
                            "usage": answer["usage"],
                            "parameters": params,
                            "prompt": instruction,
                            "request": payload,
                            "response": result,
                        },
                    )
                    break
                atomic_json(
                    self.output / "invalid_responses" / f"{journal.stem}-{attempt + 1}.json",
                    {
                        "stage": stage,
                        "finish_reason": answer["finish_reason"],
                        "content": answer["content"],
                        "usage": answer["usage"],
                    },
                )
            else:
                raise ValueError(
                    f"{stage}: responses did not cover all requested IDs with the expected schema"
                )
        if sorted(ids) != sorted(row["id"] for row in result["rows"]):
            raise ValueError("Incomplete cached response")
        return result["rows"]


    def batch(self, stage, prompt, items, size, **context):
        tasks = [
            (stage, prompt, {"items": items[i : i + size], **context})
            for i in range(0, len(items), size)
        ]
        with ThreadPoolExecutor(max_workers=3) as pool:
            return [row for group in pool.map(self.req, tasks) for row in group]


    def checked_normalization_call(self, task, check):
        stage, prompt, payload = task
        try:
            return [check(row) for row in self.req(task)]
        except (KeyError, TypeError, ValueError):
            correction = {
                **payload,
                "output_format_correction": "Use the listed kind vocabulary and return the required fields for each supplied ID.",
            }
            return [check(row) for row in self.req((stage, prompt, correction))]


    def request_referents(self, task):
        _, _, payload = task
        first = self.checked_normalization_call(task, classify_output)
        source = {item["id"]: item for item in payload["items"]}
        items = [
            {**source[row["id"]], "name": row["name"], "definition": row["definition"]}
            for row in first if row["kind"] and row["kind"] != "NonEntity"
        ]
        if not items:
            return first
        reviewed = dict(self.checked_normalization_call(
            ("normalize_type_check_v2", TYPE_PROMPT, {**payload, "items": items}), audited_kind,
        ))
        final = []
        for row in first:
            ident = row["id"]
            if ident not in reviewed or row["kind"] == reviewed[ident]:
                final.append(row)
                continue
            revised = {**row, "kind": reviewed[ident]}
            if reviewed[ident] in {AMBIGUOUS, "NonEntity"}:
                revised.update(name=source[ident]["mention"], definition="", aliases=[])
            final.append({**classify_output(revised), "initial_kind": row["kind"]})
        return final


    def resolve(self, concept_input):
        mentions = [
            mention for article in concept_input["articles"] for mention in article["mentions"]
        ]
        decisions = {}
        for m in mentions:
            if m["offset_status"] != "exact":
                decisions[m["id"]] = {
                    "id": m["id"],
                    "status": "review",
                    "name": m["mention"],
                    "kind": "",
                    "aliases": [],
                    "definition": "",
                    "reason": "原始提及缺少可验证的字符坐标，替代出现仅供复核。",
                }
        with ThreadPoolExecutor(max_workers=3) as pool:
            for result in pool.map(self.request_referents, normalization_requests(concept_input)):
                for r in result:
                    if r["status"] not in (
                        "keep",
                        "review",
                        "outside_scope",
                    ) or not isinstance(r["aliases"], list):
                        raise ValueError(f"invalid decision: {r['id']}")
                    if r["status"] == "keep" and r["kind"] not in TYPES:
                        r = {
                            **r,
                            "status": "review",
                            "model_status": "keep",
                            "scope_note": "模型提出了本轮七类之外的类型，保留为类型扩充候选。",
                        }
                    if r["status"] == "keep" and (
                        not r["name"].strip() or not r["definition"].strip()
                    ):
                        raise ValueError(f"invalid kept concept: {r['id']}")
                    decisions[r["id"]] = r
        rows = [decisions[m["id"]] for m in mentions]
        atomic_json(self.output / "normalization.json", rows)
        return rows


    def equivalence_votes(self, items):
        if not items:
            return []
        conn, params = self.request_settings()
        active = {item["id"]: item for item in items}
        cached = {}
        disagreements = set()
        # Pair evidence can be identical even when retrieval changes the surrounding batch.
        for path in sorted((self.output / "calls").glob("equivalence-*.json")):
            journal = read_json(path)
            request = journal["request"]
            call_id = key([EQUIV, request, params, endpoint(conn)])
            if path.stem != f"equivalence-{call_id}" or journal["prompt"] not in (
                EQUIV, EQUIV + JSON_RETRY,
            ):
                continue
            responses = journal["response"]["rows"]
            if sorted(row["id"] for row in responses) != sorted(row["id"] for row in request["items"]):
                continue
            prior_items = {row["id"]: row for row in request["items"]}
            for row in responses:
                ident = row["id"]
                if active.get(ident) != prior_items[ident]:
                    continue
                if ident in cached and cached[ident]["verdict"] != row["verdict"]:
                    disagreements.add(ident)
                cached.setdefault(ident, row)
        # Conflicting saved judgments need a new call with the current batch.
        for ident in disagreements:
            cached.pop(ident)
        pending = [item for item in items if item["id"] not in cached]
        cached.update({row["id"]: row for row in self.batch("equivalence", EQUIV, pending, 10)})
        return [cached[item["id"]] for item in items]


    def deduplicate(self, concept_input, normalized, *, knn_k=DEFAULT_KNN_K):
        kept = {r["id"]: r for r in normalized if r["status"] == "keep"}
        mention_evidence = {
            mention["id"]: {
                "mention": mention["mention"],
                "sentence": article["sentences"][mention["context_id"]],
            }
            for article in concept_input["articles"]
            for mention in article["mentions"]
            if mention["id"] in kept
        }
        retrieval = retrieve_pairs(list(kept.values()), self.output, self.config, knn_k=knn_k)
        items = []
        for p in retrieval["pairs"]:
            sides = []
            for side in ("left", "right"):
                row = kept[p[side]]
                sides.append(
                    {k: row[k] for k in ("name", "kind", "definition", "aliases")}
                    | mention_evidence[row["id"]]
                )
            items.append(
                {"id": p["left"] + "|" + p["right"], "left": sides[0], "right": sides[1]}
            )
        votes = self.equivalence_votes(items)
        decisions = {}
        for v in votes:
            l, r = v["id"].split("|")
            if v["verdict"] not in ("equivalent", "distinct", "uncertain"):
                raise ValueError("Unknown equivalence judgment")
            compatible = kept[l]["kind"] == kept[r]["kind"]
            decisions[l, r] = {"relation": v["verdict"] if compatible else "distinct"}
            v["applied_verdict"] = decisions[l, r]["relation"]
        groups, conflicts = equivalent_groups(list(kept), decisions)
        concepts = []
        for members in groups:
            values = [kept[i] for i in members]
            terms = {}
            for r in values:
                for s in [r["name"], *r["aliases"]]:
                    terms.setdefault(s.strip().casefold(), s.strip())

            # Prefer a supported expanded name for display; keep all model names and aliases for audit.
            def display_order(name):
                return (len(name.split()) > 1, len(name), name.casefold())

            name = max(terms.values(), key=display_order)
            representative = next(
                (r for r in values if r["name"].casefold() == name.casefold()), values[0]
            )
            concepts.append(
                {
                    "id": "C-" + key(sorted(members))[:12],
                    "name": name,
                    "kind": representative["kind"],
                    "definition": representative["definition"],
                    "aliases": [s for s in terms.values() if s != name],
                    "source_mentions": list(members),
                    "origin": "literature_mention",
                }
            )
        atomic_json(self.output / "retrieval" / "retrieved_pairs.json", retrieval)
        atomic_json(
            self.output / "equivalence.json",
            {
                "pair_judgments": votes,
                "non_clique_components_held": conflicts,
                "concepts": concepts,
            },
        )
        return concepts


    def order_for_hierarchy(self, concepts):
        conn, params = self.request_settings()
        journal_dir = self.output / "ordering" / key([endpoint(conn), conn.get("extra_body", {})])

        @lru_cache(maxsize=1)
        def provider():
            return live_provider(conn, params)

        engine = SimpleNamespace(generate=lambda *args: provider().generate(*args))
        report = rank_entity_concepts(concepts, engine, params, journal_dir)
        ranked = apply_entity_order(concepts, report)
        atomic_json(self.output / "concept_order.json", report)
        return ranked


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
