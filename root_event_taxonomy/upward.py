"""Run bottom-up event taxonomy induction without predefined roots."""

import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from itertools import islice
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from scipy.spatial.distance import pdist, squareform

from event_taxonomy.adapters.common import require_text
from event_taxonomy.induction.contracts import BudgetExceeded, CandidateError, ContractError, ProviderError
from event_taxonomy.induction.graph import assert_preserved, digest, load_taxonomy
from event_taxonomy.induction.journal import LLMJournal
from event_taxonomy.induction.providers import OpenAICompatibleProvider
from event_taxonomy.induction.storage import atomic_json, read_json, run_lock
from event_taxonomy.models import SCHEMA_VERSION, Taxonomy, UniqueConcept, stable_id, to_dict
from event_taxonomy.preparation.sources import SourceFile
from event_taxonomy.taxonomy.dag import reachable, validate_dag

from .clustering import average_tree, cut_tree
from .embedding import EmbeddingAPI, Embeddings
from .graph import GraphView, edge, intermediate, update_graph
from .leaf_deduplication import validate_leaf_links
from .upward_review import (
    PROMPTS, check_cross_review, check_parent_reviews, check_proposals, checked_call, describe, review_items,
)

DEFAULT_CUTOFFS = (0.35, 0.45, 0.55)
TEXT_FORMAT = "concept_name + newline + description"


def initial_taxonomy(dataset):
    leaves, _ = validate_leaf_links(dataset)
    if not leaves:
        raise ValueError("At least one leaf concept is required.")
    evidence = {
        row["source_id"]: {
            "evidence_id": stable_id("source-evidence", row["source_id"], row["Evidence"]),
            "source_event_id": row["source_id"], "text": row["Evidence"],
        }
        for row in dataset["source_events"]
    }
    sources = {row["source_concept_id"]: row for row in dataset["source_concepts"]}
    nodes, links, mappings = [], [], []
    for leaf in leaves.values():
        node_id = leaf["concept_id"]
        events = list(leaf["source_events"])
        evidence_ids = [evidence[event_id]["evidence_id"] for event_id in events]
        nodes.append(UniqueConcept(
            node_id, leaf["concept_name"], leaf["description"], list(leaf["source_concepts"]), [],
            node_id, "leaf_concept_export/1", events, evidence_ids, "event_type",
        ))
        mappings.append({"concept_id": node_id, "event_ids": events, "evidence_ids": evidence_ids,
                         "mapping_kind": "direct", "via_child_ids": []})
        links.extend({
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id, "concept_id": node_id, "relation": "IS_INSTANCE_OF",
            "status": "proposed", "evidence_ids": [evidence[event_id]["evidence_id"]],
            "candidate_ids": [member for member in leaf["source_concepts"]
                              if sources[member]["source_event"] == event_id],
            "decision_ref": "leaf_concepts.json",
        } for event_id in events)
    state = Taxonomy(
        taxonomy_id=stable_id("taxonomy", list(leaves)), nodes=nodes, parent_child_edges=[], predefined_roots=[],
        concept_evidence_mapping=mappings, instance_concept_links=links, unplaced_concept_ids=sorted(leaves),
        frozen_input_concept_ids=sorted(leaves), status="partial",
    )
    validate_dag(state)
    return state, list(evidence.values())


def cut_quality(partition, positions, distances):
    """Mean silhouette across all nodes, giving singletons a score of zero."""
    if len(partition) < 2:
        return 0.0
    rows = [np.array([positions[node_id] for node_id in group.members]) for group in partition]
    scores = np.zeros(len(positions))
    for index, group in enumerate(rows):
        if len(group) == 1:
            continue
        within = distances[np.ix_(group, group)].sum(axis=1) / (len(group) - 1)
        outside = np.min([distances[np.ix_(group, other)].mean(axis=1)
                          for i, other in enumerate(rows) if i != index], axis=0)
        scale = np.maximum(within, outside)
        scores[group] = np.divide(outside - within, scale, out=np.zeros_like(scale), where=scale > 0)
    return float(scores.mean())


def choose_cut(node_ids, vectors, cutoffs, max_group_size, fixed=None):
    if not node_ids:
        choices = sorted(set(cutoffs) | ({fixed} if fixed is not None else set()))
        stats = [{"cutoff": cutoff, "groups": 0, "grouped_nodes": 0, "singletons": 0,
                  "largest_group": 0, "mean_silhouette": 0.0} for cutoff in choices]
        return fixed if fixed is not None else choices[0], stats, {}, []
    condensed = np.clip(pdist(vectors, metric="cosine"), 0, 2)
    if not np.isfinite(condensed).all():
        raise ValueError("Invalid cosine distances for the current nodes.")
    tree = average_tree(node_ids, condensed)
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    distances = squareform(condensed)
    partitions, stats = {}, []
    for cutoff in sorted(set(cutoffs) | ({fixed} if fixed is not None else set())):
        partition = cut_tree(tree, cutoff, max_group_size)
        partitions[cutoff] = partition
        sizes = [len(group.members) for group in partition if len(group.members) > 1]
        stats.append({
            "cutoff": cutoff, "groups": len(sizes), "grouped_nodes": sum(sizes),
            "singletons": len(node_ids) - sum(sizes), "largest_group": max(sizes, default=0),
            "mean_silhouette": cut_quality(partition, positions, distances),
        })
    useful = [row for row in stats if row["groups"]]
    best = max(useful or stats, key=lambda row: (row["mean_silhouette"], row["grouped_nodes"], -row["cutoff"]))
    selected = fixed if fixed is not None else best["cutoff"]
    groups = {stable_id("group", group.members): group.members
              for group in partitions[selected] if len(group.members) > 1}
    singletons = [group.members[0] for group in partitions[selected] if len(group.members) == 1]
    return selected, stats, groups, singletons


def load_leaf_vectors(path, source, model, dimensions):
    if path is None:
        return {}
    with np.load(path, allow_pickle=False) as archive:
        if (archive["leaf_concepts_sha256"].item() != source.sha256 or archive["model"].item() != model
                or archive["input_format"].item() != TEXT_FORMAT):
            raise ValueError("Leaf embeddings do not match the input, model or text format.")
        node_ids, vectors = archive["concept_ids"].tolist(), archive["vectors"]
    expected = [leaf["concept_id"] for leaf in source.json()["leaf_concepts"]]
    if (node_ids != expected or vectors.shape != (len(expected), dimensions)
            or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0)):
        raise ValueError("Leaf embedding IDs, dimensions or values are invalid.")
    return dict(zip(node_ids, vectors))


def seed_groups(document, groups):
    if document is None:
        return []
    available = {row["group_id"]: groups.get(row["group_id"]) for row in document["groups"]}
    if any(members is None for members in available.values()):
        raise ValueError("Saved first-layer groups differ from the current clustering.")
    return check_proposals(document, available)


class UpwardInducer:
    def __init__(self, journal, vectors, cutoffs, max_group_size, max_rounds, *,
                 maintenance_enabled=False, excluded_ids=(), dedup_neighbors=5, max_repairs=1):
        self.journal = journal
        self.vectors = vectors
        self.cutoffs = cutoffs
        self.max_group_size = max_group_size
        self.max_rounds = max_rounds
        self.maintenance_enabled = maintenance_enabled
        self.excluded_ids = set(excluded_ids)
        self.dedup_neighbors = dedup_neighbors
        self.max_repairs = max_repairs

    def active_ids(self, view):
        return {key for key, node in view.nodes.items() if key not in self.excluded_ids
                and node.validation_status not in {"maintenance_held", "admission_held", "merged"}}

    def active_signature(self, state):
        view = GraphView(state)
        active = self.active_ids(view)
        return digest({"nodes": sorted((key, view.nodes[key].concept_label,
                                        view.nodes[key].concept_description) for key in active),
                       "edges": sorted((edge.parent_id, edge.child_id) for edge in state.parent_child_edges)})

    @staticmethod
    def definition(node):
        return node.concept_description

    def propose(self, groups, view, descendants, seed):
        by_group = {row["group_id"]: row for row in seed}
        proposals = review_items(seed, view.nodes, descendants, "first_layer_proposals")
        pending = iter(group for group in groups if group not in by_group)

        def context(ids):
            return [{"name": view.nodes[c].concept_label, "definition": self.definition(view.nodes[c])}
                    for c in sorted(ids)]

        def request_proposal(payload, requested):
            validate = lambda response: check_proposals(response, requested)
            parameters = self.journal.settings["proposal"]
            if not getattr(self.journal.provider, "config", {}).get("extra_body", {}).get("enable_thinking"):
                return checked_call(self.journal, "proposal", payload, validate)
            truncated = "Incomplete or non-stop LLM response: length"
            if self.journal.was_truncated("proposal", payload):
                retry = {**payload, "output_format_correction": truncated}
                if self.journal.has_usable_response("proposal", retry):
                    return checked_call(self.journal, "proposal", retry, validate)
                return without_thinking(payload, validate, parameters)
            try:
                return checked_call(self.journal, "proposal", payload, validate)
            except ContractError:
                if self.journal.was_truncated("proposal", payload):
                    return without_thinking(payload, validate, parameters)
                raise

        def without_thinking(payload, validate, parameters):
            self.journal.settings["proposal"] = {
                **parameters, "extra_body": {**parameters.get("extra_body", {}), "enable_thinking": False},
            }
            try:
                return checked_call(self.journal, "proposal", payload, validate)
            finally:
                self.journal.settings["proposal"] = parameters

        def propose_batch(batch):
            # Only current members get selectable IDs. Lower levels are context.
            requested = {}
            payload = {"groups": []}
            for group_index, group in enumerate(batch, 1):
                aliases = {f"G{group_index}M{index}": node_id
                           for index, node_id in enumerate(groups[group], 1)}
                requested[group] = aliases
                members = []
                for alias, node_id in aliases.items():
                    node = view.nodes[node_id]
                    members.append({"id": alias, "name": node.concept_label, "definition": self.definition(node),
                                    "direct_children": context(view.children[node_id]),
                                    "descendant_leaves": context(descendants[node_id] - {node_id})})
                payload["groups"].append({"group_id": group, "members": members})
            try:
                result, ref = request_proposal(payload, requested)
            except ContractError:
                if len(batch) == 1:
                    raise
                middle = len(batch) // 2
                propose_batch(batch[:middle])
                propose_batch(batch[middle:])
                return
            for row in result:
                aliases = requested[row["group_id"]]
                for proposal in row["proposals"]:
                    proposal["child_ids"] = [aliases[child_id] for child_id in proposal["child_ids"]]
                    if proposal["reuse_id"] is not None:
                        proposal["reuse_id"] = aliases[proposal["reuse_id"]]
            by_group.update({row["group_id"]: row for row in result})
            proposals.extend(review_items(result, view.nodes, descendants, ref))

        while batch := list(islice(pending, 4)):
            propose_batch(batch)
        return proposals, by_group

    def review(self, proposals):
        reviews = {item["proposal_id"]: [] for item in proposals}
        pieces = []
        for item in proposals:
            leaves = item["descendant_leaves"]
            for start in range(0, max(1, len(leaves)), 10):
                pieces.append({
                    key: item[key] for key in ("proposal_id", "name", "definition", "reuse_id", "children")
                } | {"descendant_leaves": leaves[start:start + 10]})

        def review_batch(batch):
            try:
                decisions, ref = checked_call(
                    self.journal, "review", {"proposals": list(batch.values())},
                    lambda response: check_parent_reviews(response, batch),
                )
            except ContractError:
                if len(batch) == 1:
                    raise
                for key, item in batch.items():
                    review_batch({key: item})
                return
            for proposal_id, decision in decisions.items():
                reviews[proposal_id].append({**decision, "review_ref": ref})

        batch, count = {}, 0
        for item in [*pieces, None]:
            size = len(item["children"]) + len(item["descendant_leaves"]) if item else 0
            item_id = item["proposal_id"] if item else None
            if batch and (item is None or len(batch) >= 2 or count + size > 12 or item_id in batch):
                review_batch(batch)
                batch, count = {}, 0
            if item is not None:
                batch[item_id] = item
                count += size
        return reviews

    def cross_review(self, proposals, view):
        if not proposals:
            return {}
        candidates = [{**{key: row[key] for key in ("proposal_id", "name", "definition", "reuse_id")},
                       "child_ids": [child["id"] for child in row["children"]]}
                      for row in proposals]
        target_ids = [row["proposal_id"] for row in candidates]
        known = set(target_ids) | view.nodes.keys()
        payload = {"proposed_parents": candidates,
                   "existing_concepts": [describe(node) for key, node in view.nodes.items()
                                         if key in self.active_ids(view)]}
        cross = {}
        for start in range(0, len(target_ids), 20):
            batch = target_ids[start:start + 20]
            checks, ref = checked_call(
                self.journal, "cross_review", {**payload, "target_ids": batch},
                lambda response: check_cross_review(response, set(batch), known),
            )
            cross.update({proposal_id: {**check, "review_ref": ref} for proposal_id, check in checks.items()})
        return cross

    def commit(self, state, proposals, reviews, cross):
        before = state
        accepted, held = [], []
        for item in proposals:
            proposal_id = item["proposal_id"]
            if not reviews.get(proposal_id) or not all(row["passes"] for row in reviews[proposal_id]):
                held.append({"proposal_id": proposal_id, "reason": "Independent review did not pass"})
                continue
            check = cross.get(proposal_id)
            if check is None or check["verdict"] != "pass":
                continue
            view = GraphView(state)
            children = [view.nodes[row["id"]] for row in item["children"]]
            node = None
            parent_id = item["reuse_id"]
            try:
                if parent_id is None:
                    node = intermediate(state, self, item, children, check["review_ref"],
                                        normalization_version="upward/1")
                    covered = set().union(*(
                        reachable(view.children, child.concept_id) | {child.concept_id} for child in children
                    ))
                    node = replace(
                        node, validation_status="llm_independently_reviewed",
                        supporting_event_ids=sorted({event_id for child in covered
                                                     for event_id in view.nodes[child].supporting_event_ids}),
                        evidence=sorted({evidence_id for child in covered for evidence_id in view.nodes[child].evidence}),
                    )
                    parent_id = node.concept_id
                review = reviews[proposal_id][0]
                additions = [edge(parent_id, result["child_id"], result["reason"], review["review_ref"],
                                  self.journal.settings["review"]["model"]) for result in review["edge_reviews"]]
                state = update_graph(state, node, additions)
            except CandidateError as error:
                held.append({"proposal_id": proposal_id, "reason": str(error)})
                continue
            accepted.append({"proposal_id": proposal_id, "parent_id": parent_id, "new": node is not None,
                             "edge_ids": [link.edge_id for link in additions]})
        assert_preserved(before, state, accepted=True)
        return state, accepted, held

    def round(self, state, number, first_cutoff=None, first_layer=None):
        before_signature = self.active_signature(state)
        original_view = GraphView(state)
        previous_generated = {node_id for node_id in self.active_ids(original_view)
                              if original_view.nodes[node_id].origin_kind == "induced_intermediate"}
        dedup_report = None
        if self.maintenance_enabled:
            from .parent_deduplication import deduplicate_parents
            state, dedup_report = deduplicate_parents(state, self, neighbors=self.dedup_neighbors)
        view = GraphView(state)
        active = self.active_ids(view)
        frontier = [node_id for node_id in view.nodes if node_id in active and not view.parents[node_id]]
        leaves = set(state.frozen_input_concept_ids)
        descendants = {node_id: (reachable(view.children, node_id) | {node_id}) & leaves for node_id in frontier}
        vectors = self.vectors({node_id: view.nodes[node_id] for node_id in frontier}) if frontier else np.empty((0, 0))
        cutoff, comparison, groups, singletons = choose_cut(
            frontier, vectors, self.cutoffs, self.max_group_size, first_cutoff,
        )
        proposals, decisions = self.propose(groups, view, descendants, seed_groups(first_layer, groups))
        reviews = self.review(proposals)
        eligible = [item for item in proposals if reviews.get(item["proposal_id"]) and
                    all(row["passes"] for row in reviews[item["proposal_id"]])]
        cross = self.cross_review(eligible, view)
        state, accepted, held = self.commit(state, eligible, reviews, cross)
        inserted = deepcopy(accepted)
        maintenance_report = None
        if self.maintenance_enabled:
            from .maintenance import maintain_graph
            state, maintenance_report = maintain_graph(state, self, max_repairs=self.max_repairs)
            state, final_dedup = deduplicate_parents(state, self, neighbors=self.dedup_neighbors)
            dedup_report = {"before_proposal": dedup_report, "after_maintenance": final_dedup}
        after = GraphView(state)
        active = self.active_ids(after)
        next_input = [node_id for node_id in after.nodes if node_id in active and not after.parents[node_id]]
        accepted = [{**row, "edge_ids": [link.edge_id for link in state.parent_child_edges
                                         if link.parent_id == row["parent_id"]]}
                    for row in accepted if row["parent_id"] in active and after.children[row["parent_id"]]]
        current_generated = {node_id for node_id in active
                             if after.nodes[node_id].origin_kind == "induced_intermediate"}
        report = {
            "round": number, "input_ids": frontier, "cutoff": cutoff, "cut_comparison": comparison,
            "groups": [{"group_id": group, "member_ids": list(members), "reason": decisions[group]["reason"]}
                       for group, members in groups.items()],
            "singleton_ids": singletons, "proposals": proposals, "reviews": reviews,
            "cross_review": cross,
            "accepted": accepted, "structural_holds": held,
            "unassigned_input_ids": [node_id for node_id in frontier if node_id in active and not after.parents[node_id]],
            "next_input_ids": next_input,
            "parent_deduplication": dedup_report, "maintenance": maintenance_report,
            "inserted_before_maintenance": inserted,
            "active_structure_changed": before_signature != self.active_signature(state),
            "new_active_parent_count": len(current_generated - previous_generated),
        }
        return state, report

    def run(self, output, result, first_cutoff, first_layer):
        state = load_taxonomy(result["taxonomy"])
        for number in range(len(result["rounds"]) + 1, self.max_rounds + 1):
            try:
                state, report = self.round(state, number, first_cutoff if number == 1 else None,
                                           first_layer if number == 1 else None)
            except (ProviderError, ContractError, BudgetExceeded) as error:
                reason = ("maximum_llm_calls" if isinstance(error, BudgetExceeded) else
                          "provider_error" if isinstance(error, ProviderError) else "invalid_model_output")
                result.update(status="paused", stop_reason=reason, issue=str(error))
                atomic_json(output, result)
                return result
            result["taxonomy"] = to_dict(state)
            result["rounds"].append(report)
            result.pop("issue", None)
            new_count = report["new_active_parent_count"]
            print(f'Round {number}: cutoff {report["cutoff"]}, {len(report["groups"])} groups, '
                  f'{new_count} new parents, {len(report["next_input_ids"])} nodes at the top.', flush=True)
            unchanged = not report["active_structure_changed"] if self.maintenance_enabled else not new_count
            reason = (("no_approved_changes" if self.maintenance_enabled else "no_approved_new_parent") if unchanged else
                      "maximum_rounds" if number == self.max_rounds else None)
            result.update(status="finished" if reason else "running", stop_reason=reason)
            atomic_json(output, result)
            if reason:
                break
        return result


def api_settings(path):
    settings = read_json(path)
    llm = settings["llm"]["provider"]
    embedding = settings.get("embedding", {})
    if llm.get("kind", "openai_compatible") != "openai_compatible":
        raise ValueError("This entry point requires an OpenAI-compatible API provider.")
    model = require_text(llm.get("model"), "Chat model")
    key_env = require_text(llm.get("api_key_env"), "API credential variable")
    base_url = require_text(llm.get("base_url") or os.environ.get(llm.get("base_url_env", "")), "API base URL")
    parts = urlsplit(base_url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Use an HTTPS API base URL without embedded credentials.")
    connection = {"base_url": base_url, "api_key_env": key_env, "timeout_seconds": llm.get("timeout_seconds", 150)}
    embedding_config = {**connection, "model": embedding.get("model", "text-embedding-v4"),
                        "dimensions": embedding.get("dimensions", 1024), "batch_size": embedding.get("batch_size", 10)}
    stage_settings = {stage: {"model": model, "temperature": 0, "max_tokens": 8000} for stage in PROMPTS}
    return {**connection, "extra_body": llm.get("extra_body", {})}, embedding_config, stage_settings


def run_directory(args, dataset, taxonomy, config, api, embedding, stages, cached, first_layer):
    directory = args.output.resolve()
    if not args.resume:
        directory.mkdir(parents=True, exist_ok=False)
    with run_lock(directory):
        output = directory / "hierarchy.json"
        if args.resume:
            result = read_json(output)
            saved_config = result["configuration"]
            limits = {"max_rounds", "max_llm_calls"}
            if ({k: v for k, v in saved_config.items() if k not in limits}
                    != {k: v for k, v in config.items() if k not in limits}
                    or any(config[key] < saved_config[key] for key in limits)):
                raise ValueError("Resume with the same input files, models, and settings.")
            assert_preserved(taxonomy, load_taxonomy(result["taxonomy"]), accepted=True)
            if result["status"] == "finished":
                if result["stop_reason"] != "maximum_rounds" or args.max_rounds <= len(result["rounds"]):
                    return result
            result["configuration"] = config
        else:
            result = {**dataset, "configuration": config, "taxonomy": to_dict(taxonomy),
                      "rounds": [], "status": "running", "stop_reason": None}
            atomic_json(output, result)
        embeddings = Embeddings(EmbeddingAPI(embedding), embedding, None, directory / "embeddings")
        embeddings.restore()
        for node in taxonomy.nodes:
            if node.concept_id in cached:
                name_and_definition = node.concept_label + "\n" + node.concept_description
                embeddings.vectors[digest([node.concept_id, name_and_definition])] = cached[node.concept_id]

        def vectorize(nodes):
            return embeddings.for_texts({node_id: node.concept_label + "\n" + node.concept_description
                                         for node_id, node in nodes.items()})

        journal = LLMJournal(directory / "llm", OpenAICompatibleProvider(api), PROMPTS, stages, args.max_llm_calls)
        from .admission import review_admission
        if "admission" not in result:
            try:
                result["admission"] = review_admission(dataset, journal,
                    batch_size=args.admission_batch_size, scope_policy=args.scope_policy)
                result.pop("issue", None)
                result.update(status="running", stop_reason=None)
                atomic_json(output, result)
            except (ProviderError, ContractError, BudgetExceeded) as error:
                reason = ("maximum_llm_calls" if isinstance(error, BudgetExceeded) else
                          "provider_error" if isinstance(error, ProviderError) else "invalid_model_output")
                result.update(status="paused", stop_reason=reason, issue=str(error))
                atomic_json(output, result)
                return result
        print(f'Admission: {len(result["admission"]["accepted_leaf_ids"])} admitted, '
              f'{len(result["admission"]["held_leaf_ids"])} held.', flush=True)
        if args.admission_only:
            result.update(status="paused", stop_reason="admission_completed")
            atomic_json(output, result)
            return result
        inducer = UpwardInducer(journal, vectorize, args.cutoffs, args.max_group_size, args.max_rounds,
                               maintenance_enabled=True, excluded_ids=result["admission"]["held_leaf_ids"],
                               dedup_neighbors=args.dedup_neighbors, max_repairs=args.max_repairs)
        result.update(status="running", stop_reason=None)
        result.pop("issue", None)
        atomic_json(output, result)
        return inducer.run(output, result, args.first_cutoff, first_layer)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaves", type=Path, required=True, help="Deduplicated leaf_concepts.json")
    parser.add_argument("--leaf-embeddings", type=Path, help="Reuse matching vectors for the initial leaves")
    parser.add_argument("--first-layer-proposals", type=Path, help="Saved proposals to independently review in round one")
    parser.add_argument("--first-cutoff", type=float, help="Fixed round-one cutoff for replaying prior groups")
    parser.add_argument("--cutoffs", nargs="+", type=float, default=DEFAULT_CUTOFFS)
    parser.add_argument("--max-group-size", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-llm-calls", type=int, default=150)
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--scope-policy", choices=("reported_events", "research_statements"), default="reported_events")
    parser.add_argument("--admission-batch-size", type=int, default=4)
    parser.add_argument("--admission-only", action="store_true", help="Save admission decisions before continuing with --resume")
    parser.add_argument("--dedup-neighbors", type=int, default=5)
    parser.add_argument("--max-repairs", type=int, default=1, help="Maximum repair proposals per reviewed parent")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-input", action="store_true", help="Validate inputs and preview the first cut without API calls")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        cutoffs = [*args.cutoffs, *([] if args.first_cutoff is None else [args.first_cutoff])]
        if (not all(math.isfinite(value) and 0 <= value <= 2 for value in cutoffs)
                or args.max_group_size < 2 or min(args.max_rounds, args.max_llm_calls, args.max_output_tokens, args.admission_batch_size,
                                                args.dedup_neighbors) < 1 or args.max_repairs not in (0, 1)):
            raise ValueError("Use cosine distance cutoffs in [0, 2], group size ≥ 2, and positive run limits.")
        if args.first_layer_proposals and (args.first_cutoff is None or args.leaf_embeddings is None):
            raise ValueError("Replay first-layer proposals with their fixed cutoff and saved leaf embeddings.")
        source = SourceFile.read(args.leaves)
        dataset = source.json()
        taxonomy, evidence = initial_taxonomy(dataset)
        api, embedding, stages = api_settings(args.model_config)
        stages = {stage: {**settings, "max_tokens": args.max_output_tokens} for stage, settings in stages.items()}
        cached = load_leaf_vectors(args.leaf_embeddings, source, embedding["model"], embedding["dimensions"])
        seed = SourceFile.read(args.first_layer_proposals) if args.first_layer_proposals else None
        first_layer = seed.json() if seed else None
        if cached and (args.check_input or args.first_layer_proposals):
            cutoff, comparison, groups, singletons = choose_cut(
                list(cached), np.stack(list(cached.values())), args.cutoffs, args.max_group_size, args.first_cutoff,
            )
            saved = seed_groups(first_layer, groups)
        if args.check_input:
            print(json.dumps({"leaves": len(taxonomy.nodes), "predefined_roots": 0,
                              "cutoff": cutoff if cached else None, "cut_comparison": comparison if cached else [],
                              "groups": len(groups) if cached else None, "singletons": len(singletons) if cached else None,
                              "saved_groups": len(saved) if cached else 0, "api_model": stages["proposal"]["model"],
                              "credential_available": bool(os.environ.get(api["api_key_env"]))}, ensure_ascii=False, indent=2))
            return 0
        if args.output is None:
            raise ValueError("--output is required for a live run.")
        if not os.environ.get(api["api_key_env"]):
            raise ValueError(f'Set {api["api_key_env"]} for the configured API before starting.')
        config = {
            "schema": "upward-taxonomy/1", "leaf_sha256": source.sha256,
            "leaf_embeddings_sha256": sha256(args.leaf_embeddings.read_bytes()).hexdigest() if args.leaf_embeddings else None,
            "first_layer_sha256": seed.sha256 if seed else None,
            "cutoffs": list(args.cutoffs), "first_cutoff": args.first_cutoff, "max_group_size": args.max_group_size,
            "max_rounds": args.max_rounds, "max_llm_calls": args.max_llm_calls,
            "embedding": embedding, "api": api, "stages": stages, "prompts_sha256": digest(PROMPTS),
            "pipeline_version": 2, "scope_policy": args.scope_policy,
            "admission_batch_size": args.admission_batch_size, "dedup_neighbors": args.dedup_neighbors,
            "max_repairs": args.max_repairs,
        }
        result = run_directory(args, {**dataset, "source_evidence": evidence}, taxonomy,
                               config, api, embedding, stages, cached, first_layer)
        print(json.dumps({key: result[key] for key in ("status", "stop_reason", "issue") if key in result},
                         ensure_ascii=False, indent=2))
        return 0 if result["status"] == "finished" else 2
    except (ContractError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
