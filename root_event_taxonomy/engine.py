"""Alternate concept induction and parent placement, then audit the proposed DAG."""
from collections import Counter
from pathlib import Path

import numpy as np

from event_taxonomy.induction.contracts import BudgetExceeded, CandidateError, ContractError, ProviderError
from event_taxonomy.induction.graph import digest, norm
from event_taxonomy.induction.storage import atomic_json, read_json
from event_taxonomy.models import to_dict
from event_taxonomy.taxonomy.dag import reachable, validate_dag

from .clustering import candidates
from .data import validate_source
from .decisions import duplicate_checks, edge_checks, group_decision, placement_decision
from .graph import GraphView, duplicate_keys, edge, finalize, intermediate, update_graph


class Inducer:
    def __init__(self, directory, config, data, embeddings, journal):
        self.directory = Path(directory)
        self.config, self.data = config, data
        self.embeddings, self.journal = embeddings, journal
        self.taxonomy = data.initial
        self.scope = set(data.working_ids)
        self.rounds = []
        self.groups_seen = set()
        self.placed = set()
        self.seek_broader = set()
        self.abstentions = {}
        self.issues = []
        self.trace_count = 0

    def trace(self, kind, **fields):
        self.trace_count += 1
        atomic_json(self.directory / "trace" / ("%06d-%s.json" % (self.trace_count, kind)), fields)

    def checkpoint(self):
        atomic_json(self.directory / "working_taxonomy.json", to_dict(self.taxonomy))

    def model(self, stage, payload, validate):
        # Stable short IDs reduce token use and transcription errors in the model.
        keys = sorted(self.context_ids(payload))
        aliases = {key: f"i{index + 1}" for index, key in enumerate(keys)}
        decoded = {alias: key for key, alias in aliases.items()}
        shortened = self.rename_ids(payload, aliases)
        current = shortened
        for attempt in range(self.config["response_retries"] + 1):
            self.journal.last_ref = None
            try:
                raw = self.journal.call(stage, current)
                response = validate(self.rename_ids(raw, decoded))
                self.trace("model-id-map", ref=self.journal.last_ref, ids=decoded)
                return response, self.journal.last_ref
            except ContractError as error:
                self.trace("invalid-response", stage=stage, attempt=attempt + 1,
                           ref=self.journal.last_ref, error=str(error), ids=decoded)
                current = {**shortened, "response_correction": {
                    "attempt": attempt + 1, "problem": str(error),
                    "instruction": "Return a complete corrected JSON object using the supplied contract.",
                }}
        self.issues.append({"stage": stage, "reason": "invalid_response", "ref": self.journal.last_ref})
        return None, self.journal.last_ref

    @staticmethod
    def context_ids(value):
        if isinstance(value, dict):
            result = set()
            for key, entry in value.items():
                if key in ("id", "edge_id", "parent_id", "child_id") and isinstance(entry, str):
                    result.add(entry)
                result.update(Inducer.context_ids(entry))
            return result
        if isinstance(value, (list, tuple)):
            return set().union(*(Inducer.context_ids(entry) for entry in value))
        return set()

    @staticmethod
    def rename_ids(value, aliases):
        if isinstance(value, dict):
            return {aliases.get(key, key): Inducer.rename_ids(entry, aliases) for key, entry in value.items()}
        if isinstance(value, (list, tuple)):
            return [Inducer.rename_ids(entry, aliases) for entry in value]
        if isinstance(value, str):
            return aliases.get(value, value)
        return value

    def rank(self, targets, view, eligible, limit):
        pool = [view.nodes[n] for n in sorted(eligible)]
        if not pool or not limit:
            return []
        # Targets can include an eligible reuse concept that is itself in a group.
        request = {node.concept_id: node for node in [*targets, *pool]}
        matrix = self.embeddings.for_nodes(list(request.values()))
        unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        positions = dict(zip(request, unit))
        source = np.vstack([positions[n.concept_id] for n in targets])
        scores = source @ np.vstack([positions[n.concept_id] for n in pool]).T
        ranked = [(node, float(score)) for node, score in zip(pool, np.clip(scores.mean(axis=0), -1, 1))]
        return sorted(ranked, key=lambda item: (-item[1], item[0].concept_id))[:limit]

    def root_context(self, view):
        return [self.data.describe(view.nodes[r]) for r in self.data.roots]

    def path_context(self, view, node):
        paths, truncated = view.paths(node.concept_id, self.config["max_ancestor_paths"])
        return {
            "paths": [[self.data.describe(view.nodes[n]) for n in path] for path in paths],
            "paths_truncated": truncated,
        }

    def propose_group(self, view, candidate, profiles):
        members = set(candidate.members)
        children = [view.nodes[n] for n in candidate.members]
        # Include the group members for exact in-group reuse. Similar non-members
        # are suggestions, even when their top-scoring root is different.
        ranked = self.rank(children, view, self.scope - members, self.config["reuse_candidates"])
        reuse_ids = members | {node.concept_id for node, _ in ranked}
        key = (candidate.members, tuple(sorted(reuse_ids)))
        if key in self.groups_seen:
            return None, None, "already_judged"
        self.groups_seen.add(key)
        vectors = self.embeddings.for_nodes(children)
        unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        payload = {
            "root_scope": [{"name": r["name"], "definition": r["definition"]} for r in self.root_context(view)],
            "members": [self.data.describe(n, evidence=True) for n in children],
            "other_existing_concepts": [self.data.describe(n) for n, _ in ranked],
        }
        self.trace("group-similarity", ids=candidate.members,
                   concept_cosine_similarity=np.clip(unit @ unit.T, -1, 1).tolist(),
                   root_similarity_by_member={n: profiles[n] for n in candidate.members})

        def validate(response):
            if isinstance(response, dict) and isinstance(response.get("proposals"), list):
                for row in response["proposals"]:
                    if isinstance(row, dict) and isinstance(row.get("parent"), str):
                        parent = row["parent"]
                        if any(norm(parent) == norm(view.nodes[n].concept_label) or parent == n for n in view.roots):
                            raise ContractError("The parent must be an intermediate type; roots are assigned separately.")
                        matches = [
                            n for n in sorted(reuse_ids)
                            if norm(parent) == norm(view.nodes[n].concept_label)
                        ]
                        if len(matches) == 1:
                            row["parent"] = matches[0]
            return group_decision(response, members, reuse_ids)

        value, ref = self.model("group", payload, validate)
        return value, ref, value["action"] if value else "invalid_response"

    def place(self, state, target_id, phase):
        view = GraphView(state)
        target = view.nodes[target_id]
        descendants = reachable(view.children, target_id)
        pool = (self.scope & view.nodes.keys()) - descendants - {target_id}
        ranked = self.rank([target], view, pool, self.config["parent_candidates"])
        shortlisted = set(self.data.roots) | {n.concept_id for n, _ in ranked} | view.parents[target_id]
        candidate_ids = [*self.data.roots, *sorted(shortlisted - view.roots)]
        candidate_nodes = [view.nodes[n] for n in candidate_ids]
        matrix = self.embeddings.for_nodes([target, *candidate_nodes])
        unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        scores = np.clip(unit[1:] @ unit[0], -1, 1)
        child_ids = sorted(view.children[target_id])
        payload = {
            "target": self.data.describe(target, evidence=True),
            "target_ancestry": self.path_context(view, target),
            "target_children": [self.data.describe(view.nodes[n]) for n in child_ids[:self.config["max_group_size"]]],
            "roots": self.root_context(view),
            "eligible_parents": [
                {**self.data.describe(node), "cosine_similarity": float(score), **self.path_context(view, node)}
                for node, score in zip(candidate_nodes, scores)
            ],
        }
        value, ref = self.model("placement", payload, lambda response: placement_decision(response, shortlisted))
        self.trace("placement", phase=phase, target_id=target_id, ref=ref, decision=value)
        return value, ref

    def placement_edges(self, view, target_id, value, ref):
        if value is None:
            return []
        return [
            edge(row["id"], target_id, row["reason"], ref, self.config["llm"]["stages"]["placement"]["model"])
            for row in value["parents"] if row["id"] not in view.parents[target_id]
        ]

    def remember_placement(self, target_id, value, ref):
        self.placed.add(target_id)
        if value:
            if value["seek_broader"]:
                self.seek_broader.add(target_id)
            else:
                self.seek_broader.discard(target_id)
            if value["parents"]:
                self.abstentions.pop(target_id, None)
            else:
                self.abstentions[target_id] = {"reason": value["reason"], "ref": ref}
        else:
            self.abstentions[target_id] = {"reason": "No valid placement response.", "ref": ref}

    def record_update(self, after, node, new_edges, phase):
        before = self.taxonomy
        self.taxonomy = after
        self.checkpoint()
        self.trace("dag-update", phase=phase, version_before=before.version, version_after=after.version,
                   added_node=to_dict(node) if node else None, added_edges=[to_dict(e) for e in new_edges],
                   graph_sha256=digest(to_dict(after)))

    def apply_proposal(self, proposal, ref):
        view = GraphView(self.taxonomy)
        children = [view.nodes[row["id"]] for row in proposal["children"]]
        node = None
        parent_id = proposal["reuse_id"]
        if parent_id is None:
            if len(self.scope - self.data.working_ids) >= self.config["max_new_nodes"]:
                raise CandidateError("maximum_new_nodes")
            node = intermediate(self.taxonomy, self.data, proposal, children, ref)
            parent_id = node.concept_id
        additions = [
            edge(parent_id, row["id"], row["reason"], ref, self.config["llm"]["stages"]["group"]["model"])
            for row in proposal["children"] if row["id"] not in view.children.get(parent_id, ())
        ]
        if not additions:
            raise CandidateError("Proposal adds no new is-a relation")
        # Validate the child links before asking the model to attach their parent.
        proposed = update_graph(self.taxonomy, node, additions)
        value, placement_ref = self.place(proposed, parent_id, "per_round")
        parent_edges = self.placement_edges(GraphView(proposed), parent_id, value, placement_ref)
        after = update_graph(proposed, edges=parent_edges) if parent_edges else proposed
        self.remember_placement(parent_id, value, placement_ref)
        if node is not None:
            self.scope.add(parent_id)
        self.record_update(after, node, additions + parent_edges, "induction")
        return parent_id

    def one_round(self, round_number, frontier):
        view = GraphView(self.taxonomy)
        nodes = [view.nodes[n] for n in [*frontier, *self.data.roots]]
        matrix = self.embeddings.for_nodes(nodes)
        cluster = self.config["clustering"]
        partition, scores = candidates(
            frontier, matrix[:len(frontier)], matrix[len(frontier):],
            cluster["distance_cutoff"], cluster["root_weight"], self.config["max_group_size"],
        )
        profiles = {n: dict(zip(self.data.roots, row.tolist())) for n, row in zip(frontier, scores)}
        stats = {"round": round_number, "frontier_size": len(frontier), "groups": 0,
                 "actions": {}, "committed_new": 0, "committed_reuse": 0}
        self.rounds.append(stats)
        self.trace("round-start", round=round_number, frontier=frontier, root_cosine_similarity=profiles,
                   groups=[{"ids": g.members, "merge_distance": g.height} for g in partition])
        # Prefer a group the placement stage flagged for possible further induction.
        groups = sorted((g for g in partition if len(g.members) >= 2),
                        key=lambda g: (-len(set(g.members) & self.seek_broader), g.height, g.members))
        pending = [(g, 0) for g in reversed(groups)]
        counts = Counter()
        while pending:
            group, retry = pending.pop()
            if set(group.members) - set(GraphView(self.taxonomy).frontier(self.scope)):
                self.trace("group-deferred", round=round_number, ids=group.members, reason="frontier_changed")
                continue
            value, ref, action = self.propose_group(GraphView(self.taxonomy), group, profiles)
            counts[action] += 1
            stats["groups"] += 1
            stats["actions"] = dict(counts)
            self.trace("group", round=round_number, ids=group.members, merge_distance=group.height,
                       split_retry=retry, ref=ref, action=action, decision=value)
            for proposal in value["proposals"] if value else ():
                try:
                    self.apply_proposal(proposal, ref)
                except CandidateError as error:
                    self.trace("transaction-rejected", round=round_number, ref=ref, error=str(error))
                else:
                    key = "committed_reuse" if proposal["reuse_id"] else "committed_new"
                    stats[key] += 1
            # A rejected cluster gets smaller candidate subclusters. No relaxed
            # semantic criteria and no repeat call with the identical shortlist.
            if retry < self.config["rejected_group_split_depth"]:
                if action in ("reject", "invalid_response"):
                    remaining = set(group.members)
                elif action in ("adjust", "split"):
                    remaining = set(group.members) & set(GraphView(self.taxonomy).frontier(self.scope))
                else:
                    remaining = set()
                pending.extend((part, retry + 1) for part in reversed(group.parts)
                               if len(part.members) >= 2 and set(part.members) <= remaining)
        stats["frontier_after"] = GraphView(self.taxonomy).frontier(self.scope)
        self.trace("round-end", **stats)

    def place_unrooted(self):
        # Topmost first; descendants already covered by a placement are skipped.
        for target in GraphView(self.taxonomy).frontier(self.scope):
            view = GraphView(self.taxonomy)
            if (target in view.rooted or target not in view.frontier(self.scope)
                    or (target in self.placed and target not in self.seek_broader)):
                continue
            value, ref = self.place(self.taxonomy, target, "unrooted_frontier")
            edges = self.placement_edges(view, target, value, ref)
            if edges:
                self.record_update(update_graph(self.taxonomy, edges=edges), None, edges, "unrooted_frontier")
            self.remember_placement(target, value, ref)
        # A rootless component can have an existing non-root parent outside scope.
        # Keep all its members in the review queue if no in-scope ancestor can be placed.

    def induction(self):
        for round_number in range(1, self.config["max_rounds"] + 1):
            frontier = GraphView(self.taxonomy).frontier(self.scope)
            if len(frontier) < 2:
                return "fewer_than_two_frontier_concepts"
            self.one_round(round_number, frontier)
            stats = self.rounds[-1]
            if not stats["groups"]:
                return "no_candidate_group"
            if not stats["committed_new"] + stats["committed_reuse"]:
                return "no_semantic_progress"
        return "maximum_rounds"

    def audit(self):
        view = GraphView(self.taxonomy)
        original_nodes = {n.concept_id for n in self.data.initial.nodes}
        introduced = sorted(view.nodes.keys() - original_nodes)
        all_edges = self.taxonomy.parent_child_edges
        checked, duplicates = {}, {}
        audit_size = self.config["audit_batch_size"]
        original_edges = {e.edge_id for e in self.data.initial.parent_child_edges}

        for start in range(0, len(all_edges), audit_size):
            batch = all_edges[start:start + audit_size]
            payload = {"edges": [{
                "edge_id": e.edge_id, "parent": self.data.describe(view.nodes[e.parent_id]),
                "child": self.data.describe(view.nodes[e.child_id]),
            } for e in batch]}
            try:
                value, ref = self.model("audit_edges", payload,
                                        lambda response: edge_checks(response, {e.edge_id for e in batch}))
            except (BudgetExceeded, ProviderError) as error:
                self.issues.append({"stage": "audit_edges", "reason": str(error), "ref": self.journal.last_ref})
                break
            if value:
                checked.update({r["edge_id"]: {**r, "ref": ref} for r in value["checks"]})
            self.trace("edge-audit", ref=ref, ids=[e.edge_id for e in batch], checks=value)

        for start in range(0, len(introduced), audit_size):
            batch = introduced[start:start + audit_size]
            payload, allowed = [], {}
            try:
                for n in batch:
                    ranked = self.rank([view.nodes[n]], view, (self.scope | view.roots) - {n},
                                       self.config["reuse_candidates"])
                    allowed[n] = {node.concept_id for node, _ in ranked}
                    payload.append({
                        "concept": self.data.describe(view.nodes[n]),
                        "similar_existing": [self.data.describe(node) for node, _ in ranked],
                    })
                value, ref = self.model("audit_duplicates", {"comparisons": payload},
                                        lambda response: duplicate_checks(response, allowed))
            except (BudgetExceeded, ProviderError) as error:
                self.issues.append({"stage": "audit_duplicates", "reason": str(error), "ref": self.journal.last_ref})
                break
            if value:
                duplicates.update({r["id"]: {**r, "ref": ref} for r in value["checks"]})
            self.trace("duplicate-audit", ref=ref, ids=batch, checks=value)

        final, cleanup = finalize(
            self.data.initial, self.taxonomy,
            {edge_id for edge_id, row in checked.items() if row["verdict"] == "valid"},
            {node_id for node_id, row in duplicates.items() if row["verdict"] == "distinct"},
        )
        checks = {
            "edge_verdicts": checked,
            "unchecked_edges": sorted({e.edge_id for e in all_edges} - checked.keys()),
            "duplicate_verdicts": duplicates,
            "unchecked_new_concepts": sorted(set(introduced) - duplicates.keys()),
            "exact_duplicates_in_final_graph": duplicate_keys(final, self.data),
            "original_edges_for_review": sorted(e for e in original_edges if checked.get(e, {}).get("verdict") != "valid"),
            **cleanup,
        }
        self.trace("final-dag-update", **cleanup, graph_sha256=digest(to_dict(final)))
        return final, checks

    def run(self):
        self.checkpoint()
        stage = "induction"
        stop = "interrupted"
        try:
            stop = self.induction()
            stage = "frontier_placement"
            self.place_unrooted()
        except (BudgetExceeded, ProviderError) as error:
            stop = "llm_call_budget" if isinstance(error, BudgetExceeded) else "provider_error"
            self.issues.append({"stage": stage, "reason": str(error), "ref": self.journal.last_ref})
            self.trace("stopped", stage=stage, stop=stop, error=str(error), ref=self.journal.last_ref)
        final, checks = self.audit()
        validate_source(self.data.dataset, final)
        atomic_json(self.directory / "final_taxonomy.json", to_dict(final))
        atomic_json(self.directory / "semantic_audit.json", checks)
        report = self.report(final, checks, stop)
        atomic_json(self.directory / "report.json", report)
        return report

    def report(self, final, checks, stop):
        view = GraphView(final)
        original = {n.concept_id for n in self.data.initial.nodes}
        selected = self.data.selected
        induced = view.nodes.keys() - original
        pending = sorted((self.scope & view.nodes.keys()) - view.rooted)
        roots = {
            root: sorted(selected & reachable(view.children, root))
            for root in self.data.roots
        }
        original_edges = {e.edge_id for e in self.data.initial.parent_child_edges}
        llm_records = [read_json(path) for path in self.journal.directory.glob("*.json")]
        embedding_records = [read_json(path) for path in self.embeddings.directory.glob("*.json")]
        llm_tokens, embedding_tokens = Counter(), Counter()
        for record in llm_records:
            llm_tokens.update({k: v for k, v in (record.get("response", {}).get("usage") or {}).items()
                               if type(v) is int})
        for record in embedding_records:
            embedding_tokens.update({k: v for k, v in (record.get("usage") or {}).items() if type(v) is int})
        return {
            "status": "partial" if self.issues or pending or checks["original_edges_for_review"] else "completed",
            "stop_reason": stop,
            "scope": self.data.scope(),
            "rounds": self.rounds,
            "structure": validate_dag(final),
            "selected_root_reachable": len(selected & view.rooted),
            "selected_unplaced": sorted(selected - view.rooted),
            "selected_by_root": roots,
            "induced_concepts_retained": sorted(induced),
            "new_edges_retained": len({e.edge_id for e in final.parent_child_edges} - original_edges),
            "multiple_parents_in_scope": {
                n: sorted(view.parents[n]) for n in sorted((selected | induced)) if len(view.parents[n]) > 1
            },
            "pending_root_review": [
                {"concept_id": n, "name": view.nodes[n].concept_label, **self.abstentions.get(n, {})}
                for n in pending
            ],
            "llm_calls": len(llm_records),
            "llm_calls_by_stage": dict(Counter(r["request"]["stage"] for r in llm_records)),
            "llm_models": sorted({r["response"]["model"] for r in llm_records if "response" in r}),
            "llm_usage": dict(llm_tokens),
            "embedding_batches": self.embeddings.batches,
            "embedding_model": self.embeddings.returned_model,
            "embedding_usage": dict(embedding_tokens),
            "issues": self.issues,
            "original_inputs_preserved": True,
            "instance_links_preserved": len(final.instance_concept_links),
            "human_semantic_review": "pending",
        }
