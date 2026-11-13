"""Five-step local refinement, with independent semantic checks before insertion."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from event_taxonomy.induction.contracts import CandidateError, obj, require, text
from event_taxonomy.induction.graph import assert_preserved, digest
from event_taxonomy.models import to_dict
from event_taxonomy.taxonomy.dag import reachable

from .decisions import duplicate_checks, placement_decision
from .discovery import knn_candidates
from .engine import Inducer
from .graph import GraphView, edge, intermediate, refresh, update_graph

PROMPT_DIR = Path(__file__).parent / "prompts"
REFINEMENT_PROMPTS = {stage: (PROMPT_DIR / filename).read_text() for stage, filename in {
    "group": "induction-proposal.xml", "validate_edges": "induction-validation.xml",
    "placement": "induction-placement.xml", "audit_duplicates": "induction-duplicates.xml",
}.items()}


def containment_checks(value, expected):
    obj(value, ("checks",))
    require(isinstance(value["checks"], list), "Expected edge checks")
    found = set()
    for row in value["checks"]:
        obj(row, ("edge_id", "child_entails_parent", "parent_entails_child", "reason"))
        text(row["edge_id"])
        text(row["reason"])
        require(row["edge_id"] in expected and row["edge_id"] not in found, "Unknown or repeated edge")
        found.add(row["edge_id"])
        forward, reverse = row["child_entails_parent"], row["parent_entails_child"]
        require(forward in ("true", "false", "uncertain") and reverse in ("true", "false", "uncertain"),
                "Use true, false or uncertain strings for both directions")
        row["verdict"] = "valid" if (forward, reverse) == ("true", "false") else (
            "uncertain" if "uncertain" in (forward, reverse) else "invalid")
    require(found == expected, "Missing edge checks")
    return value


def reduce_provisional_shortcuts(state):
    """Remove redundant provisional edges, including ancestors of reused parents."""
    after = deepcopy(state)
    removed = []
    for old in list(after.parent_child_edges):
        if old.status == "accepted":
            continue
        view = GraphView(after)
        view.children[old.parent_id].remove(old.child_id)
        if old.child_id in reachable(view.children, old.parent_id):
            after.parent_child_edges.remove(old)
            removed.append(to_dict(old))
    refresh(after)
    assert_preserved(state, after, accepted=True)
    return after, removed


def insert_transaction(before, node, additions):
    """Atomically add verified links and replace redundant provisional shortcuts.

    Confirmed edges remain until human-reviewed replacement paths exist. Other
    parents and all input concept/event/evidence records are immutable.
    """
    after, removed = reduce_provisional_shortcuts(update_graph(before, node, additions))
    assert_preserved(before, after, accepted=True)
    return after, removed


class FiveStepInducer(Inducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dirty = set(self.scope)
        self.attempted = set()
        self.validations = {}
        self.duplicates = {}
        self.rewired = []
        self.retry_pending = []

    def validate_edges(self, state, edges, phase):
        if not edges:
            return []
        view = GraphView(state)
        validated = []
        size = self.config["audit_batch_size"]
        for start in range(0, len(edges), size):
            batch = edges[start:start + size]
            # The independent call receives definitions, not the proposer's reasons.
            payload = {"edges": [{"edge_id": e.edge_id,
                                  "parent": self.data.describe(view.nodes[e.parent_id]),
                                  "child": self.data.describe(view.nodes[e.child_id])} for e in batch]}
            result, ref = self.model("validate_edges", payload,
                                     lambda r: containment_checks(r, {e.edge_id for e in batch}))
            checks = {row["edge_id"]: {**row, "ref": ref} for row in result["checks"]} if result else {}
            self.trace("concept-validation", phase=phase, checks=checks,
                       unchecked=sorted({e.edge_id for e in batch} - checks.keys()))
            self.validations.update(checks)
            for proposed in batch:
                checked = checks.get(proposed.edge_id)
                if checked and checked["verdict"] == "valid":
                    validated.append(edge(proposed.parent_id, proposed.child_id, checked["reason"], ref,
                                          self.config["llm"]["stages"]["validate_edges"]["model"]))
        return validated

    def distinct_or_reuse(self, view, node):
        ranked = self.rank([node], view, self.scope | view.roots, self.config["reuse_candidates"])
        eligible = {n.concept_id for n, _ in ranked}
        payload = {"comparisons": [{"concept": self.data.describe(node),
                                    "similar_existing": [self.data.describe(n) for n, _ in ranked]}]}
        result, ref = self.model("audit_duplicates", payload,
                                 lambda r: duplicate_checks(r, {node.concept_id: eligible}))
        if not result:
            raise CandidateError("No valid duplicate check")
        checked = result["checks"][0]
        self.duplicates[node.concept_id] = {**checked, "ref": ref}
        self.trace("concept-identity", check=checked, ref=ref)
        if checked["verdict"] == "uncertain":
            raise CandidateError("Uncertain concept identity; held for review")
        if checked["verdict"] == "duplicate":
            if checked["duplicate_of"] in view.roots:
                raise CandidateError("Proposed intermediate merely duplicates a root")
            return checked["duplicate_of"]
        return None

    def place(self, state, target_id, phase, extra=()):
        view = GraphView(state)
        target = view.nodes[target_id]
        forbidden = reachable(view.children, target_id) | {target_id}
        eligible = (self.scope | view.roots) - forbidden
        ranked = self.rank([target], view, eligible, self.config["parent_candidates"])
        shortlist = (view.roots | set(extra) | view.parents[target_id] |
                     {node.concept_id for node, _ in ranked}) - forbidden
        payload = {"target": self.data.describe(target, evidence=True),
                   "target_ancestry": self.path_context(view, target),
                   "target_children": [self.data.describe(view.nodes[n]) for n in sorted(view.children[target_id])],
                   "roots": self.root_context(view),
                   "eligible_parents": [{**self.data.describe(view.nodes[n]),
                                         **self.path_context(view, view.nodes[n])} for n in sorted(shortlist)]}
        result, ref = self.model("placement", payload, lambda r: placement_decision(r, shortlist))
        self.trace("placement", phase=phase, target_id=target_id, ref=ref, decision=result)
        return result, ref

    def apply_proposal(self, proposal, ref):
        view = GraphView(self.taxonomy)
        node = None
        parent_id = proposal["reuse_id"]
        children = [view.nodes[row["id"]] for row in proposal["children"]]
        if parent_id is None:
            node = intermediate(self.taxonomy, self.data, proposal, children, ref)
            parent_id = self.distinct_or_reuse(view, node)
            if parent_id is not None:
                node = None
            else:
                parent_id = node.concept_id
                if len(self.scope - self.data.working_ids) >= self.config["max_new_nodes"]:
                    raise CandidateError("maximum_new_nodes")
        links = [edge(parent_id, child.concept_id, "proposal", ref,
                      self.config["llm"]["stages"]["group"]["model"])
                 for child in children if child.concept_id != parent_id and
                 child.concept_id not in view.children.get(parent_id, set())]
        if not links:
            raise CandidateError("No new child links")
        # This draft is never committed. It only supplies definitions to validation.
        draft = update_graph(self.taxonomy, node, links)
        links = self.validate_edges(draft, links, "children")
        if len(links) < (2 if node else 1):
            raise CandidateError("Insufficient independently validated strict children")
        kept = {e.child_id for e in links}
        if node is not None:
            node = intermediate(self.taxonomy, self.data, proposal,
                                [view.nodes[n] for n in sorted(kept)], ref)
        draft = update_graph(self.taxonomy, node, links)
        old_parents = set().union(*(view.parents[n] for n in kept))
        placement, placement_ref = self.place(draft, parent_id, "per_round", old_parents)
        parent_links = self.placement_edges(GraphView(draft), parent_id, placement, placement_ref)
        parent_links = self.validate_edges(draft, parent_links, "placement")
        # Validate the entire batch before any mutation, including cycles formed
        # by multiple individually plausible links.
        after, removed = insert_transaction(self.taxonomy, node, links + parent_links)
        self.remember_placement(parent_id, placement, placement_ref)
        if placement and not parent_links and not GraphView(draft).parents[parent_id]:
            self.abstentions[parent_id] = {"reason": "No independently validated placement", "ref": placement_ref}
        if node:
            self.scope.add(parent_id)
        self.rewired.extend(removed)
        self.record_update(after, node, links + parent_links, "five_step_insertion")
        self.trace("hierarchy-insertion", parent_id=parent_id, removed_edges=removed,
                   retained_children=sorted(kept), excluded_children=sorted({n.concept_id for n in children} - kept))
        after_view = GraphView(after)
        affected = {parent_id} | kept | old_parents | {e.parent_id for e in parent_links}
        for n in list(affected):
            affected.update(after_view.children[n])
            affected.update(after_view.parents[n])
        self.dirty.update(affected & self.scope)
        return parent_id

    def induction(self):
        for round_number in range(1, self.config["max_rounds"] + 1):
            view = GraphView(self.taxonomy)
            ids = sorted(self.scope)
            if len(ids) < 2:
                return "fewer_than_two_concepts"
            anchors = self.dirty & self.scope
            if not anchors:
                return "no_affected_neighborhood"
            self.dirty.clear()
            vectors = self.embeddings.for_nodes([view.nodes[n] for n in [*ids, *self.data.roots]])
            forbidden = {(n, child) for n in ids for child in reachable(view.children, n)}
            discovery, cluster = self.config["discovery"], self.config["clustering"]
            groups, profiles = knn_candidates(ids, vectors[:len(ids)], vectors[len(ids):],
                k=discovery["k"], cutoff=cluster["distance_cutoff"], root_weight=cluster["root_weight"],
                max_group_size=self.config["max_group_size"], block_size=discovery["block_size"],
                anchors=anchors, forbidden=forbidden)
            groups = [g for g in groups if g.members not in self.attempted]
            pending = [(g, 0) for g in reversed(groups)] + self.retry_pending
            self.retry_pending = []
            stats = {"round": round_number, "frontier_size": len(anchors), "groups": 0,
                     "actions": {}, "committed_new": 0, "committed_reuse": 0,
                     "candidate_groups": len(pending)}
            self.rounds.append(stats)
            self.trace("candidate-discovery", round=round_number, anchors=sorted(anchors),
                       groups=[{"ids": g.members, "distance": g.height} for g in groups])
            profile_map = {n: dict(zip(self.data.roots, p.tolist())) for n, p in zip(ids, profiles)}
            counts = Counter()
            while pending and stats["groups"] < discovery["max_candidates_per_round"]:
                used = len(list(self.journal.directory.glob('*.json')))
                limit = self.journal.maximum_calls - self.config.get('placement_reserved_calls', 0)
                if used >= limit:
                    stats['pending_candidates'] = len(pending)
                    self.trace('local-refinement-budget', **stats)
                    return 'induction_budget_reserved_for_placement'
                candidate, retry = pending.pop()
                if candidate.members in self.attempted:
                    continue
                current = GraphView(self.taxonomy)
                # Recheck against changes committed earlier in this round.
                if any(b in reachable(current.children, a) for a in candidate.members for b in candidate.members if a != b):
                    continue
                self.attempted.add(candidate.members)
                value, ref, action = self.propose_group(current, candidate, profile_map)
                counts[action] += 1
                stats["groups"] += 1
                stats["actions"] = dict(counts)
                self.trace("intermediate-proposal", round=round_number, ids=candidate.members,
                           retry=retry, action=action, decision=value, ref=ref)
                committed = set()
                for proposal in value["proposals"] if value else ():
                    before = len(self.scope)
                    try:
                        parent = self.apply_proposal(proposal, ref)
                        committed.update(GraphView(self.taxonomy).children[parent])
                    except CandidateError as error:
                        self.trace("transaction-rejected", round=round_number, ref=ref, error=str(error))
                    else:
                        stats["committed_new" if len(self.scope) > before else "committed_reuse"] += 1
                remaining = set(candidate.members) - committed
                if retry < self.config["rejected_group_split_depth"]:
                    pending.extend((part, retry + 1) for part in reversed(candidate.parts)
                                   if set(part.members) <= remaining and part.members not in self.attempted)
            stats["pending_candidates"] = len(pending)
            if pending:
                self.dirty.update(n for candidate, _ in pending for n in candidate.members)
                # Preserve smaller retries even if the round's work quota is hit.
                self.retry_pending = pending
            stats["frontier_after"] = GraphView(self.taxonomy).frontier(self.scope)
            self.trace("local-refinement", **stats, dirty=sorted(self.dirty))
            if not stats["candidate_groups"]:
                return "no_new_local_candidate"
            if not stats["committed_new"] + stats["committed_reuse"] and not pending:
                return "no_valid_local_refinement"
        return "maximum_rounds"

    def place_unrooted(self):
        # Cover isolated leaves too, but only via separately validated placements.
        for target in GraphView(self.taxonomy).frontier(self.scope):
            view = GraphView(self.taxonomy)
            if target in view.rooted:
                continue
            value, ref = self.place(self.taxonomy, target, "unrooted_frontier")
            links = self.placement_edges(view, target, value, ref)
            links = self.validate_edges(self.taxonomy, links, "unrooted_frontier")
            if links:
                after, removed = insert_transaction(self.taxonomy, None, links)
                self.rewired.extend(removed)
                self.record_update(after, None, links, "unrooted_frontier")
            self.remember_placement(target, value, ref)
            if value and not links:
                self.abstentions[target] = {"reason": "No independently validated placement", "ref": ref}

    def audit(self):
        # Semantics were checked before insertion. A later comparison uses the
        # same independent rubric for both saved runs and does not alter either.
        final, removed = reduce_provisional_shortcuts(self.taxonomy)
        self.rewired.extend(removed)
        view = refresh(final)
        original_nodes = {n.concept_id for n in self.data.initial.nodes}
        for mapping in final.concept_evidence_mapping:
            node_id = mapping["concept_id"]
            if node_id in original_nodes:
                continue
            sources = [view.nodes[n] for n in reachable(view.children, node_id) & original_nodes]
            node = view.nodes[node_id]
            node.supporting_event_ids = sorted({e for n in sources for e in n.supporting_event_ids})
            node.evidence = sorted({e for n in sources for e in n.evidence})
            mapping.update(event_ids=node.supporting_event_ids, evidence_ids=node.evidence,
                           via_child_ids=sorted(view.children[node_id]))
        assert_preserved(self.data.initial, final, accepted=True)
        original = {e.edge_id for e in self.data.initial.parent_child_edges}
        unchecked = sorted(e.edge_id for e in final.parent_child_edges
                           if e.edge_id not in original and self.validations.get(e.edge_id, {}).get("verdict") != "valid")
        if unchecked:
            raise CandidateError("Unvalidated edge reached committed graph")
        return final, {"edge_verdicts": self.validations, "unchecked_edges": [],
                       "duplicate_verdicts": self.duplicates, "unchecked_new_concepts": [],
                       "original_edges_for_review": [], "quarantined_nodes": [],
                       "removed_edges": [e["edge_id"] for e in self.rewired],
                       "redundant_edges": [e["edge_id"] for e in self.rewired],
                       "validation_timing": "before_insertion"}
