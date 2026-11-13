"""Proposal, containment and cross-group checks for upward induction."""

from pathlib import Path
from xml.etree import ElementTree

from event_taxonomy.induction.contracts import ContractError, ids, obj, require, text
from event_taxonomy.models import stable_id

PROMPT_PATH = Path(__file__).with_name("prompts") / "upward.xml"
PROMPTS = {element.tag: ElementTree.tostring(element, encoding="unicode").strip()
           for element in ElementTree.parse(PROMPT_PATH).getroot()}
TRUTH_VALUES = ("true", "false", "uncertain")
VERDICTS = ("pass", "fail", "uncertain")
SCOPE_VALUES = ("has_extra_cases", "no_extra_cases", "uncertain")


def checked_call(journal, stage, payload, validate):
    try:
        return validate(journal.call(stage, payload)), journal.last_ref
    except ContractError as error:
        journal.mark_invalid(str(error))
        payload = {**payload, "output_format_correction": str(error)}
    try:
        return validate(journal.call(stage, payload)), journal.last_ref
    except ContractError as error:
        journal.mark_invalid(str(error))
        raise


def check_proposals(response, groups):
    obj(response, ("groups",))
    require(isinstance(response["groups"], list), "Expected an array of groups")
    returned = set()
    for group in response["groups"]:
        obj(group, ("group_id", "reason", "proposals"))
        group_id = group["group_id"]
        text(group_id)
        text(group["reason"])
        require(group_id in groups and group_id not in returned, "Unknown or repeated group ID")
        returned.add(group_id)
        require(isinstance(group["proposals"], list), "Expected an array of proposals")
        available = set(groups[group_id])
        for row in group["proposals"]:
            obj(row, ("name", "definition", "reuse_id", "child_ids"))
            reuse_id = row["reuse_id"]
            if reuse_id is None:
                text(row["name"])
                text(row["definition"])
            else:
                text(reuse_id)
                require(reuse_id in available and row["name"] is None and row["definition"] is None,
                        "Reuse must name an available member and keep its definition")
                available.remove(reuse_id)
            ids(row["child_ids"], available, minimum=1 if reuse_id else 2)
            available.difference_update(row["child_ids"])
    require(returned == groups.keys(), "Return one entry for every requested group")
    return response["groups"]


def describe(node):
    return {"id": node.concept_id, "name": node.concept_label, "definition": node.concept_description}


def review_items(groups, nodes, descendants, decision_ref):
    items = []
    for group in groups:
        for row in group["proposals"]:
            child_ids = sorted(row["child_ids"])
            reused = nodes.get(row["reuse_id"])
            name = reused.concept_label if reused else row["name"].strip()
            definition = reused.concept_description if reused else row["definition"].strip()
            proposal_id = stable_id("proposal", row["reuse_id"], name, definition, child_ids)
            leaf_ids = set().union(*(descendants[child_id] for child_id in child_ids)) - set(child_ids)
            items.append({
                "proposal_id": proposal_id, "group_id": group["group_id"],
                "name": name, "definition": definition, "reuse_id": row["reuse_id"],
                "children": [describe(nodes[child_id]) for child_id in child_ids],
                "descendant_leaves": [describe(nodes[leaf_id]) for leaf_id in sorted(leaf_ids)],
                "proposal_ref": decision_ref,
            })
    return items


def by_review_id(rows, field, expected):
    require(isinstance(rows, list), "Expected an array of reviews")
    found = {}
    for row in rows:
        require(isinstance(row, dict), "Expected a review object")
        identifier = row.get(field)
        text(identifier)
        require(identifier in expected and identifier not in found, "Unknown or repeated " + field)
        found[identifier] = row
    require(found.keys() == expected, "Missing review for one or more " + field + " values")
    return found


def check_counterexample(check, forward):
    text(check["reason"])
    if forward == "false":
        require(isinstance(check["counterexample"], str) and check["counterexample"].strip(),
                "Give a counterexample when a child or leaf falls outside the parent")
    elif forward == "true":
        require(check["counterexample"] is None, "A counterexample contradicts full coverage")
    elif check["counterexample"] is not None:
        text(check["counterexample"])


def check_parent_reviews(response, proposals):
    obj(response, ("parent_reviews",))
    reviews = by_review_id(response["parent_reviews"], "proposal_id", proposals.keys())
    for proposal_id, review in reviews.items():
        obj(review, ("proposal_id", "verdict", "definition_review", "edge_reviews", "leaf_reviews",
                     "suggested_action", "suggested_definition"))
        require(review["verdict"] in ("accept", "revise", "reject", "uncertain"), "Unknown parent verdict")
        text(review["definition_review"])
        text(review["suggested_action"])
        if review["suggested_definition"] is not None:
            text(review["suggested_definition"])
        proposal = proposals[proposal_id]
        edges = by_review_id(review["edge_reviews"], "child_id", {n["id"] for n in proposal["children"]})
        leaves = by_review_id(review["leaf_reviews"], "leaf_id", {n["id"] for n in proposal["descendant_leaves"]})
        for edge in edges.values():
            obj(edge, ("child_id", "child_entails_parent", "broader_scope", "parent_only_example",
                       "scope_reason", "reason", "counterexample"))
            forward, scope = edge["child_entails_parent"], edge["broader_scope"]
            require(forward in TRUTH_VALUES and scope in SCOPE_VALUES, "Invalid direct containment verdict")
            if scope == "has_extra_cases":
                require(isinstance(edge["parent_only_example"], str) and edge["parent_only_example"].strip(),
                        "Give a parent-only example when broader_scope is has_extra_cases")
                require(edge["scope_reason"] is None, "Use null for scope_reason when giving a parent-only example")
            else:
                require(edge["parent_only_example"] is None, "Leave parent_only_example null without extra cases")
                require(isinstance(edge["scope_reason"], str) and edge["scope_reason"].strip(),
                        "Explain in scope_reason why the parent is not broader or why its scope is uncertain")
            check_counterexample(edge, forward)
            edge["verdict"] = ("fail" if forward == "false" or scope == "no_extra_cases" else
                               "pass" if (forward, scope) == ("true", "has_extra_cases") else "uncertain")
        for leaf in leaves.values():
            obj(leaf, ("leaf_id", "entails_parent", "reason", "counterexample"))
            require(leaf["entails_parent"] in TRUTH_VALUES, "Invalid descendant containment verdict")
            check_counterexample(leaf, leaf["entails_parent"])
            leaf["verdict"] = dict(zip(TRUTH_VALUES, VERDICTS))[leaf["entails_parent"]]
        if review["verdict"] == "accept":
            require(review["suggested_definition"] is None and
                    all(check["verdict"] == "pass" for check in [*edges.values(), *leaves.values()]),
                    "An accepted parent must pass every child and descendant check without a definition revision")
        review["passes"] = review["verdict"] == "accept"
    return reviews


def check_cross_review(response, targets, known):
    obj(response, ("checks",))
    checks = by_review_id(response["checks"], "proposal_id", targets)
    for identifier, check in checks.items():
        obj(check, ("proposal_id", "verdict", "related_id", "reason"))
        require(check["verdict"] in VERDICTS, "Unknown cross-group verdict")
        text(check["reason"])
        related = check["related_id"]
        if related is not None:
            text(related)
            require(related != identifier and related in known, "Unknown related concept or proposal ID")
    return checks
