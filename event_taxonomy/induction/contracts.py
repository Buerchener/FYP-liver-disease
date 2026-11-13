"""Strict model contracts. Confidence never substitutes for structural checks."""
import math


class ContractError(ValueError):
    pass


class CandidateError(ValueError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


class ContextLimit(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ContractError(message)


def obj(value, keys):
    require(isinstance(value, dict) and set(value) == set(keys), "Unexpected or missing JSON object fields")


def text(value):
    require(isinstance(value, str) and bool(value.strip()), "Expected a nonempty string")


def confidence(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            "confidence must be a finite number in [0,1]")


def ids(value, known, minimum=2):
    require(isinstance(value, list) and len(value) >= minimum, "Subgroup must have at least two children")
    require(all(isinstance(x, str) for x in value) and len(set(value)) == len(value), "Duplicate/invalid child ID")
    require(set(value).issubset(known), "Subgroup references non-direct or unknown children")


def diagnosis(value, child_ids):
    obj(value, ["need_induction", "candidate_subgroups", "structural_issues", "reasoning", "confidence"])
    require(type(value["need_induction"]) is bool, "need_induction must be boolean")
    confidence(value["confidence"]); text(value["reasoning"])
    require(isinstance(value["structural_issues"], list), "structural_issues must be an array")
    for issue in value["structural_issues"]:
        text(issue)
    require(isinstance(value["candidate_subgroups"], list), "candidate_subgroups must be an array")
    seen = set()
    for group in value["candidate_subgroups"]:
        obj(group, ["group_id", "child_ids", "reasoning"])
        text(group["group_id"]); text(group["reasoning"])
        require(group["group_id"] not in seen, "Duplicate group ID")
        seen.add(group["group_id"]); ids(group["child_ids"], child_ids)
    require(value["need_induction"] or not value["candidate_subgroups"], "Negative diagnosis cannot propose groups")
    return value


def proposal(value, groups):
    obj(value, ["intermediates", "reasoning", "confidence"])
    confidence(value["confidence"]); text(value["reasoning"])
    require(isinstance(value["intermediates"], list), "intermediates must be an array")
    permitted = {frozenset(g["child_ids"]) for g in groups}
    seen_ids, seen_groups = set(), set()
    for node in value["intermediates"]:
        obj(node, ["proposal_id", "label", "description", "child_ids"])
        for key in ["proposal_id", "label", "description"]:
            text(node[key])
        require(node["proposal_id"] not in seen_ids, "Duplicate proposal ID")
        seen_ids.add(node["proposal_id"])
        ids(node["child_ids"], set().union(*permitted) if permitted else set())
        group = frozenset(node["child_ids"])
        require(group in permitted, "Proposal changed the supplied cluster membership")
        require(group not in seen_groups, "Duplicate intermediate for the same subgroup")
        seen_groups.add(group)
    return value


def validation(value, added_edges, after_children):
    obj(value, ["decision", "edge_checks", "improves_structure", "duplicates_existing_concepts",
                "remaining_subgroups", "reasoning", "confidence"])
    require(isinstance(value["decision"], str) and value["decision"] in {"REJECT", "ACCEPT_AND_CONTINUE", "ACCEPT_AND_STOP"}, "Unknown validation decision")
    confidence(value["confidence"]); text(value["reasoning"])
    for key in ["improves_structure", "duplicates_existing_concepts"]:
        require(type(value[key]) is bool, key + " must be boolean")
    require(isinstance(value["edge_checks"], list), "edge_checks must be an array")
    expected = {(e["parent_id"], e["child_id"]) for e in added_edges}
    seen = set()
    for check in value["edge_checks"]:
        obj(check, ["parent_id", "child_id", "strict_subsumption", "reasoning"])
        text(check["parent_id"]); text(check["child_id"]); text(check["reasoning"])
        pair = (check["parent_id"], check["child_id"])
        require(pair in expected and pair not in seen, "Invalid or duplicated edge check")
        require(type(check["strict_subsumption"]) is bool, "strict_subsumption must be boolean")
        seen.add(pair)
    require(isinstance(value["remaining_subgroups"], list), "remaining_subgroups must be an array")
    for group in value["remaining_subgroups"]:
        ids(group, after_children)
    if value["decision"] != "REJECT":
        require(seen == expected and all(c["strict_subsumption"] for c in value["edge_checks"]),
                "Acceptance must verify every added edge as strict subsumption")
        require(value["improves_structure"] and not value["duplicates_existing_concepts"],
                "Acceptance requires nonredundant structural improvement")
    if value["decision"] == "ACCEPT_AND_STOP":
        require(not value["remaining_subgroups"], "Stop decision contradicts remaining subgroups")
    return value
