"""Convert independently reported containment checks to a pair relation."""
from event_taxonomy.induction.contracts import obj, require, text

SCOPE_RELATIONS = {
    ("yes", "yes"): "equivalent",
    ("yes", "no"): "narrower",
    ("no", "yes"): "broader",
    ("no", "no"): "related_but_different",
}


def check_scope_decisions(response, pairs):
    obj(response, {"decisions"})
    require(isinstance(response["decisions"], list), "decisions must be an array")
    expected, decisions = set(pairs), {}
    for row in response["decisions"]:
        obj(row, {"left_id", "right_id", "left_to_right", "right_to_left"})
        text(row["left_id"])
        text(row["right_id"])
        pair = (row["left_id"], row["right_id"])
        require(pair in expected and pair not in decisions, "Unknown, reversed or duplicate pair")
        forward, reverse = row["left_to_right"], row["right_to_left"]
        for check in (forward, reverse):
            obj(check, {"verdict", "reason"})
            text(check["verdict"])
            text(check["reason"])
            require(check["verdict"] in {"yes", "no", "uncertain"}, "Unknown containment verdict")
        relation = SCOPE_RELATIONS.get((forward["verdict"], reverse["verdict"]), "uncertain")
        decisions[pair] = {
            "left_id": pair[0], "right_id": pair[1], "relation": relation, "needs_evidence": False,
            "reason": f"Left to right: {forward['reason']} Right to left: {reverse['reason']}",
            "scope_checks": {"left_to_right": forward, "right_to_left": reverse},
        }
    require(decisions.keys() == expected, "Return a decision for every requested pair")
    return decisions
