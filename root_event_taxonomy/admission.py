"""Review input concept eligibility and every source-event link without rewriting data."""

from copy import deepcopy
from pathlib import Path

from event_taxonomy.adapters.common import require_text
from event_taxonomy.induction.contracts import ContractError, obj, require, text

from .leaf_deduplication import validate_leaf_links
from .upward_review import by_review_id, checked_call


PROMPT_PATH = Path(__file__).with_name("prompts") / "concept-admission.xml"
SCOPE_POLICIES = ("reported_events", "research_statements")
VERDICTS = ("pass", "fail", "uncertain")
CONCEPT_CHECKS = ("within_scope", "reusable_type", "definition_clarity")
INSTANCE_CHECKS = ("evidence_support", "instance_of")


def combined_verdict(verdicts):
    """Any explicit failure or unresolved judgment prevents admission."""
    values = set(verdicts)
    return "fail" if "fail" in values else "uncertain" if "uncertain" in values else "pass"


def admission_inputs(dataset):
    """Return detached review payloads, retaining all distinct event witnesses."""
    leaves, _ = validate_leaf_links(dataset)
    sources = {row["source_concept_id"]: row for row in dataset["source_concepts"]}
    events = {row["source_id"]: row for row in dataset["source_events"]}
    result = []
    for leaf_id, leaf in leaves.items():
        witnesses = []
        for event_id in leaf["source_events"]:
            event = events[event_id]
            witnesses.append({
                "event_id": event_id,
                "event": require_text(event.get("Event"), "Source Event"),
                "evidence": require_text(event.get("Evidence"), "Source Evidence"),
                "trigger": event.get("Trigger"),
                "source_pmid": event.get("source_pmid"),
            })
        result.append({
            "leaf_id": leaf_id, "name": leaf["concept_name"], "definition": leaf["description"],
            "source_concepts": [{
                "source_concept_id": member,
                "event_id": sources[member]["source_event"],
                "name": sources[member].get("concept_name"),
                "definition": sources[member].get("description"),
            } for member in leaf["source_concepts"]],
            "source_events": witnesses,
        })
    return deepcopy(result)


def check_admission_reviews(response, expected):
    """Enforce complete checks, valid IDs, and a conservative combined verdict."""
    obj(response, ("reviews",))
    reviews = by_review_id(response["reviews"], "leaf_id", expected.keys())
    for leaf_id, review in reviews.items():
        obj(review, ("leaf_id", "concept_checks", "instance_reviews", "verdict", "reason"))
        obj(review["concept_checks"], CONCEPT_CHECKS)
        checks = list(review["concept_checks"].values())
        event_ids = {row["event_id"] for row in expected[leaf_id]["source_events"]}
        instances = by_review_id(review["instance_reviews"], "event_id", event_ids)
        for instance in instances.values():
            obj(instance, ("event_id", *INSTANCE_CHECKS))
            checks.extend(instance[key] for key in INSTANCE_CHECKS)
        for check in checks:
            obj(check, ("verdict", "reason"))
            require(check["verdict"] in VERDICTS, "Unknown admission check verdict")
            text(check["reason"])
        require(review["verdict"] in VERDICTS, "Unknown admission verdict")
        require(review["verdict"] == combined_verdict(check["verdict"] for check in checks),
                "Admission verdict must reflect every concept and source-event check")
        text(review["reason"])
    return reviews


def review_admission(dataset, journal, batch_size=4, scope_policy="reported_events"):
    """Review all normalized inputs, leaving rejected/uncertain records intact.

    The caller saves this report next to the unchanged dataset and uses only
    ``accepted_leaf_ids`` for induction. Journaled responses are reusable on resume.
    """
    if scope_policy not in SCOPE_POLICIES:
        raise ValueError("Unknown concept admission scope policy: " + str(scope_policy))
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Admission batch_size must be a positive integer.")
    inputs = admission_inputs(dataset)
    reviews = []

    def review_batch(batch):
        expected = {row["leaf_id"]: row for row in batch}
        payload = {"scope_policy": scope_policy, "concepts": batch}
        # A durable truncation is enough to avoid repeating an oversized request
        # on resume. Each smaller request remains independently fully validated.
        if len(batch) > 1 and journal.was_truncated("admission", payload):
            middle = len(batch) // 2
            review_batch(batch[:middle])
            review_batch(batch[middle:])
            return
        try:
            decisions, ref = checked_call(
                journal, "admission", payload,
                lambda value: check_admission_reviews(value, expected),
            )
        except ContractError:
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            review_batch(batch[:middle])
            review_batch(batch[middle:])
            return
        reviews.extend({**decisions[row["leaf_id"]], "review_ref": ref} for row in batch)

    for start in range(0, len(inputs), batch_size):
        review_batch(inputs[start:start + batch_size])
    return {
        "policy": scope_policy,
        "accepted_leaf_ids": [row["leaf_id"] for row in reviews if row["verdict"] == "pass"],
        "held_leaf_ids": [row["leaf_id"] for row in reviews if row["verdict"] != "pass"],
        "reviews": reviews,
    }
