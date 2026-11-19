"""Deterministic, offline fixture for the defense evidence review."""

from __future__ import annotations

from cognitive_agent.schema.schema_profile import schema_profile_from_dict
from cognitive_agent.verifier import KGVerifier


class _OfflineGraph:
    is_connected = False


def demo_results() -> dict:
    """Build a small, explicitly synthetic review with the real verifier."""

    trial_profile = schema_profile_from_dict({
        "name": "IllustrativeTrial",
        "contract_version": "illustrative-trial-v1",
        "semantic_target": "CURRENT_FINDING",
        "entity_types": ["ClinicalTrial", "Outcome"],
        "article_quality_mode": "none",
        "predicates": {
            "REPORTS_OUTCOME": {
                "allowed_signatures": [["ClinicalTrial", "Outcome"]],
                "explicit_patterns": [r"\breported\b"],
                "exclusion_patterns": [r"\bfailed to report\b"],
            },
        },
    })
    examples = (
        {
            "pmid": "DEMO-LIVER-001",
            "title": "Synthetic example · LiverKG compatible edge",
            "text": (
                "TITLE: Liver HCC study\n"
                "ABSTRACT: RESULTS: TP53 was associated with HCC."
            ),
            "predicate": "ASSOCIATED_WITH",
            "subject": "TP53",
            "subject_type": "Gene",
            "object": "HCC",
            "object_type": "Disease",
            "evidence": "TP53 was associated with HCC.",
            "profile": None,
        },
        {
            "pmid": "DEMO-CUSTOM-002",
            "title": "Synthetic example · Custom outcome schema",
            "text": (
                "TITLE: Illustrative trial report\n"
                "ABSTRACT: RESULTS: Trial Orion reported improved endurance."
            ),
            "predicate": "REPORTS_OUTCOME",
            "subject": "Trial Orion",
            "subject_type": "ClinicalTrial",
            "object": "improved endurance",
            "object_type": "Outcome",
            "evidence": "Trial Orion reported improved endurance.",
            "profile": trial_profile,
        },
        {
            "pmid": "DEMO-REVIEW-003",
            "title": "Synthetic example · Evidence needs review",
            "text": (
                "TITLE: Illustrative trial report\n"
                "ABSTRACT: RESULTS: Trial Vega measured endurance after exercise."
            ),
            "predicate": "REPORTS_OUTCOME",
            "subject": "Trial Vega",
            "subject_type": "ClinicalTrial",
            "object": "endurance",
            "object_type": "Outcome",
            "evidence": "Trial Vega measured endurance after exercise.",
            "profile": trial_profile,
        },
        {
            "pmid": "DEMO-REJECT-004",
            "title": "Synthetic example · Explicit exclusion",
            "text": (
                "TITLE: Illustrative trial report\n"
                "ABSTRACT: RESULTS: Trial Vega failed to report an endurance benefit."
            ),
            "predicate": "REPORTS_OUTCOME",
            "subject": "Trial Vega",
            "subject_type": "ClinicalTrial",
            "object": "endurance benefit",
            "object_type": "Outcome",
            "evidence": "Trial Vega failed to report an endurance benefit.",
            "profile": trial_profile,
        },
    )

    records = []
    for example in examples:
        profile = example["profile"]
        verifier = KGVerifier(
            _OfflineGraph(),
            verification_policy="tiered-v2",
            schema_profile=profile,
            dataset_only=profile is not None,
        )
        entities = [
            {"mention": example["subject"], "type": example["subject_type"], "attributes": {}},
            {"mention": example["object"], "type": example["object_type"], "attributes": {}},
        ]
        relation = {
            "subject": example["subject"],
            "subject_type": example["subject_type"],
            "predicate": example["predicate"],
            "object": example["object"],
            "object_type": example["object_type"],
            "evidence": example["evidence"],
            "claim_role": "CURRENT_FINDING",
            "direction": "none" if profile else "positive",
            "negated": False,
            "uncertain": False,
        }
        verified = verifier.verify(
            entities,
            [relation],
            pmid=example["pmid"],
            text=example["text"],
        )
        records.append({
            "pmid": example["pmid"],
            "title": example["title"],
            "abstract_length": len(example["text"]),
            "phases": {"verification": verified.to_dict()},
        })

    return {
        "report": {
            "run_id": "synthetic-defense-demo",
            "schema_profile": {
                "name": "Mixed built-in and illustrative dataset schemas",
                "synthetic": True,
                "neo4j_connected": False,
                "article_count": len(records),
            },
        },
        "records": records,
    }
