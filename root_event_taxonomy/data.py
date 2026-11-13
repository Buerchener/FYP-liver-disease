"""Load a Task 1 snapshot while keeping original concepts and provenance intact."""
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

from event_taxonomy.induction.graph import load_taxonomy
from event_taxonomy.models import Document, EventInstance, Evidence, ImportBundle, SourceRef
from event_taxonomy.validation import validate_bundle

VERIFIED = {"verified_exact", "verified_whitespace"}


def validate_source(dataset, taxonomy):
    docs = [
        Document(**{**row, "source_refs": [SourceRef(**ref) for ref in row["source_refs"]]})
        for row in dataset["documents"]
    ]
    events = [
        EventInstance(**{**row, "source_ref": SourceRef(**row["source_ref"]),
                         "validation_issues": deepcopy(row["validation_issues"])})
        for row in dataset["events"]
    ]
    evidence = [
        Evidence(**{**row, "source_ref": SourceRef(**row["source_ref"])})
        for row in dataset["evidence"]
    ]
    if not docs or not events or not evidence:
        raise ValueError("The snapshot must contain documents, events, and evidence.")
    # The export carries source references. The bundle validator also verifies the
    # copied document hashes, evidence spans and source/instance identities.
    source_hashes = {ref.file_sha256 for doc in docs for ref in doc.source_refs}
    source_hashes.update(row.source_ref.file_sha256 for row in events + evidence)
    validate_bundle(
        ImportBundle(documents=docs, events=events, evidence=evidence),
        [SimpleNamespace(sha256=sha256) for sha256 in source_hashes],
    )
    synthetic = dataset["data_origin"] == "synthetic"
    if {event.data_origin == "synthetic" for event in events} != {synthetic} or {
        doc.data_origin == "synthetic" for doc in docs
    } != {synthetic}:
        raise ValueError("Synthetic and real events must be run separately.")

    event_evidence = {e.event_id: set(e.evidence_ids) for e in events}
    roots = set(taxonomy.predefined_roots)
    nodes = {n.concept_id: n for n in taxonomy.nodes}
    mappings = {row["concept_id"]: row for row in taxonomy.concept_evidence_mapping}
    if len(mappings) != len(taxonomy.concept_evidence_mapping) or not mappings.keys() <= nodes.keys():
        raise ValueError("Invalid concept evidence mapping IDs.")
    for node_id, node in nodes.items():
        event_ids = set(node.supporting_event_ids)
        if not isinstance(node.concept_label, str) or not node.concept_label.strip():
            raise ValueError("Missing concept name: " + node_id)
        if node.concept_role != "event_type" or not event_ids <= event_evidence.keys():
            raise ValueError("Invalid event concept or supporting event: " + node_id)
        evidence_ids = set().union(*(event_evidence[e] for e in event_ids))
        if set(node.evidence) != evidence_ids:
            raise ValueError("Concept evidence differs from its supporting events: " + node_id)
        if node_id not in roots and (not event_ids or node_id not in mappings):
            raise ValueError("Missing concept provenance: " + node_id)
        if node_id in mappings:
            row = mappings[node_id]
            if set(row["event_ids"]) != event_ids or set(row["evidence_ids"]) != evidence_ids:
                raise ValueError("Concept evidence mapping differs from node: " + node_id)

    seen = set()
    for link in taxonomy.instance_concept_links:
        e, n = link["event_id"], link["concept_id"]
        if (e, n) in seen or n not in nodes or e not in event_evidence:
            raise ValueError("Invalid instance-concept IDs.")
        if link["relation"] != "IS_INSTANCE_OF" or not set(link["evidence_ids"]) <= event_evidence[e]:
            raise ValueError("Invalid instance-concept relation or provenance.")
        seen.add((e, n))
    for node_id in taxonomy.frozen_input_concept_ids:
        if any((e, node_id) not in seen for e in nodes[node_id].supporting_event_ids):
            raise ValueError("Missing original instance-concept links: " + node_id)


class InputData:
    def __init__(self, dataset, selected_ids, overlay, evidence_per_concept):
        self.dataset = dataset
        self.initial = load_taxonomy(dataset["taxonomy"])
        validate_source(dataset, self.initial)
        self.roots = tuple(self.initial.predefined_roots)
        frozen = set(self.initial.frozen_input_concept_ids)
        self.selected = set(selected_ids) if selected_ids is not None else frozen
        if not self.roots or not self.selected or not self.selected <= frozen:
            raise ValueError("Select at least one original input concept and supply predefined roots.")
        if selected_ids is not None and len(self.selected) != len(selected_ids):
            raise ValueError("Duplicate input concept IDs.")
        if not overlay.keys() <= self.selected:
            raise ValueError("Definition overrides must refer to selected original concepts.")
        self.overlay = overlay
        self.evidence = {e["evidence_id"]: e for e in dataset["evidence"]}
        self.events = {e["event_id"]: e for e in dataset["events"]}
        self.evidence_per_concept = evidence_per_concept
        nodes = {n.concept_id: n for n in self.initial.nodes}
        for node_id, entry in overlay.items():
            if not isinstance(entry, dict):
                raise ValueError("Invalid definition override: " + node_id)
            if not isinstance(entry.get("definition"), str) or not entry["definition"].strip():
                raise ValueError("Empty definition override: " + node_id)
            cited = entry.get("evidence_ids", [])
            if not isinstance(cited, list) or not cited or not all(isinstance(e, str) for e in cited):
                raise ValueError("An override needs a list of evidence IDs: " + node_id)
            if not set(cited) <= set(nodes[node_id].evidence) or any(
                self.evidence[e]["alignment_status"] not in VERIFIED for e in cited
            ):
                raise ValueError("An override needs verified evidence from the same concept: " + node_id)
        for node_id in self.selected | set(self.roots):
            if not self.definition(nodes[node_id]):
                raise ValueError("Concept needs a label and definition: " + node_id)
        self.existing_intermediates = {
            node_id for node_id, node in nodes.items()
            if node_id not in frozen and node_id not in self.roots and self.definition(node)
        }
        self.working_ids = self.selected | self.existing_intermediates

    def definition(self, node):
        override = self.overlay.get(node.concept_id)
        return (override["definition"] if override else node.concept_description or "").strip()

    def describe(self, node, evidence=False, excerpt_chars=900):
        result = {
            "id": node.concept_id,
            "name": node.concept_label,
            "definition": self.definition(node),
        }
        if node.aliases:
            result["aliases"] = node.aliases
        if evidence:
            override = self.overlay.get(node.concept_id, {})
            candidates = list(dict.fromkeys(override.get("evidence_ids", []) + sorted(node.evidence)))
            verified = [self.evidence[e] for e in candidates
                        if self.evidence[e]["alignment_status"] in VERIFIED and self.evidence[e]["text"]]
            samples = []
            for row in verified[:self.evidence_per_concept]:
                event = next(
                    (self.events[e] for e in node.supporting_event_ids
                     if row["evidence_id"] in self.events[e]["evidence_ids"]),
                    None,
                )
                samples.append({
                    "evidence_id": row["evidence_id"], "excerpt": row["text"][:excerpt_chars],
                    "event_id": event["event_id"] if event else None,
                    "event_text": event["event_text"][:400] if event else None,
                    "pmid": event["source_pmid"] if event else None,
                })
            result["evidence"] = samples
            result["verified_excerpts_omitted"] = len(verified) - len(samples)
            if override:
                result["definition_status"] = "provisional_evidence_based_overlay"
        return result

    def scope(self):
        selected = [n for n in self.initial.nodes if n.concept_id in self.selected]
        events = {e for n in selected for e in n.supporting_event_ids}
        evidence = {e for n in selected for e in n.evidence}
        return {
            "data_origin": self.dataset["data_origin"],
            "selected_input_ids": sorted(self.selected),
            "original_input_count": len(self.initial.frozen_input_concept_ids),
            "selected_input_count": len(selected),
            "selected_event_count": len(events),
            "selected_document_count": len({self.events[e]["document_id"] for e in events}),
            "selected_evidence_status": dict(Counter(self.evidence[e]["alignment_status"] for e in evidence)),
            "provisional_definition_count": len(self.overlay),
        }
