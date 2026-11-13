"""Integrity checks; these do not score biomedical or taxonomy semantics."""
from collections import Counter, defaultdict
import hashlib
from .models import to_dict
from .taxonomy.dag import validate_dag


def _unique(rows, field):
    result = {}
    for row in rows:
        key = getattr(row, field)
        if key in result:
            raise ValueError("Duplicate " + field + ": " + key)
        result[key] = row
    return result


def validate_bundle(bundle, sources):
    docs = _unique(bundle.documents, "document_id")
    events = _unique(bundle.events, "event_id")
    evidence = _unique(bundle.evidence, "evidence_id")
    _unique(bundle.candidates, "candidate_id")
    source_hashes = {s.sha256 for s in sources}
    for doc in docs.values():
        if doc.text_sha256 != hashlib.sha256(doc.text.encode("utf-8")).hexdigest():
            raise ValueError("Document checksum mismatch")
        if doc.data_origin == "synthetic" and doc.source_pmid is not None:
            raise ValueError("Synthetic PMID is forbidden")
        if not doc.source_refs or any(ref.file_sha256 not in source_hashes for ref in doc.source_refs):
            raise ValueError("Document provenance missing")
    for ev in evidence.values():
        if ev.document_id not in docs or ev.source_ref.file_sha256 not in source_hashes:
            raise ValueError("Evidence provenance cannot be resolved")
        doc = docs[ev.document_id]
        for span in ev.spans:
            start, end = span["start"], span["end"]
            if not 0 <= start < end <= len(doc.text):
                raise ValueError("Evidence offset outside document")
            if " ".join(doc.text[start:end].split()) != " ".join((ev.text or "").split()):
                raise ValueError("Evidence offset points to different text")
        if ev.alignment_status == "verified_exact" and (len(ev.spans) != 1 or
                doc.text[ev.spans[0]["start"]:ev.spans[0]["end"]] != ev.text):
            raise ValueError("False exact-evidence flag")
        if ev.alignment_status in {"verified_whitespace", "verified_exact"} and len(ev.spans) != 1:
            raise ValueError("Verified evidence must have one unambiguous span")
        if ev.alignment_status == "ambiguous" and len(ev.spans) < 2:
            raise ValueError("Ambiguous evidence must retain multiple spans")
        if ev.alignment_status == "passage_only" and (ev.text is not None or ev.spans):
            raise ValueError("Passage-only evidence cannot masquerade as a sentence")
    scopes = defaultdict(list)
    for event in events.values():
        if event.document_id not in docs or event.source_pmid != docs[event.document_id].source_pmid:
            raise ValueError("Event/document PMID mismatch")
        if event.source_ref.file_sha256 not in source_hashes or not event.evidence_ids:
            raise ValueError("Event provenance missing")
        for eid in event.evidence_ids:
            if eid not in evidence or evidence[eid].document_id != event.document_id:
                raise ValueError("Event references another document's evidence")
        if event.evidence_sentence is not None and event.evidence_sentence not in docs[event.document_id].text:
            raise ValueError("Event evidence_sentence is not verbatim source text")
        scopes[(event.source_profile, event.source_run_id, event.document_id)].append(event)
    # Diagnose nested references without merging/repairing events or losing invalid rows.
    for scoped in scopes.values():
        local = Counter(e.local_event_id for e in scoped if e.local_event_id)
        for event in scoped:
            for arg in event.arguments:
                if not isinstance(arg, dict):
                    raise ValueError("Event argument must be an object")
                if arg.get("Type") == "Event" and local[arg.get("Value")] != 1:
                    issue = {"kind": "unresolved_nested_event_reference", "value": arg.get("Value")}
                    if issue not in event.validation_issues:
                        event.validation_issues.append(issue)
    for candidate in bundle.candidates:
        if candidate.candidate_role not in {"event_type", "topic", "unknown"}:
            raise ValueError("Unknown concept candidate role")
        if candidate.source_ref.file_sha256 not in source_hashes or not candidate.supporting_event_ids:
            raise ValueError("Candidate provenance missing")
        supports = set()
        for event_id in candidate.supporting_event_ids:
            if event_id not in events:
                raise ValueError("Candidate references unknown event")
            supports.update(events[event_id].evidence_ids)
        if set(candidate.evidence_ids) != supports:
            raise ValueError("Candidate evidence is not its events' evidence")
    return {"documents": len(docs), "events": len(events), "evidence_records": len(evidence),
            "candidates": len(bundle.candidates), "foreign_keys_valid": True,
            "evidence_status": dict(Counter(e.alignment_status for e in bundle.evidence)),
            "candidate_roles": dict(Counter(c.candidate_role for c in bundle.candidates)),
            "event_issue_counts": dict(Counter(x["kind"] for e in bundle.events for x in e.validation_issues)),
            "synthetic_events": sum(e.data_origin == "synthetic" for e in bundle.events),
            "semantic_accuracy_evaluated": False}


def validate_preparation(bundle, concepts, mappings, taxonomy, frozen_snapshot):
    by_candidate = _unique(bundle.candidates, "candidate_id")
    by_concept = _unique(concepts, "concept_id")
    seen = set()
    for mapping in mappings:
        cid, uid = mapping["candidate_id"], mapping["concept_id"]
        if cid in seen or cid not in by_candidate:
            raise ValueError("Candidate mapping lost uniqueness")
        seen.add(cid)
        if uid is not None and (uid not in by_concept or cid not in by_concept[uid].candidate_ids):
            raise ValueError("Candidate registry mapping inconsistent")
    if seen != set(by_candidate):
        raise ValueError("Candidates disappeared during normalization")
    for concept in concepts:
        members = [by_candidate[cid] for cid in concept.candidate_ids]
        if set(concept.supporting_event_ids) != {e for c in members for e in c.supporting_event_ids}:
            raise ValueError("Concept lost event provenance")
        if set(concept.evidence) != {e for c in members for e in c.evidence_ids}:
            raise ValueError("Concept lost evidence provenance")
        if any(c.stale or c.candidate_role != concept.concept_role for c in members):
            raise ValueError("Stale or cross-role candidates in active registry")
    structural = validate_dag(taxonomy)
    input_ids = set(frozen_snapshot["concept_ids"])
    if input_ids != {c.concept_id for c in concepts if c.concept_role == "event_type"}:
        raise ValueError("Input type concepts deleted")
    actual = {c.concept_id: to_dict(c) for c in taxonomy.nodes if c.concept_id in input_ids}
    expected = {c["concept_id"]: c for c in frozen_snapshot["concepts"]}
    if actual != expected or taxonomy.concept_evidence_mapping != frozen_snapshot["concept_evidence_mapping"]:
        raise ValueError("Frozen input nodes or mappings changed")
    if taxonomy.instance_concept_links != frozen_snapshot["instance_concept_links"]:
        raise ValueError("Frozen event-concept assignments changed")
    by_event = {e.event_id: e for e in bundle.events}
    for link in taxonomy.instance_concept_links:
        if link["event_id"] not in by_event or link["concept_id"] not in input_ids:
            raise ValueError("Dangling instance-concept link")
        if not set(link["evidence_ids"]).issubset(by_event[link["event_id"]].evidence_ids):
            raise ValueError("Instance-concept evidence from a different event")
    structural.update({"unique_concepts": len(concepts),
        "unique_concept_roles": dict(Counter(c.concept_role for c in concepts)),
        "input_concepts": len(input_ids), "input_concept_retention": 1.0 if input_ids else None,
        "source_mapping_retained": True, "frozen_input_matches": True,
        "induction_ready": False, "fine_grained_inputs_confirmed": 0})
    return structural
