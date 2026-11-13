from copy import deepcopy
import hashlib
from ..models import Document, EventInstance, SourceRef, stable_id
from ..preparation.evidence import locate_evidence, primary_sentence


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a nonempty string")
    return value


def source_ref(source, profile, run_id, locator, original_id=None):
    return SourceRef(profile, run_id, source.sha256, locator,
                     str(original_id) if original_id is not None else None)


def add_document(bundle, namespace, text, pmid, title, origin, ref):
    require_text(text, "Source text")
    if origin not in {"real_literature", "synthetic", "mixed"}:
        raise ValueError("Invalid data_origin")
    if origin == "synthetic" and pmid not in (None, ""):
        raise ValueError("Synthetic Data must not use a PMID")
    pmid = str(pmid) if pmid not in (None, "") else None
    if pmid is not None and not pmid.isdecimal():
        raise ValueError("PMID must be explicitly supplied as a numeric identifier")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    doc_id = stable_id("document", namespace, pmid, digest)
    existing = next((d for d in bundle.documents if d.document_id == doc_id), None)
    if existing:
        if existing.data_origin != origin:
            raise ValueError("Conflicting document origins")
        if ref not in existing.source_refs:
            existing.source_refs.append(ref)
        return existing
    doc = Document(doc_id, pmid, str(title or ""), text, digest, origin, [ref])
    bundle.documents.append(doc)
    return doc


def add_event(bundle, document, namespace, profile, run_id, row, index, ref,
              upstream_id=None, extra_issues=None):
    text = require_text(row.get("Event"), "Event")
    local_id = row.get("Event_ID")
    if local_id is not None:
        local_id = require_text(local_id, "Event_ID")
    event_id = stable_id("event", namespace, run_id, document.document_id,
                         upstream_id or ref.record_locator, local_id, index, text)
    if any(e.event_id == event_id for e in bundle.events):
        raise ValueError("Duplicate source event identity: " + event_id)
    raw_evidence = row.get("_evidence")
    if raw_evidence is None:
        raw_evidence = [{"text": row.get("Evidence")}]
    if not isinstance(raw_evidence, list):
        raise ValueError("Evidence must be an array")
    if not raw_evidence:
        raw_evidence = [{"text": None}]
    evidence = []
    issues = deepcopy(extra_issues or [])
    for i, item in enumerate(raw_evidence):
        if not isinstance(item, dict):
            raise ValueError("Evidence record must be an object")
        evidence_ref = SourceRef(profile, run_id, ref.file_sha256,
                                 ref.record_locator + "/evidence/" + str(i), local_id)
        ev = locate_evidence(document, item.get("text"), evidence_ref, item.get("status"))
        if ev.alignment_status not in {"verified_exact", "verified_whitespace"}:
            issues.append({"kind": "evidence_" + ev.alignment_status, "evidence_id": ev.evidence_id})
        if item.get("status") == "verified_exact" and item.get("char_start") is not None:
            claimed = {"start": item["char_start"], "end": item.get("char_end")}
            if item.get("offset_unit") != "unicode_code_point" or claimed not in ev.spans:
                issues.append({"kind": "upstream_offset_mismatch", "evidence_id": ev.evidence_id})
        bundle.evidence.append(ev)
        evidence.append((ev, document))
    arguments = row.get("Arguments", [])
    if not isinstance(arguments, list):
        raise ValueError("Arguments must be an array")
    if "Entity" in row:
        if not isinstance(row["Entity"], list):
            raise ValueError("Entity must be an array")
        # Preserve untyped participation without inventing a Cause or Theme.
        arguments = [{"Role": "Participant", "Type": "Entity", "Value": x,
                      "role_provenance": "untyped_original_participation"} for x in row["Entity"]]
    assertion = row.get("assertion") or "unknown"
    if assertion == "unspecified":
        assertion = "unknown"
    event = EventInstance(event_id, text, document.source_pmid, document.document_id,
                          primary_sentence(evidence), [e.evidence_id for e, _ in evidence],
                          local_id, run_id, profile, ref, index, row.get("Trigger"),
                          deepcopy(arguments), assertion, row.get("context"),
                          row.get("review_status", "pending"), issues, document.data_origin,
                          upstream_id)
    bundle.events.append(event)
    return event
