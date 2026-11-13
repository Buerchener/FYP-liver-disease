"""Adapters for frozen upstream and role/nested Stage 2 outputs."""
from collections import Counter
from .common import add_document, add_event, source_ref


def import_official_jsonl(source, bundle, namespace, origin, run_id):
    for line, row in source.jsonl():
        if not isinstance(row, dict) or not isinstance(row.get("event_entity_dict"), list):
            raise ValueError("Official JSONL requires event_entity_dict at line " + str(line))
        if row.get("data_origin") not in (None, origin):
            raise ValueError("Source data_origin conflicts with configured origin")
        metadata = row.get("metadata", {})
        ref = source_ref(source, "official_jsonl", run_id, "line:" + str(line), row.get("id"))
        doc = add_document(bundle, namespace, row.get("original_text"), metadata.get("pmid"),
                           metadata.get("title"), origin, ref)
        # Output row IDs need not be PMIDs, and chunk_id is not persisted upstream.
        for index, event in enumerate(row["event_entity_dict"]):
            er = source_ref(source, "official_jsonl", run_id,
                            ref.record_locator + "/event_entity_dict/" + str(index))
            add_event(bundle, doc, namespace, "official_jsonl", run_id, event, index, er)
        bundle.issues.append({"kind": "chunk_identity_not_persisted",
                              "document_id": doc.document_id, "source_ref": ref.__dict__})


def import_stage2_document(source, bundle, namespace, origin, run_id, profile):
    row = source.json()
    if not isinstance(row, dict) or not isinstance(row.get("events"), list):
        raise ValueError("Stage 2 document requires an events array")
    ref = source_ref(source, profile, run_id, "/", row.get("pmid"))
    doc = add_document(bundle, namespace, row.get("text"), row.get("pmid"),
                       row.get("title"), origin, ref)
    if row.get("error"):
        bundle.issues.append({"kind": "upstream_extraction_failure", "document_id": doc.document_id,
                              "message": str(row["error"]), "source_ref": ref.__dict__})
    structural = row.get("structural_errors", [])
    if not isinstance(structural, list):
        raise ValueError("structural_errors must be an array")
    counts = Counter(e.get("Event_ID") for e in row["events"] if isinstance(e, dict))
    for index, event in enumerate(row["events"]):
        if profile == "role_nested" and not event.get("Event_ID"):
            raise ValueError("role_nested event missing Event_ID")
        extra = [x for x in structural if not x.get("event") or x["event"] == event.get("Event_ID")]
        if event.get("Event_ID") and counts[event["Event_ID"]] > 1:
            extra = extra + [{"kind": "duplicate_local_event_id"}]
        er = source_ref(source, profile, run_id, "/events/" + str(index), event.get("Event_ID"))
        add_event(bundle, doc, namespace, profile, run_id, event, index, er, extra_issues=extra)
