import csv
import json

FIELDS = ["record_type", "record_id", "source_pmid", "statement", "head_event", "tail_event",
          "source_type", "source_id", "context", "evidence", "char_start", "char_end",
          "evidence_support", "participant_correct", "direction_correct", "abstraction_quality",
          "review_status", "reviewer", "notes"]


def export_review(path, events, relations, concepts):
    labels = {e["event_id"]: e["sentence"] for e in events}
    rows = []
    for kind, records, idkey, statement in (("event", events, "event_id", "sentence"),
            ("event_relation", relations, "relation_id", "relation"),
            ("concept", concepts, "concept_id", "concept")):
        for r in records:
            evidence = r.get("evidence", {})
            rows.append({"record_type": kind, "record_id": r[idkey], "source_pmid": r["source_pmid"],
                "statement": r[statement], "head_event": labels.get(r.get("head_event_id"), ""),
                "tail_event": labels.get(r.get("tail_event_id"), ""), "source_type": r.get("source_type", ""),
                "source_id": r.get("source_id", ""), "context": r.get("context", ""),
                "evidence": evidence.get("text", ""), "char_start": evidence.get("char_start", ""),
                "char_end": evidence.get("char_end", ""), "review_status": "pending",
                "notes": json.dumps(r.get("participants", []), ensure_ascii=False) if kind == "event" else ""})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
