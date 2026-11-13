"""Adapt normalized leaves to the induction input contract without creating edges."""

from event_taxonomy.adapters.common import add_document, add_event, source_ref
from event_taxonomy.models import ConceptCandidate, ImportBundle, UniqueConcept, to_dict
from event_taxonomy.taxonomy.initialization import build_initial_taxonomy
from event_taxonomy.validation import validate_bundle

from .data import InputData


def import_events(abstracts_source, event_source, events):
    bundle = ImportBundle()
    origin = "real_literature"
    wanted_pmids = {row["source_pmid"] for row in events}
    documents = {}
    for line, row in abstracts_source.jsonl():
        pmid = row["pmid"]
        if pmid not in wanted_pmids:
            continue
        if pmid in documents:
            raise ValueError(f"Duplicate abstract for PMID {pmid}.")
        ref = source_ref(abstracts_source, "pubmed_abstract", None, f"line:{line}", pmid)
        documents[pmid] = add_document(bundle, "pubmed", row["abstract"], pmid, row.get("title"), origin, ref)
    if documents.keys() != wanted_pmids:
        raise ValueError("Missing abstracts for PMIDs: " + ", ".join(sorted(wanted_pmids - documents.keys())))

    source_events = {}
    profile = "xml_event_extraction"
    run_id = event_source.sha256
    for index, row in enumerate(events):
        source_id = row["source_id"]
        ref = source_ref(event_source, profile, run_id, f"/{index}", source_id)
        event = add_event(bundle, documents[row["source_pmid"]], "pubmed", profile, run_id,
                          {**row, "Event_ID": source_id}, index, ref, upstream_id=source_id)
        source_events[source_id] = event
    return bundle, source_events


def induction_input(leaves, source_concepts, concept_source, event_source, abstracts_source, imported, roots,
                    *, deduplicated_ids=()):
    bundle, source_events = imported
    origin = "real_literature"
    candidates = {}
    for row in source_concepts:
        source_id = row["source_concept_id"]
        event = source_events[row["source_event"]]
        ref = source_ref(concept_source, "event_conceptualization", concept_source.sha256,
                         row["record_locator"], row["source_event"])
        candidate = ConceptCandidate(
            source_id, row["concept_name"], row["description"], [event.event_id], list(event.evidence_ids),
            "event_type", ref, "pubmed", origin, description_status="generated")
        candidates[source_id] = candidate
        bundle.candidates.append(candidate)
    validate_bundle(bundle, [concept_source, event_source, abstracts_source])

    concepts = []
    for leaf in leaves:
        cid = leaf["concept_id"]
        source_ids = leaf["source_concepts"]
        members = [candidates[source_id] for source_id in source_ids]
        version = "leaf_deduplication/1" if cid in deduplicated_ids else "leaf_equivalence/1"
        concepts.append(UniqueConcept(
            cid, leaf["concept_name"], leaf["description"], source_ids,
            sorted({c.candidate_label for c in members}), cid, version,
            sorted({e for c in members for e in c.supporting_event_ids}),
            sorted({e for c in members for e in c.evidence_ids}), "event_type",
            description_provenance="canonicalization" if len(members) > 1 else "conceptualization"))
    taxonomy = build_initial_taxonomy(concepts, bundle.candidates, roots)
    dataset = {"taxonomy": to_dict(taxonomy), "data_origin": origin}
    for field in ("documents", "events", "evidence"):
        dataset[field] = [to_dict(item) for item in getattr(bundle, field)]
    InputData(dataset, None, {}, 1)
    return dataset
