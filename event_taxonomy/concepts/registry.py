from collections import defaultdict
import unicodedata
from ..models import UniqueConcept, canonical_json, stable_id


def conservative_key(label):
    # Retain case, punctuation and biomedical distinctions for new input labels.
    return " ".join(unicodedata.normalize("NFC", label).split())


def normalize_candidates(candidates):
    groups = defaultdict(list)
    mappings, issues = [], []
    for candidate in candidates:
        if candidate.stale or candidate.validation_status == "rejected":
            mappings.append({"candidate_id": candidate.candidate_id, "concept_id": None,
                             "status": "archived_not_active", "reason": "stale_or_rejected"})
            continue
        if candidate.upstream_concept_id:
            key = canonical_json([candidate.source_namespace, candidate.candidate_role,
                                  "upstream_id", candidate.upstream_concept_id,
                                  candidate.disambiguation_key])
            version = "preserve_upstream_identity/1"
        else:
            key = canonical_json([candidate.source_namespace, candidate.candidate_role,
                                  conservative_key(candidate.candidate_label),
                                  candidate.candidate_description, candidate.disambiguation_key])
            version = "NFC_whitespace_case_sensitive/1"
        groups[(version, key)].append(candidate)
    concepts = []
    for (version, key), members in sorted(groups.items()):
        descriptions = {c.candidate_description for c in members if c.candidate_description is not None}
        if len(descriptions) > 1:
            raise ValueError("Conflicting descriptions under one upstream concept identity")
        normalized_labels = {c.upstream_normalized_label for c in members if c.upstream_normalized_label is not None}
        if len(normalized_labels) > 1:
            raise ValueError("Conflicting normalized labels under one upstream concept ID")
        cid = stable_id("concept", version, key)
        aliases = sorted({c.candidate_label for c in members})
        desc = next(iter(descriptions), None)
        concept = UniqueConcept(
            cid, aliases[0], desc, sorted(c.candidate_id for c in members), aliases, key, version,
            sorted({e for c in members for e in c.supporting_event_ids}),
            sorted({e for c in members for e in c.evidence_ids}), members[0].candidate_role,
            sorted({c.upstream_concept_id for c in members if c.upstream_concept_id}),
            description_provenance="imported_candidate" if desc is not None else None)
        concepts.append(concept)
        if desc is None:
            issues.append({"kind": "concept_definition_missing", "concept_id": cid})
        if version == "preserve_upstream_identity/1" and len({conservative_key(x) for x in aliases}) > 1:
            issues.append({"kind": "upstream_normalization_aliases_need_review", "concept_id": cid,
                           "aliases": aliases})
        for c in members:
            mappings.append({"candidate_id": c.candidate_id, "concept_id": cid,
                             "status": "mapped", "normalization_version": version})
    return sorted(concepts, key=lambda c: c.concept_id), sorted(mappings, key=lambda m: m["candidate_id"]), issues
