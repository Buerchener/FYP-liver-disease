"""Attach verified LiverKG relations to ontology concepts and merge them across articles.

Each relation verified by the extractor (``phases.verification.relations``)
keeps its own evidence and write decision.  Endpoints are resolved through the
same chain the ontology used for entities:

    relation endpoint -> verification entity (canonical_key) -> mention -> concept

Relations sharing ``(subject concept, predicate, object concept)`` become one
concept-level relation.  The concept-level gate can only keep or lower the
extractor's decision: AUTO needs at least one IMPORT_READY evidence item.
"""

from collections import Counter, defaultdict
import hashlib

from cognitive_agent.extraction_quality import normalize_surface
from cognitive_agent.schema.relation_signatures import LITERATURE_CANDIDATE_SIGNATURES

from ..storage import atomic_json, read_json

VERSION = "relation-projection-v1"
PLACED_ACTIONS = {"as_parent", "insert_between", "new_under_root", "as_child"}
USABLE_WRITE_STATUS = {"IMPORT_READY", "SEMANTIC_ONLY", "HUMAN_REVIEW"}
COUNTER_EVIDENCE = {"CONTESTED", "INCONCLUSIVE"}
EVIDENCE_FIELDS = (
    "subject", "subject_type", "predicate", "object", "object_type", "direction", "evidence",
    "evidence_char_start", "evidence_char_end", "candidate_id", "candidate_version",
    "write_status", "semantic_status", "factual_status", "import_ready", "negated", "uncertain",
    "neo4j_status", "claim_role", "scope_status",
)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EndpointResolver:
    """Map an article-level endpoint surface to an ontology concept."""

    def __init__(self, inventory, concepts):
        self.concept_by_mention = {m: c["id"] for c in concepts for m in c["source_mentions"]}
        self.candidates = defaultdict(list)
        for row in inventory["verification_candidates"]:
            self.candidates[row["pmid"], row["canonical_key"]].append(row["id"])
        self.by_candidate = defaultdict(list)
        self.mentions = defaultdict(list)
        for mention in inventory["mentions"]:
            self.mentions[mention["pmid"]].append(mention)
            for candidate in mention["verification_ids"]:
                self.by_candidate[candidate].append(mention["id"])

    def resolve(self, pmid, surface, kind):
        surface_key = normalize_surface(surface)
        candidates = self.candidates.get((pmid, f"{kind}:{surface_key}"), [])
        mention_ids = [m for candidate in candidates for m in self.by_candidate[candidate]]
        method = "verification_link"
        if not mention_ids:
            method = "surface_match"
            mention_ids = [m["id"] for m in self.mentions[pmid]
                           if m["raw_type"] == kind and normalize_surface(m["mention"]) == surface_key]
        concepts = sorted({self.concept_by_mention[m] for m in mention_ids if m in self.concept_by_mention})
        base = {"method": method, "mention_ids": mention_ids}
        if len(concepts) == 1:
            return {**base, "status": "resolved", "concept_id": concepts[0]}
        if concepts:
            return {**base, "status": "ambiguous", "concept_ids": concepts}
        reason = "no_mention" if not mention_ids else "mention_not_conceptualized"
        return {**base, "status": "unresolved", "reason": reason}


def evidence_row(pmid, index, relation):
    row = {field: relation.get(field) for field in EVIDENCE_FIELDS}
    contrast = relation.get("contrastive_evidence") or {}
    return {"evidence_id": f"{pmid}:R{index:03d}", "pmid": pmid, **row,
            "contrast_status": contrast.get("status")}


def is_counter_evidence(row):
    return bool(row["negated"]) or row["contrast_status"] in COUNTER_EVIDENCE \
        or row["neo4j_status"] == "CONTRADICTING"


def concept_gate(relation, concepts, placed, decisions):
    """Return (gate, reasons). The gate never raises an extractor decision."""
    reasons = []
    evidence = relation["evidence"]
    statuses = [row["write_status"] for row in evidence]
    predicate = relation["predicate"]
    if predicate not in LITERATURE_CANDIDATE_SIGNATURES:
        reasons.append("unknown_predicate")
    sides = (("subject", relation["subject_concept_id"]), ("object", relation["object_concept_id"]))
    kinds = []
    for side, concept_id in sides:
        concept = concepts[concept_id]
        kinds.append(concept["kind"])
        if concept_id not in placed:
            action = (decisions.get(concept_id) or {}).get("action", "not_placed")
            reasons.append(f"{side}_concept_{action}")
        endpoint_types = {row[f"{side}_type"] for row in evidence}
        if endpoint_types != {concept["kind"]}:
            reasons.append(f"{side}_type_mismatch")
    signatures = LITERATURE_CANDIDATE_SIGNATURES.get(predicate, set())
    if tuple(kinds) not in signatures:
        reasons.append("signature_mismatch")
    if any(is_counter_evidence(row) for row in evidence):
        reasons.append("counter_evidence")
    if "IMPORT_READY" not in statuses:
        reasons.append("no_import_ready_evidence")
    if "unknown_predicate" in reasons or not USABLE_WRITE_STATUS.intersection(statuses):
        return "BLOCKED", reasons
    return ("REVIEW", reasons) if reasons else ("AUTO", reasons)


def novelty(relation, links, index):
    left = links.get(relation["subject_concept_id"]) or {}
    right = links.get(relation["object_concept_id"]) or {}
    if left.get("status") != "LINKED" or right.get("status") != "LINKED" or index is None:
        return "UNLINKED", []
    found = index.relations_between(
        f"{left['target']['label']}:{left['target']['id']}",
        f"{right['target']['label']}:{right['target']['id']}",
    )
    v2 = [{"type": kind, "source": source} for kind, source in found]
    same = any(kind == relation["predicate"] for kind, _ in found)
    return ("KNOWN_IN_V2" if same else "NEW"), v2


def project_relations(output, extraction_path, links, index=None):
    """Write concept_relations.json from agent_results.json and the ontology outputs."""
    records = read_json(extraction_path)["records"]
    inventory = read_json(output / "input_mentions.json")
    concepts = {c["id"]: c for c in read_json(output / "equivalence.json")["concepts"]}
    placement_path = output / "incremental-placement/placement.json"
    placement = read_json(placement_path)
    placed = {node["id"] for node in placement["nodes"]}
    decisions = {row["concept_id"]: row for row in placement["decisions"]}
    link_by_concept = {row["concept_id"]: row for row in links["links"]}
    resolver = EndpointResolver(inventory, concepts.values())

    # The built-in LiverKG predicates are all directed, so the extractor's
    # subject/object order is kept as is.
    grouped, unresolved, total = {}, [], 0
    for record in records:
        pmid = str(record["pmid"])
        verification = record.get("phases", {}).get("verification") or {}
        for index_in_article, relation in enumerate(verification.get("relations", []), 1):
            total += 1
            row = evidence_row(pmid, index_in_article, relation)
            subject = resolver.resolve(pmid, row["subject"], row["subject_type"])
            obj = resolver.resolve(pmid, row["object"], row["object_type"])
            row["subject_resolution"], row["object_resolution"] = subject, obj
            if subject["status"] != "resolved" or obj["status"] != "resolved":
                reasons = [f"{side}_{res['status']}" for side, res in (("subject", subject), ("object", obj))
                           if res["status"] != "resolved"]
                unresolved.append({**row, "reasons": reasons})
                continue
            left, right = subject["concept_id"], obj["concept_id"]
            key = f"{left}|{row['predicate']}|{right}"
            group = grouped.setdefault(key, {
                "relation_key": key, "subject_concept_id": left, "predicate": row["predicate"],
                "object_concept_id": right, "evidence": [],
            })
            group["evidence"].append(row)

    relations = []
    for key in sorted(grouped):
        relation = grouped[key]
        gate, reasons = concept_gate(relation, concepts, placed, decisions)
        state, v2 = novelty(relation, link_by_concept, index)
        subject, obj = concepts[relation["subject_concept_id"]], concepts[relation["object_concept_id"]]
        relations.append({
            **relation,
            "subject_name": subject["name"], "subject_kind": subject["kind"],
            "object_name": obj["name"], "object_kind": obj["kind"],
            "gate": gate, "gate_reasons": reasons,
            "pmids": sorted({row["pmid"] for row in relation["evidence"]}),
            "evidence_count": len(relation["evidence"]),
            "write_status_counts": dict(Counter(row["write_status"] for row in relation["evidence"])),
            "subject_link": (link_by_concept.get(relation["subject_concept_id"]) or {}).get("status"),
            "object_link": (link_by_concept.get(relation["object_concept_id"]) or {}).get("status"),
            "novelty": state, "v2_relations": v2,
        })

    projected = sum(row["evidence_count"] for row in relations)
    counts = {
        "article_relations": total, "projected_evidence": projected,
        "concept_relations": len(relations), "merged_evidence": projected - len(relations),
        "unresolved_evidence": len(unresolved),
        "gate": {gate: sum(row["gate"] == gate for row in relations) for gate in ("AUTO", "REVIEW", "BLOCKED")},
        "novelty": dict(Counter(row["novelty"] for row in relations)),
    }
    result = {
        "version": VERSION,
        "coordinate_convention": "evidence_char_start/end index TITLE: {title}\\nABSTRACT: {abstract}",
        "input_sha256": {
            "agent_results.json": file_sha256(extraction_path),
            "equivalence.json": file_sha256(output / "equivalence.json"),
            "placement.json": file_sha256(placement_path),
            "concept_links.json": file_sha256(output / "concept_links.json"),
        },
        "relations": relations, "unresolved_evidence": unresolved, "counts": counts,
    }
    atomic_json(output / "concept_relations.json", result)
    return result


def verify_relations(result, concept_ids):
    """Every projected relation must point at a known concept and keep its evidence."""
    seen = set()
    for relation in result["relations"]:
        if relation["relation_key"] in seen:
            raise ValueError("Duplicate concept relation.")
        seen.add(relation["relation_key"])
        if relation["subject_concept_id"] not in concept_ids or relation["object_concept_id"] not in concept_ids:
            raise ValueError("A concept relation refers to an unknown concept.")
        if not relation["evidence"] or relation["evidence_count"] != len(relation["evidence"]):
            raise ValueError("A concept relation lost its evidence.")
        if relation["gate"] == "AUTO" and not any(
                row["write_status"] == "IMPORT_READY" for row in relation["evidence"]):
            raise ValueError("AUTO requires IMPORT_READY evidence.")
