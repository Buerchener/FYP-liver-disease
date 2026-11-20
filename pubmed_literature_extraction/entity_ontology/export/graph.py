"""Write the run's concepts, hierarchy, relations and provenance as graph JSONL files.

Files live in ``<run>/graph/`` and are self-contained: every relation and
evidence row refers to concept and article rows in the same export.  BLOCKED
relations stay in concept_relations.json and are not exported.  Nothing here
touches a database; ``neo4j_load`` reads these files.
"""

import hashlib

from ..extraction.mentions import read_articles
from ..storage import atomic_json, atomic_jsonl, read_json

VERSION = "graph-export-v1"
GRAPH_FILES = ("articles", "concepts", "is_a", "mentions", "relations", "evidence", "same_as")
EXPORTED_GATES = ("AUTO", "REVIEW")


def build_graph(output, run_id):
    placement = read_json(output / "incremental-placement/placement.json")
    concepts = read_json(output / "equivalence.json")["concepts"]
    links = {row["concept_id"]: row for row in read_json(output / "concept_links.json")["links"]}
    projection = read_json(output / "concept_relations.json")
    inventory = read_json(output / "input_mentions.json")
    articles = read_articles((output / "articles.jsonl").read_bytes())

    def uid(ident):
        return f"{run_id}/{ident}"

    decisions = {row["concept_id"]: row for row in placement["decisions"]}
    placed = {node["id"]: node for node in placement["nodes"]}
    concept_rows, same_as = [], []
    for node in [*placement["nodes"], *(c for c in concepts if c["id"] not in placed)]:
        link = links.get(node["id"]) or {}
        target = link.get("target") or {}
        placement_state = "placed" if node["id"] in placed else (
            decisions.get(node["id"]) or {}).get("action", "not_placed")
        concept_rows.append({
            "uid": uid(node["id"]), "concept_id": node["id"], "run_id": run_id,
            "kind": node["kind"], "name": node["name"], "definition": node.get("definition", ""),
            "aliases": list(node.get("aliases", [])), "origin": node.get("origin", "literature_mention"),
            "placement": placement_state, "link_status": link.get("status"),
            "kg_label": target.get("label"), "kg_key": target.get("key"),
            "kg_node_id": target.get("id"), "kg_name": target.get("name"),
            "go_id": (link.get("external_ids") or {}).get("GO") or node.get("go_id"),
        })
        if link.get("status") == "LINKED":
            same_as.append({"concept_uid": uid(node["id"]), "label": target["label"],
                            "key": target["key"], "node_id": target["id"], "name": target["name"]})

    is_a = [{"child_uid": uid(edge["child_id"]), "parent_uid": uid(edge["parent_id"]),
             "support": edge["support"], "run_id": run_id} for edge in placement["edges"]]

    concept_of = {m: c["id"] for c in concepts for m in c["source_mentions"]}
    mentions = [{
        "uid": uid(m["id"]), "mention_id": m["id"], "pmid": m["pmid"], "text": m["mention"],
        "char_start": m["char_start"], "char_end": m["char_end"],
        "concept_uid": uid(concept_of[m["id"]]), "run_id": run_id,
    } for m in inventory["mentions"] if m["id"] in concept_of]

    relations, evidence = [], []
    for relation in projection["relations"]:
        if relation["gate"] not in EXPORTED_GATES:
            continue
        relation_uid = uid(relation["relation_key"])
        subject_uid, object_uid = uid(relation["subject_concept_id"]), uid(relation["object_concept_id"])
        relations.append({
            "uid": relation_uid, "relation_key": relation["relation_key"], "run_id": run_id,
            "subject_uid": subject_uid, "predicate": relation["predicate"], "object_uid": object_uid,
            "gate": relation["gate"], "gate_reasons": relation["gate_reasons"],
            "novelty": relation["novelty"],
            "v2_relations": [f"{row['type']}/{row['source']}" for row in relation["v2_relations"]],
            "pmids": relation["pmids"], "evidence_count": relation["evidence_count"],
            "evidence_uids": [uid(row["evidence_id"]) for row in relation["evidence"]],
            "source": "literature",
        })
        for row in relation["evidence"]:
            evidence.append({
                "uid": uid(row["evidence_id"]), "evidence_id": row["evidence_id"], "pmid": row["pmid"],
                "text": row["evidence"], "char_start": row["evidence_char_start"],
                "char_end": row["evidence_char_end"], "write_status": row["write_status"],
                "semantic_status": row["semantic_status"], "candidate_id": row["candidate_id"],
                "candidate_version": row["candidate_version"], "relation_uid": relation_uid,
                "subject_uid": subject_uid, "object_uid": object_uid, "run_id": run_id,
            })

    graph = {
        "articles": [{"pmid": str(a["pmid"]), "title": a["title"]} for a in articles],
        "concepts": concept_rows, "is_a": is_a, "mentions": mentions,
        "relations": relations, "evidence": evidence, "same_as": same_as,
    }
    verify_graph(graph)
    return graph


def verify_graph(graph):
    concepts = {row["uid"] for row in graph["concepts"]}
    pmids = {row["pmid"] for row in graph["articles"]}
    relations = {row["uid"] for row in graph["relations"]}
    if len(concepts) != len(graph["concepts"]) or len(relations) != len(graph["relations"]):
        raise ValueError("Graph export has duplicate concept or relation IDs.")
    endpoints = [(row["child_uid"], row["parent_uid"]) for row in graph["is_a"]]
    endpoints += [(row["subject_uid"], row["object_uid"]) for row in graph["relations"]]
    endpoints += [(row["subject_uid"], row["object_uid"]) for row in graph["evidence"]]
    if any(left not in concepts or right not in concepts for left, right in endpoints):
        raise ValueError("Graph export has an edge to an unknown concept.")
    if any(row["concept_uid"] not in concepts for row in [*graph["mentions"], *graph["same_as"]]):
        raise ValueError("Graph export has a mention or link to an unknown concept.")
    if any(row["pmid"] not in pmids for row in [*graph["mentions"], *graph["evidence"]]):
        raise ValueError("Graph export has provenance outside the batch articles.")
    if any(row["relation_uid"] not in relations for row in graph["evidence"]):
        raise ValueError("Graph export has evidence without its relation.")


def export_graph(output, run_id):
    graph = build_graph(output, run_id)
    folder = output / "graph"
    files = {}
    for name in GRAPH_FILES:
        path = folder / f"{name}.jsonl"
        atomic_jsonl(path, graph[name])
        files[name] = {"rows": len(graph[name]), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    gates = {gate: sum(row["gate"] == gate for row in graph["relations"]) for gate in EXPORTED_GATES}
    manifest = {"version": VERSION, "run_id": run_id, "files": files, "relation_gates": gates}
    atomic_json(folder / "manifest.json", manifest)
    return manifest
