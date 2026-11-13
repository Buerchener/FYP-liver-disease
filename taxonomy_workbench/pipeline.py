"""Read saved root-induction results and their upstream records for the workbench."""
from collections import Counter
from pathlib import Path

from event_taxonomy.induction.graph import load_taxonomy
from event_taxonomy.induction.storage import read_json
from event_taxonomy.taxonomy.dag import validate_dag


def safe_file(path, root):
    """Only read regular files inside this workbench's results directory."""
    path = Path(path)
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if ".." in relative.parts:
        return False
    if not path.is_file():
        return False
    return not any((root / Path(*relative.parts[:index])).is_symlink()
                   for index in range(1, len(relative.parts) + 1))


def optional_json(path, root, default=None):
    return read_json(path) if safe_file(path, root) else default


def current_run(runs_root):
    root = Path(runs_root).resolve()
    paths = [path.parent for path in root.glob("xml-event-only/*/final_taxonomy.json")
             if all(safe_file(path.parent / name, root)
                    for name in ("final_taxonomy.json", "input_dataset.json", "report.json", "manifest.json"))]
    return max(paths, key=lambda path: (path / "final_taxonomy.json").stat().st_mtime_ns) if paths else None


def pipeline_view(runs_root):
    root = Path(runs_root).resolve()
    run = current_run(root)
    if run is None:
        return {"available": False}

    graph = read_json(run / "final_taxonomy.json")
    structure = validate_dag(load_taxonomy(graph))
    source = read_json(run / "input_dataset.json")
    report = read_json(run / "report.json")
    manifest = read_json(run / "manifest.json")
    review = optional_json(run / "edge_definition_review.json", root, {"edges": []})
    review_by_id = {row["edge_id"]: row for row in review["edges"]}
    event_by_id = {event["event_id"]: event for event in source["events"]}

    nodes = [{
        "id": node["concept_id"], "name": node["concept_label"],
        "definition": node["concept_description"],
        "kind": {"predefined_root": "root", "input_concept": "leaf",
                 "induced_intermediate": "intermediate"}[node["origin_kind"]],
        "event_ids": node["supporting_event_ids"],
        "pmids": sorted({event_by_id[event]["source_pmid"] for event in node["supporting_event_ids"]
                         if event_by_id[event]["source_pmid"]}),
    } for node in graph["nodes"]]
    edges = []
    for edge in graph["parent_child_edges"]:
        checked = review_by_id.get(edge["edge_id"], {})
        # A review from another graph must never be attached by ID alone.
        if checked and (checked["parent_id"], checked["child_id"]) != (edge["parent_id"], edge["child_id"]):
            checked = {}
        edges.append({
            "id": edge["edge_id"], "parent": edge["parent_id"], "child": edge["child_id"],
            "status": edge["status"], "proposal_reason": edge["reason"],
            "review": checked.get("definition_review", "unreviewed"),
            "review_label": checked.get("label_zh", "未复查"),
            "review_reason": checked.get("reason_zh", ""),
        })
    parents = {edge["child"] for edge in edges}
    leaves = set(graph["frozen_input_concept_ids"])

    # Follow the recorded input lineage, without exposing model connection config.
    leaf_input = Path(manifest["config"]["dataset"])
    if not leaf_input.is_absolute():
        leaf_input = run / leaf_input
    leaf_manifest = optional_json(leaf_input.parent / "manifest.json", root, {})
    leaf_data = optional_json(leaf_input.parent / "leaf_concepts.json", root, {})
    extraction = {}
    reference = leaf_manifest.get("inputs", {}).get("events", {}).get("path")
    if reference:
        extraction = optional_json(Path(reference).parent / "report.json", root, {})
    concept_data = []
    reference = leaf_manifest.get("inputs", {}).get("concepts", {}).get("path")
    if reference:
        concept_data = optional_json(Path(reference), root, {}).get("concepts", [])

    return {
        "available": True, "run_id": run.name,
        "roots": graph["predefined_roots"], "nodes": nodes, "edges": edges,
        "unplaced_ids": graph["unplaced_concept_ids"],
        "events": [{
            "id": event["event_id"], "source_id": event["local_event_id"],
            "pmid": event["source_pmid"], "text": event["event_text"],
            "trigger": event.get("trigger", ""), "evidence": event.get("evidence_sentence", ""),
        } for event in source["events"]],
        "documents": [{"pmid": doc["source_pmid"], "title": doc["title"], "text": doc["text"]}
                      for doc in source["documents"]],
        "concepts": concept_data,
        "leaves": leaf_data.get("leaf_concepts", []),
        "stats": {
            "articles": len(source["documents"]), "events": len(source["events"]),
            "rejected_events": extraction.get("rejected_event_records"),
            "concepts": sum(bool(row.get("concept_name")) for row in concept_data),
            "abstained": sum(not row.get("concept_name") for row in concept_data),
            "leaves": len(leaves),
            "intermediates": sum(node["kind"] == "intermediate" for node in nodes),
            "edges": len(edges), "rooted": structure["root_reachable_inputs"],
            "unplaced": len(graph["unplaced_concept_ids"]), "parentless": len(leaves - parents),
            "reviews": dict(Counter(edge["review"] for edge in edges)),
        },
        "run": {
            "status": report["status"], "stop_reason": report["stop_reason"],
            "rounds": [{key: row.get(key) for key in (
                "round", "frontier_size", "groups", "committed_new", "committed_reuse")}
                for row in report["rounds"]],
            "llm_models": report.get("llm_models", []),
            "llm_calls": report["llm_calls"],
            "issues": len(report.get("issues", [])),
        },
    }
