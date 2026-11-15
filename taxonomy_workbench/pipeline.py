"""Read saved taxonomy results and their upstream records for the workbench."""
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


def load_upward_result(path):
    result = read_json(path)
    if result["configuration"]["schema"] != "upward-taxonomy/1" or result["status"] != "finished":
        raise ValueError("Expected a finished bottom-up taxonomy run.")
    load_taxonomy(result["taxonomy"])
    return result


def upward_view(result, run_id):
    graph = result["taxonomy"]
    admission = result.get("admission", {})
    held_ids = set(admission.get("held_leaf_ids", []))
    admission_reviews = {row["leaf_id"]: row for row in admission.get("reviews", [])}
    inactive_statuses = {"maintenance_held", "admission_held", "merged", "merged_duplicate"}
    events = [{
        "id": event["source_id"], "source_id": event["source_id"], "pmid": event["source_pmid"],
        "text": event["Event"], "trigger": event.get("Trigger", ""), "evidence": event.get("Evidence", ""),
    } for event in result["source_events"]]
    event_by_id = {event["id"]: event for event in events}
    concepts = [{
        "id": row["source_concept_id"], "source_id": row["source_event"],
        "concept_name": row["concept_name"], "description": row["description"],
    } for row in result["source_concepts"]]

    review_by_edge = {}
    creation_round = {}
    rounds = []
    for row in result["rounds"]:
        new_count = row.get("new_active_parent_count", sum(accepted["new"] for accepted in row["accepted"]))
        rounds.append({"round": row["round"], "cutoff": row["cutoff"], "committed_new": new_count})
        for accepted in row["accepted"]:
            if accepted["new"]:
                creation_round[accepted["parent_id"]] = row["round"]
            review = row["reviews"][accepted["proposal_id"]][0]
            checks = {check["child_id"]: check for check in review["edge_reviews"]}
            for edge_id in accepted["edge_ids"]:
                review_by_edge[edge_id] = (accepted["parent_id"], checks, review.get("review_ref"))

    nodes = [{
        "id": node["concept_id"], "name": node["concept_label"], "definition": node["concept_description"],
        "kind": "leaf" if node["concept_id"] in graph["frozen_input_concept_ids"] else "intermediate",
        "created_round": creation_round.get(node["concept_id"]),
        "event_ids": node["supporting_event_ids"],
        "pmids": sorted({event_by_id[event]["pmid"] for event in node["supporting_event_ids"]
                         if event_by_id[event]["pmid"]}),
    } for node in graph["nodes"]]
    excluded = []
    raw_nodes = {node["concept_id"]: node for node in graph["nodes"]}
    for node in nodes:
        raw = raw_nodes[node["id"]]
        if node["id"] in held_ids or raw.get("validation_status") in inactive_statuses:
            excluded.append({**node,
                "category": "admission" if node["id"] in held_ids else "inactive",
                "reason": admission_reviews.get(node["id"], {}).get("reason", "旧结构回查后已撤下，或已合并；不计入当前分类图。"),
            })
    excluded_ids = {node["id"] for node in excluded}
    nodes = [node for node in nodes if node["id"] not in excluded_ids]
    active_ids = {node["id"] for node in nodes}
    edges = []
    for edge in graph["parent_child_edges"]:
        if edge["parent_id"] not in active_ids or edge["child_id"] not in active_ids:
            continue
        parent, checks, review_ref = review_by_edge.get(edge["edge_id"], (None, {}, None))
        check = checks.get(edge["child_id"], {}) if (parent == edge["parent_id"] and review_ref == edge.get("decision_ref")) else {}
        passed = check.get("verdict") == "pass"
        edges.append({
            "id": edge["edge_id"], "parent": edge["parent_id"], "child": edge["child_id"],
            "status": edge["status"], "review": "supported" if passed else "unreviewed",
            "review_label": "模型审核通过 · 待人工复核" if passed else "已保留 · 待人工复核",
            "review_reason": edge.get("reason") or check.get("reason", ""),
            "review_ref": edge.get("decision_ref"),
            "parent_only_example": check.get("parent_only_example"),
        })
    parent_ids = {edge["parent"] for edge in edges}
    child_ids = {edge["child"] for edge in edges}
    isolated = [node["id"] for node in nodes if node["id"] not in parent_ids | child_ids]
    documents = [{"pmid": pmid, "title": ""} for pmid in sorted({event["pmid"] for event in events if event["pmid"]})]
    return {
        "available": True, "mode": "bottom_up", "run_id": run_id, "excluded_nodes": excluded,
        "roots": [id for id in graph["predefined_roots"] if id in active_ids], "nodes": nodes, "edges": edges, "unplaced_ids": isolated,
        "events": events, "concepts": concepts, "documents": documents,
        "leaves": result["leaf_concepts"],
        "stats": {
            "articles": len(documents), "events": len(events), "concepts": len(concepts),
            "abstained": len(result["unassigned_events"]), "leaves": len(result["leaf_concepts"]),
            "intermediates": sum(node["kind"] == "intermediate" for node in nodes), "edges": len(edges), "branches": len(parent_ids - child_ids),
            "isolated": len(isolated), "held": len(held_ids),
            "inactive": sum(node["category"] == "inactive" for node in excluded),
        },
        "run": {"status": result["status"], "stop_reason": result["stop_reason"], "rounds": rounds},
    }


def pipeline_view(runs_root, hierarchy_path=None):
    if hierarchy_path is not None:
        path = Path(hierarchy_path)
        return upward_view(load_upward_result(path), path.parent.name)
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
