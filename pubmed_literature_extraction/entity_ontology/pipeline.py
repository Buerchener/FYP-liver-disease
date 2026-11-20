"""Run a LiverKG batch: PubMed JSONL -> extraction -> ontology -> concept relations -> graph files."""

from datetime import datetime, timezone
from graphlib import TopologicalSorter
import hashlib
from pathlib import Path
import re

from . import DEFAULT_RUNS_DIR
from .model import ModelSession, endpoint, read_settings
from .storage import atomic_json, digest, read_json, run_lock
from .concepts.deduplicate import deduplicate
from .concepts.normalization import resolve
from .concepts.retrieve_pairs import DEFAULT_KNN_K
from .extraction.coverage import VERSION as COVERAGE_VERSION, recover as recover_coverage
from .extraction.mentions import build, conceptualization_input, read_articles
from .extraction.native import extract_articles, extractor_identity
from .export.graph import VERSION as EXPORT_VERSION, export_graph
from .hierarchy.order_entities import order_for_hierarchy
from .hierarchy.placement import run as place_incrementally
from .linking.kg_links import (
    VERSION as LINK_VERSION, V2Index, link_concepts, resolve_v2_data_dir, v2_data_digest,
)
from .relations.project import VERSION as PROJECTION_VERSION, project_relations, verify_relations


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_placement(placement, ranked):
    if placement["status"] != "complete" or placement["ranked_ids"] != [r["id"] for r in ranked]:
        raise ValueError("Incremental placement did not consume the ranked concepts.")
    decisions = placement["decisions"]
    if [row["concept_id"] for row in decisions] != placement["ranked_ids"]:
        raise ValueError("Placement decisions do not match the ranked input.")
    graph = {row["id"]: set() for row in placement["nodes"]}
    if len(graph) != len(placement["nodes"]):
        raise ValueError("The graph has duplicate concept IDs.")
    for edge in placement["edges"]:
        child, parent = edge["child_id"], edge["parent_id"]
        if parent not in graph or child not in graph or parent in graph[child]:
            raise ValueError("Invalid graph edge.")
        graph[child].add(parent)
    tuple(TopologicalSorter(graph).static_order())


def run_batch(articles_path, run_id, *, config, runs_dir=DEFAULT_RUNS_DIR, title=None,
              knn_k=DEFAULT_KNN_K, go_reference=True, entity_coverage=True, v2_data=None):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", run_id):
        raise ValueError("--run-id accepts 1–80 lowercase letters, digits, and hyphens.")
    if knn_k < 1:
        raise ValueError("--knn-k must be at least one.")

    content = articles_path.read_bytes()
    pmids = [str(row["pmid"]) for row in read_articles(content)]
    source_hash = hashlib.sha256(content).hexdigest()
    output = Path(runs_dir).resolve() / run_id
    config = Path(config).resolve()
    v2_dir = resolve_v2_data_dir(v2_data)
    extractor = extractor_identity()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    articles = output / "articles.jsonl"
    placement_path = output / "incremental-placement/placement.json"
    relations_path = output / "concept_relations.json"
    identity = {"articles_sha256": source_hash, "knn_k": knn_k,
                "extractor_fingerprint": extractor["fingerprint"], "go_reference_enabled": go_reference,
                "model_config_sha256": file_hash(config),
                "normalization_version": "v2",
                "entity_coverage": COVERAGE_VERSION if entity_coverage else None,
                "v2_data_sha256": v2_data_digest(v2_dir),
                "integration_version": f"{LINK_VERSION}/{PROJECTION_VERSION}/{EXPORT_VERSION}"}
    provider, _ = read_settings(config, "audit_edges")
    embedding = read_json(config)["embedding"]
    environment_endpoints = [endpoint(connection) for connection in (provider, embedding)
                             if not connection.get("base_url")]
    if environment_endpoints:
        identity["environment_endpoints_sha256"] = digest(environment_endpoints)
    manifest = {
        **identity, "run_id": run_id, "title": title or run_id, "pmids": pmids,
        "extractor_git": extractor["git"],
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "running", "stages": {},
    }

    with run_lock(output):
        previous = read_json(manifest_path) if manifest_path.exists() else None
        if previous is not None:
            if any(previous.get(key) != value for key, value in identity.items()):
                raise ValueError("This run ID already refers to different inputs or settings.")
            if previous["status"] == "complete":
                if (file_hash(articles) != source_hash
                        or file_hash(placement_path) != previous["placement_sha256"]
                        or file_hash(relations_path) != previous["relations_sha256"]):
                    raise ValueError("The completed batch files do not match the published manifest.")
                return previous
            manifest["created_at"] = previous["created_at"]
        if articles.exists():
            if file_hash(articles) != source_hash:
                raise ValueError("Stored articles do not match this run.")
        else:
            temp_articles = articles.with_suffix(".tmp")
            temp_articles.write_bytes(content)
            temp_articles.replace(articles)

        # Keep all model journals and intermediate results tied to this batch.
        session = ModelSession(config, output)
        stage = "extraction"

        def start(name):
            nonlocal stage
            stage = name
            manifest["phase"] = name
            manifest["stages"][name] = "running"
            atomic_json(manifest_path, manifest)
            print(f"[{run_id}] {name}", flush=True)

        def complete():
            manifest["stages"][stage] = "complete"
            atomic_json(manifest_path, manifest)

        try:
            start("extraction")
            extraction_path = extract_articles(articles, output, config=config)
            extraction = read_json(extraction_path)
            report = extraction.get("report", {})
            if report and (report["dry_run"] is not True or report["neo4j_connected"] is not False
                           or report["total_articles"] != len(pmids)):
                raise ValueError("Unexpected extraction mode or article count.")
            if any(row["phases"]["extraction"].get("error") for row in extraction["records"]):
                raise ValueError("Some abstracts reported an extraction error.")
            manifest["extraction_model"] = report.get("model") or session.request_settings()[1]["model"]
            coverage_path = None
            if entity_coverage:
                complete()
                start("entity_coverage")
                coverage_path = recover_coverage(articles, extraction_path, output, request=session.req)
            inventory = build(output, extraction_path, coverage_path=coverage_path)
            atomic_json(output / "input_mentions.json", inventory)
            compact = conceptualization_input(inventory)
            atomic_json(output / "conceptualization_input.json", compact)
            complete()

            start("conceptualization")
            normalized = resolve(compact, session=session)
            complete()

            start("embedding_and_deduplication")
            concepts = deduplicate(compact, normalized, session=session, knn_k=knn_k)
            retrieval = read_json(output / "retrieval/retrieved_pairs.json")
            complete()

            start("ordering")
            ranked = order_for_hierarchy(concepts, session=session)
            complete()

            start("incremental_placement")
            placement = place_incrementally(output, placement_path.parent, config=config, go_reference=go_reference)
            verify_placement(placement, ranked)
            complete()

            start("concept_linking")
            v2_index = V2Index(v2_dir)
            links = link_concepts(output, v2_dir, index=v2_index)
            complete()

            start("relation_projection")
            projection = project_relations(output, extraction_path, links, v2_index)
            verify_relations(projection, {concept["id"] for concept in concepts})
            complete()

            start("graph_export")
            graph = export_graph(output, run_id)
            complete()

            counts = {
                "articles": len(pmids), "mentions": len(inventory["mentions"]),
                "coverage_mentions": sum(m["upstream_status"] == "ontology_coverage"
                                         for m in inventory["mentions"]),
                "exact_mentions": sum(m["offset_status"] == "exact" for m in inventory["mentions"]),
                "kept_mentions": sum(row["status"] == "keep" for row in normalized),
                "candidate_pairs": retrieval["candidate_count"], "concepts": len(concepts),
                "placement_decisions": len(placement["decisions"]), "nodes": len(placement["nodes"]),
                "edges": len(placement["edges"]), "go_references": placement["go_reference_count"],
                "concept_links": links["counts"],
                "article_relations": projection["counts"]["article_relations"],
                "concept_relations": projection["counts"]["concept_relations"],
                "merged_evidence": projection["counts"]["merged_evidence"],
                "unresolved_evidence": projection["counts"]["unresolved_evidence"],
                "relation_gates": projection["counts"]["gate"],
                "relation_novelty": projection["counts"]["novelty"],
                "graph_rows": {name: meta["rows"] for name, meta in graph["files"].items()},
            }
            manifest.update(status="complete", phase="complete", counts=counts,
                            placement_sha256=file_hash(placement_path),
                            relations_sha256=file_hash(relations_path))
            atomic_json(manifest_path, manifest)
            return manifest
        except (Exception, KeyboardInterrupt) as error:
            manifest.update(status="failed", error_type=type(error).__name__)
            manifest["stages"][stage] = "failed"
            atomic_json(manifest_path, manifest)
            raise
