#!/usr/bin/env python3
"""Generate Neo4j Cypher that materializes ontology YAML as a schema graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def cypher_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def cypher_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def prop_map(values: dict[str, Any]) -> str:
    parts = []
    for key, value in values.items():
        if value is None:
            parts.append(f"{key}: null")
        elif isinstance(value, bool):
            parts.append(f"{key}: {cypher_bool(value)}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}: {value}")
        else:
            parts.append(f"{key}: {cypher_string(value)}")
    return "{ " + ", ".join(parts) + " }"


def merge_node(label: str, key: str, key_value: Any, props: dict[str, Any] | None = None) -> str:
    props = props or {}
    lines = [f"MERGE (n:{label} {{{key}: {cypher_string(key_value)}}})"]
    if props:
        lines.append(f"SET n += {prop_map(props)}")
    return "\n".join(lines) + ";"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=Path, default=Path("ontology/ontology_v1.0.yaml"))
    parser.add_argument("--output", type=Path, default=Path("output/neo4j_schema/load_ontology_schema.cypher"))
    args = parser.parse_args()

    ontology = yaml.safe_load(args.yaml.read_text(encoding="utf-8"))
    statements: list[str] = []

    statements.extend(
        [
            "// Generated from ontology/ontology_v1.0.yaml.",
            "// This creates ontology/schema metadata only and does not import DisGeNET records.",
            "",
            "CREATE CONSTRAINT ontology_name IF NOT EXISTS FOR (n:Ontology) REQUIRE n.name IS UNIQUE;",
            "CREATE CONSTRAINT ontology_entity_type_name IF NOT EXISTS FOR (n:OntologyEntityType) REQUIRE n.name IS UNIQUE;",
            "CREATE CONSTRAINT ontology_relation_type_name IF NOT EXISTS FOR (n:OntologyRelationType) REQUIRE n.name IS UNIQUE;",
            "CREATE CONSTRAINT ontology_attribute_key IF NOT EXISTS FOR (n:OntologyAttribute) REQUIRE (n.owner_type, n.owner_name, n.group, n.name) IS UNIQUE;",
            "CREATE CONSTRAINT evidence_level_name IF NOT EXISTS FOR (n:EvidenceLevel) REQUIRE n.name IS UNIQUE;",
            "CREATE CONSTRAINT validation_status_value IF NOT EXISTS FOR (n:ValidationStatus) REQUIRE n.value IS UNIQUE;",
            "CREATE CONSTRAINT canonical_stage_id IF NOT EXISTS FOR (n:CanonicalDiseaseStage) REQUIRE n.stage_id IS UNIQUE;",
            "",
            "// Future data constraints for ontology-defined entity labels.",
            "CREATE CONSTRAINT gene_project_id IF NOT EXISTS FOR (n:Gene) REQUIRE n.project_id IS UNIQUE;",
            "CREATE CONSTRAINT disease_project_id IF NOT EXISTS FOR (n:Disease) REQUIRE n.project_id IS UNIQUE;",
            "CREATE CONSTRAINT protein_project_id IF NOT EXISTS FOR (n:Protein) REQUIRE n.project_id IS UNIQUE;",
            "CREATE CONSTRAINT pathway_project_id IF NOT EXISTS FOR (n:Pathway) REQUIRE n.project_id IS UNIQUE;",
            "CREATE CONSTRAINT metabolite_project_id IF NOT EXISTS FOR (n:Metabolite) REQUIRE n.project_id IS UNIQUE;",
            "CREATE CONSTRAINT tissue_project_id IF NOT EXISTS FOR (n:Tissue) REQUIRE n.project_id IS UNIQUE;",
            "CREATE CONSTRAINT celltype_project_id IF NOT EXISTS FOR (n:CellType) REQUIRE n.project_id IS UNIQUE;",
            "",
        ]
    )

    meta = ontology["ontology"]
    statements.append(
        merge_node(
            "Ontology",
            "name",
            meta["name"],
            {
                "version": meta.get("version"),
                "description": meta.get("description"),
                "disease_area": meta.get("scope", {}).get("disease_area"),
            },
        )
    )

    for index, use_case in enumerate(meta.get("scope", {}).get("primary_use_cases", []), start=1):
        statements.append(
            f"""
MATCH (o:Ontology {{name: {cypher_string(meta["name"])}}})
MERGE (u:OntologyUseCase {{name: {cypher_string(use_case)}}})
SET u.position = {index}
MERGE (o)-[:HAS_USE_CASE]->(u);
""".strip()
        )

    for index, principle in enumerate(meta.get("design_principles", []), start=1):
        statements.append(
            f"""
MATCH (o:Ontology {{name: {cypher_string(meta["name"])}}})
MERGE (p:OntologyDesignPrinciple {{text: {cypher_string(principle)}}})
SET p.position = {index}
MERGE (o)-[:HAS_DESIGN_PRINCIPLE]->(p);
""".strip()
        )

    for entity_name, entity in ontology.get("entities", {}).items():
        statements.append(
            f"""
MATCH (o:Ontology {{name: {cypher_string(meta["name"])}}})
MERGE (e:OntologyEntityType {{name: {cypher_string(entity_name)}}})
SET e.description = {cypher_string(entity.get("description"))}
MERGE (o)-[:DEFINES_ENTITY_TYPE]->(e);
""".strip()
        )
        for index, candidate in enumerate(entity.get("primary_id_candidates", []), start=1):
            statements.append(
                f"""
MATCH (e:OntologyEntityType {{name: {cypher_string(entity_name)}}})
MERGE (c:PrimaryIdCandidate {{entity_type: {cypher_string(entity_name)}, name: {cypher_string(candidate)}}})
SET c.position = {index}
MERGE (e)-[:HAS_PRIMARY_ID_CANDIDATE]->(c);
""".strip()
            )
        for group, attributes in entity.get("attributes", {}).items():
            for index, attribute in enumerate(attributes, start=1):
                statements.append(
                    f"""
MATCH (e:OntologyEntityType {{name: {cypher_string(entity_name)}}})
MERGE (a:OntologyAttribute {{owner_type: "entity", owner_name: {cypher_string(entity_name)}, group: {cypher_string(group)}, name: {cypher_string(attribute)}}})
SET a.position = {index}
MERGE (e)-[:HAS_ENTITY_ATTRIBUTE]->(a);
""".strip()
                )

    for relation_name, relation in ontology.get("relations", {}).items():
        statements.append(
            f"""
MATCH (o:Ontology {{name: {cypher_string(meta["name"])}}})
MERGE (r:OntologyRelationType {{name: {cypher_string(relation_name)}}})
SET r.description = {cypher_string(relation.get("description"))},
    r.direction = {cypher_string(relation.get("direction"))},
    r.symmetric = {cypher_bool(relation.get("symmetric"))}
MERGE (o)-[:DEFINES_RELATION_TYPE]->(r);
""".strip()
        )
        statements.append(
            f"""
MATCH (r:OntologyRelationType {{name: {cypher_string(relation_name)}}})
MATCH (s:OntologyEntityType {{name: {cypher_string(relation.get("source"))}}})
MATCH (t:OntologyEntityType {{name: {cypher_string(relation.get("target"))}}})
MERGE (r)-[:FROM_ENTITY]->(s)
MERGE (r)-[:TO_ENTITY]->(t);
""".strip()
        )
        for index, source in enumerate(relation.get("allowed_sources", []), start=1):
            statements.append(
                f"""
MATCH (r:OntologyRelationType {{name: {cypher_string(relation_name)}}})
MERGE (s:AllowedSource {{name: {cypher_string(source)}}})
SET s.position = coalesce(s.position, {index})
MERGE (r)-[:ALLOWS_SOURCE]->(s);
""".strip()
            )
        for group, attributes in relation.get("attributes", {}).items():
            for index, attribute in enumerate(attributes, start=1):
                statements.append(
                    f"""
MATCH (r:OntologyRelationType {{name: {cypher_string(relation_name)}}})
MERGE (a:OntologyAttribute {{owner_type: "relation", owner_name: {cypher_string(relation_name)}, group: {cypher_string(group)}, name: {cypher_string(attribute)}}})
SET a.position = {index}
MERGE (r)-[:HAS_RELATION_ATTRIBUTE]->(a);
""".strip()
                )

    evidence_policy = ontology.get("evidence_policy", {})
    for name, description in evidence_policy.get("evidence_levels", {}).items():
        statements.append(merge_node("EvidenceLevel", "name", name, {"description": description}))

    for index, field in enumerate(evidence_policy.get("required_relation_provenance", []), start=1):
        statements.append(
            f"""
MATCH (o:Ontology {{name: {cypher_string(meta["name"])}}})
MERGE (p:RelationProvenanceField {{name: {cypher_string(field)}, requirement: "required"}})
SET p.position = {index}
MERGE (o)-[:REQUIRES_RELATION_PROVENANCE]->(p);
""".strip()
        )

    for index, field in enumerate(evidence_policy.get("recommended_relation_provenance", []), start=1):
        statements.append(
            f"""
MATCH (o:Ontology {{name: {cypher_string(meta["name"])}}})
MERGE (p:RelationProvenanceField {{name: {cypher_string(field)}, requirement: "recommended"}})
SET p.position = {index}
MERGE (o)-[:RECOMMENDS_RELATION_PROVENANCE]->(p);
""".strip()
        )

    for index, status in enumerate(evidence_policy.get("validation_status_values", []), start=1):
        statements.append(merge_node("ValidationStatus", "value", status, {"position": index}))

    progression = ontology.get("disease_progression_model", {})
    for stage in progression.get("canonical_stages", []):
        statements.append(
            merge_node(
                "CanonicalDiseaseStage",
                "stage_id",
                stage["stage_id"],
                {
                    "name": stage.get("name"),
                    "order": stage.get("order"),
                    "disease_name": stage.get("disease_name"),
                    "is_progression_stage": stage.get("is_progression_stage"),
                },
            )
        )
        statements.append(
            f"""
MATCH (o:Ontology {{name: {cypher_string(meta["name"])}}})
MATCH (s:CanonicalDiseaseStage {{stage_id: {cypher_string(stage["stage_id"])}}})
MERGE (o)-[:DEFINES_CANONICAL_STAGE]->(s);
""".strip()
        )

    for edge in progression.get("canonical_progression_edges", []):
        statements.append(
            f"""
MATCH (source:CanonicalDiseaseStage {{name: {cypher_string(edge.get("source"))}}})
MATCH (target:CanonicalDiseaseStage {{name: {cypher_string(edge.get("target"))}}})
MERGE (source)-[:CANONICALLY_PROGRESSES_TO {{relation: {cypher_string(edge.get("relation"))}}}]->(target);
""".strip()
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(statements) + "\n", encoding="utf-8")
    print(f"wrote: {args.output}")
    print(f"statements: {len([item for item in statements if item.strip() and not item.startswith('//')])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
