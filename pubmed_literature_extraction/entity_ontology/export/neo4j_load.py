"""Load an exported run into a local Neo4j database chosen explicitly by the user.

Safety rules:
- only localhost / 127.0.0.1 / ::1 URIs;
- the database must be named explicitly and must not be the curated
  ``liver-kg-core-v02`` graph (or the ``system`` database);
- only MERGE statements, so loading is repeatable and never deletes anything;
- only AUTO relations are loaded; REVIEW relations stay in the files;
- relationship types and node labels come from fixed allow-lists, never from
  free text, before they are placed in Cypher.
Concepts that link to a v2 node get ``(:Concept)-[:SAME_AS]->`` that node.  If
the database already holds a copy of the v2 graph the edge attaches to the
existing node unchanged; otherwise a ``:V2Anchor`` node with the v2 key is made.
"""

import hashlib
from urllib.parse import urlsplit

from cognitive_agent.schema.relation_signatures import LITERATURE_CANDIDATE_SIGNATURES

from ..linking.kg_links import V2_NODE_KEYS
from ..storage import read_json, read_jsonl
from .graph import GRAPH_FILES, VERSION as EXPORT_VERSION

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
PROTECTED_DATABASES = {"liver-kg-core-v02", "system"}
PREDICATES = frozenset(LITERATURE_CANDIDATE_SIGNATURES)
BATCH_SIZE = 500

CONSTRAINTS = (
    "CREATE CONSTRAINT liverkg_concept_uid IF NOT EXISTS FOR (n:Concept) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT liverkg_article_pmid IF NOT EXISTS FOR (n:Article) REQUIRE n.pmid IS UNIQUE",
    "CREATE CONSTRAINT liverkg_mention_uid IF NOT EXISTS FOR (n:Mention) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT liverkg_evidence_uid IF NOT EXISTS FOR (n:Evidence) REQUIRE n.uid IS UNIQUE",
)


def check_target(uri, database):
    host = urlsplit(uri or "").hostname
    if host not in LOCAL_HOSTS:
        raise ValueError("Loading is restricted to a local Neo4j (localhost, 127.0.0.1 or ::1).")
    if not database or not database.strip():
        raise ValueError("Name the target database explicitly with --database.")
    if database.strip().casefold() in {name.casefold() for name in PROTECTED_DATABASES}:
        raise ValueError(f"Refusing to load into {database!r}; use a separate database.")


def load_plan(run_dir):
    """Read and validate graph/*.jsonl; return the rows that will be loaded."""
    folder = run_dir / "graph"
    manifest = read_json(folder / "manifest.json")
    if manifest.get("version") != EXPORT_VERSION:
        raise ValueError("Unsupported graph export version; rerun the pipeline.")
    rows = {}
    for name in GRAPH_FILES:
        path = folder / f"{name}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"graph/{name}.jsonl does not match the export manifest.")
        rows[name] = read_jsonl(path)
    relations = [row for row in rows["relations"] if row["gate"] == "AUTO"]
    if any(row["predicate"] not in PREDICATES for row in relations):
        raise ValueError("The export contains a relation type outside the LiverKG predicates.")
    if any(V2_NODE_KEYS.get(row["label"]) != row["key"] for row in rows["same_as"]):
        raise ValueError("The export contains a v2 link with an unknown label or key.")
    loaded = {row["uid"] for row in relations}
    plan = {**rows, "relations": relations,
            "evidence": [row for row in rows["evidence"] if row["relation_uid"] in loaded]}
    plan["run_id"] = manifest["run_id"]
    plan["counts"] = {name: len(plan[name]) for name in GRAPH_FILES}
    plan["counts"]["review_relations_not_loaded"] = len(rows["relations"]) - len(relations)
    return plan


def _props(row, *exclude):
    return {key: value for key, value in row.items() if key not in exclude}


def statements(plan):
    """Yield (name, cypher, rows); labels and types come only from allow-lists."""
    yield ("articles", "UNWIND $rows AS row MERGE (a:Article {pmid: row.pmid}) SET a.title = row.title",
           plan["articles"])
    yield ("concepts", "UNWIND $rows AS row MERGE (c:Concept {uid: row.uid}) SET c += row.props",
           [{"uid": row["uid"], "props": _props(row, "uid")} for row in plan["concepts"]])
    yield ("is_a",
           "UNWIND $rows AS row MATCH (child:Concept {uid: row.child_uid}) "
           "MATCH (parent:Concept {uid: row.parent_uid}) "
           "MERGE (child)-[r:IS_A]->(parent) SET r.support = row.support, r.run_id = row.run_id",
           plan["is_a"])
    yield ("mentions",
           "UNWIND $rows AS row MATCH (c:Concept {uid: row.concept_uid}) MATCH (a:Article {pmid: row.pmid}) "
           "MERGE (m:Mention {uid: row.uid}) SET m += row.props "
           "MERGE (m)-[:REFERS_TO]->(c) MERGE (m)-[:IN_ARTICLE]->(a)",
           [{**row, "props": _props(row, "uid", "concept_uid")} for row in plan["mentions"]])
    for predicate in sorted({row["predicate"] for row in plan["relations"]}):
        if predicate not in PREDICATES:
            raise ValueError(f"Unknown predicate {predicate!r}.")
        rows = [{"uid": row["uid"], "subject_uid": row["subject_uid"], "object_uid": row["object_uid"],
                 "props": _props(row, "subject_uid", "object_uid", "predicate")}
                for row in plan["relations"] if row["predicate"] == predicate]
        yield (f"relations:{predicate}",
               "UNWIND $rows AS row MATCH (s:Concept {uid: row.subject_uid}) "
               "MATCH (o:Concept {uid: row.object_uid}) "
               f"MERGE (s)-[r:`{predicate}` {{uid: row.uid}}]->(o) SET r += row.props",
               rows)
    yield ("evidence",
           "UNWIND $rows AS row MATCH (a:Article {pmid: row.pmid}) "
           "MATCH (s:Concept {uid: row.subject_uid}) MATCH (o:Concept {uid: row.object_uid}) "
           "MERGE (e:Evidence {uid: row.uid}) SET e += row.props "
           "MERGE (e)-[:IN_ARTICLE]->(a) "
           "MERGE (e)-[:ABOUT {role: 'subject'}]->(s) MERGE (e)-[:ABOUT {role: 'object'}]->(o)",
           [{**row, "props": _props(row, "uid", "subject_uid", "object_uid")} for row in plan["evidence"]])
    for label in sorted({row["label"] for row in plan["same_as"]}):
        key = V2_NODE_KEYS[label]
        yield (f"same_as:{label}",
               "UNWIND $rows AS row MATCH (c:Concept {uid: row.concept_uid}) "
               f"MERGE (n:`{label}` {{`{key}`: row.node_id}}) "
               "ON CREATE SET n:V2Anchor, n.name = row.name, n.source = 'v2_anchor' "
               "MERGE (c)-[:SAME_AS]->(n)",
               [row for row in plan["same_as"] if row["label"] == label])


def load_run(run_dir, *, uri, database, user, password, dry_run=False, driver_factory=None):
    check_target(uri, database)
    plan = load_plan(run_dir)
    summary = {"run_id": plan["run_id"], "uri": uri, "database": database,
               "dry_run": dry_run, "counts": plan["counts"]}
    if dry_run:
        return summary
    if not password:
        raise ValueError("Set NEO4J_PASSWORD for the target database.")
    if driver_factory is None:
        from neo4j import GraphDatabase

        driver_factory = GraphDatabase.driver
    with driver_factory(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            for statement in CONSTRAINTS:
                session.run(statement).consume()
            for _, query, rows in statements(plan):
                for start in range(0, len(rows), BATCH_SIZE):
                    chunk = rows[start:start + BATCH_SIZE]
                    session.execute_write(lambda tx, q=query, c=chunk: tx.run(q, rows=c).consume())
    return summary
