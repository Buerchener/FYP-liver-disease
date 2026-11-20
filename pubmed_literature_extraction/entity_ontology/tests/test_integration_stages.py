"""Concept linking, the concept-level relation gate and the guarded Neo4j loader."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from entity_ontology.export.graph import GRAPH_FILES, VERSION as EXPORT_VERSION
from entity_ontology.export.neo4j_load import check_target, load_plan, load_run
from entity_ontology.linking.kg_links import V2Index, link_concept
from entity_ontology.relations.project import concept_gate, evidence_row
from entity_ontology.storage import atomic_json, atomic_jsonl


def write_tsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join("\t".join(row) for row in [header, *rows]) + "\n")


class ConceptLinkingTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        data = Path(self.directory.name)
        write_tsv(data / "disgenet_liver_backbone/curated/genes.tsv", ["gene_id", "gene_symbol"],
                  [["7", "KIN1"], ["8", "KIN2"], ["9", "TP53"]])
        write_tsv(data / "string_human_liver_genes_v02/gene_string_mapping.tsv",
                  ["gene_id", "string_protein_id", "preferred_name", "annotation"],
                  [["7", "9606.ENSP7", "KIN1", "Shared kinase; one"],
                   ["8", "9606.ENSP8", "KIN2", "Shared kinase; two"]])
        write_tsv(data / "disgenet_liver_backbone/curated/diseases.tsv",
                  ["stage_code", "stage_order", "disease_id", "umls_id", "disease_name", "disease_vocabularies"],
                  [["NAFLD", "1", "UMLS:C0400966", "C0400966", "Non-alcoholic Fatty Liver Disease",
                    "MESH_D065626"],
                   ["NASH", "2", "UMLS:C3241937", "C3241937", "Non-alcoholic Steatohepatitis", ""]])
        write_tsv(data / "hpa_liver_gene_context_v02/hpa_cell_types.tsv", ["cell_type_id", "name"],
                  [["HPA_CELL_TYPE:hepatocytes", "hepatocytes"]])
        self.index = V2Index(data)

    def tearDown(self):
        self.directory.cleanup()

    def concept(self, name, kind, aliases=()):
        return {"id": "C-" + name, "name": name, "kind": kind, "aliases": list(aliases)}

    def test_unique_name_links_and_shared_name_is_ambiguous(self):
        linked = link_concept(self.concept("tp53", "Gene"), self.index, [])
        self.assertEqual((linked["status"], linked["target"]["id"]), ("LINKED", "NCBIGene:9"))
        self.assertEqual(linked["target"]["key"], "gene_id")
        ambiguous = link_concept(self.concept("Shared kinase", "Gene"), self.index, [])
        self.assertEqual(ambiguous["status"], "AMBIGUOUS")
        self.assertEqual(len(ambiguous["candidates"]), 2)
        new = link_concept(self.concept("Unknown factor", "Gene"), self.index, [])
        self.assertEqual(new["status"], "NEW")

    def test_identifier_resolves_ambiguity_and_synonyms_and_plural_cell_types(self):
        by_id = link_concept(self.concept("Shared kinase", "Gene"), self.index, ["NCBIGene:8"])
        self.assertEqual((by_id["status"], by_id["target"]["id"]), ("LINKED", "NCBIGene:8"))
        self.assertEqual(by_id["matched_by"], ["normalized_id", "name"])
        masld = link_concept(self.concept("MASLD", "Disease"), self.index, [])
        self.assertEqual(masld["target"]["id"], "UMLS:C0400966")
        mesh = link_concept(self.concept("Fatty liver", "Disease"), self.index, ["MESH:D065626"])
        self.assertEqual(mesh["target"]["id"], "UMLS:C0400966")
        cell = link_concept(self.concept("Hepatocyte", "CellType"), self.index, [])
        self.assertEqual(cell["target"]["id"], "HPA_CELL_TYPE:hepatocytes")
        self.assertEqual(self.index.relations_between("Disease:UMLS:C0400966", "Disease:UMLS:C3241937"),
                         [("PROGRESSES_TO", "backbone")])


class ConceptGateTest(unittest.TestCase):
    concepts = {"A": {"kind": "Gene"}, "B": {"kind": "Disease"}}

    def relation(self, *statuses, subject_type="Gene", **extra):
        evidence = [evidence_row("1", index, {
            "subject": "a", "subject_type": subject_type, "predicate": "ASSOCIATED_WITH",
            "object": "b", "object_type": "Disease", "write_status": status, **extra,
        }) for index, status in enumerate(statuses, 1)]
        return {"subject_concept_id": "A", "predicate": "ASSOCIATED_WITH",
                "object_concept_id": "B", "evidence": evidence}

    def gate(self, relation, placed=("A", "B"), decisions=None):
        return concept_gate(relation, self.concepts, set(placed), decisions or {})

    def test_auto_needs_import_ready_evidence_on_placed_concepts(self):
        self.assertEqual(self.gate(self.relation("IMPORT_READY", "SEMANTIC_ONLY")), ("AUTO", []))
        gate, reasons = self.gate(self.relation("SEMANTIC_ONLY"))
        self.assertEqual((gate, reasons), ("REVIEW", ["no_import_ready_evidence"]))
        self.assertEqual(self.gate(self.relation("BLOCKED"))[0], "BLOCKED")

    def test_concept_problems_only_lower_the_gate(self):
        gate, reasons = self.gate(self.relation("IMPORT_READY"), placed=("B",),
                                  decisions={"A": {"action": "needs_review"}})
        self.assertEqual(gate, "REVIEW")
        self.assertIn("subject_concept_needs_review", reasons)
        gate, reasons = self.gate(self.relation("IMPORT_READY", subject_type="Protein"))
        self.assertEqual(gate, "REVIEW")
        self.assertIn("subject_type_mismatch", reasons)
        gate, reasons = self.gate(self.relation("IMPORT_READY", negated=True))
        self.assertEqual((gate, reasons), ("REVIEW", ["counter_evidence"]))


def write_export(folder, relations):
    graph = folder / "graph"
    rows = {
        "articles": [{"pmid": "1", "title": "t"}],
        "concepts": [{"uid": "r/A", "concept_id": "A", "kind": "Protein", "name": "A"},
                     {"uid": "r/B", "concept_id": "B", "kind": "Protein", "name": "B"}],
        "is_a": [], "mentions": [],
        "relations": relations,
        "evidence": [{"uid": "r/1:R001", "pmid": "1", "relation_uid": row["uid"], "subject_uid": "r/A",
                      "object_uid": "r/B", "text": "A binds B."} for row in relations],
        "same_as": [{"concept_uid": "r/A", "label": "Protein", "key": "protein_id",
                     "node_id": "9606.ENSP7", "name": "KIN1"}],
    }
    files = {}
    for name in GRAPH_FILES:
        atomic_jsonl(graph / f"{name}.jsonl", rows[name])
        files[name] = {"rows": len(rows[name]),
                       "sha256": hashlib.sha256((graph / f"{name}.jsonl").read_bytes()).hexdigest()}
    atomic_json(graph / "manifest.json", {"version": EXPORT_VERSION, "run_id": "r", "files": files})


def relation_row(uid, gate, predicate="INTERACTS_WITH"):
    return {"uid": uid, "subject_uid": "r/A", "object_uid": "r/B", "predicate": predicate, "gate": gate,
            "gate_reasons": [], "pmids": ["1"], "novelty": "NEW"}


class RecordingDriver:
    def __init__(self, uri, auth):
        self.log, self.databases = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def session(self, database):
        self.databases.append(database)
        return self

    def run(self, query, **params):
        self.log.append((query, params))
        return self

    def consume(self):
        return None

    def execute_write(self, work):
        return work(self)


class Neo4jLoaderTest(unittest.TestCase):
    def test_target_guard(self):
        for uri, database in (("bolt://10.0.0.5:7687", "liverkg-integrated"),
                              ("bolt://localhost:7687", "liver-kg-core-v02"),
                              ("bolt://localhost:7687", " ")):
            with self.assertRaises(ValueError):
                check_target(uri, database)
        check_target("bolt://127.0.0.1:7687", "liverkg-integrated")

    def test_only_auto_relations_and_allow_listed_types_are_loaded(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            write_export(folder, [relation_row("r/auto", "AUTO"), relation_row("r/review", "REVIEW")])
            dry = load_run(folder, uri="bolt://localhost:7687", database="liverkg-integrated",
                           user="neo4j", password="", dry_run=True)
            self.assertEqual(dry["counts"]["relations"], 1)
            self.assertEqual(dry["counts"]["review_relations_not_loaded"], 1)
            drivers = []

            def factory(uri, auth):
                drivers.append(RecordingDriver(uri, auth))
                return drivers[-1]

            load_run(folder, uri="bolt://localhost:7687", database="liverkg-integrated",
                     user="neo4j", password="secret", driver_factory=factory)
            driver = drivers[0]
            self.assertEqual(driver.databases, ["liverkg-integrated"])
            queries = "\n".join(query for query, _ in driver.log)
            self.assertIn("MERGE (s)-[r:`INTERACTS_WITH` {uid: row.uid}]->(o)", queries)
            self.assertIn("MERGE (n:`Protein` {`protein_id`: row.node_id})", queries)
            self.assertNotIn("DELETE", queries)
            loaded = json.dumps([params for _, params in driver.log])
            self.assertIn("r/auto", loaded)
            self.assertNotIn("r/review", loaded)

    def test_rejects_unknown_predicates_and_tampered_files(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            write_export(folder, [relation_row("r/x", "AUTO", predicate="CAUSES} DETACH DELETE")])
            with self.assertRaisesRegex(ValueError, "outside the LiverKG predicates"):
                load_plan(folder)
            write_export(folder, [relation_row("r/auto", "AUTO")])
            (folder / "graph/relations.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_plan(folder)


if __name__ == "__main__":
    unittest.main()
