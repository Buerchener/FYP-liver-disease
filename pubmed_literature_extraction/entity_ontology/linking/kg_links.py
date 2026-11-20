"""Link ontology concepts to v2 graph nodes using the curated data package offline.

The v2 knowledge graph (Neo4j ``liver-kg-core-v02``) is built from the TSV files
under the repository's ``data/`` directory.  This module rebuilds the node keys
and relations it needs from those files, so linking never connects to Neo4j.
Node keys follow the v2 import scripts (``scripts/import_*.py`` MERGE keys).
"""

from collections import defaultdict
import csv
import hashlib
import os
from pathlib import Path
import re

from ..storage import atomic_json, norm, read_json

VERSION = "kg-links-v1"
DEFAULT_V2_DATA = Path(__file__).resolve().parents[3] / "data"
V2_DATA_ENV = "LIVERKG_V2_DATA_DIR"

# Label -> MERGE key in the v2 graph.  Gene IDs carry the NCBIGene: prefix and
# Protein nodes are keyed by their STRING ID (scripts/import_*.py).
V2_NODE_KEYS = {
    "Disease": "disease_id", "Gene": "gene_id", "Protein": "protein_id",
    "Pathway": "pathway_id", "Metabolite": "metabolite_id",
    "Tissue": "tissue_id", "CellType": "cell_type_id",
}

V2_FILES = {
    "genes": "disgenet_liver_backbone/curated/genes.tsv",
    "diseases": "disgenet_liver_backbone/curated/diseases.tsv",
    "gene_disease": "disgenet_liver_backbone/curated/gene_disease_associations.tsv",
    "string_mapping": "string_human_liver_genes_v02/gene_string_mapping.tsv",
    "string_interactions": "string_human_liver_genes_v02/string_interactions.tsv",
    "kegg_pathways": "kegg_human_liver_genes_v02/kegg_pathways.tsv",
    "kegg_members": "kegg_human_liver_genes_v02/gene_kegg_pathway_memberships.tsv",
    "reactome_pathways": "reactome_human_liver_genes_v02/reactome_pathways.tsv",
    "reactome_members": "reactome_human_liver_genes_v02/gene_reactome_pathway_memberships.tsv",
    "cell_types": "hpa_liver_gene_context_v02/hpa_cell_types.tsv",
    "tissues": "hpa_liver_gene_context_v02/hpa_tissues.tsv",
    "cell_expression": "hpa_liver_gene_context_v02/gene_hpa_cell_type_expression.tsv",
    "tissue_expression": "hpa_liver_gene_context_v02/gene_hpa_liver_tissue_expression.tsv",
    "lihc_prognostic": "hpa_liver_gene_context_v02/gene_hpa_lihc_prognostic.tsv",
    "metabolites": "hmdb_human_liver_genes_v02/hmdb_metabolites.tsv",
    "gene_metabolite": "hmdb_human_liver_genes_v02/gene_hmdb_metabolite_associations.tsv",
    "disease_metabolite": "hmdb_human_liver_genes_v02/hmdb_disease_metabolite_associations.tsv",
}

_COMPACT = re.compile(r"[\s\-_/.,'’()\[\]]+")


def resolve_v2_data_dir(value=None):
    path = Path(value or os.environ.get(V2_DATA_ENV) or DEFAULT_V2_DATA).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(
            f"The v2 data package was not found at {path}. "
            f"Pass --v2-data or set {V2_DATA_ENV} to the repository's data/ directory."
        )
    return path


def v2_files(data_dir):
    return {name: data_dir / relative for name, relative in V2_FILES.items()
            if (data_dir / relative).is_file()}


def v2_data_digest(data_dir):
    """Identify the v2 snapshot by the files the linker reads."""
    digest = hashlib.sha256()
    for name, path in sorted(v2_files(data_dir).items()):
        digest.update(name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def name_keys(value, kind):
    """Matching keys for a name: normalized text plus a punctuation-free form."""
    base = norm(value or "")
    if not base:
        return set()
    keys = {base, _COMPACT.sub("", base)}
    if kind in {"CellType", "Tissue"}:
        # HPA uses plural cell-type names ("hepatocytes"); literature often does not.
        keys |= {key[:-1] for key in keys if key.endswith("s") and len(key) > 4}
    return {key for key in keys if key}


def _rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def gene_key(raw_id):
    return f"NCBIGene:{raw_id}"


def hmdb_key(raw_id):
    return raw_id if raw_id.startswith("HMDB:") else f"HMDB:{raw_id}"


def node_ref(label, node_id):
    return f"{label}:{node_id}"


class V2Index:
    """Names, identifiers and relations of the v2 graph, rebuilt from TSV files."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.files = v2_files(self.data_dir)
        self.nodes = {}
        self.names = defaultdict(lambda: defaultdict(set))
        self.ids = defaultdict(dict)
        self.edges = defaultdict(set)
        self.encodes = defaultdict(set)
        self._build()

    # -- construction -------------------------------------------------
    def _node(self, label, node_id, name, names=(), ids=()):
        ref = node_ref(label, node_id)
        self.nodes.setdefault(ref, {"label": label, "key": V2_NODE_KEYS[label],
                                    "id": node_id, "name": name})
        for value in (name, *names):
            for key in name_keys(value, label):
                self.names[label][key].add(ref)
        for value in (node_id, *ids):
            if value:
                self.ids[label][value.casefold()] = ref
        return ref

    def _edge(self, left, right, relation, source):
        if left in self.nodes and right in self.nodes:
            self.edges[frozenset((left, right))].add((relation, source))

    def _build(self):
        files = self.files
        protein_names = {}
        if "string_mapping" in files:
            for row in _rows(files["string_mapping"]):
                annotation = (row.get("annotation") or "").split(";", 1)[0].strip()
                protein_names[row["gene_id"]] = annotation
        if "genes" in files:
            for row in _rows(files["genes"]):
                extra = [protein_names.get(row["gene_id"], "")]
                self._node("Gene", gene_key(row["gene_id"]), row["gene_symbol"], extra,
                           [row["gene_id"], f"ncbigene:{row['gene_id']}"])
        if "string_mapping" in files:
            for row in _rows(files["string_mapping"]):
                protein_id = row.get("string_protein_id")
                if not protein_id:
                    continue
                uniprot = [f"UniProtKB:{value}" for value in (row.get("protein_ids_from_disgenet") or "").split(";") if value]
                protein = self._node("Protein", protein_id, row.get("preferred_name") or protein_id,
                                     [protein_names.get(row["gene_id"], "")],
                                     [f"STRING:{protein_id}", *uniprot])
                gene = node_ref("Gene", gene_key(row["gene_id"]))
                if gene in self.nodes:
                    self.encodes[gene].add(protein)
                    self.encodes[protein].add(gene)
                    self._edge(gene, protein, "ENCODES", "STRING")
        stages = []
        if "diseases" in files:
            for row in _rows(files["diseases"]):
                ids = [f"UMLS:{row['umls_id']}"] if row.get("umls_id") else []
                for vocabulary in (row.get("disease_vocabularies") or "").split(";"):
                    prefix, _, value = vocabulary.partition("_")
                    if value:
                        ids.append(f"{prefix}:{value}")
                        if prefix == "DO":
                            ids.append(f"DOID:{value}")
                self._node("Disease", row["disease_id"], row["disease_name"], [row.get("stage_code", "")], ids)
                if (row.get("stage_order") or "").isdigit():
                    stages.append((int(row["stage_order"]), node_ref("Disease", row["disease_id"])))
            self._disease_synonyms()
        for ordered in zip(sorted(stages), sorted(stages)[1:]):
            (_, left), (_, right) = ordered
            self._edge(left, right, "PROGRESSES_TO", "backbone")
        for name in ("kegg_pathways", "reactome_pathways"):
            if name in files:
                for row in _rows(files[name]):
                    self._node("Pathway", row["pathway_id"], row["name"], [row.get("pathway_name", "")],
                               [row.get("kegg_pathway_id", ""), row.get("reactome_stable_id", "")])
        if "cell_types" in files:
            for row in _rows(files["cell_types"]):
                self._node("CellType", row["cell_type_id"], row["name"], [row.get("cell_type_name", "")])
        if "tissues" in files:
            for row in _rows(files["tissues"]):
                self._node("Tissue", row["tissue_id"], row["name"], [row.get("tissue_name", "")])
        if "metabolites" in files:
            for row in _rows(files["metabolites"]):
                raw = row["metabolite_id"].removeprefix("HMDB:")
                self._node("Metabolite", row["metabolite_id"], row["name"], (), [raw])
        self._relations()

    def _disease_synonyms(self):
        # Reuse the extractor's curated backbone synonyms (NAFLD/MASLD, NASH/MASH, ...).
        from cognitive_agent.memory.kg_memory import KGMemory

        for synonym, disease_id in KGMemory.DISEASE_SYNONYMS.items():
            ref = node_ref("Disease", disease_id)
            if ref in self.nodes:
                for key in name_keys(synonym, "Disease"):
                    self.names["Disease"][key].add(ref)

    def _relations(self):
        files = self.files
        gene = lambda row: node_ref("Gene", gene_key(row["gene_id"]))  # noqa: E731
        plan = (
            ("gene_disease", lambda r: node_ref("Disease", r["disease_id"]), "ASSOCIATED_WITH", "DisGeNET"),
            ("lihc_prognostic", lambda r: node_ref("Disease", r["disease_id"]), "PROGNOSTIC_IN", "HPA"),
            ("kegg_members", lambda r: node_ref("Pathway", r["pathway_id"]), "PARTICIPATES_IN", "KEGG"),
            ("reactome_members", lambda r: node_ref("Pathway", r["pathway_id"]), "PARTICIPATES_IN", "Reactome"),
            ("cell_expression", lambda r: node_ref("CellType", r["cell_type_id"]), "EXPRESSED_IN", "HPA"),
            ("tissue_expression", lambda r: node_ref("Tissue", r["tissue_id"]), "EXPRESSED_IN", "HPA"),
            ("gene_metabolite", lambda r: node_ref("Metabolite", hmdb_key(r["hmdb_id"])),
             "ASSOCIATED_WITH_METABOLITE", "HMDB"),
        )
        for name, target, relation, source in plan:
            if name in files:
                for row in _rows(files[name]):
                    self._edge(gene(row), target(row), relation, source)
        if "disease_metabolite" in files:
            for row in _rows(files["disease_metabolite"]):
                self._edge(node_ref("Metabolite", hmdb_key(row["hmdb_id"])),
                           node_ref("Disease", row["disease_id"]), "ASSOCIATED_WITH", "HMDB")
        if "string_interactions" in files:
            for row in _rows(files["string_interactions"]):
                self._edge(node_ref("Protein", row["string_protein_id_a"]),
                           node_ref("Protein", row["string_protein_id_b"]), "INTERACTS_WITH", "STRING")

    # -- queries --------------------------------------------------------
    def match_names(self, label, terms):
        found = set()
        for term in terms:
            for key in name_keys(term, label):
                found |= self.names[label].get(key, set())
        return found

    def match_ids(self, label, identifiers):
        return {self.ids[label][value.casefold()] for value in identifiers
                if value and value.casefold() in self.ids[label]}

    def relations_between(self, left, right):
        """v2 relations between two nodes, treating a gene and its encoded protein as one entity."""
        lefts = {left, *self.encodes.get(left, ())}
        rights = {right, *self.encodes.get(right, ())}
        found = set()
        for a in lefts:
            for b in rights:
                if a != b:
                    found |= self.edges.get(frozenset((a, b)), set())
        return sorted(found)


def concept_identifiers(inventory):
    """Collect extractor-supplied normalized IDs for each mention."""
    by_candidate = {row["id"]: row.get("normalized_id") for row in inventory["verification_candidates"]}
    return {
        mention["id"]: sorted({by_candidate[v] for v in mention["verification_ids"] if by_candidate.get(v)})
        for mention in inventory["mentions"]
    }


def go_references(placement):
    found = {}
    for node in placement["nodes"]:
        if node.get("go_id"):
            found[node["id"]] = node["go_id"]
    for decision in placement["decisions"]:
        result = (decision.get("go_reference") or {}).get("result") or {}
        if result.get("status") == "MATCHED" and result.get("go_id"):
            found.setdefault(decision["concept_id"], result["go_id"])
    return found


def link_concept(concept, index, identifiers):
    label = concept["kind"]
    record = {"concept_id": concept["id"], "name": concept["name"], "kind": label,
              "normalized_ids": identifiers, "matched_by": [], "target": None, "candidates": []}
    if label not in V2_NODE_KEYS:
        return {**record, "status": "NEW"}
    by_name = index.match_names(label, [concept["name"], *concept.get("aliases", [])])
    by_id = index.match_ids(label, identifiers)
    if by_id and by_name and by_id & by_name:
        candidates, basis = by_id & by_name, ["normalized_id", "name"]
    elif by_id and by_name:
        candidates, basis = by_id | by_name, ["normalized_id_conflicts_with_name"]
    elif by_id:
        candidates, basis = by_id, ["normalized_id"]
    else:
        candidates, basis = by_name, ["name"] if by_name else []
    nodes = [index.nodes[ref] for ref in sorted(candidates)]
    if len(nodes) == 1:
        return {**record, "status": "LINKED", "matched_by": basis, "target": nodes[0]}
    if nodes:
        return {**record, "status": "AMBIGUOUS", "matched_by": basis, "candidates": nodes[:10]}
    return {**record, "status": "NEW"}


def link_concepts(output, data_dir, *, index=None):
    """Write concept_links.json for every literature concept and generated parent."""
    data_dir = resolve_v2_data_dir(data_dir)
    index = index or V2Index(data_dir)
    inventory = read_json(output / "input_mentions.json")
    concepts = read_json(output / "equivalence.json")["concepts"]
    placement = read_json(output / "incremental-placement/placement.json")
    by_mention = concept_identifiers(inventory)
    known = {concept["id"] for concept in concepts}
    concepts += [node for node in placement["nodes"]
                 if node["origin"] == "proposed_broader_class" and node["id"] not in known]
    go = go_references(placement)
    links = []
    for concept in concepts:
        identifiers = sorted({value for mention in concept.get("source_mentions", ())
                              for value in by_mention.get(mention, ())})
        link = link_concept(concept, index, identifiers)
        if concept["id"] in go:
            link["external_ids"] = {"GO": go[concept["id"]]}
        links.append(link)
    counts = {status: sum(link["status"] == status for link in links)
              for status in ("LINKED", "AMBIGUOUS", "NEW")}
    result = {
        "version": VERSION,
        "v2_data": {"sha256": v2_data_digest(data_dir), "files": sorted(index.files)},
        "links": links, "counts": counts,
    }
    atomic_json(output / "concept_links.json", result)
    return result
