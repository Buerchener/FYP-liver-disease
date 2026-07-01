#!/usr/bin/env python3
"""
Entity linking preflight for PubMed extraction results.

This script checks whether extracted entities can be linked to existing Neo4j
core KG nodes before `multi_stage_extraction_pipeline.py --write-neo4j` is run.
It is intentionally read-only: it queries Neo4j, builds a local index, enriches
the extraction JSON, and reports why candidate relations would or would not be
safe to import.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "extraction_output" / "entity_linking_preflight"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"

DEFAULT_HTTP_URL = os.environ.get("NEO4J_HTTP_URL", "http://100.104.181.96:7474")
DEFAULT_DATABASE = os.environ.get("NEO4J_DATABASE", "liver-kg-core-v02")
DEFAULT_USER = os.environ.get("NEO4J_USER", "neo4j")
DEFAULT_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

TARGET_RELATION_SIGNATURES = {
    "ASSOCIATED_WITH": {("Gene", "Disease"), ("Metabolite", "Disease")},
    "PROGNOSTIC_IN": {("Gene", "Disease")},
    "PROGRESSES_TO": {("Disease", "Disease")},
    "ENCODES": {("Gene", "Protein")},
    "INTERACTS_WITH": {("Protein", "Protein")},
    "PARTICIPATES_IN": {("Gene", "Pathway")},
    "EXPRESSED_IN": {("Gene", "Tissue"), ("Gene", "CellType")},
    "ASSOCIATED_WITH_METABOLITE": {("Gene", "Metabolite")},
}

IMPORTABLE_PREDICATES = {
    "ASSOCIATED_WITH",
    "PROGNOSTIC_IN",
    "INTERACTS_WITH",
    "PARTICIPATES_IN",
    "EXPRESSED_IN",
    "ASSOCIATED_WITH_METABOLITE",
}

BLOCKING_FLAGS = {
    "ungrounded_evidence",
    "missing_subject",
    "missing_object",
    "missing_predicate",
    "invalid_predicate",
    "subject_not_in_entities",
    "object_not_in_entities",
    "negated_relation",
    "uncertain_relation",
    "generic_background_progression",
    "non_human_or_mixed_species",
    "not_importable_policy",
    "unsupported_entity_class",
}

LABELS_TO_INDEX = ("Gene", "Protein", "Metabolite", "Pathway", "Disease", "Tissue", "CellType")

STOPWORDS = {
    "a",
    "an",
    "and",
    "associated",
    "biosynthesis",
    "disease",
    "human",
    "humans",
    "in",
    "of",
    "pathway",
    "pathways",
    "signaling",
    "signal",
    "the",
    "to",
}

GREEK = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "κ": "kappa",
    "μ": "mu",
    "ω": "omega",
}

SCHEMA_FORCED_ENTITY_CUES = {
    "Metabolite": (
        "adjuvant",
        "therapy",
        "therapies",
        "treatment",
        "drug",
        "inhibitor",
        "extract",
        "formula",
        "decoction",
    ),
    "Protein": ("inhibitor", "therapy", "treatment", "drug"),
    "Pathway": ("therapy", "treatment", "drug", "extract", "formula", "decoction"),
    "CellType": ("therapy", "treatment"),
}

# High-value biomedical aliases that help bridge common abstract mentions to
# the limited current KG index. These are treated as candidate-generation hints,
# not as write-time facts.
CURATED_ALIAS_HINTS = {
    "nrf2": {"Gene": ["NFE2L2"], "Protein": ["NFE2L2"]},
    "nuclear factor erythroid 2 related factor 2": {"Gene": ["NFE2L2"], "Protein": ["NFE2L2"]},
    "nuclear factor erythroid-2-related factor 2": {"Gene": ["NFE2L2"], "Protein": ["NFE2L2"]},
    "p38 mapk": {"Gene": ["MAPK14"], "Protein": ["MAPK14"]},
    "p38 mitogen activated protein kinase": {"Gene": ["MAPK14"], "Protein": ["MAPK14"]},
    "par 1": {"Gene": ["F2R"], "Protein": ["F2R"]},
    "protease activated receptor 1": {"Gene": ["F2R"], "Protein": ["F2R"]},
    "coagulation factor xa": {"Gene": ["F10"], "Protein": ["F10"]},
    "fxa": {"Gene": ["F10"], "Protein": ["F10"]},
    "growth hormone": {"Gene": ["GH1"], "Protein": ["GH1"]},
    "gh": {"Gene": ["GH1"], "Protein": ["GH1"]},
    "von willebrand factor": {"Gene": ["VWF"], "Protein": ["VWF"]},
}


@dataclass
class EntityEntry:
    label: str
    node_id: str
    name: str
    source: str = ""
    props: dict[str, Any] = field(default_factory=dict)
    aliases: set[str] = field(default_factory=set)

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "node_id": self.node_id,
            "name": self.name,
            "source": self.source,
            "aliases": sorted(self.aliases),
            "props": self.props,
        }


class Neo4jHTTPClient:
    def __init__(self, url: str, database: str, user: str, password: str) -> None:
        self.url = url.rstrip("/")
        self.database = database
        self.user = user
        self.password = password

    def run(self, statement: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        endpoint = f"{self.url}/db/{self.database}/tx/commit"
        payload = json.dumps(
            {"statements": [{"statement": statement, "parameters": parameters or {}}]}
        ).encode("utf-8")
        request = urllib.request.Request(endpoint, data=payload, method="POST")
        request.add_header("Content-Type", "application/json")
        token = base64.b64encode(f"{self.user}:{self.password}".encode("utf-8")).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise RuntimeError("Neo4j authentication failed. Set NEO4J_PASSWORD.") from exc
            raise
        if data.get("errors"):
            raise RuntimeError(data["errors"])
        result = data["results"][0]
        columns = result["columns"]
        return [dict(zip(columns, row["row"])) for row in result["data"]]


def fetch_neo4j_index(client: Neo4jHTTPClient) -> list[dict[str, Any]]:
    statement = """
    UNWIND $labels AS label
    CALL {
      WITH label
      MATCH (n)
      WHERE label IN labels(n)
      RETURN label AS node_label, properties(n) AS props
    }
    RETURN node_label, props
    """
    return client.run(statement, {"labels": list(LABELS_TO_INDEX)})


def load_or_fetch_index(args: argparse.Namespace) -> list[EntityEntry]:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "neo4j_entity_index.json"
    if cache_path.exists() and not args.refresh_index:
        rows = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        if not args.neo4j_password:
            raise RuntimeError(
                "No cached index found and NEO4J_PASSWORD is not set. "
                "Set NEO4J_PASSWORD or pass --refresh-index after configuring Neo4j."
            )
        client = Neo4jHTTPClient(args.neo4j_http_url, args.neo4j_database, args.neo4j_user, args.neo4j_password)
        rows = fetch_neo4j_index(client)
        cache_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    entries = [entry_from_neo4j_row(row) for row in rows]
    apply_local_alias_enrichment(entries)
    return entries


def entry_from_neo4j_row(row: dict[str, Any]) -> EntityEntry:
    label = row["node_label"]
    props = row["props"]
    node_id = (
        props.get("gene_id")
        or props.get("metabolite_id")
        or props.get("pathway_id")
        or props.get("protein_id")
        or props.get("string_protein_id")
        or props.get("disease_id")
        or props.get("tissue_id")
        or props.get("cell_type_id")
        or props.get("name")
        or ""
    )
    name = props.get("name") or props.get("disease_name") or str(node_id)
    entry = EntityEntry(label=label, node_id=str(node_id), name=str(name), source=str(props.get("source", "")), props=props)
    short_alias_keys = {
        "Gene": ("gene_id", "name", "gene_symbol", "ncbi_gene_id", "ensembl_gene_ids"),
        "Protein": ("protein_id", "string_protein_id", "name"),
        "Metabolite": ("metabolite_id", "name", "chemical_formula"),
        "Pathway": ("pathway_id", "name"),
        "Disease": ("disease_id", "name", "disease_name", "external_ids"),
        "Tissue": ("tissue_id", "name", "tissue_name"),
        "CellType": ("cell_type_id", "name", "cell_type_name"),
    }
    for key in short_alias_keys.get(label, ("name",)):
        add_alias_value(entry.aliases, props.get(key))
    if label == "Gene":
        add_alias_value(entry.aliases, props.get("gene_symbol"))
        add_alias_value(entry.aliases, props.get("ncbi_gene_id"))
    if label == "Pathway":
        add_pathway_aliases(entry.aliases, props.get("name"))
        add_pathway_aliases(entry.aliases, props.get("pathway_id"))
    if label == "Disease":
        add_alias_value(entry.aliases, props.get("disease_name"))
        add_alias_value(entry.aliases, props.get("external_ids"))
    entry.aliases = {alias for alias in entry.aliases if alias}
    return entry


def add_alias_value(aliases: set[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            add_alias_value(aliases, item)
        return
    text = str(value).strip()
    if not text:
        return
    if len(text) > 160 or text.count(" ") > 18 or "\n" in text:
        return
    if keep_alias(text):
        aliases.add(text)
    for part in re.split(r"[;|,]", text):
        part = part.strip()
        if keep_alias(part):
            aliases.add(part)


def add_pathway_aliases(aliases: set[str], value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    variants = {
        text,
        re.sub(r"\s*-\s*Homo sapiens\s*\(human\)\s*$", "", text, flags=re.I),
        re.sub(r"^KEGG:", "", text, flags=re.I),
        re.sub(r"^path:", "", text, flags=re.I),
    }
    for variant in list(variants):
        variant = variant.strip()
        if not keep_alias(variant):
            continue
        aliases.add(variant)
        for reduced in (
            re.sub(r"\s+signaling pathway$", "", variant, flags=re.I),
            re.sub(r"\s+pathway$", "", variant, flags=re.I),
            re.sub(r"\s+metabolism$", "", variant, flags=re.I),
        ):
            if keep_alias(reduced):
                aliases.add(reduced)


def keep_alias(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    compact = compact_text(text)
    if compact.isdigit():
        return False
    if len(compact) >= 3:
        return True
    return len(compact) >= 2 and text.upper() == text and any(ch.isalpha() for ch in text)


def apply_local_alias_enrichment(entries: list[EntityEntry]) -> None:
    by_id = {(entry.label, entry.node_id): entry for entry in entries}
    processed_root = SCRIPT_DIR / "workstreams" / "literature_hmdb_kegg" / "data" / "processed"
    for path in processed_root.glob("*/entities.jsonl"):
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    rec = json.loads(line)
                    label = rec.get("entity_type")
                    node_id = rec.get("project_id") or rec.get("id")
                    if label == "Gene" and rec.get("gene_id"):
                        node_id = f"NCBIGene:{rec['gene_id']}"
                    if label == "Pathway" and rec.get("pathway_id"):
                        node_id = rec.get("pathway_id")
                    entry = by_id.get((label, node_id))
                    if not entry:
                        continue
                    for key in ("name", "gene_symbol", "pathway_name", "id", "project_id"):
                        add_alias_value(entry.aliases, rec.get(key))
                    add_alias_value(entry.aliases, rec.get("aliases"))
                    if label == "Gene":
                        symbol = extract_gene_symbol(rec.get("gene_symbol") or rec.get("name"))
                        add_alias_value(entry.aliases, symbol)
                    if label == "Pathway":
                        add_pathway_aliases(entry.aliases, rec.get("pathway_name"))
        except (OSError, json.JSONDecodeError):
            continue


def extract_gene_symbol(value: Any) -> str:
    text = str(value or "").strip()
    if "\t" in text:
        return text.split("\t")[-1].strip()
    return text


def normalize_text(text: Any) -> str:
    text = str(text or "").strip()
    for src, dst in GREEK.items():
        text = text.replace(src, dst)
    text = text.replace("&", " and ")
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def compact_text(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


def token_set(text: Any) -> set[str]:
    return {token for token in normalize_text(text).split() if token and token not in STOPWORDS}


def looks_like_forced_entity_class(mention: str, entity_type: str) -> bool:
    lower = normalize_text(mention)
    return any(cue in lower for cue in SCHEMA_FORCED_ENTITY_CUES.get(entity_type, ()))


def candidate_labels_for_relation(predicate: str, side: str, other_type: str | None = None) -> set[str]:
    labels: set[str] = set()
    for source_type, target_type in TARGET_RELATION_SIGNATURES.get(predicate, set()):
        candidate = source_type if side == "subject" else target_type
        other = target_type if side == "subject" else source_type
        if other_type and other != other_type:
            continue
        labels.add(candidate)
    return labels


class EntityLinker:
    def __init__(self, entries: list[EntityEntry]) -> None:
        self.entries = entries
        self.by_label: dict[str, list[EntityEntry]] = {}
        self.alias_index: dict[tuple[str, str], list[EntityEntry]] = {}
        self.compact_index: dict[tuple[str, str], list[EntityEntry]] = {}
        self.lexical_aliases: dict[str, list[tuple[EntityEntry, str, str, str, set[str]]]] = {}
        self.id_index: dict[str, list[EntityEntry]] = {}
        for entry in entries:
            self.by_label.setdefault(entry.label, []).append(entry)
            self.id_index.setdefault(entry.node_id.lower(), []).append(entry)
            for alias in entry.aliases:
                norm = normalize_text(alias)
                comp = compact_text(alias)
                if norm:
                    self.alias_index.setdefault((entry.label, norm), []).append(entry)
                    self.lexical_aliases.setdefault(entry.label, []).append(
                        (entry, alias, norm, comp, token_set(norm))
                    )
                if comp:
                    self.compact_index.setdefault((entry.label, comp), []).append(entry)

    def link(
        self,
        mention: str,
        entity_type: str,
        *,
        normalized_id: str = "",
        context: str = "",
        allowed_labels: set[str] | None = None,
        methods: tuple[str, ...] = ("exact", "lexical", "alias_hints"),
    ) -> list[dict[str, Any]]:
        labels = sorted(allowed_labels or ({entity_type} if entity_type else set(LABELS_TO_INDEX)))
        candidates: list[dict[str, Any]] = []
        if "exact" in methods:
            candidates.extend(self._exact_candidates(mention, normalized_id, labels))
        if "alias_hints" in methods:
            candidates.extend(self._alias_hint_candidates(mention, labels))
        if "lexical" in methods:
            candidates.extend(self._lexical_candidates(mention, labels, context))
        return dedupe_candidates(candidates)[:10]

    def _exact_candidates(self, mention: str, normalized_id: str, labels: list[str]) -> list[dict[str, Any]]:
        candidates = []
        norm = normalize_text(mention)
        comp = compact_text(mention)
        ids = [normalized_id, normalized_id.replace("HGNC:", ""), normalized_id.replace("MENTION:", "")]
        for raw_id in ids:
            raw_id = str(raw_id or "").lower()
            if not raw_id:
                continue
            for entry in self.id_index.get(raw_id, []):
                if entry.label in labels:
                    candidates.append(candidate(entry, 1.0, "exact_id", matched_alias=raw_id))
        for label in labels:
            for entry in self.alias_index.get((label, norm), []):
                candidates.append(candidate(entry, 0.98, "exact_alias", matched_alias=mention))
            for entry in self.compact_index.get((label, comp), []):
                candidates.append(candidate(entry, 0.96, "compact_alias", matched_alias=mention))
        return candidates

    def _alias_hint_candidates(self, mention: str, labels: list[str]) -> list[dict[str, Any]]:
        hint = CURATED_ALIAS_HINTS.get(normalize_text(mention))
        if not hint:
            return []
        candidates = []
        for label in labels:
            for alias in hint.get(label, []):
                for entry in self.alias_index.get((label, normalize_text(alias)), []):
                    candidates.append(candidate(entry, 0.94, "curated_alias_hint", matched_alias=alias))
        return candidates

    def _lexical_candidates(self, mention: str, labels: list[str], context: str) -> list[dict[str, Any]]:
        mention_norm = normalize_text(mention)
        mention_compact = compact_text(mention)
        mention_tokens = token_set(mention)
        candidates = []
        if not mention_norm:
            return candidates
        for label in labels:
            if looks_like_forced_entity_class(mention, label):
                continue
            for entry, alias, alias_norm, alias_compact, alias_tokens in self.lexical_aliases.get(label, []):
                best_score = 0.0
                best_alias = ""
                score = lexical_score(
                    mention_norm,
                    mention_compact,
                    mention_tokens,
                    alias_norm,
                    alias_compact,
                    alias_tokens,
                )
                if score > best_score:
                    best_score = score
                    best_alias = alias
                if best_score >= threshold_for_label(label):
                    context_bonus = context_score(context, entry)
                    candidates.append(
                        candidate(
                            entry,
                            min(0.93, best_score + context_bonus),
                            "lexical_ensemble",
                            matched_alias=best_alias,
                            context_bonus=context_bonus,
                        )
                    )
        return candidates


def lexical_score(
    mention_norm: str,
    mention_compact: str,
    mention_tokens: set[str],
    alias_norm: str,
    alias_compact: str,
    alias_tokens: set[str],
) -> float:
    if not alias_norm or not alias_compact:
        return 0.0
    if mention_norm == alias_norm:
        return 0.98
    if mention_compact == alias_compact:
        return 0.96
    if len(mention_compact) >= 3 and (mention_compact in alias_compact or alias_compact in mention_compact):
        shorter = min(len(mention_compact), len(alias_compact))
        longer = max(len(mention_compact), len(alias_compact))
        return 0.72 + 0.18 * (shorter / max(longer, 1))
    jaccard = len(mention_tokens & alias_tokens) / max(len(mention_tokens | alias_tokens), 1)
    seq = difflib.SequenceMatcher(None, mention_norm, alias_norm).ratio()
    token_overlap = len(mention_tokens & alias_tokens) / max(len(mention_tokens), 1)
    return max(seq * 0.86, jaccard * 0.92, token_overlap * 0.88)


def threshold_for_label(label: str) -> float:
    if label in {"Gene", "Protein"}:
        return 0.88
    if label == "Pathway":
        return 0.84
    if label == "Metabolite":
        return 0.82
    return 0.84


def context_score(context: str, entry: EntityEntry) -> float:
    if not context:
        return 0.0
    context_tokens = token_set(context)
    alias_tokens: set[str] = set()
    for alias in entry.aliases:
        alias_tokens |= token_set(alias)
    overlap = len(context_tokens & alias_tokens)
    return min(0.04, overlap * 0.01)


def candidate(
    entry: EntityEntry,
    score: float,
    method: str,
    *,
    matched_alias: str = "",
    context_bonus: float = 0.0,
) -> dict[str, Any]:
    return {
        "label": entry.label,
        "node_id": entry.node_id,
        "name": entry.name,
        "source": entry.source,
        "score": round(score, 4),
        "method": method,
        "matched_alias": matched_alias,
        "context_bonus": round(context_bonus, 4),
    }


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for cand in candidates:
        key = (cand["label"], cand["node_id"])
        current = best.get(key)
        if current is None or cand["score"] > current["score"]:
            best[key] = cand
        elif current is not None and cand["score"] == current["score"]:
            methods = set(str(current.get("method", "")).split("+")) | {cand["method"]}
            current["method"] = "+".join(sorted(m for m in methods if m))
    return sorted(best.values(), key=lambda item: (-item["score"], item["label"], item["name"]))


def load_results(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input must be an extraction_results_*.json list")
    return data


def relation_context(record: dict[str, Any], rel: dict[str, Any]) -> str:
    return " ".join(
        str(value or "")
        for value in [
            record.get("title", ""),
            rel.get("evidence", ""),
            record.get("abstract", "")[:1200],
        ]
    )


def find_entity(extraction: dict[str, Any], mention: str, entity_type: str) -> dict[str, Any] | None:
    key = normalize_text(mention)
    for entity in extraction.get("entities", []):
        if entity_type and entity.get("type") != entity_type:
            continue
        if normalize_text(entity.get("mention", "")) == key:
            return entity
    return None


def quality_allows_import(rel: dict[str, Any]) -> bool:
    flags = set(rel.get("quality_flags", []))
    return bool(
        rel.get("evidence_grounded", True)
        and not rel.get("negated", False)
        and not rel.get("uncertain", False)
        and not (flags & BLOCKING_FLAGS)
    )


def enrich_results(
    results: list[dict[str, Any]],
    linker: EntityLinker,
    methods: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method_summaries = {name: fresh_summary() for name in ("exact", "lexical", "schema_repair", "ensemble")}
    enriched = json.loads(json.dumps(results, ensure_ascii=False))
    all_entity_rows: list[dict[str, Any]] = []
    all_relation_rows: list[dict[str, Any]] = []

    for record_index, result in enumerate(enriched):
        record = result.get("record", {})
        extraction = result.get("extraction", {})
        entity_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for entity in extraction.get("entities", []):
            mention = entity.get("mention", "")
            entity_type = entity.get("type", "")
            normalized_id = entity.get("normalized_id", "")
            context = f"{record.get('title', '')} {record.get('abstract', '')[:1200]}"
            candidates_by_method = {
                "exact": linker.link(mention, entity_type, normalized_id=normalized_id, context=context, methods=("exact",)),
                "lexical": linker.link(mention, entity_type, normalized_id=normalized_id, context=context, methods=("exact", "lexical", "alias_hints")),
                "ensemble": linker.link(mention, entity_type, normalized_id=normalized_id, context=context, methods=methods),
            }
            best = candidates_by_method["ensemble"][0] if candidates_by_method["ensemble"] else None
            entity["linking_preflight"] = {
                "best_candidate": best,
                "candidates": candidates_by_method["ensemble"][:5],
                "status": status_for_candidates(candidates_by_method["ensemble"]),
            }
            entity_cache[(normalize_text(mention), entity_type)] = candidates_by_method["ensemble"]
            for method_name, candidates in candidates_by_method.items():
                update_entity_summary(method_summaries[method_name], candidates)
            all_entity_rows.append(
                {
                    "record_index": record_index,
                    "pmid": record.get("pmid", ""),
                    "mention": mention,
                    "type": entity_type,
                    "normalized_id": normalized_id,
                    "best": best,
                    "status": entity["linking_preflight"]["status"],
                }
            )

        for rel in extraction.get("relations", []):
            relation_reports = {}
            for method_name in ("exact", "lexical", "schema_repair", "ensemble"):
                relation_reports[method_name] = evaluate_relation_linking(
                    record,
                    extraction,
                    rel,
                    linker,
                    method_name=method_name,
                    methods=methods,
                )
                update_relation_summary(method_summaries[method_name], relation_reports[method_name])
            rel["linking_preflight"] = relation_reports["ensemble"]
            all_relation_rows.append(
                {
                    "record_index": record_index,
                    "pmid": record.get("pmid", ""),
                    "subject": rel.get("subject", ""),
                    "subject_type": rel.get("subject_type", ""),
                    "predicate": rel.get("predicate", ""),
                    "object": rel.get("object", ""),
                    "object_type": rel.get("object_type", ""),
                    "ensemble_decision": relation_reports["ensemble"].get("decision"),
                    "ensemble_reason": relation_reports["ensemble"].get("reason"),
                    "subject_candidate": relation_reports["ensemble"].get("subject_candidate"),
                    "object_candidate": relation_reports["ensemble"].get("object_candidate"),
                }
            )

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method_summaries": method_summaries,
        "entity_rows": all_entity_rows,
        "relation_rows": all_relation_rows,
    }
    return enriched, report


def fresh_summary() -> dict[str, Any]:
    return {
        "entities_total": 0,
        "entities_linked": 0,
        "entities_ambiguous": 0,
        "relations_total": 0,
        "relations_endpoint_linked": 0,
        "relations_import_candidate": 0,
        "relation_reasons": {},
    }


def update_entity_summary(summary: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    summary["entities_total"] += 1
    if candidates:
        summary["entities_linked"] += 1
    if len(candidates) > 1 and candidates[0]["score"] - candidates[1]["score"] < 0.03:
        summary["entities_ambiguous"] += 1


def update_relation_summary(summary: dict[str, Any], relation_report: dict[str, Any]) -> None:
    summary["relations_total"] += 1
    if relation_report.get("endpoint_linked"):
        summary["relations_endpoint_linked"] += 1
    if relation_report.get("decision") == "import_candidate":
        summary["relations_import_candidate"] += 1
    reason = relation_report.get("reason", "unknown")
    summary["relation_reasons"][reason] = summary["relation_reasons"].get(reason, 0) + 1


def status_for_candidates(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "unlinked"
    if len(candidates) > 1 and candidates[0]["score"] - candidates[1]["score"] < 0.03:
        return "ambiguous"
    if candidates[0]["score"] >= 0.92:
        return "linked_high_confidence"
    return "linked_review"


def evaluate_relation_linking(
    record: dict[str, Any],
    extraction: dict[str, Any],
    rel: dict[str, Any],
    linker: EntityLinker,
    *,
    method_name: str,
    methods: tuple[str, ...],
) -> dict[str, Any]:
    predicate = rel.get("predicate", "")
    subject_type = rel.get("subject_type", "")
    object_type = rel.get("object_type", "")
    context = relation_context(record, rel)

    if method_name == "exact":
        method_tuple = ("exact",)
        allow_repair = False
    elif method_name == "lexical":
        method_tuple = ("exact", "lexical", "alias_hints")
        allow_repair = False
    elif method_name == "schema_repair":
        method_tuple = ("exact", "lexical", "alias_hints")
        allow_repair = True
    else:
        method_tuple = methods
        allow_repair = True

    subj_candidates = link_relation_endpoint(
        extraction,
        linker,
        rel,
        side="subject",
        entity_type=subject_type,
        context=context,
        methods=method_tuple,
        allow_repair=allow_repair,
    )
    obj_candidates = link_relation_endpoint(
        extraction,
        linker,
        rel,
        side="object",
        entity_type=object_type,
        context=context,
        methods=method_tuple,
        allow_repair=allow_repair,
    )
    subj = subj_candidates[0] if subj_candidates else None
    obj = obj_candidates[0] if obj_candidates else None
    repaired_signature = bool(subj and obj and (subj["label"], obj["label"]) in TARGET_RELATION_SIGNATURES.get(predicate, set()))
    endpoint_linked = bool(subj and obj)

    if predicate not in IMPORTABLE_PREDICATES:
        decision, reason = "review", "predicate_not_importable_policy"
    elif not repaired_signature:
        decision, reason = "review", "schema_or_endpoint_not_linked"
    elif not quality_allows_import(rel):
        decision, reason = "review", "quality_blocking_flags"
    elif ambiguous(subj_candidates) or ambiguous(obj_candidates):
        decision, reason = "review", "ambiguous_endpoint"
    else:
        decision, reason = "import_candidate", "ready"

    return {
        "method": method_name,
        "decision": decision,
        "reason": reason,
        "endpoint_linked": endpoint_linked,
        "schema_valid_after_linking": repaired_signature,
        "subject_candidate": subj,
        "object_candidate": obj,
        "subject_candidates": subj_candidates[:5],
        "object_candidates": obj_candidates[:5],
    }


def link_relation_endpoint(
    extraction: dict[str, Any],
    linker: EntityLinker,
    rel: dict[str, Any],
    *,
    side: str,
    entity_type: str,
    context: str,
    methods: tuple[str, ...],
    allow_repair: bool,
) -> list[dict[str, Any]]:
    mention = rel.get(side, "")
    entity = find_entity(extraction, mention, entity_type)
    normalized_id = ""
    if entity:
        normalized_id = entity.get("normalized_id", "")
    labels = {entity_type} if entity_type else set()
    if allow_repair:
        other_type = rel.get("object_type" if side == "subject" else "subject_type", "")
        labels |= candidate_labels_for_relation(rel.get("predicate", ""), side, other_type)
        labels |= candidate_labels_for_relation(rel.get("predicate", ""), side)
    labels = {label for label in labels if label in LABELS_TO_INDEX}
    return linker.link(
        mention,
        entity_type,
        normalized_id=normalized_id,
        context=context,
        allowed_labels=labels or None,
        methods=methods,
    )


def ambiguous(candidates: list[dict[str, Any]]) -> bool:
    return bool(len(candidates) > 1 and candidates[0]["score"] - candidates[1]["score"] < 0.03)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown_report(path: Path, input_path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entity Linking Preflight Report",
        "",
        f"Input: `{input_path}`",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Method Comparison",
        "",
        "| Method | Entities linked | Ambiguous entities | Endpoint-linked relations | Import candidates | Top blockers |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for method, summary in report["method_summaries"].items():
        blockers = ", ".join(
            f"{key}:{value}"
            for key, value in sorted(
                summary["relation_reasons"].items(),
                key=lambda item: (-item[1], item[0]),
            )[:4]
        )
        entity_cell = (
            "n/a"
            if summary["entities_total"] == 0
            else f"{summary['entities_linked']}/{summary['entities_total']}"
        )
        lines.append(
            "| {method} | {entity_cell} | {amb} | {rl}/{rt} | {ic} | {blockers} |".format(
                method=method,
                entity_cell=entity_cell,
                amb=summary["entities_ambiguous"],
                rl=summary["relations_endpoint_linked"],
                rt=summary["relations_total"],
                ic=summary["relations_import_candidate"],
                blockers=blockers or "-",
            )
        )
    lines.extend(["", "## Import-Candidate Relations", ""])
    ready = [row for row in report["relation_rows"] if row["ensemble_decision"] == "import_candidate"]
    if not ready:
        lines.append("No relation passed the ensemble preflight import gate.")
    else:
        lines.append("| PMID | Relation | Subject node | Object node |")
        lines.append("| --- | --- | --- | --- |")
        for row in ready:
            subj = row.get("subject_candidate") or {}
            obj = row.get("object_candidate") or {}
            lines.append(
                f"| {row['pmid']} | `{row['subject']} -[{row['predicate']}]-> {row['object']}` "
                f"| `{subj.get('label','')}:{subj.get('node_id','')}` | `{obj.get('label','')}:{obj.get('node_id','')}` |"
            )
    lines.extend(["", "## Review Samples", ""])
    review = [row for row in report["relation_rows"] if row["ensemble_decision"] != "import_candidate"][:25]
    lines.append("| PMID | Relation | Reason | Best subject | Best object |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in review:
        subj = row.get("subject_candidate") or {}
        obj = row.get("object_candidate") or {}
        lines.append(
            f"| {row['pmid']} | `{row['subject']} -[{row['predicate']}]-> {row['object']}` "
            f"| `{row['ensemble_reason']}` | `{subj.get('label','')}:{subj.get('node_id','')}` "
            f"| `{obj.get('label','')}:{obj.get('node_id','')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_run_id(input_path: Path) -> str:
    digest = hashlib.sha1(str(input_path).encode("utf-8")).hexdigest()[:8]
    return f"entity_linking_{time.strftime('%Y%m%d_%H%M%S')}_{digest}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only entity linking preflight for PubMed extraction results")
    parser.add_argument("--input", "-i", required=True, help="Path to extraction_results_*.json")
    parser.add_argument("--run-id", default="", help="Output run id")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Neo4j index cache directory")
    parser.add_argument("--refresh-index", action="store_true", help="Re-query Neo4j even if cache exists")
    parser.add_argument("--neo4j-http-url", default=DEFAULT_HTTP_URL)
    parser.add_argument("--neo4j-database", default=DEFAULT_DATABASE)
    parser.add_argument("--neo4j-user", default=DEFAULT_USER)
    parser.add_argument("--neo4j-password", default=DEFAULT_PASSWORD)
    parser.add_argument(
        "--methods",
        default="exact,lexical,alias_hints",
        help="Comma-separated candidate generators for ensemble mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    run_id = args.run_id or build_run_id(input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())

    results = load_results(input_path)
    entries = load_or_fetch_index(args)
    linker = EntityLinker(entries)
    enriched, report = enrich_results(results, linker, methods)

    linked_path = output_dir / f"{run_id}_linked_extraction_results.json"
    report_path = output_dir / f"{run_id}_preflight_report.json"
    entity_rows_path = output_dir / f"{run_id}_entity_candidates.jsonl"
    relation_rows_path = output_dir / f"{run_id}_relation_preflight.jsonl"
    markdown_path = output_dir / f"{run_id}_preflight_report.md"

    linked_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(entity_rows_path, report["entity_rows"])
    write_jsonl(relation_rows_path, report["relation_rows"])
    write_markdown_report(markdown_path, input_path, report)

    print(f"[OK] linked extraction: {linked_path}")
    print(f"[OK] preflight report: {report_path}")
    print(f"[OK] markdown report: {markdown_path}")
    best = report["method_summaries"]["ensemble"]
    print(
        "[SUMMARY] ensemble: "
        f"entities_linked={best['entities_linked']}/{best['entities_total']} "
        f"relations_endpoint_linked={best['relations_endpoint_linked']}/{best['relations_total']} "
        f"import_candidates={best['relations_import_candidate']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
