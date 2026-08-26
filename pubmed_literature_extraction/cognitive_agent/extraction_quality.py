#!/usr/bin/env python3
"""Deterministic Phase-A quality gates for PubMed KG extraction.

This module deliberately contains no LLM or Neo4j access.  It preserves raw
candidates for review, canonicalizes only high-confidence article-local aliases,
and validates quoted relation evidence against the current source text.
"""

from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from cognitive_agent.abbreviation_detector import AbbreviationDetector


HARD_REJECT_TERMS = frozenset({
    "cancer", "tumor", "tumour", "cells", "cell", "pathway", "pathways",
    "signaling", "response", "damage", "effect", "effects",
    "treatment", "drug", "drugs", "therapy", "therapies", "method", "methods",
    "analysis", "database", "databases", "biomarkers", "genes", "proteins",
    "molecules", "compounds", "agents", "targets", "immune cells",
    "tumor immune microenvironment", "tumour immune microenvironment",
    "tumor microenvironment", "tumour microenvironment", "immune microenvironment",
    "therapeutic agents", "immune checkpoint inhibitors", "cytokines",
    "tumor antigens", "gene targets", "drug targets", "hub genes",
})

METHOD_TERMS = frozenset({
    "network pharmacology", "molecular docking", "molecular dynamics",
    "gene ontology", "go analysis", "kegg", "kegg pathway", "gsea",
    "enrichment analysis", "functional enrichment", "pathway analysis",
    "differential expression analysis", "principal component analysis", "pca",
    "kaplan-meier", "cox regression", "logistic regression", "western blot",
    "western blot analysis", "elisa", "elisa assay", "flow cytometry",
    "mass spectrometry", "immunohistochemical staining", "rna sequencing",
    "rna-seq", "pcr", "qpcr", "tcga", "gtex", "geo database",
    "string database", "cytoscape", "metascape", "genecards", "disgenet",
})

STATISTICAL_TERMS = frozenset({
    "p value", "p-value", "confidence interval", "hazard ratio", "odds ratio",
    "risk ratio", "relative risk", "standard deviation", "standard error",
    "area under the curve", "receiver operating characteristic", "sensitivity",
    "specificity",
})

CONDITIONAL_PROCESS_TERMS = frozenset({
    "inflammation", "oxidative stress", "immune response", "angiogenesis",
    "metastasis", "injury", "fibrosis", "apoptosis", "proliferation",
    "replication", "immune activation",
})

ANATOMICAL_TERMS = frozenset({
    "liver", "hepatic tissue", "kidney", "blood", "serum", "plasma",
    "spleen", "lung", "heart", "pancreas", "intestine", "gut", "brain",
    "adipose tissue", "bone marrow", "bile duct", "portal vein",
    "aorta", "colon", "whole blood", "peripheral blood",
    "pulmonary artery", "coronary artery",
})

ANATOMICAL_RE = re.compile(
    r"(?:^|\b)(?:aorta|arter(?:y|ies)|veins?|ducts?|colon|intestin(?:e|al)|"
    r"blood|serum|plasma|liver|hepatic|kidney|renal|lung|pulmonary|heart|"
    r"cardiac|spleen|splenic|pancrea(?:s|tic)|brain|cerebral|marrow|"
    r"adipose|bile|portal)(?:\b|$)",
    re.IGNORECASE,
)

SPECIFIC_CELL_RE = re.compile(
    r"(?:hepatocytes?|kupffer cells?|macrophages?|neutrophils?|"
    r"(?:regulatory )?t cells?|b cells?|stellate cells?|endothelial cells?|"
    r"cholangiocytes?|fibroblasts?|monocytes?|natural killer cells?)$",
    re.IGNORECASE,
)

SPECIFIC_CELL_SUBSET_RE = re.compile(
    r"(?:macrophage|monocyte|neutrophil|hepatocyte|t cell|b cell|"
    r"stellate cell|endothelial cell)\s+(?:subsets?|clusters?)\s*[a-z0-9-]+$",
    re.IGNORECASE,
)

METHOD_EVIDENCE_RE = re.compile(
    r"\b(network pharmacology|molecular docking|docking|screen(?:ed|ing)?|"
    r"selected|database mining|bioinformatics|computational analysis|"
    r"enrichment analysis|gene ontology|gsea|western blot|elisa|flow cytometry)\b",
    re.IGNORECASE,
)
METHOD_ONLY_STRONG_RE = re.compile(
    r"\b(network pharmacology|molecular docking|docking|screen(?:ed|ing)?|"
    r"selected|database mining|bioinformatics|computational analysis|"
    r"enrichment analysis|gene ontology|gsea)\b",
    re.IGNORECASE,
)
PREDICTION_RE = re.compile(
    r"\b(predict(?:s|ed|ion)?|potential|candidate|in silico|may|might|could|"
    r"suggest(?:s|ed)?|possibly|likely)\b",
    re.IGNORECASE,
)
VALIDATION_RE = re.compile(
    r"\b(experimentally (?:validated|confirmed)|demonstrated|validated in|"
    r"confirmed in|we found|our results show)\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"\b(no|not|neither|without|failed to|lack of|absence of)\b",
    re.IGNORECASE,
)

PREDICATE_TRIGGERS: dict[str, tuple[str, ...]] = {
    "ASSOCIATED_WITH": (
        r"associated with", r"association (?:between|with)", r"correlat(?:ed|ion)",
        r"(?:^|[\s-])associated(?:\s|$)", r"characteri[sz]ed by", r"caused by",
        r"linked to", r"related to", r"contribut(?:es?|ed) to", r"promot(?:es?|ed)",
        r"suppress(?:es|ed)", r"reduc(?:es|ed)", r"increas(?:es|ed)",
        r"attenuat(?:es|ed)", r"ameliorat(?:es|ed)", r"protect(?:s|ed) against",
    ),
    "PROGNOSTIC_IN": (r"prognostic", r"predict(?:s|ed) survival", r"associated with survival"),
    "PROGRESSES_TO": (r"progress(?:es|ed|ion) to", r"develop(?:s|ed) into", r"evolv(?:es|ed) into"),
    "ENCODES": (r"encod(?:es|ed)",),
    "INTERACTS_WITH": (
        r"interact(?:s|ed|ion) with", r"bind(?:s|ing|bound) to",
        r"cross[- ]?talk", r"cell(?:ular)?[- ]cell communication", r"juxtapos\w*",
    ),
    "PARTICIPATES_IN": (r"participat(?:es|ed) in", r"involved in", r"mediates?", r"\bvia\b"),
    "EXPRESSED_IN": (
        r"express(?:ed|ion|ion level|ion levels|ion of).{0,100}\b(?:in|within)\b",
        r"overexpress(?:ed|ion).{0,100}\b(?:in|within)\b",
        r"localized in", r"present in",
    ),
    "ASSOCIATED_WITH_METABOLITE": (
        r"associated with", r"correlat(?:ed|ion)", r"interact(?:s|ed|ion) with",
    ),
}

INCREASE_TRIGGERS = (
    r"\bincreas(?:e|es|ed|ing)\b", r"\belevat(?:e|es|ed)\b",
    r"\bupregulat(?:e|es|ed)\b", r"\bpromot(?:e|es|ed)\b",
    r"\baccelerat(?:e|es|ed)\b", r"\benhanc(?:e|es|ed)\b",
    r"\bactivat(?:e|es|ed|ion)\b", r"\boverexpress(?:ed|ion)\b",
    r"\binduc(?:e|es|ed|tion)\b", r"\bpositive(?:ly)?\b",
)
DECREASE_TRIGGERS = (
    r"\bdecreas(?:e|es|ed|ing)\b", r"\breduc(?:e|es|ed)\b",
    r"\bdownregulat(?:e|es|ed)\b", r"\bsuppress(?:es|ed)\b",
    r"\binhibit(?:s|ed|ion)\b", r"\battenuat(?:e|es|ed|ion)\b",
    r"\bameliorat(?:e|es|ed|ion)\b", r"\bnegative(?:ly)?\b",
)

SUBJECT_BLOCKING_RE = re.compile(
    r"\b(block(?:s|ed|ing)?|inhibit(?:s|ed|ing|ion)?|knock(?:ed)?\s+down|"
    r"silenc(?:e|ed|ing))\b",
    re.IGNORECASE,
)

LIVER_SCOPE_RE = re.compile(
    r"\b(liver|hepatic|hepatitis|hepatocellular|cirrhosis|cirrhotic|"
    r"cholestasis|cholangitis|biliary|nafld|nash|masld|mash|hbv|hcv)\b",
    re.IGNORECASE,
)
NON_HUMAN_TITLE_RE = re.compile(
    r"\b(?:in mice|in rats|murine|mouse model|rat model)\b",
    re.IGNORECASE,
)
REVIEW_OR_GUIDANCE_TITLE_RE = re.compile(
    r"\b(?:narrative|scoping|umbrella)\s+review\b|\bclinical\s+(?:practice\s+)?guidelines?\b|"
    r"\bpractice\s+guidance\b|\bconsensus\s+(?:statement|recommendations?)\b|"
    r"\bscreening\s+and\s+management\b",
    re.IGNORECASE,
)
COMPOSITE_ENTITY_RE = re.compile(
    r"\b[A-Z0-9][A-Z0-9-]{1,}/[A-Z0-9][A-Z0-9-]{1,}\b"
)
GENERIC_DISEASE_CATEGORY_RE = re.compile(
    r"^(?:(?:(?:chronic|acute)\s+)?(?:liver|hepatic)?\s*diseases?|infections?)$",
    re.IGNORECASE,
)

NON_FACT_SECTIONS = {
    "BACKGROUND": "background_only",
    "INTRODUCTION": "background_only",
    "OBJECTIVE": "objective_only",
    "AIM": "objective_only",
    "AIMS": "objective_only",
    "METHOD": "method_section_only",
    "METHODS": "method_section_only",
    "MATERIALS AND METHODS": "method_section_only",
}
SECTION_MARKER_RE = re.compile(
    r"\b(BACKGROUND|INTRODUCTION|OBJECTIVE|AIMS?|MATERIALS AND METHODS|"
    r"METHODS?|RESULTS?|CONCLUSIONS?|DISCUSSION|RELEVANCE STATEMENT|KEY POINTS):",
    re.IGNORECASE,
)


@dataclass
class PreparedExtraction:
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    raw_entities: list[dict] = field(default_factory=list)
    raw_relations: list[dict] = field(default_factory=list)
    filtered_entities: list[dict] = field(default_factory=list)
    merged_entities: list[dict] = field(default_factory=list)
    mention_to_canonical: dict[str, str] = field(default_factory=dict)
    canonical_key_to_entity: dict[str, dict] = field(default_factory=dict)
    remapped_relations: list[dict] = field(default_factory=list)
    unresolved_relations: list[dict] = field(default_factory=list)
    aliases_by_canonical: dict[str, list[str]] = field(default_factory=dict)


def normalize_surface(value: Any) -> str:
    """Stable Unicode/case/punctuation normalization for equality only."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"[\-_/]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def locate_contiguous(quote: str, text: str) -> tuple[bool, int, int]:
    """Locate a continuous quote while tolerating whitespace/punctuation form."""
    quote = str(quote or "").strip()
    text = str(text or "")
    if not quote or not text or not re.search(r"\w", quote, flags=re.UNICODE):
        return False, -1, -1
    start = text.find(quote)
    if start >= 0:
        return True, start, start + len(quote)
    match = re.search(re.escape(quote), text, flags=re.IGNORECASE)
    if match:
        return True, match.start(), match.end()
    tokens = re.findall(r"\w+", unicodedata.normalize("NFKC", quote), flags=re.UNICODE)
    if not tokens:
        return False, -1, -1
    pattern = r"(?<!\w)" + r"(?:[\W_]+)".join(re.escape(token) for token in tokens) + r"(?!\w)"
    match = re.search(pattern, unicodedata.normalize("NFKC", text), flags=re.IGNORECASE)
    if match:
        return True, match.start(), match.end()
    return False, -1, -1


def _relation_references(mention: str, entity_type: str, relations: list[dict]) -> bool:
    target = normalize_surface(mention)
    for rel in relations:
        for side in ("subject", "object"):
            if normalize_surface(rel.get(side, "")) != target:
                continue
            side_type = str(rel.get(f"{side}_type", "") or "")
            if not side_type or not entity_type or side_type == entity_type:
                return True
    return False


def _cell_type_definition_long_form(mention: str, text: str) -> str:
    """Resolve article-local subtype labels defined with ``termed/called``.

    This is a narrow coreference rule for phrases such as ``a subset of liver
    endothelial cells termed \"Endo4\"``.  It never creates an entity: both the
    subtype label and the specific cell-type long form must occur verbatim in
    the current source.
    """
    mention = str(mention or "").strip()
    if not mention or not text:
        return ""
    cell_head = (
        r"(?:hepatocytes?|kupffer cells?|macrophages?|neutrophils?|"
        r"(?:regulatory )?t cells?|b cells?|stellate cells?|endothelial cells?|"
        r"cholangiocytes?|fibroblasts?|monocytes?|natural killer cells?)"
    )
    pattern = re.compile(
        rf"(?P<long>(?:[A-Za-z0-9+/-]+\s+){{0,3}}{cell_head})\s+"
        rf"(?:termed|called|named)\s+[\"']?{re.escape(mention)}[\"']?",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return ""
    long_form = matches[-1].group("long").strip()
    while True:
        cleaned = re.sub(
            r"^(?:a|an|the|of|subset of)\s+", "", long_form,
            flags=re.IGNORECASE,
        ).strip()
        if cleaned == long_form:
            break
        long_form = cleaned
    return long_form if SPECIFIC_CELL_RE.search(long_form) else ""


def _conditional_process_supported(mention: str, relations: list[dict], text: str) -> bool:
    for rel in relations:
        endpoints = (normalize_surface(rel.get("subject", "")), normalize_surface(rel.get("object", "")))
        if normalize_surface(mention) not in endpoints:
            continue
        evidence = str(rel.get("evidence", "") or "")
        grounded, _, _ = locate_contiguous(evidence, text)
        if not grounded or not locate_contiguous(mention, evidence)[0]:
            continue
        if METHOD_EVIDENCE_RE.search(evidence) and not VALIDATION_RE.search(evidence):
            continue
        predicate = str(rel.get("predicate", "") or "").upper()
        if _has_predicate_trigger(predicate, evidence):
            return True
    return False


def entity_filter_reason(entity: dict, relations: list[dict], text: str) -> tuple[str, str]:
    """Return (status, reason) for a raw entity candidate."""
    mention = str(entity.get("mention", "") or "").strip()
    entity_type = str(entity.get("type", entity.get("entity_type", "")) or "")
    normalized = normalize_surface(mention)
    if not mention:
        return "rejected", "empty_mention"
    if entity_type not in {"Gene", "Protein", "Disease", "Pathway", "Metabolite", "Tissue", "CellType"}:
        return "rejected", "invalid_entity_type"
    mention_tokens = re.findall(r"\w+", mention, flags=re.UNICODE)
    if mention_tokens:
        mention_pattern = r"[\W_]+".join(re.escape(token) for token in mention_tokens)
        explicit_gene = re.search(
            rf"(?<!\w){mention_pattern}(?:[\W_]+)gene\b", text, re.IGNORECASE
        )
        explicit_protein = re.search(
            rf"(?<!\w){mention_pattern}(?:[\W_]+)protein\b", text, re.IGNORECASE
        )
        if entity_type == "Gene" and explicit_protein and not explicit_gene:
            return "rejected", "explicit_protein_as_gene"
        if entity_type == "Protein" and explicit_gene and not explicit_protein:
            return "rejected", "explicit_gene_as_protein"
    if normalized in HARD_REJECT_TERMS:
        return "rejected", "generic_or_context_term"
    if normalized in METHOD_TERMS or any(
        len(term) >= 4 and term in normalized for term in METHOD_TERMS
    ):
        return "rejected", "method_or_database_term"
    if normalized in STATISTICAL_TERMS:
        return "rejected", "statistical_term"
    if COMPOSITE_ENTITY_RE.search(mention):
        return "rejected", "composite_entity"
    if entity_type == "Disease" and GENERIC_DISEASE_CATEGORY_RE.fullmatch(mention.strip()):
        return "rejected", "generic_disease_category"

    if entity_type == "Tissue":
        if any(term in normalized for term in ("microenvironment", "niche")):
            return "rejected", "tissue_context_not_anatomy"
        if normalized in {"tissue", "organ", "site"}:
            return "rejected", "generic_tissue_term"
        if normalized in ANATOMICAL_TERMS or ANATOMICAL_RE.search(normalized):
            return "retained", "anatomical_tissue"
        return "rejected", "tissue_not_anatomical"

    if entity_type == "CellType":
        if normalized in {"cells", "cell", "immune cells", "tumor cells", "cancer cells"}:
            return "rejected", "non_specific_cell_type"
        defined_long_form = _cell_type_definition_long_form(mention, text)
        if not (
            SPECIFIC_CELL_RE.search(mention)
            or SPECIFIC_CELL_SUBSET_RE.search(mention)
            or defined_long_form
        ):
            return "rejected", "cell_type_not_specific"
        if defined_long_form:
            return "retained", "article_local_cell_subtype_alias"

    if entity_type == "Pathway":
        if normalized in {"pathway", "pathways", "signaling", "response", "process", "mechanism"}:
            return "rejected", "generic_pathway_term"

    if entity_type == "Disease" and normalized in (
        CONDITIONAL_PROCESS_TERMS | {"damage", "effect", "symptom", "oxidative stress"}
    ):
        return "rejected", "pathological_process_not_disease"

    if normalized in CONDITIONAL_PROCESS_TERMS:
        if _conditional_process_supported(mention, relations, text):
            return "retained", "conditional_process_with_direct_evidence"
        return "rejected", "bare_process_without_direct_evidence"

    return "retained", "type_constraints_passed"


def _entity_span(entity: dict, text: str) -> tuple[bool, int, int, str]:
    mention = str(entity.get("mention", "") or "")
    start = entity.get("char_start")
    end = entity.get("char_end")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
        span = text[start:end]
        if normalize_surface(span) == normalize_surface(mention):
            return True, start, end, span
    grounded, start, end = locate_contiguous(mention, text)
    return grounded, start, end, text[start:end] if grounded else ""


def prepare_extraction(
    raw_entities: list[dict],
    raw_relations: list[dict],
    text: str = "",
) -> PreparedExtraction:
    """Filter, canonicalize, and remap an article extraction before verification."""
    prepared = PreparedExtraction(
        raw_entities=copy.deepcopy(raw_entities),
        raw_relations=copy.deepcopy(raw_relations),
    )
    endpoint_mentions = {
        (normalize_surface(rel.get(side, "")), str(rel.get(f"{side}_type", "") or ""))
        for rel in raw_relations
        for side in ("subject", "object")
        if rel.get(side)
    }

    retained: list[dict] = []
    filtered_lookup: set[tuple[str, str]] = set()
    for raw in raw_entities:
        entity = copy.deepcopy(raw)
        mention = str(entity.get("mention", "") or "").strip()
        entity_type = str(entity.get("type", entity.get("entity_type", "")) or "")
        grounded, start, end, source_span = _entity_span(entity, text)
        status, reason = entity_filter_reason(entity, raw_relations, text)
        if text and not grounded:
            status, reason = "rejected", "mention_not_in_source"
        entity.update({
            "mention": mention,
            "type": entity_type,
            "entity_type": entity_type,
            "source_span": source_span,
            "char_start": start,
            "char_end": end,
            "grounded": grounded,
            "filter_status": status,
            "filter_reason": reason,
            "was_relation_endpoint": (
                (normalize_surface(mention), entity_type) in endpoint_mentions
                or (normalize_surface(mention), "") in endpoint_mentions
            ),
        })
        if status == "rejected":
            prepared.filtered_entities.append(entity)
            filtered_lookup.add((normalize_surface(mention), entity_type))
            continue
        retained.append(entity)

    abbr_map = AbbreviationDetector().detect(text) if text else None
    explicit_abbr = dict(abbr_map.abbr_to_long) if abbr_map else {}
    for entity in retained:
        if entity.get("type") != "CellType":
            continue
        long_form = _cell_type_definition_long_form(entity.get("mention", ""), text)
        if long_form:
            explicit_abbr[str(entity.get("mention", ""))] = long_form
    groups: dict[int, list[dict]] = {}
    group_order: list[int] = []
    key_to_group: dict[tuple[str, str], int] = {}
    for entity in retained:
        mention = entity["mention"]
        entity_type = entity["type"]
        attrs = entity.get("attributes", {}) or {}
        normalized_id = str(attrs.get("normalized_id", entity.get("normalized_id", "")) or "").strip()
        long_form = explicit_abbr.get(mention, "")
        if long_form and not locate_contiguous(long_form, text)[0]:
            long_form = ""
        canonical_surface = long_form or mention
        keys = [(entity_type, f"name:{normalize_surface(canonical_surface)}")]
        if normalized_id:
            keys.append((entity_type, f"id:{normalized_id.casefold()}"))
        matched_groups = list(dict.fromkeys(
            key_to_group[key] for key in keys if key in key_to_group
        ))
        if not matched_groups:
            group_key = max(groups, default=-1) + 1
            groups[group_key] = []
            group_order.append(group_key)
        else:
            group_key = matched_groups[0]
            for extra_group in matched_groups[1:]:
                groups[group_key].extend(groups.pop(extra_group, []))
                group_order.remove(extra_group)
                for known_key, known_group in list(key_to_group.items()):
                    if known_group == extra_group:
                        key_to_group[known_key] = group_key
        for key in keys:
            key_to_group[key] = group_key
        entity["_canonical_surface"] = canonical_surface
        groups[group_key].append(entity)

    canonical_lookup: dict[tuple[str, str], str] = {}
    for group_key in group_order:
        group = groups[group_key]
        canonical_surface = max(
            (str(item.get("_canonical_surface", item["mention"])) for item in group),
            key=lambda value: (len(value.split()), len(value)),
        )
        winner = max(
            group,
            key=lambda item: (
                normalize_surface(item["mention"]) == normalize_surface(canonical_surface),
                bool(item.get("grounded")),
                len(item["mention"]),
            ),
        )
        canonical = copy.deepcopy(winner)
        if normalize_surface(canonical["mention"]) != normalize_surface(canonical_surface):
            grounded, start, end = locate_contiguous(canonical_surface, text)
            canonical.update({
                "mention": canonical_surface,
                "source_span": text[start:end] if grounded else "",
                "char_start": start,
                "char_end": end,
                "grounded": grounded,
            })
        canonical_key = f"{canonical['type']}:{normalize_surface(canonical['mention'])}"
        aliases = list(dict.fromkeys(item["mention"] for item in group))
        canonical.update({
            "canonical_key": canonical_key,
            "canonical_mentions": aliases,
            "filter_status": "retained",
            "filter_reason": canonical.get("filter_reason", "type_constraints_passed"),
        })
        canonical.pop("_canonical_surface", None)
        prepared.entities.append(canonical)
        prepared.canonical_key_to_entity[canonical_key] = copy.deepcopy(canonical)
        # Gene and Protein candidates may share a surface form but must remain
        # distinct entities.  Union their textual aliases for evidence lookup
        # so that one type cannot overwrite an explicit article abbreviation.
        prepared.aliases_by_canonical[canonical["mention"]] = list(dict.fromkeys([
            *prepared.aliases_by_canonical.get(canonical["mention"], []),
            *aliases,
        ]))
        for item in group:
            original = item["mention"]
            prepared.mention_to_canonical[original] = canonical["mention"]
            canonical_lookup[(normalize_surface(original), item["type"])] = canonical["mention"]
            canonical_lookup[(normalize_surface(canonical["mention"]), item["type"])] = canonical["mention"]
            if item is not winner or normalize_surface(original) != normalize_surface(canonical["mention"]):
                prepared.merged_entities.append({
                    "mention": original,
                    "entity_type": item["type"],
                    "canonical_mention": canonical["mention"],
                    "canonical_key": canonical_key,
                    "merge_reason": (
                        "explicit_article_abbreviation"
                        if explicit_abbr.get(original)
                        else "same_normalized_id_or_surface"
                    ),
                })

    for raw in raw_relations:
        rel = copy.deepcopy(raw)
        rel.setdefault("quality_flags", [])
        flags = set(rel.get("quality_flags", []))
        changed = False
        unresolved = False
        for side in ("subject", "object"):
            original = str(rel.get(side, "") or "").strip()
            entity_type = str(rel.get(f"{side}_type", "") or "")
            rel[f"original_{side}"] = original
            key = (normalize_surface(original), entity_type)
            canonical = canonical_lookup.get(key)
            if not canonical and (key in filtered_lookup or any(
                item_key[0] == key[0] and (not entity_type or item_key[1] == entity_type)
                for item_key in filtered_lookup
            )):
                flags.add("filtered_endpoint")
                flags.add(f"{side}_filtered")
                unresolved = True
            elif canonical:
                rel[side] = canonical
                changed = changed or canonical != original
            else:
                flags.add("unresolved_endpoint")
                flags.add(f"{side}_not_in_entities")
                unresolved = True
        rel["quality_flags"] = sorted(flags)
        rel["endpoint_remapped"] = changed
        prepared.relations.append(rel)
        if changed:
            prepared.remapped_relations.append(copy.deepcopy(rel))
        if unresolved:
            prepared.unresolved_relations.append(copy.deepcopy(rel))

    return prepared


def _has_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _has_predicate_trigger(predicate: str, evidence: str) -> bool:
    return _has_any(PREDICATE_TRIGGERS.get(predicate, ()), evidence)


def article_quality_flags(text: str) -> set[str]:
    """Return conservative article-level write blockers for PubMed records.

    The gate is applied only to the agent's explicit TITLE/ABSTRACT envelope,
    so standalone verifier unit tests and other callers keep their old scope.
    """
    title_match = re.match(r"\s*TITLE:\s*(.*?)\nABSTRACT:", text, re.DOTALL | re.IGNORECASE)
    if not title_match:
        return set()
    title = title_match.group(1).strip()
    flags: set[str] = set()
    if not LIVER_SCOPE_RE.search(title):
        flags.add("article_out_of_scope")
    if NON_HUMAN_TITLE_RE.search(title):
        flags.add("non_human_article")
    if REVIEW_OR_GUIDANCE_TITLE_RE.search(title):
        flags.add("review_article")
    return flags


def _evidence_section_flags(text: str, evidence_start: int) -> set[str]:
    """Classify a grounded quote by its explicit source section."""
    if evidence_start < 0:
        return set()
    abstract_match = re.search(r"\nABSTRACT:\s*", text, re.IGNORECASE)
    if not abstract_match:
        return set()
    if evidence_start < abstract_match.end():
        return {"title_only"}
    current_section = ""
    for match in SECTION_MARKER_RE.finditer(text, abstract_match.end(), evidence_start + 1):
        current_section = match.group(1).upper()
    flag = NON_FACT_SECTIONS.get(current_section)
    return {flag} if flag else set()


def _alias_spans(evidence: str, aliases: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for alias in aliases:
        tokens = re.findall(r"\w+", str(alias or ""), flags=re.UNICODE)
        if not tokens:
            continue
        pattern = r"(?<!\w)" + r"(?:[\W_]+)".join(
            re.escape(token) for token in tokens
        ) + r"(?!\w)"
        spans.extend((match.start(), match.end()) for match in re.finditer(
            pattern, evidence, flags=re.IGNORECASE
        ))
    return spans


def _predicate_trigger_links_endpoints(
    predicate: str,
    evidence: str,
    subject_aliases: list[str],
    object_aliases: list[str],
) -> bool:
    """Require a predicate trigger to occur between the two endpoint mentions.

    This prevents an unrelated phrase such as ``HCC associated with HBV/HCV``
    from validating a proposed ``immune surveillance ASSOCIATED_WITH HBV/HCV``
    edge merely because all words occur somewhere in the same sentence.
    """
    subject_spans = _alias_spans(evidence, subject_aliases)
    object_spans = _alias_spans(evidence, object_aliases)
    if not subject_spans or not object_spans:
        return False
    trigger_spans = [
        (match.start(), match.end())
        for pattern in PREDICATE_TRIGGERS.get(predicate, ())
        for match in re.finditer(pattern, evidence, flags=re.IGNORECASE)
    ]
    for subject_span in subject_spans:
        for object_span in object_spans:
            low = min(sum(subject_span) / 2, sum(object_span) / 2)
            high = max(sum(subject_span) / 2, sum(object_span) / 2)
            if any(trigger_end >= low and trigger_start <= high for trigger_start, trigger_end in trigger_spans):
                return True
    return False


def predicate_trigger_links_endpoints(
    predicate: str,
    evidence: str,
    subject_aliases: list[str],
    object_aliases: list[str],
) -> bool:
    """Public, read-only wrapper used by the agent's candidate lattice."""
    return _predicate_trigger_links_endpoints(
        predicate, evidence, subject_aliases, object_aliases
    )


def _subject_is_negatively_perturbed(evidence: str, aliases: list[str]) -> bool:
    """Detect an explicit block/knockdown of the relation subject.

    A sentence such as "blocking X reduces Y" supports a positive X→Y
    direction, not a negative one.  This small intervention rule prevents the
    literal word "reduces" from reversing that semantics.
    """
    for alias in aliases:
        tokens = re.findall(r"\w+", alias, flags=re.UNICODE)
        if not tokens:
            continue
        alias_pattern = r"[\W_]+".join(re.escape(token) for token in tokens)
        before = rf"{SUBJECT_BLOCKING_RE.pattern}.{{0,30}}{alias_pattern}"
        after = rf"{alias_pattern}.{{0,50}}{SUBJECT_BLOCKING_RE.pattern}"
        if re.search(before, evidence, flags=re.IGNORECASE) or re.search(
            after, evidence, flags=re.IGNORECASE
        ):
            return True
    return False


def _endpoint_support_scope(
    evidence: str,
    subject_aliases: list[str],
    object_aliases: list[str],
) -> str:
    """Return the smallest local text scope that contains both endpoints."""
    subject_spans = _alias_spans(evidence, subject_aliases)
    object_spans = _alias_spans(evidence, object_aliases)
    if not subject_spans or not object_spans:
        return evidence
    best: tuple[int, int] | None = None
    for subject_span in subject_spans:
        for object_span in object_spans:
            start = min(subject_span[0], object_span[0])
            end = max(subject_span[1], object_span[1])
            if best is None or end - start < best[1] - best[0]:
                best = (start, end)
    if best is None:
        return evidence
    sentence_start = evidence.rfind(".", 0, best[0]) + 1
    sentence_start = max(sentence_start, evidence.rfind(";", 0, best[0]) + 1)
    sentence_end_candidates = [
        index for index in (
            evidence.find(".", best[1]),
            evidence.find(";", best[1]),
            evidence.find("\n", best[1]),
        ) if index >= 0
    ]
    sentence_end = min(sentence_end_candidates) if sentence_end_candidates else len(evidence)
    clause = evidence[sentence_start:sentence_end].strip()
    return clause or evidence


def evaluate_relation_evidence(
    relation: dict,
    text: str,
    aliases_by_canonical: dict[str, list[str]],
) -> dict[str, Any]:
    """Return deterministic evidence annotations and blocking quality flags."""
    evidence = str(relation.get("evidence", "") or "").strip()
    predicate = str(relation.get("predicate", "") or "").upper()
    direction = str(relation.get("direction", "unknown") or "unknown").lower()
    flags = set(relation.get("quality_flags", []))
    grounded, start, end = locate_contiguous(evidence, text)
    if not evidence or not re.search(r"\w", evidence, flags=re.UNICODE):
        flags.add("empty_evidence")
    elif not grounded:
        flags.add("evidence_not_contiguous")

    endpoint_grounded: dict[str, bool] = {}
    for side in ("subject", "object"):
        endpoint = str(relation.get(side, "") or "")
        aliases = aliases_by_canonical.get(endpoint, [endpoint])
        aliases = list(dict.fromkeys([endpoint, *aliases]))
        present = any(locate_contiguous(alias, evidence)[0] for alias in aliases if alias)
        endpoint_grounded[side] = present
        if not present:
            flags.add(f"{side}_not_grounded")
    if not all(endpoint_grounded.values()):
        flags.add("endpoint_not_in_evidence")

    subject_aliases = list(dict.fromkeys([
        str(relation.get("subject", "") or ""),
        *aliases_by_canonical.get(str(relation.get("subject", "") or ""), []),
    ]))
    object_aliases = list(dict.fromkeys([
        str(relation.get("object", "") or ""),
        *aliases_by_canonical.get(str(relation.get("object", "") or ""), []),
    ]))
    support_scope = _endpoint_support_scope(evidence, subject_aliases, object_aliases)
    raw_trigger_present = _has_predicate_trigger(predicate, evidence)
    trigger_present = raw_trigger_present and _predicate_trigger_links_endpoints(
        predicate, evidence, subject_aliases, object_aliases
    )
    if not trigger_present:
        flags.add("trigger_missing")
        if raw_trigger_present:
            flags.add("trigger_not_linking_endpoints")
    increase_present = _has_any(INCREASE_TRIGGERS, evidence)
    decrease_present = _has_any(DECREASE_TRIGGERS, evidence)
    intervention_inversion = (
        decrease_present
        and _subject_is_negatively_perturbed(evidence, subject_aliases)
    )
    if intervention_inversion:
        increase_present, decrease_present = True, False
    direction_consistent = True
    if direction == "increase":
        direction_consistent = increase_present and not decrease_present
    elif direction == "decrease":
        direction_consistent = decrease_present and not increase_present
    elif direction == "positive" and decrease_present and not increase_present:
        direction_consistent = False
    elif direction == "negative" and increase_present and not decrease_present:
        direction_consistent = False
    if not direction_consistent:
        flags.add("trigger_direction_mismatch")

    method_signal = bool(METHOD_EVIDENCE_RE.search(support_scope))
    method_only = bool(
        method_signal
        and not VALIDATION_RE.search(support_scope)
        and (METHOD_ONLY_STRONG_RE.search(support_scope) or not trigger_present)
    )
    prediction_only = bool(
        PREDICTION_RE.search(support_scope)
        and not VALIDATION_RE.search(support_scope)
    )
    negation_text = re.sub(r"\bnot\s+only\b", "notonly", support_scope, flags=re.IGNORECASE)
    negated = bool(relation.get("negated")) or bool(NEGATION_RE.search(negation_text))
    uncertain = bool(relation.get("uncertain")) or prediction_only
    if method_only:
        flags.add("method_only")
    if prediction_only:
        flags.add("prediction_only")
    if negated:
        flags.add("negated")
        flags.add("scoped_negation")
    if uncertain:
        flags.add("uncertain")
    if grounded:
        flags.update(_evidence_section_flags(text, start))

    sentence_parts = [part for part in re.split(r"(?<=[.!?])\s+|\n+", evidence) if part.strip()]
    cross_sentence = len(sentence_parts) > 1
    if cross_sentence:
        flags.add("cross_sentence")

    level = 3
    basic_strong = (
        grounded and all(endpoint_grounded.values()) and trigger_present
        and direction_consistent and not method_only and not prediction_only
        and not negated and not uncertain
        and not ({"filtered_endpoint", "unresolved_endpoint"} & flags)
    )
    if basic_strong and not cross_sentence:
        level = 1
    elif basic_strong and cross_sentence and len(sentence_parts) <= 2:
        level = 2
    if level == 3:
        flags.add("weak_evidence")

    return {
        "evidence": evidence,
        "evidence_char_start": start,
        "evidence_char_end": end,
        "evidence_contiguous": grounded,
        "evidence_level": level,
        "subject_grounded_in_evidence": endpoint_grounded["subject"],
        "object_grounded_in_evidence": endpoint_grounded["object"],
        "trigger_present": trigger_present,
        "direction_trigger_consistent": direction_consistent,
        "negated": negated,
        "uncertain": uncertain,
        "quality_flags": sorted(flags),
    }


def ratio_metric(count: int, denominator: int) -> dict[str, float | int | None]:
    return {
        "count": count,
        "denominator": denominator,
        "value": round(count / denominator, 4) if denominator else None,
    }
