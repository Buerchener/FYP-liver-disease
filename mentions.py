"""Adapt LiverKG extraction output for evidence review and entity conceptualization."""
from collections import defaultdict
import hashlib
import json
import re


def read_articles(content):
    articles = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    if not articles:
        raise ValueError("The input must contain at least one title and abstract.")
    pmids = []
    for row in articles:
        if not isinstance(row, dict):
            raise ValueError("Each JSONL line must contain an article object.")
        pmid = str(row.get("pmid", ""))
        if not pmid.isascii() or not pmid.isdecimal():
            raise ValueError(f"Invalid PMID: {pmid!r}")
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("title", "abstract")):
            raise ValueError(f"PMID {pmid} needs a title and abstract.")
        pmids.append(pmid)
    if len(set(pmids)) != len(pmids):
        raise ValueError("Each PMID must occur only once.")
    return articles


def span_status(item, text):
    start, end = item["char_start"], item["char_end"]
    if start is None and end is None:
        return "missing"
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
        if text[start:end] == item["source_span"]:
            return "exact"
    return "mismatch"


def contexts(record, article, text):
    sections = [{"id": "title", "section": "TITLE", "start": len("TITLE: "),
                 "end": len("TITLE: " + article["title"])}]
    by_sentence = defaultdict(list)
    for unit in record["phases"]["reader"]["units"]:
        if unit["text"] != text[unit["char_start"]:unit["char_end"]]:
            raise ValueError(f"Reader offset mismatch for {record['pmid']} {unit['unit_id']}")
        by_sentence[unit["parent_sentence_id"]].append(unit)
    for sentence_id, units in by_sentence.items():
        sections.append({"id": sentence_id, "section": units[0]["section"],
                         "start": min(unit["char_start"] for unit in units),
                         "end": max(unit["char_end"] for unit in units)})
    return [dict(section, text=text[section["start"]:section["end"]]) for section in sections]


def supplementary_contexts(mention, sections, primary):
    term = re.escape(mention)
    word = re.compile(r"(?<!\w)" + term + r"(?!\w)", re.I)
    definition = re.compile(r"(?:\(\s*" + term + r"\s*\)|(?<!\w)" + term
                            + r"\s*\([\w-]{2,14}\))", re.I)
    matches = [s for s in sections if word.search(s["text"])]
    # Supply an explicit abbreviation occurrence first. Keep evidence from an
    # alternative occurrence separate from the model's original coordinates.
    extra = []
    for s in sorted(matches, key=lambda s: (not definition.search(s["text"]), s["section"] == "TITLE")):
        if primary is not None and s["id"] == primary["id"]:
            continue
        source = "parenthetical_mention" if definition.search(s["text"]) else "other_occurrence"
        extra.append(dict(s, source=source))
        if len(extra) == 2:
            break
    return extra


def article_mentions(record, article, coverage=()):
    pmid = str(record["pmid"])
    text = "TITLE: " + article["title"] + "\nABSTRACT: " + article["abstract"]
    sections = contexts(record, article, text)
    extraction = record["phases"]["extraction"]
    verification = record["phases"].get("verification")
    review = verification["review"] if verification is not None else None
    if review is not None and extraction["entities"] != review["raw_entities"]:
        raise ValueError(f"Extraction and verification inputs disagree for PMID {pmid}")
    candidates = []
    verified_entities = verification["entities"] if verification is not None else []
    filtered_entities = review["filtered_entities"] if review is not None else []
    for i, item in enumerate(verified_entities, 1):
        candidate = {key: item[key] for key in ("mention", "type", "canonical_mentions",
                      "canonical_key", "normalized_id", "char_start", "char_end", "source_span")}
        candidates.append(dict(candidate, id=f"{pmid}:V{i:02d}", pmid=pmid,
                               offset_status=span_status(item, text)))
    mentions = []
    by_span = defaultdict(list)
    primary_entities = extraction["entities"]
    for i, item in enumerate([*primary_entities, *coverage], 1):
        mention, kind = item["mention"], item["type"]
        added = i > len(primary_entities)
        start, end = item["char_start"], item["char_end"]
        status = span_status(item, text)
        primary = next((s for s in sections if status == "exact"
                        and s["start"] <= start < end <= s["end"]), None)
        if status == "exact" and primary is None:
            raise ValueError(f"Located mention lacks reader context at {pmid} {start}:{end}")
        if added and (status != "exact" or primary["id"] != item["source_sentence_id"]
                      or item["source_span"] != mention):
            raise ValueError(f"Coverage mention failed provenance validation for PMID {pmid}")
        rejected = [] if added else [v["filter_reason"] for v in filtered_entities
                    if all(v[k] == item[k] for k in ("mention", "type", "char_start", "char_end"))]
        linked = [] if rejected or added else [v for v in candidates if v["type"] == kind and
                                      mention in [v["mention"], *v["canonical_mentions"]]]
        representative = any(v["char_start"] == start and v["char_end"] == end
                             and status == "exact" for v in linked)
        if added:
            upstream = "ontology_coverage"
        elif review is None:
            upstream = "not_run"
        elif rejected:
            upstream = "filtered"
        elif not linked:
            upstream = "unresolved_link"
        elif representative:
            upstream = "retained_span"
        else:
            upstream = "linked_to_other_span"
        row = {"id": f"{pmid}:M{i:02d}", "pmid": pmid, "mention": mention,
               "raw_type": kind, "source_span": item["source_span"], "char_start": start,
               "char_end": end, "offset_status": status, "raw_grounded": item["grounded"],
               "raw_alignment_status": item["alignment_status"],
               "context": primary["text"] if primary else "",
               "context_id": primary["id"] if primary else None,
               "context_start": primary["start"] if primary else None,
               "context_end": primary["end"] if primary else None,
               "evidence_contexts": supplementary_contexts(mention, sections, primary),
               "upstream_status": upstream, "upstream_filter_reasons": rejected,
               "verification_ids": [v["id"] for v in linked], "same_span_conflicts": []}
        mentions.append(row)
        if status == "exact":
            by_span[(start, end)].append(row)
    for group in by_span.values():
        for row in group:
            row["same_span_conflicts"] = [v["id"] for v in group if v["raw_type"] != row["raw_type"]]
    return mentions, candidates


def conceptualization_input(inventory):
    """Keep one sentence table per article and link each mention to its evidence."""
    articles = {}
    for mention in inventory["mentions"]:
        pmid = mention["pmid"]
        article = articles.setdefault(pmid, {"pmid": pmid, "sentences": {}, "mentions": []})
        row = {field: mention[field] for field in (
            "id", "mention", "raw_type", "context_id", "char_start", "char_end", "offset_status",
        )}
        if row["offset_status"] == "exact":
            sentences = article["sentences"]
            sentences[row["context_id"]] = mention["context"]
            supporting = mention["evidence_contexts"]
            sentences.update({sentence["id"]: sentence["text"] for sentence in supporting})
            row["supporting_sentence_ids"] = [sentence["id"] for sentence in supporting]
            row["same_span_conflicts"] = mention["same_span_conflicts"]
        article["mentions"].append(row)
    return {
        "sources": inventory["sources"],
        "coordinate_convention": inventory["coordinate_convention"],
        "articles": list(articles.values()),
    }


def build(source, result_path, *, coverage_path=None):
    article_path = source / "articles.jsonl"
    records = json.loads(result_path.read_text())["records"]
    articles = {str(article["pmid"]): article for article in read_articles(article_path.read_bytes())}
    if len(records) != len(articles) or {str(r["pmid"]) for r in records} != set(articles):
        raise ValueError("Result PMIDs do not match source PMIDs")
    source_paths = [article_path, result_path]
    coverage_by_pmid = {}
    if coverage_path is not None:
        coverage = json.loads(coverage_path.read_text())
        if (coverage["articles_sha256"] != hashlib.sha256(article_path.read_bytes()).hexdigest()
                or coverage["extraction_sha256"] != hashlib.sha256(result_path.read_bytes()).hexdigest()):
            raise ValueError("Coverage was prepared for a different source or extraction.")
        coverage_rows = coverage["records"]
        coverage_by_pmid = {str(row["pmid"]): row["additions"] for row in coverage_rows}
        if len(coverage_rows) != len(articles) or coverage_by_pmid.keys() != articles.keys():
            raise ValueError("Coverage does not cover the corresponding PMIDs exactly once.")
        source_paths.append(coverage_path)
    mentions, candidates = [], []
    for record in records:
        pmid = str(record["pmid"])
        rows, reviewed = article_mentions(record, articles[pmid], coverage_by_pmid.get(pmid, ()))
        mentions.extend(rows)
        candidates.extend(reviewed)
    return {"sources": [{"filename": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in source_paths],
            "coordinate_convention": "Unicode [start,end) in TITLE: {title}\\nABSTRACT: {abstract}",
            "pmids": list(articles), "mentions": mentions, "verification_candidates": candidates}
