#!/usr/bin/env python3
"""Crawl and tier DisGeNET GDA records for the liver disease backbone."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENDPOINT = "https://api.disgenet.com/api/v1/gda/summary"
BACKBONE = [
    {"stage_code": "NAFLD", "stage_order": 1, "umls_id": "C0400966"},
    {"stage_code": "NASH", "stage_order": 2, "umls_id": "C3241937"},
    {"stage_code": "Fibrosis", "stage_order": 3, "umls_id": "C0239946"},
    {"stage_code": "Cirrhosis", "stage_order": 4, "umls_id": "C0023890"},
    {"stage_code": "HCC", "stage_order": 5, "umls_id": "C2239176"},
]
BACKBONE_BY_CUI = {item["umls_id"]: item for item in BACKBONE}
REQUIRED_FIELDS = ["diseaseUMLSCUI", "geneNcbiID", "symbolOfGene", "assocID", "score", "normalized_score"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_cui(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text.removeprefix("UMLS:").removeprefix("UMLS_")


def disease_param(cui: str) -> str:
    return f"UMLS_{normalize_cui(cui)}"


def request_page(api_key: str, cui: str, page_number: int, retries: int, sleep_seconds: float) -> dict[str, Any]:
    params = {"disease": disease_param(cui), "page_number": str(page_number)}
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"Authorization": api_key, "accept": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            retry_after = exc.headers.get("X-Rate-Limit-Retry-After-Seconds")
            if exc.code == 429 and attempt < retries:
                wait = float(retry_after or max(sleep_seconds, 1.0))
                time.sleep(wait)
                continue
            raise RuntimeError(f"DisGeNET API error {exc.code} for {cui} page {page_number}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(max(sleep_seconds, 1.0))
                continue
            raise RuntimeError(f"DisGeNET API request failed for {cui} page {page_number}: {exc}") from exc
    raise RuntimeError(f"DisGeNET API request failed for {cui} page {page_number}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def json_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def list_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if not isinstance(value, list):
        value = [value]
    return ";".join(str(item) for item in value)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        if not value:
            return "array[0]"
        first = value[0]
        if isinstance(first, dict):
            return f"array[object:{len(value)}]"
        return f"array[{type(first).__name__}:{len(value)}]"
    return type(value).__name__


def missing_required(row: dict[str, Any]) -> list[str]:
    return [field for field in REQUIRED_FIELDS if row.get(field) in (None, "", [], {})]


def tier_for_row(row: dict[str, Any]) -> tuple[str, str]:
    missing = missing_required(row)
    cui = normalize_cui(row.get("diseaseUMLSCUI"))
    if missing:
        return "rejected", "missing_required:" + ",".join(missing)
    if cui not in BACKBONE_BY_CUI:
        return "rejected", f"non_backbone_disease:{cui}"
    score = to_float(row.get("score"))
    num_pmids = to_int(row.get("numPMIDs")) or 0
    if score is not None and score >= 0.6 and num_pmids >= 5:
        return "high_confidence", "score>=0.6_and_numPMIDs>=5"
    if score is not None and score >= 0.3:
        return "medium_confidence", "score>=0.3"
    return "candidate", "score<0.3_or_weak_evidence"


def association_row(row: dict[str, Any], stage: dict[str, Any], retrieved_at: str, tier: str, tier_reason: str) -> dict[str, Any]:
    gene_id = str(row.get("geneNcbiID") or "")
    disease_id = f"UMLS:{stage['umls_id']}"
    assoc_id = str(row.get("assocID") or "")
    return {
        "relation_id": f"DISGENET_ASSOC:{assoc_id}" if assoc_id else "",
        "gene_id": gene_id,
        "gene_symbol": row.get("symbolOfGene") or "",
        "disease_id": disease_id,
        "disease_umls_id": stage["umls_id"],
        "stage_code": stage["stage_code"],
        "stage_order": stage["stage_order"],
        "assoc_id": assoc_id,
        "score": row.get("score", ""),
        "normalized_score": row.get("normalized_score", ""),
        "num_pmids": row.get("numPMIDs", ""),
        "year_initial": row.get("yearInitial", ""),
        "year_final": row.get("yearFinal", ""),
        "evidence_index": row.get("ei", ""),
        "disgenet_evidence_level": row.get("el", ""),
        "score_breakdown": json_text(row.get("scoreBreakdown")),
        "num_db_snp": row.get("numDBSNPsupportingAssociation", ""),
        "num_clinical_trials": row.get("numCTsupportingAssociation", ""),
        "num_chemicals": row.get("numChemsIncludedInEvidences", ""),
        "num_pmids_with_chemicals": row.get("numPMIDSWithChemsIncludedInEvidences", ""),
        "num_trials_with_chemicals": row.get("numNCTSWithChemsIncludedInEvidences", ""),
        "chemical_evidence": json_text(row.get("chemsIncludedInEvidenceBySource")),
        "source": "DisGeNET",
        "tier": tier,
        "tier_reason": tier_reason,
        "api_endpoint": ENDPOINT,
        "api_query": f"disease={disease_param(stage['umls_id'])}",
        "retrieved_at": retrieved_at,
    }


def disease_row(rows: list[dict[str, Any]], stage: dict[str, Any]) -> dict[str, Any]:
    first = rows[0] if rows else {}
    return {
        "stage_code": stage["stage_code"],
        "stage_order": stage["stage_order"],
        "disease_id": f"UMLS:{stage['umls_id']}",
        "umls_id": stage["umls_id"],
        "disease_name": first.get("diseaseName") or stage["stage_code"],
        "disease_type": first.get("diseaseType") or "",
        "disease_vocabularies": list_text(first.get("diseaseVocabularies")),
        "disease_classes_msh": list_text(first.get("diseaseClasses_MSH")),
        "disease_classes_umls_st": list_text(first.get("diseaseClasses_UMLS_ST")),
        "disease_classes_do": list_text(first.get("diseaseClasses_DO")),
        "disease_classes_hpo": list_text(first.get("diseaseClasses_HPO")),
        "source": "DisGeNET",
    }


def gene_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "gene_id": str(row.get("geneNcbiID") or ""),
        "gene_symbol": row.get("symbolOfGene") or "",
        "gene_ensembl_ids": list_text(row.get("geneEnsemblIDs")),
        "gene_ncbi_type": row.get("geneNcbiType") or "",
        "gene_protein_str_ids": list_text(row.get("geneProteinStrIDs")),
        "gene_dsi": row.get("geneDSI", ""),
        "gene_dpi": row.get("geneDPI", ""),
        "gene_pli": row.get("genepLI", ""),
        "gene_protein_class_ids": list_text(row.get("geneProteinClassIDs")),
        "gene_protein_class_names": list_text(row.get("geneProteinClassNames")),
        "source": "DisGeNET",
    }


def field_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for row in rows:
        for key, value in row.items():
            item = profile.setdefault(key, {"present": 0, "empty": 0, "types": Counter()})
            item["present"] += 1
            if value in (None, "", [], {}):
                item["empty"] += 1
            item["types"][type_name(value)] += 1
    return {
        key: {"present": value["present"], "empty": value["empty"], "types": dict(value["types"])}
        for key, value in sorted(profile.items())
    }


def crawl(api_key: str, output_dir: Path, sleep_seconds: float, retries: int, max_pages: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_dir = output_dir / "raw_pages"
    retrieved_at = now_iso()
    all_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "endpoint": ENDPOINT,
        "retrieved_at": retrieved_at,
        "source_access_note": "Academic role warning may restrict results to curated DisGeNET sources.",
        "diseases": [],
    }
    for stage in BACKBONE:
        disease_rows: list[dict[str, Any]] = []
        page_number = 0
        total_elements: int | None = None
        page_meta: list[dict[str, Any]] = []
        while True:
            data = request_page(api_key, stage["umls_id"], page_number, retries, sleep_seconds)
            payload = data.get("payload") or []
            page_payload = {
                "status": data.get("status"),
                "paging": data.get("paging"),
                "warnings": data.get("warnings"),
                "requestpar": data.get("requestpar"),
                "payload": payload,
                "httpStatus": data.get("httpStatus"),
            }
            write_json(raw_dir / stage["stage_code"] / f"page_{page_number:04d}.json", page_payload)
            paging = data.get("paging") or {}
            total_elements = int(paging.get("totalElements") or len(payload))
            disease_rows.extend(payload)
            page_meta.append(
                {
                    "page_number": page_number,
                    "rows": len(payload),
                    "paging": paging,
                    "warnings": data.get("warnings") or [],
                    "status": data.get("status"),
                }
            )
            if not payload:
                break
            if len(disease_rows) >= total_elements:
                break
            page_number += 1
            if max_pages is not None and page_number >= max_pages:
                break
            time.sleep(sleep_seconds)
        for row in disease_rows:
            row["_stage_code"] = stage["stage_code"]
            row["_stage_order"] = stage["stage_order"]
            row["_expected_umls_id"] = stage["umls_id"]
            row["_retrieved_at"] = retrieved_at
        all_rows.extend(disease_rows)
        metadata["diseases"].append(
            {
                "stage_code": stage["stage_code"],
                "stage_order": stage["stage_order"],
                "umls_id": stage["umls_id"],
                "total_elements": total_elements,
                "payload_rows": len(disease_rows),
                "pages": page_meta,
            }
        )
    return all_rows, metadata


def build_outputs(rows: list[dict[str, Any]], output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    staging_dir = output_dir / "staging"
    curated_dir = output_dir / "curated"
    tiers_dir = output_dir / "tiers"
    report_path = Path("reports/disgenet_liver_backbone_crawl_report.md")

    rows_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_stage[str(row.get("_stage_code"))].append(row)

    disease_rows = [disease_row(rows_by_stage[stage["stage_code"]], stage) for stage in BACKBONE]
    genes_by_id: dict[str, dict[str, Any]] = {}
    associations_by_id: dict[str, dict[str, Any]] = {}
    tier_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ["high_confidence", "medium_confidence", "candidate", "rejected"]}
    rejected_raw: list[dict[str, Any]] = []
    missing_ensembl: set[tuple[str, str]] = set()
    missing_protein: set[tuple[str, str]] = set()

    for raw in rows:
        stage = BACKBONE_BY_CUI.get(str(raw.get("_expected_umls_id"))) or BACKBONE_BY_CUI.get(normalize_cui(raw.get("diseaseUMLSCUI")))
        if not stage:
            stage = {"stage_code": raw.get("_stage_code") or "", "stage_order": raw.get("_stage_order") or "", "umls_id": normalize_cui(raw.get("diseaseUMLSCUI"))}
        tier, reason = tier_for_row(raw)
        assoc = association_row(raw, stage, str(raw.get("_retrieved_at") or metadata.get("retrieved_at")), tier, reason)
        if assoc["relation_id"]:
            associations_by_id[assoc["relation_id"]] = assoc
        tier_rows[tier].append(assoc)
        if tier == "rejected":
            rejected_raw.append(raw)
            continue
        gene = gene_row(raw)
        if gene["gene_id"]:
            genes_by_id[gene["gene_id"]] = gene
            if not gene["gene_ensembl_ids"]:
                missing_ensembl.add((gene["gene_id"], gene["gene_symbol"]))
            if not gene["gene_protein_str_ids"]:
                missing_protein.add((gene["gene_id"], gene["gene_symbol"]))

    gene_rows = sorted(genes_by_id.values(), key=lambda item: (item["gene_symbol"], item["gene_id"]))
    association_rows = sorted(associations_by_id.values(), key=lambda item: item["relation_id"])
    import_rows = [row for row in association_rows if row["tier"] in {"high_confidence", "medium_confidence"}]

    write_jsonl(staging_dir / "gda_raw_merged.jsonl", rows)
    write_json(staging_dir / "field_profile.json", field_profile(rows))
    write_json(staging_dir / "crawl_metadata.json", metadata)
    write_tsv(
        curated_dir / "diseases.tsv",
        disease_rows,
        [
            "stage_code",
            "stage_order",
            "disease_id",
            "umls_id",
            "disease_name",
            "disease_type",
            "disease_vocabularies",
            "disease_classes_msh",
            "disease_classes_umls_st",
            "disease_classes_do",
            "disease_classes_hpo",
            "source",
        ],
    )
    write_tsv(
        curated_dir / "genes.tsv",
        gene_rows,
        [
            "gene_id",
            "gene_symbol",
            "gene_ensembl_ids",
            "gene_ncbi_type",
            "gene_protein_str_ids",
            "gene_dsi",
            "gene_dpi",
            "gene_pli",
            "gene_protein_class_ids",
            "gene_protein_class_names",
            "source",
        ],
    )
    assoc_fields = [
        "relation_id",
        "gene_id",
        "gene_symbol",
        "disease_id",
        "disease_umls_id",
        "stage_code",
        "stage_order",
        "assoc_id",
        "score",
        "normalized_score",
        "num_pmids",
        "year_initial",
        "year_final",
        "evidence_index",
        "disgenet_evidence_level",
        "score_breakdown",
        "num_db_snp",
        "num_clinical_trials",
        "num_chemicals",
        "num_pmids_with_chemicals",
        "num_trials_with_chemicals",
        "chemical_evidence",
        "source",
        "tier",
        "tier_reason",
        "api_endpoint",
        "api_query",
        "retrieved_at",
    ]
    write_tsv(curated_dir / "gene_disease_associations.tsv", import_rows, assoc_fields)
    for tier_name, tier_items in tier_rows.items():
        write_tsv(tiers_dir / f"{tier_name}.tsv", sorted(tier_items, key=lambda item: item["relation_id"]), assoc_fields)

    per_disease = Counter(row.get("_stage_code") for row in rows)
    per_tier = {name: len(items) for name, items in tier_rows.items()}
    tier_by_disease: dict[str, Counter[str]] = defaultdict(Counter)
    for name, items in tier_rows.items():
        for item in items:
            tier_by_disease[item["stage_code"]][name] += 1

    report = render_report(
        metadata,
        per_disease,
        per_tier,
        tier_by_disease,
        len(disease_rows),
        len(gene_rows),
        len(import_rows),
        missing_ensembl,
        missing_protein,
        rejected_raw,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    return {
        "raw_rows": len(rows),
        "diseases": len(disease_rows),
        "genes": len(gene_rows),
        "associations_for_import": len(import_rows),
        "tier_counts": per_tier,
        "missing_ensembl_genes": len(missing_ensembl),
        "missing_protein_genes": len(missing_protein),
        "report_path": str(report_path),
    }


def render_report(
    metadata: dict[str, Any],
    per_disease: Counter[str],
    per_tier: dict[str, int],
    tier_by_disease: dict[str, Counter[str]],
    disease_count: int,
    gene_count: int,
    import_count: int,
    missing_ensembl: set[tuple[str, str]],
    missing_protein: set[tuple[str, str]],
    rejected_raw: list[dict[str, Any]],
) -> str:
    lines = [
        "# DisGeNET Liver Backbone Crawl Report",
        "",
        f"- Retrieved at: `{metadata.get('retrieved_at')}`",
        f"- Endpoint: `{metadata.get('endpoint')}`",
        "- Scope: NAFLD, NASH, Fibrosis, Cirrhosis, HCC",
        "- Access note: academic account responses reported curated-source access.",
        "",
        "## Crawl Counts",
        "",
        "| Stage | UMLS CUI | API totalElements | Rows crawled |",
        "|---|---:|---:|---:|",
    ]
    for item in metadata.get("diseases", []):
        lines.append(
            f"| {item['stage_code']} | {item['umls_id']} | {item.get('total_elements')} | {item.get('payload_rows')} |"
        )
    lines.extend(
        [
            "",
            "## Tier Counts",
            "",
            "| Stage | High | Medium | Candidate | Rejected |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for stage in [item["stage_code"] for item in BACKBONE]:
        counts = tier_by_disease.get(stage, Counter())
        lines.append(
            f"| {stage} | {counts.get('high_confidence', 0)} | {counts.get('medium_confidence', 0)} | {counts.get('candidate', 0)} | {counts.get('rejected', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Output Summary",
            "",
            f"- Disease rows: `{disease_count}`",
            f"- Deduplicated Gene rows: `{gene_count}`",
            f"- Import-scope Gene-Disease associations: `{import_count}`",
            f"- Raw merged GDA rows: `{sum(per_disease.values())}`",
            f"- High confidence: `{per_tier.get('high_confidence', 0)}`",
            f"- Medium confidence: `{per_tier.get('medium_confidence', 0)}`",
            f"- Candidate: `{per_tier.get('candidate', 0)}`",
            f"- Rejected: `{per_tier.get('rejected', 0)}`",
            "",
            "## Mapping Risks",
            "",
            f"- Genes missing Ensembl IDs, affecting HPA mapping: `{len(missing_ensembl)}`",
            f"- Genes missing DisGeNET protein crossrefs, affecting STRING/Reactome mapping: `{len(missing_protein)}`",
        ]
    )
    if missing_ensembl:
        sample = ", ".join(f"{symbol}({gene_id})" for gene_id, symbol in sorted(missing_ensembl)[:20])
        lines.append(f"- Missing Ensembl sample: {sample}")
    if missing_protein:
        sample = ", ".join(f"{symbol}({gene_id})" for gene_id, symbol in sorted(missing_protein)[:20])
        lines.append(f"- Missing protein crossref sample: {sample}")
    lines.extend(
        [
            "",
            "## Import Recommendation",
            "",
            "- Import only `high_confidence` and `medium_confidence` associations into the core graph.",
            "- Keep `candidate` as staging evidence for later review.",
            "- Keep `rejected` only for audit; do not import it.",
            "- Run STRING mapping conservatively after this import, using stable protein crossrefs before gene-symbol fallback.",
        ]
    )
    if rejected_raw:
        lines.extend(["", "## Rejected Examples", "", "| Stage | Assoc ID | Gene | Reason |", "|---|---|---|---|"])
        for row in rejected_raw[:20]:
            tier, reason = tier_for_row(row)
            lines.append(
                f"| {row.get('_stage_code', '')} | {row.get('assocID', '')} | {row.get('symbolOfGene', '')} | {reason} |"
            )
    return "\n".join(lines) + "\n"


def validate_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    staging_path = output_dir / "staging" / "gda_raw_merged.jsonl"
    curated_assoc_path = output_dir / "curated" / "gene_disease_associations.tsv"
    if not staging_path.exists() or not curated_assoc_path.exists():
        raise RuntimeError("Expected staging and curated output files were not created")
    tier_total = sum(int(value) for value in summary["tier_counts"].values())
    if tier_total != summary["raw_rows"]:
        raise RuntimeError(f"Tier counts {tier_total} do not match raw rows {summary['raw_rows']}")
    disease_path = output_dir / "curated" / "diseases.tsv"
    with disease_path.open("r", encoding="utf-8", newline="") as handle:
        disease_ids = [row["disease_id"] for row in csv.DictReader(handle, delimiter="\t")]
    if len(disease_ids) != len(set(disease_ids)):
        raise RuntimeError("Duplicate disease_id values found")
    genes_path = output_dir / "curated" / "genes.tsv"
    with genes_path.open("r", encoding="utf-8", newline="") as handle:
        gene_ids = [row["gene_id"] for row in csv.DictReader(handle, delimiter="\t")]
    if len(gene_ids) != len(set(gene_ids)):
        raise RuntimeError("Duplicate gene_id values found")
    with curated_assoc_path.open("r", encoding="utf-8", newline="") as handle:
        relation_ids = [row["relation_id"] for row in csv.DictReader(handle, delimiter="\t")]
    if len(relation_ids) != len(set(relation_ids)):
        raise RuntimeError("Duplicate relation_id values found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/disgenet_liver_backbone"))
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=None, help="Optional cap for smoke tests")
    args = parser.parse_args()

    api_key = os.getenv("DISGENET_API_KEY")
    if not api_key:
        print("DISGENET_API_KEY is not set", file=sys.stderr)
        return 2

    rows, metadata = crawl(api_key, args.output_dir, args.sleep_seconds, args.retries, args.max_pages)
    if not rows:
        print("No DisGeNET rows returned", file=sys.stderr)
        return 1
    summary = build_outputs(rows, args.output_dir, metadata)
    validate_outputs(args.output_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
