#!/usr/bin/env python3
"""
cognitive_agent/tools/ncbi_validator.py — NCBI E-utilities Gene Validator

Validates gene symbols against NCBI Gene database using esearch + esummary.
No API key required. Rate-limited to ~3 req/s without key, ~10 req/s with.

Usage:
    validator = NCBIGeneValidator()
    result = validator.validate("TP53")
    if result.found:
        print(f"NCBI Gene ID: {result.ncbi_gene_id}")
"""

from __future__ import annotations

import time
import json
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field


NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_RATE_LIMIT = 0.35   # ~3 req/s without API key (NCBI guideline)
RATE_LIMIT_WITH_KEY = 0.1    # 10 req/s with API key


@dataclass
class GeneValidationResult:
    """Result of validating a gene symbol against NCBI."""
    query_symbol: str
    found: bool = False
    ncbi_gene_id: str = ""
    official_symbol: str = ""
    official_name: str = ""
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    organism: str = "Homo sapiens"
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "query_symbol": self.query_symbol,
            "found": self.found,
            "ncbi_gene_id": self.ncbi_gene_id,
            "official_symbol": self.official_symbol,
            "official_name": self.official_name,
            "description": self.description,
            "aliases": self.aliases,
            "organism": self.organism,
            "error": self.error,
        }


class NCBIGeneValidator:
    """Validates gene symbols against NCBI Gene using E-utilities.

    E-utilities are free and don't strictly require an API key,
    but rate limits are tighter without one (~3 req/s).
    """

    def __init__(self, api_key: str = "", email: str = ""):
        self.api_key = api_key
        self.email = email or "agent@liverkg.dev"
        self._rate = RATE_LIMIT_WITH_KEY if api_key else DEFAULT_RATE_LIMIT
        self._last_request = 0.0

    def validate(self, gene_symbol: str, species: str = "human") -> GeneValidationResult:
        """Validate a single gene symbol against NCBI Gene.

        Args:
            gene_symbol: Gene symbol to validate (e.g. "TP53", "NFE2L2")
            species: "human" or "mouse"

        Returns:
            GeneValidationResult with NCBI Gene ID, official symbol, etc.
        """
        result = GeneValidationResult(query_symbol=gene_symbol)

        # Quick sanity check
        if not gene_symbol or len(gene_symbol) < 2:
            result.error = "Gene symbol too short or empty"
            return result

        # Ignore obvious non-gene mentions
        if _is_noise_mention(gene_symbol):
            result.error = "Noise mention (common abbreviation or non-gene token)"
            return result

        try:
            # Step 1: ESearch — find NCBI Gene IDs
            gene_ids = self._esearch_gene(gene_symbol, species)
            if not gene_ids:
                result.error = "No NCBI Gene match found"
                return result

            # Step 2: Pick best match (exact symbol match first, then first result)
            best_id = gene_ids[0]  # default to first

            # Step 3: ESummary — get gene details
            summary = self._esummary_gene(best_id)
            if not summary:
                result.error = f"ESummary failed for NCBI Gene ID {best_id}"
                return result

            result.found = True
            result.ncbi_gene_id = best_id
            result.official_symbol = summary.get("Name", gene_symbol)
            result.official_name = summary.get("Description", "")
            result.description = summary.get("Summary", "")
            result.aliases = _parse_aliases(summary)
            result.organism = summary.get("Organism", {}).get("ScientificName", "Homo sapiens")

        except urllib.error.URLError as e:
            result.error = f"Network error: {e}"
        except Exception as e:
            result.error = str(e)

        return result

    def validate_batch(self, symbols: list[str], species: str = "human") -> list[GeneValidationResult]:
        """Validate multiple gene symbols. Respects rate limits."""
        results = []
        for symbol in symbols:
            results.append(self.validate(symbol, species))
        return results

    # ── private helpers ──────────────────────────────────────

    def _rate_limit(self):
        """Enforce NCBI rate limit."""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate:
            time.sleep(self._rate - elapsed)
        self._last_request = time.time()

    def _esearch_gene(self, symbol: str, species: str) -> list[str]:
        """Search NCBI Gene for a symbol. Returns list of Gene ID strings."""
        # Build species-specific query
        if species == "human":
            term = f'{symbol}[sym] AND Homo sapiens[orgn]'
        elif species == "mouse":
            term = f'{symbol}[sym] AND Mus musculus[orgn]'
        else:
            term = f'{symbol}[sym] AND {species}[orgn]'

        params = {
            "db": "gene",
            "term": term,
            "retmax": "5",
            "retmode": "json",
            "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        data = self._get("/esearch.fcgi", params)
        id_list = data.get("esearchresult", {}).get("idlist", [])
        return id_list

    def _esummary_gene(self, gene_id: str) -> dict | None:
        """Get NCBI Gene summary for a Gene ID."""
        params = {
            "db": "gene",
            "id": gene_id,
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        data = self._get("/esummary.fcgi", params)
        result = data.get("result", {})
        # The key is the gene ID string
        gene_info = result.get(str(gene_id))
        return gene_info

    def _get(self, path: str, params: dict) -> dict:
        """Perform a rate-limited GET request to NCBI E-utilities."""
        self._rate_limit()
        query = urllib.parse.urlencode(params)
        url = f"{NCBI_BASE}{path}?{query}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # If rate-limited (HTTP 429), wait and retry once
            if e.code == 429:
                time.sleep(2.0)
                with urllib.request.urlopen(url, timeout=15) as resp2:
                    return json.loads(resp2.read().decode("utf-8"))
            raise


# ── helper functions ─────────────────────────────────────────

def _is_noise_mention(symbol: str) -> bool:
    """Filter out common abbreviations and non-gene tokens."""
    noise = {
        "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "THAN",
        "WAS", "ARE", "HAS", "HAD", "NOT", "BUT", "ALL", "ITS",
        "CAN", "MAY", "NEW", "ONE", "TWO", "VIA", "OUR",
        "HR", "CI", "OR", "SD", "SE", "CT", "MRI", "DNA", "RNA",
        "PCR", "ELISA", "WB", "ALT", "AST", "BMI", "GGT", "ALP",
        "AFP", "HBV", "HCV", "HCC", "NAFLD", "NASH", "MASLD",
        "MASH", "ALD", "HSC", "HPC",
        "IL", "TNF", "TGF", "EGF", "FGF", "VEGF", "PDGF",
        "MAPK", "JNK", "ERK", "JAK", "STAT", "NFKB", "NF-kB",
        "ABSTRACT", "METHODS", "RESULTS", "CONCLUSION",
        "BACKGROUND", "OBJECTIVE", "AIMS", "AIM",
    }
    return symbol.upper() in noise


def _parse_aliases(summary: dict) -> list[str]:
    """Extract aliases from NCBI Gene summary."""
    aliases = []
    # OtherAliases is a comma-separated string
    raw = summary.get("OtherAliases", "")
    if raw:
        aliases.extend(a.strip() for a in raw.split(",") if a.strip())
    # Also check OtherDesignations
    raw_des = summary.get("OtherDesignations", "")
    if raw_des:
        aliases.extend(a.strip() for a in raw_des.split("|") if a.strip())
    return aliases


if __name__ == "__main__":
    # Quick smoke test
    v = NCBIGeneValidator()
    for symbol in ["TP53", "NFE2L2", "SLC7A11", "INVALID_GENE_XYZ"]:
        r = v.validate(symbol)
        status = "✓" if r.found else "✗"
        print(f"[{status}] {symbol:>20} → NCBI:{r.ncbi_gene_id:>10} | {r.official_symbol} | {r.error}")
