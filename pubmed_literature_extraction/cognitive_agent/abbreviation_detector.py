#!/usr/bin/env python3
"""
cognitive_agent/abbreviation_detector.py — Biomedical Abbreviation Detection & Resolution

Implements the Schwartz-Hearst algorithm (1999, 2003) — the gold standard for
biomedical abbreviation detection. Widely used in BioCreative, UMLS, and clinical NLP.

Algorithm: scans text for "long form (ABBR)" parenthetical patterns, then validates
that the abbreviation is a valid acronym of the long form.

Also includes:
- Common biomedical abbreviation dictionary (curated from UMLS, MeSH)
- Methodology/generic term classifier for entity quality filtering
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections import OrderedDict


# ── Schwartz-Hearst Algorithm ──────────────────────────────────

# Candidate short form: 2-10 chars, starts with letter, contains at least one uppercase
SHORT_FORM_RE = re.compile(r'\b([A-Za-z][A-Za-z0-9]{1,9})\b')

# Parenthetical pattern: "text (ABBR)" or "ABBR (text)"
PARENTHETICAL_RE = re.compile(r'\(([^)]{1,30})\)')


@dataclass
class AbbreviationMap:
    """Bidirectional abbreviation ↔ long form mapping for one article."""
    abbr_to_long: OrderedDict[str, str] = field(default_factory=OrderedDict)
    long_to_abbr: OrderedDict[str, str] = field(default_factory=OrderedDict)

    def add(self, long_form: str, short_form: str):
        """Add a validated abbreviation pair."""
        lf = long_form.strip().lower()
        sf = short_form.strip()
        self.abbr_to_long[sf] = long_form.strip()
        self.long_to_abbr[lf] = sf

    def has_abbr(self, text: str) -> bool:
        """Check if text is a known abbreviation."""
        return text.strip() in self.abbr_to_long

    def has_long(self, text: str) -> bool:
        """Check if text is a known long form."""
        return text.strip().lower() in self.long_to_abbr

    def resolve_to_long(self, text: str) -> str:
        """Resolve abbreviation to long form, or return original."""
        return self.abbr_to_long.get(text.strip(), text)

    def resolve_to_short(self, text: str) -> str:
        """Resolve long form to abbreviation, or return original."""
        return self.long_to_abbr.get(text.strip().lower(), text)

    def is_duplicate_pair(self, cand1: str, cand2: str) -> bool:
        """Check if two entity mentions are abbreviation variants of each other."""
        c1 = cand1.strip()
        c2 = cand2.strip()
        # Direct match
        if self.abbr_to_long.get(c1, "").lower() == c2.lower():
            return True
        if self.abbr_to_long.get(c2, "").lower() == c1.lower():
            return True
        # Both are abbreviations of same long form
        lf1 = self.abbr_to_long.get(c1, "")
        lf2 = self.abbr_to_long.get(c2, "")
        if lf1 and lf2 and lf1.lower() == lf2.lower():
            return True
        return False

    def canonical_name(self, text: str) -> str:
        """Return canonical (long) form if available, else original."""
        return self.abbr_to_long.get(text.strip(), text)

    def to_dict(self) -> dict:
        return {
            "abbr_to_long": dict(self.abbr_to_long),
            "long_to_abbr": dict(self.long_to_abbr),
        }


class AbbreviationDetector:
    """Detect and resolve biomedical abbreviations from text.

    Implements Schwartz-Hearst algorithm:
      1. Scan for parenthetical expressions: "long form (ABBR)"
      2. Validate: ABBR chars must appear in order in long form's word initials
      3. Build bidirectional map for later entity resolution
    """

    # Minimum long form length (words)
    MIN_LONG_WORDS = 1
    MAX_LONG_WORDS = 12

    # Short form constraints
    MIN_SHORT_LEN = 2
    MAX_SHORT_LEN = 12

    # Common false-positive patterns to filter out
    # (numbers, units, statistical terms in parentheses)
    FP_PAREN_PATTERNS = re.compile(
        r'^(?:\d+[\d.,]*|'
        r'p\s*[<>=]\s*[0-9.]+|'
        r'n\s*=\s*\d+|'
        r'HR\s*[0-9.]+|'
        r'CI\s*[0-9.\-%]+|'
        r'OR\s*[0-9.]+|'
        r'SD\s*[0-9.]+|'
        r'Fig\.?\s*\d+|'
        r'Table\s*\d+|'
        r'Ref\.?\s*\d+|'
        r'et\s+al\.?|'
        r'i\.e\.|'
        r'e\.g\.|'
        r'vs\.?|'
        r'etc\.?)$'
    )

    def detect(self, text: str) -> AbbreviationMap:
        """Detect all abbreviation pairs in text.

        Implementation of Schwartz-Hearst (2003) "A Simple Algorithm for Identifying
        Abbreviation Definitions in Biomedical Text."

        Args:
            text: Full article text (title + abstract)

        Returns:
            AbbreviationMap with validated abbreviation pairs
        """
        abbr_map = AbbreviationMap()

        # Find all parenthetical expressions
        for paren_match in PARENTHETICAL_RE.finditer(text):
            paren_text = paren_match.group(1).strip()
            paren_start = paren_match.start()
            paren_end = paren_match.end()

            # Candidate short form must match pattern
            sf_match = SHORT_FORM_RE.fullmatch(paren_text)
            if not sf_match:
                continue
            short_form = sf_match.group(1)

            # Filter: must contain at least one letter
            if not any(c.isalpha() for c in short_form):
                continue

            # Filter: false positive patterns
            if self.FP_PAREN_PATTERNS.match(paren_text):
                continue

            # Find long form: text immediately before '('
            # Schwartz-Hearst: long form is within min(|SF|+5, |SF|*3) words before paren.
            # Use character window: |SF| * 40 chars (generous, covers long multi-word terms)
            max_long_len = min(len(short_form) * 40, 400)
            long_candidate = text[max(0, paren_start - max_long_len):paren_start]

            long_form = self._find_long_form(long_candidate, short_form)
            if long_form:
                abbr_map.add(long_form, short_form)

        # Curated aliases are still article-local: activate them only when both
        # forms occur in this article. This covers established liver acronyms
        # whose letters do not follow strict initials (for example MAFLD),
        # without injecting unrelated global aliases into the candidate set.
        for short_form, long_form in CURATED_ABBREVIATIONS.items():
            short_present = re.search(
                r"(?<![A-Za-z0-9])" + re.escape(short_form) + r"(?![A-Za-z0-9])",
                text, re.IGNORECASE,
            )
            long_present = re.search(re.escape(long_form), text, re.IGNORECASE)
            if short_present and long_present:
                abbr_map.add(long_form, short_form)

        return abbr_map

    def _find_long_form(self, text_before_paren: str, short_form: str) -> str:
        """Extract the long form that matches the abbreviation.

        Core of Schwartz-Hearst (2003): Starting from the end of the short form,
        walk backwards through text characters, matching each short form letter
        to a word-initial letter in the long form. Multiple short form letters
        can match the same word (e.g., 'l' and 't' both in "aminotransferase").

        Args:
            text_before_paren: Text immediately before the opening '('
            short_form: The abbreviation found inside parentheses

        Returns:
            Matched long form string, or empty string
        """
        sf = short_form.lower()
        sf_len = len(sf)
        STOPWORDS = frozenset({
            'of', 'the', 'and', 'in', 'for', 'to', 'a', 'an',
            'with', 'from', 'by', 'on', 'at', 'or', 'as',
            'is', 'was', 'are', 'were', 'been', 'via', 'per',
            'not', 'no', 'its', 'his', 'her', 'their', 'our', 'your',
            'this', 'that', 'these', 'those', 'than', 'then', 'also',
        })

        words = text_before_paren.split()
        if not words:
            return ""

        # Schwartz-Hearst: take up to min(|SF|+5, |SF|*3) words before the paren
        max_scan_words = min(sf_len * 3, len(words))
        if max_scan_words < 1:
            return ""

        # Try each possible start position from the end
        # Collect all valid matches, then pick the best (longest, fewest inside-word matches)
        all_matches: list[tuple[str, int]] = []  # (long_form, inside_match_count)

        for start_word in range(len(words) - 1, max(-1, len(words) - max_scan_words - 2), -1):
            candidate_words = []
            sf_idx = sf_len - 1
            word_idx = start_word
            inside_matches = 0  # Track how many chars matched inside words

            while sf_idx >= 0 and word_idx >= 0:
                word_lower = words[word_idx].lower().strip('(),;:.')

                if not word_lower:
                    word_idx -= 1
                    continue

                def _add_word(w):
                    if not candidate_words or candidate_words[0] != w:
                        candidate_words.insert(0, w)

                # Case 1: short form letter matches first letter of word
                if word_lower[0] == sf[sf_idx]:
                    _add_word(words[word_idx])
                    sf_idx -= 1
                    word_idx -= 1
                # Case 2: short form letter is inside the word
                elif sf[sf_idx] in word_lower[1:]:
                    _add_word(words[word_idx])
                    sf_idx -= 1
                    inside_matches += 1
                    # Don't decrement word_idx — next SF char may also match this word
                # Case 3: stopwords
                elif word_lower in STOPWORDS:
                    _add_word(words[word_idx])
                    word_idx -= 1
                else:
                    break

            if sf_idx < 0 and len(candidate_words) >= 1:
                long_text = ' '.join(candidate_words).strip('(),;:. ')
                long_words_clean = [w for w in candidate_words
                                   if w.lower() not in STOPWORDS]
                if len(long_words_clean) >= 1:
                    all_matches.append((long_text, inside_matches))

        if not all_matches:
            return ""

        # Pick best: prefer fewer inside-word matches, then longer text
        all_matches.sort(key=lambda x: (x[1], -len(x[0].split())))
        return all_matches[0][0]


# ── Curated Biomedical Abbreviation Dictionary ─────────────────

# Common biomedical abbreviations not always defined in-text
# Source: UMLS, MeSH, and manual curation
CURATED_ABBREVIATIONS: dict[str, str] = {
    # Liver diseases
    "NAFLD": "non-alcoholic fatty liver disease",
    "NASH": "non-alcoholic steatohepatitis",
    "MASLD": "metabolic dysfunction-associated steatotic liver disease",
    "MASH": "metabolic dysfunction-associated steatohepatitis",
    "MAFLD": "metabolic dysfunction-associated fatty liver disease",
    "HCC": "hepatocellular carcinoma",
    "ALD": "alcoholic liver disease",
    "PBC": "primary biliary cholangitis",
    "PSC": "primary sclerosing cholangitis",
    "AIH": "autoimmune hepatitis",
    "HBV": "hepatitis B virus",
    "HCV": "hepatitis C virus",
    "CLD": "chronic liver disease",
    "ACLF": "acute-on-chronic liver failure",
    "PH": "portal hypertension",

    # Diabetes / Metabolic
    "T2DM": "type 2 diabetes mellitus",
    "T1DM": "type 1 diabetes mellitus",
    "IR": "insulin resistance",
    "MetS": "metabolic syndrome",
    "FPG": "fasting plasma glucose",
    "HbA1c": "glycated hemoglobin",
    "BMI": "body mass index",

    # Common clinical measurements
    "ALT": "alanine aminotransferase",
    "AST": "aspartate aminotransferase",
    "GGT": "gamma-glutamyl transferase",
    "ALP": "alkaline phosphatase",
    "AFP": "alpha-fetoprotein",
    "INR": "international normalized ratio",
    "MELD": "model for end-stage liver disease",
    "CP": "Child-Pugh",

    # Common biomolecules
    "ROS": "reactive oxygen species",
    "TNF": "tumor necrosis factor",
    "IL6": "interleukin 6",
    "TGFB": "transforming growth factor beta",
    "VEGF": "vascular endothelial growth factor",
    "EGF": "epidermal growth factor",
    "FGF": "fibroblast growth factor",
    "PDGF": "platelet-derived growth factor",
    "HGF": "hepatocyte growth factor",
    "IGF": "insulin-like growth factor",
    "NFKB": "nuclear factor kappa B",
    "MAPK": "mitogen-activated protein kinase",

    # Common cell types
    "HSC": "hepatic stellate cell",
    "KC": "Kupffer cell",
    "LSEC": "liver sinusoidal endothelial cell",
    "NK": "natural killer cell",
    "Treg": "regulatory T cell",

    # Methodology (for filtering)
    "PCR": "polymerase chain reaction",
    "ELISA": "enzyme-linked immunosorbent assay",
    "IHC": "immunohistochemistry",
    "WB": "western blot",
    "qPCR": "quantitative polymerase chain reaction",
    "RNA-seq": "RNA sequencing",
    "ChIP": "chromatin immunoprecipitation",
    "GWAS": "genome-wide association study",
    "RCT": "randomized controlled trial",
    "HR": "hazard ratio",
    "OR": "odds ratio",
    "CI": "confidence interval",
    "SD": "standard deviation",
    "SE": "standard error",
    "AUC": "area under the curve",
    "ROC": "receiver operating characteristic",
}


# ── Extended Blacklists ────────────────────────────────────────

# Methodology/generic terms that should NOT be entities
METHODOLOGY_BLACKLIST: frozenset[str] = frozenset({
    # Experimental techniques
    "network pharmacology", "molecular docking", "molecular dynamics",
    "gene ontology", "go analysis", "kegg", "kegg pathway", "kegg signaling pathway",
    "gene set enrichment analysis", "gsea", "ingenuity pathway analysis", "ipa",

    # Bioinformatics / Databases
    "string database", "geo database", "tcga", "gtex", "david", "metascape",
    "cytoscape", "clusterprofiler", "reactome", "wikipathways", "biocarta",
    "genecards", "disgenet", "omics", "proteomics", "metabolomics", "transcriptomics",
    "genomics", "bioinformatics analysis", "computational analysis",

    # Statistics / Methods language
    "network analysis", "pathway analysis", "functional enrichment",
    "enrichment analysis", "differential expression analysis",
    "principal component analysis", "pca", "hierarchical clustering",
    "kaplan-meier", "cox regression", "logistic regression",
    "systematic review", "meta-analysis", "meta analysis",
    "literature review", "narrative review", "scoping review",

    # Generic descriptors (not specific entities)
    "gene symbols", "gene symbol", "gene expression", "protein expression",
    "gene targets", "drug targets", "therapeutic targets", "molecular targets",
    "disease-related targets", "disease targets", "therapeutic agents",
    "chemicals", "compounds", "molecules", "drugs", "agents", "biomarkers",
    "candidate genes", "target genes", "key genes", "hub genes",
    "differentially expressed genes", "degs",

    # Generic study descriptions
    "clinical outcomes", "clinical parameters", "biochemical parameters",
    "laboratory parameters", "clinical characteristics", "baseline characteristics",
    "demographic characteristics", "patient characteristics",
    "healthy controls", "control group", "study group", "treatment group",
    "placebo group", "experimental group",

    # Vague pathway mentions
    "signaling pathway", "signaling pathways", "signaling cascade",
    "metabolic pathway", "metabolic pathways", "biological pathway",
    "signal transduction", "cell signaling", "cellular signaling",
    "immune response", "inflammatory response", "oxidative stress response",
    "dna damage response", "unfolded protein response", "upstream regulator",
    "downstream effector", "downstream target",

    # Generic cell biology terms
    "cell proliferation", "cell apoptosis", "cell migration", "cell invasion",
    "cell cycle", "cell death", "cell survival", "cell differentiation",
    "cell growth", "cell viability", "cell senescence",
    "angiogenesis", "metastasis", "epithelial-mesenchymal transition", "emt",

    # Non-specific disease descriptors
    "liver disease", "liver diseases", "chronic liver disease",
    "liver injury", "hepatic injury", "liver damage", "hepatic damage",
    "liver dysfunction", "hepatic dysfunction", "liver failure",
    "liver fibrosis", "hepatic fibrosis", "liver steatosis", "hepatic steatosis",

    # Statistical / Research terms
    "p value", "p-value", "confidence interval", "hazard ratio", "odds ratio",
    "risk ratio", "relative risk", "standard deviation", "standard error",
    "area under curve", "receiver operating characteristic",
    "sensitivity", "specificity", "positive predictive value", "negative predictive value",

    # Non-entity extraction artifacts
    "inclusion criteria", "exclusion criteria", "eligibility criteria",
    "primary endpoint", "secondary endpoint", "primary outcome", "secondary outcome",
    "adverse events", "side effects", "safety profile", "efficacy",
    "pharmacokinetics", "pharmacodynamics", "bioavailability",
    "dose-response", "dose response", "concentration-dependent", "time-dependent",

    # Methods artifacts from machine reading
    "immunohistochemical staining", "western blot analysis", "elisa assay",
    "flow cytometry", "mass spectrometry", "chromatography",
    "spectroscopy", "microscopy", "histological analysis", "histopathology",
})


# Quality score penalties for entity names
ENTITY_QUALITY_PENALTY_PATTERNS: list[tuple[str, float, str]] = [
    # (regex pattern, penalty, reason)
    (r'^[a-z]{1,3}$', 0.5, "too short (≤3 lowercase chars)"),
    (r'^[a-z]+$', 0.2, "all lowercase common word"),
    (r'^(gene|protein|cell|tissue|pathway|disease|metabolite)\b', 0.4, "starts with generic category"),
    (r'\b(analysis|assay|test|method|technique|approach|study|trial|review)s?\b', 0.3, "contains methodology term"),
    (r'\b(database|ontology|repository|registry|atlas|portal|server)\b', 0.4, "database/resource name"),
    (r'\b(controls?|groups?|patients?|subjects?|samples?|cohorts?)$', 0.5, "study population term"),
    (r'^(all|any|some|many|several|various|different|other|such|each|every|this|these|those|the|a|an|in|on|at|by|to|for|of|and|or|not|no|is|was|are|were|been|has|had|have|can|may|will|would|could|should)\b', 0.5, "stopword-like"),
]


def is_methodology_noise(name: str) -> bool:
    """Check if an entity name looks like methodology noise."""
    name_lower = name.lower().strip()
    if name_lower in METHODOLOGY_BLACKLIST:
        return True
    # Check multi-word combinations (substring match — ONLY for pure methodology terms
    # that would never appear inside a legitimate entity name)
    for term in ("network pharmacology", "molecular docking", "gene ontology",
                 "pathway analysis", "enrichment analysis", "gene symbols",
                 "protein expression", "western blot", "flow cytometry",
                 "mass spectrometry", "principal component analysis",
                 "cox regression", "logistic regression", "kaplan-meier"):
        if term in name_lower:
            return True
    # For "kegg" — only match standalone or "KEGG pathway"/"KEGG signaling" (not "KEGG enrichment")
    if name_lower == "kegg" or name_lower.startswith("kegg "):
        return True
    return False


def score_entity_name_quality(name: str) -> tuple[float, list[str]]:
    """Score entity name quality. Returns (penalty, [reasons]).

    1.0 = perfect, 0.0 = should be rejected.
    """
    name_stripped = name.strip()
    name_lower = name_stripped.lower()
    penalty = 0.0
    reasons: list[str] = []

    for pattern, p, reason in ENTITY_QUALITY_PENALTY_PATTERNS:
        if re.search(pattern, name_lower):
            penalty += p
            reasons.append(reason)

    # Cap at 0.95
    penalty = min(penalty, 0.95)

    return round(1.0 - penalty, 3), reasons


def resolve_abbreviation_conflict(
    name1: str, name2: str, abbr_map: AbbreviationMap,
) -> str | None:
    """If name1 and name2 are abbreviation variants, return the canonical name.

    Priority: long form > first detected

    Returns:
        Canonical name, or None if they're not abbreviation variants
    """
    # Direct check (if abbr_map available)
    if abbr_map and abbr_map.is_duplicate_pair(name1, name2):
        return abbr_map.canonical_name(name1)

    # Curated dictionary check
    n1_canonical = CURATED_ABBREVIATIONS.get(name1, name1)
    n2_canonical = CURATED_ABBREVIATIONS.get(name2, name2)

    if n1_canonical.lower() == n2_canonical.lower():
        # Return the longer (more descriptive) form
        return n1_canonical if len(n1_canonical) >= len(n2_canonical) else n2_canonical

    return None
