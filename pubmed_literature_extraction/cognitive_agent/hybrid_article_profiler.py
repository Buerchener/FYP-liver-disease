#!/usr/bin/env python3
"""Three-level article profiling: features -> rules -> selective bounded LLM."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any


PRIMARY_TYPES = {
    "review", "computational", "human_omics", "clinical",
    "animal", "in_vitro", "mechanistic", "other",
}
SPECIES_SCOPES = {
    "human", "animal", "mixed_non_human", "mixed_human_in_vitro",
    "mixed_human_animal", "mixed_or_not_applicable", "not_applicable", "unclear",
}
EVIDENCE_POSTURES = {
    "synthesis_of_prior_work_no_new_experiment",
    "synthesis_of_prior_randomized_trials_no_new_experiment",
    "computational_prediction_without_experimental_validation",
    "mixed_computational_discovery_with_human_experimental_validation",
    "direct_human_observational_association_not_interventional_causality",
    "direct_human_observational_association_not_randomized_causality",
    "direct_human_interventional_evidence",
    "direct_preclinical_mechanistic_evidence",
    "direct_preclinical_therapeutic_and_mechanistic_evidence",
    "direct_in_vitro_mechanistic_evidence_with_supportive_human_data",
    "direct_in_vitro_mechanistic_evidence",
    "direct_mechanistic_evidence",
    "unclear",
}
MODALITIES = {
    "narrative_review", "systematic_review", "meta_analysis", "network_meta_analysis",
    "network_pharmacology", "database_mining", "molecular_docking", "machine_learning",
    "human_observational", "cross_sectional_baseline_analysis", "cross_sectional",
    "retrospective_cohort", "prospective_cohort", "clinical_trial", "randomized_trial",
    "epidemiology", "mediation_analysis", "bulk_transcriptomics", "single_cell_transcriptomics",
    "proteomics", "metabolomics", "mendelian_randomization", "human_tissue_wet_lab_validation",
    "mouse_in_vivo", "rat_in_vivo", "hepatocyte_in_vitro", "human_cancer_cell_lines",
    "organoid", "gene_silencing", "gene_overexpression", "genetic_overexpression",
    "genetic_knockout", "mechanistic_intervention", "formulation_characterization",
    "pharmacokinetics", "public_human_data_analysis",
    "commentary", "case_report", "health_economic_model", "mathematical_model",
    "quality_improvement", "medical_education_intervention", "human_ex_vivo",
    "clinical_imaging", "animal_imaging", "cell_line_establishment", "mouse_xenograft",
    "genome_wide_association", "polygenic_risk_score", "phenome_wide_association",
}

EVIDENCE_DESIGNS = {
    "evidence_synthesis", "computational_prediction", "health_economic_model",
    "human_observational", "human_interventional", "human_ex_vivo_experiment",
    "human_case_report", "human_omics", "preclinical_experiment",
    "in_vitro_experiment", "mixed_experiment", "mechanistic_experiment",
    "other", "unclear",
}
CAUSAL_STRENGTHS = {
    "not_applicable", "hypothesis_only", "association_only", "interventional",
    "mechanistic", "mixed_association_and_causal", "unclear",
}
VALIDATION_LEVELS = {
    "prior_work_only", "unvalidated", "population_validated",
    "direct_human_observation", "direct_human_intervention", "human_wet_lab_validated",
    "human_ex_vivo_validated", "preclinical_validated", "in_vitro_validated",
    "mixed_experimental_validation", "direct_experiment", "unclear",
}


@dataclass
class ProfileFeatures:
    structured_headings: list[str] = field(default_factory=list)
    review_cues: list[str] = field(default_factory=list)
    computational_cues: list[str] = field(default_factory=list)
    human_cues: list[str] = field(default_factory=list)
    animal_cues: list[str] = field(default_factory=list)
    in_vitro_cues: list[str] = field(default_factory=list)
    omics_cues: list[str] = field(default_factory=list)
    validation_cues: list[str] = field(default_factory=list)
    intervention_cues: list[str] = field(default_factory=list)
    char_count: int = 0
    modality_count: int = 0


@dataclass
class HybridProfile:
    primary_study_type: str
    secondary_modalities: list[str]
    species_scope: str
    evidence_posture: str
    has_structured_results: bool
    high_extraction_complexity: bool
    recommended_route: str
    confidence: float
    rationale: str
    evidence_quotes: list[str]
    source: str
    llm_trigger_reasons: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    llm_status: str = "not_called"
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    evidence_design: str = "unclear"
    causal_strength: str = "unclear"
    validation_level: str = "unclear"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hits(pattern: str, text: str) -> list[str]:
    return list(dict.fromkeys(m.group(0) for m in re.finditer(pattern, text, re.I)))


def split_evidence_posture(posture: str) -> tuple[str, str, str]:
    """Normalize legacy compound posture into three orthogonal audit fields."""
    if posture.startswith("synthesis_of_prior"):
        return "evidence_synthesis", "not_applicable", "prior_work_only"
    if posture == "computational_prediction_without_experimental_validation":
        return "computational_prediction", "hypothesis_only", "unvalidated"
    if posture == "mixed_computational_discovery_with_human_experimental_validation":
        return "human_omics", "mixed_association_and_causal", "human_wet_lab_validated"
    if posture.startswith("direct_human_observational"):
        return "human_observational", "association_only", "direct_human_observation"
    if posture == "direct_human_interventional_evidence":
        return "human_interventional", "interventional", "direct_human_intervention"
    if posture.startswith("direct_preclinical"):
        return "preclinical_experiment", "mechanistic", "preclinical_validated"
    if posture.startswith("direct_in_vitro"):
        return "in_vitro_experiment", "mechanistic", "in_vitro_validated"
    if posture == "direct_mechanistic_evidence":
        return "mechanistic_experiment", "mechanistic", "direct_experiment"
    return "unclear", "unclear", "unclear"


def combine_evidence_fields(design: str, causal: str, validation: str) -> str:
    """Preserve the legacy compound field while new code uses orthogonal fields."""
    if design == "evidence_synthesis":
        return "synthesis_of_prior_work_no_new_experiment"
    if design in {"computational_prediction", "health_economic_model"}:
        return "computational_prediction_without_experimental_validation"
    if design == "human_omics" and validation == "human_wet_lab_validated":
        return "mixed_computational_discovery_with_human_experimental_validation"
    if design == "human_observational":
        return "direct_human_observational_association_not_randomized_causality"
    if design == "human_interventional":
        return "direct_human_interventional_evidence"
    if design in {"preclinical_experiment", "mixed_experiment"}:
        return "direct_preclinical_mechanistic_evidence"
    if design in {"in_vitro_experiment", "human_ex_vivo_experiment"}:
        return "direct_in_vitro_mechanistic_evidence"
    if design == "mechanistic_experiment":
        return "direct_mechanistic_evidence"
    return "unclear"


def _with_split_fields(profile: HybridProfile) -> HybridProfile:
    profile.evidence_design, profile.causal_strength, profile.validation_level = (
        split_evidence_posture(profile.evidence_posture)
    )
    return profile


def parse_features(title: str, abstract: str) -> ProfileFeatures:
    text = f"{title}\n{abstract}"
    # Converted PubMed abstracts often store headings inline, not after newlines.
    headings = [m.upper() for m in re.findall(
        r"(?<![A-Za-z])(?:BACKGROUND(?:\s*&\s*AIMS)?|OBJECTIVES?|PURPOSE|INTRODUCTION|"
        r"METHODS?|RESULTS?|CONCLUSIONS?|DISCUSSION)\s*:", abstract, re.I
    )]
    review = _hits(r"\b(?:systematic review|scoping review|narrative review|review article|"
                   r"meta[- ]analysis|this review|we review|we summarize|review highlights?|"
                   r"this commentary|commentary on)\b", text)
    comp = _hits(r"\b(?:network pharmacology|molecular docking|in silico|bioinformatics|"
                 r"machine learning|deep learning|radiomics|diagnostic model|risk model|WGCNA|"
                 r"Mendelian randomization|GSEA|GSVA|database(?:s)?|GEO|Markov model|"
                 r"stochastic (?:differential )?model|mathematical model|polygenic risk score|"
                 r"genome[- ]wide association)\b", text)
    human = _hits(r"\b(?:patients?|participants?|subjects?|cohort|retrospective|prospective|"
                  r"case-control|clinical trial|randomi[sz]ed|human (?:liver|tissue)|controls?)\b", text)
    animal = _hits(r"\b(?:mice|mouse|murine|rats?|rat model|animal model|in vivo)\b", text)
    invitro = _hits(r"\b(?:in vitro|cell lines?|cultured cells?|organoids?|primary cells?|"
                   r"HepG2|Huh7|Huh1|PLC/PRF/5|hepatocytes?)\b", text)
    omics = _hits(r"\b(?:transcriptom\w*|proteom\w*|metabolom\w*|single[- ]cell|scRNA|"
                 r"RNA[- ]?seq(?:uencing)?|multi[- ]omics|WGCNA)\b", text)
    validation = _hits(r"\b(?:experimentally validated|experimental validation|validated (?:by|in)|"
                      r"western blot|qPCR|PCR|immunohistochemistry|rescue experiment)\b", text)
    if re.search(
        r"\b(?:require|requires|required|need|needs|needed|awaiting)\b.{0,60}"
        r"\b(?:experimental|wet[- ]lab|in vivo|in vitro) validation\b",
        text, re.I | re.S,
    ):
        validation = []
    intervention = _hits(r"\b(?:knockout|knockdown|silencing|overexpress\w*|siRNA|inhibitor|"
                        r"treated with|treatment group|randomi[sz]ed)\b", text)
    modality_groups = [review, comp, human, animal, invitro, omics, validation, intervention]
    return ProfileFeatures(
        structured_headings=headings, review_cues=review, computational_cues=comp,
        human_cues=human, animal_cues=animal, in_vitro_cues=invitro, omics_cues=omics,
        validation_cues=validation, intervention_cues=intervention, char_count=len(text),
        modality_count=sum(bool(group) for group in modality_groups),
    )


def rule_profile(title: str, abstract: str) -> HybridProfile:
    f = parse_features(title, abstract)
    text = f"{title}\n{abstract}"
    mods: list[str] = []
    def add(name: str, condition: bool) -> None:
        if condition and name not in mods:
            mods.append(name)
    add("systematic_review", bool(re.search(r"\bsystematic(?:ally)?\s+(?:review|search)", text, re.I)))
    add("network_meta_analysis", bool(re.search(r"\bnetwork meta[- ]analysis\b", text, re.I)))
    add("meta_analysis", bool(re.search(r"\bmeta[- ]analysis\b", text, re.I)))
    add("narrative_review", bool(f.review_cues) and not any(x in mods for x in ("systematic_review", "meta_analysis", "network_meta_analysis")))
    for name, pat in {
        "network_pharmacology": r"\bnetwork pharmacology\b", "molecular_docking": r"\bmolecular docking\b",
        "machine_learning": r"\bmachine learning\b", "mendelian_randomization": r"\bMendelian randomization\b",
        "single_cell_transcriptomics": r"\b(?:single[- ]cell|scRNA)\b", "bulk_transcriptomics": r"\b(?:RNA sequencing|RNA-seq|transcriptom)\w*\b",
        "retrospective_cohort": r"\bretrospective (?:analysis|cohort|study)\b", "prospective_cohort": r"\bprospective cohort\b",
        "cross_sectional": r"\bcross-sectional\b", "human_observational": r"\b(?:cohort|cross-sectional|case-control|participants?|patients?)\b",
        "randomized_trial": r"\brandomi[sz]ed controlled trial\b", "mouse_in_vivo": r"\b(?:mice|mouse|murine)\b",
        "rat_in_vivo": r"\b(?:rats?|rat model)\b", "human_cancer_cell_lines": r"\b(?:HepG2|Huh7|Huh1|PLC/PRF/5|human .*cell lines?)\b",
        "hepatocyte_in_vitro": r"\b(?:in vitro hepatocytes?|cultured hepatocytes?)\b", "gene_silencing": r"\b(?:silencing|siRNA|knockdown)\b",
        "gene_overexpression": r"\boverexpress\w*\b", "genetic_knockout": r"\b(?:knockout|KO mice)\b",
        "human_tissue_wet_lab_validation": r"\b(?:western blot|qPCR).{0,100}(?:human|patients?|tissues?)|(?:human|patients?|tissues?).{0,100}(?:western blot|qPCR)\b",
        "pharmacokinetics": r"\bpharmacokinetics?\b", "mediation_analysis": r"\bmediation analysis\b",
        "commentary": r"\bcommentary\b", "case_report": r"\bcase report\b",
        "health_economic_model": r"\b(?:cost-effectiveness|Markov model|health economic)\b",
        "mathematical_model": r"\b(?:mathematical model|stochastic differential model)\b",
        "quality_improvement": r"\bquality improvement stud(?:y|ies)\b",
        "medical_education_intervention": r"\b(?:educational|teaching) (?:session|module|intervention)\b",
        "human_ex_vivo": r"\b(?:human .{0,35})?(?:ex vivo|precision-cut .{0,20} slices?)\b",
        "clinical_imaging": r"\b(?:patients?|participants?).{0,120}\b(?:MRI|ultrasound|FibroScan|imaging)\b",
        "animal_imaging": r"\b(?:mice|mouse|rats?).{0,120}\b(?:MRI|ultrasound|imaging)\b",
        "cell_line_establishment": r"\b(?:establish(?:ed|ment)|novel).{0,40}\bcell line\b",
        "mouse_xenograft": r"\b(?:xenograft|tumorigenicity in .{0,20}mice)\b",
        "genome_wide_association": r"\bgenome-wide association\b",
        "polygenic_risk_score": r"\bpolygenic risk score\b",
        "phenome_wide_association": r"\bphenome-wide\b",
    }.items():
        add(name, bool(re.search(pat, text, re.I | re.S)))

    if f.review_cues:
        primary = "review"
        posture = ("synthesis_of_prior_randomized_trials_no_new_experiment"
                   if "network_meta_analysis" in mods else "synthesis_of_prior_work_no_new_experiment")
        species = "human" if "network_meta_analysis" in mods else "mixed_or_not_applicable"
    elif f.omics_cues and f.human_cues and f.validation_cues:
        primary, posture, species = "human_omics", "mixed_computational_discovery_with_human_experimental_validation", "human"
    elif f.computational_cues and not f.validation_cues and not f.animal_cues and not f.in_vitro_cues:
        primary, posture, species = "computational", "computational_prediction_without_experimental_validation", "not_applicable"
    elif "human_ex_vivo" in mods:
        primary, species, posture = "in_vitro", "human", "direct_in_vitro_mechanistic_evidence"
    elif f.human_cues and not f.animal_cues and not f.in_vitro_cues:
        primary, species = "clinical", "human"
        posture = ("direct_human_interventional_evidence" if "randomized_trial" in mods
                   else "direct_human_observational_association_not_randomized_causality")
    elif f.animal_cues:
        primary, species = "animal", "mixed_non_human" if f.in_vitro_cues else "animal"
        posture = "direct_preclinical_mechanistic_evidence"
    elif f.in_vitro_cues:
        primary, species, posture = "in_vitro", "mixed_human_in_vitro", "direct_in_vitro_mechanistic_evidence"
    elif f.intervention_cues:
        primary, species, posture = "mechanistic", "unclear", "direct_mechanistic_evidence"
    else:
        primary, species, posture = "other", "unclear", "unclear"

    conflict = bool(
        (f.review_cues and (f.human_cues or f.animal_cues or f.in_vitro_cues))
        or (f.computational_cues and (f.validation_cues or f.animal_cues or f.in_vitro_cues))
        or sum(bool(x) for x in (f.human_cues, f.animal_cues, f.in_vitro_cues)) >= 2
    )
    high = bool(f.modality_count >= 4 or (len(f.intervention_cues) >= 2 and f.modality_count >= 3))
    route = "FAST" if primary in {"review", "computational"} else ("DEEP" if high else "STANDARD")
    confidence = 0.62 if conflict else (0.88 if primary != "other" else 0.45)
    reasons = []
    # Explicit reviews are locally decisive even when their topic mentions cells/patients.
    adjudication_conflict = conflict and not bool(f.review_cues)
    if adjudication_conflict: reasons.append("mixed_or_conflicting_design_cues")
    if primary == "other": reasons.append("no_dominant_design")
    if f.computational_cues and f.validation_cues: reasons.append("prediction_plus_validation_requires_adjudication")
    # Keep this as an audit feature, but do not spend an LLM call on explicit review prose.
    if f.review_cues and (f.human_cues or f.in_vitro_cues):
        f.review_cues = list(dict.fromkeys([*f.review_cues, "review_topic_may_mimic_primary_study"]))
    return _with_split_fields(HybridProfile(
        primary, sorted(mods), species, posture,
        any(h.startswith("RESULT") or h.startswith("CONCLUSION") for h in f.structured_headings),
        high, route, confidence, "Deterministic multi-label rule profile.", [], "rules",
        reasons, asdict(f), "not_called",
    ))


def _route(profile: dict[str, Any]) -> str:
    if (profile["primary_study_type"] in {"review", "computational"}
            or profile.get("evidence_design") in {"evidence_synthesis", "computational_prediction", "health_economic_model"}
            or (profile.get("evidence_design") == "human_omics"
                and profile.get("causal_strength") == "association_only"
                and profile.get("validation_level") in {"population_validated", "unvalidated"})):
        return "FAST"
    return "DEEP" if profile["high_extraction_complexity"] else "STANDARD"


def build_prompt(title: str, abstract: str, rules: HybridProfile) -> str:
    return f"""Task: adjudicate a biomedical article profile, not its scientific truth.
Use only the supplied title and abstract. Distinguish ARTICLE DESIGN from TOPIC: a review about cell experiments is still a review. A computational discovery followed by wet-lab validation is not prediction-only.

Return exactly one JSON object with these keys:
primary_study_type: one of {sorted(PRIMARY_TYPES)}
secondary_modalities: unique values chosen only from {sorted(MODALITIES)}
species_scope: one of {sorted(SPECIES_SCOPES)}
evidence_design: one of {sorted(EVIDENCE_DESIGNS)}
causal_strength: one of {sorted(CAUSAL_STRENGTHS)}
validation_level: one of {sorted(VALIDATION_LEVELS)}
high_extraction_complexity: boolean; true for mixed modalities/causal layers that materially complicate extraction, not merely long prose
confidence: number 0..1
rationale: <=60 words
evidence_quotes: 1-3 exact contiguous quotes copied from title or abstract, each <=180 characters

You may adjudicate design/modalities/species/evidence fields only. Do not return recommended_route or has_structured_results; deterministic code owns those fields. Do not infer evidence beyond the supplied text.
Rule proposal (may be corrected): {json.dumps(rules.to_dict(), ensure_ascii=False)}
TITLE: {title}
ABSTRACT: {abstract}
"""


def _validated_llm(raw: dict[str, Any], title: str, abstract: str) -> dict[str, Any]:
    text = f"{title}\n{abstract}"
    primary = str(raw.get("primary_study_type", ""))
    species = str(raw.get("species_scope", ""))
    design = str(raw.get("evidence_design", ""))
    causal = str(raw.get("causal_strength", ""))
    validation = str(raw.get("validation_level", ""))
    if (primary not in PRIMARY_TYPES or species not in SPECIES_SCOPES
            or design not in EVIDENCE_DESIGNS or causal not in CAUSAL_STRENGTHS
            or validation not in VALIDATION_LEVELS):
        raise ValueError("LLM returned an out-of-vocabulary profile field")
    modalities = list(dict.fromkeys(str(x) for x in raw.get("secondary_modalities", [])))
    if any(x not in MODALITIES for x in modalities):
        raise ValueError("LLM returned an out-of-vocabulary modality")
    quotes = [str(x).strip() for x in raw.get("evidence_quotes", []) if str(x).strip()]
    if not 1 <= len(quotes) <= 3 or any(q not in text or len(q) > 180 for q in quotes):
        raise ValueError("LLM evidence quote is absent from source or violates bounds")
    confidence = float(raw.get("confidence", 0.0))
    if not 0 <= confidence <= 1:
        raise ValueError("LLM confidence is out of range")
    return {
        "primary_study_type": primary, "secondary_modalities": modalities,
        "species_scope": species,
        "evidence_posture": combine_evidence_fields(design, causal, validation),
        "evidence_design": design, "causal_strength": causal,
        "validation_level": validation,
        "high_extraction_complexity": bool(raw.get("high_extraction_complexity")),
        "confidence": confidence, "rationale": str(raw.get("rationale", ""))[:500],
        "evidence_quotes": quotes,
    }


def profile_article(title: str, abstract: str, *, api_key: str = "", api_base: str = "",
                    model_id: str = "deepseek-v4-flash", force_llm: bool = False) -> HybridProfile:
    rules = rule_profile(title, abstract)
    if not (force_llm or rules.llm_trigger_reasons):
        return rules
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    api_base = (api_base or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
    if not api_key:
        rules.llm_status = "skipped_missing_api_key"
        return rules
    started = time.perf_counter()
    base_prompt = build_prompt(title, abstract, rules)
    try:
        validation_error = ""
        total_prompt_tokens = total_output_tokens = 0
        for attempt in range(2):
            repair = (
                "\nYour previous response failed validation: " + validation_error
                + "\nReturn corrected JSON. Every evidence quote must be copied character-for-character from TITLE or ABSTRACT."
                if validation_error else ""
            )
            body = {
                "model": model_id, "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": "You are a precision-first biomedical study-design adjudicator. Output strict JSON only."},
                    {"role": "user", "content": base_prompt + repair},
                ],
                "response_format": {"type": "json_object"},
            }
            req = urllib.request.Request(
                f"{api_base}/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                payload = json.loads(response.read())
            total_prompt_tokens += int(payload.get("usage", {}).get("prompt_tokens", 0) or 0)
            total_output_tokens += int(payload.get("usage", {}).get("completion_tokens", 0) or 0)
            content = payload["choices"][0]["message"].get("content") or ""
            try:
                judged = _validated_llm(json.loads(content), title, abstract)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                validation_error = f"{type(exc).__name__}: {str(exc)[:180]}"
                if attempt == 1:
                    raise
        result = HybridProfile(
            **judged, has_structured_results=rules.has_structured_results,
            recommended_route="", source="llm_adjudicated", llm_trigger_reasons=rules.llm_trigger_reasons,
            features=rules.features, llm_status="success", latency_seconds=time.perf_counter() - started,
            prompt_tokens=total_prompt_tokens, output_tokens=total_output_tokens,
        )
        result.recommended_route = _route(result.to_dict())
        return result
    except Exception as exc:
        rules.llm_status = f"fallback:{type(exc).__name__}:{str(exc)[:180]}"
        rules.latency_seconds = time.perf_counter() - started
        return rules
