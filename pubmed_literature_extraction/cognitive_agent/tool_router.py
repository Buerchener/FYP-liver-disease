#!/usr/bin/env python3
"""Fast, deterministic tool routing for the article-level cognitive agent.

The router is deliberately not an LLM.  It keeps planning latency negligible,
routes expensive memory/model calls only when they can change a decision, and
leaves all final import-readiness decisions to the deterministic verifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cognitive_agent.hybrid_article_profiler import HybridProfile, rule_profile


CALL = "CALL"
SKIP = "SKIP"
DEFER = "DEFER"


@dataclass(frozen=True)
class ArticleProfile:
    study_type: str
    char_count: int
    sentence_count: int
    max_sentence_words: int
    entity_signal_count: int
    mechanistic_signal_count: int
    evidence_signal_count: int
    has_structured_results: bool
    high_complexity: bool
    reason_codes: tuple[str, ...] = ()
    secondary_modalities: tuple[str, ...] = ()
    species_scope: str = "unclear"
    evidence_design: str = "unclear"
    causal_strength: str = "unclear"
    validation_level: str = "unclear"
    profile_confidence: float = 0.0
    profile_source: str = "legacy_rules"
    complexity_vector: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "study_type": self.study_type,
            "char_count": self.char_count,
            "sentence_count": self.sentence_count,
            "max_sentence_words": self.max_sentence_words,
            "entity_signal_count": self.entity_signal_count,
            "mechanistic_signal_count": self.mechanistic_signal_count,
            "evidence_signal_count": self.evidence_signal_count,
            "has_structured_results": self.has_structured_results,
            "high_complexity": self.high_complexity,
            "reason_codes": list(self.reason_codes),
            "secondary_modalities": list(self.secondary_modalities),
            "species_scope": self.species_scope,
            "evidence_design": self.evidence_design,
            "causal_strength": self.causal_strength,
            "validation_level": self.validation_level,
            "profile_confidence": self.profile_confidence,
            "profile_source": self.profile_source,
            "complexity_vector": self.complexity_vector,
        }


@dataclass(frozen=True)
class ComplexityDimension:
    score: float = 0.0
    reasons: tuple[str, ...] = ()
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": round(max(0.0, min(1.0, float(self.score))), 4),
            "reasons": list(self.reasons),
            "features": self.features,
        }


@dataclass(frozen=True)
class ComplexityVector:
    design_complexity: ComplexityDimension = field(default_factory=ComplexityDimension)
    extraction_complexity: ComplexityDimension = field(default_factory=ComplexityDimension)
    evidence_complexity: ComplexityDimension = field(default_factory=ComplexityDimension)
    linking_complexity: ComplexityDimension = field(default_factory=ComplexityDimension)
    runtime_uncertainty: ComplexityDimension = field(default_factory=ComplexityDimension)

    @property
    def max_score(self) -> float:
        return max(
            self.design_complexity.score,
            self.extraction_complexity.score,
            self.evidence_complexity.score,
            self.linking_complexity.score,
            self.runtime_uncertainty.score,
        )

    def to_dict(self) -> dict:
        return {
            "design_complexity": self.design_complexity.to_dict(),
            "extraction_complexity": self.extraction_complexity.to_dict(),
            "evidence_complexity": self.evidence_complexity.to_dict(),
            "linking_complexity": self.linking_complexity.to_dict(),
            "runtime_uncertainty": self.runtime_uncertainty.to_dict(),
            "max_score": round(self.max_score, 4),
        }


@dataclass(frozen=True)
class ToolDecision:
    tool: str
    decision: str
    reason: str
    cost_class: str = "low"
    expected_value: float = 0.0
    budget_level: str = "minimal"
    plan_status: str = "production"
    hard_masked: bool = False
    candidate_pool: bool = False
    utility: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "decision": self.decision,
            "reason": self.reason,
            "cost_class": self.cost_class,
            "expected_value": round(float(self.expected_value), 4),
            "budget_level": self.budget_level,
            "plan_status": self.plan_status,
            "hard_masked": self.hard_masked,
            "candidate_pool": self.candidate_pool,
            "utility": self.utility,
        }


@dataclass
class ToolPlan:
    stage: str
    route: str
    profile: ArticleProfile
    decisions: dict[str, ToolDecision] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    plan_status: str = "production"
    early_stop_reasons: list[str] = field(default_factory=list)
    legacy_route: str = ""
    routing_version: str = "legacy"
    layer_trace: dict = field(default_factory=dict)

    def should_call(self, tool: str) -> bool:
        item = self.decisions.get(tool)
        return bool(item and item.decision == CALL)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "route": self.route,
            "profile": self.profile.to_dict(),
            "reason_codes": list(dict.fromkeys(self.reason_codes)),
            "plan_status": self.plan_status,
            "early_stop_reasons": list(dict.fromkeys(self.early_stop_reasons)),
            "legacy_route": self.legacy_route,
            "route_changed": bool(self.legacy_route and self.legacy_route != self.route),
            "routing_version": self.routing_version,
            "four_layer_trace": self.layer_trace,
            "tools": [item.to_dict() for item in self.decisions.values()],
            "called_tools": [
                name for name, item in self.decisions.items() if item.decision == CALL
            ],
            "skipped_tools": [
                name for name, item in self.decisions.items() if item.decision == SKIP
            ],
            "deferred_tools": [
                name for name, item in self.decisions.items() if item.decision == DEFER
            ],
        }


class ArticleToolRouter:
    """Two-stage execution with a four-layer, evidence-aware shadow policy.

    The four logical layers are: immutable safety masks, a cheap candidate
    pool, post-extraction sufficiency judgment, and net-utility gating.  The
    policy is deliberately deterministic and audit-only until validated.
    """

    ROUTING_VERSION = "four-layer-shadow-v1"
    SAFETY_INVARIANTS = (
        "deterministic_verifier_cannot_be_bypassed",
        "safe_write_policy_cannot_be_overridden",
        "tools_cannot_invent_missing_evidence_or_endpoints",
        "external_graph_context_is_not_article_evidence",
    )

    _REVIEW = re.compile(
        r"\b(systematic review|scoping review|narrative review|meta[- ]analysis|"
        r"review article|this review|we review|we summarize)\b", re.I
    )
    _COMPUTATIONAL = re.compile(
        r"\b(network pharmacology|molecular docking|in silico|bioinformatics|"
        r"machine learning|deep learning|radiomics|diagnostic model|risk model)\b",
        re.I,
    )
    _CLINICAL = re.compile(
        r"\b(patient|patients|cohort|clinical trial|randomi[sz]ed|prospective|"
        r"retrospective|case-control)\b",
        re.I,
    )
    _OMICS = re.compile(
        r"\b(transcriptom|proteom|metabolom|single[- ]cell|rna[- ]seq|multi[- ]omics)\w*\b",
        re.I,
    )
    _ANIMAL = re.compile(r"\b(mice|mouse|murine|rats?|animal model|in vivo)\b", re.I)
    _IN_VITRO = re.compile(
        r"\b(in vitro|cell lines?|cultured cells?|organoids?|primary cells?)\b", re.I
    )
    _MECHANISTIC = re.compile(
        r"\b(knock(?:ed)?down|knockout|overexpress\w*|silenc\w*|mechanis\w*|"
        r"mediate\w*|interact\w*|bind\w*|phosphorylat\w*|signali[sz]ing|"
        r"inhibit\w*|suppress\w*|activat\w*|promote\w*|regulat\w*)\b",
        re.I,
    )
    _EVIDENCE = re.compile(
        r"\b(we found|we observed|we demonstrate|we showed|results showed|"
        r"significantly|was associated with|were associated with|confirmed|validated)\b",
        re.I,
    )
    _ENTITY = re.compile(
        r"\b(?:[A-Z][A-Z0-9-]{1,9}|HCC|NAFLD|NASH|MASLD|MASH|cirrhosis|"
        r"fibrosis|hepatitis|hepatocellular carcinoma|liver cancer)\b"
    )

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    @staticmethod
    def _evidence_fields(posture: str) -> tuple[str, str, str]:
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

    @staticmethod
    def _score(value: float) -> float:
        return round(max(0.0, min(1.0, value)), 4)

    def complexity_before_extraction(
        self, title: str, abstract: str, profile: HybridProfile | None = None,
    ) -> ComplexityVector:
        profile = profile or rule_profile(title, abstract)
        f = profile.features or {}
        text = f"{title}\n{abstract}"
        modality_count = int(f.get("modality_count", 0) or 0)
        mixed_design = len([
            key for key in ("human_cues", "animal_cues", "in_vitro_cues", "computational_cues")
            if f.get(key)
        ])
        design_reasons: list[str] = []
        design_score = 0.12 * max(0, modality_count - 1) + 0.12 * max(0, mixed_design - 1)
        if mixed_design >= 2:
            design_reasons.append("mixed_study_modalities")
        if f.get("computational_cues") and f.get("validation_cues"):
            design_score += 0.30
            design_reasons.append("prediction_plus_experimental_validation")
        if f.get("review_cues") and mixed_design >= 2:
            design_score += 0.12
            design_reasons.append("review_topic_mimics_primary_design")

        entity_signals = len(self._ENTITY.findall(text))
        relation_triggers = len(self._MECHANISTIC.findall(text)) + len(self._EVIDENCE.findall(text))
        outcome_signals = len(re.findall(
            r"\b(?:primary outcomes?|secondary outcomes?|mortality|survival|odds ratio|"
            r"hazard ratio|AUC|sensitivity|specificity|pharmacokinetics?|biodistribution)\b",
            text, re.I,
        ))
        contrast_signals = len(re.findall(r"\b(?:whereas|however|conversely|compared with|versus)\b", text, re.I))
        extraction_score = (
            min(entity_signals, 30) / 30 * 0.30
            + min(relation_triggers, 12) / 12 * 0.30
            + min(outcome_signals, 8) / 8 * 0.25
            + min(contrast_signals, 4) / 4 * 0.15
        )
        extraction_reasons = [
            reason for active, reason in (
                (entity_signals >= 12, "dense_entity_mentions"),
                (relation_triggers >= 6, "dense_relation_triggers"),
                (outcome_signals >= 3, "multiple_outcomes_or_endpoints"),
                (contrast_signals >= 2, "comparative_or_contrasting_claims"),
            ) if active
        ]

        evidence_score = 0.0
        evidence_reasons: list[str] = []
        if f.get("review_cues"):
            evidence_score += 0.25
            evidence_reasons.append("prior_work_vs_current_article_boundary")
        if f.get("computational_cues"):
            evidence_score += 0.18
            evidence_reasons.append("prediction_vs_observation_boundary")
        if f.get("validation_cues") and f.get("computational_cues"):
            evidence_score += 0.28
            evidence_reasons.append("mixed_prediction_and_validation_evidence")
        if re.search(r"\b(?:may|might|could|suggest\w*|potential|appears? to)\b", text, re.I):
            evidence_score += 0.14
            evidence_reasons.append("hedged_claims")
        if re.search(r"\b(?:not associated|no significant|did not|failed to)\b", text, re.I):
            evidence_score += 0.15
            evidence_reasons.append("negative_or_null_findings")

        abbreviation_pairs = len(re.findall(r"\b[A-Za-z][A-Za-z0-9 -]{3,80}\s*\([A-Z][A-Z0-9-]{1,9}\)", text))
        short_symbols = len(re.findall(r"\b[A-Z][A-Z0-9-]{1,5}\b", text))
        gene_protein_ambiguity = len(re.findall(r"\b(?:gene|protein|expression|encoded by|encodes)\b", text, re.I))
        linking_score = (
            min(abbreviation_pairs, 6) / 6 * 0.25
            + min(short_symbols, 20) / 20 * 0.45
            + min(gene_protein_ambiguity, 8) / 8 * 0.30
        )
        linking_reasons = [
            reason for active, reason in (
                (abbreviation_pairs >= 2, "multiple_abbreviation_definitions"),
                (short_symbols >= 8, "dense_short_biomedical_symbols"),
                (gene_protein_ambiguity >= 3, "gene_protein_surface_ambiguity"),
            ) if active
        ]
        return ComplexityVector(
            design_complexity=ComplexityDimension(
                self._score(design_score), tuple(design_reasons),
                {"modality_group_count": modality_count, "mixed_design_group_count": mixed_design},
            ),
            extraction_complexity=ComplexityDimension(
                self._score(extraction_score), tuple(extraction_reasons),
                {"entity_signals": entity_signals, "relation_triggers": relation_triggers,
                 "outcome_signals": outcome_signals, "contrast_signals": contrast_signals},
            ),
            evidence_complexity=ComplexityDimension(
                self._score(evidence_score), tuple(evidence_reasons),
                {"has_review_cues": bool(f.get("review_cues")),
                 "has_computational_cues": bool(f.get("computational_cues")),
                 "has_validation_cues": bool(f.get("validation_cues"))},
            ),
            linking_complexity=ComplexityDimension(
                self._score(linking_score), tuple(linking_reasons),
                {"abbreviation_pairs": abbreviation_pairs, "short_symbols": short_symbols,
                 "gene_protein_cues": gene_protein_ambiguity},
            ),
        )

    def _shadow_article_profile(
        self, title: str, abstract: str, profile: HybridProfile | None = None,
        vector: ComplexityVector | None = None,
    ) -> ArticleProfile:
        hybrid = profile or rule_profile(title, abstract)
        vector = vector or self.complexity_before_extraction(title, abstract, hybrid)
        legacy = self.profile(title, abstract)
        evidence_design = hybrid.evidence_design
        causal_strength = hybrid.causal_strength
        validation_level = hybrid.validation_level
        return ArticleProfile(
            study_type=hybrid.primary_study_type,
            char_count=legacy.char_count,
            sentence_count=legacy.sentence_count,
            max_sentence_words=legacy.max_sentence_words,
            entity_signal_count=legacy.entity_signal_count,
            mechanistic_signal_count=legacy.mechanistic_signal_count,
            evidence_signal_count=legacy.evidence_signal_count,
            has_structured_results=hybrid.has_structured_results,
            high_complexity=vector.max_score >= 0.55,
            reason_codes=tuple(dict.fromkeys([
                f"study_type_{hybrid.primary_study_type}", *hybrid.llm_trigger_reasons,
            ])),
            secondary_modalities=tuple(hybrid.secondary_modalities),
            species_scope=hybrid.species_scope,
            evidence_design=evidence_design,
            causal_strength=causal_strength,
            validation_level=validation_level,
            profile_confidence=hybrid.confidence,
            profile_source=hybrid.source,
            complexity_vector=vector.to_dict(),
        )

    @staticmethod
    def _utility(
        quality_gain: float, *, latency_cost: float = 0.0,
        monetary_cost: float = 0.0, safety_risk: float = 0.0,
        threshold: float = 0.0, basis: tuple[str, ...] = (),
    ) -> dict:
        """Return an auditable rule prior, never an uncalibrated probability."""
        net = quality_gain - latency_cost - monetary_cost - safety_risk
        return {
            "quality_gain": round(max(0.0, min(1.0, quality_gain)), 4),
            "latency_cost": round(max(0.0, latency_cost), 4),
            "monetary_cost": round(max(0.0, monetary_cost), 4),
            "safety_risk": round(max(0.0, safety_risk), 4),
            "net_utility": round(net, 4),
            "call_threshold": round(threshold, 4),
            "positive": bool(net >= threshold),
            "basis": list(basis),
            "estimator": "auditable_rule_prior_not_learned_probability",
        }

    @staticmethod
    def _candidate_scores(entity: dict) -> list[float]:
        scores = []
        for candidate in entity.get("candidates", []) or []:
            try:
                scores.append(float(candidate.get("score", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        return sorted(scores, reverse=True)

    def _linking_uncertainty(self, entities: list[dict]) -> tuple[float, dict]:
        """SkewRoute-style score-distribution signal for graph lookup value."""
        ambiguous = 0
        close_margins = 0
        low_top_scores = 0
        candidate_entities = 0
        margins: list[float] = []
        for entity in entities:
            status = str(entity.get("neo4j_status", "") or "").upper()
            scores = self._candidate_scores(entity)
            if status == "AMBIGUOUS" or entity.get("ambiguity_reason"):
                ambiguous += 1
            if not scores:
                continue
            candidate_entities += 1
            if scores[0] < 0.82:
                low_top_scores += 1
            if len(scores) >= 2:
                margin = scores[0] - scores[1]
                margins.append(round(margin, 4))
                if margin < 0.08:
                    close_margins += 1
        denominator = max(len(entities), 1)
        score = self._score(
            0.65 * min(1.0, ambiguous / denominator * 3)
            + 0.25 * min(1.0, close_margins / denominator * 4)
            + 0.10 * min(1.0, low_top_scores / denominator * 2)
        )
        return score, {
            "entity_count": len(entities),
            "candidate_entity_count": candidate_entities,
            "ambiguous_entity_count": ambiguous,
            "close_top2_margin_count": close_margins,
            "low_top_score_count": low_top_scores,
            "top2_margins": margins,
        }

    @staticmethod
    def _mask_reason(masked: dict[str, str], tool: str) -> str:
        return masked.get(tool, "")

    def shadow_plan_before_extraction(
        self, title: str, abstract: str, *, legacy_plan: ToolPlan,
        memory_available: bool, rag_enabled: bool, second_llm_enabled: bool,
        reviewer_enabled: bool = False, chunk_max_chars: int = 1800,
        profile: HybridProfile | None = None,
    ) -> ToolPlan:
        vector = self.complexity_before_extraction(title, abstract, profile)
        article_profile = self._shadow_article_profile(title, abstract, profile, vector)
        prediction_only = article_profile.evidence_design in {"computational_prediction", "health_economic_model"}
        synthesis_only = article_profile.evidence_design == "evidence_synthesis"
        population_omics_only = bool(
            article_profile.evidence_design == "human_omics"
            and article_profile.causal_strength == "association_only"
            and article_profile.validation_level in {"population_validated", "unvalidated"}
        )
        mixed_experimental_design = bool(
            article_profile.study_type in {"human_omics", "animal", "in_vitro", "mechanistic"}
            and article_profile.causal_strength in {"mechanistic", "mixed_association_and_causal"}
            and vector.design_complexity.score >= 0.35
        )
        if prediction_only or synthesis_only or population_omics_only:
            route = "FAST"
        elif (profile or rule_profile(title, abstract)).high_extraction_complexity or mixed_experimental_design:
            route = "DEEP"
        else:
            route = "STANDARD"
        plan = ToolPlan(
            stage="pre_extraction", route=route, profile=article_profile,
            reason_codes=list(article_profile.reason_codes), plan_status="shadow",
            legacy_route=legacy_plan.route, routing_version=self.ROUTING_VERSION,
        )
        # Layer 1: evidence policy is an immutable action mask.  It cannot skip
        # extraction/verification, but it can forbid unsupported deep tools.
        hard_masks: dict[str, str] = {}
        if synthesis_only:
            hard_masks.update({
                "article_chunker": "review_fast_path_does_not_need_deep_extraction_windows",
                "context_memory": "review_fast_path_defers_graph_context",
                "relation_recovery": "review_has_no_current_experiment",
                "causal_reasoner": "review_cannot_support_article_level_causality",
            })
        if prediction_only:
            hard_masks.update({
                "article_chunker": "prediction_only_fast_path_does_not_need_deep_extraction_windows",
                "context_memory": "prediction_only_fast_path_defers_graph_context",
                "relation_recovery": "prediction_only_has_no_experimental_relation_target",
                "causal_reasoner": "prediction_only_cannot_support_causal_inference",
            })
        if population_omics_only:
            hard_masks.update({
                "article_chunker": "population_omics_fast_path_avoids_deep_extraction_windows",
                "context_memory": "population_omics_fast_path_defers_graph_context",
                "causal_reasoner": "population_association_is_not_mechanistic_causality",
            })
        if not rag_enabled or not memory_available:
            hard_masks["neo4j_rag"] = "graph_memory_unavailable_or_disabled"
        if not second_llm_enabled:
            hard_masks["second_llm_refiner"] = "second_model_disabled"
        if not reviewer_enabled:
            hard_masks["debug_reviewer"] = "debug_reviewer_disabled"

        # Layer 2: cheap taxonomy and complexity features only narrow the tool
        # pool.  Deferred tools still require post-extraction evidence.
        section_count = len(
            (profile or rule_profile(title, abstract)).features.get("structured_headings", [])
        )
        needs_chunking = bool(
            len(f"TITLE: {title}\nABSTRACT: {abstract}") > chunk_max_chars
            and (
                vector.extraction_complexity.score >= 0.50
                or (section_count >= 4 and vector.extraction_complexity.score >= 0.40)
                or (
                    vector.design_complexity.score >= 0.55
                    and vector.extraction_complexity.score >= 0.40
                )
            )
        )
        candidate_pool = {
            "article_chunker": bool(needs_chunking and "article_chunker" not in hard_masks),
            "context_memory": bool(
                memory_available and vector.linking_complexity.score >= 0.62
                and "context_memory" not in hard_masks
            ),
            "neo4j_rag": "neo4j_rag" not in hard_masks,
            "second_llm_refiner": "second_llm_refiner" not in hard_masks,
            "relation_recovery": "relation_recovery" not in hard_masks,
            "causal_reasoner": "causal_reasoner" not in hard_masks,
            "conflict_resolver": True,
            "debug_reviewer": "debug_reviewer" not in hard_masks,
        }
        plan.layer_trace = {
            "layer_1_safety_mask": {
                "invariants": list(self.SAFETY_INVARIANTS),
                "masked_tools": hard_masks,
            },
            "layer_2_candidate_pool": {
                "tools": candidate_pool,
                "profile_source": article_profile.profile_source,
                "profile_confidence": article_profile.profile_confidence,
                "complexity_vector": vector.to_dict(),
            },
            "layer_3_sufficiency_judgment": {
                "status": "deferred_until_deterministic_verification",
            },
            "layer_4_net_utility": {
                "status": "deferred_until_candidate_level_evidence",
            },
        }
        def decide(tool: str, decision: str, reason: str, cost: str,
                   value: float, budget: str = "minimal", *,
                   masked: bool = False, candidate: bool = False,
                   utility: dict | None = None) -> None:
            plan.decisions[tool] = ToolDecision(
                tool, decision, reason, cost, value, budget, "shadow",
                masked, candidate, utility or {},
            )
        decide("abbreviation_detector", CALL, "mandatory lossless normalization", "low", 0.95)
        decide("langextract_candidate_generator", CALL, "primary high-recall generator", "high", 0.95, "standard")
        decide("deterministic_verifier", CALL, "mandatory evidence and schema gate", "low", 1.0)
        decide("decision_engine", CALL, "mandatory write-policy gate", "low", 1.0)
        decide("golden_example_selector", CALL, "profile-matched examples", "low", 0.70)
        chunk_call = candidate_pool["article_chunker"]
        decide(
            "article_chunker", CALL if chunk_call else SKIP,
            "long evidence-dense or multi-section article" if chunk_call
            else hard_masks.get("article_chunker", "length alone does not justify chunking"),
            "medium", 0.68 if chunk_call else 0.05, "standard" if chunk_call else "minimal",
            masked=bool(hard_masks.get("article_chunker")), candidate=chunk_call,
            utility=self._utility(
                0.68 if chunk_call else 0.05, latency_cost=0.08,
                threshold=0.25, basis=("long_evidence_dense_document",) if chunk_call else (),
            ),
        )
        context_call = candidate_pool["context_memory"]
        decide(
            "context_memory", CALL if context_call else SKIP,
            "high pre-extraction linking ambiguity" if context_call
            else hard_masks.get("context_memory", "defer graph memory until verified ambiguity"),
            "medium", 0.62 if context_call else 0.10,
            masked=bool(hard_masks.get("context_memory")), candidate=context_call,
            utility=self._utility(
                0.62 if context_call else 0.10, latency_cost=0.10,
                threshold=0.30, basis=("pre_extraction_linking_ambiguity",) if context_call else (),
            ),
        )
        for tool, cost, enabled, reason in (
            ("neo4j_rag", "medium", candidate_pool["neo4j_rag"], "post-verification linking gate"),
            ("second_llm_refiner", "high", candidate_pool["second_llm_refiner"], "post-verification semantic-risk gate"),
            ("relation_recovery", "high", candidate_pool["relation_recovery"], "requires a verified extraction gap and explicit trigger"),
            ("causal_reasoner", "low", candidate_pool["causal_reasoner"], "requires import-ready causal evidence"),
            ("conflict_resolver", "low", True, "requires graph-eligible relation"),
            ("debug_reviewer", "high", candidate_pool["debug_reviewer"], "post-verification debug only"),
        ):
            masked_reason = self._mask_reason(hard_masks, tool)
            decide(
                tool, DEFER if enabled else SKIP, masked_reason or reason, cost, 0.0,
                masked=bool(masked_reason), candidate=enabled,
            )
        if synthesis_only:
            plan.early_stop_reasons.append("evidence_synthesis_has_no_current_experiment")
        if prediction_only:
            plan.early_stop_reasons.append("prediction_only_without_experimental_validation")
        if population_omics_only:
            plan.early_stop_reasons.append("population_level_omics_without_wet_lab_validation")
        return plan

    def profile(self, title: str, abstract: str) -> ArticleProfile:
        text = f"{title}\n{abstract}"
        hybrid = rule_profile(title, abstract)
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
        word_counts = [len(re.findall(r"\b\w+[\w-]*\b", item)) for item in sentences]
        max_words = max(word_counts, default=0)
        entity_signals = len(self._ENTITY.findall(text))
        mechanisms = len(self._MECHANISTIC.findall(text))
        evidence = len(self._EVIDENCE.findall(text))
        structured_results = bool(re.search(r"(?:^|\n)\s*(RESULTS?|CONCLUSIONS?)\s*[:.]", abstract, re.I))

        # Use the same multi-label deterministic profiler as preprocessing.
        # This prevents a single computational keyword from masking later
        # human/wet-lab validation and recognizes explicit review prose in the
        # abstract as article design rather than experimental evidence.
        study_type = hybrid.primary_study_type

        conjunction_load = len(re.findall(r"\b(?:and|whereas|while|but|however)\b", text, re.I))
        high_complexity = bool(
            hybrid.high_extraction_complexity
            or max_words >= 55 or mechanisms >= 4 or conjunction_load >= 12
        )
        reasons = [f"study_type_{study_type}"]
        if structured_results:
            reasons.append("structured_results_present")
        if high_complexity:
            reasons.append("complex_sentences_or_mechanism_density")
        if entity_signals >= 4:
            reasons.append("high_entity_signal_density")

        return ArticleProfile(
            study_type=study_type,
            char_count=len(text),
            sentence_count=len(sentences),
            max_sentence_words=max_words,
            entity_signal_count=entity_signals,
            mechanistic_signal_count=mechanisms,
            evidence_signal_count=evidence,
            has_structured_results=structured_results,
            high_complexity=high_complexity,
            reason_codes=tuple(reasons),
            secondary_modalities=tuple(hybrid.secondary_modalities),
            species_scope=hybrid.species_scope,
            evidence_design=hybrid.evidence_design,
            causal_strength=hybrid.causal_strength,
            validation_level=hybrid.validation_level,
            profile_confidence=hybrid.confidence,
            profile_source=hybrid.source,
        )

    def plan_before_extraction(
        self,
        title: str,
        abstract: str,
        *,
        memory_available: bool,
        rag_enabled: bool,
        second_llm_enabled: bool,
        reviewer_enabled: bool,
        chunk_max_chars: int = 1800,
        chunk_complexity_min_chars: int = 1400,
    ) -> ToolPlan:
        profile = self.profile(title, abstract)
        if not self.enabled:
            route = "LEGACY"
        elif (
            profile.study_type in {"review", "computational"}
            and profile.evidence_signal_count == 0
            and not profile.has_structured_results
        ):
            route = "FAST"
        elif profile.high_complexity and profile.study_type in {
            "mechanistic", "human_omics", "clinical", "animal", "in_vitro"
        }:
            route = "DEEP"
        else:
            route = "STANDARD"

        plan = ToolPlan(
            stage="pre_extraction",
            route=route,
            profile=profile,
            reason_codes=list(profile.reason_codes),
        )
        always = {
            "abbreviation_detector": "cheap deterministic normalization",
            "langextract_candidate_generator": "primary high-recall candidate generator",
            "deterministic_verifier": "mandatory schema and evidence gate",
            "decision_engine": "mandatory write-policy gate",
        }
        for tool, reason in always.items():
            plan.decisions[tool] = ToolDecision(tool, CALL, reason, "low" if tool != "langextract_candidate_generator" else "high")
        plan.decisions["golden_example_selector"] = ToolDecision(
            "golden_example_selector", CALL,
            "select 3 positive demonstrations and 1 boundary example by article profile",
            "low",
        )
        needs_chunking = bool(
            profile.char_count > chunk_max_chars
            or (
                profile.high_complexity
                and profile.char_count >= chunk_complexity_min_chars
            )
        )
        plan.decisions["article_chunker"] = ToolDecision(
            "article_chunker", CALL if needs_chunking else SKIP,
            (
                "long or high-complexity abstract requires section-aligned extraction windows"
                if needs_chunking else "short/simple abstract is safer and faster as one window"
            ),
            "medium" if needs_chunking else "low",
        )

        pre_context = bool(
            not self.enabled
            or (
                memory_available
                and (
                    route == "DEEP"
                    and profile.entity_signal_count >= 3
                    and (
                        (
                            profile.study_type in {"mechanistic", "animal", "in_vitro"}
                            and profile.mechanistic_signal_count >= 3
                        )
                        or (
                            profile.study_type == "human_omics"
                            and profile.mechanistic_signal_count >= 1
                        )
                    )
                )
            )
        )
        plan.decisions["context_memory"] = ToolDecision(
            "context_memory",
            CALL if pre_context else SKIP,
            (
                "complex evidence-bearing article benefits from pre-extraction entity memory"
                if pre_context
                else "defer memory until verified entities show ambiguity or high value"
            ),
            "medium",
        )
        plan.decisions["neo4j_rag"] = ToolDecision(
            "neo4j_rag", DEFER if rag_enabled else SKIP,
            "decide after deterministic verification" if rag_enabled else "RAG disabled",
            "medium",
        )
        plan.decisions["second_llm_refiner"] = ToolDecision(
            "second_llm_refiner", DEFER if second_llm_enabled else SKIP,
            "call only for recovery, structural repair, or high-value review"
            if second_llm_enabled else "second model disabled",
            "high",
        )
        plan.decisions["debug_reviewer"] = ToolDecision(
            "debug_reviewer", DEFER if reviewer_enabled else SKIP,
            "debug reviewer is post-verification only" if reviewer_enabled else "debug reviewer disabled",
            "high",
        )
        return plan

    def plan_after_verification(
        self,
        pre_plan: ToolPlan,
        extraction: dict,
        verification: dict,
        *,
        memory_available: bool,
        rag_enabled: bool,
        second_llm_enabled: bool,
        second_llm_mode: str = "conditional",
        reviewer_enabled: bool = False,
        recovery_candidate_count: int = 0,
    ) -> ToolPlan:
        profile = pre_plan.profile
        plan = ToolPlan(
            stage="post_verification",
            route=pre_plan.route,
            profile=profile,
            decisions=dict(pre_plan.decisions),
            reason_codes=list(pre_plan.reason_codes),
        )
        raw_entities = extraction.get("entities", []) or []
        relations = verification.get("relations", []) or []
        pair_core_active = any(
            "pair_classifier_candidate" in set(item.get("quality_flags", []) or [])
            for item in relations
        )
        extraction_failed = bool(extraction.get("error"))
        parse_failed = any("parse" in str(item).casefold() for item in extraction.get("warnings", []) or [])
        zero_entities = not raw_entities

        endpoint_flags = {
            "filtered_endpoint", "unresolved_endpoint", "subject_endpoint_missing",
            "object_endpoint_missing",
        }
        evidence_flags = {"empty_evidence", "evidence_not_contiguous", "endpoint_not_grounded"}
        has_endpoint_issue = any(endpoint_flags & set(item.get("quality_flags", [])) for item in relations)
        has_schema_issue = any("schema_mismatch" in item.get("quality_flags", []) for item in relations)
        has_evidence_issue = any(evidence_flags & set(item.get("quality_flags", [])) for item in relations)
        semantic_hard_blockers = {
            "schema_mismatch", "negated", "scoped_negation", "evidence_contradicted",
            "subject_endpoint_missing", "object_endpoint_missing", "empty_evidence",
            "filtered_endpoint", "unresolved_endpoint",
        }
        # Semantic uncertainty (weak trigger heuristics, hedging, judge
        # uncertainty, high-risk predicates) is exactly what the bounded
        # second model is for.  Deterministic hard blockers stay local.
        reviewable_relations = [
            item for item in relations
            if item.get("schema_valid", True)
            and not (set(item.get("quality_flags", [])) & semantic_hard_blockers)
            and (
                bool(set(item.get("quality_flags", [])) & {
                    "trigger_missing", "trigger_not_linking_endpoints",
                    "trigger_direction_mismatch", "weak_evidence", "uncertain",
                    "pair_low_confidence", "pair_ambiguous_predicate",
                    "judge_uncertain", "judge_verifier_conflict",
                })
                or str(item.get("predicate", "")).upper() in {
                    "PROGNOSTIC_IN", "INTERACTS_WITH", "EXPRESSED_IN", "PROGRESSES_TO",
                }
                or bool(re.search(
                    r"\b(?:potential|candidate|implicated|may|might|could|suggest\w*|"
                    r"target for|diagnos\w*|treatment)\b",
                    str(item.get("evidence", "") or ""),
                    re.IGNORECASE,
                ))
            )
        ]
        has_reviewable_relation = bool(reviewable_relations)
        has_recovery_candidate = recovery_candidate_count > 0
        high_value = any(
            item.get("import_ready")
            and item.get("predicate") in {"PROGNOSTIC_IN", "PROGRESSES_TO", "ENCODES"}
            for item in relations
        )
        import_ready = sum(bool(item.get("import_ready")) for item in relations)
        hard_issue = has_endpoint_issue or has_schema_issue or has_evidence_issue or parse_failed

        if extraction_failed or (zero_entities and profile.study_type not in {"review", "computational"}):
            plan.route = "RECOVERY"
            plan.reason_codes.append("primary_extractor_failure_quarantined")
        elif (has_reviewable_relation or has_recovery_candidate) and (
            hard_issue or high_value or profile.high_complexity or has_recovery_candidate
        ):
            plan.route = "DEEP"
            plan.reason_codes.append(
                "bounded_relation_recovery_required"
                if has_recovery_candidate
                else "grounded_candidates_require_semantic_adjudication"
            )

        rag_call = bool(
            rag_enabled
            and (
                not self.enabled
                or (
                    memory_available
                    and reviewable_relations
                    and (
                        high_value
                        or (pre_plan.route == "DEEP" and import_ready > 0)
                    )
                )
            )
        )
        plan.decisions["neo4j_rag"] = ToolDecision(
            "neo4j_rag",
            CALL if rag_call else SKIP,
            (
                "verified endpoints need linking/type/conflict hints"
                if rag_call else "no verified ambiguity or high-value relation needs graph memory"
            ),
            "medium",
        )

        llm_call = bool(
            second_llm_enabled
            and (
                not self.enabled
                or (
                    not extraction_failed
                    and (has_reviewable_relation or has_recovery_candidate)
                )
            )
        )
        plan.decisions["second_llm_refiner"] = ToolDecision(
            "second_llm_refiner",
            CALL if llm_call else SKIP,
            (
                "bounded adjudication of grounded triples and relation-gap choices"
                if llm_call else "no reviewable triple or bounded recovery choice"
            ),
            "high",
        )

        review_call = bool(
            reviewer_enabled
            and (not self.enabled or plan.route in {"DEEP", "RECOVERY"})
        )
        plan.decisions["debug_reviewer"] = ToolDecision(
            "debug_reviewer", CALL if review_call else SKIP,
            "debug inspection for a difficult article" if review_call else "clean fast/standard article",
            "high",
        )
        causal_call = bool(not self.enabled or import_ready)
        plan.decisions["causal_reasoner"] = ToolDecision(
            "causal_reasoner", CALL if causal_call else SKIP,
            "infer only from import-ready relations" if import_ready else "no import-ready relation",
            "low",
        )
        conflict_call = bool(not self.enabled or relations)
        plan.decisions["conflict_resolver"] = ToolDecision(
            "conflict_resolver", CALL if conflict_call else SKIP,
            "relations require deterministic conflict policy" if relations else "no relation candidate",
            "low",
        )
        return plan

    def complexity_after_verification(
        self, pre_vector: ComplexityVector, extraction: dict, verification: dict,
        *, recovery_candidate_count: int = 0,
    ) -> ComplexityVector:
        raw_entities = extraction.get("entities", []) or []
        relations = verification.get("relations", []) or []
        warnings = extraction.get("warnings", []) or []
        parse_failed = any("parse" in str(item).casefold() for item in warnings)
        extraction_failed = bool(extraction.get("error"))
        all_flags = [set(item.get("quality_flags", []) or []) for item in relations]
        hard_flags = {
            "schema_mismatch", "negated", "scoped_negation", "evidence_contradicted",
            "empty_evidence", "subject_endpoint_missing", "object_endpoint_missing",
            "filtered_endpoint", "unresolved_endpoint",
        }
        semantic_flags = {
            "trigger_missing", "trigger_not_linking_endpoints", "trigger_direction_mismatch",
            "weak_evidence", "uncertain", "pair_low_confidence",
            "pair_ambiguous_predicate",
        }
        linking_flags = {
            "subject_endpoint_missing", "object_endpoint_missing", "unresolved_endpoint",
            "ambiguous_endpoint", "subject_ambiguous", "object_ambiguous",
        }
        hard_count = sum(bool(flags & hard_flags) for flags in all_flags)
        semantic_count = sum(bool(flags & semantic_flags) for flags in all_flags)
        linking_count = sum(bool(flags & linking_flags) for flags in all_flags)
        invalid_count = sum(not bool(item.get("schema_valid", True)) for item in relations)
        import_ready = sum(bool(item.get("import_ready")) for item in relations)
        denominator = max(len(relations), 1)
        score = (
            0.35 * extraction_failed + 0.20 * parse_failed
            + 0.20 * (hard_count / denominator)
            + 0.15 * (semantic_count / denominator)
            + 0.10 * (invalid_count / denominator)
        )
        if raw_entities and not relations and recovery_candidate_count:
            score += 0.25
        runtime_reasons = [
            reason for active, reason in (
                (extraction_failed, "primary_extractor_failed"),
                (parse_failed, "structured_output_parse_warning"),
                (hard_count > 0, "deterministic_hard_blockers_present"),
                (semantic_count > 0, "semantic_risk_candidates_present"),
                (invalid_count > 0, "schema_invalid_candidates_present"),
                (bool(raw_entities and not relations and recovery_candidate_count), "bounded_relation_gap_present"),
            ) if active
        ]
        post_linking_score = max(
            pre_vector.linking_complexity.score,
            self._score(linking_count / denominator),
        )
        linking_reasons = list(pre_vector.linking_complexity.reasons)
        if linking_count:
            linking_reasons.append("verified_endpoint_linking_ambiguity")
        return ComplexityVector(
            design_complexity=pre_vector.design_complexity,
            extraction_complexity=pre_vector.extraction_complexity,
            evidence_complexity=pre_vector.evidence_complexity,
            linking_complexity=ComplexityDimension(
                post_linking_score, tuple(dict.fromkeys(linking_reasons)),
                {**pre_vector.linking_complexity.features,
                 "verified_linking_issue_count": linking_count},
            ),
            runtime_uncertainty=ComplexityDimension(
                self._score(score), tuple(runtime_reasons),
                {
                    "entity_count": len(raw_entities), "relation_count": len(relations),
                    "import_ready_count": import_ready, "hard_blocked_count": hard_count,
                    "semantic_risk_count": semantic_count, "linking_issue_count": linking_count,
                    "schema_invalid_count": invalid_count, "parse_failed": parse_failed,
                    "extraction_failed": extraction_failed,
                    "recovery_candidate_count": recovery_candidate_count,
                },
            ),
        )

    def shadow_plan_after_verification(
        self, shadow_pre_plan: ToolPlan, legacy_post_plan: ToolPlan,
        extraction: dict, verification: dict, *, memory_available: bool,
        rag_enabled: bool, second_llm_enabled: bool, reviewer_enabled: bool = False,
        recovery_candidate_count: int = 0,
    ) -> ToolPlan:
        raw_vector = shadow_pre_plan.profile.complexity_vector or {}
        def dim(name: str) -> ComplexityDimension:
            value = raw_vector.get(name, {}) or {}
            return ComplexityDimension(
                float(value.get("score", 0.0) or 0.0),
                tuple(value.get("reasons", []) or []),
                dict(value.get("features", {}) or {}),
            )
        pre_vector = ComplexityVector(
            design_complexity=dim("design_complexity"),
            extraction_complexity=dim("extraction_complexity"),
            evidence_complexity=dim("evidence_complexity"),
            linking_complexity=dim("linking_complexity"),
            runtime_uncertainty=dim("runtime_uncertainty"),
        )
        vector = self.complexity_after_verification(
            pre_vector, extraction, verification,
            recovery_candidate_count=recovery_candidate_count,
        )
        profile = ArticleProfile(
            **{
                **shadow_pre_plan.profile.__dict__,
                "high_complexity": vector.max_score >= 0.55,
                "complexity_vector": vector.to_dict(),
            }
        )
        entities = verification.get("entities", []) or []
        relations = verification.get("relations", []) or []
        relation_flags = [set(item.get("quality_flags", []) or []) for item in relations]
        semantic_flags = {
            "trigger_missing", "trigger_not_linking_endpoints", "trigger_direction_mismatch",
            "weak_evidence", "uncertain", "pair_low_confidence",
            "pair_ambiguous_predicate", "judge_uncertain", "judge_verifier_conflict",
        }
        hard_flags = {
            "schema_mismatch", "negated", "scoped_negation", "evidence_contradicted",
            "subject_endpoint_missing", "object_endpoint_missing", "empty_evidence",
            "filtered_endpoint", "unresolved_endpoint",
        }
        # Deterministic hard blockers never reach the second model.  Semantic
        # uncertainty (weak trigger heuristics, hedging, judge uncertainty,
        # high-risk predicates) always may: starving the adjudicator turned
        # out to cost far more precision than the calls it saved.
        reviewable = [
            item for item, flags in zip(relations, relation_flags)
            if item.get("schema_valid", True) and not (flags & hard_flags)
            and (
                flags & semantic_flags
                or item.get("predicate") in {
                    "PROGNOSTIC_IN", "INTERACTS_WITH", "EXPRESSED_IN", "PROGRESSES_TO",
                }
            )
        ]
        # Layer 1 carries forward immutable article-policy masks and adds
        # candidate-level masks. A later utility score can never undo them.
        pre_trace = shadow_pre_plan.layer_trace or {}
        hard_masks = dict(
            pre_trace.get("layer_1_safety_mask", {}).get("masked_tools", {}) or {}
        )
        if profile.evidence_design == "evidence_synthesis":
            hard_masks.update({
                "relation_recovery": "review_has_no_current_experiment",
                "causal_reasoner": "review_cannot_support_article_level_causality",
            })
        if profile.evidence_design in {"computational_prediction", "health_economic_model"}:
            hard_masks.update({
                "relation_recovery": "prediction_only_has_no_experimental_relation_target",
                "causal_reasoner": "prediction_only_cannot_support_causal_inference",
            })
        if (
            profile.evidence_design == "human_omics"
            and profile.causal_strength == "association_only"
            and profile.validation_level in {"population_validated", "unvalidated"}
        ):
            hard_masks["causal_reasoner"] = "population_association_is_not_mechanistic_causality"
        linking_issues = sum(bool(flags & {
            "ambiguous_endpoint", "subject_ambiguous", "object_ambiguous", "type_ambiguous",
        }) for flags in relation_flags)
        import_ready = [item for item in relations if item.get("import_ready")]
        all_hard_blocked = bool(relations) and all(bool(flags & hard_flags) for flags in relation_flags)
        repairable_linking_flags = {
            "ambiguous_endpoint", "subject_ambiguous", "object_ambiguous", "type_ambiguous",
        }
        non_linking_hard_flags = hard_flags - repairable_linking_flags
        all_non_linking_hard_blocked = bool(relations) and all(
            bool(flags & non_linking_hard_flags) for flags in relation_flags
        )
        extraction_failed = bool(extraction.get("error"))
        parse_failed = any(
            "parse" in str(item).casefold()
            for item in extraction.get("warnings", []) or []
        )
        if extraction_failed:
            hard_masks.update({
                "neo4j_rag": "primary_extraction_failed_no_grounded_candidate",
                "second_llm_refiner": "primary_extraction_failed_open_regeneration_forbidden",
                "relation_recovery": "primary_extraction_failed_open_regeneration_forbidden",
                "causal_reasoner": "primary_extraction_failed",
                "conflict_resolver": "primary_extraction_failed",
            })
        if all_non_linking_hard_blocked:
            hard_masks["second_llm_refiner"] = "all_candidates_are_deterministic_hard_blocks"
            hard_masks["neo4j_rag"] = "hard_evidence_or_schema_failures_are_not_linking_problems"
        if not import_ready:
            hard_masks["causal_reasoner"] = "no_import_ready_relation"
            hard_masks["conflict_resolver"] = "no_graph_eligible_relation"

        linking_uncertainty, linking_features = self._linking_uncertainty(entities)
        if linking_issues:
            linking_uncertainty = max(linking_uncertainty, 0.72)
            linking_features["verified_relation_linking_issue_count"] = linking_issues

        causal_posture = bool(
            profile.causal_strength in {"interventional", "mechanistic", "mixed_association_and_causal"}
            and profile.validation_level not in {"prior_work_only", "unvalidated"}
        )
        existing_graph_relations = [
            item for item in import_ready
            if str(item.get("neo4j_status", "") or "").upper()
            in {"KNOWN", "INVERTED", "CONTRADICTING"}
        ]
        recovery_gap = bool(
            not extraction_failed and recovery_candidate_count > 0 and not relations
        )
        if recovery_candidate_count <= 0:
            hard_masks["relation_recovery"] = "no_explicit_trigger_grounded_recovery_candidate"
        elif relations:
            hard_masks["relation_recovery"] = "primary_extractor_already_returned_relation_candidates"

        # Layer 2 candidate pool: a tool must have both capability and a
        # relevant evidence class before its value is estimated.
        candidate_pool = {
            "neo4j_rag": bool(
                rag_enabled and memory_available and linking_uncertainty >= 0.20
                and "neo4j_rag" not in hard_masks
            ),
            "second_llm_refiner": bool(
                second_llm_enabled and reviewable
                and "second_llm_refiner" not in hard_masks
            ),
            "relation_recovery": bool(
                recovery_gap and "relation_recovery" not in hard_masks
            ),
            "causal_reasoner": bool(
                causal_posture and import_ready
                and "causal_reasoner" not in hard_masks
            ),
            "conflict_resolver": bool(
                existing_graph_relations and "conflict_resolver" not in hard_masks
            ),
            "debug_reviewer": bool(
                reviewer_enabled and not extraction_failed
                and "debug_reviewer" not in hard_masks
            ),
        }

        # Layer 3 asks whether the current result is already sufficient.  It
        # uses verifier state, never model self-confidence alone.
        sufficiency = {
            "extraction_failed": extraction_failed,
            "parse_failed": parse_failed,
            "entity_count": len(entities),
            "relation_count": len(relations),
            "import_ready_count": len(import_ready),
            "hard_blocked_count": sum(bool(flags & hard_flags) for flags in relation_flags),
            "all_hard_blocked": all_hard_blocked,
            "semantic_reviewable_count": len(reviewable),
            "recovery_candidate_count": recovery_candidate_count,
            "bounded_relation_gap": recovery_gap,
            "linking_uncertainty": linking_uncertainty,
            "linking_features": linking_features,
            "existing_graph_relation_count": len(existing_graph_relations),
            "causal_evidence_posture": causal_posture,
            "all_candidates_import_ready": bool(relations) and len(import_ready) == len(relations),
        }
        deep_audit_required = bool(
            shadow_pre_plan.route != "FAST"
            and not extraction_failed
            and (
                vector.runtime_uncertainty.score >= 0.10
                or len(import_ready) >= 2
                or (
                    shadow_pre_plan.route == "DEEP"
                    and vector.design_complexity.score >= 0.35
                    and vector.extraction_complexity.score >= 0.40
                )
            )
        )
        sufficiency["deep_audit_required"] = deep_audit_required

        # Layer 4: rule priors expose the cost-quality trade-off. They are
        # designed for later replacement by a calibrated model trained on
        # tool_marginal_benefit traces.
        utilities = {
            "neo4j_rag": self._utility(
                0.30 + 0.50 * linking_uncertainty,
                latency_cost=0.12, monetary_cost=0.01, safety_risk=0.03,
                threshold=0.20,
                basis=("retrieval_score_distribution", "verified_linking_ambiguity"),
            ),
            "second_llm_refiner": self._utility(
                min(0.82, 0.38 + 0.12 * len(reviewable)),
                latency_cost=0.18, monetary_cost=0.08, safety_risk=0.08,
                threshold=0.15,
                basis=("grounded_import_ready_semantic_risk_candidates",),
            ),
            "relation_recovery": self._utility(
                min(0.80, 0.42 + 0.10 * recovery_candidate_count),
                latency_cost=0.18, monetary_cost=0.08, safety_risk=0.12,
                threshold=0.18,
                basis=("explicit_trigger_but_no_primary_relation",),
            ),
            "causal_reasoner": self._utility(
                min(0.65, 0.24 + 0.10 * len(import_ready)),
                latency_cost=0.04, safety_risk=0.08, threshold=0.15,
                basis=("validated_causal_posture", "import_ready_relations"),
            ),
            "conflict_resolver": self._utility(
                min(0.85, 0.35 + 0.16 * len(existing_graph_relations)),
                latency_cost=0.02, safety_risk=0.02, threshold=0.15,
                basis=("existing_or_contradicting_graph_relation",),
            ),
            "debug_reviewer": self._utility(
                0.12, latency_cost=0.20, monetary_cost=0.08,
                threshold=0.20, basis=("debug_only",),
            ),
        }
        calls = {
            tool: bool(
                candidate_pool.get(tool)
                and not hard_masks.get(tool)
                and utilities[tool]["positive"]
            )
            for tool in utilities
        }
        rag_call = calls["neo4j_rag"]
        llm_call = calls["second_llm_refiner"]
        recovery_call = calls["relation_recovery"]
        causal_call = calls["causal_reasoner"]
        conflict_call = calls["conflict_resolver"]
        if extraction_failed:
            route = "RECOVERY"
        elif llm_call or recovery_call:
            route = "DEEP"
        elif rag_call:
            # Aggregate route remains a compatibility label; a single bounded
            # linking lookup does not make the full article a DEEP workflow.
            route = shadow_pre_plan.route
        elif deep_audit_required:
            # DEEP is an audit/attention label, not a fixed tool bundle.  The
            # tool gates below may still call no remote service.
            route = "DEEP"
        elif causal_call or conflict_call:
            route = "STANDARD"
        else:
            # Post-extraction evidence has more authority than the coarse
            # pre-route. No expensive tool value means STANDARD, except for
            # evidence-policy FAST articles.
            route = "FAST" if shadow_pre_plan.route == "FAST" else "STANDARD"
        plan = ToolPlan(
            stage="post_verification", route=route, profile=profile,
            decisions=dict(shadow_pre_plan.decisions),
            reason_codes=list(shadow_pre_plan.reason_codes), plan_status="shadow",
            legacy_route=legacy_post_plan.route,
            early_stop_reasons=list(shadow_pre_plan.early_stop_reasons),
            routing_version=self.ROUTING_VERSION,
        )
        plan.layer_trace = {
            "layer_1_safety_mask": {
                "invariants": list(self.SAFETY_INVARIANTS),
                "masked_tools": hard_masks,
            },
            "layer_2_candidate_pool": {"tools": candidate_pool},
            "layer_3_sufficiency_judgment": sufficiency,
            "layer_4_net_utility": {
                "utilities": utilities,
                "calls": calls,
                "policy": "call_only_if_unmasked_candidate_and_net_utility_meets_threshold",
            },
        }
        def decide(tool: str, decision: str, reason: str, cost: str,
                   value: float, budget: str = "minimal") -> None:
            plan.decisions[tool] = ToolDecision(
                tool, decision, reason, cost, value, budget, "shadow",
                bool(hard_masks.get(tool)), bool(candidate_pool.get(tool)),
                utilities.get(tool, {}),
            )
        decide(
            "neo4j_rag", CALL if rag_call else SKIP,
            "verified endpoint ambiguity requires graph linking hints" if rag_call
            else (
                hard_masks.get("neo4j_rag")
                or ("no verified linking/type ambiguity" if not candidate_pool["neo4j_rag"]
                    else "net utility below call threshold")
            ),
            "medium", utilities["neo4j_rag"]["net_utility"], "standard" if rag_call else "minimal",
        )
        decide(
            "second_llm_refiner", CALL if llm_call else SKIP,
            "bounded semantic candidates can change a decision" if llm_call
            else (
                hard_masks.get("second_llm_refiner")
                or ("no grounded import-ready semantic-risk candidate"
                    if not candidate_pool["second_llm_refiner"]
                    else "net utility below call threshold")
            ),
            "high", utilities["second_llm_refiner"]["net_utility"], "standard" if llm_call else "minimal",
        )
        decide(
            "relation_recovery", CALL if recovery_call else SKIP,
            "explicit preconstructed relation gap after deterministic blocking" if recovery_call
            else (
                hard_masks.get("relation_recovery")
                or ("no bounded extraction gap with explicit trigger"
                    if not candidate_pool["relation_recovery"]
                    else "net utility below call threshold")
            ),
            "high", utilities["relation_recovery"]["net_utility"], "standard" if recovery_call else "minimal",
        )
        decide(
            "causal_reasoner", CALL if causal_call else SKIP,
            "validated causal posture and import-ready relations" if causal_call
            else hard_masks.get("causal_reasoner", "causal evidence is insufficient or low value"),
            "low", utilities["causal_reasoner"]["net_utility"],
        )
        decide(
            "conflict_resolver", CALL if conflict_call else SKIP,
            "existing graph relation requires conflict policy" if conflict_call
            else hard_masks.get("conflict_resolver", "all graph-eligible relations are novel"),
            "low", utilities["conflict_resolver"]["net_utility"],
        )
        review_call = calls["debug_reviewer"]
        decide(
            "debug_reviewer", CALL if review_call else SKIP,
            "debug difficult shadow route" if review_call else "reviewer disabled or low-value",
            "high", utilities["debug_reviewer"]["net_utility"],
        )
        if relations and not reviewable and not import_ready and all_hard_blocked:
            plan.early_stop_reasons.append("all_relations_deterministically_hard_blocked")
        if relations and len(import_ready) == len(relations):
            plan.early_stop_reasons.append("all_relations_deterministically_import_ready")
        return plan

    @staticmethod
    def compare_plans(legacy: ToolPlan, shadow: ToolPlan) -> dict:
        tools = sorted(set(legacy.decisions) | set(shadow.decisions))
        differences = []
        for tool in tools:
            old = legacy.decisions.get(tool)
            new = shadow.decisions.get(tool)
            old_decision = old.decision if old else "MISSING"
            new_decision = new.decision if new else "MISSING"
            if old_decision != new_decision:
                differences.append({
                    "tool": tool, "legacy_decision": old_decision,
                    "shadow_decision": new_decision,
                    "shadow_reason": new.reason if new else "not planned",
                })
        return {
            "legacy_route": legacy.route,
            "shadow_route": shadow.route,
            "route_changed": legacy.route != shadow.route,
            "tool_differences": differences,
            "production_plan_unchanged": True,
        }

    @staticmethod
    def prompt_guidance(plan: ToolPlan) -> str:
        profile = plan.profile
        if profile.study_type == "review":
            instruction = (
                "This is a review-like article. Prefer relations asserted as review findings; "
                "do not turn cited background statements into article-level discoveries."
            )
        elif profile.study_type == "computational":
            instruction = (
                "This is a computational/prediction-heavy article. Keep predicted or docking "
                "associations uncertain unless the same abstract reports experimental validation."
            )
        elif profile.study_type in {"clinical", "human_omics"}:
            instruction = (
                "Separate cohort association/prognosis from molecular causation and preserve "
                "the exact Results/Conclusion evidence span."
            )
        elif profile.study_type in {"animal", "in_vitro"}:
            instruction = (
                "Preserve the experimental context; do not silently generalize animal or "
                "in-vitro effects to human disease."
            )
        else:
            instruction = "Extract only article-supported claims with exact contiguous evidence."
        return (
            "Article tool-router guidance:\n"
            f"- route: {plan.route}\n"
            f"- study_type: {profile.study_type}\n"
            f"- instruction: {instruction}\n"
        )
