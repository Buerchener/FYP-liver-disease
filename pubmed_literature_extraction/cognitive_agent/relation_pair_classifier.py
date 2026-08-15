#!/usr/bin/env python3
"""BioRED-style document relation classification over grounded entity pairs.

LangExtract remains responsible for entity discovery and span alignment.  This
module constructs a bounded, evidence-local entity-pair lattice and classifies
each pair into one project predicate or ``NO_RELATION``.  The deterministic
backend is deliberately conservative: it is a safe fallback and a shadow-mode
baseline, not a claim to reproduce the BioRED PubMedBERT model.

The public result is backend-neutral so that a calibrated BioRED/BioREx model
can be plugged in without changing the Agent, verifier, or DeepSeek routing.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Protocol

from cognitive_agent.abbreviation_detector import AbbreviationDetector
from cognitive_agent.evidence_units import EvidenceUnit
from cognitive_agent.evidence_selector import EvidenceSelector
from cognitive_agent.extraction_quality import normalize_surface
from cognitive_agent.rule_memory import RuleMatch, RuleMemory
from cognitive_agent.schema.relation_signatures import RELATION_SIGNATURES


NO_RELATION = "NO_RELATION"
RESULT_SECTIONS = frozenset({"RESULT", "RESULTS", "CONCLUSION", "CONCLUSIONS", "DISCUSSION"})
BACKGROUND_SECTIONS = frozenset({"BACKGROUND", "INTRODUCTION", "OBJECTIVE", "OBJECTIVES", "AIM", "AIMS", "PURPOSE"})

# Predicate definitions serve two purposes: local fallback classification and
# ontology-constrained predicate retrieval.  The latter avoids asking a model
# to score every relation label for every entity pair.
PREDICATE_PATTERNS: dict[str, tuple[str, ...]] = {
    "ENCODES": (r"\bencod(?:e|es|ed|ing)\b", r"\bprotein product\b"),
    "PROGNOSTIC_IN": (
        r"\bprognos(?:is|tic)\b", r"\bsurvival\b", r"\brecurren(?:ce|t)\b",
        r"\bpredict(?:s|ed|ive|or)?\b.{0,45}\b(?:outcome|mortality|survival)\b",
    ),
    "PROGRESSES_TO": (r"\bprogress(?:es|ed|ion)?\s+(?:in)?to\b", r"\bevolv(?:e|es|ed)\s+into\b"),
    "INTERACTS_WITH": (
        r"\binteract(?:s|ed|ion)?\s+with\b", r"\bbind(?:s|ing|bound)?\s+(?:to|with)\b",
        r"\bcomplex(?:es)?\s+with\b",
    ),
    "PARTICIPATES_IN": (
        r"\bparticipat(?:e|es|ed|ing)\s+in\b", r"\b(?:regulat|mediat|activat|inhibit)(?:e|es|ed|ing|ion)?\b",
        r"\b(?:component|member)\s+of\b",
    ),
    "EXPRESSED_IN": (
        r"\bexpress(?:ed|ion|es|ing)?\s+(?:in|by|within)\b", r"\blocali[sz](?:e|ed|ation)\s+(?:in|to)\b",
    ),
    "ASSOCIATED_WITH_METABOLITE": (
        r"\b(?:metabolic|metabolite)\s+association\b", r"\bassociated\s+with\b",
        r"\bcorrelat(?:e|es|ed|ion)\s+with\b",
    ),
    "ASSOCIATED_WITH": (
        r"\bassociated\s+with\b", r"\bassociation\s+(?:between|with)\b",
        r"\bcorrelat(?:e|es|ed|ion)\s+with\b", r"\blinked\s+to\b",
        r"\brelated\s+to\b", r"\b(?:increase|decrease)d?\b.{0,35}\bin\b",
    ),
}

NEGATION_RE = re.compile(r"\b(?:no|not|neither|without|failed to|did not)\b", re.IGNORECASE)
HEDGE_RE = re.compile(r"\b(?:may|might|could|possibly|potential|suggests?|predicted|putative)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PairClassifierConfig:
    enabled: bool = True
    mode: str = "shadow"  # off | shadow | active
    backend: str = "deterministic"
    model_path: str = ""
    high_confidence_threshold: float = 0.78
    relation_threshold: float = 0.48
    uncertainty_floor: float = 0.35
    max_candidates: int = 64
    max_low_confidence_candidates: int = 12
    include_adjacent_units: bool = False


@dataclass
class RelationPairCandidate:
    candidate_id: str
    subject: str
    subject_type: str
    object: str
    object_type: str
    allowed_predicates: list[str]
    evidence: str
    evidence_unit_id: str
    evidence_section: str
    evidence_char_start: int
    evidence_char_end: int
    source_predicates: list[str] = field(default_factory=list)
    source_directions: list[str] = field(default_factory=list)
    same_sentence: bool = True
    endpoint_distance: int = -1
    evidence_confidence: float = 0.0
    evidence_entailment: str = "NOT_ENOUGH_INFORMATION"
    evidence_trigger_predicate: str = ""

    def to_dict(self) -> dict:
        return {key: value for key, value in self.__dict__.items()}


@dataclass
class PairPrediction:
    candidate_id: str
    label: str
    confidence: float
    relation_probability: float
    no_relation_probability: float
    margin: float
    direction: str = "unknown"
    backend: str = "deterministic"
    routed_to_llm: bool = False
    reason_codes: list[str] = field(default_factory=list)
    predicate_scores: dict[str, float] = field(default_factory=dict)
    evidence_confidence: float = 0.0
    rule_score_delta: float = 0.0
    rule_matches: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {key: value for key, value in self.__dict__.items()}


@dataclass
class PairClassificationResult:
    mode: str = "shadow"
    backend: str = "deterministic"
    candidates: list[RelationPairCandidate] = field(default_factory=list)
    predictions: list[PairPrediction] = field(default_factory=list)
    accepted_relations: list[dict] = field(default_factory=list)
    low_confidence_relations: list[dict] = field(default_factory=list)
    truncated_candidates: int = 0
    fallback_reason: str = ""

    def to_dict(self) -> dict:
        relation_predictions = [p for p in self.predictions if p.label != NO_RELATION]
        return {
            "mode": self.mode,
            "backend": self.backend,
            "candidate_count": len(self.candidates),
            "positive_prediction_count": len(relation_predictions),
            "accepted_relation_count": len(self.accepted_relations),
            "low_confidence_count": len(self.low_confidence_relations),
            "no_relation_count": sum(p.label == NO_RELATION for p in self.predictions),
            "deepseek_routing_rate": round(
                len(self.low_confidence_relations) / max(len(self.candidates), 1), 4
            ),
            "truncated_candidates": self.truncated_candidates,
            "fallback_reason": self.fallback_reason,
            "candidates": [item.to_dict() for item in self.candidates],
            "predictions": [item.to_dict() for item in self.predictions],
        }


class PairPredictionBackend(Protocol):
    name: str

    def predict(self, candidate: RelationPairCandidate) -> PairPrediction: ...


class DeterministicPairBackend:
    """Auditable fallback with abstention; not a learned BioRED baseline."""

    name = "deterministic_evidence_pair_v1"

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _direction(text: str) -> str:
        lowered = text.casefold()
        if re.search(r"\b(?:decreas|reduc|downregulat|suppress|inhibit)\w*\b", lowered):
            return "decrease"
        if re.search(r"\b(?:increas|elevat|upregulat|enhanc|activat)\w*\b", lowered):
            return "increase"
        return "unknown"

    def predict(self, candidate: RelationPairCandidate) -> PairPrediction:
        evidence = candidate.evidence
        scores: dict[str, float] = {}
        reasons: list[str] = []
        for predicate in candidate.allowed_predicates:
            score = -1.35
            patterns = PREDICATE_PATTERNS.get(predicate, ())
            hits = sum(bool(re.search(pattern, evidence, re.IGNORECASE)) for pattern in patterns)
            if hits:
                score += 1.55 + min(hits - 1, 2) * 0.25
            if predicate in candidate.source_predicates:
                score += 0.72
            if candidate.evidence_section in RESULT_SECTIONS:
                score += 0.25
            elif candidate.evidence_section in BACKGROUND_SECTIONS:
                score -= 0.28
            if 0 <= candidate.endpoint_distance <= 120:
                score += 0.18
            elif candidate.endpoint_distance > 240:
                score -= 0.20
            if NEGATION_RE.search(evidence):
                score -= 1.25
            if HEDGE_RE.search(evidence):
                score -= 0.38
            # The broad label must not win from co-occurrence alone.
            if predicate == "ASSOCIATED_WITH" and not hits:
                score -= 0.25
            scores[predicate] = round(self._sigmoid(score), 6)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best_label, best_score = ranked[0] if ranked else (NO_RELATION, 0.0)
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        no_relation = max(0.03, min(0.97, 1.0 - best_score + (0.12 if not candidate.source_predicates else 0.0)))
        if best_score <= no_relation:
            label = NO_RELATION
            confidence = no_relation
        else:
            label = best_label
            confidence = best_score
        margin = abs(best_score - max(no_relation, second_score))
        if best_label in candidate.source_predicates:
            reasons.append("langextract_pair_hint")
        if any(re.search(pattern, evidence, re.IGNORECASE) for pattern in PREDICATE_PATTERNS.get(best_label, ())):
            reasons.append("explicit_predicate_trigger")
        if candidate.evidence_section in RESULT_SECTIONS:
            reasons.append("result_section")
        if NEGATION_RE.search(evidence):
            reasons.append("negation_signal")
        if HEDGE_RE.search(evidence):
            reasons.append("hedging_signal")
        if label == NO_RELATION:
            reasons.append("no_relation_wins")
        return PairPrediction(
            candidate_id=candidate.candidate_id,
            label=label,
            confidence=round(confidence, 6),
            relation_probability=round(best_score, 6),
            no_relation_probability=round(no_relation, 6),
            margin=round(margin, 6),
            direction=self._direction(evidence),
            backend=self.name,
            reason_codes=reasons,
            predicate_scores=dict(ranked),
        )


class SklearnPairBackend:
    """Small calibrated classifier trained on pair-level JSONL/BioRED rows.

    The artifact is a joblib dictionary containing a fitted sklearn pipeline
    under ``model``.  Imports are lazy, so the production fallback has no
    sklearn startup or memory cost.
    """

    name = "sklearn_calibrated_pair_v1"

    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError("sklearn pair backend requires model_path")
        import joblib

        artifact = joblib.load(model_path)
        self.model = artifact["model"] if isinstance(artifact, dict) else artifact
        self.metadata = artifact.get("metadata", {}) if isinstance(artifact, dict) else {}

    @staticmethod
    def feature_text(candidate: RelationPairCandidate) -> str:
        hints = " ".join(candidate.source_predicates) or "NONE"
        return (
            f"SUBJECT_TYPE={candidate.subject_type} OBJECT_TYPE={candidate.object_type} "
            f"SECTION={candidate.evidence_section} HINT={hints} "
            f"[SUBJECT] {candidate.subject} [/SUBJECT] "
            f"[OBJECT] {candidate.object} [/OBJECT] TEXT {candidate.evidence}"
        )

    def _prediction(self, candidate: RelationPairCandidate, probabilities) -> PairPrediction:
        classes = [str(value) for value in self.model.classes_]
        raw = {label: float(score) for label, score in zip(classes, probabilities)}
        allowed_scores = {
            label: raw.get(label, 0.0) for label in candidate.allowed_predicates
        }
        ranked = sorted(allowed_scores.items(), key=lambda item: (-item[1], item[0]))
        best_label, relation_probability = ranked[0] if ranked else (NO_RELATION, 0.0)
        no_relation_probability = raw.get(NO_RELATION, max(0.0, 1.0 - relation_probability))
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if no_relation_probability >= relation_probability:
            label, confidence = NO_RELATION, no_relation_probability
        else:
            label, confidence = best_label, relation_probability
        return PairPrediction(
            candidate_id=candidate.candidate_id,
            label=label,
            confidence=round(confidence, 6),
            relation_probability=round(relation_probability, 6),
            no_relation_probability=round(no_relation_probability, 6),
            margin=round(abs(relation_probability - max(no_relation_probability, second)), 6),
            backend=self.name,
            direction=DeterministicPairBackend._direction(candidate.evidence),
            reason_codes=["calibrated_pair_model", "ontology_label_mask"],
            predicate_scores={key: round(value, 6) for key, value in ranked},
        )

    def predict(self, candidate: RelationPairCandidate) -> PairPrediction:
        probabilities = self.model.predict_proba([self.feature_text(candidate)])[0]
        return self._prediction(candidate, probabilities)

    def predict_many(self, candidates: list[RelationPairCandidate]) -> list[PairPrediction]:
        if not candidates:
            return []
        matrix = self.model.predict_proba([self.feature_text(item) for item in candidates])
        return [self._prediction(candidate, row) for candidate, row in zip(candidates, matrix)]


def build_pair_backend(config: PairClassifierConfig) -> PairPredictionBackend:
    if config.backend == "deterministic":
        return DeterministicPairBackend()
    if config.backend == "sklearn":
        return SklearnPairBackend(config.model_path)
    raise ValueError(f"unsupported pair classifier backend: {config.backend}")


class BioREDPairClassifier:
    """Evidence-local candidate generator plus calibrated/abstaining backend."""

    def __init__(
        self,
        config: PairClassifierConfig | None = None,
        backend: PairPredictionBackend | None = None,
        evidence_selector: EvidenceSelector | None = None,
        rule_memory: RuleMemory | None = None,
        evidence_selector_enabled: bool = True,
    ):
        self.config = config or PairClassifierConfig()
        self.backend = backend or build_pair_backend(self.config)
        self.evidence_selector = evidence_selector or EvidenceSelector()
        self.evidence_selector_enabled = bool(evidence_selector_enabled)
        self.rule_memory = rule_memory
        if self.config.mode not in {"off", "shadow", "active"}:
            raise ValueError("pair classifier mode must be off, shadow, or active")

    @staticmethod
    def _mentions(entity: dict) -> list[str]:
        values = [entity.get("mention", ""), *(entity.get("canonical_mentions", []) or [])]
        return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))

    @staticmethod
    def _mention_span(unit: EvidenceUnit, entity: dict) -> tuple[int, int] | None:
        matches: list[tuple[int, int]] = []
        for mention in BioREDPairClassifier._mentions(entity):
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(mention) + r"(?![A-Za-z0-9])", re.IGNORECASE)
            match = pattern.search(unit.text)
            if match:
                matches.append((match.start(), match.end()))
        return min(matches, key=lambda value: value[0], default=None)

    @staticmethod
    def _allowed(subject_type: str, object_type: str) -> list[str]:
        return sorted(
            predicate for predicate, pairs in RELATION_SIGNATURES.items()
            if (subject_type, object_type) in pairs
        )

    @staticmethod
    def _hint_index(relations: list[dict]) -> dict[tuple[str, str, str, str], list[dict]]:
        output: dict[tuple[str, str, str, str], list[dict]] = {}
        for rel in relations:
            key = (
                normalize_surface(rel.get("subject", "")), str(rel.get("subject_type", "")),
                normalize_surface(rel.get("object", "")), str(rel.get("object_type", "")),
            )
            output.setdefault(key, []).append(rel)
        return output

    def build_candidates(
        self,
        entities: list[dict],
        relations: list[dict],
        units: list[EvidenceUnit],
        source_text: str = "",
    ) -> tuple[list[RelationPairCandidate], int]:
        hints = self._hint_index(relations)
        # Pairing must see article-local abbreviations. Otherwise an entity
        # discovered at its long-form mention cannot pair with its short form
        # in a later result sentence (Icaritin/ICT, PBC, T2DM, and similar).
        # This is deterministic source-derived alias expansion, not entity
        # generation.
        abbreviation_map = AbbreviationDetector().detect(source_text) if source_text else None
        pairing_entities: list[dict] = []
        for entity in entities:
            enriched = dict(entity)
            mentions = list(enriched.get("canonical_mentions", []) or [])
            mention = str(enriched.get("mention", "") or "")
            if abbreviation_map and mention:
                mentions.extend([
                    abbreviation_map.resolve_to_long(mention),
                    abbreviation_map.resolve_to_short(mention),
                ])
            enriched["canonical_mentions"] = list(dict.fromkeys(
                str(value).strip() for value in mentions
                if str(value or "").strip()
                and str(value).strip().casefold() != mention.casefold()
            ))
            pairing_entities.append(enriched)
        candidates: list[RelationPairCandidate] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for unit in units:
            local: list[tuple[dict, tuple[int, int]]] = []
            for entity in pairing_entities:
                span = self._mention_span(unit, entity)
                if span:
                    local.append((entity, span))
            for subject, subject_span in local:
                for obj, object_span in local:
                    if subject is obj:
                        continue
                    subject_type = str(subject.get("type", subject.get("entity_type", "")) or "")
                    object_type = str(obj.get("type", obj.get("entity_type", "")) or "")
                    allowed = self._allowed(subject_type, object_type)
                    if not allowed:
                        continue
                    key = (
                        normalize_surface(subject.get("mention", "")), subject_type,
                        normalize_surface(obj.get("mention", "")), object_type, unit.unit_id,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    pair_hints = hints.get(key[:4], [])
                    digest = hashlib.sha1("|".join(key).encode("utf-8")).hexdigest()[:10]
                    evidence_text = unit.text
                    evidence_start = unit.char_start
                    evidence_end = unit.char_end
                    evidence_confidence = 0.0
                    evidence_entailment = "NOT_ENOUGH_INFORMATION"
                    evidence_trigger_predicate = ""
                    if source_text and self.evidence_selector_enabled:
                        selections = []
                        for predicate in allowed:
                            selected = self.evidence_selector.select(
                                candidate_id=f"p-{digest}",
                                subject_mentions=self._mentions(subject),
                                object_mentions=self._mentions(obj),
                                predicate=predicate, units=[unit], source_text=source_text,
                            )
                            if selected:
                                selections.append((
                                    selected.trigger_span is None,
                                    selected.section not in RESULT_SECTIONS,
                                    len(selected.text), predicate, selected,
                                ))
                        if selections:
                            _, _, _, evidence_trigger_predicate, selected = min(selections)
                            evidence_text = selected.text
                            evidence_start = selected.char_start
                            evidence_end = selected.char_end
                            evidence_confidence = selected.evidence_confidence
                            evidence_entailment = selected.local_label
                    candidates.append(RelationPairCandidate(
                        candidate_id=f"p-{digest}",
                        subject=str(subject.get("mention", "")), subject_type=subject_type,
                        object=str(obj.get("mention", "")), object_type=object_type,
                        allowed_predicates=allowed,
                        evidence=evidence_text,
                        evidence_unit_id=unit.unit_id,
                        evidence_section=unit.section,
                        evidence_char_start=evidence_start,
                        evidence_char_end=evidence_end,
                        source_predicates=list(dict.fromkeys(
                            str(item.get("predicate", "")).upper() for item in pair_hints
                            if str(item.get("predicate", "")).upper() in allowed
                        )),
                        source_directions=list(dict.fromkeys(
                            str(item.get("direction", "unknown")) for item in pair_hints
                        )),
                        endpoint_distance=max(
                            0, max(subject_span[0], object_span[0]) - min(subject_span[1], object_span[1])
                        ),
                        evidence_confidence=evidence_confidence,
                        evidence_entailment=evidence_entailment,
                        evidence_trigger_predicate=evidence_trigger_predicate,
                    ))

        # Evidence-local pairs with a LangExtract hint are most valuable, then
        # result/conclusion assertions, then close pairs.  This bounds N^2.
        candidates.sort(key=lambda item: (
            not bool(item.source_predicates),
            item.evidence_section not in RESULT_SECTIONS,
            not bool(item.evidence_trigger_predicate),
            -float(item.evidence_confidence or 0.0),
            item.endpoint_distance if item.endpoint_distance >= 0 else 10_000,
            item.candidate_id,
        ))
        truncated = max(0, len(candidates) - self.config.max_candidates)
        return candidates[: self.config.max_candidates], truncated

    @staticmethod
    def _as_relation(candidate: RelationPairCandidate, prediction: PairPrediction) -> dict:
        flags = ["pair_classifier_candidate"]
        if prediction.routed_to_llm:
            flags.extend(["pair_low_confidence", "manual_review"])
        if len([value for value in prediction.predicate_scores.values() if value >= 0.45]) > 1:
            flags.append("pair_ambiguous_predicate")
        return {
            "subject": candidate.subject,
            "subject_type": candidate.subject_type,
            "predicate": prediction.label,
            "object": candidate.object,
            "object_type": candidate.object_type,
            "direction": prediction.direction,
            "negated": bool(NEGATION_RE.search(candidate.evidence)),
            "uncertain": bool(HEDGE_RE.search(candidate.evidence)),
            "evidence": candidate.evidence,
            "evidence_unit_id": candidate.evidence_unit_id,
            "evidence_role": candidate.evidence_section,
            "candidate_id": candidate.candidate_id,
            "classifier_source": prediction.backend,
            "classifier_confidence": prediction.confidence,
            "relation_probability": prediction.relation_probability,
            "no_relation_probability": prediction.no_relation_probability,
            "classifier_margin": prediction.margin,
            "evidence_confidence": prediction.evidence_confidence,
            "evidence_entailment": candidate.evidence_entailment,
            "evidence_char_start": candidate.evidence_char_start,
            "evidence_char_end": candidate.evidence_char_end,
            "rule_score_delta": prediction.rule_score_delta,
            "rule_matches": prediction.rule_matches,
            "predicate_candidates": prediction.predicate_scores,
            "quality_flags": sorted(set(flags)),
        }

    def _apply_rule_priors(
        self, candidate: RelationPairCandidate, prediction: PairPrediction,
    ) -> PairPrediction:
        prediction.evidence_confidence = candidate.evidence_confidence
        if not self.rule_memory or self.rule_memory.mode == "off":
            return prediction
        adjusted: dict[str, float] = {}
        all_matches: dict[str, RuleMatch] = {}
        deltas: dict[str, float] = {}
        for predicate, score in prediction.predicate_scores.items():
            matches = self.rule_memory.retrieve({
                "predicate": predicate,
                "subject_type": candidate.subject_type,
                "object_type": candidate.object_type,
                "section": candidate.evidence_section,
                "evidence": candidate.evidence,
                "evidence_confidence": candidate.evidence_confidence,
                "both_endpoints_in_evidence": True,
            })
            delta = sum(
                float(item.value or 0.0) for item in matches
                if item.kind == "pair_prior" and item.action == "ADJUST_PAIR_SCORE"
            )
            delta = max(-0.20, min(0.20, delta))
            deltas[predicate] = delta
            adjusted[predicate] = round(max(0.0, min(1.0, score + delta)), 6)
            for item in matches:
                all_matches[item.rule_id] = item
        if adjusted and self.rule_memory.mode == "active":
            prediction.predicate_scores = dict(sorted(adjusted.items(), key=lambda item: (-item[1], item[0])))
            best_label, best_score = next(iter(prediction.predicate_scores.items()))
            prediction.relation_probability = best_score
            prediction.no_relation_probability = round(max(0.03, min(0.97, 1.0 - best_score)), 6)
            prediction.label = best_label if best_score > prediction.no_relation_probability else NO_RELATION
            prediction.confidence = max(best_score, prediction.no_relation_probability)
            prediction.rule_score_delta = round(deltas.get(best_label, 0.0), 6)
            if prediction.rule_score_delta:
                prediction.reason_codes.append("active_rule_prior")
        prediction.rule_matches = [all_matches[key].to_dict() for key in sorted(all_matches)]
        if self.rule_memory.mode == "active" and any(
            item.action == "REJECT" for item in all_matches.values()
        ):
            prediction.label = NO_RELATION
            prediction.reason_codes.append("active_rule_reject")
        if any(item.action in {"REVIEW", "ABSTAIN", "CALL_DEEPSEEK", "CALL_QWEN_CRITIC"}
               for item in all_matches.values()):
            prediction.routed_to_llm = True
            prediction.reason_codes.append("active_rule_route_or_downgrade")
        return prediction

    def classify(
        self,
        entities: list[dict],
        relations: list[dict],
        units: list[EvidenceUnit],
        source_text: str = "",
    ) -> PairClassificationResult:
        result = PairClassificationResult(mode=self.config.mode, backend=self.backend.name)
        if not self.config.enabled or self.config.mode == "off":
            result.fallback_reason = "pair_classifier_disabled"
            return result
        candidates, truncated = self.build_candidates(entities, relations, units, source_text)
        result.candidates = candidates
        result.truncated_candidates = truncated
        predictions = (
            self.backend.predict_many(candidates)
            if hasattr(self.backend, "predict_many")
            else [self.backend.predict(candidate) for candidate in candidates]
        )
        for candidate, prediction in zip(candidates, predictions):
            prediction = self._apply_rule_priors(candidate, prediction)
            plausible_relation = prediction.relation_probability >= self.config.uncertainty_floor
            prediction.routed_to_llm = prediction.routed_to_llm or bool(
                plausible_relation
                and (
                    prediction.confidence < self.config.high_confidence_threshold
                    or prediction.margin < 0.18
                )
            )
            result.predictions.append(prediction)
            if prediction.label == NO_RELATION:
                # CoRE also routes uncertain negative decisions.  Represent the
                # strongest ontology-compatible alternative as a provisional,
                # write-blocked relation; DeepSeek may KEEP/REJECT it, but it is
                # never accepted locally while NO_RELATION wins.
                if prediction.routed_to_llm and prediction.predicate_scores:
                    alternative = next(iter(prediction.predicate_scores))
                    provisional = PairPrediction(
                        **{
                            **prediction.__dict__,
                            "label": alternative,
                            "confidence": prediction.relation_probability,
                            "reason_codes": [
                                *prediction.reason_codes,
                                "no_relation_abstention_routed",
                            ],
                        }
                    )
                    relation = self._as_relation(candidate, provisional)
                    relation.setdefault("quality_flags", []).append(
                        "pair_no_relation_abstention"
                    )
                    relation["quality_flags"] = sorted(set(relation["quality_flags"]))
                    result.low_confidence_relations.append(relation)
                continue
            relation = self._as_relation(candidate, prediction)
            if prediction.relation_probability >= self.config.relation_threshold:
                result.accepted_relations.append(relation)
            if prediction.routed_to_llm:
                result.low_confidence_relations.append(relation)
        result.low_confidence_relations = result.low_confidence_relations[
            : self.config.max_low_confidence_candidates
        ]
        return result
