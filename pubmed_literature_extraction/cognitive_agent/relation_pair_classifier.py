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
from cognitive_agent.evidence_units import ArticleEvidenceReader, EvidenceUnit
from cognitive_agent.evidence_selector import EvidenceSelector
from cognitive_agent.extraction_quality import (
    article_quality_flags,
    locate_contiguous,
    normalize_surface,
    predicate_trigger_links_endpoints,
)
from cognitive_agent.rule_memory import RuleMatch, RuleMemory
from cognitive_agent.schema.relation_signatures import RELATION_SIGNATURES


NO_RELATION = "NO_RELATION"
RESULT_SECTIONS = frozenset({"RESULT", "RESULTS", "CONCLUSION", "CONCLUSIONS", "DISCUSSION"})
BACKGROUND_SECTIONS = frozenset({"BACKGROUND", "INTRODUCTION", "OBJECTIVE", "OBJECTIVES", "AIM", "AIMS", "PURPOSE"})
METHOD_SECTIONS = frozenset({"METHOD", "METHODS", "MATERIALS_AND_METHODS"})
SYMMETRIC_PREDICATES = frozenset({"ASSOCIATED_WITH", "INTERACTS_WITH"})

PRIOR_WORK_RE = re.compile(
    r"\b(?:previous|prior|earlier) (?:stud(?:y|ies)|work|research)|"
    r"\b(?:has|have) been (?:reported|shown|demonstrated)|\bis known to\b",
    re.IGNORECASE,
)
REVIEW_ARTICLE_RE = re.compile(
    r"\b(?:systematic |scoping |narrative )?review\b|\bthis review\b",
    re.IGNORECASE,
)
PREDICTION_ONLY_RE = re.compile(
    r"\b(?:in silico|bioinformatics|computational|molecular docking|"
    r"predicted?|prediction)\b",
    re.IGNORECASE,
)
DISTINCTIVE_DESCRIPTOR_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z][A-Za-z0-9-]{2,}$")
LIGHT_COREFERENCE_RE = re.compile(
    r"\b(?:these|those|such) (?:cells?|populations?|subsets?)\b|"
    r"\b(?:they|them|their)\b",
    re.IGNORECASE,
)

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
        r"\bcomplex(?:es)?\s+with\b", r"\bcross[- ]?talk\b",
        r"\bcell(?:ular)?[- ]cell communication\b", r"\bjuxtapos\w*\b",
    ),
    "PARTICIPATES_IN": (
        r"\bparticipat(?:e|es|ed|ing)\s+in\b", r"\b(?:regulat|mediat|activat|inhibit)(?:e|es|ed|ing|ion)?\b",
        r"\b(?:component|member)\s+of\b",
    ),
    "EXPRESSED_IN": (
        r"\bexpress(?:ed|ion|es|ing)?\s+(?:in|by|within)\b",
        r"\b(?:high|low)?\s*expression\s+of\b.{0,100}\b(?:in|within)\b",
        r"\bsource\s+of\b",
        r"\blocali[sz](?:e|ed|ation)\s+(?:in|to)\b",
    ),
    "ASSOCIATED_WITH_METABOLITE": (
        r"\b(?:metabolic|metabolite)\s+association\b", r"\bassociated\s+with\b",
        r"\bcorrelat(?:e|es|ed|ion)\s+with\b",
    ),
    "ASSOCIATED_WITH": (
        r"\bassociated\s+with\b", r"\bassociation\s+(?:between|with)\b",
        r"\bcorrelat(?:e|es|ed|ion)\s+with\b", r"\blinked\s+to\b",
        r"\bclosely\s+linked\s+to\b", r"\brelated\s+to\b",
        r"\b(?:increase|decrease)d?\b.{0,35}\bin\b",
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
    max_candidates: int = 128
    max_low_confidence_candidates: int = 24
    # High-recall candidate generation: pair entities that co-occur in the
    # same clause, the same sentence, or adjacent sentences.  A missing
    # trigger must never delete an ASSOCIATED_WITH-style candidate before the
    # downstream judge sees it.
    include_parent_sentences: bool = True
    include_adjacent_windows: bool = True
    adjacent_windows_require_trigger: bool = True
    max_incomplete_evidence_candidates: int = 12


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
    claim_role: str = "CURRENT_FINDING"
    quality_flags: list[str] = field(default_factory=list)

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
    out_of_scope_relations: list[dict] = field(default_factory=list)
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
            "out_of_scope_count": len(self.out_of_scope_relations),
            "no_relation_count": sum(p.label == NO_RELATION for p in self.predictions),
            "deepseek_routing_rate": round(
                len(self.low_confidence_relations) / max(len(self.candidates), 1), 4
            ),
            "truncated_candidates": self.truncated_candidates,
            "fallback_reason": self.fallback_reason,
            "candidates": [item.to_dict() for item in self.candidates],
            "predictions": [item.to_dict() for item in self.predictions],
            "out_of_scope_relations": list(self.out_of_scope_relations),
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
    def _distinctive_descriptor_aliases(entity: dict, source_text: str) -> list[str]:
        """Return conservative article-local subtype anchors such as ``Endo4``.

        Biomedical cell subsets are often introduced with a long descriptive
        name and referred to by a distinctive alphanumeric token in the next
        sentence (for example ``Endo4 liver endothelial cells`` -> ``Endo4
        marker``).  Only tokens that occur at least twice in the current source
        are admitted, so this cannot inject an article-external alias.
        """
        if not source_text or str(entity.get("type", "")) != "CellType":
            return []
        aliases: list[str] = []
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", str(entity.get("mention", ""))):
            if not DISTINCTIVE_DESCRIPTOR_RE.fullmatch(token):
                continue
            occurrences = re.findall(
                r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])",
                source_text,
                re.IGNORECASE,
            )
            if len(occurrences) >= 2:
                aliases.append(token)
        return aliases

    @classmethod
    def _deduplicate_pairing_entities(
        cls, entities: list[dict], abbreviation_map: object | None, source_text: str,
    ) -> list[dict]:
        """Collapse only same-type, source-derived alias families for pairing.

        This is deliberately narrower than entity linking: stable IDs and
        article-local abbreviation families may merge, but Gene/Protein views
        and unrelated same-surface types remain separate.
        """
        groups: dict[tuple[str, str], list[dict]] = {}
        order: list[tuple[str, str]] = []
        for original in entities:
            entity = dict(original)
            mention = str(entity.get("mention", "") or "").strip()
            entity_type = str(entity.get("type", entity.get("entity_type", "")) or "")
            if not mention or not entity_type:
                continue
            attrs = entity.get("attributes", {}) or {}
            normalized_id = str(
                attrs.get("normalized_id", entity.get("normalized_id", "")) or ""
            ).strip()
            canonical = (
                abbreviation_map.canonical_name(mention)
                if abbreviation_map is not None else mention
            )
            family = f"id:{normalized_id.casefold()}" if normalized_id else (
                f"name:{normalize_surface(canonical)}"
            )
            key = (entity_type, family)
            if key not in groups:
                groups[key] = []
                order.append(key)
            aliases = [mention, *(entity.get("canonical_mentions", []) or [])]
            if abbreviation_map is not None:
                aliases.extend([
                    abbreviation_map.resolve_to_long(mention),
                    abbreviation_map.resolve_to_short(mention),
                ])
            aliases.extend(cls._distinctive_descriptor_aliases(entity, source_text))
            entity["canonical_mentions"] = list(dict.fromkeys(
                str(value).strip() for value in aliases if str(value or "").strip()
            ))
            groups[key].append(entity)

        output: list[dict] = []
        for key in order:
            variants = groups[key]
            winner = max(variants, key=lambda item: (
                len(str(item.get("mention", "")).split()),
                len(str(item.get("mention", ""))),
                bool(item.get("grounded", False)),
            ))
            merged = dict(winner)
            merged["canonical_mentions"] = list(dict.fromkeys(
                alias
                for item in variants
                for alias in cls._mentions(item)
                if alias.casefold() != str(winner.get("mention", "")).casefold()
            ))
            output.append(merged)
        return output

    @staticmethod
    def _infer_claim_role(evidence: str, section: str, source_text: str) -> str:
        if section in METHOD_SECTIONS:
            return "METHOD"
        if PRIOR_WORK_RE.search(evidence):
            return "PRIOR_WORK"
        if PREDICTION_ONLY_RE.search(evidence):
            return "PREDICTION"
        if section in BACKGROUND_SECTIONS or REVIEW_ARTICLE_RE.search(source_text):
            return "BACKGROUND"
        return "CURRENT_FINDING"

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

    @staticmethod
    def _pairing_windows(
        source_text: str, units: list[EvidenceUnit], config: "PairClassifierConfig",
    ) -> list[EvidenceUnit]:
        """Clause units plus lossless sentence and adjacent-sentence windows.

        The lattice is a high-recall candidate generator: pairing windows are
        widened (clause -> sentence -> adjacent sentences) so that coordinated
        endpoints split by clause boundaries or sentence boundaries still form
        a candidate.  Every window remains an exact source substring, so the
        evidence contract is untouched.
        """
        windows: list[EvidenceUnit] = list(units or [])
        parents: list[EvidenceUnit] = []
        # Parent/adjacent windows are exact source substrings; without the
        # source text they would be empty and must never replace clauses.
        if source_text and config.include_parent_sentences:
            parents = ArticleEvidenceReader.parent_units(source_text, windows)
            windows.extend(parents)
        if source_text and config.include_adjacent_windows and parents:
            windows.extend(
                ArticleEvidenceReader.adjacent_sentence_windows(source_text, parents)
            )
        unique: dict[tuple[int, int], EvidenceUnit] = {}
        for window in windows:
            key = (window.char_start, window.char_end)
            previous = unique.get(key)
            if previous is None or len(window.text) < len(previous.text):
                unique[key] = window
        return sorted(unique.values(), key=lambda item: (item.char_start, item.char_end))

    @staticmethod
    def _has_explicit_relation_trigger(text: str) -> bool:
        """Whether a widened cross-sentence window contains a schema cue."""
        return any(
            re.search(pattern, text or "", re.IGNORECASE)
            for patterns in PREDICATE_PATTERNS.values()
            for pattern in patterns
        )

    def build_candidates(
        self,
        entities: list[dict],
        relations: list[dict],
        units: list[EvidenceUnit],
        source_text: str = "",
    ) -> tuple[list[RelationPairCandidate], int]:
        # Pairing must see article-local abbreviations. Otherwise an entity
        # discovered at its long-form mention cannot pair with its short form
        # in a later result sentence (Icaritin/ICT, PBC, T2DM, and similar).
        # This is deterministic source-derived alias expansion, not entity
        # generation.
        abbreviation_map = AbbreviationDetector().detect(source_text) if source_text else None
        pairing_entities = self._deduplicate_pairing_entities(
            entities, abbreviation_map, source_text,
        )
        entity_by_alias_type: dict[tuple[str, str], dict] = {}
        for entity in pairing_entities:
            entity_type = str(entity.get("type", entity.get("entity_type", "")) or "")
            for alias in self._mentions(entity):
                entity_by_alias_type.setdefault(
                    (normalize_surface(alias), entity_type), entity,
                )

        # Canonicalise extractor hints through the same article-local alias
        # registry.  Otherwise an HCC hint would not reach a pair represented
        # by its long form after abbreviation deduplication.
        hints: dict[tuple[str, str, str, str], list[dict]] = {}
        for relation in relations:
            subject_type = str(relation.get("subject_type", "") or "")
            object_type = str(relation.get("object_type", "") or "")
            subject = entity_by_alias_type.get((
                normalize_surface(relation.get("subject", "")), subject_type,
            ))
            obj = entity_by_alias_type.get((
                normalize_surface(relation.get("object", "")), object_type,
            ))
            if subject is None or obj is None:
                continue
            key = (
                normalize_surface(subject.get("mention", "")), subject_type,
                normalize_surface(obj.get("mention", "")), object_type,
            )
            hints.setdefault(key, []).append(relation)

        windows = self._pairing_windows(source_text, units, self.config)
        article_flags = article_quality_flags(source_text)

        # Pass 1: enumerate schema-compatible pairs per window.  The type
        # signature mask stays a hard constraint; co-occurrence alone is
        # enough to form a candidate.  No trigger is required here.
        pair_rows: dict[tuple[str, str, str, str], dict] = {}
        for window in windows:
            if (
                self.config.adjacent_windows_require_trigger
                and window.unit_id.startswith("w")
                and not self._has_explicit_relation_trigger(window.text)
            ):
                continue
            local: list[tuple[dict, tuple[int, int]]] = []
            for entity in pairing_entities:
                span = self._mention_span(window, entity)
                if span:
                    local.append((entity, span))
            local.sort(key=lambda item: (
                item[1][0], item[1][1], str(item[0].get("mention", "")),
            ))
            for subject, subject_span in local:
                for obj, object_span in local:
                    if subject is obj:
                        continue
                    if max(subject_span[0], object_span[0]) < min(
                        subject_span[1], object_span[1]
                    ):
                        # One textual mention cannot supply both endpoints
                        # (for example Nrf2 inside "Nrf2 pathway").  Treating
                        # the overlap as a pair creates tautological pseudo-
                        # relations and inflates the lattice.
                        continue
                    subject_type = str(subject.get("type", subject.get("entity_type", "")) or "")
                    object_type = str(obj.get("type", obj.get("entity_type", "")) or "")
                    allowed = self._allowed(subject_type, object_type)
                    if not allowed:
                        continue
                    pair_key = (
                        normalize_surface(subject.get("mention", "")), subject_type,
                        normalize_surface(obj.get("mention", "")), object_type,
                    )
                    pair_hints = hints.get(pair_key, [])
                    reverse_key = (
                        normalize_surface(obj.get("mention", "")), object_type,
                        normalize_surface(subject.get("mention", "")), subject_type,
                    )
                    # Purely symmetric type signatures need one candidate, not
                    # A->B and B->A duplicates.  Mixed signatures that include
                    # a directional predicate (notably Disease->Disease with
                    # PROGRESSES_TO) retain both orientations.
                    if set(allowed) <= SYMMETRIC_PREDICATES and reverse_key in pair_rows:
                        pair_rows[reverse_key]["windows"].append(
                            (window, object_span, subject_span)
                        )
                        continue
                    row = pair_rows.get(pair_key)
                    if row is None:
                        digest = hashlib.sha1("|".join(pair_key).encode("utf-8")).hexdigest()[:10]
                        pair_rows[pair_key] = {
                            "candidate_id": f"p-{digest}",
                            "subject": subject,
                            "subject_type": subject_type,
                            "object": obj,
                            "object_type": object_type,
                            "allowed": allowed,
                            "pair_hints": pair_hints,
                            "windows": [],
                        }
                        row = pair_rows[pair_key]
                    row["windows"].append((window, subject_span, object_span))

        # Pass 2: choose the best pairing window per pair and run the
        # minimal-span EvidenceSelector over it best-effort.  A missing
        # trigger downgrades ranking but never deletes the candidate.
        candidates: list[RelationPairCandidate] = []
        for pair_key in sorted(pair_rows):
            row = pair_rows[pair_key]
            pair_windows = row["windows"]
            pair_windows.sort(key=lambda item: (
                item[0].section not in RESULT_SECTIONS,
                len(item[0].text),
                item[0].char_start,
            ))
            window, subject_span, object_span = pair_windows[0]
            subject, obj = row["subject"], row["object"]
            evidence_text = window.text
            evidence_unit_id = window.unit_id
            evidence_section = window.section
            evidence_start = window.char_start
            evidence_end = window.char_end
            evidence_confidence = 0.0
            evidence_entailment = "NOT_ENOUGH_INFORMATION"
            evidence_trigger_predicate = ""
            evidence_reason_codes: list[str] = []
            selectable = [
                item[0] for item in pair_windows
                if BioREDPairClassifier._mention_span(item[0], subject)
                and BioREDPairClassifier._mention_span(item[0], obj)
            ]
            if source_text and self.evidence_selector_enabled:
                other_mentions = [
                    mention
                    for entity in pairing_entities
                    if entity is not subject and entity is not obj
                    for mention in self._mentions(entity)
                ]
                selections = []
                for predicate in row["allowed"]:
                    selected = self.evidence_selector.select(
                        candidate_id=row["candidate_id"],
                        subject_mentions=self._mentions(subject),
                        object_mentions=self._mentions(obj),
                        predicate=predicate, units=selectable or [window],
                        source_text=source_text,
                        other_mentions=other_mentions,
                    )
                    if selected:
                        selections.append((
                            selected.trigger_span is None,
                            selected.section not in RESULT_SECTIONS,
                            len(selected.text), predicate, selected,
                        ))
                if selections:
                    _, _, _, selected_predicate, selected = min(selections)
                    evidence_trigger_predicate = (
                        selected_predicate if selected.trigger_span is not None else ""
                    )
                    evidence_text = selected.text
                    evidence_start = selected.char_start
                    evidence_end = selected.char_end
                    evidence_confidence = selected.evidence_confidence
                    evidence_entailment = selected.local_label
                    evidence_reason_codes = list(selected.reason_codes)
                    if "trigger_attachment_ambiguous" in selected.reason_codes:
                        evidence_trigger_predicate = ""
            # A grounded extractor hint that already contains both endpoints
            # and a pair-linking trigger is stronger than a shorter generic
            # window.  Preserve it instead of letting minimal-span selection
            # accidentally borrow a nearby third entity's relation wording.
            exact_hints: list[tuple[bool, int, int, dict, EvidenceUnit | None]] = []
            for hint in row["pair_hints"]:
                predicate = str(hint.get("predicate", "") or "").upper()
                evidence = str(hint.get("evidence", "") or "").strip()
                grounded, start, end = locate_contiguous(evidence, source_text)
                if not grounded or predicate not in row["allowed"]:
                    continue
                if not predicate_trigger_links_endpoints(
                    predicate, evidence, self._mentions(subject), self._mentions(obj),
                ):
                    continue
                container = ArticleEvidenceReader.containing_unit(evidence, windows)
                section = container.section if container else window.section
                exact_hints.append((
                    section not in RESULT_SECTIONS, len(evidence), start, hint, container,
                ))
            if exact_hints:
                _, _, hint_start, hint, container = min(exact_hints)
                evidence_text = str(hint.get("evidence", "") or "").strip()
                evidence_start = hint_start
                evidence_end = hint_start + len(evidence_text)
                evidence_unit_id = container.unit_id if container else window.unit_id
                evidence_section = container.section if container else window.section
                evidence_confidence = max(evidence_confidence, 0.94)
                evidence_entailment = (
                    "ENTAILED" if evidence_section in RESULT_SECTIONS
                    else "NOT_ENOUGH_INFORMATION"
                )
                evidence_trigger_predicate = str(hint.get("predicate", "") or "").upper()
                evidence_reason_codes = [
                    *evidence_reason_codes, "grounded_complete_extractor_evidence_preserved",
                ]
            cross_sentence = window.unit_id.startswith("w")
            quality_flags = ["cross_sentence"] if cross_sentence else []
            quality_flags.extend(sorted(article_flags))
            if "trigger_attachment_ambiguous" in evidence_reason_codes:
                quality_flags.extend(["trigger_attachment_ambiguous", "manual_review"])
            if cross_sentence and LIGHT_COREFERENCE_RE.search(window.text):
                quality_flags.append("light_coreference_window")
            candidates.append(RelationPairCandidate(
                candidate_id=row["candidate_id"],
                subject=str(subject.get("mention", "")), subject_type=row["subject_type"],
                object=str(obj.get("mention", "")), object_type=row["object_type"],
                allowed_predicates=row["allowed"],
                evidence=evidence_text,
                evidence_unit_id=evidence_unit_id,
                evidence_section=evidence_section,
                evidence_char_start=evidence_start,
                evidence_char_end=evidence_end,
                source_predicates=list(dict.fromkeys(
                    str(item.get("predicate", "")).upper() for item in row["pair_hints"]
                    if str(item.get("predicate", "")).upper() in row["allowed"]
                )),
                source_directions=list(dict.fromkeys(
                    str(item.get("direction", "unknown")) for item in row["pair_hints"]
                )),
                same_sentence=not cross_sentence,
                endpoint_distance=max(
                    0, max(subject_span[0], object_span[0]) - min(subject_span[1], object_span[1])
                ),
                evidence_confidence=evidence_confidence,
                evidence_entailment=evidence_entailment,
                evidence_trigger_predicate=evidence_trigger_predicate,
                claim_role=self._infer_claim_role(
                    evidence_text, evidence_section, source_text,
                ),
                quality_flags=quality_flags,
            ))

        # Review fallback for extractor-proposed relations whose quote is an
        # exact source span but covers only one local endpoint.  Both typed
        # endpoints must still resolve to grounded article entities, so this
        # never relaxes missing-endpoint or no-source-trace hard gates.  These
        # candidates are explicitly routed and cannot be locally write-ready.
        existing_pair_keys = {
            (
                normalize_surface(item.subject), item.subject_type,
                normalize_surface(item.object), item.object_type,
            )
            for item in candidates
        }
        fallback_candidates: list[RelationPairCandidate] = []
        for pair_key, pair_hints in sorted(hints.items()):
            if pair_key in existing_pair_keys:
                continue
            subject = entity_by_alias_type.get((pair_key[0], pair_key[1]))
            obj = entity_by_alias_type.get((pair_key[2], pair_key[3]))
            if subject is None or obj is None:
                continue
            allowed = self._allowed(pair_key[1], pair_key[3])
            for hint in pair_hints:
                predicate = str(hint.get("predicate", "") or "").upper()
                if predicate not in allowed:
                    continue
                evidence = str(hint.get("evidence", "") or "").strip()
                grounded, start, end = locate_contiguous(evidence, source_text)
                if not grounded:
                    continue
                subject_present = self._mention_span(EvidenceUnit(
                    unit_id="hint", section="ABSTRACT", text=evidence,
                    char_start=start, char_end=end, parent_sentence_id="hint",
                ), subject)
                object_present = self._mention_span(EvidenceUnit(
                    unit_id="hint", section="ABSTRACT", text=evidence,
                    char_start=start, char_end=end, parent_sentence_id="hint",
                ), obj)
                if bool(subject_present) == bool(object_present):
                    continue
                # The endpoints must each be traceable somewhere in the
                # current article even though this local quote is incomplete.
                if not all(
                    any(locate_contiguous(alias, source_text)[0] for alias in self._mentions(entity))
                    for entity in (subject, obj)
                ):
                    continue
                container = ArticleEvidenceReader.containing_unit(evidence, units)
                section = container.section if container else "ABSTRACT"
                digest = hashlib.sha1("|".join(pair_key).encode("utf-8")).hexdigest()[:10]
                missing_side = "object" if subject_present else "subject"
                fallback_candidates.append(RelationPairCandidate(
                    candidate_id=f"p-{digest}",
                    subject=str(subject.get("mention", "")),
                    subject_type=pair_key[1],
                    object=str(obj.get("mention", "")),
                    object_type=pair_key[3],
                    allowed_predicates=allowed,
                    evidence=evidence,
                    evidence_unit_id=(container.unit_id if container else f"h{len(fallback_candidates):03d}"),
                    evidence_section=section,
                    evidence_char_start=start,
                    evidence_char_end=end,
                    source_predicates=[predicate],
                    source_directions=[str(hint.get("direction", "unknown") or "unknown")],
                    same_sentence=True,
                    endpoint_distance=-1,
                    evidence_confidence=0.2,
                    evidence_entailment="NOT_ENOUGH_INFORMATION",
                    evidence_trigger_predicate=(
                        predicate if any(
                            re.search(pattern, evidence, re.IGNORECASE)
                            for pattern in PREDICATE_PATTERNS.get(predicate, ())
                        ) else ""
                    ),
                    claim_role=self._infer_claim_role(evidence, section, source_text),
                    quality_flags=[
                        "incomplete_evidence_boundary", "endpoint_not_in_evidence",
                        f"{missing_side}_not_grounded", "manual_review",
                        *sorted(article_flags),
                    ],
                ))
                existing_pair_keys.add(pair_key)
                break
            if len(fallback_candidates) >= self.config.max_incomplete_evidence_candidates:
                break
        candidates.extend(fallback_candidates)

        # Candidates with a LangExtract hint are most valuable, then
        # result/conclusion assertions, then trigger-bearing pairs, then
        # close pairs.  The raised cap absorbs the widened windows.
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
        flags = ["pair_classifier_candidate", *candidate.quality_flags]
        if not candidate.same_sentence:
            flags.append("cross_sentence")
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
            "claim_role": candidate.claim_role,
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
            if "incomplete_evidence_boundary" in candidate.quality_flags:
                prediction.routed_to_llm = True
                prediction.reason_codes.append("incomplete_evidence_routed")
            plausible_relation = prediction.relation_probability >= self.config.uncertainty_floor
            # A deterministic NO_RELATION score is terminal for bare
            # co-occurrence even in a RESULTS section.  Only extractor support
            # or an explicit evidence trigger justifies a bounded model call;
            # otherwise broad entity pools create quadratic review traffic.
            supported_negative_abstention = (
                prediction.label == NO_RELATION
                and bool(
                    candidate.source_predicates
                    or candidate.evidence_trigger_predicate
                    or "incomplete_evidence_boundary" in candidate.quality_flags
                )
            )
            prediction.routed_to_llm = prediction.routed_to_llm or bool(
                plausible_relation
                and (
                    prediction.confidence < self.config.high_confidence_threshold
                    or prediction.margin < 0.18
                )
            )
            if supported_negative_abstention:
                prediction.routed_to_llm = True
                prediction.reason_codes.append("supported_no_relation_routed")
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
                    if "article_out_of_scope" in candidate.quality_flags:
                        relation["scope_status"] = "OUT_OF_SCOPE"
                        relation["quality_flags"] = sorted(set([
                            *relation["quality_flags"], "scope_bucket",
                        ]))
                        result.out_of_scope_relations.append(relation)
                    else:
                        result.low_confidence_relations.append(relation)
                continue
            relation = self._as_relation(candidate, prediction)
            if "article_out_of_scope" in candidate.quality_flags:
                relation["scope_status"] = "OUT_OF_SCOPE"
                relation["quality_flags"] = sorted(set([
                    *relation.get("quality_flags", []), "scope_bucket",
                ]))
                result.out_of_scope_relations.append(relation)
                continue
            if prediction.relation_probability >= self.config.relation_threshold:
                result.accepted_relations.append(relation)
            if prediction.routed_to_llm:
                result.low_confidence_relations.append(relation)
        result.low_confidence_relations = result.low_confidence_relations[
            : self.config.max_low_confidence_candidates
        ]
        return result
