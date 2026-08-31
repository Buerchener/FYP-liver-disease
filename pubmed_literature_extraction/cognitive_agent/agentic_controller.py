#!/usr/bin/env python3
"""Budgeted observe -> act controller for one PubMed article.

The controller is deliberately cheap and deterministic.  It does not replace
the extractor or verifier.  It observes their state and exposes only two
bounded recovery actions:

1. select a better *existing source span* for a relation whose evidence was
   truncated or used an article-local abbreviation;
2. construct a small relation-choice lattice from already extracted entities
   that co-occur in one evidence unit and have an explicit predicate cue.

The second LLM may select from that lattice, but cannot create endpoints,
evidence text, or predicates outside the ontology.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from cognitive_agent.abbreviation_detector import AbbreviationMap
from cognitive_agent.evidence_units import ArticleEvidenceReader, EvidenceUnit
from cognitive_agent.extraction_quality import (
    PREDICATE_TRIGGERS,
    locate_contiguous,
    normalize_surface,
    predicate_trigger_links_endpoints,
)
from cognitive_agent.schema.relation_signatures import RELATION_SIGNATURES


NON_FACT_SECTIONS = frozenset({
    "TITLE", "BACKGROUND", "INTRODUCTION", "OBJECTIVE", "OBJECTIVES",
    "AIM", "AIMS", "PURPOSE", "METHOD", "METHODS", "MATERIALS_AND_METHODS",
})
SECTION_RANK = {
    "RESULT": 0, "RESULTS": 0, "CONCLUSION": 0, "CONCLUSIONS": 0,
    "DISCUSSION": 1, "ABSTRACT": 2,
}
SYMMETRIC_PREDICATES = frozenset({"ASSOCIATED_WITH", "INTERACTS_WITH"})
RECOVERY_SIGNAL_RE = re.compile(
    r"\b(?:risk of|driver of|more frequent|less frequent|higher in|lower in|"
    r"enriched in|depleted in|coexpression|co-expression)\b",
    re.IGNORECASE,
)


def partition_recovery_relations(
    raw_relations: list[dict], verified_relations: list[dict], mode: str
) -> tuple[list[dict], list[dict]]:
    """Keep Agent-added edges out of production unless recall is explicit."""
    recovered = [
        copy.deepcopy(relation) for relation in verified_relations
        if "agent_recovered_relation" in set(relation.get("quality_flags", []) or [])
    ]
    if mode == "shadow-agent":
        production = [
            copy.deepcopy(relation) for relation in raw_relations
            if "agent_recovered_relation" not in set(relation.get("quality_flags", []) or [])
        ]
    else:
        production = copy.deepcopy(raw_relations)
    return production, recovered


@dataclass
class AgentAction:
    tool: str
    decision: str
    reason: str
    input_count: int = 0
    output_count: int = 0
    latency_class: str = "local"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class AgenticPlan:
    repaired_relations: list[dict] = field(default_factory=list)
    evidence_repairs: list[dict] = field(default_factory=list)
    recovery_candidates: list[dict] = field(default_factory=list)
    actions: list[AgentAction] = field(default_factory=list)
    max_rounds: int = 2
    current_round: int = 1

    def to_dict(self) -> dict:
        return {
            "policy": "observe_act_verify_stop",
            "max_rounds": self.max_rounds,
            "current_round": self.current_round,
            "evidence_repairs": self.evidence_repairs,
            "recovery_candidate_count": len(self.recovery_candidates),
            "recovery_candidates": self.recovery_candidates,
            "actions": [item.to_dict() for item in self.actions],
            "stop_condition": "stop after deterministic re-verification or budget exhaustion",
        }


class AgenticArticleController:
    """Generate a bounded article-local action plan without an LLM call."""

    def __init__(
        self,
        max_recovery_candidates: int = 24,
        max_repairs: int = 16,
        enable_evidence_repair: bool = True,
        enable_recovery: bool = True,
        min_recovery_score: float = 0.55,
    ):
        self.max_recovery_candidates = max(1, int(max_recovery_candidates))
        self.max_repairs = max(1, int(max_repairs))
        self.enable_evidence_repair = bool(enable_evidence_repair)
        self.enable_recovery = bool(enable_recovery)
        self.min_recovery_score = max(0.0, min(1.0, float(min_recovery_score)))
        self.reader = ArticleEvidenceReader()

    @staticmethod
    def _predicate_has_signal(predicate: str, text: str) -> bool:
        patterns = PREDICATE_TRIGGERS.get(str(predicate).upper(), ())
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    @classmethod
    def _fallback_signal_links(
        cls, text: str, subject_aliases: list[str], object_aliases: list[str]
    ) -> bool:
        subject = cls._find_alias(text, subject_aliases)
        object_match = cls._find_alias(text, object_aliases)
        if not subject or not object_match:
            return False
        low, high = sorted((subject[1], object_match[1]))
        return any(
            match.end() >= low and match.start() <= high
            for match in RECOVERY_SIGNAL_RE.finditer(text)
        )

    @staticmethod
    def _alias_pattern(value: str) -> re.Pattern | None:
        tokens = re.findall(r"\w+", str(value or ""), flags=re.UNICODE)
        if not tokens:
            return None
        pattern = r"(?<!\w)" + r"(?:[\W_]+)".join(
            re.escape(token) for token in tokens
        ) + r"(?!\w)"
        return re.compile(pattern, re.IGNORECASE)

    @classmethod
    def _find_alias(cls, text: str, aliases: list[str]) -> tuple[str, int] | None:
        matches: list[tuple[str, int]] = []
        for alias in aliases:
            pattern = cls._alias_pattern(alias)
            if not pattern:
                continue
            match = pattern.search(text)
            if match:
                matches.append((alias, match.start()))
        return min(matches, key=lambda item: item[1], default=None)

    @staticmethod
    def _entity_aliases(entity: dict, abbr_map: AbbreviationMap) -> list[str]:
        mention = str(entity.get("mention", "") or "")
        aliases = [mention, *(entity.get("canonical_mentions", []) or [])]
        if mention:
            aliases.extend([
                abbr_map.resolve_to_long(mention),
                abbr_map.resolve_to_short(mention),
            ])
        for short, long_form in abbr_map.abbr_to_long.items():
            if normalize_surface(mention) in {
                normalize_surface(short), normalize_surface(long_form),
            }:
                aliases.extend([short, long_form])
        return list(dict.fromkeys(item.strip() for item in aliases if str(item).strip()))

    @staticmethod
    def _parent_units(text: str, units: list[EvidenceUnit]) -> list[EvidenceUnit]:
        grouped: dict[str, list[EvidenceUnit]] = {}
        for unit in units:
            grouped.setdefault(unit.parent_sentence_id, []).append(unit)
        parents: list[EvidenceUnit] = []
        for parent_id, children in grouped.items():
            ordered = sorted(children, key=lambda item: item.char_start)
            start, end = ordered[0].char_start, ordered[-1].char_end
            parents.append(EvidenceUnit(
                unit_id=f"p{parent_id[1:]}",
                section=ordered[0].section,
                text=text[start:end],
                char_start=start,
                char_end=end,
                parent_sentence_id=parent_id,
            ))
        return parents

    def _evidence_choices(self, text: str, units: list[EvidenceUnit]) -> list[EvidenceUnit]:
        # Clause units are preferred; parent sentences provide a lossless
        # fallback for coordinated entities split across clauses.
        combined = [*units, *self._parent_units(text, units)]
        unique: dict[tuple[int, int], EvidenceUnit] = {}
        for unit in combined:
            key = (unit.char_start, unit.char_end)
            previous = unique.get(key)
            if previous is None or len(unit.text) < len(previous.text):
                unique[key] = unit
        return list(unique.values())

    def repair_relation_evidence(
        self,
        text: str,
        raw_entities: list[dict],
        raw_relations: list[dict],
        abbr_map: AbbreviationMap,
        units: list[EvidenceUnit],
    ) -> tuple[list[dict], list[dict]]:
        repaired = copy.deepcopy(raw_relations)
        repairs: list[dict] = []
        choices = self._evidence_choices(text, units)
        entity_alias_index: dict[tuple[str, str], list[str]] = {}
        for entity in raw_entities:
            key = (normalize_surface(entity.get("mention", "")), str(entity.get("type", "")))
            entity_alias_index.setdefault(key, []).extend(self._entity_aliases(entity, abbr_map))

        for raw_index, relation in enumerate(repaired):
            if len(repairs) >= self.max_repairs:
                break
            evidence = str(relation.get("evidence", "") or "").strip()
            subject = str(relation.get("subject", "") or "")
            object_name = str(relation.get("object", "") or "")
            subject_type = str(relation.get("subject_type", "") or "")
            object_type = str(relation.get("object_type", "") or "")
            subject_aliases = list(dict.fromkeys([
                subject,
                abbr_map.resolve_to_long(subject),
                abbr_map.resolve_to_short(subject),
                *entity_alias_index.get((normalize_surface(subject), subject_type), []),
            ]))
            object_aliases = list(dict.fromkeys([
                object_name,
                abbr_map.resolve_to_long(object_name),
                abbr_map.resolve_to_short(object_name),
                *entity_alias_index.get((normalize_surface(object_name), object_type), []),
            ]))
            source_exact = bool(locate_contiguous(evidence, text)[0])
            endpoints_exact = bool(
                self._find_alias(evidence, subject_aliases)
                and self._find_alias(evidence, object_aliases)
            )
            if source_exact and endpoints_exact:
                continue

            candidates: list[EvidenceUnit] = []
            for unit in choices:
                if not self._find_alias(unit.text, subject_aliases):
                    continue
                if not self._find_alias(unit.text, object_aliases):
                    continue
                candidates.append(unit)
            if not candidates:
                continue
            predicate = str(relation.get("predicate", "") or "").upper()
            candidates.sort(key=lambda unit: (
                SECTION_RANK.get(unit.section, 8),
                not self._predicate_has_signal(predicate, unit.text),
                len(unit.text),
                unit.char_start,
            ))
            selected = candidates[0]
            relation["evidence"] = selected.text
            relation.setdefault("quality_flags", []).append("agent_evidence_repaired")
            repairs.append({
                "raw_index": raw_index,
                "subject": subject,
                "predicate": predicate,
                "object": object_name,
                "old_evidence": evidence,
                "new_evidence": selected.text,
                "evidence_unit_id": selected.unit_id,
                "section": selected.section,
                "reason": "source_span_or_endpoint_grounding_repair",
            })
        return repaired, repairs

    def build_recovery_candidates(
        self,
        text: str,
        verified_entities: list[dict],
        verified_relations: list[dict],
        abbr_map: AbbreviationMap,
        units: list[EvidenceUnit],
    ) -> list[dict]:
        entities: list[dict[str, Any]] = []
        for index, entity in enumerate(verified_entities):
            mention = str(entity.get("mention", "") or "").strip()
            entity_type = str(entity.get("type", entity.get("entity_type", "")) or "")
            if not mention or not entity_type:
                continue
            entities.append({
                "index": index,
                "mention": mention,
                "type": entity_type,
                "aliases": self._entity_aliases({
                    "mention": mention,
                    "canonical_mentions": entity.get("canonical_mentions", []),
                }, abbr_map),
            })

        existing_pairs: dict[tuple[str, str, str, str], list[dict]] = {}
        for relation in verified_relations:
            pair = (
                normalize_surface(relation.get("subject", "")),
                str(relation.get("subject_type", "")),
                normalize_surface(relation.get("object", "")),
                str(relation.get("object_type", "")),
            )
            existing_pairs.setdefault(pair, []).append(relation)

        if not self.enable_recovery:
            return []

        proposals: list[dict] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        evidence_choices = [
            unit for unit in self._evidence_choices(text, units)
            if unit.section not in NON_FACT_SECTIONS
        ]
        evidence_choices.sort(key=lambda unit: (
            SECTION_RANK.get(unit.section, 8), len(unit.text), unit.char_start,
        ))

        for unit in evidence_choices:
            mentions: list[tuple[dict, int]] = []
            for entity in entities:
                match = self._find_alias(unit.text, entity["aliases"])
                if match:
                    mentions.append((entity, match[1]))
            if len(mentions) < 2 or len(mentions) > 10:
                continue
            mentions.sort(key=lambda item: item[1])
            for left_index, (left, left_pos) in enumerate(mentions):
                for right, right_pos in mentions[left_index + 1:]:
                    orientations = [(left, right), (right, left)]
                    selected: tuple[dict, dict, list[str]] | None = None
                    for subject, object_entity in orientations:
                        allowed = [
                            predicate for predicate, signatures in RELATION_SIGNATURES.items()
                            if (subject["type"], object_entity["type"]) in signatures
                            and (
                                predicate_trigger_links_endpoints(
                                    predicate,
                                    unit.text,
                                    subject["aliases"],
                                    object_entity["aliases"],
                                )
                                or (
                                    predicate == "ASSOCIATED_WITH"
                                    and self._fallback_signal_links(
                                        unit.text,
                                        subject["aliases"],
                                        object_entity["aliases"],
                                    )
                                )
                            )
                        ]
                        if allowed:
                            selected = (subject, object_entity, allowed)
                            break
                    if not selected:
                        continue
                    subject, object_entity, allowed = selected
                    pair = (
                        normalize_surface(subject["mention"]), subject["type"],
                        normalize_surface(object_entity["mention"]), object_entity["type"],
                    )
                    inverse_pair = (pair[2], pair[3], pair[0], pair[1])
                    pair_relations = [
                        *existing_pairs.get(pair, []),
                        *existing_pairs.get(inverse_pair, []),
                    ]
                    # A pair with any surviving candidate is handled by normal
                    # adjudication.  Recovery is reserved for genuinely absent
                    # or deterministically blocked pair claims.
                    if any(
                        str(item.get("factual_status", "VALID")).upper() != "REJECTED"
                        for item in pair_relations
                    ):
                        continue
                    key = (*pair, unit.parent_sentence_id)
                    if key in seen:
                        continue
                    if any(pred in SYMMETRIC_PREDICATES for pred in allowed):
                        reverse_key = (*inverse_pair, unit.parent_sentence_id)
                        if reverse_key in seen:
                            continue
                    seen.add(key)
                    endpoint_distance = abs(right_pos - left_pos)
                    score = 0.25  # explicit predicate trigger links both endpoints
                    score_reasons = ["explicit_trigger_links_endpoints:+0.25"]
                    section_weight = {
                        "RESULT": 0.30, "RESULTS": 0.30,
                        "CONCLUSION": 0.30, "CONCLUSIONS": 0.30,
                        "DISCUSSION": 0.16, "ABSTRACT": 0.10,
                    }.get(unit.section, 0.05)
                    score += section_weight
                    score_reasons.append(
                        f"section_{unit.section.casefold()}:+{section_weight:.2f}"
                    )
                    distance_weight = max(0.0, 0.18 * (1.0 - min(endpoint_distance, 180) / 180))
                    score += distance_weight
                    score_reasons.append(f"endpoint_distance_{endpoint_distance}:+{distance_weight:.2f}")
                    if any(predicate in {
                        "ENCODES", "INTERACTS_WITH", "EXPRESSED_IN",
                        "PROGNOSTIC_IN", "PROGRESSES_TO",
                    } for predicate in allowed):
                        score += 0.12
                        score_reasons.append("specific_predicate:+0.12")
                    human_signal = bool(re.search(
                        r"\b(?:patients?|participants?|subjects?|cohort|clinical|human|"
                        r"healthy controls?|patient-derived|liver tissues?)\b",
                        unit.text,
                        re.IGNORECASE,
                    ))
                    nonhuman_signal = bool(re.search(
                        r"\b(?:mice|mouse|murine|rats?|rat model|in vitro|cell lines?)\b",
                        unit.text,
                        re.IGNORECASE,
                    ))
                    if human_signal:
                        score += 0.12
                        score_reasons.append("human_study_signal:+0.12")
                    if nonhuman_signal:
                        score -= 0.35
                        score_reasons.append("nonhuman_or_in_vitro:-0.35")
                    score = round(max(0.0, min(1.0, score)), 4)
                    if score < self.min_recovery_score:
                        continue
                    proposals.append({
                        "candidate_id": "",
                        "candidate_kind": "recovery",
                        "raw_index": -1,
                        "subject": subject["mention"],
                        "subject_type": subject["type"],
                        "predicate": "NONE",
                        "object": object_entity["mention"],
                        "object_type": object_entity["type"],
                        "direction": "unknown",
                        "evidence": unit.text,
                        "evidence_unit_id": unit.unit_id,
                        "section": unit.section,
                        "import_ready": False,
                        "quality_flags": ["relation_gap_candidate"],
                        "ranking_score": score,
                        "ranking_reasons": score_reasons,
                        "allowed_predicates": allowed[:4],
                        "rank": (
                            -score,
                            SECTION_RANK.get(unit.section, 8),
                            0 if "Disease" in {subject["type"], object_entity["type"]} else 1,
                            len(unit.text),
                            left_pos,
                            right_pos,
                        ),
                    })

        proposals.sort(key=lambda item: item.pop("rank"))
        per_sentence: dict[str, int] = {}
        bounded: list[dict] = []
        for proposal in proposals:
            sentence_id = str(proposal.get("evidence_unit_id", "") or "")
            if per_sentence.get(sentence_id, 0) >= 6:
                continue
            per_sentence[sentence_id] = per_sentence.get(sentence_id, 0) + 1
            bounded.append(proposal)
            if len(bounded) >= self.max_recovery_candidates:
                break
        proposals = bounded
        for index, proposal in enumerate(proposals):
            proposal["candidate_id"] = f"p{index:03d}"
        return proposals

    def plan(
        self,
        text: str,
        raw_entities: list[dict],
        raw_relations: list[dict],
        abbr_map: AbbreviationMap,
        units: list[EvidenceUnit],
    ) -> AgenticPlan:
        if self.enable_evidence_repair:
            repaired, repairs = self.repair_relation_evidence(
                text, raw_entities, raw_relations, abbr_map, units
            )
        else:
            repaired, repairs = copy.deepcopy(raw_relations), []
        plan = AgenticPlan(repaired_relations=repaired, evidence_repairs=repairs)
        plan.actions.append(AgentAction(
            tool="evidence_span_repair",
            decision="CALL" if repairs else "SKIP",
            reason=(
                "repair relation evidence from exact article-local spans"
                if repairs else (
                    "disabled by precision mode"
                    if not self.enable_evidence_repair
                    else "all candidate spans already grounded or no safe repair exists"
                )
            ),
            input_count=len(raw_relations), output_count=len(repairs),
        ))
        return plan

    def add_recovery_observation(
        self,
        plan: AgenticPlan,
        text: str,
        verified_entities: list[dict],
        verified_relations: list[dict],
        abbr_map: AbbreviationMap,
        units: list[EvidenceUnit],
    ) -> AgenticPlan:
        plan.recovery_candidates = self.build_recovery_candidates(
            text, verified_entities, verified_relations, abbr_map, units
        ) if self.enable_recovery else []
        plan.actions.append(AgentAction(
            tool="relation_gap_scan",
            decision="CALL" if self.enable_recovery else "SKIP",
            reason=(
                "scan entity-first evidence units for ranked schema-constrained missing relations"
                if self.enable_recovery else "disabled by precision mode"
            ),
            input_count=len(verified_entities),
            output_count=len(plan.recovery_candidates),
        ))
        return plan
