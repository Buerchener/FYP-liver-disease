#!/usr/bin/env python3
"""Entity Coverage Critic → Missing Entity Recovery (training-free, LLM-centric).

The primary extractor (LangExtract + Gemini) leaves some schema entities out
of the inventory.  On the frozen dev snapshot the offline lattice diagnostic
counted 8/63 gold relations whose endpoint was never extracted (12.7%):
missed endpoints are predominantly Pathway / Metabolite / Tissue / Protein
names (``NK cell activation``, ``hepatic ferroptosis``, ``DAM``, ``fibrotic
liver``, ...).  This module runs a single bounded LLM pass per article that

1. critiques the entity inventory for coverage gaps (Entity Coverage Critic),
2. proposes missing entity spans (Missing Entity Recovery), and
3. validates every proposal deterministically: contiguous source span
   (`locate_contiguous`), schema type whitelist, generic-term filter, and
   alias-aware dedup against the existing inventory.

The recovery pass MAY add entities; it can NEVER generate relations or
predicates.  Downstream the widened inventory feeds the ordinary pair lattice
and every candidate pair is still decided by the judge + deterministic
verifier, so the Verifier authority and the Safe Write boundary are untouched.

Modes: off | shadow (record counterfactual proposals, do not add) |
active (append validated entities to the inventory before pairing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from cognitive_agent.abbreviation_detector import AbbreviationMap
from cognitive_agent.extraction_kernel import GENERIC_ENTITY_TERMS
from cognitive_agent.extraction_quality import (
    entity_filter_reason,
    locate_contiguous,
    normalize_surface,
)
from cognitive_agent.schema.entity_classes import ENTITY_CLASSES

ENTITY_TYPES = frozenset(item["label"] for item in ENTITY_CLASSES.values())

# Recovery-specific generic terms on top of the extractor's list: things an
# LLM recovery pass may propose but which can never be schema entities.
RECOVERY_GENERIC_TERMS = frozenset({
    "patients", "patient", "subjects", "subject", "controls", "control",
    "participants", "mice", "mouse", "rats", "rat", "humans", "human",
    "cohort", "cohorts", "groups", "group", "samples", "sample", "specimens",
    "expression", "levels", "level", "activity", "results", "findings",
    "data", "study", "studies", "analysis", "analyses", "models", "model",
    "experiments", "experiment", "cells", "cell", "tissue", "tissues",
    "liver", "serum", "plasma", "blood", "protein", "proteins", "gene",
    "genes", "disease", "diseases", "cancer", "pathway", "pathways",
})
RECOVERY_BLOCKED_TERMS = GENERIC_ENTITY_TERMS | RECOVERY_GENERIC_TERMS

# Do not embed development-gold endpoint strings in prompts.  Such examples
# leak evaluation information whenever an article later becomes part of a
# held-out set.  The optional guidance below is deliberately generic and
# contains no PMID, article-specific span, predicate, or gold annotation.
GENERIC_RECOVERY_GUIDANCE = (
    "Check only for concrete, verbatim biomedical endpoint mentions such as "
    "named genes/proteins, diseases, pathways, metabolites, tissues, drugs, "
    "or organisms that are absent from the inventory."
)


@dataclass
class EntityRecoveryConfig:
    mode: str = "off"  # off | shadow | active
    model_id: str = "deepseek-v4-flash"
    max_proposals: int = 12
    max_accepted: int = 10
    golden_shot: bool = False
    min_name_chars: int = 2
    max_name_chars: int = 100
    max_text_chars: int = 8000


@dataclass
class EntityRecoveryResult:
    pmid: str = ""
    status: str = "OFF"
    coverage_gaps: list[dict] = field(default_factory=list)
    proposals: list[dict] = field(default_factory=list)
    recovered: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    model_id: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "status": self.status,
            "mode": "entity_recovery",
            "coverage_gaps": self.coverage_gaps,
            "proposal_count": len(self.proposals),
            "recovered_count": len(self.recovered),
            "recovered": self.recovered,
            "rejected": self.rejected,
            "stats": {
                "proposals": len(self.proposals),
                "accepted": len(self.recovered),
                "rejected": len(self.rejected),
                "reject_reasons": self._reject_reason_counts(),
            },
            "model_id": self.model_id,
            "latency_s": round(self.latency_s, 4),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
        }

    def _reject_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.rejected:
            reason = str(item.get("reason", "other"))
            counts[reason] = counts.get(reason, 0) + 1
        return counts


class EntityRecovery:
    """Bounded LLM entity recovery with deterministic validation."""

    def __init__(
        self,
        config: EntityRecoveryConfig,
        registry: Optional[Any] = None,
        abbr_detector: Optional[Any] = None,
    ):
        if config.mode not in {"off", "shadow", "active"}:
            raise ValueError("entity_recovery mode must be off, shadow, or active")
        self.config = config
        self.registry = registry
        self.abbr_detector = abbr_detector

    # ── public API ────────────────────────────────────────────────────────
    def run(
        self,
        *,
        pmid: str,
        text: str,
        entities: list[dict],
        abbr_map: Optional[AbbreviationMap] = None,
        study_type: str = "",
    ) -> EntityRecoveryResult:
        result = EntityRecoveryResult(pmid=str(pmid))
        if self.config.mode == "off" or self.registry is None:
            result.status = "OFF"
            return result

        inventory = self._inventory_view(entities, abbr_map)
        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(
            pmid, text, inventory, study_type,
        )
        call = self.registry.call_json(
            "recovery",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_hint={
                "coverage_gaps": [{"gap": "string", "why_relevant": "string"}],
                "entities": [
                    {"name": "string (verbatim source span)",
                     "type": "string (one of the 7 schema types)",
                     "evidence_quote": "string (verbatim sentence fragment)"}
                ],
            },
        )
        result.status = call.status
        result.model_id = call.model_id
        result.latency_s = call.latency_s
        result.prompt_tokens = call.prompt_tokens
        result.output_tokens = call.output_tokens
        result.error = call.error
        if call.status != "OK":
            # Fail closed: never invent entities when the recovery model fails.
            return result

        payload = call.payload or {}
        result.coverage_gaps = [
            {"gap": str(g.get("gap", ""))[:500], "why_relevant": str(g.get("why_relevant", ""))[:500]}
            for g in payload.get("coverage_gaps", []) or []
            if isinstance(g, dict)
        ][: 8]
        result.proposals = [
            {
                "name": str(item.get("name", "") or "").strip(),
                "type": str(item.get("type", "") or "").strip(),
                "evidence_quote": str(item.get("evidence_quote", "") or "").strip(),
            }
            for item in payload.get("entities", []) or []
            if isinstance(item, dict)
        ][: self.config.max_proposals]

        existing = [(str(e.get("mention", "")), str(e.get("type", "") or "")) for e in entities]
        accepted_names: list[str] = []
        for proposal in result.proposals:
            validated, reason = self._validate(
                proposal, text, existing, accepted_names, abbr_map,
            )
            if validated is None:
                result.rejected.append({"name": proposal["name"], "reason": reason})
                continue
            if len(result.recovered) >= self.config.max_accepted:
                result.rejected.append({"name": proposal["name"], "reason": "cap_reached"})
                continue
            result.recovered.append(validated)
            accepted_names.append(normalize_surface(validated["mention"]))
        return result

    # ── prompts ───────────────────────────────────────────────────────────
    def _system_prompt(self) -> str:
        type_list = ", ".join(sorted(ENTITY_TYPES))
        lines = [
            "You are an entity coverage critic for a biomedical relation-extraction pipeline "
            "over liver-disease PubMed abstracts.",
            f"Entity types allowed by the schema: {type_list}.",
            "The pipeline extracts 8 relation types (ASSOCIATED_WITH, INTERACTS_WITH, "
            "PROGRESSES_TO, PARTICIPATES_IN, EXPRESSED_IN, PROGNOSTIC_IN, "
            "ASSOCIATED_WITH_METABOLITE, ENCODES) between pairs of these entities.",
            "",
            "Your task has two parts:",
            "1. CRITIQUE: list concrete coverage gaps in the current entity inventory — "
            "entities that appear in the abstract, fit the schema types, are plausible "
            "endpoints of the 8 relation types, but are missing from the inventory.",
            "2. RECOVER: propose the missing entities themselves.",
            "",
            "HARD RULES:",
            f"- Propose AT MOST {self.config.max_proposals} entities.",
            "- Every proposed name must appear VERBATIM as a contiguous span in the article "
            "text (title or abstract).  Copy the exact surface form, e.g. 'fibrotic liver' "
            "not 'Liver Fibrosis' if only the former is in the text.",
            "- Types must be exactly one of the schema types listed above.",
            "- Only propose entities that plausibly participate in one of the 8 relation "
            "types with entities already in the inventory.  Do NOT list every noun.",
            "- Do NOT propose entities already present in the inventory under any spelling, "
            "case or abbreviation variant — those are duplicates, not gaps.",
            "- Do NOT invent entities not in the text, do NOT normalise names, do NOT "
            "propose generic terms like 'patients', 'cells', 'expression'.",
            "- Output ONLY entities.  Never output relations, predicates or evidence "
            "judgements — relation decisions are made downstream and your output must "
            "not pre-empt them.",
        ]
        if self.config.golden_shot:
            lines += ["", "Additional non-gold guidance:", GENERIC_RECOVERY_GUIDANCE]
        return "\n".join(lines)

    def _user_prompt(
        self, pmid: str, text: str, inventory: list[dict], study_type: str,
    ) -> str:
        shown = text
        truncated = False
        if len(shown) > self.config.max_text_chars:
            shown = shown[: self.config.max_text_chars]
            truncated = True
        inventory_block = "\n".join(
            f"- {item['mention']} ({item['type']})"
            for item in inventory[:80]
        ) or "(empty inventory)"
        lines = [
            f"PMID: {pmid}",
            f"Study type: {study_type or 'unknown'}",
            "",
            "Current entity inventory:",
            inventory_block,
            "",
            "Article text:",
            shown,
        ]
        if truncated:
            lines.append("[text truncated — use only spans visible above]")
        return "\n".join(lines)

    # ── deterministic validation ──────────────────────────────────────────
    def _validate(
        self,
        proposal: dict,
        text: str,
        existing: list[tuple[str, str]],
        accepted_names: list[str],
        abbr_map: Optional[AbbreviationMap],
    ) -> tuple[Optional[dict], str]:
        name = proposal.get("name", "")
        entity_type = proposal.get("type", "")
        if not (self.config.min_name_chars <= len(name) <= self.config.max_name_chars):
            return None, "bad_length"
        if entity_type not in ENTITY_TYPES:
            return None, "bad_type"
        if name.casefold() in RECOVERY_BLOCKED_TERMS:
            return None, "generic_term"
        # Mirror prepare_extraction's deterministic filters: anything the
        # downstream chain would drop must never be proposed (fail-closed;
        # keeps recovery slots for entities that survive verification).
        status, reason = entity_filter_reason(
            {"mention": name, "type": entity_type}, [], text,
        )
        if status == "rejected":
            return None, f"downstream_{reason}"
        located, start, end = locate_contiguous(name, text)
        if not located:
            return None, "span_not_in_source"

        norm = normalize_surface(name)
        # Duplicate of an existing inventory entity (alias-aware)?
        for mention, mention_type in existing:
            if normalize_surface(mention) == norm:
                return None, "duplicate_existing"
            if abbr_map is not None:
                canonical = abbr_map.canonical_name(name)
                if normalize_surface(canonical) == normalize_surface(mention):
                    return None, "duplicate_existing_alias"
        # Duplicate of an already-accepted recovery proposal?
        if norm in accepted_names:
            return None, "duplicate_proposal"

        return {
            "mention": name,
            "type": entity_type,
            "extraction_class": "entity_recovery",
            "attributes": {},
            "grounded": True,
            "source_span": text[start:end],
            "alignment_status": "RECOVERED",
            "char_start": start,
            "char_end": end,
            "recovered": True,
        }, ""

    def _inventory_view(
        self, entities: list[dict], abbr_map: Optional[AbbreviationMap],
    ) -> list[dict]:
        seen: set[str] = set()
        view: list[dict] = []
        for entity in entities:
            mention = str(entity.get("mention", "") or "")
            if not mention:
                continue
            norm = normalize_surface(mention)
            if norm in seen:
                continue
            seen.add(norm)
            view.append({"mention": mention, "type": str(entity.get("type", "") or "")})
        return view
