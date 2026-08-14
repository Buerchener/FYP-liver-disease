#!/usr/bin/env python3
"""
cognitive_agent/self_reflection.py — Phase 6: Metacognitive Reflection

Evaluates extraction quality across articles, adjusts thresholds,
proposes schema extensions, and detects emerging patterns.

Extracted from and significantly expanded beyond the original ~50-line
_reflect() method in agent.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from cognitive_agent.decision_engine import ExecutionLog
from cognitive_agent.context_activator import ContextCard


@dataclass
class StrategyUpdate:
    """Strategy adjustments proposed by the self-reflection phase."""
    threshold_adjustments: dict[str, float] = field(default_factory=dict)
    schema_proposals: list[str] = field(default_factory=list)
    example_adjustments: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def adjust_threshold(self, key: str, delta: float):
        """Record a threshold adjustment (e.g. entity_creation_confidence: -0.1)."""
        self.threshold_adjustments[key] = delta

    def propose_schema_extension(self, extension: str):
        """Propose a new relation signature or entity type."""
        self.schema_proposals.append(extension)

    def add_suggestion(self, suggestion: str):
        """Add a strategy suggestion."""
        self.suggestions.append(suggestion)

    def note_example_change(self, change: str):
        """Note a few-shot example change."""
        self.example_adjustments.append(change)

    def has_changes(self) -> bool:
        """Return True if any adjustments were made."""
        return bool(self.threshold_adjustments or self.schema_proposals
                    or self.example_adjustments or self.suggestions)

    def to_dict(self) -> dict:
        return {
            "threshold_adjustments": self.threshold_adjustments,
            "schema_proposals": self.schema_proposals,
            "example_adjustments": self.example_adjustments,
            "suggestions": self.suggestions,
        }


class SelfReflection:
    """Phase 6: Metacognitive reflection and strategy adaptation.

    Evaluates recent extraction quality and proposes adjustments to:
    - Entity creation thresholds
    - Few-shot example selection
    - Schema coverage
    """

    def __init__(self):
        pass

    def reflect(
        self,
        execution_log: ExecutionLog,
        context_card: ContextCard,
        agent_state,
        episodic_memory=None,
    ) -> StrategyUpdate:
        """Analyze recent performance and propose strategy updates.

        Args:
            execution_log: Latest article's execution decisions
            context_card: Latest article's context card
            agent_state: AgentState with cumulative statistics
            episodic_memory: Optional EpisodicMemory for cross-document analysis

        Returns:
            StrategyUpdate with proposed adjustments
        """
        update = StrategyUpdate()

        # ── 1. Compute per-article quality metrics ──
        total_actions = len(execution_log.actions)
        if total_actions == 0:
            return update  # nothing to analyze

        discard_count = execution_log.discarded
        dispute_count = execution_log.disputed
        create_count = execution_log.entities_created + execution_log.relations_created

        discard_rate = discard_count / max(total_actions, 1)
        create_rate = create_count / max(total_actions, 1)
        dispute_rate = dispute_count / max(total_actions, 1)

        # ── 2. Threshold adaptation ──
        entity_link_rate = context_card.coverage_score

        if entity_link_rate < 0.3 and agent_state.total_articles > 10:
            # In a medical KG, low linkage is ambiguity/novelty evidence, not
            # permission to lower the creation bar.
            update.add_suggestion(
                f"Low entity link rate ({entity_link_rate:.1%}) — "
                "route unresolved mentions to entity disambiguation; keep the creation threshold"
            )

        if discard_rate > 0.5:
            # A high-recall generator naturally emits noise.  Do not interpret
            # this signal as automatic evidence that the ontology is incomplete.
            update.add_suggestion(
                f"High discard rate ({discard_rate:.1%}) — "
                "mine hard negatives before proposing any schema extension"
            )

        if dispute_rate > 0.2:
            update.adjust_threshold("relation_creation_confidence", 0.05)
            update.add_suggestion(
                f"Elevated dispute rate ({dispute_rate:.1%}) — "
                "raised the relation threshold and require curator review"
            )

        if create_rate == 0 and agent_state.total_articles > 20:
            update.add_suggestion(
                "No entities/relations created in the recent batch; this is a valid outcome "
                "and does not justify lowering evidence thresholds"
            )

        # ── 3. Cross-document pattern discovery ──
        if episodic_memory and agent_state.total_articles > 0:
            if agent_state.total_articles % 20 == 0:
                # Check for emerging entities across articles
                emerging_entities = episodic_memory.find_emerging_entities(min_frequency=3)
                for ent in emerging_entities:
                    update.propose_schema_extension(
                        f"Frequent novel entity: {ent['name']} ({ent['type']}) — "
                        f"appeared in {ent['frequency']} articles"
                    )

                # Check for emerging relation patterns
                emerging_rels = episodic_memory.find_emerging_relations(min_frequency=3)
                for rel in emerging_rels:
                    update.propose_schema_extension(
                        f"Frequent relation type: {rel['predicate']} — "
                        f"appeared in {rel['frequency']} articles"
                    )

                # Check for contradiction patterns
                contradictions = episodic_memory.find_contradiction_patterns()
                for c in contradictions:
                    update.add_suggestion(
                        f"Repeated dispute: {c['entity']} -[{c['predicate']}]-> "
                        f"{c['object']} disputed in {c['frequency']} articles — "
                        f"may need human review"
                    )

        # ── 4. Few-shot example adaptation ──
        recent_quality = agent_state.quality_scores[-10:] if agent_state.quality_scores else []
        if recent_quality and sum(recent_quality) / len(recent_quality) < 0.2:
            update.note_example_change(
                "Low average quality — consider expanding few-shot examples "
                "to include more diverse relation types"
            )

        return update
