#!/usr/bin/env python3
"""
cognitive_agent/strategy_manager.py — Dynamic Strategy Manager

Manages extraction strategy evolution across the agent's lifecycle:
- Threshold state (entity/relation creation confidence, dispute thresholds)
- Strategy snapshots for rollback
- Extraction mode selection based on ContextCard coverage
- Error pattern tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from cognitive_agent.context_activator import ContextCard


@dataclass
class StrategyState:
    """Mutable strategy parameters that evolve during a run."""
    entity_creation_confidence: float = 0.7
    relation_creation_confidence: float = 0.7
    auto_update_threshold: float = 0.8
    dispute_threshold: float = 0.6
    use_extended_examples: bool = False
    conflict_resolution_mode: str = "balanced"  # conservative | balanced | aggressive
    extraction_mode: str = "balanced"            # exploratory | focused | balanced

    def to_dict(self) -> dict:
        return {
            "entity_creation_confidence": self.entity_creation_confidence,
            "relation_creation_confidence": self.relation_creation_confidence,
            "auto_update_threshold": self.auto_update_threshold,
            "dispute_threshold": self.dispute_threshold,
            "use_extended_examples": self.use_extended_examples,
            "conflict_resolution_mode": self.conflict_resolution_mode,
            "extraction_mode": self.extraction_mode,
        }


class StrategyManager:
    """Manages extraction strategy with snapshot/rollback capability.

    Applies StrategyUpdates from SelfReflection and adjusts the
    active strategy parameters accordingly.
    """

    def __init__(self, config=None):
        self.state = StrategyState()
        self._snapshots: list[StrategyState] = []
        self._error_patterns: list[dict] = []

        # Apply initial config if provided
        if config:
            if hasattr(config, 'entity_creation_min_confidence'):
                self.state.entity_creation_confidence = config.entity_creation_min_confidence
            if hasattr(config, 'relation_creation_min_confidence'):
                self.state.relation_creation_confidence = config.relation_creation_min_confidence

    def get_strategy(self, context_card: ContextCard) -> dict:
        """Return the current strategy parameters adapted to the context.

        Args:
            context_card: Phase 1 context card with coverage score and gaps

        Returns:
            Strategy dict with extraction mode, thresholds, etc.
        """
        strategy = {
            "entity_confidence_threshold": self.state.entity_creation_confidence,
            "relation_confidence_threshold": self.state.relation_creation_confidence,
            "extraction_mode": self._pick_extraction_mode(context_card),
            "conflict_resolution_mode": self.state.conflict_resolution_mode,
            "use_extended_examples": self.state.use_extended_examples,
        }

        # KG coverage controls retrieval focus, not evidence standards.  Low
        # coverage must not silently switch a medical extractor to broad mode.
        coverage = context_card.coverage_score
        if coverage > 0.6:
            strategy["extraction_mode"] = "focused"
        elif coverage < 0.2:
            strategy["extraction_mode"] = "balanced"

        return strategy

    def apply_update(self, update) -> bool:
        """Apply a StrategyUpdate. Snapshots current state before changes.

        Args:
            update: StrategyUpdate from SelfReflection

        Returns:
            True if any changes were applied
        """
        if not update.has_changes():
            return False

        # Snapshot current state
        self._snapshots.append(deepcopy(self.state))

        # Apply threshold adjustments
        for key, delta in update.threshold_adjustments.items():
            self._apply_threshold(key, delta)

        # Apply example mode changes
        for change in update.example_adjustments:
            if "expand" in change.lower():
                self.state.use_extended_examples = True

        # Apply conflict resolution mode changes
        for suggestion in update.suggestions:
            if "conservative" in suggestion.lower():
                self.state.conflict_resolution_mode = "conservative"
            elif "aggressive" in suggestion.lower():
                self.state.conflict_resolution_mode = "aggressive"

        return True

    def handle_error(self, error: Exception, article: dict):
        """Record an error and potentially adjust strategy."""
        self._error_patterns.append({
            "pmid": article.get("pmid", "unknown"),
            "error": str(error),
            "error_type": type(error).__name__,
        })

        # If same error type appears >5 times, suggest conservative mode
        error_types = [e["error_type"] for e in self._error_patterns[-10:]]
        from collections import Counter
        type_counts = Counter(error_types)
        for error_type, count in type_counts.items():
            if count >= 5:
                self.state.conflict_resolution_mode = "conservative"

    def rollback(self) -> bool:
        """Roll back to the previous strategy state. Returns True if successful."""
        if not self._snapshots:
            return False
        self.state = self._snapshots.pop()
        return True

    def get_snapshot_count(self) -> int:
        """Return number of saved snapshots (for monitoring strategy churn)."""
        return len(self._snapshots)

    def get_error_count(self) -> int:
        """Return total error count."""
        return len(self._error_patterns)

    def _pick_extraction_mode(self, context_card: ContextCard) -> str:
        """Choose extraction mode based on context."""
        if "exploratory_extraction" in context_card.extraction_goals:
            return "balanced"
        if "focused_extraction" in context_card.extraction_goals:
            return "focused"
        return "balanced"

    def _apply_threshold(self, key: str, delta: float):
        """Apply a threshold adjustment, clamped to [0.3, 0.95]."""
        attr_map = {
            "entity_creation_confidence": "entity_creation_confidence",
            "relation_creation_confidence": "relation_creation_confidence",
            "auto_update_threshold": "auto_update_threshold",
            "dispute_threshold": "dispute_threshold",
        }
        attr = attr_map.get(key)
        if attr and hasattr(self.state, attr):
            current = getattr(self.state, attr)
            new_value = max(0.3, min(0.95, current + delta))
            setattr(self.state, attr, new_value)
