#!/usr/bin/env python3
"""
cognitive_agent/memory/episodic_memory.py — Episodic Memory

Persistent (session-scoped) record of all agent decisions.
Used by SelfReflection for cross-document pattern discovery:
- find_emerging_entities(): entities that appear as NOVEL across multiple articles
- find_emerging_relations(): relation types that appear frequently
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class Episode:
    """A single decision recorded by the agent."""
    pmid: str
    timestamp: float
    article_title: str = ""
    action_type: str = ""           # CREATE_ENTITY | CREATE_RELATION | UPDATE_RELATION | MARK_DISPUTED | DISCARD | PROPOSE_HYPOTHESIS | NO_ACTION
    entity_type: str = ""
    entity_name: str = ""
    predicate: str = ""
    object_name: str = ""
    object_type: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    evidence: str = ""
    study_type: str = ""
    outcome: str = ""              # accepted | rejected | manual_review
    reason_code: str = ""
    supervision_source: str = "agent"  # agent | deterministic | curator | gold

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "action_type": self.action_type,
            "entity_type": self.entity_type,
            "entity_name": self.entity_name,
            "predicate": self.predicate,
            "object_name": self.object_name,
            "object_type": self.object_type,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "study_type": self.study_type,
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "supervision_source": self.supervision_source,
        }


@dataclass
class EpisodicMemory:
    """Session-scoped decision history for pattern discovery (thread-safe)."""

    episodes: list[Episode] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, episode: Episode):
        """Record a decision episode."""
        with self._lock:
            self.episodes.append(episode)

    def find_emerging_entities(self, min_frequency: int = 3) -> list[dict]:
        """Find novel entities that appear across multiple articles.

        Returns entities with frequency >= min_frequency, sorted by frequency.
        """
        novel_counter: Counter = Counter()
        for ep in self.episodes:
            if ep.action_type == "CREATE_ENTITY":
                novel_counter[(ep.entity_name, ep.entity_type)] += 1

        return [
            {"name": name, "type": etype, "frequency": freq}
            for (name, etype), freq in novel_counter.most_common()
            if freq >= min_frequency
        ]

    def find_emerging_relations(self, min_frequency: int = 3) -> list[dict]:
        """Find relation patterns that appear frequently across articles.

        Returns (subject_type, predicate, object_type) tuples with frequency >= min_frequency.
        """
        rel_counter: Counter = Counter()
        for ep in self.episodes:
            if ep.action_type in ("CREATE_RELATION", "PROPOSE_HYPOTHESIS"):
                rel_counter[ep.predicate] += 1

        return [
            {"predicate": pred, "frequency": freq}
            for pred, freq in rel_counter.most_common()
            if freq >= min_frequency
        ]

    def find_contradiction_patterns(self) -> list[dict]:
        """Find entities or relations involved in multiple conflicts/disputes."""
        disputed: Counter = Counter()
        for ep in self.episodes:
            if ep.action_type == "MARK_DISPUTED":
                disputed[(ep.entity_name, ep.predicate, ep.object_name)] += 1

        return [
            {"entity": name, "predicate": pred, "object": obj, "frequency": freq}
            for (name, pred, obj), freq in disputed.most_common()
            if freq >= 2
        ]

    def recent_decisions(self, n: int = 20) -> list[Episode]:
        """Return the most recent N decisions."""
        return self.episodes[-n:] if self.episodes else []

    def trusted_hard_negatives(
        self,
        subject_type: str,
        object_type: str,
        predicate: str = "",
        limit: int = 4,
    ) -> list[Episode]:
        """Return only curator/gold rejections for contrastive prompting.

        Model self-judgments are deliberately excluded so an error cannot be
        reinforced across articles.
        """
        matches = [
            episode for episode in self.episodes
            if episode.outcome == "rejected"
            and episode.supervision_source in {"curator", "gold"}
            and episode.entity_type == subject_type
            and episode.object_type == object_type
            and (not predicate or episode.predicate == predicate)
        ]
        return matches[-max(0, limit):]

    def count_by_action(self, action_type: str) -> int:
        """Count decisions of a given action type."""
        return sum(1 for ep in self.episodes if ep.action_type == action_type)

    def total_actions(self) -> int:
        return len(self.episodes)
