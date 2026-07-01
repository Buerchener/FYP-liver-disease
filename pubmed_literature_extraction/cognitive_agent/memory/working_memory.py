#!/usr/bin/env python3
"""
cognitive_agent/memory/working_memory.py — Agent Working Memory

Transient, per-article scratchpad cache.
Cleared at the start of each process_article() invocation.
Caches Neo4j lookups to avoid redundant queries within a single article.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkingMemory:
    """Per-article working memory — transient cache for the current article."""

    current_article_pmid: str = ""

    # Cached Neo4j entity lookups: key = "mention|entity_type" -> dict
    cached_entity_lookups: dict[str, dict] = field(default_factory=dict)

    # Cached Neo4j relation lookups: key = "subject|predicate|object" -> dict
    cached_relation_lookups: dict[str, dict] = field(default_factory=dict)

    # Extraction targets derived from ContextCard
    extraction_targets: list[str] = field(default_factory=list)

    # Quality scores for relations in this article
    recent_quality: list[float] = field(default_factory=list)

    # Conflict flags
    has_conflicts: bool = False

    def cache_entity(self, key: str, value: dict):
        """Cache a Neo4j entity lookup result."""
        self.cached_entity_lookups[key] = value

    def get_cached_entity(self, key: str) -> Optional[dict]:
        """Get a cached entity lookup, or None."""
        return self.cached_entity_lookups.get(key)

    def cache_relation(self, key: str, value: dict):
        """Cache a Neo4j relation lookup result."""
        self.cached_relation_lookups[key] = value

    def get_cached_relation(self, key: str) -> Optional[dict]:
        """Get a cached relation lookup, or None."""
        return self.cached_relation_lookups.get(key)

    def clear_article_session(self):
        """Reset per-article state. Called at start of each process_article()."""
        self.current_article_pmid = ""
        self.cached_entity_lookups.clear()
        self.cached_relation_lookups.clear()
        self.extraction_targets.clear()
        self.recent_quality.clear()
        self.has_conflicts = False
