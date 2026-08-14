#!/usr/bin/env python3
"""Content-addressed cache and parallel deterministic article preparation."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from cognitive_agent.hybrid_article_profiler import HybridProfile, rule_profile


CACHE_VERSION = "article-preparation-v2"


@dataclass
class _PreparationFlight:
    event: threading.Event = field(default_factory=threading.Event)
    value: Any = None
    error: BaseException | None = None


@dataclass(frozen=True)
class PreparedArticle:
    content_hash: str
    profile: HybridProfile
    evidence_units: list
    abbreviation_map: Any
    golden_selection: Any
    cache_hits: dict[str, bool]


class ContentAddressedCache:
    """Thread-safe process-local cache; values never contain credentials."""

    def __init__(self, version: str = CACHE_VERSION, max_entries: int = 1024):
        self.version = version
        self.max_entries = max(1, int(max_entries))
        self._values: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._inflight: dict[tuple[str, str], _PreparationFlight] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._singleflight_waits = 0

    def content_hash(self, title: str, abstract: str) -> str:
        payload = f"{self.version}\0{title}\0{abstract}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get_or_compute(
        self, content_hash: str, component: str, factory: Callable[[], Any],
    ) -> tuple[Any, bool]:
        key = (content_hash, component)
        with self._lock:
            if key in self._values and self._values[key] is not None:
                self._hits += 1
                self._values.move_to_end(key)
                return self._values[key], True
            if key in self._values:
                # A partial/corrupted in-process cache entry is safe to evict;
                # deterministic components will be recomputed below.
                del self._values[key]
            flight = self._inflight.get(key)
            owner = flight is None
            if owner:
                flight = _PreparationFlight()
                self._inflight[key] = flight
            else:
                self._singleflight_waits += 1
        if not owner:
            flight.event.wait()
            with self._lock:
                if flight.error is not None:
                    raise flight.error
                self._hits += 1
                if key in self._values:
                    self._values.move_to_end(key)
                return flight.value, True
        try:
            value = factory()
            if value is None:
                raise ValueError(f"component {component} returned no cacheable value")
        except BaseException as exc:
            with self._lock:
                flight.error = exc
                self._inflight.pop(key, None)
                flight.event.set()
            raise
        with self._lock:
            flight.value = value
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)
                self._evictions += 1
            self._misses += 1
            self._inflight.pop(key, None)
            flight.event.set()
        return value, False

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "version": self.version,
                "entries": len(self._values),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "singleflight_waits": self._singleflight_waits,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }


class ParallelArticlePreprocessor:
    """Run independent local preparation tasks concurrently and cache them."""

    def __init__(self, evidence_reader, abbreviation_detector, golden_selector, *,
                 max_workers: int = 4, cache_max_entries: int = 1024):
        self.evidence_reader = evidence_reader
        self.abbreviation_detector = abbreviation_detector
        self.golden_selector = golden_selector
        self.cache = ContentAddressedCache(max_entries=cache_max_entries)
        self._executor = ThreadPoolExecutor(
            max_workers=max(2, int(max_workers)), thread_name_prefix="article-prep"
        )

    def prepare(
        self, *, title: str, abstract: str, text: str, study_type: str,
        max_examples: int, document_id: str,
    ) -> PreparedArticle:
        digest = self.cache.content_hash(title, abstract)
        specs = {
            "profile:rules-v2": lambda: rule_profile(title, abstract),
            "evidence_units:reader-v1": lambda: self.evidence_reader.read(text),
            "abbreviations:schwartz-hearst-v1": lambda: self.abbreviation_detector.detect(text),
            f"golden_examples:v1:{study_type}:{max_examples}:{document_id}": lambda: self.golden_selector.select(
                text, study_type, max_examples=max_examples, document_id=document_id,
            ),
        }
        futures = {
            name: self._executor.submit(self.cache.get_or_compute, digest, name, factory)
            for name, factory in specs.items()
        }
        values: dict[str, Any] = {}
        hits: dict[str, bool] = {}
        for name, future in futures.items():
            values[name], hits[name] = future.result()
        profile_key, evidence_key, abbreviation_key, golden_key = specs.keys()
        return PreparedArticle(
            content_hash=digest, profile=values[profile_key],
            evidence_units=values[evidence_key],
            abbreviation_map=values[abbreviation_key],
            golden_selection=values[golden_key], cache_hits={
                "profile": hits[profile_key], "evidence_units": hits[evidence_key],
                "abbreviations": hits[abbreviation_key], "golden_examples": hits[golden_key],
            },
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
