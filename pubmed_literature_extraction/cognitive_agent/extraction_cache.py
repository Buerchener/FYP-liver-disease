#!/usr/bin/env python3
"""Lightweight bounded cache for expensive extraction results.

Memory mode is the production default and creates no files. Persistent mode is
an opt-in experiment aid backed by Python's built-in sqlite3 module.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class _Flight:
    event: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
    envelope: dict | None = None


class LightweightExtractionCache:
    """Bounded L1 cache plus optional bounded/expiring SQLite L2 cache."""

    VALID_MODES = frozenset({"off", "memory", "persistent"})

    def __init__(
        self, *, mode: str = "memory", path: str | Path = "",
        memory_max_entries: int = 256, persistent_max_entries: int = 2000,
        persistent_max_mb: int = 200, ttl_days: int = 30,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"cache mode must be one of {sorted(self.VALID_MODES)}")
        self.mode = mode
        self.path = Path(path) if path else None
        self.memory_max_entries = max(1, int(memory_max_entries))
        self.persistent_max_entries = max(1, int(persistent_max_entries))
        self.persistent_max_bytes = max(1, int(persistent_max_mb)) * 1024 * 1024
        self.ttl_seconds = max(1, int(ttl_days)) * 86400
        self._memory: OrderedDict[str, dict] = OrderedDict()
        self._inflight: dict[str, _Flight] = {}
        self._lock = threading.RLock()
        self._db_lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self._stats = {
            "memory_hits": 0, "persistent_hits": 0, "misses": 0,
            "writes": 0, "evictions": 0, "singleflight_waits": 0,
            "remote_calls_avoided": 0, "estimated_latency_saved_s": 0.0,
        }
        # Deliberately create no directory/file outside persistent mode.
        if self.mode == "persistent":
            if self.path is None:
                raise ValueError("persistent cache mode requires a path")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path), timeout=10, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS extraction_cache ("
                "cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                "created_at REAL NOT NULL, accessed_at REAL NOT NULL, size_bytes INTEGER NOT NULL)"
            )
            self._db.commit()
            self._prune_persistent()

    def _remember(self, key: str, envelope: dict) -> None:
        with self._lock:
            self._memory[key] = copy.deepcopy(envelope)
            self._memory.move_to_end(key)
            while len(self._memory) > self.memory_max_entries:
                self._memory.popitem(last=False)
                self._stats["evictions"] += 1

    def _lookup(self, key: str) -> tuple[dict | None, str]:
        if self.mode == "off":
            return None, "off"
        with self._lock:
            envelope = self._memory.get(key)
            if envelope is not None:
                created_at = float(envelope.get("created_at", 0.0) or 0.0)
                if (
                    self.mode == "persistent" and created_at
                    and time.time() - created_at > self.ttl_seconds
                ):
                    del self._memory[key]
                    envelope = None
            if envelope is not None:
                self._memory.move_to_end(key)
                self._stats["memory_hits"] += 1
                self._record_avoided(envelope)
                return copy.deepcopy(envelope), "memory_hit"
        if self.mode != "persistent" or self._db is None:
            return None, "miss"
        now = time.time()
        with self._db_lock:
            row = self._db.execute(
                "SELECT payload, created_at FROM extraction_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row and now - float(row[1]) > self.ttl_seconds:
                self._db.execute("DELETE FROM extraction_cache WHERE cache_key = ?", (key,))
                self._db.commit()
                row = None
            if row:
                self._db.execute(
                    "UPDATE extraction_cache SET accessed_at = ? WHERE cache_key = ?", (now, key)
                )
                self._db.commit()
        if not row:
            return None, "miss"
        try:
            envelope = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            with self._db_lock:
                self._db.execute("DELETE FROM extraction_cache WHERE cache_key = ?", (key,))
                self._db.commit()
            return None, "miss"
        envelope.setdefault("created_at", float(row[1]))
        with self._lock:
            self._stats["persistent_hits"] += 1
            self._record_avoided(envelope)
        self._remember(key, envelope)
        return copy.deepcopy(envelope), "persistent_hit"

    def _record_avoided(self, envelope: dict) -> None:
        self._stats["remote_calls_avoided"] += 1
        self._stats["estimated_latency_saved_s"] += float(
            envelope.get("source_latency_s", 0.0) or 0.0
        )

    def _store(self, key: str, envelope: dict) -> None:
        self._remember(key, envelope)
        if self.mode == "persistent" and self._db is not None:
            serialized = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
            now = time.time()
            with self._db_lock:
                self._db.execute(
                    "INSERT OR REPLACE INTO extraction_cache "
                    "(cache_key, payload, created_at, accessed_at, size_bytes) VALUES (?, ?, ?, ?, ?)",
                    (key, serialized, now, now, len(serialized.encode("utf-8"))),
                )
                self._db.commit()
            self._prune_persistent()
        with self._lock:
            self._stats["writes"] += 1

    def _prune_persistent(self) -> None:
        if self._db is None:
            return
        cutoff = time.time() - self.ttl_seconds
        evicted = 0
        with self._db_lock:
            cursor = self._db.execute("DELETE FROM extraction_cache WHERE created_at < ?", (cutoff,))
            evicted += max(0, int(cursor.rowcount or 0))
            count, size = self._db.execute(
                "SELECT count(*), coalesce(sum(size_bytes), 0) FROM extraction_cache"
            ).fetchone()
            while count > self.persistent_max_entries or size > self.persistent_max_bytes:
                rows = self._db.execute(
                    "SELECT cache_key, size_bytes FROM extraction_cache ORDER BY accessed_at ASC LIMIT 100"
                ).fetchall()
                if not rows:
                    break
                for cache_key, row_size in rows:
                    if count <= self.persistent_max_entries and size <= self.persistent_max_bytes:
                        break
                    self._db.execute("DELETE FROM extraction_cache WHERE cache_key = ?", (cache_key,))
                    count -= 1
                    size -= int(row_size or 0)
                    evicted += 1
            self._db.commit()
        if evicted:
            with self._lock:
                self._stats["evictions"] += evicted

    def get_or_compute(
        self, key: str, factory: Callable[[], tuple[dict, float]],
        *, cacheable: Callable[[dict], bool],
    ) -> tuple[dict, str]:
        """Return one result per key; concurrent callers share the same work."""
        if self.mode == "off":
            payload, _ = factory()
            return payload, "disabled"
        envelope, status = self._lookup(key)
        if envelope is not None:
            return envelope["payload"], status
        with self._lock:
            flight = self._inflight.get(key)
            owner = flight is None
            if owner:
                flight = _Flight()
                self._inflight[key] = flight
                self._stats["misses"] += 1
            else:
                self._stats["singleflight_waits"] += 1
        if not owner:
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            envelope, wait_status = self._lookup(key)
            if envelope is None:
                # Non-cacheable results still need one shared in-flight value.
                envelope = getattr(flight, "envelope", None)
                if envelope is None:
                    raise RuntimeError("single-flight completed without a result")
                return copy.deepcopy(envelope["payload"]), "singleflight_shared"
            return envelope["payload"], "singleflight_shared"
        try:
            payload, latency_s = factory()
            envelope = {
                "payload": payload,
                "source_latency_s": float(latency_s or 0.0),
                "created_at": time.time(),
            }
            flight.envelope = copy.deepcopy(envelope)
            if cacheable(payload):
                self._store(key, envelope)
            return payload, "miss"
        except BaseException as exc:
            flight.error = exc
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)
                flight.event.set()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            data = dict(self._stats)
            completed = (
                self._stats["memory_hits"]
                + self._stats["persistent_hits"]
                + self._stats["misses"]
            )
            data.update({
                "mode": self.mode, "memory_entries": len(self._memory),
                "memory_max_entries": self.memory_max_entries,
                "hit_rate": round(
                    (self._stats["memory_hits"] + self._stats["persistent_hits"])
                    / completed, 4,
                ) if completed else 0.0,
                "estimated_latency_saved_s": round(
                    float(self._stats["estimated_latency_saved_s"]), 3
                ),
            })
        if self.mode == "persistent" and self._db is not None:
            with self._db_lock:
                count, size = self._db.execute(
                    "SELECT count(*), coalesce(sum(size_bytes), 0) FROM extraction_cache"
                ).fetchone()
            data.update({
                "persistent_entries": int(count),
                "persistent_size_mb": round(int(size) / 1024 / 1024, 3),
                "persistent_max_entries": self.persistent_max_entries,
                "persistent_max_mb": round(self.persistent_max_bytes / 1024 / 1024, 1),
                "ttl_days": round(self.ttl_seconds / 86400, 1),
                "path": str(self.path),
            })
        return data

    def close(self) -> None:
        if self._db is not None:
            with self._db_lock:
                self._db.close()
                self._db = None
