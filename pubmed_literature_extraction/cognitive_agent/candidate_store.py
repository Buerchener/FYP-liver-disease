#!/usr/bin/env python3
"""Persistent ledger for literature-derived relation candidates.

The ledger is deliberately separate from Neo4j.  It records every verified
relation candidate, including semantic-only and human-review cases, so broad
literature observations remain auditable without expanding the main KG schema.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_STORE_MODES = frozenset({"off", "sqlite"})


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _candidate_key(pmid: str, relation: dict[str, Any]) -> str:
    candidate_id = str(relation.get("candidate_id", "") or "").strip()
    if candidate_id:
        version = max(1, int(relation.get("candidate_version", 1) or 1))
        return f"{pmid}:{candidate_id}:v{version}"
    fields = (
        pmid,
        str(relation.get("subject", "") or "").casefold(),
        str(relation.get("subject_type", "") or ""),
        str(relation.get("predicate", "") or "").upper(),
        str(relation.get("object", "") or "").casefold(),
        str(relation.get("object_type", "") or ""),
        str(relation.get("evidence", "") or "").casefold(),
    )
    digest = hashlib.sha1("|".join(fields).encode("utf-8")).hexdigest()[:16]
    return f"{pmid}:c-{digest}"


@dataclass(frozen=True)
class CandidateStoreWriteSummary:
    mode: str
    path: str
    pmid: str
    relation_count: int
    import_ready_count: int
    human_review_count: int
    semantic_only_count: int
    blocked_count: int
    candidate_schema_valid_count: int = 0
    write_contract_valid_count: int = 0
    candidate_only_count: int = 0
    out_of_scope_count: int = 0
    written: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class CandidateRelationStore:
    """Append/update SQLite store for final verified article candidates."""

    def __init__(self, *, mode: str = "off", path: str | Path = ""):
        if mode not in VALID_STORE_MODES:
            raise ValueError(f"candidate store mode must be one of {sorted(VALID_STORE_MODES)}")
        self.mode = mode
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self._db: sqlite3.Connection | None = None
        if self.mode == "sqlite":
            if self.path is None:
                raise ValueError("sqlite candidate store requires a path")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path), timeout=10, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()

    def _init_schema(self) -> None:
        if self._db is None:
            return
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_relations (
                candidate_key TEXT PRIMARY KEY,
                pmid TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                candidate_version INTEGER NOT NULL DEFAULT 1,
                parent_version INTEGER NOT NULL DEFAULT 0,
                candidate_lane TEXT NOT NULL DEFAULT 'extracted_hint',
                subject TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                object_type TEXT NOT NULL,
                relation_direction TEXT NOT NULL DEFAULT 'UNKNOWN',
                association_sign TEXT NOT NULL DEFAULT 'UNKNOWN',
                expression_change TEXT NOT NULL DEFAULT 'UNKNOWN',
                activity_change TEXT NOT NULL DEFAULT 'UNKNOWN',
                factual_status TEXT NOT NULL,
                semantic_status TEXT NOT NULL,
                write_status TEXT NOT NULL,
                scope_status TEXT NOT NULL DEFAULT 'IN_SCOPE',
                claim_role TEXT NOT NULL,
                import_ready INTEGER NOT NULL,
                schema_valid INTEGER NOT NULL,
                candidate_schema_valid INTEGER NOT NULL DEFAULT 0,
                write_contract_valid INTEGER NOT NULL DEFAULT 0,
                evidence TEXT NOT NULL,
                evidence_spans_json TEXT NOT NULL,
                evidence_pack_json TEXT NOT NULL DEFAULT '{}',
                quality_flags_json TEXT NOT NULL,
                semantic_reasons_json TEXT NOT NULL,
                write_reasons_json TEXT NOT NULL,
                schema_gap_reasons_json TEXT NOT NULL DEFAULT '[]',
                write_contract_version TEXT NOT NULL DEFAULT '',
                verification_policy_version TEXT NOT NULL DEFAULT '',
                relation_json TEXT NOT NULL,
                run_id TEXT NOT NULL,
                title TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._ensure_column(
            "candidate_schema_valid", "INTEGER NOT NULL DEFAULT 0"
        )
        self._ensure_column(
            "write_contract_valid", "INTEGER NOT NULL DEFAULT 0"
        )
        self._ensure_column(
            "schema_gap_reasons_json", "TEXT NOT NULL DEFAULT '[]'"
        )
        self._ensure_column(
            "write_contract_version", "TEXT NOT NULL DEFAULT ''"
        )
        self._ensure_column(
            "scope_status", "TEXT NOT NULL DEFAULT 'IN_SCOPE'"
        )
        for name, definition in (
            ("candidate_version", "INTEGER NOT NULL DEFAULT 1"),
            ("parent_version", "INTEGER NOT NULL DEFAULT 0"),
            ("candidate_lane", "TEXT NOT NULL DEFAULT 'extracted_hint'"),
            ("relation_direction", "TEXT NOT NULL DEFAULT 'UNKNOWN'"),
            ("association_sign", "TEXT NOT NULL DEFAULT 'UNKNOWN'"),
            ("expression_change", "TEXT NOT NULL DEFAULT 'UNKNOWN'"),
            ("activity_change", "TEXT NOT NULL DEFAULT 'UNKNOWN'"),
            ("evidence_pack_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("verification_policy_version", "TEXT NOT NULL DEFAULT ''"),
        ):
            self._ensure_column(name, definition)
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidate_relations_pmid "
            "ON candidate_relations(pmid)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidate_relations_status "
            "ON candidate_relations(write_status, semantic_status, factual_status)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidate_relations_predicate "
            "ON candidate_relations(predicate, subject_type, object_type)"
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_lineage_audit (
                audit_key TEXT PRIMARY KEY,
                pmid TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                candidate_version INTEGER NOT NULL,
                parent_candidate_id TEXT NOT NULL DEFAULT '',
                parent_version INTEGER NOT NULL DEFAULT 0,
                candidate_lane TEXT NOT NULL DEFAULT '',
                disposition TEXT NOT NULL DEFAULT 'KEPT',
                provenance_json TEXT NOT NULL DEFAULT '{}',
                run_id TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL
            )
            """
        )
        self._db.commit()

    def _ensure_column(self, name: str, definition: str) -> None:
        if self._db is None:
            return
        columns = {
            row[1] for row in self._db.execute("PRAGMA table_info(candidate_relations)")
        }
        if name not in columns:
            self._db.execute(
                f"ALTER TABLE candidate_relations ADD COLUMN {name} {definition}"
            )

    def record_verified(
        self, *, pmid: str, title: str = "", run_id: str = "",
        verified: Any,
    ) -> CandidateStoreWriteSummary:
        relations = [
            item.to_dict() if hasattr(item, "to_dict") else dict(item)
            for item in getattr(verified, "relations", []) or []
        ]
        for relation in relations:
            flags = set(relation.get("quality_flags", []) or [])
            relation["scope_status"] = str(
                relation.get("scope_status")
                or ("OUT_OF_SCOPE" if "article_out_of_scope" in flags else "IN_SCOPE")
            ).upper()
        summary = CandidateStoreWriteSummary(
            mode=self.mode,
            path=str(self.path or ""),
            pmid=str(pmid or ""),
            relation_count=len(relations),
            import_ready_count=sum(r.get("write_status") == "IMPORT_READY" for r in relations),
            human_review_count=sum(
                r.get("write_status") == "HUMAN_REVIEW"
                and r.get("scope_status") != "OUT_OF_SCOPE"
                for r in relations
            ),
            semantic_only_count=sum(r.get("write_status") == "SEMANTIC_ONLY" for r in relations),
            blocked_count=sum(r.get("write_status") == "BLOCKED" for r in relations),
            candidate_schema_valid_count=sum(
                bool(r.get("candidate_schema_valid", r.get("schema_valid", False)))
                for r in relations
            ),
            write_contract_valid_count=sum(
                bool(r.get("write_contract_valid", False)) for r in relations
            ),
            candidate_only_count=sum(
                not bool(r.get("write_contract_valid", False))
                and str(r.get("factual_status", "")).upper() != "REJECTED"
                and str(r.get("semantic_status", "")).upper() != "REJECTED"
                for r in relations
            ),
            out_of_scope_count=sum(
                r.get("scope_status") == "OUT_OF_SCOPE" for r in relations
            ),
            written=self.mode == "sqlite",
        )
        if self.mode == "off" or self._db is None:
            return summary

        now = time.time()
        rows = []
        for relation in relations:
            rows.append((
                _candidate_key(summary.pmid, relation),
                summary.pmid,
                str(relation.get("candidate_id", "") or ""),
                max(1, int(relation.get("candidate_version", 1) or 1)),
                max(0, int(relation.get("parent_version", 0) or 0)),
                str(relation.get("candidate_lane", "extracted_hint") or "extracted_hint"),
                str(relation.get("subject", "") or ""),
                str(relation.get("subject_type", "") or ""),
                str(relation.get("predicate", "") or "").upper(),
                str(relation.get("object", "") or ""),
                str(relation.get("object_type", "") or ""),
                str(relation.get("relation_direction", "UNKNOWN") or "UNKNOWN"),
                str(relation.get("association_sign", "UNKNOWN") or "UNKNOWN"),
                str(relation.get("expression_change", "UNKNOWN") or "UNKNOWN"),
                str(relation.get("activity_change", "UNKNOWN") or "UNKNOWN"),
                str(relation.get("factual_status", "") or ""),
                str(relation.get("semantic_status", "") or ""),
                str(relation.get("write_status", "") or ""),
                str(relation.get("scope_status", "IN_SCOPE") or "IN_SCOPE"),
                str(relation.get("claim_role", "") or ""),
                int(bool(relation.get("import_ready", False))),
                int(bool(relation.get("schema_valid", False))),
                int(bool(relation.get("candidate_schema_valid", relation.get("schema_valid", False)))),
                int(bool(relation.get("write_contract_valid", False))),
                str(relation.get("evidence", "") or ""),
                _stable_json(relation.get("evidence_spans", []) or []),
                _stable_json(relation.get("evidence_pack", {}) or {}),
                _stable_json(relation.get("quality_flags", []) or []),
                _stable_json(relation.get("semantic_reasons", []) or []),
                _stable_json(relation.get("write_reasons", []) or []),
                _stable_json(relation.get("schema_gap_reasons", []) or []),
                str(relation.get("write_contract_version", "") or ""),
                str(relation.get("verification_policy_version", "") or ""),
                _stable_json(relation),
                str(run_id or ""),
                str(title or ""),
                now,
            ))
        with self._lock:
            self._db.executemany(
                """
                INSERT OR REPLACE INTO candidate_relations (
                    candidate_key, pmid, candidate_id, candidate_version,
                    parent_version, candidate_lane, subject, subject_type,
                    predicate, object, object_type, relation_direction,
                    association_sign, expression_change, activity_change, factual_status,
                    semantic_status, write_status, scope_status, claim_role, import_ready,
                    schema_valid, candidate_schema_valid, write_contract_valid,
                    evidence, evidence_spans_json, evidence_pack_json,
                    quality_flags_json, semantic_reasons_json, write_reasons_json,
                    schema_gap_reasons_json, write_contract_version,
                    verification_policy_version, relation_json,
                    run_id, title, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            lineage_rows = []
            for relation in relations:
                instances = relation.get("claim_instances", []) or [relation]
                for instance in instances:
                    candidate_id = str(instance.get("candidate_id", "") or "")
                    if not candidate_id:
                        continue
                    version = max(1, int(instance.get("candidate_version", 1) or 1))
                    audit_key = f"{summary.pmid}|{candidate_id}|v{version}"
                    lineage_rows.append((
                        audit_key, summary.pmid, candidate_id, version,
                        str(instance.get("parent_candidate_id", "") or ""),
                        max(0, int(instance.get("parent_version", 0) or 0)),
                        str(instance.get("candidate_lane", "") or ""),
                        str(instance.get("candidate_disposition", "KEPT") or "KEPT"),
                        _stable_json(instance), str(run_id or ""), now,
                    ))
            self._db.executemany(
                """
                INSERT OR REPLACE INTO candidate_lineage_audit (
                    audit_key, pmid, candidate_id, candidate_version,
                    parent_candidate_id, parent_version, candidate_lane,
                    disposition, provenance_json, run_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                lineage_rows,
            )
            self._db.commit()
        return summary

    def stats(self) -> dict[str, Any]:
        if self.mode == "off" or self._db is None:
            return {"mode": self.mode, "path": str(self.path or ""), "entries": 0}
        with self._lock:
            entries = self._db.execute("SELECT count(*) FROM candidate_relations").fetchone()[0]
            by_status = {
                row[0]: row[1]
                for row in self._db.execute(
                    "SELECT write_status, count(*) FROM candidate_relations GROUP BY write_status"
                ).fetchall()
            }
            by_scope = {
                row[0]: row[1]
                for row in self._db.execute(
                    "SELECT scope_status, count(*) FROM candidate_relations GROUP BY scope_status"
                ).fetchall()
            }
            gap_rows = self._db.execute(
                "SELECT schema_gap_reasons_json FROM candidate_relations "
                "WHERE schema_gap_reasons_json != '[]'"
            ).fetchall()
        gap_counts: dict[str, int] = {}
        for (raw_reasons,) in gap_rows:
            try:
                reasons = json.loads(raw_reasons)
            except (TypeError, json.JSONDecodeError):
                reasons = []
            for reason in reasons if isinstance(reasons, list) else []:
                gap_counts[str(reason)] = gap_counts.get(str(reason), 0) + 1
        return {
            "mode": self.mode,
            "path": str(self.path or ""),
            "entries": int(entries),
            "by_write_status": by_status,
            "by_scope_status": by_scope,
            "by_schema_gap_reason": dict(sorted(gap_counts.items())),
        }

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None
