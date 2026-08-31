#!/usr/bin/env python3
"""Span-preserving semantic chunk plans over deterministic sentence IDs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from cognitive_agent.article_chunker import ArticleChunk
from cognitive_agent.evidence_units import ArticleEvidenceReader, EvidenceUnit


PROMPT_VERSION = "semantic-chunk-boundaries-v1"
SYSTEM_PROMPT = """You plan semantic boundaries for biomedical abstract extraction.
Return strict JSON only. Group consecutive sentence IDs into 2 or 3 coherent
chunks. Preserve every ID exactly once and in order. Never rewrite, summarize,
drop, duplicate, or reorder a sentence. Prefer boundaries at section or topic
changes, but keep entity chains and relation evidence together. Keep each chunk
under 2000 source characters when possible."""


class SemanticChunkPlanError(ValueError):
    """The model response cannot be converted to a safe source-span plan."""


@dataclass(frozen=True)
class SemanticSentence:
    sentence_id: str
    section: str
    text: str
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "section": self.section,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "char_count": self.char_end - self.char_start,
        }


@dataclass(frozen=True)
class SemanticGroup:
    start_id: str
    end_id: str
    topic: str
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_id": self.start_id,
            "end_id": self.end_id,
            "topic": self.topic,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "char_count": self.char_end - self.char_start,
        }


@dataclass(frozen=True)
class SemanticChunkPlan:
    source_sha256: str
    prompt_version: str
    sentences: tuple[SemanticSentence, ...]
    groups: tuple[SemanticGroup, ...]
    chunks: tuple[ArticleChunk, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "prompt_version": self.prompt_version,
            "sentences": [item.to_dict() for item in self.sentences],
            "groups": [item.to_dict() for item in self.groups],
            "chunks": [item.to_dict() for item in self.chunks],
        }


def source_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parent_sentences(text: str, units: list[EvidenceUnit]) -> list[SemanticSentence]:
    parents = ArticleEvidenceReader.parent_units(text, units)
    return [
        SemanticSentence(
            sentence_id=item.parent_sentence_id,
            section=item.section,
            text=item.text,
            char_start=item.char_start,
            char_end=item.char_end,
        )
        for item in parents
    ]


def build_user_prompt(*, pmid: str, title: str, sentences: list[SemanticSentence]) -> str:
    payload = {
        "pmid": str(pmid),
        "title": str(title or ""),
        "constraints": {
            "group_count": "2_or_3",
            "consecutive_only": True,
            "complete_coverage": True,
            "target_chars": [800, 1800],
            "max_chars": 2000,
        },
        "sentences": [item.to_dict() for item in sentences],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def schema_hint() -> dict[str, Any]:
    return {
        "groups": [
            {"start_id": "s001", "end_id": "s004", "topic": "short topic label"}
        ]
    }


def plan_from_payload(
    text: str,
    units: list[EvidenceUnit],
    payload: dict[str, Any],
    *,
    max_group_chars: int = 2000,
    overlap_parent_sentences: int = 1,
) -> SemanticChunkPlan:
    sentences = parent_sentences(text, units)
    if len(sentences) < 2:
        raise SemanticChunkPlanError("semantic chunking requires at least two sentences")
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or len(raw_groups) not in {2, 3}:
        raise SemanticChunkPlanError("groups must contain exactly 2 or 3 ranges")

    indexes = {item.sentence_id: index for index, item in enumerate(sentences)}
    expected_start = 0
    groups: list[SemanticGroup] = []
    ranges: list[tuple[int, int, str]] = []
    for raw in raw_groups:
        if not isinstance(raw, dict):
            raise SemanticChunkPlanError("each semantic group must be an object")
        start_id = str(raw.get("start_id", ""))
        end_id = str(raw.get("end_id", ""))
        if start_id not in indexes or end_id not in indexes:
            raise SemanticChunkPlanError("semantic group references an unknown sentence ID")
        start_index, end_index = indexes[start_id], indexes[end_id]
        if start_index != expected_start or end_index < start_index:
            raise SemanticChunkPlanError("semantic groups must be ordered, contiguous, and non-overlapping")
        start = 0 if not groups else sentences[start_index].char_start
        end = len(text) if len(groups) == len(raw_groups) - 1 else sentences[end_index].char_end
        sentence_span = max(
            item.char_end - item.char_start for item in sentences[start_index:end_index + 1]
        )
        if end - start > max_group_chars and sentence_span <= max_group_chars:
            raise SemanticChunkPlanError("semantic group exceeds the maximum source length")
        topic = str(raw.get("topic", "") or "").strip()[:160]
        groups.append(SemanticGroup(start_id, end_id, topic, start, end))
        ranges.append((start_index, end_index, topic))
        expected_start = end_index + 1
    if expected_start != len(sentences):
        raise SemanticChunkPlanError("semantic groups do not cover every sentence")

    chunks: list[ArticleChunk] = []
    overlap = max(0, min(1, int(overlap_parent_sentences)))
    for index, (start_index, end_index, topic) in enumerate(ranges):
        overlapped_start = start_index if index == 0 else max(0, start_index - overlap)
        start = 0 if index == 0 else sentences[overlapped_start].char_start
        end = len(text) if index == len(ranges) - 1 else sentences[end_index].char_end
        selected = sentences[overlapped_start:end_index + 1]
        sections = tuple(dict.fromkeys(item.section for item in selected)) or ("ABSTRACT",)
        chunks.append(ArticleChunk(
            chunk_id=f"c{index + 1:03d}",
            text=text[start:end],
            char_start=start,
            char_end=end,
            sections=sections,
            overlapped_parent_sentences=start_index - overlapped_start,
            topic=topic,
            owner_sentence_ids=tuple(
                item.sentence_id for item in sentences[start_index:end_index + 1]
            ),
            context_sentence_ids=tuple(
                item.sentence_id for item in sentences[overlapped_start:start_index]
            ),
            strategy="llm_sentence_boundary_plan",
        ))

    return SemanticChunkPlan(
        source_sha256=source_sha256(text),
        prompt_version=PROMPT_VERSION,
        sentences=tuple(sentences),
        groups=tuple(groups),
        chunks=tuple(chunks),
    )


def chunks_from_manifest(
    text: str,
    units: list[EvidenceUnit],
    record: dict[str, Any],
) -> list[ArticleChunk]:
    expected_hash = str(record.get("source_sha256", ""))
    if expected_hash != source_sha256(text):
        raise SemanticChunkPlanError("semantic chunk source hash mismatch")
    payload = {"groups": record.get("groups")}
    plan = plan_from_payload(text, units, payload)
    manifest_chunks = record.get("chunks") or []
    if manifest_chunks:
        expected = [item.to_dict() for item in plan.chunks]
        stable_keys = ("chunk_id", "char_start", "char_end", "overlapped_parent_sentences", "topic")
        observed_stable = [{key: item.get(key) for key in stable_keys} for item in manifest_chunks]
        expected_stable = [{key: item.get(key) for key in stable_keys} for item in expected]
        if observed_stable != expected_stable:
            raise SemanticChunkPlanError("semantic chunk manifest offsets do not match source")
    return list(plan.chunks)
