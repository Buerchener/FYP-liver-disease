#!/usr/bin/env python3
"""Section/sentence aligned extraction chunks that remain exact source spans."""

from __future__ import annotations

from dataclasses import dataclass

from cognitive_agent.evidence_units import EvidenceUnit


@dataclass(frozen=True)
class ArticleChunk:
    chunk_id: str
    text: str
    char_start: int
    char_end: int
    sections: tuple[str, ...]
    overlapped_parent_sentences: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "char_count": self.char_end - self.char_start,
            "sections": list(self.sections),
            "overlapped_parent_sentences": self.overlapped_parent_sentences,
        }


class ArticleChunker:
    """Create 2–3 contiguous chunks without rewriting article text."""

    def __init__(
        self,
        max_chars: int = 1800,
        overlap_parent_sentences: int = 1,
        complexity_min_chars: int = 1400,
        max_chunks: int = 3,
    ):
        self.max_chars = max(800, int(max_chars))
        self.overlap_parent_sentences = max(0, min(2, int(overlap_parent_sentences)))
        self.complexity_min_chars = max(600, int(complexity_min_chars))
        self.max_chunks = max(2, min(4, int(max_chunks)))

    def should_chunk(self, text: str, high_complexity: bool) -> bool:
        return len(text) > self.max_chars or (
            high_complexity and len(text) >= self.complexity_min_chars
        )

    def build(
        self,
        text: str,
        units: list[EvidenceUnit],
        *,
        high_complexity: bool = False,
    ) -> list[ArticleChunk]:
        if not text or not self.should_chunk(text, high_complexity) or not units:
            return [ArticleChunk("c001", text, 0, len(text), self._sections(units))]

        parents: list[dict] = []
        by_parent: dict[str, list[EvidenceUnit]] = {}
        for unit in units:
            by_parent.setdefault(unit.parent_sentence_id, []).append(unit)
        for parent_id, children in by_parent.items():
            ordered = sorted(children, key=lambda item: item.char_start)
            parents.append({
                "parent_id": parent_id,
                "start": ordered[0].char_start,
                "end": ordered[-1].char_end,
                "sections": tuple(dict.fromkeys(item.section for item in ordered)),
            })
        parents.sort(key=lambda item: item["start"])
        if len(parents) < 2:
            return [ArticleChunk("c001", text, 0, len(text), self._sections(units))]

        chunks: list[ArticleChunk] = []
        next_parent = 0
        while next_parent < len(parents) and len(chunks) < self.max_chunks:
            is_first = not chunks
            overlap_start_index = (
                next_parent if is_first
                else max(0, next_parent - self.overlap_parent_sentences)
            )
            start = 0 if is_first else parents[overlap_start_index]["start"]
            end_index = next_parent
            while end_index + 1 < len(parents):
                proposed_end = parents[end_index + 1]["end"]
                if proposed_end - start > self.max_chars and end_index >= next_parent:
                    break
                end_index += 1
            if len(chunks) == self.max_chunks - 1:
                end_index = len(parents) - 1
            end = len(text) if end_index == len(parents) - 1 else parents[end_index]["end"]
            selected = parents[overlap_start_index:end_index + 1]
            sections = tuple(dict.fromkeys(
                section for parent in selected for section in parent["sections"]
            ))
            chunks.append(ArticleChunk(
                chunk_id=f"c{len(chunks) + 1:03d}",
                text=text[start:end],
                char_start=start,
                char_end=end,
                sections=sections,
                overlapped_parent_sentences=(
                    next_parent - overlap_start_index if not is_first else 0
                ),
            ))
            next_parent = end_index + 1

        return chunks

    @staticmethod
    def _sections(units: list[EvidenceUnit]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(unit.section for unit in units)) or ("ABSTRACT",)
