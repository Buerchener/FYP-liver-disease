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
    topic: str = ""
    owner_sentence_ids: tuple[str, ...] = ()
    context_sentence_ids: tuple[str, ...] = ()
    strategy: str = "one_shot"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "char_count": self.char_end - self.char_start,
            "sections": list(self.sections),
            "overlapped_parent_sentences": self.overlapped_parent_sentences,
            "topic": self.topic,
            "owner_sentence_ids": list(self.owner_sentence_ids),
            "context_sentence_ids": list(self.context_sentence_ids),
            "strategy": self.strategy,
        }


class ArticleChunker:
    """Build exact adaptive windows with disjoint owner sentences."""

    def __init__(
        self,
        max_chars: int = 1800,
        overlap_parent_sentences: int = 1,
        complexity_min_chars: int = 1400,
        max_chunks: int = 6,
        one_shot_max_chars: int = 2200,
        one_shot_max_sentences: int = 12,
    ):
        self.max_chars = max(800, int(max_chars))
        self.overlap_parent_sentences = max(0, min(2, int(overlap_parent_sentences)))
        self.complexity_min_chars = max(600, int(complexity_min_chars))
        self.max_chunks = max(2, min(8, int(max_chunks)))
        self.one_shot_max_chars = max(800, int(one_shot_max_chars))
        self.one_shot_max_sentences = max(2, int(one_shot_max_sentences))

    def should_chunk(
        self, text: str, high_complexity: bool = False,
        parent_sentence_count: int | None = None,
    ) -> bool:
        if parent_sentence_count is not None:
            return not (
                len(text) <= self.one_shot_max_chars
                and parent_sentence_count <= self.one_shot_max_sentences
            )
        return len(text) > self.one_shot_max_chars or (
            high_complexity and len(text) >= self.complexity_min_chars
        )

    @staticmethod
    def should_use_llm_planner(text: str, units: list[EvidenceUnit]) -> bool:
        """LLM planning is reserved for long abstracts without section structure."""
        parent_count = len({item.parent_sentence_id for item in units})
        sections = {item.section for item in units if item.section}
        unstructured = not sections or sections == {"ABSTRACT"}
        return unstructured and (len(text) > 4000 or parent_count > 20)

    def build(
        self,
        text: str,
        units: list[EvidenceUnit],
        *,
        high_complexity: bool = False,
    ) -> list[ArticleChunk]:
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
        parent_ids = tuple(item["parent_id"] for item in parents)
        if (
            not text or not units or len(parents) < 2
            or not self.should_chunk(text, high_complexity, len(parents))
        ):
            return [ArticleChunk(
                "c001", text, 0, len(text), self._sections(units),
                owner_sentence_ids=parent_ids,
                strategy="one_shot",
            )]

        chunks: list[ArticleChunk] = []
        next_parent = 0
        while next_parent < len(parents) and len(chunks) < self.max_chunks:
            owner_start_index = next_parent
            owner_end_index = next_parent
            owner_start = parents[owner_start_index]["start"]
            if owner_start_index == 0:
                owner_start = 0
            end_index = owner_end_index
            while end_index + 1 < len(parents):
                proposed_end = parents[end_index + 1]["end"]
                if proposed_end - owner_start > self.max_chars and end_index >= next_parent:
                    break
                end_index += 1
            if len(chunks) == self.max_chunks - 1:
                end_index = len(parents) - 1
            owner_end_index = end_index
            context_start_index = max(
                0, owner_start_index - self.overlap_parent_sentences
            )
            context_end_index = min(
                len(parents) - 1, owner_end_index + self.overlap_parent_sentences
            )
            start = 0 if context_start_index == 0 else parents[context_start_index]["start"]
            end = (
                len(text) if context_end_index == len(parents) - 1
                else parents[context_end_index]["end"]
            )
            selected = parents[context_start_index:context_end_index + 1]
            owner_ids = tuple(
                item["parent_id"] for item in parents[owner_start_index:owner_end_index + 1]
            )
            context_ids = tuple(
                item["parent_id"] for index, item in enumerate(parents)
                if context_start_index <= index <= context_end_index
                and not owner_start_index <= index <= owner_end_index
            )
            sections = tuple(dict.fromkeys(
                section for parent in selected for section in parent["sections"]
            ))
            chunks.append(ArticleChunk(
                chunk_id=f"c{len(chunks) + 1:03d}",
                text=text[start:end],
                char_start=start,
                char_end=end,
                sections=sections,
                overlapped_parent_sentences=len(context_ids),
                owner_sentence_ids=owner_ids,
                context_sentence_ids=context_ids,
                strategy="deterministic_sentence_windows",
            ))
            next_parent = owner_end_index + 1

        return chunks

    @staticmethod
    def _sections(units: list[EvidenceUnit]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(unit.section for unit in units)) or ("ABSTRACT",)
