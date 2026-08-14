#!/usr/bin/env python3
"""Extractive, span-preserving evidence units for PubMed abstracts.

The reader is deliberately deterministic.  It identifies structured abstract
sections, sentences, and (for long sentences) clause-sized spans without
rewriting the source.  LLM components may select a unit, but they never invent
or paraphrase evidence text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


SECTION_RE = re.compile(
    r"(?:(?<=\n)|(?<=\s))"
    r"(BACKGROUND|INTRODUCTION|OBJECTIVE|OBJECTIVES|AIM|AIMS|PURPOSE|"
    r"METHOD|METHODS|MATERIALS AND METHODS|RESULT|RESULTS|"
    r"CONCLUSION|CONCLUSIONS|DISCUSSION)\s*:\s*",
    re.IGNORECASE,
)
SENTENCE_RE = re.compile(r"[^\n]+?(?:[.!?](?=\s|$)|$)", re.DOTALL)
CLAUSE_BOUNDARY_RE = re.compile(
    r"\s*(?:;|(?=\b(?:whereas|while|however|but)\b))\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    section: str
    text: str
    char_start: int
    char_end: int
    parent_sentence_id: str

    def to_dict(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "section": self.section,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "parent_sentence_id": self.parent_sentence_id,
        }


class ArticleEvidenceReader:
    """Create exact source spans suitable for extraction and adjudication."""

    def read(self, text: str) -> list[EvidenceUnit]:
        units: list[EvidenceUnit] = []
        abstract_match = re.search(r"\nABSTRACT:\s*", text, re.IGNORECASE)
        abstract_start = abstract_match.end() if abstract_match else 0
        sections = self._sections(text, abstract_start)
        sentence_no = 0
        unit_no = 0
        for section, start, end in sections:
            segment = text[start:end]
            for sentence_match in SENTENCE_RE.finditer(segment):
                raw = sentence_match.group(0)
                left = len(raw) - len(raw.lstrip())
                right = len(raw.rstrip())
                if right <= left:
                    continue
                sentence = raw[left:right]
                sentence_start = start + sentence_match.start() + left
                sentence_end = start + sentence_match.start() + right
                sentence_no += 1
                parent_id = f"s{sentence_no:03d}"
                clause_spans = self._clause_spans(sentence)
                for local_start, local_end in clause_spans:
                    clause = sentence[local_start:local_end]
                    trim_left = len(clause) - len(clause.lstrip(" ;,"))
                    trim_right = len(clause.rstrip())
                    if trim_right <= trim_left:
                        continue
                    clause = clause[trim_left:trim_right]
                    char_start = sentence_start + local_start + trim_left
                    char_end = sentence_start + local_start + trim_right
                    if not re.search(r"\w", clause, re.UNICODE):
                        continue
                    unit_no += 1
                    units.append(EvidenceUnit(
                        unit_id=f"u{unit_no:03d}",
                        section=section,
                        text=clause,
                        char_start=char_start,
                        char_end=char_end,
                        parent_sentence_id=parent_id,
                    ))
                # A defensive fallback for pathological splitting.
                if not clause_spans and sentence_start < sentence_end:
                    unit_no += 1
                    units.append(EvidenceUnit(
                        unit_id=f"u{unit_no:03d}", section=section, text=sentence,
                        char_start=sentence_start, char_end=sentence_end,
                        parent_sentence_id=parent_id,
                    ))
        return units

    @staticmethod
    def _sections(text: str, abstract_start: int) -> list[tuple[str, int, int]]:
        matches = list(SECTION_RE.finditer(text, abstract_start))
        if not matches:
            return [("ABSTRACT", abstract_start, len(text))]
        sections: list[tuple[str, int, int]] = []
        if matches[0].start() > abstract_start:
            sections.append(("ABSTRACT", abstract_start, matches[0].start()))
        for index, match in enumerate(matches):
            section = match.group(1).upper().replace(" ", "_")
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            sections.append((section, match.end(), end))
        return sections

    @staticmethod
    def _clause_spans(sentence: str) -> list[tuple[int, int]]:
        # Short sentences remain intact.  Long biomedical sentences are split
        # only at explicit boundaries so every unit remains a source substring.
        # Structured abstracts often compress two separately assertable
        # findings into only 20–30 words.  Make those clause-local early enough
        # for evidence adjudication while leaving short/simple prose intact.
        if len(re.findall(r"\b\w+[\w-]*\b", sentence)) < 16:
            return [(0, len(sentence))]
        spans: list[tuple[int, int]] = []
        cursor = 0
        for boundary in CLAUSE_BOUNDARY_RE.finditer(sentence):
            if boundary.start() > cursor:
                spans.append((cursor, boundary.start()))
            cursor = boundary.end()
        if cursor < len(sentence):
            spans.append((cursor, len(sentence)))
        return spans or [(0, len(sentence))]

    @staticmethod
    def containing_unit(evidence: str, units: list[EvidenceUnit]) -> EvidenceUnit | None:
        evidence = str(evidence or "").strip()
        if not evidence:
            return None
        exact = [unit for unit in units if evidence in unit.text]
        if exact:
            return min(exact, key=lambda item: len(item.text))
        reverse = [unit for unit in units if unit.text in evidence]
        return max(reverse, key=lambda item: len(item.text), default=None)
