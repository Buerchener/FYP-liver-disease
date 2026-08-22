#!/usr/bin/env python3
"""Inference-time demonstration retrieval for the pairwise LLM judge.

Development gold annotations are used for few-shot prompting, never for
fine-tuning.  For each candidate pair the retriever selects a small set of
demonstrations:

    * one positive example (same predicate, same endpoint-type pair);
    * one confusable example (same type pair, different predicate);
    * one hard NO_RELATION example — in `error_pattern` mode this slot is
      filled with a negative that carries the SAME deterministic error
      signature as the candidate's own focus sentence (cohort sampling,
      background phrasing, computational prediction, ...).

Error-pattern retrieval replaces text-similarity retrieval: the dominant
judge failure modes are categorical (cohort/background co-occurrence dressed
as ASSOCIATED_WITH), and a matched same-class hard negative teaches the
annotation policy far better than ngram-similar text.  Character-ngram
similarity remains only as a tie-breaker inside a class.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognitive_agent.evidence_units import ArticleEvidenceReader, EvidenceUnit
from cognitive_agent.extraction_quality import locate_contiguous, normalize_surface
from cognitive_agent.schema.relation_signatures import RELATION_SIGNATURES

# ── Deterministic error signatures ──
# Each category captures a measured annotation-policy failure mode of the
# first judge run.  Signatures are regex/section rules over the candidate's
# focus sentence; they never call a model.
ERROR_CATEGORIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "cohort_context_false_association": (
        "共现只出现在队列/样本描述中（'in patients with Y'），不代表关系结论",
        (
            r"\bmeasured in patients with\b", r"\bin patients with\b",
            r"\bamong patients with\b", r"\bcohort\b",
            r"\b(?:serum|plasma|tissue) samples? (?:of|from)\b",
            r"\blevels? of\b.{0,60}\bin\b.{0,60}\bpatients\b",
        ),
    ),
    "measurement_without_claim": (
        "句子只报告测量值/表达量，没有关系性断言",
        (
            r"\b(?:levels?|expression|concentration|activity) of\b"
            r".{0,80}\b(?:measured|assessed|detected|examined|evaluated|compared)\b",
        ),
    ),
    "background_relation": (
        "引言/背景句中的领域知识，不是本文新发现",
        (
            r"\bknown to\b", r"\bhas been shown\b", r"\bit has been reported\b",
            r"\bprevious(?:ly)? (?:studies|reports|work)\b",
            r"\bwell[- ]established\b", r"\bplays? (?:an? )?(?:important|critical|key) role\b",
            r"\bcommonly\b.{0,40}\b(?:associat|linked|involved)\b",
        ),
    ),
    "prediction_not_observation": (
        "计算预测/docking/数据库筛选，不是实验观察",
        (
            r"\bdocking\b", r"\bin silico\b", r"\bcomputational(?:ly)?\b",
            r"\bpredicted (?:to|binding)\b", r"\benrichment\b",
            r"\b(?:public )?databases?\b", r"\bbioinformatic",
            r"\bhigh[- ]throughput screen\b",
        ),
    ),
    "association_vs_causality": (
        "因果动词不是 ASSOCIATED_WITH（相关 ≠ 因果）",
        (
            r"\b(?:causes?|caused|leads? to|led to|drives?|induce[sd]?|trigger[sd]?|"
            r"results? in|resulted in)\b",
        ),
    ),
    "direction_confusion": (
        "被动/反向措辞容易把方向写反",
        (
            r"\b(?:regulated|activated|inhibited|induced|modulated|controlled|"
            r"downregulated|upregulated) by\b",
            r"\b(?:substrate|target|ligand|receptor) (?:for|of)\b",
            r"\b(?:downstream|upstream) (?:of|targets?)\b",
        ),
    ),
    "method_only_relation": (
        "方法/材料句中的提及（试剂、细胞系、实验手段）不是研究发现",
        (
            r"\b(?:antibod\w+|western blot|rt-?pcr|immunohistochem|reagent|primer\b)",
            r"\b(?:sirna|shrna|transfect|plasmid|knockout mice|cell lines?)\b",
            r"\b(?:was|were) used to\b", r"\busing\b.{0,40}\b(?:assay|staining)\b",
        ),
    ),
    "mere_cooccurrence": (
        "纯共现：同一句出现但没有任何关系性动词",
        (),
    ),
}

RELATIONAL_CUE_RE = re.compile(
    r"\b(?:associat|correlat|linked|related|bind|interact|express|progress|"
    r"encod|participat|mediat|predict|prognostic|caus|leads? to|regulat|"
    r"inhibit|activates?|induc|promot|suppress|reduc|increas|attenuat)\w*\b",
    re.IGNORECASE,
)

BACKGROUND_SECTION_NAMES = frozenset({"INTRODUCTION", "BACKGROUND", "INTRO"})


def detect_error_signatures(
    sentence: str, section: str = "", study_type: str = "",
) -> list[str]:
    """Deterministic error-signature detection over one sentence.

    Ordered by specificity; `mere_cooccurrence` only fires when no other
    signature matched and the sentence contains no relational verb at all.
    """
    hits: list[str] = []
    lowered = (sentence or "").casefold()
    for category, (_, patterns) in ERROR_CATEGORIES.items():
        if category == "mere_cooccurrence":
            continue
        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in patterns):
            hits.append(category)
    if not lowered:
        return hits
    if (
        "background_relation" not in hits
        and section.upper() in BACKGROUND_SECTION_NAMES
        and RELATIONAL_CUE_RE.search(sentence)
    ):
        hits.append("background_relation")
    if not hits and not RELATIONAL_CUE_RE.search(sentence):
        hits.append("mere_cooccurrence")
    return hits


@dataclass
class FewShotExample:
    pmid: str
    sentence: str
    subject: str
    subject_type: str
    predicate: str  # "NO_RELATION" for hard negatives
    object: str
    object_type: str
    evidence_quote: str = ""
    import_ready: bool = False
    exclusion_reason: str = ""
    study_type: str = ""
    error_category: str = ""          # hard negatives: which policy error this teaches
    sentence_error_signatures: str = ""  # positives: traps present in its own sentence
    _ngrams: frozenset | None = field(default=None, repr=False)

    def render(self) -> str:
        if self.predicate == "NO_RELATION":
            if self.error_category:
                description = ERROR_CATEGORIES.get(
                    self.error_category, ("", ())
                )[0]
                verdict = f"NO_RELATION（错误模式 {self.error_category}: {description}）"
            else:
                verdict = "NO_RELATION（仅共现/背景/方法表述，无文章支持的关联）"
            tail = ""
        else:
            verdict = f"{self.subject} -[{self.predicate}]-> {self.object}"
            tail = f" | 证据: \"{self.evidence_quote}\""
            if self.import_ready:
                tail += " | import_ready=true"
            elif self.exclusion_reason:
                tail += f" | import_ready=false({self.exclusion_reason})"
        return f"EX 句: \"{self.sentence}\" | 判定: {verdict}{tail}"


def char_ngrams(text: str, size: int = 4) -> set[str]:
    compact = "".join(text.casefold().split())
    if len(compact) <= size:
        return {compact}
    return {compact[index:index + size] for index in range(len(compact) - size + 1)}


def ngram_similarity(first: str, second: str, size: int = 4) -> float:
    left, right = char_ngrams(first, size), char_ngrams(second, size)
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def ngram_set_similarity(query_ngrams: set[str], stored_ngrams: set[str]) -> float:
    if not query_ngrams or not stored_ngrams:
        return 0.0
    return len(query_ngrams & stored_ngrams) / math.sqrt(
        len(query_ngrams) * len(stored_ngrams)
    )


class FewShotRetriever:
    """Compact demonstration index over development gold annotations."""

    def __init__(
        self,
        pool_path: str = "",
        *,
        source_path: str = "",
        max_examples: int = 4,
        exclude_pmids: set[str] | None = None,
    ):
        self.pool_path = str(pool_path or "")
        self.source_path = str(source_path or "")
        self.max_examples = max(0, int(max_examples))
        self.exclude_pmids = set(exclude_pmids or set())
        self.examples: list[FewShotExample] = []
        self.by_type_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.by_predicate: dict[str, list[int]] = defaultdict(list)
        self.by_error_category: dict[str, list[int]] = defaultdict(list)
        self._loaded = False

    def _source_rows(self) -> dict[str, dict]:
        if not self.source_path:
            return {}
        path = Path(self.source_path)
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as handle:
            return {
                str(item.get("pmid", "")): item
                for item in (json.loads(line) for line in handle if line.strip())
                if item.get("pmid")
            }

    def load(self) -> None:
        self.examples = []
        self.by_type_pair = defaultdict(list)
        self.by_predicate = defaultdict(list)
        if self._loaded or not self.pool_path:
            self._loaded = True
            return
        path = Path(self.pool_path)
        if not path.exists():
            return
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        source_by_pmid = self._source_rows()
        reader = ArticleEvidenceReader()
        for row in rows:
            pmid = str(row.get("pmid", ""))
            if not pmid or pmid in self.exclude_pmids:
                continue
            source = source_by_pmid.get(pmid, {})
            title = str(row.get("title", "") or source.get("title", "") or "")
            abstract = str(row.get("abstract", "") or source.get("abstract", "") or "")
            source_text = str(row.get("text", "") or "")
            if not source_text and title and abstract:
                source_text = f"TITLE: {title}\nABSTRACT: {abstract}"
            if not source_text:
                continue
            units = reader.read(source_text)
            sentences = ArticleEvidenceReader.parent_units(source_text, units)
            gold_entities = row.get("entities", []) or []
            gold_relations = row.get("relations", []) or []
            study_type = str(row.get("study_type", "") or row.get("study_context", "") or "")

            # ── positive examples: one per gold relation with its evidence ──
            for relation in gold_relations:
                subject = str(relation.get("subject", "") or "")
                object_ = str(relation.get("object", "") or "")
                subject_type = str(relation.get("subject_type", "") or "")
                object_type = str(relation.get("object_type", "") or "")
                predicate = str(relation.get("predicate", "") or "").upper()
                evidence = str(relation.get("evidence", "") or "")
                if not all((subject, object_, predicate)):
                    continue
                grounded, start, end = locate_contiguous(evidence, source_text)
                if not grounded:
                    continue
                sentence = self._covering_sentence(
                    start, end, sentences, source_text
                )
                sentence_section = self._section_at(start, end, sentences)
                signatures = detect_error_signatures(sentence, sentence_section)
                self._append(FewShotExample(
                    pmid=pmid, sentence=sentence,
                    subject=subject, subject_type=subject_type,
                    predicate=predicate, object=object_, object_type=object_type,
                    evidence_quote=evidence,
                    import_ready=bool(relation.get("import_ready")),
                    exclusion_reason=str(relation.get("exclusion_reason", "") or ""),
                    study_type=study_type,
                    sentence_error_signatures=",".join(signatures),
                ))

            # ── hard NO_RELATION examples: co-occurring gold entity pairs that
            #    the article never asserts a relation for.  Negative notes make
            #    deliberately zero-relation articles first-class negatives. ──
            gold_keys = {
                (
                    normalize_surface(item.get("subject", "")),
                    normalize_surface(item.get("object", "")),
                )
                for item in gold_relations
            }
            entity_forms: list[tuple[dict, set[str]]] = []
            for entity in gold_entities:
                mention = str(entity.get("mention", "") or "")
                canonical = str(entity.get("canonical", "") or "")
                aliases = {
                    normalize_surface(value)
                    for value in (mention, canonical, *(entity.get("aliases", []) or []))
                    if str(value or "").strip()
                }
                if aliases:
                    entity_forms.append((entity, aliases))
            for unit in units:
                for left_index, (left, left_aliases) in enumerate(entity_forms):
                    if not any(alias and alias in normalize_surface(unit.text) for alias in left_aliases):
                        continue
                    for right, right_aliases in entity_forms[left_index + 1:]:
                        if not str(left.get("type", "") or "") or not str(right.get("type", "") or ""):
                            continue
                        if not any(alias and alias in normalize_surface(unit.text) for alias in right_aliases):
                            continue
                        pair = (
                            normalize_surface(left.get("canonical", left.get("mention", ""))),
                            normalize_surface(right.get("canonical", right.get("mention", ""))),
                        )
                        if pair in gold_keys or pair[::-1] in gold_keys:
                            continue
                        allowed = [
                            predicate for predicate, pairs in RELATION_SIGNATURES.items()
                            if (str(left.get("type", "")), str(right.get("type", ""))) in pairs
                            or (str(right.get("type", "")), str(left.get("type", ""))) in pairs
                        ]
                        if not allowed:
                            continue
                        # Tag the negative with the deterministic error
                        # signature of its own clause so retrieval can match
                        # same-class negatives to same-class candidates.
                        signature = detect_error_signatures(unit.text, unit.section)
                        self._append(FewShotExample(
                            pmid=pmid, sentence=unit.text,
                            subject=str(left.get("canonical", left.get("mention", "")) or ""),
                            subject_type=str(left.get("type", "") or ""),
                            predicate="NO_RELATION",
                            object=str(right.get("canonical", right.get("mention", "")) or ""),
                            object_type=str(right.get("type", "") or ""),
                            study_type=study_type,
                            error_category=signature[0] if signature else "",
                        ))
        self._loaded = True

    def _append(self, example: FewShotExample) -> None:
        index = len(self.examples)
        example._ngrams = frozenset(char_ngrams(example.sentence))
        self.examples.append(example)
        pair = (example.subject_type, example.object_type)
        self.by_type_pair[pair].append(index)
        self.by_predicate[example.predicate].append(index)
        if example.error_category:
            self.by_error_category[example.error_category].append(index)

    @staticmethod
    def _covering_sentence(
        start: int, end: int, sentences: list[EvidenceUnit], source_text: str,
    ) -> str:
        for sentence in sentences:
            if sentence.char_start <= start and end <= sentence.char_end:
                return sentence.text
        if start >= 0:
            return source_text[start:end]
        return ""

    @staticmethod
    def _section_at(
        start: int, end: int, sentences: list[EvidenceUnit],
    ) -> str:
        for sentence in sentences:
            if sentence.char_start <= start and end <= sentence.char_end:
                return sentence.section
        return ""

    def retrieve(
        self,
        *,
        subject: str,
        subject_type: str,
        object_: str,
        object_type: str,
        allowed_predicates: list[str],
        source_predicates: list[str],
        focus_sentence: str = "",
        study_type: str = "",
        exclude_pmid: str = "",
        mode: str = "retrieval",  # retrieval | error_pattern
        section: str = "",
    ) -> list[FewShotExample]:
        if not self._loaded:
            self.load()
        if not self.examples or self.max_examples <= 0:
            return []
        excluded = self.exclude_pmids | ({exclude_pmid} if exclude_pmid else set())
        type_pair = (subject_type, object_type)
        reverse_pair = (object_type, subject_type)
        candidate_predicates = set(source_predicates or allowed_predicates or [])
        detected = (
            set(detect_error_signatures(focus_sentence, section))
            if mode == "error_pattern"
            else set()
        )

        # Cheap structural pre-filter: the pool is scanned per candidate pair,
        # so similarity is only computed on examples that can occupy a slot.
        pool = [
            example for example in self.examples
            if example.pmid not in excluded
            and (
                (example.subject_type, example.object_type) in {type_pair, reverse_pair}
                or example.predicate in candidate_predicates
                or example.predicate == "NO_RELATION"
            )
        ]
        if not pool:
            return []
        focus_ngrams = frozenset(char_ngrams(focus_sentence)) if focus_sentence else None

        def score(example: FewShotExample) -> float:
            value = 0.0
            example_pair = (example.subject_type, example.object_type)
            if example_pair == type_pair:
                value += 2.0
            elif example_pair == reverse_pair:
                value += 1.0
            if example.predicate in candidate_predicates:
                value += 1.5
            elif example.predicate == "NO_RELATION":
                value += 0.8
            if study_type and example.study_type == study_type:
                value += 0.4
            if mode == "error_pattern":
                # Same-class hard negatives dominate their slot: the annotation
                # policy lesson beats surface similarity.
                if example.error_category and example.error_category in detected:
                    value += 3.0
                # Prefer positives that are clean direct findings.
                if example.predicate != "NO_RELATION" and not example.sentence_error_signatures:
                    value += 0.5
                if focus_ngrams is not None and example._ngrams:
                    value += 0.3 * ngram_set_similarity(focus_ngrams, example._ngrams)
            elif focus_ngrams is not None and example._ngrams:
                value += 0.6 * ngram_set_similarity(focus_ngrams, example._ngrams)
            return value

        ranked = sorted(pool, key=score, reverse=True)
        selected: list[FewShotExample] = []
        slots = {
            "positive": None,
            "confusable": None,
            "hard_negative": None,
        }
        positive_predicates = set(source_predicates or allowed_predicates or [])
        for example in ranked:
            if len(selected) >= self.max_examples:
                break
            if example.predicate in positive_predicates and slots["positive"] is None:
                slots["positive"] = example
            elif (
                example.predicate != "NO_RELATION"
                and example.predicate not in positive_predicates
                and (example.subject_type, example.object_type) in {type_pair, reverse_pair}
                and slots["confusable"] is None
            ):
                slots["confusable"] = example
            elif example.predicate == "NO_RELATION" and slots["hard_negative"] is None:
                slots["hard_negative"] = example
            if all(slots.values()):
                break
        for key in ("positive", "hard_negative", "confusable"):
            example = slots[key]
            if example is not None:
                selected.append(example)
        return selected[: self.max_examples]

    def audit(self) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        return {
            "pool_path": self.pool_path,
            "loaded": self._loaded,
            "example_count": len(self.examples),
            "positive_count": sum(
                example.predicate != "NO_RELATION" for example in self.examples
            ),
            "hard_negative_count": sum(
                example.predicate == "NO_RELATION" for example in self.examples
            ),
            "exclude_pmid_count": len(self.exclude_pmids),
        }
