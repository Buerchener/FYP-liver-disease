#!/usr/bin/env python3
"""
cognitive_agent/extraction_kernel.py — LangExtract 提取内核 v2

改进:
- 重试机制: 0 实体时自动重试 (最多 3 次)
- 宽松 Schema: user_schema_constraints=False, 去掉 resolver_params
- 容错解析: 属性格式错误时仍保留实体
- 更好的错误分类: error vs warning
"""

from __future__ import annotations
import os
import re
from typing import Optional
from dataclasses import asdict, dataclass, field, is_dataclass
import copy
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import time
import langextract as lx
from langextract.factory import ModelConfig
from cognitive_agent.schema.examples import KG_EXTRACTION_PROMPT, DEFAULT_EXAMPLES
from cognitive_agent.schema.entity_classes import EXTRACTION_CLASS_TO_LABEL
from cognitive_agent.article_chunker import ArticleChunk
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.extraction_cache import LightweightExtractionCache
from cognitive_agent.golden_examples import GOLDEN_EXAMPLE_VERSION
from cognitive_agent.schema.ontology import ONTOLOGY_VERSION
from cognitive_agent.provider_errors import is_retryable_provider_error

# These phrases describe broad context or intervention classes, not KG entities.
# Keep this deterministic because model output is otherwise prone to over-labeling them.
GENERIC_ENTITY_TERMS = frozenset({
    "cancer", "tumor", "tumour", "immune regulation", "immunomodulation",
    "immune modulation", "immune suppression", "immunosuppression",
    "immune activation", "immune evasion", "immune cell infiltration",
    "antitumor immunity", "tumor immune microenvironment",
    "tumor microenvironment", "immune microenvironment", "immunotherapy",
    "immune checkpoint inhibitors", "cytokines", "tumor antigens",
})

# A tissue must be anatomical; microenvironments are contextual descriptions.
TISSUE_CONTEXT_TERMS = frozenset({
    "tumor immune microenvironment", "tumor microenvironment", "immune microenvironment",
})

# Empty responses get one retry. Transport/provider failures have a separate,
# configurable budget so a transient 429/502 does not empty an entire run.
EMPTY_RESULT_MAX_RETRIES = 1
EXTRACTION_CACHE_KEY_VERSION = "langextract-candidates-v3"
PROMPT_VERSION = "kg-extraction-prompt-v2"


def _langextract_version() -> str:
    try:
        return package_version("langextract")
    except PackageNotFoundError:
        return "unknown"


@dataclass
class RawExtraction:
    pmid: str = ""
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    raw_lx_result: Optional[object] = None
    error: str = ""
    retry_count: int = 0          # 实际重试次数
    warnings: list[str] = field(default_factory=list)  # 非致命警告
    chunk_count: int = 1
    chunks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "entities": self.entities,
            "relations": self.relations,
            "error": self.error,
            "retry_count": self.retry_count,
            "warnings": self.warnings,
            "chunk_count": self.chunk_count,
            "chunks": self.chunks,
        }


class ExtractionKernel:
    """LangExtract + Gemini 提取内核 — v2 重试 + 容错版"""

    def __init__(self, model_config: ModelConfig,
                 cache: LightweightExtractionCache | None = None,
                 inner_max_workers: int = 2):
        self.model_config = model_config
        self.cache = cache or LightweightExtractionCache(mode="off")
        self.inner_max_workers = max(1, int(inner_max_workers))

    @staticmethod
    def _stable_value(value):
        if is_dataclass(value):
            return ExtractionKernel._stable_value(asdict(value))
        if isinstance(value, dict):
            return {str(k): ExtractionKernel._stable_value(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [ExtractionKernel._stable_value(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _cache_key(self, *, text: str, examples: list, prompt: str,
                   retry_on_empty: bool) -> str:
        provider_kwargs = getattr(self.model_config, "provider_kwargs", {}) or {}
        http_options = provider_kwargs.get("http_options", {}) or {}
        endpoint = provider_kwargs.get("base_url") or http_options.get("base_url") or ""
        endpoint = str(endpoint).rstrip("/")
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        examples_value = self._stable_value(examples)
        examples_hash = hashlib.sha256(json.dumps(
            examples_value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        payload = {
            "version": EXTRACTION_CACHE_KEY_VERSION,
            "langextract_version": _langextract_version(),
            "provider": str(getattr(self.model_config, "provider", "") or ""),
            "model_id": str(getattr(self.model_config, "model_id", "") or ""),
            # Endpoint identity affects model behavior; credentials deliberately
            # do not participate in the key and are never cached.
            "endpoint_identity": endpoint,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "golden_shot_version": GOLDEN_EXAMPLE_VERSION,
            "golden_shot_hash": examples_hash,
            "ontology_version": ONTOLOGY_VERSION,
            "chunk_content_hash": text_hash,
            "temperature": 0, "schema_constraints": False,
            "fence_output": True,
            "extraction_passes": 1,
            "max_char_buffer": self._max_char_buffer(),
            "max_output_tokens": provider_kwargs.get("max_output_tokens"),
            "reasoning_effort": provider_kwargs.get("reasoning_effort"),
            "inner_max_workers": self.inner_max_workers,
            "retry_on_empty": bool(retry_on_empty),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _max_char_buffer() -> int:
        # Agent-level chunks are already bounded and span aligned. Avoid the
        # LangExtract 1,000-character default splitting each chunk a second time.
        return max(1000, int(os.environ.get("PRIMARY_LLM_MAX_CHAR_BUFFER", "4000")))

    @staticmethod
    def _payload(result: RawExtraction) -> dict:
        return {
            "entities": copy.deepcopy(result.entities),
            "relations": copy.deepcopy(result.relations),
            "error": result.error, "retry_count": result.retry_count,
            "warnings": list(result.warnings), "chunk_count": result.chunk_count,
            "chunks": copy.deepcopy(result.chunks),
        }

    @staticmethod
    def _from_payload(payload: dict, document_id: str, cache_status: str) -> RawExtraction:
        result = RawExtraction(
            pmid=document_id,
            entities=copy.deepcopy(payload.get("entities", [])),
            relations=copy.deepcopy(payload.get("relations", [])),
            error=str(payload.get("error", "") or ""),
            retry_count=int(payload.get("retry_count", 0) or 0),
            warnings=list(payload.get("warnings", []) or []),
            chunk_count=int(payload.get("chunk_count", 1) or 1),
            chunks=copy.deepcopy(payload.get("chunks", []) or []),
        )
        result.warnings.append(f"extraction_cache:{cache_status}")
        return result

    @staticmethod
    def _is_cacheable_payload(payload: dict) -> bool:
        """Accept replayable zero-result responses, but never cache failures.

        A valid empty extraction is a deterministic model result and must be
        reusable for a 100% warm replay.  Transport/provider errors, parse
        failures and malformed payloads remain deliberately non-cacheable.
        """
        if payload.get("error"):
            return False
        if not isinstance(payload.get("entities"), list):
            return False
        if not isinstance(payload.get("relations"), list):
            return False
        fatal_warning_tokens = (
            "parse error:", "unexpected lx.extract return type",
            "unexpected element in results:", "attempt 1 error:",
        )
        warnings = [str(item).casefold() for item in payload.get("warnings", []) or []]
        return not any(
            token in warning
            for warning in warnings
            for token in fatal_warning_tokens
        )

    def extract(
        self,
        text: str,
        document_id: str = "",
        examples: list | None = None,
        prompt: str = KG_EXTRACTION_PROMPT,
        retry_on_empty: bool = True,
    ) -> RawExtraction:
        """提取实体和关系，0 实体时自动重试。

        Args:
            text: 文章文本
            document_id: PMID
            examples: few-shot 示例
            prompt: 提取提示词

        Returns:
            RawExtraction 结果
        """
        if examples is None:
            examples = DEFAULT_EXAMPLES

        cache_key = self._cache_key(
            text=text, examples=examples, prompt=prompt, retry_on_empty=retry_on_empty,
        )
        def compute() -> tuple[dict, float]:
            started = time.perf_counter()
            result = self._extract_uncached(
                text=text, document_id=document_id, examples=examples,
                prompt=prompt, retry_on_empty=retry_on_empty,
            )
            return self._payload(result), time.perf_counter() - started
        payload, cache_status = self.cache.get_or_compute(
            cache_key, compute,
            cacheable=self._is_cacheable_payload,
        )
        return self._from_payload(payload, document_id, cache_status)

    def _extract_uncached(
        self, *, text: str, document_id: str, examples: list,
        prompt: str, retry_on_empty: bool,
    ) -> RawExtraction:
        """Execute one uncached LangExtract request sequence."""

        result = RawExtraction(pmid=document_id)

        empty_retry_budget = EMPTY_RESULT_MAX_RETRIES if retry_on_empty else 0
        provider_retry_budget = max(0, int(os.environ.get("PRIMARY_LLM_MAX_RETRIES", "5")))
        timeout_retry_budget = max(
            0, int(os.environ.get("PRIMARY_LLM_TIMEOUT_MAX_RETRIES", "1"))
        )
        retry_base_delay = max(0.0, float(os.environ.get("PRIMARY_LLM_RETRY_BASE_DELAY_S", "2")))
        retry_max_delay = max(
            retry_base_delay,
            float(os.environ.get("PRIMARY_LLM_RETRY_MAX_DELAY_S", "45")),
        )
        empty_retries = provider_retries = total_retries = 0
        while True:

            try:
                doc = lx.data.Document(document_id=document_id, text=text)
                lx_result = lx.extract(
                    text_or_documents=[doc],
                    prompt_description=prompt,
                    examples=examples,
                    config=self.model_config,
                    temperature=0,
                    max_workers=self.inner_max_workers,
                    use_schema_constraints=False,  # 宽松模式
                    # Compatible proxies may still wrap otherwise valid JSON
                    # in a single ```json fence.  LangExtract's lenient fence
                    # mode accepts both fenced and raw JSON.
                    fence_output=True,
                    show_progress=False,
                    extraction_passes=1,           # 单次提取（稳定性优先）
                    max_char_buffer=self._max_char_buffer(),
                    # 去掉 resolver_params — fuzzy alignment 会导致 chunk 被丢弃
                )

                # ── 归一化返回值 ──
                if isinstance(lx_result, list):
                    annotated_docs = lx_result
                elif hasattr(lx_result, 'extractions') and hasattr(lx_result, 'document_id'):
                    annotated_docs = [lx_result]
                elif hasattr(lx_result, 'extractions'):
                    annotated_docs = [lx_result]
                else:
                    result.error = f"Unexpected lx.extract return type: {type(lx_result).__name__}"
                    return result

                result.raw_lx_result = lx_result

                # ── 解析提取结果 ──
                for annotated in annotated_docs:
                    if not hasattr(annotated, 'extractions'):
                        result.warnings.append(
                            f"Unexpected element in results: {type(annotated).__name__}"
                        )
                        continue

                    try:
                        self._parse(annotated, result, source_text=text)
                    except Exception as parse_err:
                        result.warnings.append(f"Parse error: {parse_err}")
                        # 不 break — 尝试解析剩余的 extractions

                # ── 重试判断：有实体 → 成功，无实体 → 可能重试 ──
                if len(result.entities) > 0:
                    break  # 成功，跳出重试循环
                if empty_retries < empty_retry_budget:
                    empty_retries += 1
                    total_retries += 1
                    result.warnings.append(
                        f"Attempt {total_retries}: 0 entities extracted"
                    )
                else:
                    result.warnings.append(
                        f"All {empty_retry_budget + 1} attempts produced 0 entities"
                    )
                    break

            except Exception as e:
                result.error = str(e)
                retryable = is_retryable_provider_error(result.error)
                is_timeout = any(
                    token in result.error.casefold()
                    for token in ("timeout", "timed out")
                )
                effective_retry_budget = (
                    min(provider_retry_budget, timeout_retry_budget)
                    if is_timeout else provider_retry_budget
                )
                if not retryable or provider_retries >= effective_retry_budget:
                    result.warnings.append(f"Extraction error: {e}")
                    print(f"    [Extract] Retries exhausted for {document_id}: {e}")
                    break
                provider_retries += 1
                total_retries += 1
                result.warnings.append(f"Retryable provider error: {e}")

            delay = min(retry_max_delay, retry_base_delay * (2 ** max(0, total_retries - 1)))
            retry_after = re.search(r"retry[- ]after\s*[:=]\s*(\d+(?:\.\d+)?)", result.error, re.IGNORECASE)
            if retry_after:
                delay = min(retry_max_delay, max(0.0, float(retry_after.group(1))))
            print(
                f"    [Extract] Retry {total_retries} for {document_id} "
                f"(sleep {delay:.1f}s)..."
            )
            time.sleep(delay)
            # Clear partial state before the next independent request.
            result.entities = []
            result.relations = []
            result.error = ""
            result.warnings = []

        result.retry_count = total_retries

        return result

    def extract_chunked(
        self,
        chunks: list[ArticleChunk],
        *,
        full_text: str,
        document_id: str = "",
        examples: list | None = None,
        prompt: str = KG_EXTRACTION_PROMPT,
    ) -> RawExtraction:
        """Extract exact source chunks, rebase spans, then deduplicate overlap."""
        if len(chunks) <= 1:
            result = self.extract(
                full_text, document_id=document_id, examples=examples, prompt=prompt
            )
            result.chunk_count = 1
            result.chunks = [chunks[0].to_dict()] if chunks else []
            return result

        combined = RawExtraction(
            pmid=document_id,
            chunk_count=len(chunks),
            chunks=[chunk.to_dict() for chunk in chunks],
        )
        parent_units = ArticleEvidenceReader.parent_units(
            full_text, ArticleEvidenceReader().read(full_text),
        )
        raw_results = []
        for chunk in chunks:
            partial = self.extract(
                chunk.text,
                document_id=f"{document_id}:{chunk.chunk_id}",
                examples=examples,
                prompt=(
                    f"{prompt}\n\nCurrent extraction window: {chunk.chunk_id}; "
                    f"sections: {', '.join(chunk.sections)}; "
                    f"owner_sentence_ids: {', '.join(chunk.owner_sentence_ids)}; "
                    f"context_only_sentence_ids: {', '.join(chunk.context_sentence_ids)}. "
                    "Extract a relation only when its main assertion or resolving "
                    "reference occurs in an owner sentence. Context-only sentences "
                    "may resolve aliases/coreference but must not emit relations."
                ),
                retry_on_empty=False,
            )
            raw_results.append(partial.raw_lx_result)
            for entity in partial.entities:
                rebased = copy.deepcopy(entity)
                if isinstance(rebased.get("char_start"), int):
                    rebased["char_start"] += chunk.char_start
                if isinstance(rebased.get("char_end"), int):
                    rebased["char_end"] += chunk.char_start
                combined.entities.append(rebased)
            for raw_relation in partial.relations:
                relation = copy.deepcopy(raw_relation)
                evidence = str(relation.get("evidence", "") or "").strip()
                local_start = chunk.text.find(evidence) if evidence else -1
                evidence_parent_ids: list[str] = []
                if local_start >= 0:
                    absolute_start = chunk.char_start + local_start
                    absolute_end = absolute_start + len(evidence)
                    evidence_parent_ids = [
                        item.parent_sentence_id for item in parent_units
                        if item.char_start < absolute_end and item.char_end > absolute_start
                    ]
                if (
                    evidence_parent_ids
                    and chunk.owner_sentence_ids
                    and not set(evidence_parent_ids) & set(chunk.owner_sentence_ids)
                ):
                    combined.warnings.append(
                        f"{chunk.chunk_id}: context_only_relation_suppressed"
                    )
                    continue
                relation.update({
                    "candidate_lane": "extracted_hint",
                    "owner_sentence_ids": list(chunk.owner_sentence_ids),
                    "context_sentence_ids": list(chunk.context_sentence_ids),
                    "evidence_parent_sentence_ids": evidence_parent_ids,
                    "source_chunk_id": chunk.chunk_id,
                })
                combined.relations.append(relation)
            combined.warnings.extend(
                f"{chunk.chunk_id}: {warning}" for warning in partial.warnings
            )
            if partial.error:
                combined.warnings.append(f"{chunk.chunk_id}: {partial.error}")
            combined.retry_count += partial.retry_count

        combined.raw_lx_result = raw_results
        combined.entities = self._merge_overlap_entities(combined.entities)
        combined.relations = self._dedupe_relations(combined.relations)
        if not combined.entities:
            fallback = self.extract(
                full_text, document_id=document_id, examples=examples, prompt=prompt
            )
            fallback.chunk_count = len(chunks)
            fallback.chunks = [chunk.to_dict() for chunk in chunks]
            fallback.warnings.insert(0, "chunked extraction empty; used one-shot fallback")
            return fallback
        return combined

    @staticmethod
    def _merge_overlap_entities(entities: list[dict]) -> list[dict]:
        merged: dict[tuple, dict] = {}
        for entity in entities:
            key = (
                str(entity.get("mention", "")).casefold(), entity.get("type", ""),
                entity.get("char_start"), entity.get("char_end"),
            )
            if key not in merged:
                merged[key] = copy.deepcopy(entity)
                continue
            target = merged[key].setdefault("attributes", {})
            for name, value in (entity.get("attributes", {}) or {}).items():
                if name not in target:
                    target[name] = copy.deepcopy(value)
                elif isinstance(target[name], list) and isinstance(value, list):
                    seen = {json.dumps(item, sort_keys=True, default=str) for item in target[name]}
                    target[name].extend(
                        copy.deepcopy(item) for item in value
                        if json.dumps(item, sort_keys=True, default=str) not in seen
                    )
        return list(merged.values())

    @staticmethod
    def _dedupe_relations(relations: list[dict]) -> list[dict]:
        output: list[dict] = []
        seen: dict[tuple, dict] = {}
        for relation in relations:
            key = tuple(str(relation.get(name, "") or "").casefold() for name in (
                "subject", "subject_type", "predicate", "object", "object_type",
                "direction",
            ))
            if key not in seen:
                value = copy.deepcopy(relation)
                evidence = str(value.get("evidence", "") or "").strip()
                value["evidence_candidates"] = [evidence] if evidence else []
                value["source_chunk_ids"] = [
                    value.get("source_chunk_id")
                ] if value.get("source_chunk_id") else []
                seen[key] = value
                output.append(value)
                continue
            target = seen[key]
            evidence_values = [
                *target.get("evidence_candidates", []),
                str(relation.get("evidence", "") or "").strip(),
            ]
            target["evidence_candidates"] = list(dict.fromkeys(
                item for item in evidence_values if item
            ))[:3]
            target["source_chunk_ids"] = list(dict.fromkeys([
                *target.get("source_chunk_ids", []),
                str(relation.get("source_chunk_id", "") or ""),
            ]))
        return output

    def _parse(self, annotated, result: RawExtraction, source_text: str = ""):
        """解析 LangExtract 返回的 AnnotatedDocument。

        容错: 单个 extraction 解析失败不影响其他 extraction。
        """
        for ext in annotated.extractions:
            try:
                has_interval = (
                    ext.char_interval is not None
                    and ext.char_interval.start_pos is not None
                    and ext.char_interval.end_pos is not None
                )
                # 类型映射
                raw_class = getattr(ext, 'extraction_class', 'unknown')
                entity_type = EXTRACTION_CLASS_TO_LABEL.get(raw_class, raw_class)

                # Preserve every raw candidate.  The Phase-A quality gate records
                # rejected entities and whether relations referenced them; dropping
                # candidates here would destroy that audit trail.
                mention = str(getattr(ext, 'extraction_text', '') or '').strip()
                aligned_source = ""
                if has_interval:
                    aligned_source = source_text[
                        ext.char_interval.start_pos:ext.char_interval.end_pos
                    ]
                # LangExtract MATCH_LESSER may expand a source token (e.g.
                # "TNF") into a broader unsupported concept ("TNF signalling").
                # Preserve it for audit but never call it grounded.
                has_g = bool(
                    has_interval
                    and " ".join(mention.casefold().split())
                    == " ".join(aligned_source.casefold().split())
                )
                if mention.casefold() in GENERIC_ENTITY_TERMS:
                    result.warnings.append(f"Generic entity candidate retained for review: {mention}")
                if entity_type == "Tissue" and mention.casefold() in TISSUE_CONTEXT_TERMS:
                    result.warnings.append(f"Contextual tissue candidate retained for review: {mention}")

                try:
                    attrs = dict(ext.attributes) if ext.attributes else {}
                except Exception:
                    attrs = {}

                # 过滤掉值为 "null" 字符串的属性 (常见于模型输出错误)
                clean_attrs = {}
                for k, v in attrs.items():
                    if v is None or v == "null" or v == []:
                        continue
                    # 过滤列表中的 "null" 字符串
                    if isinstance(v, list):
                        clean_list = [
                            x for x in v
                            if x is not None and x != "null" and x != {}
                        ]
                        if clean_list:
                            clean_attrs[k] = clean_list
                    elif isinstance(v, dict):
                        if v:  # 非空 dict
                            clean_attrs[k] = v
                    else:
                        clean_attrs[k] = v

                entity = {
                    "mention": ext.extraction_text,
                    "type": entity_type,
                    "extraction_class": raw_class,
                    "attributes": clean_attrs,
                    "grounded": has_g,
                    "source_span": aligned_source if has_interval else "",
                    "alignment_status": (
                        ext.alignment_status.name
                        if hasattr(ext, "alignment_status") and ext.alignment_status
                        else "UNKNOWN"
                    ),
                    "char_start": (
                        ext.char_interval.start_pos
                        if has_g and ext.char_interval
                        else None
                    ),
                    "char_end": (
                        ext.char_interval.end_pos
                        if has_g and ext.char_interval
                        else None
                    ),
                }
                result.entities.append(entity)

                # 提取关系
                self._extract_relations(ext, entity, result)

            except Exception as parse_err:
                result.warnings.append(f"Entity parse warning: {parse_err}")

    def _extract_relations(self, extraction, entity: dict, result: RawExtraction):
        """从实体属性中提取关系。

        容错: 跳过非 dict 的关系条目，不中断整体解析。
        """
        attrs = entity.get("attributes", {})
        rel_map = {
            "associated_with": "ASSOCIATED_WITH",
            "encodes": "ENCODES",
            "participates_in": "PARTICIPATES_IN",
            "interacts_with": "INTERACTS_WITH",
            "expressed_in": "EXPRESSED_IN",
            "prognostic_in": "PROGNOSTIC_IN",
            "progresses_to": "PROGRESSES_TO",
            "associated_with_metabolite": "ASSOCIATED_WITH_METABOLITE",
        }

        for attr_key, pred in rel_map.items():
            rel_list = attrs.get(attr_key, [])
            if not rel_list:
                continue
            if not isinstance(rel_list, list):
                rel_list = [rel_list]

            for rd in rel_list:
                if not isinstance(rd, dict):
                    # 跳过字符串 "null"、数字等非 dict 条目
                    continue
                try:
                    target_type = rd.get("target_type", "")
                    result.relations.append({
                        "subject": entity["mention"],
                        "subject_type": entity["type"],
                        "predicate": pred,
                        "object": rd.get("target_entity", ""),
                        "object_type": (
                            EXTRACTION_CLASS_TO_LABEL.get(target_type, target_type)
                            if target_type else ""
                        ),
                        "direction": rd.get("direction", "unknown"),
                        "negated": rd.get("negated", False),
                        "uncertain": rd.get("uncertain", False),
                        "disease_stage": rd.get("disease_stage", ""),
                        "evidence": rd.get("evidence", ""),
                        "species": rd.get("species", attrs.get("species", "")),
                        "confidence": rd.get("confidence", attrs.get("confidence", 0.7)),
                        "grounded": entity.get("grounded", False),
                    })
                except Exception:
                    # 单个关系解析失败不影响其他
                    pass
