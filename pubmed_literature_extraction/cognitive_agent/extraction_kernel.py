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
from typing import Optional
from dataclasses import dataclass, field
import time
import langextract as lx
from langextract.factory import ModelConfig
from cognitive_agent.schema.examples import KG_EXTRACTION_PROMPT, DEFAULT_EXAMPLES
from cognitive_agent.schema.entity_classes import EXTRACTION_CLASS_TO_LABEL

# 重试配置 (原生 Gemini 模式: schema 错误已消除, 仅应对网络瞬时故障)
MAX_RETRIES = 1
RETRY_BASE_DELAY = 2.0
RETRY_BACKOFF = 1.5


@dataclass
class RawExtraction:
    pmid: str = ""
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    raw_lx_result: Optional[object] = None
    error: str = ""
    retry_count: int = 0          # 实际重试次数
    warnings: list[str] = field(default_factory=list)  # 非致命警告

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "entities": self.entities,
            "relations": self.relations,
            "error": self.error,
            "retry_count": self.retry_count,
        }


class ExtractionKernel:
    """LangExtract + Gemini 提取内核 — v2 重试 + 容错版"""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def extract(
        self,
        text: str,
        document_id: str = "",
        examples: list | None = None,
        prompt: str = KG_EXTRACTION_PROMPT,
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

        result = RawExtraction(pmid=document_id)

        for attempt in range(1 + MAX_RETRIES):
            if attempt > 0:
                delay = RETRY_BASE_DELAY * (RETRY_BACKOFF ** (attempt - 1))
                print(f"    [Extract] Retry {attempt}/{MAX_RETRIES} for {document_id} "
                      f"(sleep {delay:.1f}s)...")
                time.sleep(delay)
                # 清空之前的实体，准备重新提取
                result.entities = []
                result.relations = []
                result.error = ""
                result.warnings = []

            try:
                doc = lx.data.Document(document_id=document_id, text=text)
                lx_result = lx.extract(
                    text_or_documents=[doc],
                    prompt_description=prompt,
                    examples=examples,
                    config=self.model_config,
                    temperature=0,
                    max_workers=2,                # 降低并发：proxy 模型更稳定
                    use_schema_constraints=False,  # 宽松模式
                    show_progress=False,
                    extraction_passes=1,           # 单次提取（稳定性优先）
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
                        self._parse(annotated, result)
                    except Exception as parse_err:
                        result.warnings.append(f"Parse error: {parse_err}")
                        # 不 break — 尝试解析剩余的 extractions

                # ── 重试判断：有实体 → 成功，无实体 → 可能重试 ──
                if len(result.entities) > 0:
                    result.retry_count = attempt
                    break  # 成功，跳出重试循环
                elif attempt < MAX_RETRIES:
                    result.warnings.append(
                        f"Attempt {attempt+1}: 0 entities extracted"
                    )
                else:
                    result.warnings.append(
                        f"All {MAX_RETRIES+1} attempts produced 0 entities"
                    )

            except Exception as e:
                result.error = str(e)
                result.warnings.append(f"Attempt {attempt+1} error: {e}")
                if attempt >= MAX_RETRIES:
                    print(f"    [Extract] All retries exhausted for {document_id}: {e}")

        return result

    def _parse(self, annotated, result: RawExtraction):
        """解析 LangExtract 返回的 AnnotatedDocument。

        容错: 单个 extraction 解析失败不影响其他 extraction。
        """
        for ext in annotated.extractions:
            try:
                has_g = ext.char_interval is not None and ext.char_interval.start_pos is not None
                # 类型映射
                raw_class = getattr(ext, 'extraction_class', 'unknown')
                entity_type = EXTRACTION_CLASS_TO_LABEL.get(raw_class, raw_class)

                # 属性安全转换
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
                        "grounded": entity.get("grounded", False),
                    })
                except Exception:
                    # 单个关系解析失败不影响其他
                    pass
