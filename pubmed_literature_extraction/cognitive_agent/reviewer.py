#!/usr/bin/env python3
"""独立的 LLM 抽取质量审稿器。

Reviewer 只返回质量意见，不修改抽取结果、不访问 Neo4j，也不执行任何写入。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


REVIEW_DECISIONS = frozenset({"ACCEPT", "REVIEW", "REJECT"})
MAX_TEXT_CHARS = 12000
MAX_EVIDENCE_CHARS = 1000
MAX_LIST_ITEMS = 50
MAX_RATIONALE_CHARS = 2000


@dataclass(frozen=True)
class ReviewerConfig:
    """审稿模型配置；api_key 不会写入审稿结果。"""

    api_key: str = ""
    api_base: str = ""
    model_id: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 2048


@dataclass
class ReviewResult:
    """审稿结果，字段保持 JSON-safe。"""

    status: str = "DISABLED"  # DISABLED | OK | FALLBACK | ERROR
    overall_score: float = 0.0
    decision: str = "REVIEW"
    entity_reviews: list[dict] = field(default_factory=list)
    relation_reviews: list[dict] = field(default_factory=list)
    missing_items: list[str] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    rationale: str = ""
    error: str = ""
    model_id: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "overall_score": self.overall_score,
            "decision": self.decision,
            "entity_reviews": self.entity_reviews,
            "relation_reviews": self.relation_reviews,
            "missing_items": self.missing_items,
            "hallucination_flags": self.hallucination_flags,
            "rationale": self.rationale,
            "error": self.error,
            "model_id": self.model_id,
        }


class ExtractionReviewer:
    """使用独立 LLM 审查抽取结果；模型不可用时安全降级。"""

    def __init__(
        self,
        config: ReviewerConfig | None = None,
        generate: Optional[Callable[[str], Any]] = None,
    ):
        self.config = config or ReviewerConfig()
        self._generate = generate

    @property
    def enabled(self) -> bool:
        return bool(self._generate or (self.config.api_key and self.config.model_id))

    def review(
        self,
        text: str,
        extraction: dict,
        verification: dict | None = None,
        pmid: str = "",
    ) -> ReviewResult:
        if not self.enabled:
            return ReviewResult(model_id=self.config.model_id)

        prompt = self._build_prompt(text, extraction, verification or {}, pmid)
        try:
            raw = self._call_model(prompt)
            payload = self._parse_json(raw)
            result = self._normalize(payload)
            result.status = "OK"
            result.model_id = self.config.model_id
            return result
        except Exception as exc:
            return ReviewResult(
                status="FALLBACK",
                decision="REVIEW",
                rationale="审稿模型不可用，保留规则验证结果并转人工复核。",
                error=str(exc)[:500],
                model_id=self.config.model_id,
            )

    def _call_model(self, prompt: str) -> Any:
        if self._generate is not None:
            return self._generate(prompt)

        from google import genai
        from google.genai import types

        kwargs = {"api_key": self.config.api_key}
        if self.config.api_base:
            kwargs["http_options"] = {"base_url": self.config.api_base}
        client = genai.Client(**kwargs)
        response = client.models.generate_content(
            model=self.config.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                response_mime_type="application/json",
            ),
        )
        return getattr(response, "text", response)

    @staticmethod
    def _parse_json(raw: Any) -> dict:
        if isinstance(raw, dict):
            return raw
        text = getattr(raw, "text", raw)
        if not isinstance(text, str):
            raise ValueError("reviewer returned a non-text response")
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("reviewer JSON must be an object")
        return payload

    @staticmethod
    def _score(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _normalize(cls, payload: dict) -> ReviewResult:
        decision = str(payload.get("decision", "REVIEW")).upper()
        if decision not in REVIEW_DECISIONS:
            decision = "REVIEW"

        def clean_dicts(value: Any) -> list[dict]:
            if not isinstance(value, list):
                return []
            return [item for item in value[:MAX_LIST_ITEMS] if isinstance(item, dict)]

        def clean_strings(value: Any) -> list[str]:
            if not isinstance(value, list):
                return []
            return [str(item)[:500] for item in value[:MAX_LIST_ITEMS] if item is not None]

        return ReviewResult(
            overall_score=cls._score(payload.get("overall_score", 0.0)),
            decision=decision,
            entity_reviews=clean_dicts(payload.get("entity_reviews")),
            relation_reviews=clean_dicts(payload.get("relation_reviews")),
            missing_items=clean_strings(payload.get("missing_items")),
            hallucination_flags=clean_strings(payload.get("hallucination_flags")),
            rationale=str(payload.get("rationale", ""))[:MAX_RATIONALE_CHARS],
        )

    @staticmethod
    def _build_prompt(text: str, extraction: dict, verification: dict, pmid: str) -> str:
        safe_extraction = json.dumps(extraction, ensure_ascii=False, default=str)[:16000]
        safe_verification = json.dumps(verification, ensure_ascii=False, default=str)[:12000]
        return f"""你是生物医学知识图谱抽取审稿人。只依据给定文章原文审查抽取结果，不补写事实。
请输出严格 JSON 对象，不要 Markdown。decision 只能是 ACCEPT、REVIEW 或 REJECT，overall_score 为 0 到 1。
检查实体是否在原文有依据、关系方向和证据是否成立、关系类型是否符合语义、是否存在幻觉或遗漏。
JSON 字段：overall_score, decision, entity_reviews, relation_reviews, missing_items, hallucination_flags, rationale。
每个 entity/relation review 至少可包含 item、score、issue、evidence。
PMID: {pmid}
文章原文（最多 {MAX_TEXT_CHARS} 字符）：
{text[:MAX_TEXT_CHARS]}
抽取结果：
{safe_extraction}
规则验证结果：
{safe_verification}
"""
