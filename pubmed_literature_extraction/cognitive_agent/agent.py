#!/usr/bin/env python3
"""
cognitive_agent/agent.py — 自主认知知识管理 Agent 主循环

基于 LangExtract + 原生 Gemini-compatible provider，以 Neo4j 为外部动态记忆，
实现 7 阶段闭环：Context → Extract → Verify → Conflict → Decision → Reflection → Adapt。

用法:
    python3 -m cognitive_agent.agent \
        --input extraction_output/pubmed_converted_500.jsonl \
        --limit 50 \
        --run-id agent_v1_50 \
        --skip-neo4j-write
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
import threading
from urllib.parse import urlparse
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from cognitive_agent.memory.kg_memory import KGMemory
from cognitive_agent.memory.working_memory import WorkingMemory
from cognitive_agent.memory.episodic_memory import Episode, EpisodicMemory
from cognitive_agent.context_activator import ContextActivator, ContextCard
from cognitive_agent.extraction_kernel import ExtractionKernel, RawExtraction
from cognitive_agent.verifier import KGVerifier, VerifiedExtraction
from cognitive_agent.reviewer import ExtractionReviewer, ReviewerConfig, ReviewResult
from cognitive_agent.collaborative_extractor import (
    CollaborationResult,
    CollaborativeConfig,
    CollaborativeExtractor,
)
from cognitive_agent.rag_context import ControlledNeo4jRAG, RAGConfig, RAGContext
from cognitive_agent.decision_engine import DecisionEngine, ExecutionLog
from cognitive_agent.causal_reasoner import CausalReasoner
from cognitive_agent.conflict_resolver import ConflictResolver, ResolutionResult
from cognitive_agent.self_reflection import SelfReflection
from cognitive_agent.strategy_manager import StrategyManager
from cognitive_agent.tool_router import ArticleToolRouter, ToolPlan
from cognitive_agent.schema.examples import KG_EXTRACTION_PROMPT, DEFAULT_EXAMPLES, ALL_EXAMPLES
from cognitive_agent.abbreviation_detector import AbbreviationDetector, AbbreviationMap
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.agentic_controller import (
    AgentAction,
    AgenticArticleController,
    partition_recovery_relations,
)
from cognitive_agent.article_chunker import ArticleChunker
from cognitive_agent.golden_examples import GoldenExampleSelector
from cognitive_agent.article_preprocessing import ParallelArticlePreprocessor
from cognitive_agent.extraction_cache import LightweightExtractionCache
from cognitive_agent.relation_pair_classifier import (
    BioREDPairClassifier,
    PairClassifierConfig,
)
from cognitive_agent.central_agent_v2 import CentralAgentV2

import langextract as lx
from langextract.factory import ModelConfig


@dataclass
class AgentConfig:
    """Agent 配置"""
    # Gemini-compatible proxy
    api_key: str = ""
    api_base: str = "https://new.bitexingai.com"  # 原生 Gemini: SDK 自动追加 /v1beta/...
    model_id: str = "[按次]gemini-2.5-flash"  # 原生 Gemini 2.5 支持 response_schema

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # 行为
    skip_neo4j_write: bool = True
    max_workers: int = 5  # 并发数 (API 限制约 3-5)
    extraction_inner_max_workers: int = 2
    temperature: float = 0.0
    reflection_interval: int = 10  # 每 N 篇做一次策略反思
    tool_router_enabled: bool = True
    shadow_router_enabled: bool = True
    # legacy executes the old plan; active_shadow executes the audited plan.
    # Active mode is intentionally accepted only for dry-run/read-only experiments.
    router_execution_mode: str = "legacy"
    router_pre_context_max_mentions: int = 8

    # Central Agent v2. Legacy remains the compatibility default; shadow only
    # records a counterfactual plan, while active v2 is restricted to dry-run.
    execution_mode: str = "legacy"
    agent_budget_profile: str = "quality"
    agent_max_actions: int = 0
    agent_max_aux_remote_calls: int = 0
    agent_max_neo4j_calls: int = 0
    agent_soft_timeout: float = 0.0
    agent_hard_timeout: float = 180.0

    # 调试期可选的 LLM 抽取审稿器；默认关闭，不属于最终生产链路
    reviewer_enabled: bool = False
    reviewer_api_key: str = ""
    reviewer_api_base: str = ""
    reviewer_model_id: str = ""
    reviewer_max_output_tokens: int = 2048

    # Phase B: conditional second-model collaboration; disabled by default.
    second_llm_enabled: bool = False
    second_llm_provider: str = "openai"
    second_llm_api_key: str = ""
    second_llm_api_base: str = "https://api.deepseek.com"
    second_llm_model_id: str = "deepseek-v4-flash"
    second_llm_mode: str = "conditional"
    second_llm_timeout: float = 45.0
    # Omit provider max_tokens by default; DeepSeek enforces its model limit.
    second_llm_max_output_tokens: int | None = None
    second_llm_thinking_enabled: bool = False

    # Phase C: bounded read-only Neo4j RAG; disabled by default.
    neo4j_rag_enabled: bool = False
    rag_max_entities: int = 12
    rag_max_candidates_per_entity: int = 3
    rag_max_total_candidates: int = 20
    rag_max_neighbors_per_candidate: int = 4
    rag_max_evidence_chars: int = 500
    rag_max_total_chars: int = 6000
    agent_max_recovery_candidates: int = 12
    agent_max_evidence_repairs: int = 16
    agent_min_recovery_score: float = 0.55
    # precision: review/delete only; shadow-agent: evaluate recovery but never
    # expose it to the write path; recall: recovery may enter final decisions.
    agent_mode: str = "precision"
    golden_shot_enabled: bool = True
    golden_shot_max_examples: int = 4
    chunked_extraction_enabled: bool = True
    extraction_chunk_max_chars: int = 1800
    extraction_chunk_complexity_min_chars: int = 1400
    extraction_chunk_max_chunks: int = 3
    # Lightweight cache: memory is bounded and creates no files. Persistent
    # SQLite is opt-in for repeatable experiments only.
    extraction_cache_mode: str = "memory"
    extraction_cache_path: str = ".cache/langextract_candidates.sqlite3"
    extraction_cache_memory_entries: int = 256
    extraction_cache_max_entries: int = 2000
    extraction_cache_max_mb: int = 200
    extraction_cache_ttl_days: int = 30

    # BioRED-style entity-pair relation classification.  Shadow is the safe
    # default: it records counterfactual predictions without changing the
    # production relation path.  Active remains dry-run only until calibrated.
    pair_classifier_enabled: bool = True
    pair_classifier_mode: str = "shadow"  # off | shadow | active
    pair_classifier_backend: str = "deterministic"
    pair_classifier_model_path: str = ""
    pair_classifier_high_confidence: float = 0.78
    pair_classifier_relation_threshold: float = 0.48
    pair_classifier_uncertainty_floor: float = 0.35
    pair_classifier_max_candidates: int = 64
    pair_classifier_max_llm_candidates: int = 12

    # 阈值
    entity_creation_min_confidence: float = 0.7
    relation_creation_min_confidence: float = 0.7


@dataclass
class AgentState:
    """Agent 运行时状态（线程安全）"""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    total_articles: int = 0
    total_extractions: int = 0
    total_entities_extracted: int = 0
    total_relations_extracted: int = 0
    total_entities_created: int = 0
    total_relations_created: int = 0
    total_relations_updated: int = 0
    total_disputed: int = 0
    total_discarded: int = 0
    total_import_ready: int = 0
    quality_scores: list[float] = field(default_factory=list)
    structural_scores: list[float] = field(default_factory=list)
    semantic_scores: list[float] = field(default_factory=list)
    evidence_scores: list[float] = field(default_factory=list)
    quality_metric_totals: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)

    def safe_add(self, **kwargs):
        """线程安全地原子递增多个计数器"""
        with self._lock:
            for key, delta in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, getattr(self, key) + delta)

    def safe_append(self, attr: str, value):
        """线程安全地向列表属性追加元素"""
        with self._lock:
            lst = getattr(self, attr)
            lst.append(value)

    def safe_merge_metrics(self, metrics: dict):
        """Accumulate explicit metric numerators and denominators."""
        with self._lock:
            for name, metric in metrics.items():
                target = self.quality_metric_totals.setdefault(
                    name, {"count": 0, "denominator": 0}
                )
                target["count"] += int(metric.get("count", 0) or 0)
                target["denominator"] += int(metric.get("denominator", 0) or 0)


class CognitiveAgent:
    """
    自主认知知识管理 Agent

    对每篇 PubMed 摘要执行完整的认知推理循环:
    Context → Extract → Verify → Decide → (Reflect → Adapt)
    """

    def __init__(self, config: AgentConfig):
        if config.agent_mode not in {"precision", "shadow-agent", "recall"}:
            raise ValueError("agent_mode must be precision, shadow-agent, or recall")
        if config.router_execution_mode not in {"legacy", "active_shadow"}:
            raise ValueError("router_execution_mode must be legacy or active_shadow")
        if config.router_execution_mode == "active_shadow" and not config.skip_neo4j_write:
            raise ValueError("active_shadow router is restricted to dry-run execution")
        if config.extraction_cache_mode not in LightweightExtractionCache.VALID_MODES:
            raise ValueError("invalid extraction_cache_mode")
        if config.pair_classifier_mode not in {"off", "shadow", "active"}:
            raise ValueError("pair_classifier_mode must be off, shadow, or active")
        if config.pair_classifier_mode == "active" and not config.skip_neo4j_write:
            raise ValueError("active pair classifier is restricted to dry-run execution")
        if config.execution_mode not in CentralAgentV2.VALID_MODES:
            raise ValueError("invalid central Agent v2 execution_mode")
        if config.execution_mode == "agent-v2" and not config.skip_neo4j_write:
            raise ValueError("active Agent v2 is restricted to dry-run execution")
        if config.agent_budget_profile not in CentralAgentV2.VALID_PROFILES:
            raise ValueError("invalid central Agent v2 budget profile")
        if min(
            config.agent_max_actions, config.agent_max_aux_remote_calls,
            config.agent_max_neo4j_calls, config.agent_soft_timeout,
        ) < 0 or config.agent_hard_timeout <= 0:
            raise ValueError("Agent v2 budget overrides must be non-negative")
        if not config.skip_neo4j_write:
            host = urlparse(config.neo4j_uri).hostname
            if host not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("Neo4j write is restricted to localhost")
            if config.neo4j_database != "neo4j":
                raise ValueError("Neo4j write is restricted to database 'neo4j'")
        self.config = config
        self.state = AgentState()
        self._history_lock = threading.Lock()  # 保护 history 列表的并发写入

        # 初始化组件
        self.kg_memory = KGMemory(
            uri=config.neo4j_uri,
            user=config.neo4j_user,
            password=config.neo4j_password,
            database=config.neo4j_database,
        )
        # Working memory is created per article in process_article().
        # It must not be shared across ThreadPoolExecutor workers.
        self.episodic_memory = EpisodicMemory()
        self.context_activator = ContextActivator(self.kg_memory)

        # LangExtract provider is selected from the endpoint protocol.  The
        # new `api-666.cc/v1` endpoint is OpenAI-compatible even though the
        # served model is Gemini; the historical proxy uses native Gemini.
        api_base_normalized = str(config.api_base or "").rstrip("/").casefold()
        openai_compatible_endpoint = api_base_normalized.endswith("/v1")
        extraction_provider = "openai" if openai_compatible_endpoint else "gemini"
        extraction_provider_kwargs = (
            {
                "api_key": config.api_key,
                "base_url": config.api_base,
                "temperature": 0.0,
            }
            if openai_compatible_endpoint
            else {
                "api_key": config.api_key,
                "http_options": {"base_url": config.api_base},
                "temperature": 0.0,
            }
        )
        # LangExtract model configuration.  Provider choice is transport-level;
        # `count.gmcli-gemini-3-flash-preview` remains the model identifier.
        lx_config = ModelConfig(
            provider=extraction_provider,
            model_id=config.model_id,
            provider_kwargs=extraction_provider_kwargs,
        )
        self.extraction_cache = LightweightExtractionCache(
            mode=config.extraction_cache_mode,
            path=config.extraction_cache_path,
            memory_max_entries=config.extraction_cache_memory_entries,
            persistent_max_entries=config.extraction_cache_max_entries,
            persistent_max_mb=config.extraction_cache_max_mb,
            ttl_days=config.extraction_cache_ttl_days,
        )
        self.extraction_kernel = ExtractionKernel(
            lx_config,
            cache=self.extraction_cache,
            inner_max_workers=config.extraction_inner_max_workers,
        )
        self.pair_classifier = BioREDPairClassifier(PairClassifierConfig(
            enabled=config.pair_classifier_enabled,
            mode=config.pair_classifier_mode,
            backend=config.pair_classifier_backend,
            model_path=config.pair_classifier_model_path,
            high_confidence_threshold=config.pair_classifier_high_confidence,
            relation_threshold=config.pair_classifier_relation_threshold,
            uncertainty_floor=config.pair_classifier_uncertainty_floor,
            max_candidates=config.pair_classifier_max_candidates,
            max_low_confidence_candidates=config.pair_classifier_max_llm_candidates,
        ))
        self.verifier = KGVerifier(self.kg_memory)
        self.reviewer = ExtractionReviewer(
            ReviewerConfig(
                api_key=config.reviewer_api_key or config.api_key,
                api_base=config.reviewer_api_base or config.api_base,
                model_id=config.reviewer_model_id,
                max_output_tokens=config.reviewer_max_output_tokens,
            )
        ) if config.reviewer_enabled else ExtractionReviewer()
        self.collaborative_extractor = CollaborativeExtractor(
            CollaborativeConfig(
                enabled=config.second_llm_enabled,
                provider=config.second_llm_provider,
                api_key=config.second_llm_api_key,
                api_base=config.second_llm_api_base,
                model_id=config.second_llm_model_id,
                mode=config.second_llm_mode,
                timeout=config.second_llm_timeout,
                max_output_tokens=config.second_llm_max_output_tokens,
                thinking_enabled=config.second_llm_thinking_enabled,
            )
        )
        self.rag_context_builder = ControlledNeo4jRAG(
            self.kg_memory,
            RAGConfig(
                enabled=config.neo4j_rag_enabled,
                max_entities=config.rag_max_entities,
                max_candidates_per_entity=config.rag_max_candidates_per_entity,
                max_total_candidates=config.rag_max_total_candidates,
                max_neighbors_per_candidate=config.rag_max_neighbors_per_candidate,
                max_evidence_chars=config.rag_max_evidence_chars,
                max_total_chars=config.rag_max_total_chars,
            ),
        )
        self.causal_reasoner = CausalReasoner(self.kg_memory)
        self.conflict_resolver = ConflictResolver()
        self.decision_engine = DecisionEngine(
            self.kg_memory, skip_neo4j_write=config.skip_neo4j_write
        )
        self.self_reflection = SelfReflection()
        self.strategy_manager = StrategyManager(config)
        self.abbreviation_detector = AbbreviationDetector()
        self.evidence_reader = ArticleEvidenceReader()
        self.agentic_controller = AgenticArticleController(
            max_recovery_candidates=config.agent_max_recovery_candidates,
            max_repairs=config.agent_max_evidence_repairs,
            enable_evidence_repair=config.agent_mode != "precision",
            enable_recovery=config.agent_mode != "precision",
            min_recovery_score=config.agent_min_recovery_score,
        )
        self.tool_router = ArticleToolRouter(enabled=config.tool_router_enabled)
        self.golden_example_selector = GoldenExampleSelector()
        self.article_chunker = ArticleChunker(
            max_chars=config.extraction_chunk_max_chars,
            complexity_min_chars=config.extraction_chunk_complexity_min_chars,
            max_chunks=config.extraction_chunk_max_chunks,
        )
        self.article_preprocessor = ParallelArticlePreprocessor(
            self.evidence_reader,
            self.abbreviation_detector,
            self.golden_example_selector,
            max_workers=min(8, max(2, config.max_workers)),
            cache_max_entries=max(4, config.extraction_cache_memory_entries * 4),
        )
        self.central_agent_v2 = CentralAgentV2(
            execution_mode=config.execution_mode,
            budget_profile=config.agent_budget_profile,
            max_actions=config.agent_max_actions,
            max_aux_remote_calls=config.agent_max_aux_remote_calls,
            max_neo4j_calls=config.agent_max_neo4j_calls,
            soft_timeout_s=config.agent_soft_timeout,
            hard_timeout_s=config.agent_hard_timeout,
        )

        # 运行时状态
        self.history: list[dict] = []
        self.current_examples = list(DEFAULT_EXAMPLES)

    def _collaborate_cache_first(self, *, enabled: bool, max_retries: int = 0, **kwargs):
        """Run the auxiliary adjudicator through the shared bounded cache.

        Legacy execution intentionally bypasses this wrapper. Agent v2 keys
        include model/tool versions and the full candidate state; credentials
        are never included. Only successful strict-JSON results are cached.
        """
        if not enabled:
            started = time.perf_counter()
            result = self.collaborative_extractor.collaborate(**kwargs)
            return result, "disabled", time.perf_counter() - started, 0

        cache_payload = {
            "tool_version": "second-llm-adjudicator-v2",
            "provider": self.config.second_llm_provider,
            "model_id": self.config.second_llm_model_id,
            "temperature": 0.0,
            "pmid": kwargs.get("pmid", ""),
            "text": kwargs.get("text", ""),
            "verification": kwargs.get("verification", {}),
            "recovery_candidates": kwargs.get("recovery_candidates", []),
            "pair_review_candidates": kwargs.get("pair_review_candidates", []),
            "rag_context": kwargs.get("rag_context", {}),
        }
        key = self.central_agent_v2.tool_cache_key("second_llm_refiner", cache_payload)

        def factory():
            started = time.perf_counter()
            retries = 0
            value = self.collaborative_extractor.collaborate(**kwargs)
            while retries < max(0, min(2, int(max_retries))) and value.status == "FALLBACK":
                retryable = any(token in str(value.error).casefold() for token in (
                    "timeout", "timed out", "rate", "429", "json", "temporar", "connection",
                ))
                if not retryable:
                    break
                retries += 1
                value = self.collaborative_extractor.collaborate(**kwargs)
            payload = value.to_dict()
            payload["_agent_v2_retry_count"] = retries
            return payload, time.perf_counter() - started

        started = time.perf_counter()
        payload, cache_status = self.extraction_cache.get_or_compute(
            key,
            factory,
            cacheable=lambda value: value.get("status") == "OK",
        )
        fields = CollaborationResult.__dataclass_fields__
        result = CollaborationResult(**{
            key: value for key, value in payload.items() if key in fields
        })
        return (
            result,
            cache_status,
            time.perf_counter() - started,
            int(payload.get("_agent_v2_retry_count", 0) or 0),
        )

    def _review_cache_first(self, *, enabled: bool, **kwargs):
        """Cache successful diagnostic reviews without changing their authority."""
        if not enabled:
            started = time.perf_counter()
            result = self.reviewer.review(**kwargs)
            return result, "disabled", time.perf_counter() - started
        payload_for_key = {
            "tool_version": "debug-reviewer-v2",
            "model_id": self.config.reviewer_model_id,
            "pmid": kwargs.get("pmid", ""),
            "text": kwargs.get("text", ""),
            "extraction": kwargs.get("extraction", {}),
            "verification": kwargs.get("verification", {}),
        }
        key = self.central_agent_v2.tool_cache_key("debug_reviewer", payload_for_key)

        def factory():
            started = time.perf_counter()
            value = self.reviewer.review(**kwargs)
            return value.to_dict(), time.perf_counter() - started

        started = time.perf_counter()
        payload, cache_status = self.extraction_cache.get_or_compute(
            key, factory, cacheable=lambda value: value.get("status") == "OK",
        )
        fields = ReviewResult.__dataclass_fields__
        result = ReviewResult(**{key: value for key, value in payload.items() if key in fields})
        return result, cache_status, time.perf_counter() - started

    def process_article(self, article: dict) -> dict:
        """
        处理单篇 PubMed 文章 — 完整的认知推理循环

        Args:
            article: {"pmid": ..., "title": ..., "abstract": ..., "source": "PubMed"}

        Returns:
            processing_record: 包含所有阶段的完整处理记录
        """
        pmid = article.get("pmid", "unknown")
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        text = f"TITLE: {title}\nABSTRACT: {abstract}"

        record = {
            "pmid": pmid,
            "title": title[:120],
            "abstract_length": len(abstract),
            "timestamp": time.time(),
            "phases": {},
        }

        t_start = time.time()

        try:
            # Per-article working memory: never share this scratchpad between workers.
            working_memory = WorkingMemory(current_article_pmid=pmid)
            working_memory.clear_article_session()
            working_memory.current_article_pmid = pmid

            # The planning layer is deterministic and adds no model/API call.
            # It records why each expensive tool is called, deferred, or skipped.
            pre_plan = self.tool_router.plan_before_extraction(
                title=title,
                abstract=abstract,
                memory_available=self.kg_memory.is_connected,
                rag_enabled=self.config.neo4j_rag_enabled,
                second_llm_enabled=self.config.second_llm_enabled,
                reviewer_enabled=self.config.reviewer_enabled,
                chunk_max_chars=self.config.extraction_chunk_max_chars,
                chunk_complexity_min_chars=self.config.extraction_chunk_complexity_min_chars,
            )
            record["phases"]["tool_plan_pre"] = pre_plan.to_dict()

            prepared = self.article_preprocessor.prepare(
                title=title, abstract=abstract, text=text,
                study_type=pre_plan.profile.study_type,
                max_examples=self.config.golden_shot_max_examples,
                document_id=str(pmid),
            )
            record["phases"]["local_preprocessing"] = {
                "mode": "parallel_content_addressed",
                "content_hash": prepared.content_hash,
                "cache_hits": prepared.cache_hits,
                "cached_components": sorted(
                    key for key, hit in prepared.cache_hits.items() if hit
                ),
            }
            shadow_pre_plan = None
            if self.config.shadow_router_enabled:
                shadow_pre_plan = self.tool_router.shadow_plan_before_extraction(
                    title=title, abstract=abstract, legacy_plan=pre_plan,
                    memory_available=self.kg_memory.is_connected,
                    rag_enabled=self.config.neo4j_rag_enabled,
                    second_llm_enabled=self.config.second_llm_enabled,
                    reviewer_enabled=self.config.reviewer_enabled,
                    chunk_max_chars=self.config.extraction_chunk_max_chars,
                    profile=prepared.profile,
                )
                record["phases"]["shadow_tool_plan_pre"] = shadow_pre_plan.to_dict()
                record["phases"]["shadow_tool_plan_pre_comparison"] = (
                    self.tool_router.compare_plans(pre_plan, shadow_pre_plan)
                )
            execution_pre_plan = (
                shadow_pre_plan
                if self.config.router_execution_mode == "active_shadow" and shadow_pre_plan is not None
                else pre_plan
            )
            record["phases"]["router_execution"] = {
                "mode": self.config.router_execution_mode,
                "pre_plan_source": execution_pre_plan.plan_status,
                "dry_run_guard": self.config.skip_neo4j_write,
            }

            # ═══════════════════════════════════════════════════
            # Phase 1: Context Activation — 先验知识激活
            # ═══════════════════════════════════════════════════
            t1 = time.time()
            if execution_pre_plan.should_call("context_memory"):
                context_card = self.context_activator.activate(
                    text,
                    pmid=pmid,
                    max_mentions=self.config.router_pre_context_max_mentions,
                )
                context_payload = context_card.to_dict()
                context_payload["status"] = "OK"
            else:
                # Neutral context prevents a skipped KG lookup from being
                # misinterpreted as low KG coverage/exploratory extraction.
                context_card = ContextCard(
                    pmid=pmid,
                    extraction_goals=["balanced_extraction"],
                    coverage_score=0.35,
                )
                context_payload = context_card.to_dict()
                context_payload.update({
                    "status": "SKIPPED_BY_ROUTER",
                    "reason": execution_pre_plan.decisions["context_memory"].reason,
                })
            record["phases"]["context"] = context_payload
            # Save extraction targets in working memory
            working_memory.extraction_targets = context_card.extraction_goals
            t1_end = time.time()

            # ── Strategy adaptation from context ──
            strategy = self.strategy_manager.get_strategy(context_card)
            evidence_units = prepared.evidence_units
            golden_selection = prepared.golden_selection
            extraction_examples = (
                golden_selection.examples
                if self.config.golden_shot_enabled
                else self._select_examples(strategy)
            )
            extraction_chunks = (
                self.article_chunker.build(
                    text, evidence_units,
                    high_complexity=execution_pre_plan.profile.high_complexity,
                )
                if (self.config.chunked_extraction_enabled
                    and execution_pre_plan.should_call("article_chunker"))
                else []
            )
            section_counts: dict[str, int] = {}
            for unit in evidence_units:
                section_counts[unit.section] = section_counts.get(unit.section, 0) + 1
            record["phases"]["reader"] = {
                "mode": "extractive_span_preserving",
                "unit_count": len(evidence_units),
                "section_counts": section_counts,
                "complex_sentence_units": sum(
                    1 for unit in evidence_units
                    if sum(x.parent_sentence_id == unit.parent_sentence_id for x in evidence_units) > 1
                ),
                "units": [unit.to_dict() for unit in evidence_units],
                "chunking": {
                    "enabled": self.config.chunked_extraction_enabled,
                    "used": len(extraction_chunks) > 1,
                    "chunk_count": len(extraction_chunks) or 1,
                    "chunks": [chunk.to_dict() for chunk in extraction_chunks],
                },
            }
            extraction_prompt = self._build_strategy_prompt(
                KG_EXTRACTION_PROMPT,
                context_card=context_card,
                strategy=strategy,
                tool_plan=execution_pre_plan,
                evidence_units=evidence_units,
            )
            record["phases"]["strategy"] = {
                "active_strategy": strategy,
                "example_count": len(extraction_examples),
                "example_policy": (
                    "dynamic_3plus1_or_2plus2_golden_boundary"
                    if self.config.golden_shot_enabled else "legacy_strategy_examples"
                ),
                "example_names": (
                    golden_selection.names if self.config.golden_shot_enabled else []
                ),
                "example_selection_reasons": (
                    golden_selection.reasons if self.config.golden_shot_enabled else []
                ),
                "self_example_excluded": str(pmid) in {"41650163", "41482383"},
                "extraction_goals": context_card.extraction_goals,
            }

            # ── v3: Abbreviation detection (Schwartz-Hearst) ──
            abbr_map = prepared.abbreviation_map

            # ═══════════════════════════════════════════════════
            # Phase 2: Extract + Ground — LangExtract 提取
            # ═══════════════════════════════════════════════════
            t2 = time.time()
            if len(extraction_chunks) > 1:
                raw_extraction = self.extraction_kernel.extract_chunked(
                    chunks=extraction_chunks,
                    full_text=text,
                    document_id=pmid,
                    examples=extraction_examples,
                    prompt=extraction_prompt,
                )
            else:
                raw_extraction = self.extraction_kernel.extract(
                    text=text,
                    document_id=pmid,
                    examples=extraction_examples,
                    prompt=extraction_prompt,
                )
                raw_extraction.chunk_count = 1
                raw_extraction.chunks = [
                    chunk.to_dict() for chunk in extraction_chunks
                ]
            record["phases"]["extraction"] = raw_extraction.to_dict()
            t2_end = time.time()

            # BioRED-style relation core: classify grounded, schema-compatible
            # entity pairs.  LangExtract relations are hints, never labels.
            pair_result = self.pair_classifier.classify(
                entities=raw_extraction.entities,
                relations=raw_extraction.relations,
                units=evidence_units,
            )
            record["phases"]["relation_pair_classification"] = pair_result.to_dict()
            if self.config.pair_classifier_mode == "active":
                relation_core_relations = []
                relation_core_keys: set[tuple] = set()
                for relation in [
                    *pair_result.accepted_relations,
                    *pair_result.low_confidence_relations,
                ]:
                    key = (
                        relation.get("candidate_id"), relation.get("predicate"),
                        relation.get("subject"), relation.get("object"),
                    )
                    if key not in relation_core_keys:
                        relation_core_keys.add(key)
                        relation_core_relations.append(relation)
            else:
                relation_core_relations = raw_extraction.relations
            record["phases"]["relation_core_selection"] = {
                "mode": self.config.pair_classifier_mode,
                "production_source": (
                    "biored_entity_pair_classifier"
                    if self.config.pair_classifier_mode == "active"
                    else "langextract_legacy_relations"
                ),
                "langextract_relation_count": len(raw_extraction.relations),
                "pair_relation_count": len(pair_result.accepted_relations),
                "production_relation_count": len(relation_core_relations),
                "dry_run_guard": self.config.skip_neo4j_write,
            }

            # The central controller observes the first-pass state and may
            # select only lossless local tools before deterministic checking.
            agent_plan = self.agentic_controller.plan(
                text=text,
                raw_entities=raw_extraction.entities,
                raw_relations=relation_core_relations,
                abbr_map=abbr_map,
                units=evidence_units,
            )

            # ═══════════════════════════════════════════════════
            # Phase 3: Verify → conditional second model → re-verify
            # ═══════════════════════════════════════════════════
            t3 = time.time()
            initial_verified = self.verifier.verify(
                raw_entities=raw_extraction.entities,
                raw_relations=agent_plan.repaired_relations,
                pmid=pmid,
                text=text,
            )
            self.agentic_controller.add_recovery_observation(
                plan=agent_plan,
                text=text,
                verified_entities=[item.to_dict() for item in initial_verified.entities],
                verified_relations=[item.to_dict() for item in initial_verified.relations],
                abbr_map=abbr_map,
                units=evidence_units,
            )

            post_plan = self.tool_router.plan_after_verification(
                pre_plan=pre_plan,
                extraction=raw_extraction.to_dict(),
                verification=initial_verified.to_dict(),
                memory_available=self.kg_memory.is_connected,
                rag_enabled=self.config.neo4j_rag_enabled,
                second_llm_enabled=self.config.second_llm_enabled,
                second_llm_mode=self.config.second_llm_mode,
                reviewer_enabled=self.config.reviewer_enabled,
                recovery_candidate_count=len(agent_plan.recovery_candidates),
            )
            record["phases"]["tool_plan_post"] = post_plan.to_dict()
            shadow_post_plan = None
            if shadow_pre_plan is not None:
                shadow_post_plan = self.tool_router.shadow_plan_after_verification(
                    shadow_pre_plan=shadow_pre_plan,
                    legacy_post_plan=post_plan,
                    extraction=raw_extraction.to_dict(),
                    verification=initial_verified.to_dict(),
                    memory_available=self.kg_memory.is_connected,
                    rag_enabled=self.config.neo4j_rag_enabled,
                    second_llm_enabled=self.config.second_llm_enabled,
                    reviewer_enabled=self.config.reviewer_enabled,
                    recovery_candidate_count=len(agent_plan.recovery_candidates),
                )
                record["phases"]["shadow_tool_plan_post"] = shadow_post_plan.to_dict()
                record["phases"]["shadow_tool_plan_post_comparison"] = (
                    self.tool_router.compare_plans(post_plan, shadow_post_plan)
                )
            execution_post_plan = (
                shadow_post_plan
                if self.config.router_execution_mode == "active_shadow" and shadow_post_plan is not None
                else post_plan
            )
            record["phases"]["router_execution"].update({
                "post_plan_source": execution_post_plan.plan_status,
                "production_execution_unchanged": self.config.router_execution_mode == "legacy",
            })

            # Agent v2 observes the exact same grounded candidates as legacy.
            # Shadow mode only records counterfactual decisions; active mode may
            # replace post-verification tool gates, but can never bypass verify.
            v2_state = None
            if self.central_agent_v2.enabled:
                v2_route_source = shadow_post_plan or post_plan
                v2_state = self.central_agent_v2.start(pmid, v2_route_source.route)
                self.central_agent_v2.observe(
                    v2_state,
                    entities=[item.to_dict() for item in initial_verified.entities],
                    candidate_pairs=[item.to_dict() for item in pair_result.candidates],
                    verification=initial_verified.to_dict(),
                    extraction_error=raw_extraction.error,
                    recovery_candidate_count=len(agent_plan.recovery_candidates),
                )
                primary_cache_statuses: list[str] = []
                for warning in raw_extraction.warnings:
                    warning_text = str(warning)
                    if "extraction_cache:" not in warning_text:
                        continue
                    status = warning_text.rsplit("extraction_cache:", 1)[-1].strip()
                    primary_cache_statuses.append(status)
                    counter = {
                        "memory_hit": "memory_hits",
                        "persistent_hit": "persistent_hits",
                        "miss": "misses",
                        "singleflight_shared": "singleflight_shared",
                    }.get(status)
                    if counter:
                        v2_state.cache[counter] += 1
                    if status in {"memory_hit", "persistent_hit", "singleflight_shared"}:
                        v2_state.cache["remote_calls_avoided"] += 1
                bootstrap_fingerprint = v2_state.fingerprint()
                for tool, details in (
                    ("article_preprocessing", {"cache_hits": prepared.cache_hits}),
                    ("langextract_candidate_generator", {
                        "chunk_count": raw_extraction.chunk_count,
                        "primary_remote_requests": (
                            sum(status not in {
                                "memory_hit", "persistent_hit", "singleflight_shared",
                            } for status in primary_cache_statuses)
                            if primary_cache_statuses else (raw_extraction.chunk_count or 1)
                        ),
                        "cache_statuses": primary_cache_statuses,
                    }),
                    ("biored_pair_classifier", {"candidate_count": len(pair_result.candidates)}),
                    ("deterministic_verifier", {
                        "relation_count": len(initial_verified.relations),
                    }),
                ):
                    self.central_agent_v2.record_action(
                        v2_state, tool=tool, decision="CALL", reason="mandatory_bootstrap",
                        before=bootstrap_fingerprint, after=bootstrap_fingerprint,
                        details=details,
                    )

            rag_call = execution_post_plan.should_call("neo4j_rag")
            rag_reason = execution_post_plan.decisions["neo4j_rag"].reason
            if v2_state is not None:
                v2_rag_call, v2_rag_reason = self.central_agent_v2.should_use_rag(
                    v2_state,
                    enabled=self.config.neo4j_rag_enabled,
                    memory_available=self.kg_memory.is_connected,
                )
                if self.central_agent_v2.active:
                    rag_call, rag_reason = v2_rag_call, v2_rag_reason
                elif not v2_rag_call:
                    rag_reason = v2_rag_reason
            rag_started = time.perf_counter()
            if rag_call:
                rag_context = self.rag_context_builder.build(
                    entities=[item.to_dict() for item in initial_verified.entities],
                    known_entities=context_card.known_entities,
                    focus_relations=[item.to_dict() for item in initial_verified.relations],
                )
            else:
                rag_context = RAGContext(
                    status="SKIPPED_BY_ROUTER",
                    error=rag_reason,
                )
            record["phases"]["rag_context"] = rag_context.to_dict()

            if v2_state is not None:
                self.central_agent_v2.record_action(
                    v2_state,
                    tool="neo4j_rag",
                    decision=(
                        "CALL" if rag_call and self.central_agent_v2.active
                        else ("OBSERVED_LEGACY_CALL" if rag_call else (
                            "WOULD_CALL" if v2_rag_call else "SKIP"
                        ))
                    ),
                    reason=rag_reason,
                    latency_s=time.perf_counter() - rag_started,
                    neo4j=bool(rag_call and self.central_agent_v2.active),
                    result_status=rag_context.status,
                    details={"shadow": not self.central_agent_v2.active},
                )

            llm_call = execution_post_plan.should_call("second_llm_refiner")
            llm_reason = execution_post_plan.decisions["second_llm_refiner"].reason
            if v2_state is not None:
                v2_llm_call, v2_llm_reason = self.central_agent_v2.should_adjudicate(
                    v2_state, enabled=self.config.second_llm_enabled,
                )
                if (
                    self.central_agent_v2.active and not v2_llm_call
                    and "budget_exhausted" in v2_llm_reason
                    and self.central_agent_v2.maybe_escalate(
                        v2_state, "high_value_unresolved_relation_requires_adjudication"
                    )
                ):
                    v2_llm_call, v2_llm_reason = self.central_agent_v2.should_adjudicate(
                        v2_state, enabled=self.config.second_llm_enabled,
                    )
                if self.central_agent_v2.active:
                    llm_call, llm_reason = v2_llm_call, v2_llm_reason
                elif not v2_llm_call:
                    llm_reason = v2_llm_reason

            collaboration_cache_status = "not_applicable"
            collaboration_actual_latency = 0.0
            collaboration_retry_count = 0
            v2_llm_before = v2_state.fingerprint() if v2_state is not None else ""
            if llm_call:
                collaboration, collaboration_cache_status, collaboration_actual_latency, collaboration_retry_count = (
                    self._collaborate_cache_first(
                        enabled=self.central_agent_v2.active,
                        max_retries=(
                            min(2, max(0, v2_state.budget.max_aux_remote_calls - v2_state.aux_remote_calls - 1))
                            if v2_state is not None and self.central_agent_v2.active else 0
                        ),
                        text=text,
                        extraction=raw_extraction.to_dict(),
                        verification=initial_verified.to_dict(),
                        pmid=pmid,
                        context=context_card.to_dict(),
                        rag_context=(
                            {
                                "usage_policy": rag_context.usage_policy,
                                "entity_contexts": rag_context.entity_contexts,
                            }
                            if rag_context.status == "OK" else {}
                        ),
                        router_reasons=execution_post_plan.reason_codes,
                        recovery_candidates=agent_plan.recovery_candidates,
                        pair_review_candidates=(
                            pair_result.low_confidence_relations
                            if self.config.pair_classifier_mode == "active" else []
                        ),
                    )
                )
            else:
                collaboration = CollaborationResult(
                    status="SKIPPED_BY_ROUTER",
                    triggered=False,
                    trigger_reasons=execution_post_plan.reason_codes,
                    review_reason=llm_reason,
                    model_id=self.config.second_llm_model_id,
                )
            merged = self.collaborative_extractor.merge(
                raw_entities=raw_extraction.entities,
                raw_relations=agent_plan.repaired_relations,
                initial_verification=initial_verified.to_dict(),
                collaboration=collaboration,
            )
            collaboration_payload = collaboration.to_dict()
            collaboration_payload["merge"] = merged.to_dict()
            collaboration_payload["phase_a_reverification_required"] = bool(
                merged.entity_additions
                or merged.relation_additions
                or merged.relation_edits
                or merged.relation_rejections
                or merged.deterministic_rejections
                or merged.second_model_rejections
            )
            record["phases"]["collaboration"] = collaboration_payload
            record["phases"]["tool_marginal_benefit"] = {
                "second_llm_refiner": {
                    "called": bool(collaboration.triggered),
                    "input_candidates": len(collaboration.review_candidates),
                    "output_decisions": len(collaboration.review_decisions),
                    "relation_additions": merged.relation_additions,
                    "relation_edits": merged.relation_edits,
                    "relation_rejections": merged.relation_rejections,
                    "latency_seconds": float(collaboration.latency_s or 0.0),
                    "prompt_tokens": int(collaboration.prompt_tokens or 0),
                    "output_tokens": int(collaboration.output_tokens or 0),
                    "state_changes_per_second": round(
                        (merged.relation_additions + merged.relation_edits + merged.relation_rejections)
                        / max(float(collaboration.latency_s or 0.0), 0.001), 4,
                    ) if collaboration.triggered else 0.0,
                }
            }

            agent_plan.actions.append(AgentAction(
                tool="deepseek_relation_adjudicator",
                decision="CALL" if collaboration.triggered else "SKIP",
                reason=collaboration.review_reason or collaboration.status,
                input_count=len(collaboration.review_candidates),
                output_count=len(collaboration.review_decisions),
                latency_class="remote",
            ))

            # Deterministic blockers apply even when the second model is
            # skipped or falls back.  Every edit/removal must therefore pass
            # through Phase-A verification before decisions are made.
            if collaboration_payload["phase_a_reverification_required"]:
                verified = self.verifier.verify(
                    raw_entities=merged.entities,
                    raw_relations=merged.relations,
                    pmid=pmid,
                    text=text,
                )
            else:
                verified = initial_verified

            # Active Agent v2 may perform several bounded adjudication rounds.
            # A new round is legal only after the previous round changed the
            # verified state; identical states are cache-deduplicated.
            collaboration_rounds = [{
                "round": 1,
                "status": collaboration.status,
                "triggered": collaboration.triggered,
                "cache_status": collaboration_cache_status,
                "relation_additions": merged.relation_additions,
                "relation_edits": merged.relation_edits,
                "relation_rejections": merged.relation_rejections,
                "latency_s": collaboration_actual_latency,
                "prompt_tokens": collaboration.prompt_tokens,
                "output_tokens": collaboration.output_tokens,
                "retry_count": collaboration_retry_count,
            }]
            aggregate_merge = {
                "entity_additions": 0,
                "relation_additions": merged.relation_additions,
                "relation_edits": merged.relation_edits,
                "relation_rejections": merged.relation_rejections,
            }
            if v2_state is not None:
                self.central_agent_v2.observe(
                    v2_state,
                    entities=[item.to_dict() for item in verified.entities],
                    candidate_pairs=[item.to_dict() for item in pair_result.candidates],
                    verification=verified.to_dict(),
                    extraction_error=raw_extraction.error,
                    recovery_candidate_count=len(agent_plan.recovery_candidates),
                )
                v2_llm_after = v2_state.fingerprint()
                v2_state.fingerprints_seen.add(v2_llm_before)
                self.central_agent_v2.record_action(
                    v2_state,
                    tool="second_llm_refiner",
                    decision=(
                        "CALL" if llm_call and self.central_agent_v2.active
                        else ("OBSERVED_LEGACY_CALL" if llm_call else (
                            "WOULD_CALL" if v2_llm_call else "SKIP"
                        ))
                    ),
                    reason=llm_reason,
                    before=v2_llm_before,
                    after=v2_llm_after,
                    latency_s=collaboration_actual_latency,
                    cache_status=collaboration_cache_status,
                    remote=bool(llm_call and self.central_agent_v2.active),
                    result_status=collaboration.status,
                    prompt_tokens=collaboration.prompt_tokens,
                    output_tokens=collaboration.output_tokens,
                    attempt_count=(1 + collaboration_retry_count) if collaboration.triggered else 0,
                    retry_count=collaboration_retry_count,
                    details={"round": 1, "shadow": not self.central_agent_v2.active},
                )

            if self.central_agent_v2.active and v2_state is not None:
                for round_index in range(2, 5):
                    allowed, next_reason = self.central_agent_v2.should_adjudicate(
                        v2_state, enabled=self.config.second_llm_enabled,
                    )
                    if (
                        not allowed and "budget_exhausted" in next_reason
                        and self.central_agent_v2.maybe_escalate(
                            v2_state, "new_verified_state_requires_another_adjudication_round"
                        )
                    ):
                        allowed, next_reason = self.central_agent_v2.should_adjudicate(
                            v2_state, enabled=self.config.second_llm_enabled,
                        )
                    before = v2_state.fingerprint()
                    if not allowed:
                        self.central_agent_v2.record_action(
                            v2_state, tool="second_llm_refiner", decision="SKIP",
                            reason=next_reason, before=before, after=before,
                            details={"round": round_index},
                        )
                        break
                    if before in v2_state.fingerprints_seen:
                        self.central_agent_v2.record_action(
                            v2_state, tool="second_llm_refiner", decision="SKIP",
                            reason="identical_state_deduplicated", before=before, after=before,
                            details={"round": round_index},
                        )
                        break
                    v2_state.fingerprints_seen.add(before)
                    round_extraction = raw_extraction.to_dict()
                    round_extraction["entities"] = merged.entities
                    round_extraction["relations"] = merged.relations
                    next_collaboration, next_cache_status, next_actual_latency, next_retry_count = (
                        self._collaborate_cache_first(
                            enabled=True,
                            max_retries=min(2, max(
                                0,
                                v2_state.budget.max_aux_remote_calls
                                - v2_state.aux_remote_calls - 1,
                            )),
                            text=text,
                            extraction=round_extraction,
                            verification=verified.to_dict(),
                            pmid=pmid,
                            context=context_card.to_dict(),
                            rag_context=(
                                {
                                    "usage_policy": rag_context.usage_policy,
                                    "entity_contexts": rag_context.entity_contexts,
                                }
                                if rag_context.status == "OK" else {}
                            ),
                            router_reasons=[*execution_post_plan.reason_codes, f"agent_v2_round_{round_index}"],
                            recovery_candidates=agent_plan.recovery_candidates,
                            pair_review_candidates=(
                                pair_result.low_confidence_relations
                                if self.config.pair_classifier_mode == "active" else []
                            ),
                        )
                    )
                    next_merged = self.collaborative_extractor.merge(
                        raw_entities=merged.entities,
                        raw_relations=merged.relations,
                        initial_verification=verified.to_dict(),
                        collaboration=next_collaboration,
                    )
                    round_changes = (
                        next_merged.relation_additions
                        + next_merged.relation_edits
                        + next_merged.relation_rejections
                    )
                    if round_changes:
                        merged = next_merged
                        verified = self.verifier.verify(
                            raw_entities=merged.entities,
                            raw_relations=merged.relations,
                            pmid=pmid,
                            text=text,
                        )
                    self.central_agent_v2.observe(
                        v2_state,
                        entities=[item.to_dict() for item in verified.entities],
                        candidate_pairs=[item.to_dict() for item in pair_result.candidates],
                        verification=verified.to_dict(),
                        extraction_error=raw_extraction.error,
                        recovery_candidate_count=len(agent_plan.recovery_candidates),
                    )
                    after = v2_state.fingerprint()
                    self.central_agent_v2.record_action(
                        v2_state, tool="second_llm_refiner", decision="CALL",
                        reason=next_reason, before=before, after=after,
                        latency_s=next_actual_latency, cache_status=next_cache_status,
                        remote=True, result_status=next_collaboration.status,
                        prompt_tokens=next_collaboration.prompt_tokens,
                        output_tokens=next_collaboration.output_tokens,
                        attempt_count=(1 + next_retry_count) if next_collaboration.triggered else 0,
                        retry_count=next_retry_count,
                        details={"round": round_index},
                    )
                    collaboration_rounds.append({
                        "round": round_index,
                        "status": next_collaboration.status,
                        "triggered": next_collaboration.triggered,
                        "cache_status": next_cache_status,
                        "relation_additions": next_merged.relation_additions,
                        "relation_edits": next_merged.relation_edits,
                        "relation_rejections": next_merged.relation_rejections,
                        "latency_s": next_actual_latency,
                        "prompt_tokens": next_collaboration.prompt_tokens,
                        "output_tokens": next_collaboration.output_tokens,
                        "retry_count": next_retry_count,
                    })
                    aggregate_merge["relation_additions"] += next_merged.relation_additions
                    aggregate_merge["relation_edits"] += next_merged.relation_edits
                    aggregate_merge["relation_rejections"] += next_merged.relation_rejections
                    if not round_changes:
                        break

            collaboration_payload["rounds"] = collaboration_rounds
            collaboration_payload["round_count"] = len(collaboration_rounds)
            collaboration_payload["merge"].update(aggregate_merge)
            collaboration_payload["phase_a_reverification_required"] = bool(
                collaboration_payload["phase_a_reverification_required"]
                or aggregate_merge["relation_additions"]
                or aggregate_merge["relation_edits"]
                or aggregate_merge["relation_rejections"]
            )
            record["phases"]["collaboration"] = collaboration_payload
            record["phases"]["tool_marginal_benefit"]["second_llm_refiner"].update({
                "round_count": len(collaboration_rounds),
                "relation_additions": aggregate_merge["relation_additions"],
                "relation_edits": aggregate_merge["relation_edits"],
                "relation_rejections": aggregate_merge["relation_rejections"],
                "latency_seconds": round(sum(
                    float(item.get("latency_s", 0.0) or 0.0) for item in collaboration_rounds
                ), 4),
                "prompt_tokens": sum(
                    int(item.get("prompt_tokens", 0) or 0) for item in collaboration_rounds
                ),
                "output_tokens": sum(
                    int(item.get("output_tokens", 0) or 0) for item in collaboration_rounds
                ),
            })

            # Canonicalization itself can reveal a new duplicate (for example
            # PBC and its long form).  Run at most two local finalize/verify
            # passes: this is an agent loop, but it has a hard latency bound.
            finalization_passes: list[dict] = []
            finalization_changed = False
            for pass_index in range(2):
                finalized_relations, pass_audit = (
                    self.collaborative_extractor.finalize_after_reverification(
                        merged.relations, verified.to_dict(), source_text=text
                    )
                )
                pass_audit["pass"] = pass_index + 1
                pass_changed = bool(
                    pass_audit["rolled_back_count"]
                    or pass_audit["duplicate_relations_removed"]
                    or pass_audit["symmetric_orientation_changes"]
                )
                pass_audit["changed"] = pass_changed
                finalization_passes.append(pass_audit)
                if not pass_changed:
                    break
                finalization_changed = True
                merged.relations = finalized_relations
                verified = self.verifier.verify(
                    raw_entities=merged.entities,
                    raw_relations=merged.relations,
                    pmid=pmid,
                    text=text,
                )
            finalization_audit = {
                "passes": finalization_passes,
                "rolled_back_count": sum(
                    item["rolled_back_count"] for item in finalization_passes
                ),
                "duplicate_relations_removed": sum(
                    item["duplicate_relations_removed"] for item in finalization_passes
                ),
                "symmetric_orientation_changes": sum(
                    item["symmetric_orientation_changes"] for item in finalization_passes
                ),
            }
            collaboration_payload["post_action_finalization"] = finalization_audit
            shadow_payload = {
                "mode": self.config.agent_mode,
                "candidate_count": len(agent_plan.recovery_candidates),
                "adjudicated_additions": merged.relation_additions,
                "relations": [],
                "write_eligible": self.config.agent_mode == "recall",
            }
            if self.config.agent_mode == "shadow-agent":
                # Preserve the fully re-verified Agent result for offline
                # evaluation, then rebuild the production state without any
                # recovered edge.  This makes write separation structural,
                # rather than relying on a later flag check.
                production_relations, shadow_relations = partition_recovery_relations(
                    merged.relations,
                    [relation.to_dict() for relation in verified.relations],
                    self.config.agent_mode,
                )
                shadow_payload["relations"] = shadow_relations
                if len(production_relations) != len(merged.relations):
                    merged.relations = production_relations
                    verified = self.verifier.verify(
                        raw_entities=merged.entities,
                        raw_relations=merged.relations,
                        pmid=pmid,
                        text=text,
                    )
            elif self.config.agent_mode == "recall":
                shadow_payload["relations"] = [
                    relation.to_dict() for relation in verified.relations
                    if "agent_recovered_relation" in set(relation.quality_flags or [])
                ]
            collaboration_payload["recovery_partition"] = shadow_payload
            record["phases"]["verification"] = verified.to_dict()
            agent_plan.current_round = 2 if collaboration_payload[
                "phase_a_reverification_required"
            ] or finalization_changed else 1
            agent_plan.actions.append(AgentAction(
                tool="post_action_rollback_and_dedup",
                decision="CALL" if finalization_changed else "SKIP",
                reason=(
                    "rollback verifier-failed actions and consolidate duplicate triples"
                    if finalization_changed else "all actions survived and no duplicate triple remained"
                ),
                input_count=(
                    len(verified.relations)
                    + finalization_audit["rolled_back_count"]
                    + finalization_audit["duplicate_relations_removed"]
                ),
                output_count=len(verified.relations),
            ))
            agent_plan.actions.append(AgentAction(
                tool="deterministic_reverification",
                decision=(
                    "CALL" if collaboration_payload["phase_a_reverification_required"] else "SKIP"
                ),
                reason=(
                    "verify every repaired, removed, edited, or recovered relation"
                    if collaboration_payload["phase_a_reverification_required"]
                    else "no state-changing action"
                ),
                input_count=len(merged.relations),
                output_count=len(verified.relations),
            ))
            record["phases"]["agent_controller"] = agent_plan.to_dict()

            # The second model gets a chance to recover a total first-model
            # failure.  If it cannot, retain the original failure semantics.
            if raw_extraction.error and not merged.entities:
                record["error"] = raw_extraction.error
                self.state.safe_append(
                    "errors",
                    {"pmid": pmid, "phase": "extraction", "error": raw_extraction.error},
                )
                record["phases"]["causal_reasoning"] = {"error": "skipped"}
                record["phases"]["conflict_resolution"] = {"error": "skipped"}
                record["phases"]["execution"] = {"error": "skipped"}
                if v2_state is not None:
                    self.central_agent_v2.finalize(v2_state, "primary_extraction_failed")
                    record["phases"]["agent_v2"] = v2_state.to_dict()
                record["total_time_s"] = time.time() - t_start
                with self._history_lock:
                    self.history.append(record)
                self.state.safe_add(total_articles=1)
                return record

            if v2_state is not None:
                self.central_agent_v2.observe(
                    v2_state,
                    entities=[item.to_dict() for item in verified.entities],
                    candidate_pairs=[item.to_dict() for item in pair_result.candidates],
                    verification=verified.to_dict(),
                    extraction_error=raw_extraction.error,
                    recovery_candidate_count=len(agent_plan.recovery_candidates),
                )

            # 调试期可选：LLM 只审稿并记录意见，不直接改变抽取或写入权限。
            reviewer_call = execution_post_plan.should_call("debug_reviewer")
            reviewer_reason = execution_post_plan.decisions["debug_reviewer"].reason
            v2_reviewer_call = False
            if v2_state is not None:
                v2_reviewer_call, v2_reviewer_reason = self.central_agent_v2.should_review(
                    v2_state, enabled=self.config.reviewer_enabled,
                )
                if self.central_agent_v2.active:
                    reviewer_call, reviewer_reason = v2_reviewer_call, v2_reviewer_reason
            reviewer_before = v2_state.fingerprint() if v2_state is not None else ""
            reviewer_cache_status = "not_applicable"
            reviewer_latency = 0.0
            if reviewer_call:
                review, reviewer_cache_status, reviewer_latency = self._review_cache_first(
                    enabled=self.central_agent_v2.active,
                    text=text,
                    extraction=raw_extraction.to_dict(),
                    verification=verified.to_dict(),
                    pmid=pmid,
                )
                record["phases"]["review"] = review.to_dict()
                if v2_state is not None and review.decision in {"REVIEW", "REJECT"}:
                    v2_state.unresolved_issues = list(dict.fromkeys([
                        *v2_state.unresolved_issues, "debug_reviewer_attention",
                    ]))
            if v2_state is not None:
                reviewer_after = v2_state.fingerprint()
                self.central_agent_v2.record_action(
                    v2_state, tool="debug_reviewer",
                    decision=(
                        "CALL" if reviewer_call and self.central_agent_v2.active
                        else ("OBSERVED_LEGACY_CALL" if reviewer_call else (
                            "WOULD_CALL" if v2_reviewer_call else "SKIP"
                        ))
                    ),
                    reason=reviewer_reason,
                    before=reviewer_before,
                    after=reviewer_after,
                    latency_s=reviewer_latency,
                    cache_status=reviewer_cache_status,
                    remote=bool(
                        reviewer_call and self.central_agent_v2.active
                        and review.status != "DISABLED"
                    ),
                    result_status=(review.status if reviewer_call else "SKIPPED"),
                    details={"shadow": not self.central_agent_v2.active},
                )

            # Phase 3+: Causal chain inference
            causal_call = execution_post_plan.should_call("causal_reasoner")
            causal_reason = execution_post_plan.decisions["causal_reasoner"].reason
            v2_causal_call = False
            if v2_state is not None:
                v2_causal_call, v2_causal_reason = self.central_agent_v2.should_run_causal(v2_state)
                if self.central_agent_v2.active:
                    causal_call, causal_reason = v2_causal_call, v2_causal_reason
            causal_before = v2_state.fingerprint() if v2_state is not None else ""
            causal_chains = (
                self.causal_reasoner.infer_causal_chains(
                    [relation for relation in verified.relations if relation.import_ready]
                )
                if causal_call
                else []
            )
            verified.causal_chains = [c.to_dict() for c in causal_chains]
            hypothesis_candidates = [{
                "subject": chain.inferred_subject,
                "subject_type": chain.inferred_subject_type,
                "predicate": chain.inferred_predicate,
                "object": chain.inferred_object,
                "object_type": chain.inferred_object_type,
                "evidence": "",
                "direction": "unknown",
                "inferred": True,
                "import_ready": False,
                "quality_flags": ["causal_hypothesis", "missing_direct_article_evidence"],
                "derivation": chain.derivation,
                "confidence": chain.confidence,
            } for chain in causal_chains]
            hypothesis_verification = []
            if hypothesis_candidates:
                checked_hypotheses = self.verifier.verify(
                    raw_entities=[item.to_dict() for item in verified.entities],
                    raw_relations=hypothesis_candidates,
                    pmid=pmid,
                    text=text,
                )
                hypothesis_verification = [item.to_dict() for item in checked_hypotheses.relations]
                # Keep hypotheses structurally separate even if a future
                # verifier becomes more permissive.
                for item in hypothesis_verification:
                    item["import_ready"] = False
                    item["inferred"] = True
                    item["quality_flags"] = sorted(set(item.get("quality_flags", [])) | {
                        "causal_hypothesis", "missing_direct_article_evidence",
                    })
            if v2_state is not None:
                v2_state.hypothesis_relations = hypothesis_verification
                causal_after = v2_state.fingerprint()
                self.central_agent_v2.record_action(
                    v2_state, tool="causal_reasoner",
                    decision=(
                        "CALL" if causal_call and self.central_agent_v2.active
                        else ("OBSERVED_LEGACY_CALL" if causal_call else (
                            "WOULD_CALL" if v2_causal_call else "SKIP"
                        ))
                    ),
                    reason=causal_reason, before=causal_before, after=causal_after,
                    result_status="OK" if causal_call else "SKIPPED",
                    details={"hypothesis_count": len(hypothesis_verification)},
                )
            record["phases"]["causal_reasoning"] = {
                "chains_inferred": len(causal_chains),
                "chains": verified.causal_chains,
                "hypothesis_relations": hypothesis_verification,
                "write_eligible": False,
            }
            t3_end = time.time()

            # ═══════════════════════════════════════════════════
            # Phase 4: Conflict Resolution — 冲突检测与解决
            # ═══════════════════════════════════════════════════
            t4 = time.time()
            conflict_call = execution_post_plan.should_call("conflict_resolver")
            conflict_reason = execution_post_plan.decisions["conflict_resolver"].reason
            v2_conflict_call = False
            if v2_state is not None:
                v2_conflict_call, v2_conflict_reason = self.central_agent_v2.should_run_conflict(v2_state)
                if self.central_agent_v2.active:
                    conflict_call, conflict_reason = v2_conflict_call, v2_conflict_reason
            conflict_before = v2_state.fingerprint() if v2_state is not None else ""
            resolution = (
                self.conflict_resolver.resolve(verified)
                if conflict_call
                else ResolutionResult()
            )
            resolution_payload = resolution.to_dict()
            if not conflict_call:
                resolution_payload["status"] = "SKIPPED_BY_ROUTER"
                resolution_payload["reason"] = conflict_reason
            record["phases"]["conflict_resolution"] = resolution_payload
            if v2_state is not None:
                v2_state.disputed_relations = [
                    item.to_dict() for item in resolution.items
                    if item.decision in {"DISPUTE", "KEEP_OLD", "REVIEW"}
                ]
                conflict_after = v2_state.fingerprint()
                self.central_agent_v2.record_action(
                    v2_state, tool="conflict_resolver",
                    decision=(
                        "CALL" if conflict_call and self.central_agent_v2.active
                        else ("OBSERVED_LEGACY_CALL" if conflict_call else (
                            "WOULD_CALL" if v2_conflict_call else "SKIP"
                        ))
                    ),
                    reason=conflict_reason, before=conflict_before, after=conflict_after,
                    result_status="OK" if conflict_call else "SKIPPED",
                    details={"decision_counts": resolution_payload.get("decisions", {})},
                )
            t4_end = time.time()

            # ═══════════════════════════════════════════════════
            # Phase 5: Decision — 知识决策 (Create/Update/Dispute/...)
            # ═══════════════════════════════════════════════════
            t5 = time.time()
            write_eligible_endpoints = {
                (relation.subject, relation.subject_type)
                for relation in verified.relations
                if relation.import_ready
            } | {
                (relation.object, relation.object_type)
                for relation in verified.relations
                if relation.import_ready
            }
            execution_log = self.decision_engine.decide(
                verified_entities=verified.entities,
                verified_relations=verified.relations,
                pmid=pmid,
                strategy=strategy,
                conflict_resolution=resolution,
                abbr_map=abbr_map,
                restrict_entities_to_import_ready_endpoints=True,
            )
            # Phase 5+: Execute — 执行 Neo4j 写入
            execution_log = self.decision_engine.execute(execution_log)
            execution_payload = execution_log.to_dict()
            execution_payload.update({
                "dry_run": self.config.skip_neo4j_write,
                "entities_proposed": execution_log.entities_created,
                "relations_proposed": execution_log.relations_created,
                "relations_updates_proposed": execution_log.relations_updated,
                "entities_written": 0 if self.config.skip_neo4j_write else execution_log.entities_created,
                "relations_written": 0 if self.config.skip_neo4j_write else execution_log.relations_created,
                "relations_updated_in_neo4j": (
                    0 if self.config.skip_neo4j_write else execution_log.relations_updated
                ),
                "write_eligible_entity_count": len(write_eligible_endpoints),
            })
            record["phases"]["execution"] = execution_payload
            t5_end = time.time()

            if v2_state is not None:
                decision_before = v2_state.fingerprint()
                self.central_agent_v2.observe(
                    v2_state,
                    entities=[item.to_dict() for item in verified.entities],
                    candidate_pairs=[item.to_dict() for item in pair_result.candidates],
                    verification=verified.to_dict(),
                    extraction_error=raw_extraction.error,
                    recovery_candidate_count=0,
                )
                decision_after = v2_state.fingerprint()
                self.central_agent_v2.record_action(
                    v2_state, tool="decision_engine", decision="CALL",
                    reason="deterministic_safe_write_gate",
                    before=decision_before, after=decision_after,
                    result_status="DRY_RUN" if self.config.skip_neo4j_write else "EXECUTED",
                    details={
                        "actions": len(execution_log.actions),
                        "relations_proposed": execution_log.relations_created,
                        "updates_proposed": execution_log.relations_updated,
                        "disputed": execution_log.disputed,
                    },
                )
                self.central_agent_v2.finalize(v2_state)
                record["phases"]["agent_v2"] = v2_state.to_dict()

            # 更新状态 + 记录情景记忆
            self._update_state(raw_extraction, verified, execution_log, pmid=pmid, title=title)

            # Reflection is intentionally deferred to run_batch(), after all workers finish.

            # 时间统计
            record["timing"] = {
                "phase1_context_s": round(t1_end - t1, 2),
                "phase2_extract_s": round(t2_end - t2, 2),
                "phase3_verify_s": round(t3_end - t3, 2),
                "phase4_conflict_s": round(t4_end - t4, 2),
                "phase5_decide_s": round(t5_end - t5, 2),
                "total_s": round(time.time() - t_start, 2),
            }

        except Exception as e:
            record["error"] = str(e)
            self.state.safe_append("errors", {"pmid": pmid, "phase": "unknown", "error": str(e)})

        with self._history_lock:
            self.history.append(record)
        self.state.safe_add(total_articles=1)

        return record

    def _update_state(
        self,
        raw: RawExtraction,
        verified: VerifiedExtraction,
        log: ExecutionLog,
        pmid: str = "",
        title: str = "",
    ):
        """更新 Agent 全局状态统计 + 记录情景记忆（线程安全）"""
        self.state.safe_add(
            total_entities_extracted=len(raw.entities),
            total_relations_extracted=len(raw.relations),
            total_entities_created=log.entities_created,
            total_relations_created=log.relations_created,
            total_relations_updated=log.relations_updated,
            total_disputed=log.disputed,
            total_discarded=log.discarded,
            total_import_ready=verified.summary.get("import_ready", 0),
        )

        # Keep structural, semantic, evidence, and linking signals separate.
        summary = verified.summary
        structural = float(summary.get("structural_score", 0.0))
        semantic = float(summary.get("semantic_score", 0.0))
        evidence = float(summary.get("evidence_score", 0.0))
        self.state.safe_append("structural_scores", structural)
        self.state.safe_append("semantic_scores", semantic)
        self.state.safe_append("evidence_scores", evidence)
        self.state.safe_append("quality_scores", semantic)  # legacy reflection input
        self.state.safe_merge_metrics(summary.get("quality_metrics", {}))

        # ── 记录情景记忆（EpisodicMemory 内部保护） ──
        now = time.time()
        for action in log.actions:
            self.episodic_memory.record(Episode(
                pmid=pmid,
                timestamp=now,
                article_title=title[:120] if title else "",
                action_type=action.type,
                entity_type=action.entity.get("type", "") if action.entity else "",
                entity_name=action.entity.get("mention", "") if action.entity else "",
                predicate=action.relation.get("predicate", "") if action.relation else "",
                object_name=action.relation.get("object", "") if action.relation else "",
                confidence=action.confidence,
                reasoning=action.reason,
            ))

    def _compute_quality_score(self, verified: VerifiedExtraction) -> float:
        """Backward-compatible accessor for the semantic score."""
        return float(verified.summary.get("semantic_score", 0.0))

    def _select_examples(self, strategy: dict) -> list:
        """Select few-shot examples from the active strategy."""
        if (
            strategy.get("use_extended_examples")
            or strategy.get("extraction_mode") == "exploratory"
        ):
            return list(ALL_EXAMPLES)
        return list(self.current_examples)

    def _build_strategy_prompt(
        self,
        base_prompt: str,
        context_card: ContextCard,
        strategy: dict,
        tool_plan: ToolPlan | None = None,
        evidence_units: list | None = None,
    ) -> str:
        """Append runtime extraction guidance derived from Phase 1 context."""
        mode = strategy.get("extraction_mode", "balanced")
        goals = ", ".join(context_card.extraction_goals) or "full_extraction"

        if mode == "exploratory":
            mode_instruction = (
                "The current KG has low coverage for this article. Extract novel "
                "entities broadly when they are explicitly grounded in the text, "
                "but do not force unsupported relation types."
            )
        elif mode == "focused":
            mode_instruction = (
                "The current KG already covers much of this article. Prioritize "
                "relations among known entities and knowledge gaps. Avoid broad "
                "background-only associations."
            )
        else:
            mode_instruction = (
                "Use balanced extraction: capture directly supported entities and "
                "relations while keeping evidence exact."
            )

        prompt = (
            f"{base_prompt}\n\n"
            "Runtime strategy from KG context:\n"
            f"- extraction_mode: {mode}\n"
            f"- extraction_goals: {goals}\n"
            f"- instruction: {mode_instruction}\n"
        )
        if tool_plan is not None:
            prompt += "\n" + self.tool_router.prompt_guidance(tool_plan)
        if evidence_units is not None:
            sections = sorted({getattr(unit, "section", "ABSTRACT") for unit in evidence_units})
            prompt += (
                "\nExtractive reader guidance:\n"
                "- Identify and normalize the entity inventory first.\n"
                "- Then consider relations only between inventory entities grounded in the same sentence or explicit clause.\n"
                "- Resolve article-local abbreviations/coreference before assigning endpoints.\n"
                "- Prefer RESULTS/CONCLUSION evidence; BACKGROUND/OBJECTIVE/METHODS text is not an article finding.\n"
                "- For compound sentences, reason clause-by-clause and do not connect entities across unrelated clauses.\n"
                f"- detected_sections: {', '.join(sections) or 'ABSTRACT'}; evidence_units: {len(evidence_units)}\n"
            )
        return prompt

    def _reflect(self) -> dict:
        """已废弃 — 保留兼容性。元认知反思已迁移至 SelfReflection 模块。

        详见 cognitive_agent/self_reflection.py: SelfReflection.reflect()
        """
        return {"status": "delegated_to_SelfReflection_module"}

    def run_batch(
        self,
        articles: list[dict],
        run_id: str = "agent_v1",
        output_dir: Path = Path("extraction_output"),
    ) -> dict:
        """
        批量处理 PubMed 文章（并发模式）。

        Args:
            articles: PubMed JSONL 记录列表
            run_id: 运行标识
            output_dir: 输出目录

        Returns:
            batch_report: 批量处理报告
        """
        max_workers = self.config.max_workers
        total = len(articles)

        print(f"\n{'='*65}")
        print(f"  🧠 Cognitive Agent — {run_id}")
        print(f"  Articles: {total}")
        print(f"  Workers:  {max_workers} (concurrent)")
        print(f"  Model:   {self.config.model_id} via LangExtract")
        print(f"  Neo4j:   {'CONNECTED' if self.kg_memory.is_connected else 'DISCONNECTED'}")
        print(f"  Neo4j Write: {'ENABLED' if not self.config.skip_neo4j_write else 'DISABLED'}")
        print(f"  Tool Router: {'ENABLED' if self.config.tool_router_enabled else 'LEGACY'}")
        print(f"  Shadow Router: {'AUDIT-ONLY' if self.config.shadow_router_enabled else 'DISABLED'}")
        print(f"  Router Execute: {self.config.router_execution_mode}")
        print(
            f"  Central Agent v2: {self.config.execution_mode} "
            f"({self.config.agent_budget_profile})"
        )
        print(f"  Extraction Cache: {self.config.extraction_cache_mode}")
        print(
            f"  Pair Classifier: {self.config.pair_classifier_mode.upper()} "
            f"({self.config.pair_classifier_backend})"
        )
        print(
            f"  Second LLM: "
            f"{'ENABLED (' + self.config.second_llm_mode + ')' if self.config.second_llm_enabled else 'DISABLED'}"
        )
        print(f"  Neo4j RAG: {'ENABLED (read-only)' if self.config.neo4j_rag_enabled else 'DISABLED'}")
        print(f"{'='*65}\n")

        t_batch_start = time.time()
        completed = [0]  # mutable counter for thread-safe progress
        completed_lock = threading.Lock()

        # ── 并发处理 ──
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_article = {
                executor.submit(self._process_article_with_progress, article, i, total,
                                completed, completed_lock): article
                for i, article in enumerate(articles)
            }

            for future in as_completed(future_to_article):
                try:
                    future.result()  # 异常在此处抛出
                except Exception as e:
                    article = future_to_article[future]
                    pmid = article.get("pmid", "?")
                    print(f"[ERR] PMID:{pmid} — unhandled error: {e}")
                    with self._history_lock:
                        self.state.safe_append("errors", {"pmid": pmid, "phase": "batch", "error": str(e)})

        t_batch_end = time.time()

        # ── 按原始顺序排列 history（可选） ──
        self.history.sort(key=lambda r: r.get("_batch_index", 0))

        # ── 批次边界 Reflection：所有 worker 完成后才改变策略 ──
        reflection = None
        interval = self.config.reflection_interval
        if total and interval > 0 and total >= interval:
            recent = [r for r in self.history if r.get("_batch_index") is not None]
            if recent:
                latest = recent[-1]
                reflection = self.self_reflection.reflect(
                    execution_log=self._execution_log_from_record(latest),
                    context_card=self._context_card_from_record(latest),
                    agent_state=self.state,
                    episodic_memory=self.episodic_memory,
                )
                # Agent v2 treats batch reflection as an offline advisory so a
                # concurrent batch never changes policy midway or silently
                # carries an unvalidated rule into the next experiment.
                if reflection.has_changes() and self.config.execution_mode == "legacy":
                    self.strategy_manager.apply_update(reflection)

        # 生成报告
        report = self._generate_report(run_id, articles, t_batch_end - t_batch_start)
        if reflection is not None:
            report["reflection"] = {
                "strategy_update": reflection.to_dict(),
                "current_strategy": self.strategy_manager.state.to_dict(),
                "applied": bool(
                    reflection.has_changes() and self.config.execution_mode == "legacy"
                ),
                "mode": (
                    "legacy_apply" if self.config.execution_mode == "legacy"
                    else "agent_v2_advisory_only"
                ),
            }

        # 保存结果
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"agent_results_{run_id}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "report": report,
                "config": {
                    "model_id": self.config.model_id,
                    "agent_mode": self.config.agent_mode,
                    "golden_shot_enabled": self.config.golden_shot_enabled,
                    "golden_shot_max_examples": self.config.golden_shot_max_examples,
                    "chunked_extraction_enabled": self.config.chunked_extraction_enabled,
                    "extraction_chunk_max_chars": self.config.extraction_chunk_max_chars,
                    "extraction_chunk_complexity_min_chars": (
                        self.config.extraction_chunk_complexity_min_chars
                    ),
                    "extraction_chunk_max_chunks": self.config.extraction_chunk_max_chunks,
                    "extraction_cache_mode": self.config.extraction_cache_mode,
                    "extraction_cache_memory_entries": self.config.extraction_cache_memory_entries,
                    "extraction_cache_max_entries": self.config.extraction_cache_max_entries,
                    "extraction_cache_max_mb": self.config.extraction_cache_max_mb,
                    "extraction_cache_ttl_days": self.config.extraction_cache_ttl_days,
                    "skip_neo4j_write": self.config.skip_neo4j_write,
                    "reflection_interval": self.config.reflection_interval,
                    "max_workers": max_workers,
                    "extraction_inner_max_workers": self.config.extraction_inner_max_workers,
                    "tool_router_enabled": self.config.tool_router_enabled,
                    "shadow_router_enabled": self.config.shadow_router_enabled,
                    "router_execution_mode": self.config.router_execution_mode,
                    "router_pre_context_max_mentions": self.config.router_pre_context_max_mentions,
                    "execution_mode": self.config.execution_mode,
                    "agent_budget_profile": self.config.agent_budget_profile,
                    "agent_budget_overrides": {
                        "max_actions": self.config.agent_max_actions,
                        "max_aux_remote_calls": self.config.agent_max_aux_remote_calls,
                        "max_neo4j_calls": self.config.agent_max_neo4j_calls,
                        "soft_timeout": self.config.agent_soft_timeout,
                        "hard_timeout": self.config.agent_hard_timeout,
                    },
                    "pair_classifier_enabled": self.config.pair_classifier_enabled,
                    "pair_classifier_mode": self.config.pair_classifier_mode,
                    "pair_classifier_backend": self.config.pair_classifier_backend,
                    "pair_classifier_high_confidence": self.config.pair_classifier_high_confidence,
                    "pair_classifier_relation_threshold": self.config.pair_classifier_relation_threshold,
                    "pair_classifier_uncertainty_floor": self.config.pair_classifier_uncertainty_floor,
                    "pair_classifier_max_candidates": self.config.pair_classifier_max_candidates,
                    "pair_classifier_max_llm_candidates": self.config.pair_classifier_max_llm_candidates,
                    "reviewer_enabled": self.config.reviewer_enabled,
                    "reviewer_model_id": self.config.reviewer_model_id if self.config.reviewer_enabled else "",
                    "second_llm_enabled": self.config.second_llm_enabled,
                    "second_llm_provider": self.config.second_llm_provider,
                    "second_llm_model_id": (
                        self.config.second_llm_model_id if self.config.second_llm_enabled else ""
                    ),
                    "second_llm_mode": self.config.second_llm_mode,
                    "neo4j_rag_enabled": self.config.neo4j_rag_enabled,
                    "rag_limits": {
                        "max_entities": self.config.rag_max_entities,
                        "max_candidates_per_entity": self.config.rag_max_candidates_per_entity,
                        "max_total_candidates": self.config.rag_max_total_candidates,
                        "max_neighbors_per_candidate": self.config.rag_max_neighbors_per_candidate,
                        "max_evidence_chars": self.config.rag_max_evidence_chars,
                        "max_total_chars": self.config.rag_max_total_chars,
                    },
                },
                "records": self.history,
            }, f, indent=2, ensure_ascii=False, default=str)

        report_path = output_dir / f"agent_report_{run_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n[OK] Results → {results_path}")
        print(f"[OK] Report  → {report_path}")

        self._print_report(report)
        return report

    def _process_article_with_progress(
        self,
        article: dict,
        index: int,
        total: int,
        completed: list,
        lock: threading.Lock,
    ):
        """包装 process_article，添加线程安全的进度输出和 history 写入"""
        pmid = article.get("pmid", "?")
        n = index + 1

        record = self.process_article(article)
        record["_batch_index"] = index

        # 线程安全：更新进度（history 已在 process_article 内部写入）
        with lock:
            completed[0] += 1
            done = completed[0]

        # 简要输出
        exec_phase = record.get("phases", {}).get("execution", {})
        verif_phase = record.get("phases", {}).get("verification", {})
        timing = record.get("timing", {})

        entities = len(record.get("phases", {}).get("extraction", {}).get("entities", []))
        rels = len(record.get("phases", {}).get("extraction", {}).get("relations", []))
        ir = verif_phase.get("summary", {}).get("import_ready", 0)
        created = exec_phase.get("entities_created", 0) + exec_phase.get("relations_created", 0)
        disputed = exec_phase.get("disputed", 0)
        discarded = exec_phase.get("discarded", 0)
        err = record.get("error", "")

        status = "❌" if err else "✓"
        print(f"[{done:>3}/{total}] PMID:{pmid} → {entities}E/{rels}R | {ir} IR | "
              f"+{created} ⚡{disputed} ✗{discarded} | {timing.get('total_s', 0):.1f}s {status}")

    @staticmethod
    def _execution_log_from_record(record: dict) -> ExecutionLog:
        """Rebuild the minimal reflection input from a completed record."""
        phase = record.get("phases", {}).get("execution", {})
        return ExecutionLog(
            pmid=record.get("pmid", ""),
            entities_created=phase.get("entities_created", 0),
            relations_created=phase.get("relations_created", 0),
            relations_updated=phase.get("relations_updated", 0),
            disputed=phase.get("disputed", 0),
            discarded=phase.get("discarded", 0),
        )

    @staticmethod
    def _context_card_from_record(record: dict) -> ContextCard:
        """Rebuild reflection context without sharing worker-local objects."""
        context = record.get("phases", {}).get("context", {})
        return ContextCard(
            pmid=record.get("pmid", ""),
            coverage_score=context.get("coverage_score", 0.0),
            extraction_goals=context.get("extraction_goals", []),
        )

    def _generate_report(
        self,
        run_id: str,
        articles: list[dict],
        total_time_s: float,
    ) -> dict:
        """生成批量处理报告"""
        n = len(articles)
        s = self.state
        aggregate_metrics = {}
        for name, metric in s.quality_metric_totals.items():
            denominator = metric["denominator"]
            aggregate_metrics[name] = {
                **metric,
                "value": round(metric["count"] / denominator, 4) if denominator else None,
            }
        proposed_entities = s.total_entities_created
        proposed_relations = s.total_relations_created
        proposed_updates = s.total_relations_updated
        article_reviews = [
            self._build_article_review(record)
            for record in sorted(
                self.history,
                key=lambda item: item.get("_batch_index", len(self.history)),
            )
        ]
        collaboration_phases = [
            record.get("phases", {}).get("collaboration", {})
            for record in self.history
            if record.get("phases", {}).get("collaboration")
        ]
        collaboration_report = {
            "enabled": self.config.second_llm_enabled,
            "mode": self.config.second_llm_mode,
            "triggered_articles": sum(bool(item.get("triggered")) for item in collaboration_phases),
            "successful_articles": sum(item.get("status") == "OK" for item in collaboration_phases),
            "fallback_articles": sum(item.get("status") == "FALLBACK" for item in collaboration_phases),
            "invalid_json_attempts": sum(
                int(item.get("invalid_json_attempts", 0) or 0)
                for item in collaboration_phases
            ),
            "prompt_tokens": sum(
                int(item.get("prompt_tokens", 0) or 0)
                for item in collaboration_phases
            ),
            "output_tokens": sum(
                int(item.get("output_tokens", 0) or 0)
                for item in collaboration_phases
            ),
            "total_latency_s": round(sum(
                float(item.get("latency_s", 0.0) or 0.0)
                for item in collaboration_phases
            ), 3),
            "entity_additions": sum(
                int(item.get("merge", {}).get("entity_additions", 0) or 0)
                for item in collaboration_phases
            ),
            "relation_additions": sum(
                int(item.get("merge", {}).get("relation_additions", 0) or 0)
                for item in collaboration_phases
            ),
            "relation_edits": sum(
                int(item.get("merge", {}).get("relation_edits", 0) or 0)
                for item in collaboration_phases
            ),
            "relation_rejections": sum(
                int(item.get("merge", {}).get("relation_rejections", 0) or 0)
                for item in collaboration_phases
            ),
            "deterministic_rejections": sum(
                int(item.get("merge", {}).get("deterministic_rejection_count", 0) or 0)
                for item in collaboration_phases
            ),
            "manual_review_count": sum(
                len(item.get("merge", {}).get("manual_review", []))
                + len(item.get("merge", {}).get("second_model_rejections", []))
                for item in collaboration_phases
            ),
        }
        rag_phases = [
            record.get("phases", {}).get("rag_context", {})
            for record in self.history
            if record.get("phases", {}).get("rag_context")
        ]
        rag_report = {
            "enabled": self.config.neo4j_rag_enabled,
            "ok_articles": sum(item.get("status") == "OK" for item in rag_phases),
            "offline_articles": sum(item.get("status") == "OFFLINE" for item in rag_phases),
            "error_articles": sum(item.get("status") == "ERROR" for item in rag_phases),
            "candidate_count": sum(int(item.get("candidate_count", 0) or 0) for item in rag_phases),
            "truncated_articles": sum(bool(item.get("truncated")) for item in rag_phases),
            "max_total_chars": self.config.rag_max_total_chars,
            "schema_profile": next((
                item.get("schema_profile", {})
                for item in rag_phases if item.get("schema_profile")
            ), (
                self.kg_memory.get_schema_profile()
                if self.kg_memory.is_connected else {}
            )),
            "entity_lookup_cache": self.kg_memory.entity_lookup_cache_stats(),
        }
        route_phases = [
            record.get("phases", {}).get("tool_plan_post", {})
            for record in self.history
            if record.get("phases", {}).get("tool_plan_post")
        ]
        route_counts: dict[str, int] = {}
        tool_call_counts: dict[str, int] = {}
        for item in route_phases:
            route = item.get("route", "UNKNOWN")
            route_counts[route] = route_counts.get(route, 0) + 1
            for tool in item.get("called_tools", []):
                tool_call_counts[tool] = tool_call_counts.get(tool, 0) + 1
        router_report = {
            "enabled": self.config.tool_router_enabled,
            "route_counts": route_counts,
            "tool_call_counts": tool_call_counts,
            "planned_articles": len(route_phases),
        }
        shadow_pre_phases = [
            record.get("phases", {}).get("shadow_tool_plan_pre", {})
            for record in self.history
            if record.get("phases", {}).get("shadow_tool_plan_pre")
        ]
        shadow_post_phases = [
            record.get("phases", {}).get("shadow_tool_plan_post", {})
            for record in self.history
            if record.get("phases", {}).get("shadow_tool_plan_post")
        ]
        shadow_route_counts: dict[str, int] = {}
        shadow_tool_call_counts: dict[str, int] = {}
        for item in shadow_post_phases:
            route = item.get("route", "UNKNOWN")
            shadow_route_counts[route] = shadow_route_counts.get(route, 0) + 1
            for tool in item.get("called_tools", []) or []:
                shadow_tool_call_counts[tool] = shadow_tool_call_counts.get(tool, 0) + 1
        shadow_router_report = {
            "enabled": self.config.shadow_router_enabled,
            "mode": self.config.router_execution_mode,
            "routing_version": ArticleToolRouter.ROUTING_VERSION,
            "production_execution_unchanged": self.config.router_execution_mode == "legacy",
            "planned_articles": len(shadow_post_phases),
            "pre_route_changes": sum(bool(item.get("route_changed")) for item in shadow_pre_phases),
            "post_route_changes": sum(bool(item.get("route_changed")) for item in shadow_post_phases),
            "route_counts": shadow_route_counts,
            "deep_rate": round(
                shadow_route_counts.get("DEEP", 0) / max(len(shadow_post_phases), 1), 4
            ),
            "tool_call_counts": shadow_tool_call_counts,
            "early_stop_counts": {
                reason: sum(reason in (item.get("early_stop_reasons", []) or []) for item in shadow_post_phases)
                for reason in sorted({
                    reason for item in shadow_post_phases
                    for reason in (item.get("early_stop_reasons", []) or [])
                })
            },
            "four_layer": {
                "safety_mask_counts": {
                    tool: sum(
                        tool in (
                            item.get("four_layer_trace", {})
                            .get("layer_1_safety_mask", {})
                            .get("masked_tools", {})
                        )
                        for item in shadow_post_phases
                    )
                    for tool in sorted({
                        tool for item in shadow_post_phases
                        for tool in (
                            item.get("four_layer_trace", {})
                            .get("layer_1_safety_mask", {})
                            .get("masked_tools", {})
                        )
                    })
                },
                "candidate_pool_counts": {
                    tool: sum(bool(
                        item.get("four_layer_trace", {})
                        .get("layer_2_candidate_pool", {})
                        .get("tools", {}).get(tool)
                    ) for item in shadow_post_phases)
                    for tool in sorted({
                        tool for item in shadow_post_phases
                        for tool in (
                            item.get("four_layer_trace", {})
                            .get("layer_2_candidate_pool", {})
                            .get("tools", {})
                        )
                    })
                },
                "positive_utility_counts": {
                    tool: sum(bool(
                        item.get("four_layer_trace", {})
                        .get("layer_4_net_utility", {})
                        .get("utilities", {}).get(tool, {}).get("positive")
                    ) for item in shadow_post_phases)
                    for tool in sorted({
                        tool for item in shadow_post_phases
                        for tool in (
                            item.get("four_layer_trace", {})
                            .get("layer_4_net_utility", {})
                            .get("utilities", {})
                        )
                    })
                },
            },
        }
        marginal_phases = [
            record.get("phases", {}).get("tool_marginal_benefit", {}).get("second_llm_refiner", {})
            for record in self.history
            if record.get("phases", {}).get("tool_marginal_benefit")
        ]
        tool_value_report = {
            "second_llm_refiner": {
                "called_articles": sum(bool(item.get("called")) for item in marginal_phases),
                "state_changes": sum(
                    int(item.get("relation_additions", 0) or 0)
                    + int(item.get("relation_edits", 0) or 0)
                    + int(item.get("relation_rejections", 0) or 0)
                    for item in marginal_phases
                ),
                "zero_change_calls": sum(
                    bool(item.get("called")) and not (
                        int(item.get("relation_additions", 0) or 0)
                        + int(item.get("relation_edits", 0) or 0)
                        + int(item.get("relation_rejections", 0) or 0)
                    ) for item in marginal_phases
                ),
                "latency_seconds": round(sum(
                    float(item.get("latency_seconds", 0.0) or 0.0) for item in marginal_phases
                ), 3),
                "prompt_tokens": sum(int(item.get("prompt_tokens", 0) or 0) for item in marginal_phases),
                "output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in marginal_phases),
            }
        }
        reader_phases = [
            record.get("phases", {}).get("reader", {})
            for record in self.history
            if record.get("phases", {}).get("reader")
        ]
        strategy_phases = [
            record.get("phases", {}).get("strategy", {})
            for record in self.history
            if record.get("phases", {}).get("strategy")
        ]
        extraction_planning_report = {
            "golden_shot_enabled": self.config.golden_shot_enabled,
            "golden_shot_examples_per_article": self.config.golden_shot_max_examples,
            "chunked_extraction_enabled": self.config.chunked_extraction_enabled,
            "chunked_articles": sum(
                bool(item.get("chunking", {}).get("used")) for item in reader_phases
            ),
            "total_extraction_windows": sum(
                int(item.get("chunking", {}).get("chunk_count", 1) or 1)
                for item in reader_phases
            ),
            "example_usage": {
                name: sum(name in (item.get("example_names", []) or []) for item in strategy_phases)
                for name in sorted({
                    name
                    for item in strategy_phases
                    for name in (item.get("example_names", []) or [])
                })
            },
        }
        controller_phases = [
            record.get("phases", {}).get("agent_controller", {})
            for record in self.history
            if record.get("phases", {}).get("agent_controller")
        ]
        controller_action_counts: dict[str, int] = {}
        for phase in controller_phases:
            for action in phase.get("actions", []) or []:
                key = f"{action.get('tool', 'unknown')}:{action.get('decision', 'unknown')}"
                controller_action_counts[key] = controller_action_counts.get(key, 0) + 1
        agent_controller_report = {
            "policy": "observe_act_verify_stop",
            "mode": self.config.agent_mode,
            "max_rounds": 2,
            "planned_articles": len(controller_phases),
            "second_round_articles": sum(
                int(item.get("current_round", 1) or 1) > 1 for item in controller_phases
            ),
            "evidence_repairs": sum(
                len(item.get("evidence_repairs", []) or []) for item in controller_phases
            ),
            "recovery_candidates": sum(
                int(item.get("recovery_candidate_count", 0) or 0)
                for item in controller_phases
            ),
            "recovered_relations": collaboration_report["relation_additions"],
            "post_action_rollbacks": sum(
                int(item.get("post_action_finalization", {}).get("rolled_back_count", 0) or 0)
                for item in collaboration_phases
            ),
            "canonical_duplicates_removed": sum(
                int(item.get("post_action_finalization", {}).get(
                    "duplicate_relations_removed", 0
                ) or 0)
                for item in collaboration_phases
            ),
            "symmetric_orientation_changes": sum(
                int(item.get("post_action_finalization", {}).get(
                    "symmetric_orientation_changes", 0
                ) or 0)
                for item in collaboration_phases
            ),
            "shadow_relations": sum(
                len(item.get("recovery_partition", {}).get("relations", []) or [])
                for item in collaboration_phases
                if not item.get("recovery_partition", {}).get("write_eligible", False)
            ),
            "action_counts": controller_action_counts,
        }
        pair_phases = [
            record.get("phases", {}).get("relation_pair_classification", {})
            for record in self.history
            if record.get("phases", {}).get("relation_pair_classification")
        ]
        pair_classifier_report = {
            "enabled": self.config.pair_classifier_enabled,
            "mode": self.config.pair_classifier_mode,
            "backend": self.config.pair_classifier_backend,
            "production_execution_unchanged": self.config.pair_classifier_mode != "active",
            "articles": len(pair_phases),
            "candidate_count": sum(int(item.get("candidate_count", 0) or 0) for item in pair_phases),
            "positive_prediction_count": sum(
                int(item.get("positive_prediction_count", 0) or 0) for item in pair_phases
            ),
            "accepted_relation_count": sum(
                int(item.get("accepted_relation_count", 0) or 0) for item in pair_phases
            ),
            "no_relation_count": sum(
                int(item.get("no_relation_count", 0) or 0) for item in pair_phases
            ),
            "low_confidence_count": sum(
                int(item.get("low_confidence_count", 0) or 0) for item in pair_phases
            ),
            "truncated_candidates": sum(
                int(item.get("truncated_candidates", 0) or 0) for item in pair_phases
            ),
        }
        pair_classifier_report["deepseek_routing_rate"] = round(
            pair_classifier_report["low_confidence_count"]
            / max(pair_classifier_report["candidate_count"], 1), 4
        )
        v2_phases = [
            record.get("phases", {}).get("agent_v2", {})
            for record in self.history
            if record.get("phases", {}).get("agent_v2")
        ]
        v2_action_counts: dict[str, int] = {}
        v2_remote_totals: dict[str, dict[str, float | int]] = {}
        v2_cache_totals: dict[str, int] = {}
        v2_primary_remote_requests = 0
        v2_primary_extraction_windows = 0
        for phase in v2_phases:
            for action in phase.get("action_trace", []) or []:
                key = f"{action.get('tool', 'unknown')}:{action.get('decision', 'unknown')}"
                v2_action_counts[key] = v2_action_counts.get(key, 0) + 1
                if action.get("tool") == "langextract_candidate_generator":
                    details = action.get("details", {}) or {}
                    v2_primary_remote_requests += int(
                        details.get("primary_remote_requests", 0) or 0
                    )
                    v2_primary_extraction_windows += int(details.get("chunk_count", 1) or 1)
            for tool, usage in (phase.get("remote_usage", {}) or {}).items():
                target = v2_remote_totals.setdefault(tool, {
                    "attempted": 0, "successful": 0, "retried": 0,
                    "cached": 0, "timeouts": 0, "prompt_tokens": 0,
                    "output_tokens": 0, "latency_s": 0.0,
                    "state_changes": 0, "zero_change_calls": 0,
                })
                for key, value in usage.items():
                    target[key] = target.get(key, 0) + value
            for key, value in (phase.get("cache", {}) or {}).items():
                v2_cache_totals[key] = v2_cache_totals.get(key, 0) + int(value or 0)
        remote_attempts = sum(int(item.get("attempted", 0)) for item in v2_remote_totals.values())
        zero_change = sum(int(item.get("zero_change_calls", 0)) for item in v2_remote_totals.values())
        agent_v2_report = {
            "execution_mode": self.config.execution_mode,
            "budget_profile": self.config.agent_budget_profile,
            "planned_articles": len(v2_phases),
            "route_counts": {
                route: sum(item.get("route") == route for item in v2_phases)
                for route in ("FAST", "STANDARD", "DEEP")
            },
            "budget_escalations": sum(
                len(item.get("budget_escalations", []) or []) for item in v2_phases
            ),
            "action_counts": v2_action_counts,
            "remote_usage": v2_remote_totals,
            "primary_extraction": {
                "windows": v2_primary_extraction_windows,
                "remote_requests": v2_primary_remote_requests,
                "cache_avoided_requests": max(
                    0, v2_primary_extraction_windows - v2_primary_remote_requests
                ),
            },
            "cache": v2_cache_totals,
            "zero_change_call_rate": round(zero_change / max(remote_attempts, 1), 4),
            "termination_counts": {
                status: sum(
                    item.get("termination", {}).get("status") == status for item in v2_phases
                )
                for status in ("accepted", "human_review", "running")
            },
        }
        article_times = sorted(
            float(record.get("timing", {}).get("total_s", 0.0) or 0.0)
            for record in self.history if record.get("timing")
        )

        def nearest_percentile(values: list[float], fraction: float) -> float:
            if not values:
                return 0.0
            index = max(0, min(len(values) - 1, int((len(values) - 1) * fraction + 0.5)))
            return round(values[index], 3)

        latency_report = {
            "article_p50_s": nearest_percentile(article_times, 0.50),
            "article_p95_s": nearest_percentile(article_times, 0.95),
            "article_p99_s": nearest_percentile(article_times, 0.99),
            "article_max_s": round(max(article_times), 3) if article_times else 0.0,
        }
        return {
            "run_id": run_id,
            "agent_mode": self.config.agent_mode,
            "model": self.config.model_id,
            "neo4j_connected": self.kg_memory.is_connected,
            "dry_run": self.config.skip_neo4j_write,
            "total_articles": n,
            "total_time_s": round(total_time_s, 1),
            "avg_time_per_article_s": round(total_time_s / max(n, 1), 1),
            "extraction": {
                "total_entities_extracted": s.total_entities_extracted,
                "total_relations_extracted": s.total_relations_extracted,
                "avg_entities_per_article": round(s.total_entities_extracted / max(n, 1), 1),
                "avg_relations_per_article": round(s.total_relations_extracted / max(n, 1), 1),
            },
            "decisions": {
                "total_import_ready": s.total_import_ready,
                "entities_proposed": proposed_entities,
                "relations_proposed": proposed_relations,
                "relation_updates_proposed": proposed_updates,
                "entities_created": 0 if self.config.skip_neo4j_write else proposed_entities,
                "relations_created": 0 if self.config.skip_neo4j_write else proposed_relations,
                "relations_updated": 0 if self.config.skip_neo4j_write else proposed_updates,
                "disputed": s.total_disputed,
                "discarded": s.total_discarded,
                "import_ready_rate": round(s.total_import_ready / max(s.total_relations_extracted, 1), 3),
                "discard_rate": round(
                    s.total_discarded
                    / max(s.total_entities_extracted + s.total_relations_extracted, 1),
                    3,
                ),
            },
            "quality": {
                "avg_structural_score": round(
                    sum(s.structural_scores) / max(len(s.structural_scores), 1), 3
                ),
                "avg_semantic_score": round(
                    sum(s.semantic_scores) / max(len(s.semantic_scores), 1), 3
                ),
                "avg_evidence_score": round(
                    sum(s.evidence_scores) / max(len(s.evidence_scores), 1), 3
                ),
                "metrics": aggregate_metrics,
                "error_count": len(s.errors),
                "errors": s.errors[:10],  # 仅展示前10个错误
            },
            "collaboration": collaboration_report,
            "neo4j_rag": rag_report,
            "tool_router": router_report,
            "shadow_tool_router": shadow_router_report,
            "tool_marginal_benefit": tool_value_report,
            "preprocessing_cache": self.article_preprocessor.cache.stats(),
            "extraction_cache": self.extraction_cache.stats(),
            "extraction_planning": extraction_planning_report,
            "agent_controller": agent_controller_report,
            "relation_pair_classifier": pair_classifier_report,
            "agent_v2": agent_v2_report,
            "latency": latency_report,
            "article_reviews": article_reviews,
        }

    @staticmethod
    def _build_article_review(record: dict) -> dict:
        """Build the fixed human-review view required by Phase A."""
        phases = record.get("phases", {})
        verification = phases.get("verification", {})
        review = verification.get("review", {})
        summary = verification.get("summary", {})
        filtered = {
            (item.get("mention", ""), item.get("type", item.get("entity_type", ""))): item
            for item in review.get("filtered_entities", [])
        }
        canonical_entities = verification.get("entities", [])
        canonical_by_alias = {}
        for item in canonical_entities:
            entity_type = item.get("type", "")
            for alias in item.get("canonical_mentions", []) or [item.get("mention", "")]:
                canonical_by_alias[(alias, entity_type)] = item

        entity_reviews = []
        for raw in review.get("raw_entities", phases.get("extraction", {}).get("entities", [])):
            mention = raw.get("mention", "")
            entity_type = raw.get("type", raw.get("entity_type", ""))
            rejected = filtered.get((mention, entity_type))
            canonical = canonical_by_alias.get((mention, entity_type), {})
            entity_reviews.append({
                "mention": mention,
                "entity_type": entity_type,
                "source_span": (rejected or canonical or raw).get("source_span", ""),
                "char_start": (rejected or canonical or raw).get("char_start", -1),
                "char_end": (rejected or canonical or raw).get("char_end", -1),
                "grounded": bool((rejected or canonical or raw).get("grounded", False)),
                "filter_status": "rejected" if rejected else "retained",
                "filter_reason": (
                    rejected.get("filter_reason", "")
                    if rejected
                    else canonical.get("filter_reason", "type_constraints_passed")
                ),
                "canonical_entity": canonical.get("mention", "") if not rejected else "",
                "canonical_key": canonical.get("canonical_key", "") if not rejected else "",
            })

        relation_reviews = [
            {
                key: relation.get(key)
                for key in (
                    "subject", "subject_type", "predicate", "object", "object_type",
                    "direction", "evidence", "evidence_char_start", "evidence_char_end",
                    "evidence_level", "quality_flags", "schema_valid", "import_ready",
                )
            }
            for relation in verification.get("relations", [])
        ]
        return {
            "pmid": record.get("pmid", ""),
            "title": record.get("title", ""),
            "tool_plan_pre": phases.get("tool_plan_pre", {}),
            "tool_plan_post": phases.get("tool_plan_post", {}),
            "agent_controller": phases.get("agent_controller", {}),
            "collaboration": phases.get("collaboration", {}),
            "rag_context": phases.get("rag_context", {}),
            "entities": entity_reviews,
            "relations": relation_reviews,
            "structural_score": summary.get("structural_score", 0.0),
            "semantic_score": summary.get("semantic_score", 0.0),
            "evidence_score": summary.get("evidence_score", 0.0),
            "quality_metrics": summary.get("quality_metrics", {}),
            "linking_stats": summary.get("linking_stats", {}),
            "error": record.get("error", ""),
        }

    def _print_report(self, report: dict):
        """打印人可读报告"""
        ext = report["extraction"]
        dec = report["decisions"]
        q = report["quality"]

        print(f"\n{'='*65}")
        print(f"  📊 Agent 运行报告 — {report['run_id']}")
        print(f"{'='*65}")
        print(f"  处理文章:        {report['total_articles']}")
        print(f"  总耗时:          {report['total_time_s']:.0f}s ({report['avg_time_per_article_s']:.0f}s/article)")
        print(f"")
        print(f"  ── 提取 ──")
        print(f"  实体总数:        {ext['total_entities_extracted']} ({ext['avg_entities_per_article']}/article)")
        print(f"  关系总数:        {ext['total_relations_extracted']} ({ext['avg_relations_per_article']}/article)")
        print(f"  Import-Ready:    {dec['total_import_ready']} ({dec['import_ready_rate']:.1%})")
        print(f"")
        print(f"  ── 决策 ──")
        print(f"  Dry-run:         {'是（未写 Neo4j）' if report.get('dry_run') else '否'}")
        print(f"  拟创建实体:      {dec['entities_proposed']}")
        print(f"  拟创建关系:      {dec['relations_proposed']}")
        print(f"  实际创建实体:    {dec['entities_created']}")
        print(f"  实际创建关系:    {dec['relations_created']}")
        print(f"  实际更新关系:    {dec['relations_updated']}")
        print(f"  标记争议:        {dec['disputed']}")
        print(f"  丢弃:            {dec['discarded']} ({dec['discard_rate']:.1%})")
        print(f"")
        print(f"  ── 质量 ──")
        print(f"  结构合规分:      {q['avg_structural_score']:.2f}")
        print(f"  语义质量分:      {q['avg_semantic_score']:.2f}")
        print(f"  Evidence 分:     {q['avg_evidence_score']:.2f}")
        print(f"  错误数:          {q['error_count']}")
        print(f"  Neo4j 连接:      {'✅' if report['neo4j_connected'] else '❌ (offline mode)'}")
        collaboration = report.get("collaboration", {})
        print(
            f"  第二模型:        "
            f"{'开启' if collaboration.get('enabled') else '关闭'} | "
            f"触发 {collaboration.get('triggered_articles', 0)}, "
            f"成功 {collaboration.get('successful_articles', 0)}, "
            f"降级 {collaboration.get('fallback_articles', 0)}"
        )
        rag = report.get("neo4j_rag", {})
        print(
            f"  Neo4j RAG:       {'开启' if rag.get('enabled') else '关闭'} | "
            f"候选 {rag.get('candidate_count', 0)}, "
            f"离线 {rag.get('offline_articles', 0)}, "
            f"错误 {rag.get('error_articles', 0)}"
        )
        router = report.get("tool_router", {})
        print(
            f"  Tool Router:      {'开启' if router.get('enabled') else '兼容模式'} | "
            f"路径 {router.get('route_counts', {})}"
        )
        controller = report.get("agent_controller", {})
        print(
            f"  Agent 闭环:       修复 {controller.get('evidence_repairs', 0)}, "
            f"恢复 {controller.get('recovered_relations', 0)}, "
            f"回滚 {controller.get('post_action_rollbacks', 0)}, "
            f"去重 {controller.get('canonical_duplicates_removed', 0)}"
        )
        if report.get("article_reviews"):
            print(f"")
            print(f"  ── 逐篇人工审查索引 ──")
            for article in report["article_reviews"]:
                ready = sum(1 for rel in article["relations"] if rel.get("import_ready"))
                print(
                    f"  PMID {article['pmid']}: {len(article['entities'])} raw entities, "
                    f"{len(article['relations'])} relations, {ready} import-ready | "
                    f"S={article['structural_score']:.2f} "
                    f"M={article['semantic_score']:.2f} E={article['evidence_score']:.2f}"
                )
        print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🧠 Cognitive Agent — Autonomous KG Knowledge Manager"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="PubMed JSONL 输入文件",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=50,
        help="处理文章数 (默认 50)",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="运行标识 (默认自动生成)",
    )
    parser.add_argument(
        "--output-dir", default="extraction_output",
        help="输出目录",
    )
    parser.add_argument(
        "--skip-neo4j-write", action="store_true", default=True,
        help="跳过 Neo4j 写入 (默认开启)",
    )
    parser.add_argument(
        "--write-neo4j", action="store_true",
        help="执行 Neo4j 写入",
    )
    parser.add_argument(
        "--api-key", default="",
        help="Gemini API Key (默认从 GEMINI_API_KEY 环境变量读取)",
    )
    parser.add_argument(
        "--neo4j-password", default="",
        help="Neo4j 密码 (默认从 NEO4J_PASSWORD 环境变量读取)",
    )
    parser.add_argument(
        "--neo4j-uri", default="",
        help="Neo4j Bolt URI (默认从 NEO4J_URI 环境变量读取)",
    )
    parser.add_argument(
        "--neo4j-user", default="",
        help="Neo4j 用户 (默认从 NEO4J_USER 环境变量读取)",
    )
    parser.add_argument(
        "--neo4j-database", default="",
        help="Neo4j 数据库 (默认从 NEO4J_DATABASE 环境变量读取)",
    )
    parser.add_argument(
        "--model-id", default="[按次]gemini-2.5-flash",
        help="模型 ID",
    )
    parser.add_argument(
        "--api-base", default="",
        help="API Base URL (覆盖默认值)",
    )
    parser.add_argument(
        "--reflection-interval", type=int, default=10,
        help="反思间隔 (每 N 篇执行一次策略反思)",
    )
    parser.add_argument(
        "--agent-mode", choices=("precision", "shadow-agent", "recall"),
        default=os.environ.get("AGENT_MODE", "precision"),
        help="Agent 行动模式；默认 precision",
    )
    parser.add_argument(
        "--disable-golden-shot", action="store_true",
        help="关闭动态 3+1 golden-shot，回退到旧固定示例",
    )
    parser.add_argument(
        "--golden-shot-max-examples", type=int,
        default=int(os.environ.get("GOLDEN_SHOT_MAX_EXAMPLES", "4")),
        help="每篇使用 3–4 个动态示范（默认 4）",
    )
    parser.add_argument(
        "--disable-chunked-extraction", action="store_true",
        help="关闭长摘要的章节/证据单元分块抽取",
    )
    parser.add_argument(
        "--extraction-chunk-max-chars", type=int,
        default=int(os.environ.get("EXTRACTION_CHUNK_MAX_CHARS", "1800")),
    )
    parser.add_argument(
        "--extraction-chunk-complexity-min-chars", type=int,
        default=int(os.environ.get("EXTRACTION_CHUNK_COMPLEXITY_MIN_CHARS", "1400")),
    )
    parser.add_argument(
        "--extraction-chunk-max-chunks", type=int,
        default=int(os.environ.get("EXTRACTION_CHUNK_MAX_CHUNKS", "3")),
    )
    parser.add_argument(
        "--extraction-cache-mode", choices=("off", "memory", "persistent"),
        default=os.environ.get("EXTRACTION_CACHE_MODE", "memory"),
        help="主抽取缓存：memory默认轻量且不落盘；persistent仅用于重复实验",
    )
    parser.add_argument(
        "--extraction-cache-path",
        default=os.environ.get("EXTRACTION_CACHE_PATH", ".cache/langextract_candidates.sqlite3"),
    )
    parser.add_argument(
        "--extraction-cache-memory-entries", type=int,
        default=int(os.environ.get("EXTRACTION_CACHE_MEMORY_ENTRIES", "256")),
    )
    parser.add_argument(
        "--extraction-cache-max-entries", type=int,
        default=int(os.environ.get("EXTRACTION_CACHE_MAX_ENTRIES", "2000")),
    )
    parser.add_argument(
        "--extraction-cache-max-mb", type=int,
        default=int(os.environ.get("EXTRACTION_CACHE_MAX_MB", "200")),
    )
    parser.add_argument(
        "--extraction-cache-ttl-days", type=int,
        default=int(os.environ.get("EXTRACTION_CACHE_TTL_DAYS", "30")),
    )
    parser.add_argument(
        "--disable-tool-router", action="store_true",
        help="关闭文章级工具路由，恢复每篇固定调用链",
    )
    parser.add_argument(
        "--disable-shadow-router", action="store_true",
        help="关闭多维复杂度影子路由审计；默认开启且不影响生产调用",
    )
    parser.add_argument(
        "--router-execution-mode", choices=("legacy", "active_shadow"),
        default=os.environ.get("ROUTER_EXECUTION_MODE", "legacy"),
        help="legacy保持生产旧路由；active_shadow仅允许dry-run激活实验",
    )
    parser.add_argument(
        "--router-pre-context-max-mentions", type=int, default=8,
        help="路由器预抽取记忆查询的实体上限（默认 8）",
    )
    parser.add_argument(
        "--execution-mode", choices=("legacy", "agent-v2-shadow", "agent-v2"),
        default=os.environ.get("AGENT_EXECUTION_MODE", "legacy"),
        help="中央 Agent v2 执行模式；legacy 保持原行为",
    )
    parser.add_argument(
        "--agent-budget-profile", choices=("quality", "balanced", "speed"),
        default=os.environ.get("AGENT_BUDGET_PROFILE", "quality"),
    )
    parser.add_argument(
        "--agent-max-actions", type=int,
        default=int(os.environ.get("AGENT_MAX_ACTIONS", "0")),
        help="Agent v2 动作覆盖值；0 使用路由预算",
    )
    parser.add_argument(
        "--agent-max-aux-remote-calls", type=int,
        default=int(os.environ.get("AGENT_MAX_AUX_REMOTE_CALLS", "0")),
        help="辅助 LLM 请求覆盖值；0 使用路由预算",
    )
    parser.add_argument(
        "--agent-max-neo4j-calls", type=int,
        default=int(os.environ.get("AGENT_MAX_NEO4J_CALLS", "0")),
        help="Neo4j 查询覆盖值；0 使用路由预算",
    )
    parser.add_argument(
        "--agent-soft-timeout", type=float,
        default=float(os.environ.get("AGENT_SOFT_TIMEOUT", "0")),
        help="远程动作软截止秒数；0 使用路由预算",
    )
    parser.add_argument(
        "--agent-hard-timeout", type=float,
        default=float(os.environ.get("AGENT_HARD_TIMEOUT", "180")),
        help="单篇 Agent v2 硬截止秒数（默认 180）",
    )
    parser.add_argument(
        "--pair-classifier-mode", choices=("off", "shadow", "active"),
        default=os.environ.get("PAIR_CLASSIFIER_MODE", "shadow"),
        help="BioRED式实体对分类：shadow默认只审计；active仅允许dry-run",
    )
    parser.add_argument(
        "--pair-classifier-backend", choices=("deterministic", "sklearn"),
        default=os.environ.get("PAIR_CLASSIFIER_BACKEND", "deterministic"),
    )
    parser.add_argument(
        "--pair-classifier-model-path",
        default=os.environ.get("PAIR_CLASSIFIER_MODEL_PATH", ""),
    )
    parser.add_argument(
        "--pair-classifier-high-confidence", type=float,
        default=float(os.environ.get("PAIR_CLASSIFIER_HIGH_CONFIDENCE", "0.78")),
        help="高于该值直接接受，低置信候选才交给DeepSeek",
    )
    parser.add_argument(
        "--pair-classifier-relation-threshold", type=float,
        default=float(os.environ.get("PAIR_CLASSIFIER_RELATION_THRESHOLD", "0.48")),
    )
    parser.add_argument(
        "--pair-classifier-uncertainty-floor", type=float,
        default=float(os.environ.get("PAIR_CLASSIFIER_UNCERTAINTY_FLOOR", "0.35")),
    )
    parser.add_argument(
        "--pair-classifier-max-candidates", type=int,
        default=int(os.environ.get("PAIR_CLASSIFIER_MAX_CANDIDATES", "64")),
    )
    parser.add_argument(
        "--pair-classifier-max-llm-candidates", type=int,
        default=int(os.environ.get("PAIR_CLASSIFIER_MAX_LLM_CANDIDATES", "12")),
    )
    parser.add_argument(
        "--reviewer-enabled", action="store_true",
        help="调试期开启 LLM 抽取质量审稿；默认关闭，不属于最终生产链路",
    )
    parser.add_argument(
        "--reviewer-model-id", default="",
        help="审稿模型 ID；仅在 --reviewer-enabled 时使用",
    )
    parser.add_argument(
        "--reviewer-api-base", default="",
        help="审稿模型 API Base；为空时复用抽取模型 API Base",
    )
    parser.add_argument(
        "--second-llm-enabled", action="store_true",
        help="开启 Phase-B 条件式第二模型；默认关闭",
    )
    parser.add_argument(
        "--second-llm-provider", choices=("openai", "gemini"), default="openai",
        help="第二模型协议；DeepSeek 使用 openai（默认）",
    )
    parser.add_argument(
        "--second-llm-model-id", default="deepseek-v4-flash",
        help="第二模型 ID（也可用 SECOND_LLM_MODEL_ID）",
    )
    parser.add_argument(
        "--second-llm-api-base", default="https://api.deepseek.com",
        help="第二模型 API Base（也可用 SECOND_LLM_API_BASE）",
    )
    parser.add_argument(
        "--second-llm-mode", choices=("conditional", "always"), default="conditional",
        help="第二模型触发模式；默认 conditional",
    )
    parser.add_argument(
        "--second-llm-timeout", type=float, default=45.0,
        help="第二模型请求超时秒数",
    )
    parser.add_argument(
        "--neo4j-rag-enabled", action="store_true",
        help="开启 Phase-C 受限只读 Neo4j RAG；默认关闭",
    )
    parser.add_argument("--rag-max-entities", type=int, default=12)
    parser.add_argument("--rag-max-candidates-per-entity", type=int, default=3)
    parser.add_argument("--rag-max-total-candidates", type=int, default=20)
    parser.add_argument("--rag-max-neighbors-per-candidate", type=int, default=4)
    parser.add_argument("--rag-max-evidence-chars", type=int, default=500)
    parser.add_argument("--rag-max-total-chars", type=int, default=6000)
    parser.add_argument(
        "--max-workers", type=int, default=5,
        help="并发 worker 数 (默认 5，设 1 为串行)",
    )
    parser.add_argument(
        "--extraction-inner-max-workers", type=int,
        default=int(os.environ.get("EXTRACTION_INNER_MAX_WORKERS", "2")),
        help="LangExtract 内层并发数（稳定性矩阵应设为 1）",
    )
    args = parser.parse_args()

    # ── 环境变量检查（GEMINI_API_KEY → DEEPSEEK_API_KEY 级联回退） ──
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    neo4j_password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD", "")
    neo4j_uri = args.neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = args.neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
    neo4j_database = args.neo4j_database or os.environ.get("NEO4J_DATABASE", "neo4j")
    second_llm_enabled = args.second_llm_enabled or os.environ.get(
        "SECOND_LLM_ENABLED", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    second_llm_provider = os.environ.get(
        "SECOND_LLM_PROVIDER", args.second_llm_provider
    ).strip().lower()
    second_llm_api_key = (
        os.environ.get("SECOND_LLM_API_KEY", "")
        or (os.environ.get("DEEPSEEK_API_KEY", "") if second_llm_provider == "openai" else "")
    )
    second_llm_model_id = os.environ.get(
        "SECOND_LLM_MODEL_ID", args.second_llm_model_id
    )
    second_llm_api_base = os.environ.get(
        "SECOND_LLM_API_BASE", args.second_llm_api_base
    )
    second_llm_mode = os.environ.get("SECOND_LLM_MODE", args.second_llm_mode).strip().lower()
    try:
        second_llm_timeout = float(
            os.environ.get("SECOND_LLM_TIMEOUT", str(args.second_llm_timeout))
        )
    except ValueError:
        print("[ERROR] SECOND_LLM_TIMEOUT must be numeric")
        sys.exit(1)
    neo4j_rag_enabled = args.neo4j_rag_enabled or os.environ.get(
        "NEO4J_RAG_ENABLED", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    try:
        rag_limits = {
            "rag_max_entities": int(os.environ.get("RAG_MAX_ENTITIES", args.rag_max_entities)),
            "rag_max_candidates_per_entity": int(os.environ.get(
                "RAG_MAX_CANDIDATES_PER_ENTITY", args.rag_max_candidates_per_entity
            )),
            "rag_max_total_candidates": int(os.environ.get(
                "RAG_MAX_TOTAL_CANDIDATES", args.rag_max_total_candidates
            )),
            "rag_max_neighbors_per_candidate": int(os.environ.get(
                "RAG_MAX_NEIGHBORS_PER_CANDIDATE", args.rag_max_neighbors_per_candidate
            )),
            "rag_max_evidence_chars": int(os.environ.get(
                "RAG_MAX_EVIDENCE_CHARS", args.rag_max_evidence_chars
            )),
            "rag_max_total_chars": int(os.environ.get(
                "RAG_MAX_TOTAL_CHARS", args.rag_max_total_chars
            )),
        }
    except ValueError:
        print("[ERROR] RAG limit environment variables must be integers")
        sys.exit(1)

    if not api_key:
        print("[ERROR] No API key found. Set one of:")
        print("  export GEMINI_API_KEY='...'")
        print("  export DEEPSEEK_API_KEY='...'")
        print("  Or pass: --api-key '...'")
        sys.exit(1)

    if second_llm_enabled and (not second_llm_api_key or not second_llm_model_id):
        print("[ERROR] Phase-B collaboration requires SECOND_LLM_API_KEY and SECOND_LLM_MODEL_ID")
        sys.exit(1)
    if second_llm_mode not in {"conditional", "always"}:
        print("[ERROR] SECOND_LLM_MODE must be 'conditional' or 'always'")
        sys.exit(1)
    if second_llm_provider not in {"openai", "gemini"}:
        print("[ERROR] SECOND_LLM_PROVIDER must be 'openai' or 'gemini'")
        sys.exit(1)
    if any(value <= 0 for value in rag_limits.values()):
        print("[ERROR] RAG limits must be positive integers")
        sys.exit(1)
    if args.router_pre_context_max_mentions <= 0:
        print("[ERROR] --router-pre-context-max-mentions must be positive")
        sys.exit(1)
    if min(
        args.agent_max_actions, args.agent_max_aux_remote_calls,
        args.agent_max_neo4j_calls, args.agent_soft_timeout,
    ) < 0 or args.agent_hard_timeout <= 0:
        print("[ERROR] Agent v2 budget overrides must be non-negative and hard timeout positive")
        sys.exit(1)
    if not all(0 < value < 1 for value in (
        args.pair_classifier_high_confidence,
        args.pair_classifier_relation_threshold,
        args.pair_classifier_uncertainty_floor,
    )):
        print("[ERROR] pair-classifier probability thresholds must be between 0 and 1")
        sys.exit(1)
    if min(args.pair_classifier_max_candidates, args.pair_classifier_max_llm_candidates) <= 0:
        print("[ERROR] pair-classifier candidate limits must be positive")
        sys.exit(1)
    if args.pair_classifier_backend == "sklearn" and not args.pair_classifier_model_path:
        print("[ERROR] sklearn pair classifier requires --pair-classifier-model-path")
        sys.exit(1)
    if not 3 <= args.golden_shot_max_examples <= 4:
        print("[ERROR] --golden-shot-max-examples must be 3 or 4")
        sys.exit(1)
    if args.extraction_chunk_max_chars < 800:
        print("[ERROR] --extraction-chunk-max-chars must be >= 800")
        sys.exit(1)
    if not 2 <= args.extraction_chunk_max_chunks <= 4:
        print("[ERROR] --extraction-chunk-max-chunks must be between 2 and 4")
        sys.exit(1)
    if min(
        args.extraction_cache_memory_entries, args.extraction_cache_max_entries,
        args.extraction_cache_max_mb, args.extraction_cache_ttl_days,
    ) <= 0:
        print("[ERROR] extraction cache limits must be positive")
        sys.exit(1)
    if args.extraction_inner_max_workers <= 0:
        print("[ERROR] --extraction-inner-max-workers must be positive")
        sys.exit(1)

    if not neo4j_password:
        print("[WARN]  NEO4J_PASSWORD not set — Neo4j will be DISABLED (offline verification mode)")
        print("  Set via: export NEO4J_PASSWORD='...'")
        print("  Or pass: --neo4j-password '...'")

    if args.write_neo4j:
        args.skip_neo4j_write = False
        if not neo4j_password:
            print("[ERROR] --write-neo4j requires NEO4J_PASSWORD")
            sys.exit(1)
        write_host = urlparse(neo4j_uri).hostname
        if write_host not in {"localhost", "127.0.0.1", "::1"}:
            print("[ERROR] --write-neo4j is restricted to localhost")
            sys.exit(1)
        if neo4j_database != "neo4j":
            print("[ERROR] --write-neo4j is restricted to database 'neo4j'")
            sys.exit(1)

    run_id = args.run_id or f"agent_{args.limit}_{time.strftime('%Y%m%d_%H%M%S')}"

    # ── 加载输入 ──
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    articles = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    articles = articles[:args.limit]
    print(f"[INFO] Loaded {len(articles)} articles from {input_path}")

    # ── 初始化 Agent ──
    config = AgentConfig(
        api_key=api_key,
        model_id=args.model_id,
        api_base=args.api_base or AgentConfig.api_base,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        skip_neo4j_write=args.skip_neo4j_write,
        max_workers=args.max_workers,
        extraction_inner_max_workers=args.extraction_inner_max_workers,
        reflection_interval=args.reflection_interval,
        agent_mode=args.agent_mode,
        golden_shot_enabled=not args.disable_golden_shot,
        golden_shot_max_examples=args.golden_shot_max_examples,
        chunked_extraction_enabled=not args.disable_chunked_extraction,
        extraction_chunk_max_chars=args.extraction_chunk_max_chars,
        extraction_chunk_complexity_min_chars=args.extraction_chunk_complexity_min_chars,
        extraction_chunk_max_chunks=args.extraction_chunk_max_chunks,
        extraction_cache_mode=args.extraction_cache_mode,
        extraction_cache_path=args.extraction_cache_path,
        extraction_cache_memory_entries=args.extraction_cache_memory_entries,
        extraction_cache_max_entries=args.extraction_cache_max_entries,
        extraction_cache_max_mb=args.extraction_cache_max_mb,
        extraction_cache_ttl_days=args.extraction_cache_ttl_days,
        tool_router_enabled=not args.disable_tool_router,
        shadow_router_enabled=not args.disable_shadow_router,
        router_execution_mode=args.router_execution_mode,
        router_pre_context_max_mentions=args.router_pre_context_max_mentions,
        execution_mode=args.execution_mode,
        agent_budget_profile=args.agent_budget_profile,
        agent_max_actions=args.agent_max_actions,
        agent_max_aux_remote_calls=args.agent_max_aux_remote_calls,
        agent_max_neo4j_calls=args.agent_max_neo4j_calls,
        agent_soft_timeout=args.agent_soft_timeout,
        agent_hard_timeout=args.agent_hard_timeout,
        pair_classifier_mode=args.pair_classifier_mode,
        pair_classifier_enabled=args.pair_classifier_mode != "off",
        pair_classifier_backend=args.pair_classifier_backend,
        pair_classifier_model_path=args.pair_classifier_model_path,
        pair_classifier_high_confidence=args.pair_classifier_high_confidence,
        pair_classifier_relation_threshold=args.pair_classifier_relation_threshold,
        pair_classifier_uncertainty_floor=args.pair_classifier_uncertainty_floor,
        pair_classifier_max_candidates=args.pair_classifier_max_candidates,
        pair_classifier_max_llm_candidates=args.pair_classifier_max_llm_candidates,
        reviewer_enabled=args.reviewer_enabled,
        reviewer_model_id=args.reviewer_model_id,
        reviewer_api_base=args.reviewer_api_base,
        second_llm_enabled=second_llm_enabled,
        second_llm_provider=second_llm_provider,
        second_llm_api_key=second_llm_api_key,
        second_llm_api_base=second_llm_api_base,
        second_llm_model_id=second_llm_model_id,
        second_llm_mode=second_llm_mode,
        second_llm_timeout=second_llm_timeout,
        neo4j_rag_enabled=neo4j_rag_enabled,
        **rag_limits,
    )
    agent = CognitiveAgent(config)

    # ── 运行 ──
    try:
        report = agent.run_batch(
            articles=articles,
            run_id=run_id,
            output_dir=Path(args.output_dir),
        )
    finally:
        agent.article_preprocessor.close()
        agent.extraction_cache.close()
        agent.kg_memory.close()

    return report


if __name__ == "__main__":
    main()
