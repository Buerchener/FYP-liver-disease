#!/usr/bin/env python3
"""
cognitive_agent/agent.py — 自主认知知识管理 Agent 主循环

基于 LangExtract + DeepSeek，以 Neo4j 为外部动态记忆，
实现 6 阶段闭环认知循环：
  Phase 1: Context Activation  — Neo4j 先验知识激活
  Phase 2: Extract + Ground     — LangExtract 高精度提取
  Phase 3: Verify + Reason      — 图谱溯源验证
  Phase 4: Decision             — 知识决策 (Create/Update/Dispute/...)
  Phase 5: Reflection           — 元认知反思 (每 N 篇)
  Phase 6: Adapt                — 策略自适应

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
from cognitive_agent.decision_engine import DecisionEngine, ExecutionLog
from cognitive_agent.causal_reasoner import CausalReasoner
from cognitive_agent.conflict_resolver import ConflictResolver
from cognitive_agent.self_reflection import SelfReflection
from cognitive_agent.strategy_manager import StrategyManager
from cognitive_agent.schema.examples import KG_EXTRACTION_PROMPT, DEFAULT_EXAMPLES, ALL_EXAMPLES
from cognitive_agent.abbreviation_detector import AbbreviationDetector, AbbreviationMap

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
    neo4j_uri: str = "bolt://100.104.181.96:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "liver-kg-core-v02"

    # 行为
    skip_neo4j_write: bool = True
    max_workers: int = 5  # 并发数 (API 限制约 3-5)
    temperature: float = 0.0
    reflection_interval: int = 10  # 每 N 篇做一次策略反思

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


class CognitiveAgent:
    """
    自主认知知识管理 Agent

    对每篇 PubMed 摘要执行完整的认知推理循环:
    Context → Extract → Verify → Decide → (Reflect → Adapt)
    """

    def __init__(self, config: AgentConfig):
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
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.context_activator = ContextActivator(self.kg_memory)

        # LangExtract 模型配置 — 原生 Gemini provider
        lx_config = ModelConfig(
            provider="gemini",
            model_id=config.model_id,
            provider_kwargs={
                "api_key": config.api_key,
                "http_options": {"base_url": config.api_base},  # 代理 base URL
                "temperature": 0.0,
            },
        )
        self.extraction_kernel = ExtractionKernel(lx_config)
        self.verifier = KGVerifier(self.kg_memory)
        self.causal_reasoner = CausalReasoner(self.kg_memory)
        self.conflict_resolver = ConflictResolver()
        self.decision_engine = DecisionEngine(
            self.kg_memory, skip_neo4j_write=config.skip_neo4j_write
        )
        self.self_reflection = SelfReflection()
        self.strategy_manager = StrategyManager(config)
        self.abbreviation_detector = AbbreviationDetector()

        # 运行时状态
        self.history: list[dict] = []
        self.current_examples = list(DEFAULT_EXAMPLES)

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
            # ── Per-article working memory reset ──
            self.working_memory.clear_article_session()
            self.working_memory.current_article_pmid = pmid

            # ═══════════════════════════════════════════════════
            # Phase 1: Context Activation — 先验知识激活
            # ═══════════════════════════════════════════════════
            t1 = time.time()
            context_card = self.context_activator.activate(text, pmid=pmid)
            record["phases"]["context"] = context_card.to_dict()
            # Save extraction targets in working memory
            self.working_memory.extraction_targets = context_card.extraction_goals
            t1_end = time.time()

            # ── Strategy adaptation from context ──
            strategy = self.strategy_manager.get_strategy(context_card)
            extraction_examples = self._select_examples(strategy)
            extraction_prompt = self._build_strategy_prompt(
                KG_EXTRACTION_PROMPT,
                context_card=context_card,
                strategy=strategy,
            )
            record["phases"]["strategy"] = {
                "active_strategy": strategy,
                "example_count": len(extraction_examples),
                "extraction_goals": context_card.extraction_goals,
            }

            # ── v3: Abbreviation detection (Schwartz-Hearst) ──
            abbr_map = self.abbreviation_detector.detect(text)

            # ═══════════════════════════════════════════════════
            # Phase 2: Extract + Ground — LangExtract 提取
            # ═══════════════════════════════════════════════════
            t2 = time.time()
            raw_extraction = self.extraction_kernel.extract(
                text=text,
                document_id=pmid,
                examples=extraction_examples,
                prompt=extraction_prompt,
            )
            record["phases"]["extraction"] = raw_extraction.to_dict()
            t2_end = time.time()

            # Only skip pipeline if BOTH: extraction completely failed AND no entities
            if raw_extraction.error and len(raw_extraction.entities) == 0:
                record["error"] = raw_extraction.error
                self.state.errors.append({"pmid": pmid, "phase": "extraction", "error": raw_extraction.error})
                record["phases"]["verification"] = {"error": "skipped due to extraction failure"}
                record["phases"]["causal_reasoning"] = {"error": "skipped"}
                record["phases"]["conflict_resolution"] = {"error": "skipped"}
                record["phases"]["execution"] = {"error": "skipped"}
                record["total_time_s"] = time.time() - t_start
                self.history.append(record)
                self.state.total_articles += 1
                return record

            # ═══════════════════════════════════════════════════
            # Phase 3: Verify + Reason — 图谱溯源验证 + 因果推理
            # ═══════════════════════════════════════════════════
            t3 = time.time()
            verified = self.verifier.verify(
                raw_entities=raw_extraction.entities,
                raw_relations=raw_extraction.relations,
                pmid=pmid,
            )
            record["phases"]["verification"] = verified.to_dict()

            # Phase 3+: Causal chain inference
            causal_chains = self.causal_reasoner.infer_causal_chains(verified.relations)
            verified.causal_chains = [c.to_dict() for c in causal_chains]
            record["phases"]["causal_reasoning"] = {
                "chains_inferred": len(causal_chains),
                "chains": verified.causal_chains,
            }
            t3_end = time.time()

            # ═══════════════════════════════════════════════════
            # Phase 4: Conflict Resolution — 冲突检测与解决
            # ═══════════════════════════════════════════════════
            t4 = time.time()
            resolution = self.conflict_resolver.resolve(verified)
            record["phases"]["conflict_resolution"] = resolution.to_dict()
            t4_end = time.time()

            # ═══════════════════════════════════════════════════
            # Phase 5: Decision — 知识决策 (Create/Update/Dispute/...)
            # ═══════════════════════════════════════════════════
            t5 = time.time()
            execution_log = self.decision_engine.decide(
                verified_entities=verified.entities,
                verified_relations=verified.relations,
                pmid=pmid,
                strategy=strategy,
                conflict_resolution=resolution,
                abbr_map=abbr_map,
            )
            # Phase 5+: Execute — 执行 Neo4j 写入
            execution_log = self.decision_engine.execute(execution_log)
            record["phases"]["execution"] = execution_log.to_dict()
            t5_end = time.time()

            # 更新状态 + 记录情景记忆
            self._update_state(raw_extraction, verified, execution_log, pmid=pmid, title=title)

            # ═══════════════════════════════════════════════════
            # Phase 6: Reflect & Adapt — 元认知反思 (每 N 篇)
            # ═══════════════════════════════════════════════════
            if self.state.total_articles > 0 and self.state.total_articles % self.config.reflection_interval == 0:
                strategy_update = self.self_reflection.reflect(
                    execution_log=execution_log,
                    context_card=context_card,
                    agent_state=self.state,
                    episodic_memory=self.episodic_memory,
                )
                if strategy_update.has_changes():
                    self.strategy_manager.apply_update(strategy_update)
                record["phases"]["reflection"] = {
                    "strategy_update": strategy_update.to_dict(),
                    "current_strategy": self.strategy_manager.state.to_dict(),
                }

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

        # 质量评分
        qs = self._compute_quality_score(verified)
        self.state.safe_append("quality_scores", qs)

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
        """计算单篇提取质量分数"""
        total_rel = max(verified.summary.get("total_relations", 0), 1)
        schema_rate = verified.summary.get("schema_valid", 0) / total_rel
        import_rate = verified.summary.get("import_ready", 0) / total_rel

        total_ent = max(verified.summary.get("total_entities", 0), 1)
        link_rate = (
            verified.summary.get("exact_matches", 0) + verified.summary.get("fuzzy_matches", 0)
        ) / total_ent

        return 0.4 * schema_rate + 0.3 * import_rate + 0.3 * link_rate

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

        return (
            f"{base_prompt}\n\n"
            "Runtime strategy from KG context:\n"
            f"- extraction_mode: {mode}\n"
            f"- extraction_goals: {goals}\n"
            f"- instruction: {mode_instruction}\n"
        )

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

        # 生成报告
        report = self._generate_report(run_id, articles, t_batch_end - t_batch_start)

        # 保存结果
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"agent_results_{run_id}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "report": report,
                "config": {
                    "model_id": self.config.model_id,
                    "skip_neo4j_write": self.config.skip_neo4j_write,
                    "reflection_interval": self.config.reflection_interval,
                    "max_workers": max_workers,
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

    def _generate_report(
        self,
        run_id: str,
        articles: list[dict],
        total_time_s: float,
    ) -> dict:
        """生成批量处理报告"""
        n = len(articles)
        s = self.state
        return {
            "run_id": run_id,
            "model": self.config.model_id,
            "neo4j_connected": self.kg_memory.is_connected,
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
                "entities_created": s.total_entities_created,
                "relations_created": s.total_relations_created,
                "relations_updated": s.total_relations_updated,
                "disputed": s.total_disputed,
                "discarded": s.total_discarded,
                "import_ready_rate": round(s.total_import_ready / max(s.total_relations_extracted, 1), 3),
                "discard_rate": round(s.total_discarded / max(s.total_relations_extracted, 1), 3),
            },
            "quality": {
                "avg_quality_score": round(
                    sum(s.quality_scores) / max(len(s.quality_scores), 1), 3
                ),
                "error_count": len(s.errors),
                "errors": s.errors[:10],  # 仅展示前10个错误
            },
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
        print(f"  创建实体:        {dec['entities_created']}")
        print(f"  创建关系:        {dec['relations_created']}")
        print(f"  更新关系:        {dec['relations_updated']}")
        print(f"  标记争议:        {dec['disputed']}")
        print(f"  丢弃:            {dec['discarded']} ({dec['discard_rate']:.1%})")
        print(f"")
        print(f"  ── 质量 ──")
        print(f"  平均质量分:      {q['avg_quality_score']:.2f}")
        print(f"  错误数:          {q['error_count']}")
        print(f"  Neo4j 连接:      {'✅' if report['neo4j_connected'] else '❌ (offline mode)'}")
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
        "--max-workers", type=int, default=5,
        help="并发 worker 数 (默认 5，设 1 为串行)",
    )
    args = parser.parse_args()

    # ── 环境变量检查（GEMINI_API_KEY → DEEPSEEK_API_KEY 级联回退） ──
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    neo4j_password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD", "")

    if not api_key:
        print("[ERROR] No API key found. Set one of:")
        print("  export GEMINI_API_KEY='...'")
        print("  export DEEPSEEK_API_KEY='...'")
        print("  Or pass: --api-key '...'")
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
        neo4j_password=neo4j_password,
        skip_neo4j_write=args.skip_neo4j_write,
        max_workers=args.max_workers,
        reflection_interval=args.reflection_interval,
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
        agent.kg_memory.close()

    return report


if __name__ == "__main__":
    main()
