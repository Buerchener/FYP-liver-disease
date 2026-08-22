#!/usr/bin/env python3
"""Pair-centric LLM relation judgement for the high-recall candidate lattice.

The lattice proposes schema-compatible entity pairs; the judge decides the
predicate (or NO_RELATION), the exact evidence quote, the orientation and the
entailment label.  The judge may not add endpoints, may not use predicates
outside the pair's schema shortlist, and may not rewrite evidence — every
returned quote is realigned to the source text, and anything the judge says
still passes through the deterministic verifier.

This module has no write authority.  `refine_pair_result` replaces the local
classifier's predictions inside a PairClassificationResult, so every
downstream stage (evidence realignment, bounded DeepSeek adjudication, the
verifier, the Decision Engine and Safe Write) keeps working unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from cognitive_agent.abbreviation_detector import AbbreviationDetector
from cognitive_agent.aux_model_registry import AuxModelRegistry
from cognitive_agent.evidence_units import ArticleEvidenceReader, EvidenceUnit
from cognitive_agent.extraction_quality import locate_contiguous, normalize_surface
from cognitive_agent.few_shot_retriever import FewShotRetriever, detect_error_signatures
from cognitive_agent.relation_pair_classifier import (
    NO_RELATION,
    RESULT_SECTIONS,
    PairClassificationResult,
    PairPrediction,
    RelationPairCandidate,
)

try:
    from cognitive_agent.collaborative_extractor import RELATION_DESCRIPTIONS
except Exception:  # pragma: no cover - registry decoupling fallback
    RELATION_DESCRIPTIONS: dict[str, str] = {}


JUDGE_BACKEND = "pairwise_judge_v1"
ENTAILED = "ENTAILED"
CONTRADICTED = "CONTRADICTED"
NOT_ENOUGH_INFORMATION = "NOT_ENOUGH_INFORMATION"
JUDGE_DECISION_LABELS = frozenset({ENTAILED, CONTRADICTED, NOT_ENOUGH_INFORMATION})
ORIENTATION_LABELS = frozenset({"A_TO_B", "B_TO_A", "NONE"})
SYMMETRIC_PREDICATES = frozenset({"ASSOCIATED_WITH", "INTERACTS_WITH"})

# ── Stage A: Claim Gate v2 ──
# Round-4: two INDEPENDENT judgements.  (A) relation_asserted — does the text
# explicitly assert a relation between A and B at all?  (B) claim_role — what
# role does that assertion play in this article?  The old DIRECT_FINDING-only
# gate conflated the two, killing legitimate prior-work / background relations
# (Round-3 gate recall was 50%).
#
# relation_asserted: ASSERTED | NOT_ASSERTED | UNCERTAIN
#   "Previous studies showed X is associated with Y." → ASSERTED (it *does*
#   assert a relation), regardless of whether it is the article's own finding.
# claim_role: CURRENT_FINDING | PRIOR_WORK | BACKGROUND | METHOD |
#             PREDICTION | SPECULATIVE | OTHER
CLAIM_ASSERTED = "ASSERTED"
CLAIM_NOT_ASSERTED = "NOT_ASSERTED"
CLAIM_UNCERTAIN = "UNCERTAIN"
CLAIM_ASSERTED_LABELS = frozenset({CLAIM_ASSERTED, CLAIM_NOT_ASSERTED, CLAIM_UNCERTAIN})
CLAIM_ROLE_CURRENT_FINDING = "CURRENT_FINDING"
CLAIM_ROLE_PRIOR_WORK = "PRIOR_WORK"
CLAIM_ROLE_BACKGROUND = "BACKGROUND"
CLAIM_ROLE_METHOD = "METHOD"
CLAIM_ROLE_PREDICTION = "PREDICTION"
CLAIM_ROLE_SPECULATIVE = "SPECULATIVE"
CLAIM_ROLE_OTHER = "OTHER"
CLAIM_ROLE_LABELS = frozenset({
    CLAIM_ROLE_CURRENT_FINDING, CLAIM_ROLE_PRIOR_WORK, CLAIM_ROLE_BACKGROUND,
    CLAIM_ROLE_METHOD, CLAIM_ROLE_PREDICTION, CLAIM_ROLE_SPECULATIVE,
    CLAIM_ROLE_OTHER,
})
# claim_roles that should NOT be treated as this article's new writeable
# evidence even when the relation is asserted.  They may still form a valid
# *semantic* relation, but are not import-ready.
NON_CURRENT_FINDING_ROLES = frozenset({
    CLAIM_ROLE_PRIOR_WORK, CLAIM_ROLE_BACKGROUND, CLAIM_ROLE_METHOD,
    CLAIM_ROLE_PREDICTION, CLAIM_ROLE_SPECULATIVE, CLAIM_ROLE_OTHER,
})
# Backwards-compat: the old DIRECT_FINDING status set is retained as an alias
# so existing consumers (gate_table, audits) keep working.
CLAIM_STATUS_DIRECT_FINDING = "DIRECT_FINDING"
CLAIM_GATE_STATUSES = frozenset({
    CLAIM_STATUS_DIRECT_FINDING,
    "BACKGROUND",          # background/prior-knowledge sentence, not a new finding
    "PRIOR_WORK",          # explicitly cited prior work ("previous studies have shown")
    "COHORT_CONTEXT",      # sampling/measurement description ("X was measured in patients with Y")
    "METHOD",              # methods/materials mention (reagents, cell lines, assays)
    "PREDICTION_ONLY",     # computational prediction / docking / enrichment
    "SPECULATIVE",         # hedged wording (may/could/potential/hypothesized)
    "NO_EXPLICIT_RELATION",  # bare co-occurrence, no relational statement
})
GATE_TERMINAL_STATUSES = CLAIM_GATE_STATUSES - {CLAIM_STATUS_DIRECT_FINDING}

# Judge-only entailment triggers, deliberately narrower than the verifier's
# PREDICATE_TRIGGERS.  The first judge run massively over-accepted
# ASSOCIATED_WITH from directional verbs ("increased in", "promotes") and
# PARTICIPATES_IN from "involved in": an ENTAILED decision therefore requires
# an explicit association-class / predicate-specific wording inside the quote.
# Everything else stays positive but is routed to the bounded adjudicator.
JUDGE_ENTAILMENT_TRIGGERS: dict[str, tuple[str, ...]] = {
    "ASSOCIATED_WITH": (
        r"\bassociat\w+ (?:with|between)\b", r"\bcorrelat\w+ (?:with|between)\b",
        r"\blinked to\b", r"\brelated to\b", r"\brisk factor (?:for|of)\b",
        r"\bindependent(?:ly)? associated\b",
    ),
    "ASSOCIATED_WITH_METABOLITE": (
        r"\b(?:metabolic|metabolite) association\b", r"\bassociat\w+ with\b",
        r"\bcorrelat\w+ with\b",
    ),
    "PROGNOSTIC_IN": (
        r"\bprognos\w+",
        r"\bpredict\w* (?:of|for)? (?:survival|outcome|mortality|recurrence)\b",
        r"\bassociated with (?:overall )?(?:survival|outcome)\b",
    ),
    "PROGRESSES_TO": (
        r"\bprogress\w* (?:in)?to\b", r"\bdevelop\w* into\b", r"\bevolv\w* into\b",
    ),
    "ENCODES": (r"\bencod\w+",),
    "INTERACTS_WITH": (r"\binteract\w* with\b", r"\bbind\w* (?:to|with)\b"),
    "PARTICIPATES_IN": (
        r"\bparticipat\w* in\b", r"\bmediat\w+", r"\bplays? a role in\b",
    ),
    "EXPRESSED_IN": (
        r"\bexpress\w+ (?:in|by|within)\b", r"\bpresent in\b",
        r"\blocali[sz]\w+ (?:in|to)\b",
    ),
}


@dataclass(frozen=True)
class PairwiseJudgeConfig:
    mode: str = "off"  # off | shadow | active
    model_id: str = "deepseek-v4-flash"
    api_base: str = ""
    max_pairs_per_call: int = 32
    max_calls_per_article: int = 2
    min_confidence: float = 0.70
    write_endorsement_confidence: float = 0.85
    temperature: float = 0.0
    timeout_s: float = 120.0
    few_shot_mode: str = "off"  # off | retrieval
    few_shot_pool: str = ""
    few_shot_source: str = ""
    few_shot_max_examples: int = 4
    include_rule_context: bool = False
    max_context_sentences: int = 4
    max_context_chars: int = 1200
    # Only judge pairs carrying an extractor hint or an explicit trigger.
    # Bare co-occurrence pairs then abstain as NO_RELATION without a remote
    # call: the judge massively over-accepts ASSOCIATED_WITH when asked to
    # label dozens of hint-less co-occurrence pairs per article.
    judge_only_hinted: bool = False
    # Two-stage judgement: Stage A Claim Gate classifies every candidate's
    # claim status (DIRECT_FINDING / BACKGROUND / ... / NO_EXPLICIT_RELATION);
    # only DIRECT_FINDING reaches Stage B (predicate judgement).  All other
    # statuses are gated to NO_RELATION deterministically.
    claim_gate_enabled: bool = False
    # Ablation switch: with the gate on but the predicate stage off,
    # DIRECT_FINDING survivors keep their local backend prediction instead of
    # receiving a predicate decision (isolates the gate's own effect).
    predicate_stage_enabled: bool = True


@dataclass
class JudgePairDecision:
    candidate_id: str
    label: str
    confidence: float
    direction: str
    decision: str
    evidence_quote: str
    rationale: str
    valid: bool = True
    reason_codes: list[str] = field(default_factory=list)
    claim_status: str = ""       # Stage A output; "" when no gate was run
    claim_stage: str = "predicate"  # gate | predicate
    # Claim Gate v2 outputs (filled when claim_gate_enabled):
    relation_asserted: str = ""    # ASSERTED | NOT_ASSERTED | UNCERTAIN
    claim_role: str = ""           # CURRENT_FINDING | PRIOR_WORK | ...
    non_current_finding: bool = False  # True when claim_role in NON_CURRENT_FINDING_ROLES

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class JudgeBatchAudit:
    model_id: str = ""
    status: str = "NOT_RUN"
    stage: str = "predicate"  # gate | predicate
    pairs_requested: int = 0
    pairs_parsed: int = 0
    pairs_valid: int = 0
    positive_count: int = 0
    no_relation_count: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    provider_cache_read_tokens: int = 0
    provider_cache_write_tokens: int = 0
    provider_cache_miss_tokens: int = 0
    provider_cache_hit_rate: float = 0.0
    uncached_input_tokens: int = 0
    local_result_hit: bool = False
    provider_prompt_hit: bool = False
    singleflight_shared: bool = False
    latency_s: float = 0.0
    error: str = ""
    few_shot_count: int = 0
    gate_counts: dict[str, int] = field(default_factory=dict)  # claim_status tally
    asserted_counts: dict[str, int] = field(default_factory=dict)  # relation_asserted tally
    role_counts: dict[str, int] = field(default_factory=dict)  # claim_role tally
    gate_pass_count: int = 0  # relation_asserted == ASSERTED

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class PairwiseJudge:
    """Closed-label pairwise relation judge with exact-quote discipline."""

    def __init__(
        self,
        config: PairwiseJudgeConfig,
        registry: AuxModelRegistry | None = None,
        *,
        few_shot_retriever: FewShotRetriever | None = None,
    ):
        if config.mode not in {"off", "shadow", "active"}:
            raise ValueError("pairwise judge mode must be off, shadow, or active")
        self.config = config
        self.registry = registry
        self.few_shot_retriever = few_shot_retriever
        self.audits: list[JudgeBatchAudit] = []

    # ────────────────────────── context building ──────────────────────────

    @staticmethod
    def _entity_alias_index(entities: list[dict], abbr_map: Any) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for entity in entities:
            mention = str(entity.get("mention", "") or "")
            values = [
                mention,
                *(str(item or "") for item in (entity.get("canonical_mentions", []) or [])),
            ]
            if abbr_map and mention:
                values.extend([
                    abbr_map.resolve_to_long(mention),
                    abbr_map.resolve_to_short(mention),
                ])
            aliases = list(dict.fromkeys(
                str(value).strip() for value in values if str(value or "").strip()
            ))
            for alias in aliases:
                surface = normalize_surface(alias)
                existing = index.get(surface, [])
                for value in aliases:
                    if value not in existing:
                        existing.append(value)
                index[surface] = existing
        return index

    @staticmethod
    def _mentions_for(mention: str, alias_index: dict[str, list[str]]) -> list[str]:
        surface = normalize_surface(mention)
        values = [mention]
        if surface in alias_index:
            values.extend(item for item in alias_index[surface] if item != mention)
        return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))

    def build_context(
        self,
        candidate: RelationPairCandidate,
        *,
        text: str,
        sentences: list[EvidenceUnit],
        alias_index: dict[str, list[str]],
        claim_status: str = "",
    ) -> dict[str, Any]:
        subject_aliases = self._mentions_for(candidate.subject, alias_index)
        object_aliases = self._mentions_for(candidate.object, alias_index)

        def find_alias(unit_text: str, aliases: list[str]) -> bool:
            lowered = " ".join(unit_text.casefold().split())
            for alias in aliases:
                pattern = " ".join(normalize_surface(alias).split())
                if pattern and pattern in lowered:
                    return True
            return False

        relevant: list[str] = []
        focus_sentence = ""
        chars = 0
        for sentence in sentences:
            has_subject = find_alias(sentence.text, subject_aliases)
            has_object = find_alias(sentence.text, object_aliases)
            if not has_subject and not has_object:
                continue
            label = sentence.section
            rendered = f"[{label}] {sentence.text}"
            if has_subject and has_object:
                focus_sentence = focus_sentence or sentence.text
                rendered = f"[{label}|BOTH] {sentence.text}"
            if chars + len(rendered) > self.config.max_context_chars:
                break
            relevant.append(rendered)
            chars += len(rendered)
            if len(relevant) >= self.config.max_context_sentences:
                break
        if not relevant and candidate.evidence:
            relevant.append(f"[{candidate.evidence_section}|WINDOW] {candidate.evidence}")
        if not focus_sentence and candidate.evidence:
            focus_sentence = candidate.evidence

        allowed = [
            {
                "predicate": predicate,
                "meaning": RELATION_DESCRIPTIONS.get(predicate, predicate),
            }
            for predicate in candidate.allowed_predicates
        ]
        return {
            "candidate_id": candidate.candidate_id,
            "claim_status": claim_status,
            "claim_role": claim_status.split("|", 1)[1] if "|" in claim_status else "",
            "section": candidate.evidence_section,
            "error_signature_hints": detect_error_signatures(
                focus_sentence, candidate.evidence_section
            ),
            "A": {
                "mention": candidate.subject,
                "type": candidate.subject_type,
                "aliases": [item for item in subject_aliases if item != candidate.subject][:6],
            },
            "B": {
                "mention": candidate.object,
                "type": candidate.object_type,
                "aliases": [item for item in object_aliases if item != candidate.object][:6],
            },
            "allowed_predicates": allowed,
            "extractor_hints": candidate.source_predicates,
            "focus_sentence": focus_sentence,
            "sentences": relevant,
        }

    # ────────────────────────────── prompting ──────────────────────────────

    def _render_prompt(
        self,
        contexts: list[dict[str, Any]],
        *,
        pmid: str,
        study_type: str,
        rule_context: str = "",
    ) -> str:
        few_shot_lines: list[str] = []
        if (
            self.config.few_shot_mode == "retrieval"
            and self.few_shot_retriever is not None
        ):
            for context in contexts:
                examples = self.few_shot_retriever.retrieve(
                    subject=str(context["A"]["mention"]),
                    subject_type=str(context["A"]["type"]),
                    object_=str(context["B"]["mention"]),
                    object_type=str(context["B"]["type"]),
                    allowed_predicates=[
                        str(item["predicate"]) for item in context["allowed_predicates"]
                    ],
                    source_predicates=list(context.get("extractor_hints") or []),
                    focus_sentence=str(context.get("focus_sentence") or ""),
                    study_type=study_type,
                    exclude_pmid=pmid,
                    mode=self.config.few_shot_mode,
                    section=str(context.get("section") or ""),
                )
                for example in examples:
                    rendered = example.render()
                    if rendered not in few_shot_lines:
                        few_shot_lines.append(rendered)
        few_shot_block = (
            "\n".join(f"  {line}" for line in few_shot_lines[: self.config.few_shot_max_examples])
            if few_shot_lines
            else "  （无可用示范）"
        )
        rule_block = (
            f"\n已激活规则（soft guidance，不改变 schema）:\n{rule_context}"
            if self.config.include_rule_context and rule_context
            else ""
        )
        return f"""你是肝病知识图谱的实体对谓词裁判（Predicate Judge），不是抽取器，也不判定"到底有没有关系"。

Claim Gate 已经判定过该对的关系是否被原文**明确表达**。你只负责一件事：
如果 Claim Gate 说 relation_asserted=ASSERTED，选一个最精确的谓词（"是什么关系"）。
不要再用 NO_RELATION 去推翻 gate 的 ASSERTED 判定——那属于确定性 verifier 的职责。
若你无法从 allowed_predicates 中明确选择一个谓词，返回 NOT_ENOUGH_INFORMATION，
不要为了提高召回强行选 ASSOCIATED_WITH。

对每个候选对返回一个决定：
- predicate：必须从该候选的 allowed_predicates 中选一个，或 NO_RELATION（仅当文本与该对确实无任何支持时）；
- evidence_quote：原文中支持该判定的最短连续引文，必须逐字来自原文（可用文中显式缩写指代端点），且同时覆盖两个端点与该谓词的语义；
- direction：A_TO_B 表示 A→B，B_TO_A 表示 B→A，NONE 表示无方向；
- decision：ENTAILED（引文支持该谓词）、CONTRADICTED（引文与该谓词矛盾）、NOT_ENOUGH_INFORMATION（证据不足）；
- confidence：0-1 的判定置信度。

claim_role（Claim Gate 提供，仅作语义背景，不改变你的谓词选择）：
- CURRENT_FINDING：本文实验/观察结果 → 通常应选精确谓词；
- PRIOR_WORK / BACKGROUND：既往研究或领域知识 → 语义关系仍可成立，但不要把它当成本文新发现去过度选关联词；
- METHOD / PREDICTION / SPECULATIVE：方法、预测或推测 → 通常证据不足，倾向 NOT_ENOUGH_INFORMATION。

关键硬负例：
1. "potential target for diagnosis/treatment" 不等于 PROGNOSTIC_IN；
2. 两个分子在同一句中表达改变，不等于 INTERACTS_WITH；
3. 共同出现、同一列表、同一研究背景，不等于 ASSOCIATED_WITH；
4. 数据库筛选、富集、docking、计算预测不是当前文章实验事实；
5. 如果只是共现而没有任何谓词语义，返回 NOT_ENOUGH_INFORMATION 而不是编造谓词。

强制规则：只有 ENTAILED 判定才需要 evidence_quote，且 quote 中必须出现与
predicate 对应的显式触发表述（例如 ASSOCIATED_WITH 需要 associated with/
correlated with/linked to/related to 等 association 类表述，EXPRESSED_IN 需要
expressed in/present in 等，INTERACTS_WITH 需要 interacts with/binds to 等，
PARTICIPATES_IN 需要 participates in/mediates 等）。两个端点仅仅出现在同一句里
不构成触发；找不到触发表述的候选必须返回 NOT_ENOUGH_INFORMATION。confidence
只在触发表述真实存在时给高。

ASSOCIATED_WITH 特别警示：它是宽泛谓词、误报代价最高。increased/decreased/
promotes/suppresses 等方向性动词只描述方向，不自动构成 ASSOCIATED_WITH；
没有 association 类表述的候选一律 NOT_ENOUGH_INFORMATION，宁可漏报不可误报。
confidence 低于 0.7 时不得给 ENTAILED。

标注示范（仅作参考，不得复制其证据文本）:
{few_shot_block}{rule_block}

PMID: {pmid}
候选对: {json.dumps(contexts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}
"""

    def _render_gate_prompt(
        self, contexts: list[dict[str, Any]], *, pmid: str, study_type: str,
    ) -> str:
        return f"""你是肝病知识图谱的实体对"论断门控 v2"（Claim Gate v2），不是抽取器，也不是谓词裁判。

你的任务对每个候选对做两个**独立**判断，仅输出 JSON：

（A）relation_asserted — 本文章是否**明确表达**实体 A 与 B 之间存在某种关系？
   - ASSERTED：原文直接陈述了 A 与 B 的关系（无论是本文发现、背景知识、既往研究还是方法）。
     例如："Previous studies showed that X is associated with Y." → ASSERTED。
     例如："X levels were measured in Y patients." → NOT_ASSERTED（仅测量对象，无关系结论）。
   - NOT_ASSERTED：只有共同出现 / 并列 / 列表 / 测量描述，没有关系性陈述。
   - UNCERTAIN：关系性 wording 模糊，无法确定。

（B）claim_role — 该关系断言在本文章中扮演哪个角色？
   - CURRENT_FINDING：本文自己的实验/观察结果。
   - PRIOR_WORK：明确引用前人工作的结果。
   - BACKGROUND：作为领域知识在引言提及，不属本文发现。
   - METHOD：方法/材料/Assay/细胞系/动物模式提及，非研究结论。
   - PREDICTION：计算预测 / 数据库筛选 / docking / 富集分析。
   - SPECULATIVE：推测性表述（may / could / potential / hypothesized / …）。
   - OTHER：无法归类。

判断 A 与 B **解藕**：ASSERTED + PRIOR_WORK 是合法组合（原文确实表达了关系，
但该关系来自既往研究）。只有 relation_asserted=NOT_ASSERTED/UNCERTAIN 的候选
才跳过 Predicate Judge 而直接 NO_RELATION。

判定规则（保守）：
- "在…患者中检测/测量/表达…" → NOT_ASSERTED（仅测量对象）；
- "Previous studies / It has been reported that … associated with …" → ASSERTED + PRIOR_WORK；
- "is known to play a role in …" → ASSERTED + BACKGROUND；
- 并列 / 列表 / 无关系动词 → NOT_ASSERTED；
- "may / could / potential" → ASSERTED + SPECULATIVE。

PMID: {pmid}
研究类型: {study_type or 'unknown'}
候选对: {json.dumps(contexts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}
"""

    # ────────────────────────────── remote call ──────────────────────────────

    def judge(
        self,
        candidates: list[RelationPairCandidate],
        *,
        text: str,
        entities: list[dict],
        pmid: str,
        study_type: str = "",
        rule_context: str = "",
        claim_status_by_id: dict[str, str] | None = None,
    ) -> tuple[dict[str, JudgePairDecision], JudgeBatchAudit]:
        """Stage B: predicate judgement for one batch of candidates.

        When `claim_status_by_id` is given (two-stage mode) every context
        carries the Claim Gate status and the predicate stage must not
        overturn it: gated candidates never arrive here.
        """
        audit = JudgeBatchAudit(
            model_id=self.config.model_id,
            status="NOT_RUN",
            stage="predicate",
            pairs_requested=len(candidates),
        )
        if not candidates or self.registry is None or not self.registry.configured("judge"):
            audit.status = "UNCONFIGURED" if candidates else "EMPTY"
            audit.error = "judge model unavailable"
            return {}, audit
        reader = ArticleEvidenceReader()
        units = reader.read(text)
        sentences = ArticleEvidenceReader.parent_units(text, units)
        abbr_map = AbbreviationDetector().detect(text)
        alias_index = self._entity_alias_index(entities, abbr_map)
        contexts = [
            self.build_context(
                candidate, text=text, sentences=sentences, alias_index=alias_index,
                claim_status=(claim_status_by_id or {}).get(candidate.candidate_id, ""),
            )
            for candidate in candidates
        ]
        prompt = self._render_prompt(
            contexts, pmid=pmid, study_type=study_type, rule_context=rule_context,
        )
        result = self.registry.call_json(
            "judge",
            system_prompt=(
                "Closed-label biomedical relation judgement. Return strict JSON only. "
                "Never invent candidates, predicates, endpoints or evidence."
            ),
            user_prompt=prompt,
            schema_hint={
                "decisions": [{
                    "candidate_id": "",
                    "subject": "", "object": "", "predicate": "",
                    "evidence_quote": "", "direction": "A_TO_B",
                    "decision": "ENTAILED", "confidence": 0.0, "rationale": "",
                }]
            },
        )
        audit.model_id = result.model_id
        audit.prompt_tokens = result.prompt_tokens
        audit.output_tokens = result.output_tokens
        audit.provider_cache_read_tokens = result.provider_cache_read_tokens
        audit.provider_cache_write_tokens = result.provider_cache_write_tokens
        audit.provider_cache_miss_tokens = result.provider_cache_miss_tokens
        audit.provider_cache_hit_rate = result.provider_cache_hit_rate
        audit.uncached_input_tokens = result.uncached_input_tokens
        audit.local_result_hit = result.local_result_hit
        audit.provider_prompt_hit = result.provider_prompt_hit
        audit.singleflight_shared = result.singleflight_shared
        audit.latency_s = result.latency_s
        if result.status != "OK":
            audit.status = "FALLBACK"
            audit.error = result.error
            return {}, audit
        audit.status = "OK"
        decisions: dict[str, JudgePairDecision] = {}
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        for raw in result.payload.get("decisions", []) or []:
            if not isinstance(raw, dict):
                continue
            audit.pairs_parsed += 1
            candidate_id = str(raw.get("candidate_id", "") or "")
            candidate = by_id.get(candidate_id)
            if candidate is None:
                continue
            decision = self._validate(
                raw, candidate, text=text, alias_index=alias_index,
            )
            decisions[candidate_id] = decision
            if decision.valid:
                audit.pairs_valid += 1
                if decision.label == NO_RELATION:
                    audit.no_relation_count += 1
                else:
                    audit.positive_count += 1
        self.audits.append(audit)
        return decisions, audit

    def judge_claim_gate(
        self,
        candidates: list[RelationPairCandidate],
        *,
        text: str,
        entities: list[dict],
        pmid: str,
        study_type: str = "",
        max_calls: int | None = None,
    ) -> tuple[dict[str, JudgePairDecision], list[JudgeBatchAudit]]:
        """Stage A: classify every candidate's claim status.

        Bounded batches of max_pairs_per_call.  Terminal statuses are turned
        into valid NO_RELATION decisions here; DIRECT_FINDING decisions are
        placeholders that Stage B replaces.
        """
        audits: list[JudgeBatchAudit] = []
        decisions: dict[str, JudgePairDecision] = {}
        if not candidates or self.registry is None or not self.registry.configured("judge"):
            audit = JudgeBatchAudit(
                model_id=self.config.model_id, status="UNCONFIGURED", stage="gate",
                pairs_requested=len(candidates), error="judge model unavailable",
            )
            audits.append(audit)
            self.audits.append(audit)
            return {}, audits
        reader = ArticleEvidenceReader()
        units = reader.read(text)
        sentences = ArticleEvidenceReader.parent_units(text, units)
        abbr_map = AbbreviationDetector().detect(text)
        alias_index = self._entity_alias_index(entities, abbr_map)
        allowed_calls = max(0, int(
            self.config.max_calls_per_article if max_calls is None else max_calls
        ))
        for call_index, offset in enumerate(
            range(0, len(candidates), self.config.max_pairs_per_call)
        ):
            if call_index >= allowed_calls:
                break
            batch = candidates[offset:offset + self.config.max_pairs_per_call]
            audit = JudgeBatchAudit(
                model_id=self.config.model_id, status="NOT_RUN", stage="gate",
                pairs_requested=len(batch),
            )
            contexts = [
                self.build_context(
                    candidate, text=text, sentences=sentences, alias_index=alias_index,
                )
                for candidate in batch
            ]
            prompt = self._render_gate_prompt(contexts, pmid=pmid, study_type=study_type)
            result = self.registry.call_json(
                "judge",
                system_prompt=(
                    "Claim gating for biomedical relation extraction. Return strict "
                    "JSON only. For each candidate pair output two independent fields: "
                    "relation_asserted (ASSERTED | NOT_ASSERTED | UNCERTAIN) and "
                    "claim_role (CURRENT_FINDING | PRIOR_WORK | BACKGROUND | METHOD | "
                    "PREDICTION | SPECULATIVE | OTHER), plus a rationale. "
                    "Never output predicates or evidence."
                ),
                user_prompt=prompt,
                schema_hint={
                    "decisions": [{
                        "candidate_id": "",
                        "relation_asserted": "ASSERTED",
                        "claim_role": "CURRENT_FINDING",
                        "rationale": "",
                    }]
                },
            )
            audit.model_id = result.model_id
            audit.prompt_tokens = result.prompt_tokens
            audit.output_tokens = result.output_tokens
            audit.provider_cache_read_tokens = result.provider_cache_read_tokens
            audit.provider_cache_write_tokens = result.provider_cache_write_tokens
            audit.provider_cache_miss_tokens = result.provider_cache_miss_tokens
            audit.provider_cache_hit_rate = result.provider_cache_hit_rate
            audit.uncached_input_tokens = result.uncached_input_tokens
            audit.local_result_hit = result.local_result_hit
            audit.provider_prompt_hit = result.provider_prompt_hit
            audit.singleflight_shared = result.singleflight_shared
            audit.latency_s = result.latency_s
            if result.status != "OK":
                audit.status = "FALLBACK"
                audit.error = result.error
                audits.append(audit)
                self.audits.append(audit)
                break
            audit.status = "OK"
            by_id = {candidate.candidate_id: candidate for candidate in batch}
            for raw in result.payload.get("decisions", []) or []:
                if not isinstance(raw, dict):
                    continue
                audit.pairs_parsed += 1
                candidate_id = str(raw.get("candidate_id", "") or "")
                candidate = by_id.get(candidate_id)
                if candidate is None:
                    continue
                decision = self._validate_gate(raw, candidate)
                decisions[candidate_id] = decision
                audit.pairs_valid += 1
                audit.gate_counts[decision.claim_status] = audit.gate_counts.get(decision.claim_status, 0) + 1
                audit.asserted_counts[decision.relation_asserted] = audit.asserted_counts.get(decision.relation_asserted, 0) + 1
                audit.role_counts[decision.claim_role] = audit.role_counts.get(decision.claim_role, 0) + 1
                if decision.relation_asserted == CLAIM_ASSERTED:
                    audit.gate_pass_count += 1
                else:
                    audit.no_relation_count += 1
            audits.append(audit)
            self.audits.append(audit)
        return decisions, audits

    def _validate_gate(
        self, raw: dict[str, Any], candidate: RelationPairCandidate,
    ) -> JudgePairDecision:
        relation_asserted = str(raw.get("relation_asserted", "") or "").upper()
        claim_role = str(raw.get("claim_role", "") or "").upper()
        rationale = str(raw.get("rationale", "") or "")[:300]
        # Legacy single-field fallback: claim_status (DIRECT_FINDING etc.).
        legacy_status = str(raw.get("claim_status", "") or "").upper()
        if relation_asserted not in CLAIM_ASSERTED_LABELS:
            # Fail closed: accept legacy DIRECT_FINDING mapping.
            if legacy_status == CLAIM_STATUS_DIRECT_FINDING:
                relation_asserted = CLAIM_ASSERTED
                claim_role = claim_role or CLAIM_ROLE_CURRENT_FINDING
            elif legacy_status in GATE_TERMINAL_STATUSES:
                relation_asserted = CLAIM_NOT_ASSERTED
                claim_role = claim_role or self._legacy_status_to_role(legacy_status)
            else:
                relation_asserted = CLAIM_NOT_ASSERTED
                claim_role = claim_role or CLAIM_ROLE_OTHER
                legacy_status = "NO_EXPLICIT_RELATION"
        # Normalize claim_status to a clean label.
        if legacy_status not in CLAIM_GATE_STATUSES:
            legacy_status = "NO_EXPLICIT_RELATION"
        if claim_role not in CLAIM_ROLE_LABELS:
            claim_role = CLAIM_ROLE_OTHER
        non_current = claim_role in NON_CURRENT_FINDING_ROLES
        if relation_asserted == CLAIM_ASSERTED:
            # Placeholder decision; Stage B fills the predicate.
            return JudgePairDecision(
                candidate_id=candidate.candidate_id,
                label="",  # placeholder; Stage B fills the predicate
                confidence=0.0,
                direction="NONE",
                decision=NOT_ENOUGH_INFORMATION,
                evidence_quote="",
                rationale=rationale,
                valid=True,
                reason_codes=[] if not non_current else ["non_current_finding_role"],
                claim_status=legacy_status or CLAIM_STATUS_DIRECT_FINDING,
                claim_stage="gate",
                relation_asserted=relation_asserted,
                claim_role=claim_role,
                non_current_finding=non_current,
            )
        # NOT_ASSERTED or UNCERTAIN → NO_RELATION.
        reason_codes = [f"claim_gate_{relation_asserted.lower()}"]
        return JudgePairDecision(
            candidate_id=candidate.candidate_id,
            label=NO_RELATION,
            confidence=0.9 if relation_asserted == CLAIM_NOT_ASSERTED else 0.6,
            direction="NONE",
            decision=NOT_ENOUGH_INFORMATION,
            evidence_quote="",
            rationale=rationale,
            valid=True,
            reason_codes=reason_codes,
            claim_status=legacy_status or "NO_EXPLICIT_RELATION",
            claim_stage="gate",
            relation_asserted=relation_asserted,
            claim_role=claim_role,
            non_current_finding=non_current,
        )

    @staticmethod
    def _legacy_status_to_role(status: str) -> str:
        """Map old CLAIM_GATE_STATUSES to new claim_role labels."""
        mapping = {
            "BACKGROUND": CLAIM_ROLE_BACKGROUND,
            "PRIOR_WORK": CLAIM_ROLE_PRIOR_WORK,
            "COHORT_CONTEXT": CLAIM_ROLE_OTHER,
            "METHOD": CLAIM_ROLE_METHOD,
            "PREDICTION_ONLY": CLAIM_ROLE_PREDICTION,
            "SPECULATIVE": CLAIM_ROLE_SPECULATIVE,
            "NO_EXPLICIT_RELATION": CLAIM_ROLE_OTHER,
        }
        return mapping.get(status, CLAIM_ROLE_OTHER)

    def _validate(
        self,
        raw: dict[str, Any],
        candidate: RelationPairCandidate,
        *,
        text: str,
        alias_index: dict[str, list[str]],
    ) -> JudgePairDecision:
        reason_codes: list[str] = []
        label = str(raw.get("predicate", "") or "").upper()
        direction = str(raw.get("direction", "") or "NONE").upper()
        decision = str(raw.get("decision", "") or "").upper()
        quote = str(raw.get("evidence_quote", "") or "")
        rationale = str(raw.get("rationale", "") or "")[:300]
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence <= 0.0:
            confidence = {"ENTAILED": 0.80, "CONTRADICTED": 0.90,
                          NOT_ENOUGH_INFORMATION: 0.40}.get(decision, 0.4)
        confidence = max(0.0, min(1.0, confidence))

        valid = True
        # The judge may echo endpoints but can never change them: any
        # disagreement is recorded and the lattice endpoints win.
        raw_subject = str(raw.get("subject", "") or "")
        raw_object = str(raw.get("object", "") or "")
        if raw_subject and (
            normalize_surface(raw_subject) != normalize_surface(candidate.subject)
            and normalize_surface(raw_subject) not in {
                normalize_surface(item) for item in alias_index.get(
                    normalize_surface(candidate.subject), []
                )
            }
        ):
            reason_codes.append("judge_endpoint_disagreement_subject")
        if raw_object and (
            normalize_surface(raw_object) != normalize_surface(candidate.object)
            and normalize_surface(raw_object) not in {
                normalize_surface(item) for item in alias_index.get(
                    normalize_surface(candidate.object), []
                )
            }
        ):
            reason_codes.append("judge_endpoint_disagreement_object")
        if label != NO_RELATION and label not in candidate.allowed_predicates:
            # Predicate not in schema shortlist: do NOT force a wrong predicate.
            # The Claim Gate already said the relation is asserted (has
            # semantic content), so downgrade to NOT_ENOUGH_INFORMATION with a
            # NO_RELATION label and mark manual_review; the verifier/adjudicator
            # decides rather than emitting a fabricated predicate.
            reason_codes.append("predicate_not_in_shortlist")
            label = NO_RELATION
            decision = NOT_ENOUGH_INFORMATION
            reason_codes.append("manual_review")
        if decision not in JUDGE_DECISION_LABELS:
            reason_codes.append("invalid_decision_label")
            decision = NOT_ENOUGH_INFORMATION
        if direction not in ORIENTATION_LABELS:
            reason_codes.append("invalid_direction")
            direction = "NONE"
        grounded, start, end = locate_contiguous(quote, text)
        if label != NO_RELATION and decision == ENTAILED:
            if not grounded or not re.search(r"\w", quote, flags=re.UNICODE):
                # A positive semantic decision whose quote cannot be located
                # in the source is not trustworthy enough to override the
                # deterministic heuristics: downgrade it to adjudication.
                reason_codes.append("judge_quote_not_in_source")
                quote = candidate.evidence
                grounded, start, end = locate_contiguous(quote, text)
                if not grounded:
                    valid = False
                    decision = NOT_ENOUGH_INFORMATION
                    label = NO_RELATION
                else:
                    decision = NOT_ENOUGH_INFORMATION
            elif not any(
                re.search(pattern, quote, re.IGNORECASE)
                for pattern in JUDGE_ENTAILMENT_TRIGGERS.get(label, ())
            ):
                # ENTAILED without explicit predicate-specific wording in the
                # quoted span is co-occurrence dressed as a relation (the
                # dominant ASSOCIATED_WITH failure mode).  Do not let it
                # override the deterministic heuristics: send it to the
                # bounded adjudicator instead.
                reason_codes.append("judge_entailment_without_trigger_support")
                decision = NOT_ENOUGH_INFORMATION
        if decision == CONTRADICTED:
            # Evidence contradicts the proposed predicate: the pair is a
            # negative at this level.  Keep the contradiction for audit.
            reason_codes.append("judge_contradicted")
            label = NO_RELATION
        if decision == NOT_ENOUGH_INFORMATION and label == NO_RELATION:
            reason_codes.append("no_relation_or_insufficient")
        return JudgePairDecision(
            candidate_id=candidate.candidate_id,
            label=label,
            confidence=round(confidence, 6),
            direction=direction,
            decision=decision if label != NO_RELATION else NOT_ENOUGH_INFORMATION,
            evidence_quote=quote,
            rationale=rationale,
            valid=valid,
            reason_codes=reason_codes,
            claim_status=str(raw.get("claim_status", "") or "").upper(),
        )

    # ─────────────────────── PairClassificationResult refine ───────────────────────

    def _as_judge_relation(
        self,
        candidate: RelationPairCandidate, decision: JudgePairDecision,
        *, quote_start: int, quote_end: int, section: str, uncertain: bool,
    ) -> dict[str, Any]:
        subject = candidate.subject
        object_ = candidate.object
        direction = "unknown"
        if decision.direction == "B_TO_A" and decision.label not in SYMMETRIC_PREDICATES:
            subject, object_ = object_, subject
        flags = ["pair_classifier_candidate", "pairwise_judge"]
        if uncertain:
            flags.extend(["pair_low_confidence", "judge_uncertain"])
        else:
            flags.append("judge_entailed")
        if decision.confidence < self.config.write_endorsement_confidence:
            # Automatic writes need either very high judge confidence or an
            # independent second-model endorsement; the deterministic write
            # gate stays untouched otherwise.
            flags.append("judge_no_write_endorsement")
        if decision.non_current_finding:
            # Claim Gate v2: the relation is asserted but its claim_role
            # (PRIOR_WORK / BACKGROUND / METHOD / PREDICTION / SPECULATIVE /
            # OTHER) means it is not THIS article's new writeable evidence.
            # It may still form a valid semantic relation, but is not
            # import-ready unless a second model independently upgrades it.
            flags.append("non_current_finding_role")
            flags.append("manual_review")
        flags.extend(decision.reason_codes)
        return {
            "subject": subject,
            "subject_type": candidate.subject_type,
            "predicate": decision.label,
            "object": object_,
            "object_type": candidate.object_type,
            "direction": direction,
            "negated": False,
            "uncertain": uncertain or decision.decision != ENTAILED,
            "evidence": decision.evidence_quote,
            "evidence_unit_id": candidate.evidence_unit_id,
            "evidence_role": section,
            "candidate_id": candidate.candidate_id,
            "classifier_source": JUDGE_BACKEND,
            "classifier_confidence": decision.confidence,
            "relation_probability": decision.confidence,
            "no_relation_probability": round(max(0.0, 1.0 - decision.confidence), 6),
            "classifier_margin": round(abs(decision.confidence - 0.5), 6),
            "evidence_confidence": (
                decision.confidence if decision.decision == ENTAILED else 0.5
            ),
            "evidence_entailment": decision.decision,
            "evidence_char_start": quote_start,
            "evidence_char_end": quote_end,
            "predicate_candidates": {decision.label: decision.confidence},
            "claim_role": decision.claim_role,
            "quality_flags": sorted(set(flags)),
        }

    def refine_pair_result(
        self,
        pair_result: PairClassificationResult,
        *,
        text: str,
        entities: list[dict],
        pmid: str,
        study_type: str = "",
        rule_context: str = "",
    ) -> PairClassificationResult:
        """Replace local classifier predictions with judge decisions.

        Falls back to the original pair_result unchanged when the judge is
        unavailable, so the deterministic backend remains the safety net.
        """
        if self.config.mode == "off" or not pair_result.candidates:
            return pair_result
        # This is a hard article-level request budget, shared by Claim Gate
        # and predicate judgement.  The prior implementation applied the cap
        # independently to both stages and could make twice as many remote
        # requests as configured.
        total_call_budget = max(0, int(self.config.max_calls_per_article))
        eligible = [
            candidate for candidate in pair_result.candidates
            if not self.config.judge_only_hinted
            or candidate.source_predicates
            or candidate.evidence_trigger_predicate
        ]
        if self.config.claim_gate_enabled:
            gate_call_budget = min(total_call_budget, 1 if total_call_budget > 1 else total_call_budget)
            predicate_call_budget = max(0, total_call_budget - gate_call_budget)
            gate_candidate_budget = self.config.max_pairs_per_call * gate_call_budget
            judged_candidates = eligible[:gate_candidate_budget]
        else:
            gate_call_budget = 0
            predicate_call_budget = total_call_budget
            judged_candidates = eligible[: self.config.max_pairs_per_call * total_call_budget]
        skipped = len(eligible) - len(judged_candidates)
        unjudged_ids = {
            candidate.candidate_id for candidate in pair_result.candidates
        } - {
            candidate.candidate_id for candidate in judged_candidates
        }
        decision_by_id: dict[str, JudgePairDecision] = {}
        batch_audits: list[JudgeBatchAudit] = []
        if self.config.claim_gate_enabled:
            # ── Stage A: Claim Gate ──
            gate_decisions, gate_audits = self.judge_claim_gate(
                judged_candidates, text=text, entities=entities, pmid=pmid,
                study_type=study_type, max_calls=gate_call_budget,
            )
            decision_by_id.update(gate_decisions)
            batch_audits.extend(gate_audits)
            gate_ok = bool(gate_decisions)
            stage_b_candidates = (
                [
                    candidate for candidate in judged_candidates
                    if decision_by_id.get(candidate.candidate_id) is not None
                    and decision_by_id[candidate.candidate_id].relation_asserted
                    == CLAIM_ASSERTED
                ]
                if gate_ok
                else list(judged_candidates)  # gate failed: degrade to single-stage
            )
            claim_status_by_id = (
                {
                    candidate_id: f"{decision.relation_asserted}|{decision.claim_role}"
                    for candidate_id, decision in gate_decisions.items()
                }
                if gate_ok else None
            )
            # ── Stage B: Predicate Judge (relation_asserted=ASSERTED only) ──
            if self.config.predicate_stage_enabled and predicate_call_budget > 0:
                for call_index, offset in enumerate(
                    range(0, len(stage_b_candidates), self.config.max_pairs_per_call)
                ):
                    if call_index >= predicate_call_budget:
                        break
                    batch = stage_b_candidates[offset:offset + self.config.max_pairs_per_call]
                    decisions, audit = self.judge(
                        batch, text=text, entities=entities, pmid=pmid,
                        study_type=study_type, rule_context=rule_context,
                        claim_status_by_id=claim_status_by_id,
                    )
                    decision_by_id.update(decisions)
                    batch_audits.append(audit)
                    if audit.status != "OK":
                        break
            # Gate-only ablation: Stage B is off, so DIRECT_FINDING
            # placeholders stay in decision_by_id (their claim_status feeds
            # the gate_table in phase_payload) and the prediction loop keeps
            # the local backend prediction for them (baseline treatment).
        else:
            for offset in range(0, len(judged_candidates), self.config.max_pairs_per_call):
                batch = judged_candidates[offset:offset + self.config.max_pairs_per_call]
                decisions, audit = self.judge(
                    batch, text=text, entities=entities, pmid=pmid,
                    study_type=study_type, rule_context=rule_context,
                )
                decision_by_id.update(decisions)
                batch_audits.append(audit)
                if audit.status != "OK":
                    break

        refined = PairClassificationResult(
            mode=self.config.mode,
            backend=JUDGE_BACKEND,
            candidates=list(pair_result.candidates),
            truncated_candidates=pair_result.truncated_candidates,
        )
        if not decision_by_id:
            refined.fallback_reason = "judge_unavailable_kept_local_predictions"
            refined.predictions = list(pair_result.predictions)
            refined.accepted_relations = list(pair_result.accepted_relations)
            refined.low_confidence_relations = list(pair_result.low_confidence_relations)
            refined.judge_audits = batch_audits
            refined.judge_decisions = dict(decision_by_id)
            return refined

        reader = ArticleEvidenceReader()
        units = reader.read(text)
        sentences = ArticleEvidenceReader.parent_units(text, units)
        by_char: dict[tuple[int, int], str] = {
            (sentence.char_start, sentence.char_end): sentence.section
            for sentence in sentences
        }
        by_char.update({(unit.char_start, unit.char_end): unit.section for unit in units})

        for candidate in pair_result.candidates:
            decision = decision_by_id.get(candidate.candidate_id)
            if (
                decision is not None
                and (
                    not self.config.predicate_stage_enabled
                    or not decision.label
                )
                and decision.claim_stage == "gate"
                and decision.relation_asserted == CLAIM_ASSERTED
                and not decision.label
            ):
                # Gate-only ablation: the ASSERTED survivor keeps its local
                # backend prediction instead of a predicate decision.
                local = next(
                    (item for item in pair_result.predictions
                     if item.candidate_id == candidate.candidate_id), None
                )
                if local is not None:
                    refined.predictions.append(local)
                continue
            if decision is None:
                # Not judged (beyond budget, or abstained under
                # judge_only_hinted): audit as an abstained NO_RELATION in
                # judge mode.  The local backend prediction is retained only
                # for the shadow audit and never produces a relation.
                local = next(
                    (item for item in pair_result.predictions
                     if item.candidate_id == candidate.candidate_id), None
                )
                if local is not None:
                    refined.predictions.append(local)
                elif candidate.candidate_id in unjudged_ids:
                    refined.predictions.append(PairPrediction(
                        candidate_id=candidate.candidate_id,
                        label=NO_RELATION,
                        confidence=0.5,
                        relation_probability=0.5,
                        no_relation_probability=0.5,
                        margin=0.0,
                        direction="unknown",
                        backend=JUDGE_BACKEND,
                        reason_codes=["judge_abstained_unjudged_candidate"],
                        predicate_scores={},
                    ))
                continue
            no_relation_probability = round(max(0.0, 1.0 - decision.confidence), 6)
            prediction = PairPrediction(
                candidate_id=candidate.candidate_id,
                label=decision.label,
                confidence=max(decision.confidence, no_relation_probability),
                relation_probability=decision.confidence,
                no_relation_probability=no_relation_probability,
                margin=round(abs(decision.confidence - 0.5), 6),
                direction="unknown",
                backend=JUDGE_BACKEND,
                routed_to_llm=bool(
                    decision.label != NO_RELATION
                    and (
                        decision.confidence < self.config.min_confidence
                        or decision.decision != ENTAILED
                    )
                ),
                reason_codes=list(decision.reason_codes),
                predicate_scores={decision.label: decision.confidence},
            )
            refined.predictions.append(prediction)
            if decision.label == NO_RELATION:
                continue
            quote_start, quote_end = locate_contiguous(decision.evidence_quote, text)[1:]
            if quote_start < 0:
                quote_start, quote_end = candidate.evidence_char_start, candidate.evidence_char_end
            section = candidate.evidence_section
            for (start, end), value in sorted(by_char.items()):
                if start <= quote_start and quote_end <= end:
                    section = value
                    break
            uncertain = (
                decision.confidence < self.config.min_confidence
                or decision.decision != ENTAILED
            )
            relation = self._as_judge_relation(
                candidate, decision,
                quote_start=quote_start, quote_end=quote_end,
                section=section, uncertain=uncertain,
            )
            if uncertain:
                refined.low_confidence_relations.append(relation)
            else:
                refined.accepted_relations.append(relation)
        refined.low_confidence_relations = refined.low_confidence_relations[:24]
        refined.judge_audits = batch_audits
        refined.judge_decisions = dict(decision_by_id)
        refined.skipped_judge_candidates = skipped
        return refined

    def phase_payload(self, refined: PairClassificationResult) -> dict[str, Any]:
        audits = getattr(refined, "judge_audits", []) or []
        gate_counts: dict[str, int] = {}
        gate_pass = 0
        for audit in audits:
            if audit.stage != "gate":
                continue
            for status, count in audit.gate_counts.items():
                gate_counts[status] = gate_counts.get(status, 0) + count
            gate_pass += audit.gate_pass_count
        payload: dict[str, Any] = {
            "mode": self.config.mode,
            "model_id": self.config.model_id,
            "few_shot_mode": self.config.few_shot_mode,
            "claim_gate_enabled": self.config.claim_gate_enabled,
            "audits": [audit.to_dict() for audit in audits],
            "candidate_count": len(refined.candidates),
            "positive_prediction_count": sum(
                prediction.label != NO_RELATION for prediction in refined.predictions
            ),
            "no_relation_count": sum(
                prediction.label == NO_RELATION for prediction in refined.predictions
            ),
            "accepted_relation_count": len(refined.accepted_relations),
            "low_confidence_count": len(refined.low_confidence_relations),
            "fallback_reason": refined.fallback_reason,
        }
        if gate_counts or gate_pass:
            payload["claim_gate"] = {
                "status_counts": gate_counts,
                "direct_finding_pass": gate_pass,
                "gated_relation_suppressed": sum(
                    count for status, count in gate_counts.items()
                    if status != CLAIM_STATUS_DIRECT_FINDING
                ),
            }
        decisions = getattr(refined, "judge_decisions", {}) or {}
        if self.config.claim_gate_enabled:
            predictions = {item.candidate_id: item for item in refined.predictions}
            payload["gate_table"] = [
                {
                    "subject": candidate.subject,
                    "subject_type": candidate.subject_type,
                    "object": candidate.object,
                    "object_type": candidate.object_type,
                    "relation_asserted": (
                        decisions[candidate.candidate_id].relation_asserted
                        if candidate.candidate_id in decisions else ""
                    ),
                    "claim_status": (
                        decisions[candidate.candidate_id].claim_status
                        if candidate.candidate_id in decisions else ""
                    ),
                    "claim_role": (
                        decisions[candidate.candidate_id].claim_role
                        if candidate.candidate_id in decisions else ""
                    ),
                    "predicate": (
                        predictions[candidate.candidate_id].label
                        if candidate.candidate_id in predictions else ""
                    ),
                    "reason_codes": list(
                        predictions[candidate.candidate_id].reason_codes
                        if candidate.candidate_id in predictions else []
                    ),
                }
                for candidate in refined.candidates
            ]
        return payload
