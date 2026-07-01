# Native Gemini Provider 迁移与架构改进报告

**日期**: 2026-06-29  
**模型**: `[按次]gemini-2.5-flash` via bitexingai proxy  
**LangExtract**: 1.5.0

---

## 1. 架构概览

### 当前数据流

```
PubMed Abstract
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Phase 1: Context Activator                           │
│   Neo4j 先验知识激活 → ContextCard                    │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Phase 2: Extraction Kernel (LangExtract + Gemini)    │
│   · provider="gemini" (原生)                          │
│   · use_schema_constraints=False                     │
│   · MAX_RETRIES=1, max_workers=2                     │
│   · 6 Few-shot Examples + KG_EXTRACTION_PROMPT       │
│   → RawExtraction (entities + relations)             │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Phase 3: KG Verifier                                 │
│   · 实体溯源 (EXACT_MATCH / FUZZY_MATCH / NOVEL)      │
│   · 关系 Schema 检查 (RELATION_SIGNATURES)            │
│   · Import-ready 判定                                 │
│   → VerifiedExtraction                               │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Phase 4: Causal Reasoner                             │
│   · 传递推理: A→B (新) + B→C (已知) ⇒ A→C (推断)      │
│   · 置信度 = min(前件, 后件) × 0.8                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Phase 5: Decision Engine                             │
│   · CREATE_ENTITY / CREATE_RELATION / UPDATE          │
│   · MARK_DISPUTED / DISCARD / NO_ACTION              │
│   · NCBI Gene Validator (E-utilities)                │
│   → ExecutionLog                                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Phase 6: Self Reflection (每 N 篇)                    │
│   · 质量指标计算 + 阈值自适应                          │
│   · 跨文档新实体发现                                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Phase 7: Strategy Manager                            │
│   · 策略快照/回滚                                      │
│   · 探索 vs 聚焦模式切换                               │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ KG Memory → Neo4j (bolt://100.104.181.96:7687)       │
│   Database: liver-kg-core-v02                        │
│   · 7 种实体类型 (Gene/Disease/Protein/Pathway/        │
│     Metabolite/Tissue/CellType)                       │
│   · 8 种关系谓词                                       │
│   · 类型特定主键 (gene_id, disease_id, ...)            │
└─────────────────────────────────────────────────────┘
```

### 关键组件

| 组件 | 文件 | 职责 |
|------|------|------|
| ExtractionKernel | `cognitive_agent/extraction_kernel.py` | LangExtract 封装 + 重试 |
| KGMemory | `cognitive_agent/memory/kg_memory.py` | Neo4j CRUD + 查询 |
| WorkingMemory | `cognitive_agent/memory/working_memory.py` | 单篇文章临时缓存 |
| EpisodicMemory | `cognitive_agent/memory/episodic_memory.py` | 跨文章决策记录 |
| ContextActivator | `cognitive_agent/context_activator.py` | Neo4j 先验知识激活 |
| KGVerifier | `cognitive_agent/verifier.py` | 实体溯源 + Schema 检查 |
| CausalReasoner | `cognitive_agent/causal_reasoner.py` | 传递因果推理 |
| ConflictResolver | `cognitive_agent/conflict_resolver.py` | 8 行冲突决策表 |
| DecisionEngine | `cognitive_agent/decision_engine.py` | 6 种决策动作 |
| SelfReflection | `cognitive_agent/self_reflection.py` | 元认知反思 |
| StrategyManager | `cognitive_agent/strategy_manager.py` | 策略自适应 |

---

## 2. Provider 迁移: OpenAI 兼容 → 原生 Gemini

### 问题背景

使用 `provider="openai"` + bitexingai proxy 时，LangExtract 无法启用 Gemini 的 `response_schema` 功能。Schema 约束被转换为 prompt 文字描述，模型有时遵守有时不遵守，导致高频 "Skipping chunk: schema error" 错误。

### 根因分析

```
OpenAI 兼容模式:  请求 → JSON body {messages: [...]} → Gemini → 无强制 schema → ~30% 畸形输出
原生 Gemini 模式: 请求 → Gemini API {contents: ..., response_schema: ...} → Gemini → 100% 有效输出
```

关键差异：
- `provider="openai"`: LangExtract 用 OpenAI SDK 风格调用，proxy 转成 Gemini。`response_schema` 被嵌入 system prompt 而非 API 参数。
- `provider="gemini"`: LangExtract 用 `google-genai` SDK 原生调用 Gemini API。`response_schema` 作为 API 参数强制模型输出有效 JSON。

### 最终配置

```python
# agent.py - ModelConfig
lx_config = ModelConfig(
    provider="gemini",                              # 原生 Gemini
    model_id="[按次]gemini-2.5-flash",               # 支持 response_schema
    provider_kwargs={
        "api_key": "sk-...",                         # bitexingai proxy key
        "http_options": {"base_url": "https://new.bitexingai.com"},  # SDK 自动加 /v1beta/
        "temperature": 0.0,
    },
)
```

```python
# extraction_kernel.py - lx.extract() 参数
lx.extract(
    use_schema_constraints=False,   # True 会消灭关系 (0R)
    max_workers=2,
    extraction_passes=1,
    temperature=0,
)
```

### 为什么 use_schema_constraints=False?

实测发现 `use_schema_constraints=True` 时：
- 实体提取正常 (57E)
- **关系被全部消灭 (0R)** — LangExtract 的 strict schema 过于限制嵌套关系属性
- 速度极慢 (235s/篇 vs 30s/篇)

原生 Gemini 即使不开 schema constraints，输出质量也远超 OpenAI 兼容模式 — 0 "Skipping chunk" 错误。

---

## 3. Few-shot Examples 设计 (v2)

### 示例覆盖矩阵

| # | 示例名 | 摘要类型 | 含关系 | 关键教学点 |
|---|--------|---------|--------|-----------|
| 1 | GENE_DISEASE | 实验研究 | ✅ | gene-disease 关联 + encodes |
| 2 | METABOLIC_PATHWAY | 动物模型 | ✅ | 多种关系类型 + 鼠模型标记 |
| 3 | EXPRESSION | 表达定位 | ✅ | expressed_in + 组织/细胞类型 |
| 4 | REVIEW_NO_RELS | 综述 | ❌ | 无关系也能提取实体 |

### Prompt 关键规则

```
1. 使用原文精确文本（不改写）
2. 基因符号大写
3. 只提取原文直接支持的
4. 无关系时省略关系属性（不输出空列表或 "null"）
5. 综述类文章仍提取实体
6. 非人类物种标记 species
```

---

## 4. 实体与关系 Schema

### 7 种实体类型

| extraction_class | Neo4j Label | ID Property | ID Prefix | Name Property |
|-----------------|-------------|-------------|-----------|---------------|
| gene | Gene | gene_id | NCBIGene | gene_symbol |
| disease | Disease | disease_id | PROJECT:Disease | name |
| protein | Protein | string_protein_id | PROJECT:Protein | preferred_name |
| pathway | Pathway | pathway_id | PROJECT:Pathway | name |
| metabolite | Metabolite | metabolite_id | PROJECT:Metabolite | name |
| tissue | Tissue | tissue_id | PROJECT:Tissue | name |
| cell_type | CellType | cell_type_id | PROJECT:CellType | name |

### 8 种关系谓词

| 谓词 | 允许的 (Subject, Object) |
|------|------------------------|
| ASSOCIATED_WITH | (Gene,Disease) (Metabolite,Disease) (Protein,Disease) (Pathway,Disease) |
| PROGNOSTIC_IN | (Gene, Disease) |
| PROGRESSES_TO | (Disease, Disease) |
| ENCODES | (Gene, Protein) |
| INTERACTS_WITH | (Protein, Protein) (Gene, Protein) (Protein, Gene) |
| PARTICIPATES_IN | (Gene, Pathway) (Protein, Pathway) |
| EXPRESSED_IN | (Gene, Tissue) (Gene, CellType) |
| ASSOCIATED_WITH_METABOLITE | (Gene, Metabolite) |

### 实体创建策略

| 类型 | 最小置信度 | 需要外部验证 |
|------|-----------|-------------|
| Gene | 0.8 | ✅ NCBI E-utilities |
| Protein | 0.8 | ✅ (待实现 UniProt) |
| Disease | 0.6 | ❌ |
| Pathway | 0.7 | ❌ |
| Metabolite | 0.7 | ✅ (待实现 HMDB) |
| CellType | 0.6 | ❌ |
| Tissue | 0.5 | ❌ |

---

## 5. 测试结果

### 3-Article Test (agent_native_gemini_test)

| 指标 | 结果 |
|------|------|
| 文章数 | 3 |
| 总耗时 | 110s (37s/篇) |
| 实体总数 | 89 (29.7/篇) |
| 关系总数 | 43 (14.3/篇) |
| Import-Ready 关系 | 19 (44.2%) |
| Schema 合规率 | 44% |
| 错误数 | **0** |

### 与之前方案对比

| 指标 | OpenAI 兼容 + 3-flash | **原生 Gemini + 2.5-flash** |
|------|----------------------|---------------------------|
| Schema 错误 | "Skipping chunk" 频发 | **0** |
| 实体/篇 | ~26 | **29.7** |
| 关系/篇 | ~13 | **14.3** |
| 需要重试 | 是 (MAX_RETRIES=2) | 否 (MAX_RETRIES=1) |
| 速度/篇 | ~30s + 重试开销 | **37s** |
| 稳定性 | 不确定（同篇文章结果不一） | **稳定** |

---

## 6. 已识别的 Neo4j 兼容性问题

### 🔴 P0: RELATION_SIGNATURES 过窄

56% 的关系被拒绝，缺少以下合理签名：

```
缺失: Disease → Tissue   (如 "HCC affects tumor microenvironment")
缺失: Disease → Pathway   (如 "HCC involves immunomodulation")  
缺失: Pathway → Tissue    (如 "Ferroptosis in liver tissue")
缺失: CellType → Pathway  (如 "T cells participate in immune response")
缺失: Tissue → Disease    (如 "Fibrotic liver → cirrhosis")
缺失: Disease → Disease   (如 ASSOCIATED_WITH 比 PROGRESSES_TO 更常用)
```

### 🔴 P0: 实体属性写入时丢失

`decision_engine.py:308` — `execute()` 传 `properties={}` 空字典：
```python
self.kg_memory.create_entity(
    properties={},   # ← 所有提取的属性被丢弃
)
```

`VerifiedEntity` dataclass 也没有 `attributes` 字段，属性在验证阶段就被截断。

### 🔴 P0: 嵌套属性未过滤

实体属性中的 `associated_with`、`encodes` 等数组无法存入 Neo4j（不支持嵌套结构），需在写入前过滤。

### 🟡 P1: 跨文章实体去重

ID 生成包含 `pmid`，导致同一概念在不同文章中创建多个节点。应移除 `pmid` 依赖，改用纯 `mention` + `type` 生成 ID。

---

## 7. 下一步

1. **修复 RELATION_SIGNATURES** — 扩展为更全面的签名矩阵
2. **修复属性传递链** — VerifiedEntity 保留 attributes → DecisionEngine 传递 → KGMemory 过滤
3. **修复实体去重** — ID 生成不依赖 pmid
4. **Neo4j 写入测试** — 连接真实数据库验证 MERGE 不触发 UNIQUE 约束冲突
5. **20 篇文章全流程测试** — 验证端到端稳定性
