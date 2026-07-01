# PubMed 文献知识抽取系统：从 Pipeline 到 Cognitive Agent

## Biomedical Knowledge Graph Construction for Liver Disease Progression

**本科毕业设计项目 — PubMed 文献三元组抽取模块完整技术报告**

> 生成日期：2026-06-29  
> 实验基准：liver-kg-core-v02 (836 Gene / 793 Protein / 1,721 Pathway / 36 Metabolite / 5 Disease)  
> 测试规模：30 篇 A/B 实际执行（第一代完成；第二代因 Gemini token 认证失败未形成质量对比）  
> 系统版本：v1.0 (Pipeline) → v2.0 (Cognitive Agent) → v2.1 (Neo4j Write + Quality Guard) → v2.2 (Closed-loop Agent)

---

## 执行摘要

本报告记录的是同一个 PubMed 文献抽取目标的两代实现：

- **第一代 Pipeline**：目标是安全、保守、可审计地从 PubMed abstract 中抽取候选三元组。它采用线性 5-stage 处理，强依赖 schema hard validation 和独立的 entity-linking preflight。其优点是稳定、透明、不会污染 Neo4j；缺点是无状态、不能主动补齐节点、对当前 KG 覆盖不足非常敏感。
- **第二代 Cognitive Agent**：目标是让系统像一个知识 curator 一样读文献。它先激活 Neo4j 先验知识，再抽取、验证、推理、冲突裁决、决策、写入，并通过 episodic memory 和 strategy manager 形成跨文章闭环。v2.2 之后，Context Activation 生成的 strategy 已经会影响 prompt、few-shot examples 和决策阈值；ConflictResolver 的结果也会真正驱动 DecisionEngine。

当前最重要的工程结论：

1. 第一代仍然适合作为 **baseline / audit pipeline**，尤其适合评估“在不创建新节点的前提下，哪些关系可安全进入现有 KG”。
2. 第二代应作为后续主线，因为它能处理第一代暴露出的根本问题：KG 端点覆盖不足、跨文章记忆缺失、冲突处理不足、无法自适应。
3. 两代系统的指标不能只看“实体/关系数量”。更合理的对比应同时看：schema 合规率、import-ready 率、discard/review 率、证据质量、零产出文章比例、错误数、每篇耗时，以及是否产生可写入/可审核的高价值候选。
4. 本次 30 篇 A/B 已按同一输入、同一批前 30 篇、dry-run/offline write 执行。第一代 pipeline 完成并产生可审计 baseline；第二代 agent 跑完整个命令路径，但 30/30 篇均在 Gemini provider 认证层返回 401 Invalid token，因此本次不能把第二代 0 产出解释为模型或 agent 质量问题，只能记录为环境配置失败。

---

## 目录

### 第一部分：项目背景
1. [动机与问题定义](#1-动机与问题定义)

### 第二部分：第一代系统 — Pipeline 模式
2. [系统架构：双脚本协同](#2-系统架构双脚本协同)
3. [开发历程与问题解决](#3-开发历程与问题解决)
4. [创新点](#4-创新点)
5. [30 篇 A/B 实际执行结果](#5-30-篇-ab-实际执行结果)
6. [瓶颈与局限](#6-瓶颈与局限)

### 第三部分：第二代系统 — Cognitive Agent 模式
7. [架构重构：从 Pipeline 到 Cognitive Loop](#7-架构重构从-pipeline-到-cognitive-loop)
8. [Provider 迁移：从 DeepSeek 到 Gemini 原生的完整旅程](#8-provider-迁移从-deepseek-到-gemini-原生的完整旅程)
9. [Cognitive Agent 开发全程问题排查](#9-cognitive-agent-开发全程问题排查)
10. [两代系统全面对比](#10-两代系统全面对比)

### 第三部分续：Neo4j 写入验证
10.5 [Neo4j 写入验证与质量控制](#105-neo4j-写入验证与质量控制)  
10.6 [30 篇 A/B 对比实验方案与命令](#106-30-篇-ab-对比实验方案与命令)

### 第四部分：展望与附录
11. [当前局限与未来方向](#11-当前局限与未来方向)
12. [附录：关键指标汇总](#12-附录关键指标汇总)

---

## 1. 动机与问题定义

### 1.1 为什么需要 PubMed 文献抽取

现有肝病知识图谱 `liver-kg-core-v02` 已整合 DisGeNET、STRING、KEGG、Reactome、Human Protein Atlas、HMDB 六大公共数据库，形成了包含 3,546 个节点和 1,363 条关系的核心图谱。然而，这些数据库存在一个共性问题：**它们收录的是已被充分验证、广泛引用的"教科书级"知识**，更新速度滞后于前沿研究。

以肝病进展链条 `Normal → NAFLD → NASH → Fibrosis → Cirrhosis → HCC` 为例：

- DisGeNET 提供了 gene-disease association，但无法区分同一基因在 NAFLD 和 HCC 中的不同角色
- STRING 提供了 protein-protein interaction，但不包含疾病上下文
- KEGG/Reactome 提供了 pathway membership，但不涉及时序和 stage-specific 证据

**PubMed 文献中蕴含着公共数据库尚未收录的 stage-aware 知识**——某项研究可能揭示了某个基因在 NASH→Fibrosis 转变中的关键作用，或某个代谢物在 HCC 早期诊断中的潜在价值。这些知识只有通过文献抽取才能获取。

### 1.2 核心问题

> 如何从 PubMed 非结构化摘要文本中，自动抽取符合目标图数据库 schema 的结构化三元组（实体-关系-实体），并以可溯源、可校验、可审核的方式写入 Neo4j？

这包含五个子问题：

| 子问题 | 挑战 |
|---|---|
| **Q1. 文本→结构化** | 如何让 LLM 从自由文本中准确识别生物医学实体和关系 |
| **Q2. Schema 约束** | 如何确保 LLM 输出符合目标 KG 的关系签名，不产生垃圾数据 |
| **Q3. 质量保障** | 如何检测否定、不确定、非人类物种、证据不可溯等问题 |
| **Q4. 实体链接** | 如何将 LLM 抽取的实体 mention（如 "Nrf2"）链接到 Neo4j 已有节点（如 NFE2L2） |
| **Q5. 可溯源性** | 如何保留每条关系的证据来源、使得后续可以人工审核 |

### 1.3 设计原则演变

系统的设计原则经历了一次根本性的转变：

**第一代 (v1.0)**：保守导入策略
1. 不创建新实体节点，只在已有核心节点之间写入候选关系
2. 证据锚定：每条关系携带 evidence sentence + PMID + 字符偏移
3. 分层校验：LLM 抽取 → 规则校验 → Schema 校验 → 实体链接 → Gatekeeper
4. 可重复可审计：独立 JSON 结果 + 质量报告 + HTML 溯源报告

**第二代 (v2.0-v2.2)**：主动知识构建与闭环自适应策略
1. 主动创建新实体节点（类型特定主键 + MERGE 语义）
2. 三记忆系统：Working（单篇） → Episodic（跨篇） → KG Memory（持久化）
3. 认知循环：提取 → 验证 → 因果推理 → 冲突裁决 → 决策 → 反思 → 自适应
4. 外部验证：NCBI E-utilities 基因验证 + 可扩展验证框架
5. v2.2 闭环：ContextCard 生成的 strategy 会改变 prompt、few-shot examples、实体/关系创建阈值；ConflictResolver 的裁决会进入 DecisionEngine 并实际驱动 create/update/dispute/discard

---

## 2. 系统架构：双脚本协同

### 2.1 总体架构

第一代系统由两个独立的 Python 脚本组成，分别承担「知识生成」和「知识校验」职责：

```
┌─────────────────────────────────────────────────────────────┐
│                  PubMed 文献知识抽取系统 (v1.0)               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  PubMed JSONL ──→ multi_stage_extraction_pipeline.py          │
│                   │  5-Stage LLM 抽取引擎                     │
│                   │  (生成知识)                               │
│                   ↓                                           │
│              extraction_results.json                          │
│                   │                                           │
│                   └──→ entity_linking_preflight.py            │
│                        Ensemble Entity Linker                 │
│                        (校验知识能否入库)                       │
│                        ↓                                      │
│                   import_candidates → Neo4j                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 multi_stage_extraction_pipeline.py — 五阶段抽取引擎

```
PubMed JSONL (title + abstract + pmid)
  │
  ├─ Stage 0: 文本分类 (规则)
  │    输入: 文本
  │    输出: clinical_note / imaging_report / literature_abstract
  │
  ├─ Stage 1: Few-shot LLM 初提取 (DeepSeek API, temperature=0)
  │    输入: 分类标签 + Schema 约束 + 2个 Few-shot 示例 + 文本
  │    输出: {entities: [...], relations: [...]}
  │    限制: 每篇最多 8 条关系
  │
  ├─ Stage 2: 证据校验与字段补全 (纯规则, 15+ 检查项)
  │    · 证据句定位 (字符级子串匹配)
  │    · 否定检测 / 不确定检测 (正则)
  │    · 物种归一化 (Human → Mouse → Rat → Mixed)
  │    · Schema 签名校验
  │    · 实体类型强制分类检测 (防止 drug/therapy → Metabolite)
  │    · Quality flags 打分 + import_ready 判定
  │
  ├─ Stage 3: 实体标准化 (字典驱动)
  │    Gene mention → HGNC ID (25 个已知 Gene)
  │    Disease mention → UMLS CUI (13 个已知 Disease)
  │    Fallback: MENTION:Type_Name
  │
  ├─ Stage 4: 冲突检测 (规则)
  │    与已有 KG 的对比标记 (negated_claim, low_confidence, etc.)
  │
  ├─ Stage 5: Neo4j 写入 (Cypher MERGE)
  │    保守模式: 不创建新节点, 不创建非白名单关系
  │    携带完整 provenance: evidence + confidence + species + direction + disease_stage
  │
  └─ 输出三件套:
       extraction_results_{run_id}.json
       quality_report_{run_id}.json
       provenance_report_{run_id}.html
```

### 2.3 entity_linking_preflight.py — 实体链接预检层

写入 Neo4j 前的最后一道防线，使用 **ensemble linker** 架构：

```
抽取结果 JSON
  │
  ├─ Indexer Agent: 从 Neo4j 只读导出核心节点索引, 本地缓存
  │
  ├─ Exact Matcher: 主键 / NCBI Gene ID / HMDB ID / KEGG ID 精确匹配
  ├─ Lexical Matcher: 规范化 fuzzy (Greek字母, 大小写, pathway后缀)
  ├─ Alias Hint Matcher: 高价值 biomedical alias (Nrf2→NFE2L2, PAR-1→F2R)
  ├─ Schema Repair Critic: 类型修正候选 (被错标为Protein的EGFR→Gene)
  │
  └─ Gatekeeper: predicate + schema + quality flags + 端点链接 → import_candidate
```

**关键设计选择**：
- 只读操作，绝不修改 Neo4j
- 因为它是独立的，可以在不消耗 LLM token 的情况下快速迭代匹配策略
- 一次运行输出 exact / lexical / ensemble 三种策略的对比

### 2.4 关系 Schema 设计（初版 8 种签名）

| 关系类型 | 允许的 (Subject, Object) 组合 | 导入策略 |
|---|---|---|
| ASSOCIATED_WITH | (Gene, Disease), (Metabolite, Disease) | 可写候选 |
| PROGNOSTIC_IN | (Gene, Disease) | 可写候选 |
| INTERACTS_WITH | (Protein, Protein) | 可写候选 |
| PARTICIPATES_IN | (Gene, Pathway) | 可写候选 |
| EXPRESSED_IN | (Gene, Tissue), (Gene, CellType) | 可写候选 |
| ASSOCIATED_WITH_METABOLITE | (Gene, Metabolite) | 可写候选 |
| PROGRESSES_TO | (Disease, Disease) | 只识别，不自动写入 |
| ENCODES | (Gene, Protein) | 只识别，不自动写入 |

### 2.5 数据流：从 PubMed XML 到 Neo4j

```
PubMed Entrez API
  │  检索: liver disease + HCC + NAFLD + fibrosis + cirrhosis...
  ▼
1,936 篇 PubMed XML (11 批次, 含全文 Abstract)
  │  convert_pubmed_xml_to_jsonl.py
  │  质量筛选: abstract≥200字符 + 英文 + 排除erratum/correction
  ▼
pubmed_converted_500.jsonl (当前 A/B 取前 30 篇；历史基准取 500 篇)
  │  multi_stage_extraction_pipeline.py
  │  DeepSeek API × 30 次调用（当前 A/B）
  ▼
extraction_results.json (当前 118 条候选关系；历史 2,029 条)
  │  entity_linking_preflight.py
  │  Ensemble Matcher vs Neo4j (836 Gene, 793 Protein, 5 Disease...)
  ▼
import_candidates (估计 60-120 条可安全写入)
```

---

## 3. 开发历程与问题解决

### 3.1 开发迭代历程

系统开发经历了至少 10 轮迭代：

```
dryrun → v1 → v2 → v3 → v4 → v5 (schema-compliant 系列)
api_probe_original → api_probe_quality_patch → api_probe_quality_patch_v2 (API 探测系列)
pubmed_quality_test (10 篇 PubMed 测试)
mock_after_quality_patch (模拟数据测试)
```

关键转折点出现在 **从"宽松"到"保守"的导入策略转变**：早期版本允许 LLM 创建 `Article`、`LLMEntity` 等自定义节点标签，导致 Neo4j 中出现不可控的新节点类型。后续改为保守模式——不创建新实体节点，只在已有核心节点之间写入候选关系。

### 3.2 遇到的五个核心问题

#### 问题 1：LLM 输出不受 Schema 约束

**现象**：DeepSeek 频繁输出 `Protein → Disease`、`Pathway → Disease`、`CellType → Disease` 等关系类型，这些不在 8 种目标签名中。即使 prompt 明确列出了允许的签名，模型仍倾向于输出它认为「生物学上正确」的关系。

**解决**：Stage 2 引入 `RELATION_SIGNATURES` 硬校验——任何不在白名单中的 (subject_type, predicate, object_type) 组合直接被标记为 `schema_mismatch` 并拒绝。这导致 894 条关系（44%）被拒绝。

**启示**：对于 schema-constrained extraction，**out-of-band validation is non-negotiable**。Prompt 指令是软约束，代码校验是硬约束。

#### 问题 2：实体类型系统混淆（Drug → Metabolite, Therapy → Protein）

**现象**：LLM 将 "adjuvant"、"CAR-T cell therapy"、"immune checkpoint inhibitors"、"Hedyotis diffusa"（中草药）等干预/治疗概念错误分类为 Metabolite、Protein 或 CellType，仅仅因为需要把关系塞进某个允许的 schema 槽位。

**解决**：引入 `SCHEMA_FORCED_ENTITY_CUES` 字典——当实体的 mention 包含 "therapy"、"drug"、"inhibitor"、"extract"、"decoction" 等词汇时，标记为 `unsupported_entity_class`。在 prompt 中增加明确的负面指令。

#### 问题 3：实体链接是真正的瓶颈

**现象**：即使一条关系通过了所有校验（`import_ready=True`），到了 entity_linking_preflight 阶段仍然会失败，因为端点实体根本不在 Neo4j 中。

实测数据：
- Preflight on schema-compliant v5 (25 实体, 14 关系): 14/25 实体链接, 1/14 两端可链接, **0 条 import_candidate**
- Preflight on PubMed 10 篇测试 (82 实体, 74 关系): 26/82 实体链接, 2/74 两端可链接, **0 条 import_candidate**
- 正例控制 (人工构造 TP53→HCC): **1/1 import_candidate** ✅ — 证明 Gatekeeper 逻辑正确

**原因**：Neo4j Disease 只有 5 个节点（Healthy liver, NAFLD, NASH, Fibrosis, Cirrhosis, HCC），而 PubMed 文章中的 Disease 对象远超这个范围。Metabolite 只有 36 个，PubMed 中出现大量非核心代谢物。

**解决**：这是系统性问题。根因在于上游 PubMed 采样策略与下游 KG 节点覆盖之间存在 mismatch。这直接促成了第二代架构中"主动创建实体"的设计决策。

#### 问题 4：Gene vs Protein 边界模糊

**现象**：同一分子（如 EGFR、AKT1、NFE2L2/Nrf2）在 LLM 输出中时而标 Gene、时而标 Protein。`Gene → Disease` 是允许的，但 `Protein → Disease` 不是。导致 240 条关系被拒绝。

**部分解决**：entity_linking_preflight 的 SchemaRepairCritic 会检查：如果被拒绝的 `Protein → Disease` 关系中，同名 Gene 在 Neo4j 中存在，则提议修正为 `Gene → Disease`。

#### 问题 5：PubMed 文章类型与研究范式错位

**现象**：500 篇文章内容分类：

| 类型 | 占比 | 与 molecular schema 匹配度 |
|---|---|---|
| gene/protein/molecular | 30.4% | ✅ 高 |
| drug/therapy/treatment | 21.0% | ⚠️ 低 (产生 TREATS/TARGETS) |
| review/bibliometric | 11.0% | ❌ 无原始实验证据 |
| herbal/TCM | 4.8% | ❌ 不匹配 |
| imaging/diagnosis | 8.2% | ❌ 不匹配 |

**根因**：PubMed 检索关键词（liver disease + HCC + NAFLD + fibrosis + cirrhosis）会召回大量治疗/药物/综述研究，65% 的文章与 molecular KG 存在根本性 mismatch。

### 3.3 解决过程总结

```
迭代 1-3:    调 prompt → 改善不大 (LLM 不受控)
迭代 4-5:    加 Stage 2 硬校验 → import-ready 从 40% 降到 23% (更严格)
迭代 6-7:    加 unsupported_entity_class 检测 → 减少错误分类
迭代 8-9:    加 entity_linking_preflight → 发现真正的瓶颈
迭代 10:    跑 500 篇全量 → 获得统计意义的基线数据
```

核心理念演变：
```
"让 LLM 抽取得更多" → "让 LLM 抽取得更准" → "理解为什么抽取得对的东西也进不了库"
```

---

## 4. 创新点

### 4.1 Schema-Constrained Multi-Stage Extraction with Out-of-Band Validation

区别于大多数文献抽取系统依赖 prompt engineering 做 schema 约束，本系统采用**双层约束架构**：

- **Soft constraint (Stage 1)**：prompt 中的 Few-shot 示例 + 允许签名列表
- **Hard constraint (Stage 2)**：15+ 项代码层校验，任何不满足的直接标记为 review

这种设计使得即使 LLM 输出不稳定，系统输出仍然可控。代价是 import-ready 比例偏低（23%），但比让错误数据进入 KG 好。

### 4.2 Evidence-Grounded Extraction with Character-Level Provenance

每条关系必须通过**证据自检**：系统在原文中做子串匹配定位证据句的字符偏移量。如果证据句无法在原文中找到（LLM 幻觉或改写），关系被标记为 `ungrounded_evidence` 并拒绝。

```
evidence: "TP53 mutations are strongly associated with hepatocellular carcinoma progression."
           ↑                                                                    ↑
     char_start = 342                                                    char_end = 434
```

### 4.3 Decoupled Extraction–Linking Architecture

将知识抽取（LLM 密集型、高成本）和实体链接（数据库查询、低成本）拆分为两个独立脚本：

| 维度 | Extraction Pipeline | Linking Preflight |
|---|---|---|
| LLM 调用 | 500 次 × ~3s | 0 次 |
| Neo4j 查询 | 0 次（skip模式） | N 次（可缓存） |
| 单次运行时间 | 30-40 分钟 | < 30 秒 |
| 迭代成本 | 高 (API 费用) | 零 |

### 4.4 Ensemble Entity Linking

6 个专业化匹配器组成的 ensemble：Exact (ID) → Lexical (文本) → Alias (领域知识) → Schema Repair (类型层面) → Dense Retriever (语义层面, 待实现) → Gatekeeper。

### 4.5 Conservative Import Policy with Full Provenance

写入 Neo4j 的关系携带 12 个元数据字段，所有关系以 `validation_status='candidate'` 写入，明确标注为 LLM 候选知识而非 curated fact。

---

## 5. 30 篇 A/B 实际执行结果

### 5.1 实验设置

本次实跑使用同一份输入文件的前 30 篇文献，关闭 Neo4j 写入，只比较两代系统在 dry-run/offline write 条件下的输出路径。

| 条件 | 设置 |
|---|---|
| 输入文件 | `extraction_output/pubmed_converted_500.jsonl` |
| 样本 | 前 30 篇 |
| 第一代 run id | `v1_pipeline_pubmed30_20260629_155656` |
| 第二代 run id | `v2_agent_pubmed30_20260629_155656` |
| Neo4j 写入 | 关闭 |
| Neo4j 读取 | 未连接（本机未设置 `NEO4J_PASSWORD`） |
| 输出汇总 | `extraction_output/abtest_compare_20260629_155656.md` |

### 5.2 第一代 Pipeline 结果

第一代 pipeline 完成 30/30 篇抽取，使用 `LLM_API_KEY` 映射为 `DEEPSEEK_API_KEY` 后真实调用模型，没有进入 mock extraction。

| 指标 | 数值 | 解释 |
|---|---:|---|
| 输入文章 | 30 | 同一输入文件前 30 篇 |
| 产出实体 | 200 | 6.7/篇 |
| 产出关系 | 118 | 3.9/篇 |
| Schema-valid 关系 | 67 | 56.8% of relations |
| Import-ready 关系 | 37 | 31.4% of relations |
| Review 关系 | 81 | 68.6% of relations |
| 零关系文章 | 1 | 3.3% |
| 至少 1 条关系 | 29 | 96.7% |
| 至少 3 条关系 | 20 | 66.7% |
| 总运行时间 | 168.5s | 约 5.6s/篇 |

本次 30 篇里，第一代的 import-ready rate 比历史 500 篇 baseline 更高（31.4% vs 22.9%），主要说明这 30 篇样本与目标 schema 的匹配程度更好，不能直接外推为全量 corpus 的提升。

### 5.3 第一代拒绝原因分布（N=81 条 review 关系）

| 拒绝 flag | 次数 | 解释 |
|---|---:|---|
| `non_human_or_mixed_species` | 32 | 非人类/混合物种证据，合理拦截 |
| `uncertain_relation` | 17 | 证据表达不确定，合理进入 review |
| `schema_mismatch:Protein->Disease` | 13 | 旧 schema 过窄，第二代 42 对签名可覆盖 |
| `schema_mismatch:Disease->Disease` | 11 | 疾病-疾病关联/进展关系需要更细分谓词 |
| `schema_mismatch:Pathway->Disease` | 9 | 旧 schema 过窄，第二代已支持 |
| `not_importable_policy` | 6 | 保守导入策略拒绝 |
| `schema_mismatch:Protein->Tissue` | 3 | 端点类型不在第一代目标签名内 |
| `unsupported_entity_class` | 3 | 实体类型不属于导入策略 |

这组 flag 继续支持前面的架构判断：第一代不是“抽不出”，而是大量候选关系被保守 schema 和 endpoint policy 拦在 review 层。

### 5.4 第二代 Agent 结果与认证失败说明

第二代 agent 命令完整执行到 30/30 篇并生成报告，但每篇都在 LangExtract/Gemini provider 认证阶段失败，错误类别一致：`Gemini API 401 Invalid token`。

| 指标 | 数值 | 解释 |
|---|---:|---|
| 尝试文章 | 30 | 同一输入文件前 30 篇 |
| 成功抽取文章 | 0 | 30 篇均在 provider 认证层失败 |
| 错误数 | 30 | 全部为 Gemini token 认证失败 |
| 产出实体 | 0 | 不是模型质量结论 |
| 产出关系 | 0 | 不是 agent 质量结论 |
| Import-ready | 0 | 因 extraction 未成功 |
| 总运行时间 | 120.2s | 约 4.0s/篇，主要是失败重试等待 |

关键解释：项目当前 `.env` 只有 `LLM_API_KEY`，没有独立 `GEMINI_API_KEY`。该 key 可用于第一代 DeepSeek-compatible 调用，但本次不能用于第二代 Gemini-compatible proxy。因此，本次 30 篇 A/B 只能作为：

1. 第一代 30 篇真实 baseline；
2. 第二代命令链路、日志、报告落盘路径验证；
3. 第二代 provider 配置问题记录。

它不能作为“第二代抽取质量低于第一代”的证据。要形成公平质量对比，需要补充一个对 `GEMINI_API_BASE=https://new.bitexingai.com` 和 `GEMINI_MODEL=[按次]gemini-2.5-flash` 有效的 `GEMINI_API_KEY` 后重跑第二代。

### 5.5 输出文件

| 文件 | 用途 |
|---|---|
| `extraction_output/extraction_results_v1_pipeline_pubmed30_20260629_155656.json` | 第一代逐篇抽取结果 |
| `extraction_output/quality_report_v1_pipeline_pubmed30_20260629_155656.json` | 第一代质量汇总 |
| `extraction_output/provenance_report_v1_pipeline_pubmed30_20260629_155656.html` | 第一代证据溯源 HTML |
| `logs/v1_pipeline_pubmed30_20260629_155656.log` | 第一代运行日志 |
| `extraction_output/agent_results_v2_agent_pubmed30_20260629_155656.json` | 第二代逐篇失败记录 |
| `extraction_output/agent_report_v2_agent_pubmed30_20260629_155656.json` | 第二代失败汇总 |
| `logs/v2_agent_pubmed30_20260629_155656.log` | 第二代运行日志 |
| `extraction_output/abtest_compare_20260629_155656.md` | 本次 A/B 统一对比页 |

### 5.6 历史 500 篇结果的定位

历史 500 篇第一代实验仍有参考价值，尤其用于说明 broad PubMed sampling 下的长期瓶颈：3,224 实体、2,029 关系、1,135 schema-valid、465 import-ready、327 篇零产出。但从本版报告开始，主实验口径切换为 30 篇 A/B；500 篇结果只作为历史背景和架构动机，不再作为当前两代公平比较的主表。

### 5.7 正例控制验证

人为构造 `TP53(Gene) → HCC(Disease)` 正例输入 preflight：

```
✅ Gatekeeper: PASS → import_candidate = true
```

证明系统的 Gatekeeper 逻辑是正确的——当端点真实存在、schema 匹配、质量合格时，系统能正确放行。

---

## 6. 瓶颈与局限

### 6.1 关系 Schema 签名过窄

本次 30 篇中，第一代仍出现 `schema_mismatch:Protein->Disease` 13 次、`schema_mismatch:Pathway->Disease` 9 次、`schema_mismatch:Disease->Disease` 11 次。历史 500 篇中同类问题规模更大（Protein→Disease ~240、Pathway→Disease ~100），说明这不是偶然样本噪声，而是第一代 schema 过窄造成的系统性 review 压力。

### 6.2 Neo4j 节点覆盖不足

| 节点类型 | 当前数量 | 30 篇实验中的表现 | 说明 |
|---|---:|---|---|
| Disease | 5 | 出现 Disease→Disease、Protein→Disease、Pathway→Disease 等 review flags | 目标疾病骨架过窄，无法覆盖 HBV/HCV、其他器官纤维化等 broad PubMed 内容 |
| Metabolite | 36 | 可抽取但端点覆盖有限 | 代谢物空间远大于当前 KG |
| Gene/Protein | 836 / 793 | 可抽取实体较多，但 mention→canonical ID 仍需 linking | 仍需要 NCBI、alias、dense retriever 辅助标准化 |

即使一条关系通过所有校验，如果疾病对象不在 Neo4j 中，仍然无法入库。这是促成第二代系统"主动创建实体"策略的根本原因。

### 6.3 PubMed 采样策略

本次前 30 篇只有 1 篇零关系文章，但 review flags 中仍有 32 次 `non_human_or_mixed_species`。这说明 30 篇局部样本比历史 500 篇更“可抽取”，但 broad liver disease 检索仍会混入动物实验、其他器官纤维化、病毒性肝炎、影像学筛查等与核心进展链条不完全一致的内容。

### 6.4 缺少语义匹配能力

纯词法/规则匹配对 `Nrf2→NFE2L2`、`hepatic fibrosis↔liver fibrosis` 等语义等价关系无力，需要 BioSyn/SapBERT 等 dense retriever。

### 6.5 架构层面的五个结构性问题

这些瓶颈不仅是参数配置问题，更是架构设计问题：

| # | 瓶颈 | 表现 |
|---|------|------|
| 1 | **无状态** | 每篇文章独立处理。同一实体在第 50 篇中出现时，系统不知道前 49 篇已见过它 |
| 2 | **被动过滤** | 只能拒绝，不能修正或推理。无法尝试"Protein→Disease 可能是 Gene→Disease" |
| 3 | **无因果推理** | 无法从 A→B + B→C 推断 A→C |
| 4 | **无自优化** | Prompt 固定不变。不会因为连续遇到综述文章而调整策略 |
| 5 | **割裂的实体链接** | 抽取阶段不知道 KG 里有什么，大量 effort 浪费在不可链接的实体上 |

这些问题直接促成了 2026-06-28 的架构重构。

---

## 7. 架构重构：从 Pipeline 到 Cognitive Loop

### 7.1 重构动机

第一代系统在历史 500 篇测试中暴露出 broad sampling 与 KG 覆盖不匹配的问题；本次 30 篇实跑又进一步显示，即便样本更可抽取，仍有 81/118 条关系需要 review。更深层的问题在于——Pipeline 模式假设"世界是静态的"，但知识抽取是一个动态认知过程：

- 读到第 10 篇文章时，应该已经学到前 9 篇的模式
- 发现一条矛盾关系时，应该能回溯之前的相关决策
- 连续遇到低质量文章时，应该自动调整提取策略

### 7.2 新架构：7 阶段 Cognitive Agent（v2.2 闭环版）

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Cognitive Agent — 自主认知知识管理 Agent                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PubMed JSONL (title + abstract + pmid)                                    │
│      │                                                                     │
│      ▼                                                                     │
│  ╔══════════════════════════════════════════════════════════════════════╗  │
│  ║  Phase 1: Context Activation — Neo4j 先验知识激活                      ║  │
│  ║    · 查询 KG 中已有实体/关系 → ContextCard                              ║  │
│  ║    · 指导后续提取：已知实体精准匹配，新实体谨慎创建                       ║  │
│  ╚══════════════════════════════════════════════════════════════════════╝  │
│      │                                                                     │
│      ▼                                                                     │
│  ╔══════════════════════════════════════════════════════════════════════╗  │
│  ║  Phase 2: Extract + Ground — LangExtract + Gemini 原生 API            ║  │
│  ║    · provider="gemini" (google-genai SDK)                              ║  │
│  ║    · 6 Few-shot Examples + 类型特定 Schema                              ║  │
│  ║    · 0 实体时自动重试 (MAX_RETRIES=1, exponential backoff)              ║  │
│  ║    · 属性清洗 (过滤 "null" 字符串、空列表、非 dict 关系条目)             ║  │
│  ║    → RawExtraction (entities + relations + warnings)                    ║  │
│  ╚══════════════════════════════════════════════════════════════════════╝  │
│      │                                                                     │
│      ▼                                                                     │
│  ╔══════════════════════════════════════════════════════════════════════╗  │
│  ║  Phase 3: Verify + Reason — 图谱溯源验证 + 因果推理                     ║  │
│  ║    · 实体溯源: EXACT_MATCH / FUZZY_MATCH / NOVEL                        ║  │
│  ║    · 关系 Schema 检查: RELATION_SIGNATURES 硬校验 (42 对签名)             ║  │
│  ║    · 传递推理: A→B (新提取) + B→C (KG已知) ⇒ A→C (推断)                  ║  │
│  ║    → VerifiedExtraction (entities + relations + causal_chains)          ║  │
│  ╚══════════════════════════════════════════════════════════════════════╝  │
│      │                                                                     │
│      ▼                                                                     │
│  ╔══════════════════════════════════════════════════════════════════════╗  │
│  ║  Phase 4: Conflict Resolution — 冲突检测与裁决                          ║  │
│  ║    · DIRECT_CONTRADICTION / EVIDENCE_STRENGTH /                        ║  │
│  ║      METHODOLOGICAL_DIFF / TEMPORAL_DRIFT                               ║  │
│  ║    · 8 行决策表裁决                                                     ║  │
│  ╚══════════════════════════════════════════════════════════════════════╝  │
│      │                                                                     │
│      ▼                                                                     │
│  ╔══════════════════════════════════════════════════════════════════════╗  │
│  ║  Phase 5: Decision — 知识决策引擎                                       ║  │
│  ║    · CREATE_ENTITY / CREATE_RELATION / UPDATE_RELATION                  ║  │
│  ║    · MARK_DISPUTED / PROPOSE_HYPOTHESIS / DISCARD / NO_ACTION          ║  │
│  ║    · NCBI Gene Validator (E-utilities, 无 API key)                      ║  │
│  ║    · 属性过滤: 自动剥离关系属性键 + 嵌套结构                              ║  │
│  ║    → ExecutionLog                                                      ║  │
│  ╚══════════════════════════════════════════════════════════════════════╝  │
│      │                                                                     │
│      ▼                                                                     │
│  ╔══════════════════════════════════════════════════════════════════════╗  │
│  ║  Phase 6: Reflection — 元认知反思 (每 N 篇触发)                         ║  │
│  ║    · 质量指标计算 (实体/关系覆盖率, Schema 合规率)                        ║  │
│  ║    · 阈值自适应调节                                                      ║  │
│  ║    · 跨文档新实体发现 (emerging entities)                                ║  │
│  ╚══════════════════════════════════════════════════════════════════════╝  │
│      │                                                                     │
│      ▼                                                                     │
│  ╔══════════════════════════════════════════════════════════════════════╗  │
│  ║  Phase 7: Adapt — 策略自适应                                            ║  │
│  ║    · 根据 ContextCard 选择探索/聚焦模式                                  ║  │
│  ║    · 影响下一轮 prompt / examples / 创建阈值 / 冲突处理模式              ║  │
│  ║    · 策略快照与回滚                                                      ║  │
│  ╚══════════════════════════════════════════════════════════════════════╝  │
│      │                                                                     │
│      ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  KG Memory (Neo4j) — 外部动态记忆                                      │  │
│  │    · 7 实体类型 (Gene / Disease / Protein / Pathway / Metabolite /    │  │
│  │      Tissue / CellType) · 8 关系谓词 · 类型特定主键 · 全量 CRUD        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

v2.2 之前，`StrategyManager.get_strategy()` 和 `ConflictResolver.resolve()` 已经存在，但更多是报告层面的诊断信息。v2.2 将这两个模块接入主控制流：

```
ContextCard
  → StrategyManager.get_strategy()
  → _build_strategy_prompt() / _select_examples()
  → ExtractionKernel.extract()
  → KGVerifier.verify()
  → ConflictResolver.resolve()
  → DecisionEngine.decide(strategy, conflict_resolution)
  → DecisionEngine.execute()
```

闭环后的行为变化：

| 闭环点 | v2.1 之前 | v2.2 之后 |
|--------|-----------|-----------|
| Strategy 对抽取的影响 | 计算后未真正使用 | `extraction_mode` 进入 prompt；低覆盖时使用 `ALL_EXAMPLES` |
| Strategy 对决策的影响 | 不影响实体/关系创建阈值 | 影响 `entity_confidence_threshold` 与 `relation_confidence_threshold` |
| ConflictResolver | 结果只写入 report | 结果驱动 `CREATE / UPDATE / DISPUTE / DISCARD / KEEP_OLD` |
| UPDATE_RELATION | 只在计数层面出现 | 调用 `KGMemory.update_relation()` 更新证据与置信度 |
| MARK_DISPUTED | 只在计数层面出现 | 调用 `KGMemory.mark_disputed()` 写入争议标记 |
| 自适应回路 | Reflection 更新 StrategyManager 状态，但影响有限 | 下一篇文章会使用更新后的 strategy |

### 7.3 三记忆系统

新系统引入三种记忆，模拟人类认知的三层记忆模型。这是与第一代系统最根本的架构差异——从"无状态函数"变为"有状态的认知实体"。

| 记忆层 | 类比 | 生命周期 | 数据结构 | 职责 |
|--------|------|---------|---------|------|
| **Working Memory** | 阅读单篇论文的即时记忆 | `process_article()` 开始→结束 | 临时缓存 (Neo4j 查询结果 + 提取目标) | 单篇上下文 |
| **Episodic Memory** | 读完一批论文后的交叉理解 | Agent 启动→关闭 | Episode 列表 (所有决策记录) | 跨文档模式发现 |
| **KG Memory** | 领域知识库 / 长期记忆 | 跨 Agent 会话 | Neo4j (7 实体类型 + 8 关系谓词) | 持久化结构化知识 |

### 7.4 实体类型 Schema（7 种）

| extraction_class | Neo4j Label | ID Property | ID Prefix | Name Property | 外部验证 |
|-----------------|-------------|-------------|-----------|---------------|---------|
| gene | Gene | gene_id | NCBIGene | gene_symbol | ✅ NCBI E-utilities |
| disease | Disease | disease_id | PROJECT:Disease | name | ❌ |
| protein | Protein | string_protein_id | PROJECT:Protein | preferred_name | ⚠️ 待实现 |
| pathway | Pathway | pathway_id | PROJECT:Pathway | name | ❌ |
| metabolite | Metabolite | metabolite_id | PROJECT:Metabolite | name | ⚠️ 待实现 |
| tissue | Tissue | tissue_id | PROJECT:Tissue | name | ❌ |
| cell_type | CellType | cell_type_id | PROJECT:CellType | name | ❌ |

### 7.5 关系 Schema（v2 扩展版 42 对签名）

第一代系统 8 种关系谓词、13 对 subject-object 签名 → 第二代仍保留 8 种核心谓词，但将允许的 subject-object 组合扩展为 **42 对签名**。核心变化：

```
ASSOCIATED_WITH: 2→22 对 (新增 Disease→Tissue, Disease→Pathway, Pathway→Tissue,
                         CellType→Disease, Gene→Tissue, Pathway→Pathway, 等)
INTERACTS_WITH:  1→6 对  (新增 Gene→Gene, CellType→CellType, Metabolite→Protein)
PARTICIPATES_IN: 1→4 对  (新增 CellType→Pathway, Metabolite→Pathway)
EXPRESSED_IN:    2→4 对  (新增 Protein→Tissue, Protein→CellType)
PROGNOSTIC_IN:   1→2 对  (新增 Protein→Disease)
ASSOCIATED_WITH_METABOLITE: 1→2 对 (新增 Protein→Metabolite)
```

完整签名矩阵见附录 §12.5。

### 7.6 组件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `cognitive_agent/agent.py` | 681 | 主循环 + CLI + 组件编排 + strategy prompt 注入 |
| `cognitive_agent/extraction_kernel.py` | 268 | LangExtract 封装 + 重试 + 属性清洗 |
| `cognitive_agent/verifier.py` | 246 | 实体溯源 + Schema 检查 + Import-ready |
| `cognitive_agent/decision_engine.py` | 661 | 决策 + strategy 阈值 + conflict resolution 消费 + Neo4j 执行 |
| `cognitive_agent/causal_reasoner.py` | 213 | 传递推理 + 置信度计算 |
| `cognitive_agent/conflict_resolver.py` | 191 | 8 行决策表冲突裁决 |
| `cognitive_agent/self_reflection.py` | 170 | 质量指标 + 阈值自适应 |
| `cognitive_agent/strategy_manager.py` | 177 | 策略快照 + 模式切换 |
| `cognitive_agent/context_activator.py` | 223 | Neo4j 先验知识激活 |
| `cognitive_agent/memory/kg_memory.py` | 687 | Neo4j CRUD + 查询 + 类型特定 Schema |
| `cognitive_agent/memory/working_memory.py` | 60 | 单篇文章临时缓存 |
| `cognitive_agent/memory/episodic_memory.py` | 108 | 跨文章决策记录 |
| `cognitive_agent/schema/entity_classes.py` | ~83 | 7 种实体类型 + 创建策略 |
| `cognitive_agent/schema/relation_signatures.py` | 105 | 42 对关系签名矩阵 |
| `cognitive_agent/schema/examples.py` | 395 | 6 个 Few-shot 示例 |
| `cognitive_agent/tools/ncbi_validator.py` | 236 | NCBI E-utilities 基因验证 |
| `tests/test_cognitive_agent_loop.py` | 198 | v2.2 闭环单元测试 |

**总计：~4,400 行 Cognitive Agent 相关 Python（不含第一代 pipeline 和 workstream）。**

---

## 8. Provider 迁移：从 DeepSeek 到 Gemini 原生的完整旅程

这是整个项目中最复杂、耗时最长的单项问题排查。此处完整记录每一个错误、排查过程和最终解决方案。

### 8.1 起点：DeepSeek → Gemini + bitexingai 代理

第一代使用 DeepSeek Chat API。切换到 LangExtract 后，最初选择 Gemini 模型通过 bitexingai 代理访问（用户提供的 API key 为 `sk-` 格式）。

**初始配置（失败）**：
```python
provider="openai"
model_id="gemini-2.5-flash"
provider_kwargs={
    "api_key": "sk-...",
    "base_url": "https://new.bitexingai.com/v1",
}
```

### 8.2 问题排查时间线

#### 错误 1：400 INVALID_ARGUMENT — "API key not valid"

**错误信息**：Google API 返回 400，声称 `sk-` 格式的 key 无效。

**排查**：`sk-` 是 OpenAI 风格的 key 格式，Google 原生 API 期望 `AIza...` 格式。bitexingai 代理支持 OpenAI 兼容端点（`/v1/chat/completions`），因此需要将 provider 设为 `"openai"` 让 LangExtract 走 OpenAI SDK 路径。

**解决**：`provider="openai"` + `base_url="https://new.bitexingai.com/v1"`

#### 错误 2：503 "No available channel for model gemini-2.5-flash"

**错误信息**：代理返回 `model_not_found`。

**排查**：bitexingai 代理使用自定义模型名。模型需要 `[按次]` 前缀。

**解决**：`model_id="[按次]gemini-2.5-flash"`

#### 错误 3：高频 "Skipping chunk: schema error"

**现象**：每批 3 篇文章中，1-2 篇出现 `Skipping chunk: schema error: Extraction text must be a string, integer, or float`。模型有时能正常输出，有时输出畸形 JSON（如 `associated_with: ['null', 'null']`）。同一篇文章（PMID 41810002）第一次运行 7E，第二次 0E——完全不确定。

**排查过程**：

1. **阅读 LangExtract 源码** — 发现 `suppress_parse_errors` 默认为 True（1.2.0+），错误被静默跳过而非抛出异常
2. **分析具体失败案例** — 畸形输出模式：
   - `"null"` 字符串出现在关系列表中
   - 空 JSON 响应（`char 0`）
   - JSON 被截断（`Unterminated string`）
3. **社区调研** — LangExtract GitHub issues：
   - [#222](https://github.com/google/langextract/issues/222)：Malformed JSON from model
   - [#301](https://github.com/google/langextract/issues/301)：Empty responses at char 0
   - [#287](https://github.com/google/langextract/issues/287)：Truncated output

**根因定位**：`provider="openai"` 时，LangExtract 通过 OpenAI SDK 发送请求。Gemini 的 `response_schema` 功能被代理转换为 prompt 中的文字描述（而非 API 参数），失去了对模型输出的强制约束力。模型有时遵守 schema，有时不遵守。

#### 错误 4：`Episode.__init__() got an unexpected keyword argument 'reason'`

**现象**：Agent 启动后立即崩溃，所有文章处理失败。

**排查**：`agent.py` 创建 Episode 时传参 `reason=action.reason`，但 `Episode` dataclass 的字段名为 `reasoning`。

**解决**：一行修改 — `reason=action.reason` → `reasoning=action.reason`

#### 缓解措施（在切换到原生 Gemini 之前）

在理解根因但尚未解决之前，实施了 4 层缓解：

1. **重试机制** — `MAX_RETRIES=3`，exponential backoff。0 实体时自动重试。实测：5 篇中 3 篇在重试后成功。
2. **属性清洗** — `_parse()` 中过滤 `"null"` 字符串、None、空 dict。减少 schema error。
3. **宽松 Schema** — `use_schema_constraints=False`，移除 `resolver_params`。减少 fuzzy alignment 导致的 chunk 丢弃。
4. **降低并发** — `max_workers=2`（从 8），减少代理压力。

**缓解效果**（3 篇文章测试）：
- 提取质量提升：+76% 实体，+34% 关系
- Import-ready 比例：从 11% → 79.5%
- 但根本问题未解决：Gemini 仍然可能输出畸形 JSON

### 8.3 关键洞察：切换到原生 Gemini Provider

用户的反馈："就用原生的gemini啊，我用的也是gemini的模型"。

bitexingai 代理同时支持 `["gemini", "openai"]` 两种端点类型。这意味着可以直接使用 LangExtract 的原生 Gemini provider，让 `google-genai` SDK 直接与代理通信。

**协议层面差异**：
```
OpenAI 兼容模式:
  请求 → JSON body {messages: [...], response_format: {...}}
       → 代理转换为 Gemini 格式
       → response_schema 被嵌入 prompt 文字 → 软约束 → 模型可能忽略

原生 Gemini 模式:
  请求 → google-genai SDK → Gemini API {contents: ..., response_schema: {...}}
       → 代理直接转发
       → response_schema 作为 API 参数 → 硬约束 → 模型强制遵守
```

关键不在代理本身（代理支持两种协议），而在 **SDK 如何将 schema 传递给模型**。

### 8.4 迁移过程中的三次尝试

#### 尝试 1：`provider="gemini"` + `use_schema_constraints=True` ❌

```python
lx_config = ModelConfig(
    provider="gemini",
    model_id="[按次]gemini-2.5-flash",
    provider_kwargs={
        "api_key": "sk-...",
        "http_options": {"base_url": "https://new.bitexingai.com"},
        "temperature": 0.0,
    },
)
lx.extract(use_schema_constraints=True)
```

**结果**：57 实体 / **0 关系** / 235s/篇

**原因**：Gemini 原生 `response_schema` 的严格约束与 LangExtract 内部生成的 schema 结合后，对嵌套关系属性的限制过于死板。模型无法在严格 schema 内表达关系，选择了不输出关系。

#### 尝试 2：`provider="gemini"` + `use_schema_constraints=False` ✅

**结果**：76 实体 / **51 关系** / ~30s/篇 / **0 错误**

**原因**：即使不开 `use_schema_constraints`，原生 Gemini 协议的内置输出控制也远超 OpenAI 兼容模式。模型从 Few-shot 示例学习输出格式，原生协议确保一致性。**"Skipping chunk" 从此完全消失。**

#### 尝试 3：`[按次]gemini-3-flash-preview` ❌

**结果**：模型完全忽略 `response_schema`，返回自然语言而非 JSON

**结论**：bitexingai 代理的 `gemini-3-flash-preview` 端点不支持结构化输出。只有 `gemini-2.5-flash` 可用。

### 8.5 最终配置

```python
# agent.py — ModelConfig
lx_config = ModelConfig(
    provider="gemini",                              # 原生 Google genai SDK
    model_id="[按次]gemini-2.5-flash",               # 代理模型名（必须含前缀）
    provider_kwargs={
        "api_key": "sk-...",                         # bitexingai proxy key
        "http_options": {"base_url": "https://new.bitexingai.com"},
        "temperature": 0.0,
    },
)

# extraction_kernel.py — lx.extract()
lx.extract(
    text_or_documents=[doc],
    prompt_description=prompt,
    examples=examples,                               # 4 个核心 Few-shot
    config=self.model_config,
    temperature=0,
    max_workers=2,                                   # 低并发保稳定
    use_schema_constraints=False,                    # 原生协议已足够
    show_progress=False,
    extraction_passes=1,
)
```

### 8.6 代理下原生 Gemini API 的已知限制

| 项目 | 状态 | 说明 |
|------|------|------|
| `response_schema` 支持 | ⚠️ 部分 | gemini-2.5-flash ✅ / gemini-3-flash-preview ❌ |
| `use_schema_constraints=True` | ❌ 不推荐 | 消灭关系 (0R) |
| 模型名格式 | 需 `[按次]` 前缀 | 无前缀 → 503 |
| `base_url` 格式 | 不含 `/v1` | SDK 自动追加 `/v1beta/models/...` |
| `max_workers` | ≤ 2 | 高并发触发限流 |
| 非确定性 | 已消除 | 原生协议 + `temperature=0` |

---

## 9. Cognitive Agent 开发全程问题排查

### 9.1 问题全景图

本节按发现顺序，记录 Cognitive Agent 开发过程中遇到的全部问题及其解决过程。

### 9.2 API 与连接层（5 个问题）

| # | 问题 | 错误表现 | 根因 | 解决方案 | 章节 |
|---|------|---------|------|---------|------|
| 1 | API key 被拒 | 400 INVALID_ARGUMENT | `sk-` 格式不兼容 Google | `provider="openai"` + proxy | §8.2 |
| 2 | 模型不可用 | 503 model_not_found | 代理需要 `[按次]` 前缀 | `model_id="[按次]gemini-2.5-flash"` | §8.2 |
| 3 | Skipping chunk | 畸形 JSON 被静默丢弃 | 代理不支持 `response_schema` | 切换到原生 Gemini provider | §8.2-8.4 |
| 4 | EpisodicMemory 崩溃 | `unexpected keyword argument 'reason'` | 字段名不匹配 | `reason=` → `reasoning=` | §8.2 |
| 5 | base_url 路径重复 | 404 `/v1/v1beta/...` | SDK 自动追加路径 | `base_url` 去掉 `/v1` 后缀 | §8.4 |

### 9.3 提取质量层（3 个问题）

#### 问题 6：非确定性输出

**现象**：同一篇文章 PMID 41810002，第一次 7E，第二次 0E。

**排查**：OpenAI 兼容模式下，`response_schema` 被转为 prompt 文字。模型生成的 JSON 结构有时不符合 LangExtract 内部 parser 的期望，被静默丢弃。

**解决**：切换到原生 Gemini（§8.4）+ 添加 retry 机制作为安全网。原生协议下非确定性完全消除。

#### 问题 7：0 关系产出

**现象**：`use_schema_constraints=True` 时，所有文章关系数 = 0。

**排查**：LangExtract 内部生成的 GeminiSchema 与示例中的嵌套关系属性不完全匹配。严格约束下，模型选择放弃输出关系而非冒险输出不合规 JSON。

**解决**：`use_schema_constraints=False`（参见 §8.4 尝试 2）。原生 Gemini 即使不开 schema constraints，输出一致性也远超 OpenAI 兼容模式。

#### 问题 8：gemini-3-flash-preview 无法结构化输出

**现象**：返回自然语言 "Here are the entities..." 而非 JSON。

**排查**：向代理发送 `response_schema` 参数时，gemini-3-flash-preview 端点忽略该参数。

**解决**：切换为 `[按次]gemini-2.5-flash`。速度稍慢但功能完整。

### 9.4 Schema 兼容层（1 个核心问题 + 修复）

#### 问题 9：RELATION_SIGNATURES 过窄（56% 关系被误杀）

**现象**：3 篇文章 43 条关系中，只有 19 条 (44%) 通过 schema 校验。24 条 (56%) 因签名不匹配被拒绝。

**被拒绝的高频关系**：
- Disease → Tissue (如 "HCC affects tumor microenvironment")
- Disease → Pathway (如 "MASLD involves ferroptosis")
- Pathway → Tissue, Tissue → Disease, CellType → Pathway...

**根因**：初始 8 种签名设计过于保守，ASSOCIATED_WITH 仅覆盖 4 对。而生物医学文献中实体间的关联远比此丰富。

**解决（2026-06-29，后续 v2.2 校准）**：将签名矩阵从 8 种 13 对扩展到 8 种 **42 对**。详见 §7.5。

**修复验证**：
```
修复前: 19/43 合规 (44%)
修复后: 42/43 合规 (97%)
唯一拒绝: CellType -[EXPRESSED_IN]-> Tissue (模型方向搞错，正确拒绝)
```

### 9.5 Neo4j 数据完整性层（3 个问题 + 修复）

#### 问题 10：实体属性在决策阶段全部丢失

**问题链路**：

```
Extraction Kernel         →  Verifier            →  Decision Engine      →  KG Memory
attributes: {             VerifiedEntity:           execute():
  disease_name: "HCC",    mention: str              create_entity(
  disease_stage: "HCC"    entity_type: str            properties={}  ← 空!
}                         # ❌ 无 attributes 字段      )
```

**三重断链**：
1. `VerifiedEntity` dataclass 没有 `attributes` 字段
2. `_decide_entity()` 只组装 `mention + type + validated_props`
3. `execute()` 硬编码 `properties={}`

**影响**：写入 Neo4j 的所有节点都缺少类型特定属性（`disease_name`、`gene_symbol`、`pathway_name` 等）。

**解决（2026-06-29）**：
- `VerifiedEntity` 添加 `attributes: dict` 字段
- `_verify_entity()` 保留原始 `entity.get("attributes", {})`
- `_decide_entity()` 合并 `entity.attributes` 到 `entity_props`
- `execute()` 传递过滤后的 `entity_props` 给 `create_entity(properties=...)`

#### 问题 11：跨文章实体重复创建

**现象**：同一概念 "Hepatocellular carcinoma" 在两篇文章中创建了两个独立 Neo4j 节点。

**根因**：`create_entity()` 的 ID 生成包含 `pmid`：
```python
id_base = f"{label}:{name}:{pmid or 'agent'}"  # 不同 pmid 产生不同 hash
```

**解决**：移除 `pmid`。ID 仅依赖 `label:name`。`ON MATCH SET` 追加 evidence 即可区分来源。
```python
id_base = f"{label}:{name}"  # 同一概念跨文章共享 ID
```

#### 问题 12：Neo4j 收到嵌套结构（关系属性未过滤）

**现象**：实体属性中的 `associated_with`、`encodes` 等关系列表（数组 + 嵌套 dict）会被传入 `create_entity()`。Neo4j 不支持嵌套结构，会抛出类型错误。

**解决**：在 `kg_memory.py` 添加：
```python
RELATION_ATTRIBUTE_KEYS: frozenset[str] = frozenset({
    "associated_with", "encodes", "participates_in",
    "interacts_with", "expressed_in", "prognostic_in",
    "progresses_to", "associated_with_metabolite",
})

# create_entity() 步骤 6:
for k, v in properties.items():
    if k in self.RELATION_ATTRIBUTE_KEYS:  # 剥离关系属性
        continue
    if isinstance(v, (list, dict)):        # 剥离任何嵌套结构
        continue
    if k not in skip_keys and k not in props:
        props[k] = v
```

### 9.6 修复验证：3 篇文章全流程测试

```
测试日期: 2026-06-29
配置: provider="gemini", gemini-2.5-flash, use_schema_constraints=False
修复内容: §9.4-9.5 的全部 4 项修复
```

| 指标 | 修复前 | 修复后 | 变化 |
|------|--------|--------|------|
| 实体/篇 | 29.7 | 29.7 | — |
| 关系/篇 | 14.3 | 14.3 | — |
| Schema 合规率 | 44% | **97%** | +120% |
| Import-Ready | 19 (44.2%) | **35 (81.4%)** | +84% |
| 丢弃率 | 55.8% | **18.6%** | -67% |
| 质量分 | 0.31 | **0.58** | +87% |
| 错误数 | 0 | 0 | 保持 |
| 零产出文章 | 0/3 | 0/3 | 保持 |

### 9.7 v2.2 闭环修复：Strategy 与 Conflict 真正进入主流程

v2.1 之后系统虽然已经拥有 `StrategyManager`、`SelfReflection`、`ConflictResolver` 等模块，但它们与主流程的连接仍不完整：

| 模块 | v2.1 状态 | 问题 |
|------|-----------|------|
| `StrategyManager.get_strategy()` | 每篇文章都会计算 strategy | 结果没有改变 prompt、examples 或创建阈值 |
| `SelfReflection.apply_update()` | 可以更新 StrategyManager 状态 | 更新后的状态对下一篇文章影响有限 |
| `ConflictResolver.resolve()` | 生成 ResolutionResult 并写入 record | DecisionEngine 不消费该结果 |
| `UPDATE_RELATION` / `MARK_DISPUTED` | 决策阶段可计数 | execute 阶段未真正调用 Neo4j 更新/争议标记 |

v2.2 修复后形成完整闭环：

```python
context_card = self.context_activator.activate(text, pmid=pmid)
strategy = self.strategy_manager.get_strategy(context_card)
extraction_examples = self._select_examples(strategy)
extraction_prompt = self._build_strategy_prompt(KG_EXTRACTION_PROMPT, context_card, strategy)

raw_extraction = self.extraction_kernel.extract(
    text=text,
    document_id=pmid,
    examples=extraction_examples,
    prompt=extraction_prompt,
)

verified = self.verifier.verify(...)
resolution = self.conflict_resolver.resolve(verified)
execution_log = self.decision_engine.decide(
    verified_entities=verified.entities,
    verified_relations=verified.relations,
    pmid=pmid,
    strategy=strategy,
    conflict_resolution=resolution,
)
execution_log = self.decision_engine.execute(execution_log)
```

关键行为：

1. **低 KG 覆盖时进入 exploratory mode**：使用全部 6 个 examples，降低实体创建阈值，prompt 明确鼓励提取文本中有证据的新实体。
2. **高 KG 覆盖时进入 focused mode**：prompt 强调优先已知实体之间的关系、避免 broad background association。
3. **ConflictResolver 驱动 DecisionEngine**：`CREATE`、`CREATE_WITH_FLAG`、`UPDATE`、`DISPUTE`、`KEEP_OLD`、`DISCARD` 不再只是报告标签。
4. **Neo4j 执行层支持更新/争议**：`UPDATE_RELATION` 调 `KGMemory.update_relation()`；`MARK_DISPUTED` 调 `KGMemory.mark_disputed()`。
5. **新增闭环单元测试**：`tests/test_cognitive_agent_loop.py` 覆盖 strategy 选择、prompt 注入、阈值改变、conflict resolution 驱动决策、update/dispute 执行。

验证结果：

```text
compileall: OK
root cognitive agent tests: 6 OK
workstream tests: 28 OK
agent CLI --help: OK
./run_cognitive_agent.sh 0: OK
```

---

## 10. 两代系统全面对比

### 10.1 架构对比

| 维度 | 第一代 (Pipeline) | 第二代 (Cognitive Agent) |
|------|-------------------|-------------------------|
| **范式** | 线性 5-Stage Pipeline | 7-Phase Closed-loop Cognitive Agent |
| **LLM 引擎** | DeepSeek Chat API | LangExtract + **Gemini 原生 API** |
| **API 协议** | 原生 OpenAI 兼容 | **google-genai SDK** (原生 Gemini) |
| **Schema 约束** | Prompt 软约束 + 15+ 规则硬校验 | Gemini 原生协议 + 42 对签名硬校验 |
| **实体策略** | 保守 — 不创建新节点 | 主动创建 — 类型特定主键 + MERGE |
| **实体类型** | 依赖已有 KG 节点 | 7 种独立实体类型 |
| **关系签名** | 8 种 13 对 | 8 种 **42 对** |
| **重试机制** | ❌ 单次调用 | ✅ 0 实体自动重试 (exponential backoff) |
| **因果推理** | ❌ | ✅ 传递推理 A→B + B→C ⇒ A→C |
| **冲突解决** | 简单标记 | ✅ 4 类冲突 + 8 行决策表，结果驱动 DecisionEngine |
| **自我反思** | ❌ | ✅ 每 N 篇质量分析 + 阈值自适应 |
| **策略自适应** | ❌ | ✅ 探索/聚焦模式切换，并影响 prompt/examples/阈值 |
| **记忆系统** | 无状态 | 三记忆系统 (Working + Episodic + KG) |
| **外部验证** | ❌ | ✅ NCBI E-utilities 基因验证 |
| **属性完整性** | 12 元数据字段 | 12 元数据 + 类型特定属性 |
| **并发** | 同步单线程 | 可配置 max_workers |
| **Few-shot** | 2 个固定示例 | 6 个示例 (4 核心 + 2 补充) |
| **代码规模** | ~2,600 行核心脚本 (2 文件) | ~4,400 行 Agent 相关代码 + 闭环测试 |

### 10.2 质量指标对比

| 指标 | 第一代 Pipeline（30 篇实跑） | 第二代 Agent（30 篇实跑） |
|------|---------------:|-----------------:|
| 尝试文章 | 30 | 30 |
| 成功抽取文章 | 30 | 0 |
| 实体/篇 | 6.7 | 0 |
| 关系/篇 | 3.9 | 0 |
| Schema-valid 关系 | 67 | n/a |
| Schema 合规率 | 56.8% | n/a |
| Import-ready 关系 | 37 | 0 |
| Import-ready/篇 | 1.23 | 0 |
| Review/Discard | 81 review | 0 discard |
| 错误数 | 0 | 30 |
| 主要失败/拦截原因 | schema/review flags | Gemini API 401 Invalid token |

> **解释**：本表是 2026-06-29 的实际 30 篇执行结果。第二代的 0 实体/0 关系来自 provider 认证失败，不是抽取策略或 agent 架构失败。因此，本次只能确认第一代 30 篇 baseline 和第二代运行链路；要比较两代输出质量，必须先配置有效 `GEMINI_API_KEY` 后重跑第二代。

### 10.3 设计理念的演变

```
第一代: "如何在已有 KG 框架内安全地添加知识"
        → 保守导入、硬校验、宁可漏掉不可写错

第二代: "如何让 Agent 像研究员一样阅读和积累知识"
        → 主动创建、记忆系统、认知循环
```

### 10.4 对比实验应关注的指标

两代系统输出结构不同，不能只比较“抽到了多少实体/关系”。建议将指标分成四层：

| 层级 | 指标 | 解释 |
|------|------|------|
| 抽取产出 | entities/article, relations/article | 粗略衡量召回能力，但会受抽取粒度影响 |
| Schema 质量 | schema_valid_rate, import_ready_rate | 衡量输出是否符合目标 KG 关系签名 |
| 审核负担 | review_rate, discard_rate, flag_counts | 衡量人工审核压力 |
| 运行可靠性 | error_count, zero_output_articles, avg_time_per_article | 衡量批处理稳定性与成本 |
| KG 价值 | create/update/dispute counts, evidence quality | 第二代特有，衡量是否能主动构建知识 |

30 篇 A/B 测试的目标不是证明第二代“数量更大”，而是回答三个更具体的问题：

1. 同样 30 篇文章，哪一代的 **schema 合规率** 更高？
2. 哪一代产生的 **import-ready / create-ready** 候选更多？
3. 第二代增加的实体创建、冲突处理和自适应策略，是否带来更低的零产出率和更高的可审核价值？

---

## 10.5 Neo4j 写入验证与质量控制

> **测试日期**: 2026-06-29  
> **Neo4j 数据库**: liver-kg-core-v02 (`bolt://100.104.181.96:7687`)  
> **测试规模**: 分阶段验证——3 篇初始测试 → 10 篇质量控制测试  

前文 §12.2 报告的 50 篇离线测试取得了 86.5% Import-Ready 率，但当时 Neo4j 不可用，所有写入均在 `--skip-neo4j-write` 模式下运行。§12.2 中的 "创建实体/关系" 数据来自**决策阶段计数**（`_decide_entity()` 中的 `log.entities_created += 1`），而非实际 Neo4j 写入结果。Neo4j 恢复连接后，我们进行了系统的写入验证测试。

### 10.5.1 关键 Bug 发现与修复

首次 Neo4j 写入测试（2026-06-29）发现了三个关键缺陷：

#### Bug 1: `execute()` 从未被调用

**症状**: 报告显示实体和关系已创建，但 Neo4j 中无任何写入。

**根因**: `agent.py` 的 `process_article()` 方法调用了 `self.decision_engine.decide()`（生成决策），但从未调用 `self.decision_engine.execute()`（执行写入）。`decide()` 返回的 `ExecutionLog` 中包含 `entities_created` / `relations_created` 计数，但这些只是 DECISION 阶段的计划，并未真正写入 Neo4j。

```python
# agent.py (修复前)
execution_log = self.decision_engine.decide(...)  # 仅决策，不写入
record["phases"]["execution"] = execution_log.to_dict()

# agent.py (修复后)
execution_log = self.decision_engine.decide(...)
execution_log = self.decision_engine.execute(execution_log)  # ← 新增：真正写入
record["phases"]["execution"] = execution_log.to_dict()
```

#### Bug 2: 关系写入未实现

**症状**: 实体写入修复后，关系仍然为零。

**根因**: `decision_engine.py` 的 `execute()` 方法中，`CREATE_RELATION` 分支仅包含 `pass` 语句，注释写道 "在实际实现中：查找两端实体 element_id → 创建关系"。

**修复**: 实现完整的关系创建流程：
1. 先执行所有 `CREATE_ENTITY`（Phase 1），确保实体已存在于 Neo4j
2. 再执行 `CREATE_RELATION`（Phase 2），通过 `find_entity()` 查找两端实体的 `element_id`
3. 调用 `kg_memory.create_relation()` 创建关系
4. 若查找失败或写入失败，将 action 标记为 `NO_ACTION` 并附失败原因

#### Bug 3: 报告计数在写入失败时未更新

**症状**: 实体写入失败时，报告仍计入 "创建实体" 计数。

**根因**: `entities_created` 计数器在 `_decide_entity()` 阶段递增，`execute()` 中写入失败只改变 `action.type` 为 `NO_ACTION`，未回退计数。

**修复**: 在 `execute()` 末尾重新统计实际成功写入的 action 数量：
```python
log.entities_created = sum(1 for a in log.actions if a.type == "CREATE_ENTITY")
log.relations_created = sum(1 for a in log.actions if a.type == "CREATE_RELATION")
```

### 10.5.2 Neo4j Schema 兼容性

#### 属性键不匹配警告

最初运行时产生大量 Neo4j 警告：
```
warn: property key does not exist. The property `confidence` does not exist...
```

**根因**: `kg_memory.py` 的 `check_relation_exists()` 方法查询了 `r.confidence`、`r.evidence`、`r.direction` 三个属性，但这些属性在队友通过 import 脚本创建的现有关系中不存在（现有关系属性为 `score`、`source`、`relation_id` 等）。

**修复**: 将 RETURN 子句从读取三个不存在的属性简化为仅返回 `elementId(r)`。方向矛盾检测暂时移除（待现有 Schema 升级后恢复）。

### 10.5.3 写入质量控制三层防线

修复上述 Bug 后，系统在 6/10 篇文章测试中发现三个数据质量问题，逐一建立防线：

#### 防线 1: 通用术语黑名单（GENERIC_TERM_BLACKLIST）

**问题**: LLM 从文献中提取了 "cancer"、"tumor"、"malignancies"、"cells"、"patients" 等泛化术语作为独立实体。这些概念虽然高频出现，但作为知识图谱节点毫无信息量。

**修复**: 在 `KGMemory` 类中添加 60+ 通用术语黑名单。`create_entity()` 在创建前检查名称（小写）是否在黑名单中，命中则静默拒绝。

```python
GENERIC_TERM_BLACKLIST: frozenset[str] = frozenset({
    "cancer", "tumor", "tumour", "cells", "cell",
    "patients", "patient", "controls", "disease", "diseases",
    "inflammation", "stress", "infection",
    "expression", "level", "levels", "activity",
    "response", "effect", "factor", "mechanism",
    "treatment", "therapy", "survival", "prognosis",
    "liver", "blood", "serum", "plasma", "tissue",
    # ... 60+ terms total
})
```

**效果**: 0 个通用术语泄漏至 Neo4j。

#### 防线 2: 大小写不敏感去重（find_entity_by_name_ci）

**问题**: 同一篇文章中，LLM 可能提取 "Immunomodulation" 和 "immunomodulation" 两个仅大小写不同的实体，导致重复节点。

**根因深入**: `verifier._verify_entity()` 在提取批次内不检查实体间重复——它仅将每个实体与 Neo4j 已有数据比对。由于同一批次内的实体尚未写入 Neo4j，两者均被判定为 NOVEL。

**修复历程**:
1. **第一次尝试**: 在 `create_entity()` 中添加 `find_entity_by_name_ci()` 方法，创建前做大小写不敏感查重。但该方法因 Cypher 语法错误 (`n.{*}` 不是合法的属性投影) 静默失败。
2. **第二次修复**: 修正为仅返回 `elementId(n)`，避免语法错误。
3. **验证**: 创建 "DedupFixTest" 后立即尝试创建 "dedupfixtest"——成功返回同一 `element_id`。

**Cypher 查询**:
```cypher
MATCH (n:Pathway)
WHERE toLower(coalesce(n.name, n.gene_symbol, n.disease_name)) = toLower($name)
RETURN elementId(n) AS element_id
LIMIT 1
```

**效果**: 0 个大小写重复。

#### 防线 3: 骨干疾病同义词匹配（DISEASE_SYNONYMS）

**问题**: "Hepatocellular carcinoma" 应匹配到骨干 HCC 节点（`UMLS:C2239176`），而非创建新的 `PROJECT:Disease:xxx`。同时 "type 2 diabetes" 和 "T2DM" 是同一种疾病但被创建为两个节点。

**根因**: 现有 HCC 节点使用 UMLS ID，LLM 提取的名称 "Hepatocellular carcinoma" 与现有节点名 "HCC" 不匹配。词法匹配（CONTAINS）存在方向性——`"HCC" CONTAINS "Hepatocellular carcinoma"` 为 False。

**修复**: 三管齐下——
1. **双向模糊匹配**: 修改 `find_entity_fuzzy()` 增加反向 CONTAINS 条件（`$mention CONTAINS toLower(n.name)`），使 "Hepatocellular carcinoma" 能反向匹配到 "HCC"
2. **同义词字典**: 添加 25+ 骨干疾病同义词映射（NAFLD/NASH/Fibrosis/Cirrhosis/HCC 各 4-5 个变体），在 `create_entity()` 中匹配到同义词时直接复用骨干 UMLS ID
3. **缩写展开**: "MASLD"→NAFLD, "MASH"→NASH 等

```python
DISEASE_SYNONYMS: dict[str, str] = {
    "nafld": "UMLS:C0400966",
    "non-alcoholic fatty liver disease": "UMLS:C0400966",
    "masld": "UMLS:C0400966",
    # ...
    "hcc": "UMLS:C2239176",
    "hepatocellular carcinoma": "UMLS:C2239176",
    "liver cancer": "UMLS:C2239176",
    # ...
}
```

**效果**: "Hepatocellular carcinoma" 和 "HCC" 不再创建新节点，均映射到骨干 UMLS ID。部分真实新疾病（如 "hepatitis B virus infection"）正确匹配到已有 UMLS ID（`UMLS:C0019163`）。

### 10.5.4 输出质量检查方法

所有写入测试均通过以下 5 项 Neo4j 查询进行自动化质量审计：

| 检查项 | Cypher 逻辑 | 目标 |
|--------|------------|------|
| **写入确认** | `MATCH (n) WHERE n.source STARTS WITH "PubMed" RETURN count(n)` | 验证实际写入 vs 报告声称 |
| **大小写重复** | `MATCH (n) WHERE n.source STARTS WITH "PubMed" WITH toLower(n.name) AS ln, collect(n) AS nodes, count(n) AS cnt WHERE cnt > 1 RETURN ln, nodes` | 0 个重复 |
| **通用术语泄漏** | `MATCH (n) WHERE n.source STARTS WITH "PubMed" AND toLower(n.name) IN ["cancer","tumor",...] RETURN n.name` | 0 个泄漏 |
| **骨干疾病匹配** | `MATCH (n:Disease) WHERE n.source STARTS WITH "PubMed" AND NOT n.disease_id STARTS WITH "UMLS:" AND <名称含骨干关键词> RETURN n` | 所有骨干疾病应使用 UMLS ID |
| **孤立节点** | `MATCH (n) WHERE n.source STARTS WITH "PubMed" AND NOT (n)--() RETURN count(n)` | 监控比例，排查异常 |

**Cypher 审计查询示例**（大小写重复检测）：

```cypher
MATCH (n)
WHERE n.source STARTS WITH "PubMed"
WITH toLower(n.name) AS name_lower,
     collect(DISTINCT n.name) AS original_names,
     count(n) AS node_count
WHERE node_count > 1
RETURN name_lower, original_names, node_count
ORDER BY node_count DESC
```

此查询利用 Neo4j 的聚合能力，在一次扫描中检测所有大小写变体重复，比逐对比较效率高 O(n) vs O(n²)。

### 10.5.5 写入测试最终结果

| 指标 | 初始测试（Bug 存在） | 修复后（v2.1） |
|------|---------------------|----------------|
| 实际写入实体 | **0**（全部丢失） | **109** (7篇) |
| 实际写入关系 | **0**（全部丢失） | **43** (7篇) |
| 大小写重复 | 4 组 (8 条) | **0** ✅ |
| 通用术语泄漏 | "cancer", "tumor" 等 | **0** ✅ |
| 骨干疾病重复 | "Hepatocellular carcinoma" | **均匹配到 UMLS** ✅ |
| Neo4j Schema 警告 | 每次查询 >50 条 | **0** ✅ |
| Skipping chunk 错误 | ~8% | ~10%（偶发，非致命） |
| 报告-实际一致性 | ❌（报告计数 ≠ Neo4j 数据） | **✅** |

---

## 10.6 30 篇 A/B 对比实验方案与命令

本节先记录 2026-06-29 已执行的一次 30 篇 A/B，再给出可重复命令。两代入口分别是：

- **第一代 Pipeline**：`multi_stage_extraction_pipeline.py`
- **第二代 Agent**：`cognitive_agent.agent`，通过 `run_cognitive_agent.sh` 启动

### 10.6.0 本次实际执行结果

| 项目 | 第一代 Pipeline | 第二代 Cognitive Agent |
|---|---|---|
| Run ID | `v1_pipeline_pubmed30_20260629_155656` | `v2_agent_pubmed30_20260629_155656` |
| 输入 | `extraction_output/pubmed_converted_500.jsonl` 前 30 篇 | 同左 |
| Neo4j 写入 | 关闭 | 关闭 |
| 运行状态 | 完成 | 命令完成，但 extraction 全部失败 |
| 实体 | 200 | 0 |
| 关系 | 118 | 0 |
| Import-ready | 37 | 0 |
| 错误数 | 0 | 30 |
| 主要说明 | 可作为本次 baseline | `GEMINI_API_KEY` 缺失/无效，实际使用 `LLM_API_KEY` 访问 Gemini proxy 返回 401 |

统一结果页：`extraction_output/abtest_compare_20260629_155656.md`。

### 10.6.1 实验原则

为了公平比较，建议采用以下固定条件：

| 条件 | 设置 |
|------|------|
| 输入文件 | `extraction_output/pubmed_converted_500.jsonl` |
| 样本 | 同一文件的前 30 篇 |
| Neo4j 写入 | 关闭，全部 dry-run/offline write |
| Neo4j 读取 | 可开启；若设置 `NEO4J_PASSWORD`，第二代会读取 KG 做 context/verification，但不会写入 |
| 第一代 preflight | 推荐单独运行，只读 Neo4j，用来评估第一代抽取结果的 endpoint linking |
| 输出目录 | `extraction_output/` |
| 日志目录 | `logs/` |

> 注意：如果不设置 `DEEPSEEK_API_KEY`，第一代 pipeline 会进入 mock extraction 模式，不能作为真实质量对比。第二代虽然会读取 `GEMINI_API_KEY` 或 `LLM_API_KEY`，但这两个变量不能盲目等价；token 必须对 `GEMINI_API_BASE` 和 `GEMINI_MODEL` 对应的 Gemini-compatible proxy 有效，否则会出现本次 30/30 的 401 Invalid token 失败。

### 10.6.2 一次性环境准备

```bash
cd /Users/a1234/FYP/liver_disease_kg_project

# 可选：加载本地 .env。该文件不要提交。
if [ -f workstreams/literature_hmdb_kegg/.env ]; then
  set -a
  source workstreams/literature_hmdb_kegg/.env
  set +a
fi

# 第一代 pipeline 使用 DeepSeek；若 .env 中只有 LLM_API_KEY，可复用。
# 注意：DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL 必须彼此匹配。
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-${LLM_API_KEY:-}}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"

# 第二代 agent 使用 Gemini-compatible proxy。
# 只有当 LLM_API_KEY 本身就是该 Gemini proxy 的有效 token 时，才可复用。
# 若确认可以复用，再手动执行：export GEMINI_API_KEY="$LLM_API_KEY"
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
export GEMINI_MODEL="${GEMINI_MODEL:-[按次]gemini-2.5-flash}"
export GEMINI_API_BASE="${GEMINI_API_BASE:-https://new.bitexingai.com}"

# 可选：设置后只读连接 Neo4j；本实验仍然不写入。
# export NEO4J_PASSWORD="<local-password>"

mkdir -p logs extraction_output

export INPUT="extraction_output/pubmed_converted_500.jsonl"
export STAMP="$(date +%Y%m%d_%H%M%S)"
export V1_RUN="v1_pipeline_pubmed30_${STAMP}"
export V2_RUN="v2_agent_pubmed30_${STAMP}"
```

检查环境：

```bash
./check_env.sh

test -n "$DEEPSEEK_API_KEY" || echo "[WARN] DEEPSEEK_API_KEY missing: v1 will use mock extraction"
test -n "$GEMINI_API_KEY" || echo "[ERROR] GEMINI_API_KEY missing: v2 cannot produce a valid comparison"
```

### 10.6.3 运行第一代 Pipeline（30 篇，dry-run）

```bash
time .venv-cognitive/bin/python multi_stage_extraction_pipeline.py \
  --input "$INPUT" \
  --limit 30 \
  --run-id "$V1_RUN" \
  --skip-neo4j \
  --output-dir extraction_output \
  | tee "logs/${V1_RUN}.log"
```

第一代输出：

```text
extraction_output/extraction_results_${V1_RUN}.json
extraction_output/quality_report_${V1_RUN}.json
extraction_output/provenance_report_${V1_RUN}.html
logs/${V1_RUN}.log
```

建议先看：

```bash
.venv-cognitive/bin/python - <<'PY'
import json, os
run = os.environ["V1_RUN"]
path = f"extraction_output/quality_report_{run}.json"
q = json.load(open(path, encoding="utf-8"))
print(json.dumps({
    "run_id": q["run_id"],
    "records": q["records"],
    "entities": q["entities"],
    "relations": q["relations"],
    "schema_valid_relations": q["schema_valid_relations"],
    "import_ready_relations": q["import_ready_relations"],
    "review_relations": q["review_relations"],
    "top_flags": sorted(q["flag_counts"].items(), key=lambda x: x[1], reverse=True)[:10],
}, indent=2, ensure_ascii=False))
PY
```

### 10.6.4 可选：运行第一代 Entity Linking Preflight（只读 Neo4j）

如果你想比较“第一代抽取结果最终有多少关系能链接到现有 KG 端点”，运行 preflight。该步骤不写 Neo4j，但需要 `NEO4J_PASSWORD`。

```bash
test -n "${NEO4J_PASSWORD:-}" || echo "[WARN] NEO4J_PASSWORD missing: skip preflight or set it first"

time .venv-cognitive/bin/python entity_linking_preflight.py \
  --input "extraction_output/extraction_results_${V1_RUN}.json" \
  --run-id "${V1_RUN}_preflight" \
  --refresh-index \
  | tee "logs/${V1_RUN}_preflight.log"
```

Preflight 输出：

```text
extraction_output/entity_linking_preflight/${V1_RUN}_preflight_preflight_report.json
extraction_output/entity_linking_preflight/${V1_RUN}_preflight_preflight_report.md
extraction_output/entity_linking_preflight/${V1_RUN}_preflight_linked_extraction_results.json
```

如果已经刷新过 Neo4j index，后续可以去掉 `--refresh-index`，速度会更快。

### 10.6.5 运行第二代 Cognitive Agent（30 篇，dry-run）

默认 `run_cognitive_agent.sh` 是 `--skip-neo4j-write`，即使设置了 `NEO4J_PASSWORD` 也不会写库，只会读 KG 做 context activation 与 verification。

```bash
test -n "${GEMINI_API_KEY:-}" || {
  echo "[ERROR] Set a valid GEMINI_API_KEY for the Gemini-compatible proxy before running V2"
  exit 1
}

time COGNITIVE_AGENT_INPUT="$INPUT" \
  COGNITIVE_AGENT_RUN_ID="$V2_RUN" \
  ./run_cognitive_agent.sh 30 \
  | tee "logs/${V2_RUN}.log"
```

第二代输出：

```text
extraction_output/agent_results_${V2_RUN}.json
extraction_output/agent_report_${V2_RUN}.json
logs/${V2_RUN}.log
```

建议先看：

```bash
.venv-cognitive/bin/python - <<'PY'
import json, os
run = os.environ["V2_RUN"]
path = f"extraction_output/agent_report_{run}.json"
r = json.load(open(path, encoding="utf-8"))
print(json.dumps({
    "run_id": r["run_id"],
    "total_articles": r["total_articles"],
    "total_time_s": r["total_time_s"],
    "avg_time_per_article_s": r["avg_time_per_article_s"],
    "extraction": r["extraction"],
    "decisions": r["decisions"],
    "quality": r["quality"],
}, indent=2, ensure_ascii=False))
PY
```

### 10.6.6 生成统一对比报告

两代输出 JSON 结构不同，因此建议用一个小脚本把核心指标归一化。下面命令会生成：

```text
extraction_output/abtest_compare_${STAMP}.md
```

```bash
.venv-cognitive/bin/python - <<'PY'
import json
import os
from pathlib import Path

out = Path("extraction_output")
v1_run = os.environ["V1_RUN"]
v2_run = os.environ["V2_RUN"]
stamp = os.environ["STAMP"]

v1 = json.load(open(out / f"quality_report_{v1_run}.json", encoding="utf-8"))
v2_report = json.load(open(out / f"agent_report_{v2_run}.json", encoding="utf-8"))
v2_results = json.load(open(out / f"agent_results_{v2_run}.json", encoding="utf-8"))

def pct(num, den):
    return round(num / den * 100, 1) if den else 0.0

v1_zero_rel = sum(1 for r in v1.get("records_detail", []) if r.get("relations", 0) == 0)
v2_records = v2_results.get("records", [])
v2_zero_rel = sum(
    1 for r in v2_records
    if r.get("phases", {}).get("extraction", {}).get("relation_count", 0) == 0
)
v2_schema_valid = sum(
    r.get("phases", {}).get("verification", {}).get("summary", {}).get("schema_valid", 0)
    for r in v2_records
)

v1_rel = v1.get("relations", 0)
v2_rel = v2_report.get("extraction", {}).get("total_relations_extracted", 0)
v1_ir = v1.get("import_ready_relations", 0)
v2_ir = v2_report.get("decisions", {}).get("total_import_ready", 0)
v2_errors = v2_report.get("quality", {}).get("error_count", 0)
v2_valid_extraction = v2_errors == 0 and v2_rel > 0

def v2_pct(num, den):
    return f"{pct(num, den)}%" if v2_valid_extraction else "n/a"

lines = []
lines.append(f"# PubMed 30-Article A/B Comparison — {stamp}")
lines.append("")
lines.append(f"- V1 run: `{v1_run}`")
lines.append(f"- V2 run: `{v2_run}`")
lines.append(f"- Input: `{os.environ.get('INPUT', '')}`")
lines.append("")
lines.append("| Metric | V1 Pipeline | V2 Cognitive Agent |")
lines.append("|---|---:|---:|")
lines.append(f"| Articles | {v1.get('records', 0)} | {v2_report.get('total_articles', 0)} |")
lines.append(f"| Total entities | {v1.get('entities', 0)} | {v2_report.get('extraction', {}).get('total_entities_extracted', 0)} |")
lines.append(f"| Entities/article | {round(v1.get('entities', 0)/max(v1.get('records', 1),1), 1)} | {v2_report.get('extraction', {}).get('avg_entities_per_article', 0)} |")
lines.append(f"| Total relations | {v1_rel} | {v2_rel} |")
lines.append(f"| Relations/article | {round(v1_rel/max(v1.get('records', 1),1), 1)} | {v2_report.get('extraction', {}).get('avg_relations_per_article', 0)} |")
lines.append(f"| Schema-valid relations | {v1.get('schema_valid_relations', 0)} | {v2_schema_valid} |")
lines.append(f"| Schema-valid rate | {pct(v1.get('schema_valid_relations', 0), v1_rel)}% | {v2_pct(v2_schema_valid, v2_rel)} |")
lines.append(f"| Import-ready relations | {v1_ir} | {v2_ir} |")
lines.append(f"| Import-ready rate | {pct(v1_ir, v1_rel)}% | {v2_pct(v2_ir, v2_rel)} |")
lines.append(f"| Review/discard count | {v1.get('review_relations', 0)} | {v2_report.get('decisions', {}).get('discarded', 0)} |")
lines.append(f"| Zero-relation articles | {v1_zero_rel} | {v2_zero_rel} |")
lines.append(f"| Error count | 0 | {v2_errors} |")
lines.append(f"| Avg quality score | n/a | {v2_report.get('quality', {}).get('avg_quality_score', 0)} |")
lines.append(f"| Total time seconds | see log | {v2_report.get('total_time_s', 0)} |")
lines.append(f"| Avg time/article seconds | see log | {v2_report.get('avg_time_per_article_s', 0)} |")
lines.append("")
lines.append("## V1 top quality flags")
lines.append("")
for flag, count in sorted(v1.get("flag_counts", {}).items(), key=lambda x: x[1], reverse=True)[:15]:
    lines.append(f"- `{flag}`: {count}")
lines.append("")
lines.append("## V2 decision summary")
lines.append("")
for key, val in v2_report.get("decisions", {}).items():
    lines.append(f"- `{key}`: {val}")
lines.append("")
lines.append("## Interpretation Checklist")
lines.append("")
lines.append("- If V2 has many more entities/relations, inspect whether they are evidence-grounded rather than broad background terms.")
lines.append("- If V1 schema-valid is low, inspect `flag_counts` to see whether schema expansion or sampling is the bottleneck.")
lines.append("- If V2 import-ready is high but write is disabled, run a separate 3-5 article Neo4j write test before bulk import.")
lines.append("- Compare HTML provenance for V1 and per-record phases in V2 before making claims about biological correctness.")
if v2_errors:
    lines.append("- V2 has extraction errors; do not interpret zero output as an agent quality result until provider authentication is fixed.")

compare_path = out / f"abtest_compare_{stamp}.md"
compare_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(compare_path)
PY
```

### 10.6.7 推荐人工检查顺序

30 篇结果出来后，不要只看总表。建议按下面顺序检查：

1. 打开第一代 `provenance_report_${V1_RUN}.html`，随机检查 10 条 import-ready 关系的 evidence 是否真能支撑关系。
2. 打开第二代 `agent_results_${V2_RUN}.json`，检查每篇的 `phases.strategy`、`phases.verification.summary`、`phases.conflict_resolution`、`phases.execution.actions`。
3. 对比零产出文章：如果第一代 0R、第二代有输出，判断第二代是否确实提取了有用知识，而不是过度抽取。
4. 对比 V1 `flag_counts` 与 V2 `discarded`：如果 V1 大量 `schema_mismatch:Protein->Disease`，说明第二代 42 对签名可能确实缓解了第一代 schema 过窄问题。
5. 若准备写 Neo4j，只取第二代中 3-5 篇高质量文章先跑 `--write-neo4j`，再做 Cypher 审计，不要直接 30 篇全写。

### 10.6.8 3-5 篇 Neo4j 写入验证命令（可选）

如果 30 篇 dry-run 结果满意，可以单独挑小批量写入。该步骤会修改 Neo4j：

```bash
export NEO4J_PASSWORD="<local-password>"
export WRITE_RUN="v2_agent_write_probe5_$(date +%Y%m%d_%H%M%S)"

time COGNITIVE_AGENT_INPUT="$INPUT" \
  COGNITIVE_AGENT_RUN_ID="$WRITE_RUN" \
  ./run_cognitive_agent.sh 5 --write-neo4j \
  | tee "logs/${WRITE_RUN}.log"
```

写入后至少检查：

```cypher
MATCH (n)
WHERE n.source STARTS WITH "PubMed"
RETURN labels(n) AS labels, count(n) AS count
ORDER BY count DESC;

MATCH ()-[r]->()
WHERE r.source STARTS WITH "PubMed"
RETURN type(r) AS rel_type, count(r) AS count
ORDER BY count DESC;
```

---

## 11. 当前局限与未来方向

### 11.1 当前局限

| 局限 | 说明 | 优先级 |
|------|------|--------|
| **gemini-2.5-flash 绑死** | gemini-3-flash-preview 不支持 response_schema | P1 |
| **批内语义去重缺失** | "type 2 diabetes" 和 "T2DM" 仍为两个节点 | P1 |
| **批量写入延迟** | 每篇文章 60-130s（LangExtract + Neo4j 往返），大面积入库耗时长 | P2 |
| **Gene/Protein 歧义** | 生物医学固有歧义，当前无自动修正 | P2 |
| **PubMed 采样质量** | 50%+ 文章与 molecular KG 范式不匹配 | P2 |
| **无 Dense Semantic Matching** | 对 Nrf2→NFE2L2 等需要语义匹配 | P2 |
| **无人工反馈回路** | Few-shot 固定，无法从审核中学习 | P3 |
| **30 篇公平 A/B 尚未完成** | 当前已有历史基准和小样本验证，但仍需同一输入下的 30 篇直接对比 | P1 |

### 11.2 短期优化 (1-2 周)

1. ~~**Neo4j 连接恢复 + 写入测试**~~ ✅ 已完成（2026-06-29）
2. **语义去重** — 对同批内实体做 SapBERT embedding 余弦相似度聚类，合并 "T2DM" / "type 2 diabetes" 等缩写展开对
3. **扩充 Disease 节点** — 从 5 个扩展到 ~20 个（HBV/HCV/ALD/AIH 等）

### 11.3 中期优化 (1-2 月)

1. **SapBERT 语义链接器** — 离线生成 KG 节点 embedding，替代纯词法匹配
2. **Gene→Protein 自动修正** — SchemaRepairCritic 主动修正类型歧义
3. **主动学习回路** — 人工审核 → 自动加入 Few-shot 示例库
4. **多模型对比** — GPT-4o / Claude 对比 schema 符合率

### 11.4 长期优化 (3-6 月)

1. **Evidence Node 设计** — 参考 EvidenceNet，将扁平三元组升级为 `(Article)--[REPORTS]-->(Evidence)--[SUPPORTS]-->(Gene)--[ASSOCIATED_WITH]-->(Disease)` 带 study_design/cohort_size/p_value 等上下文
2. **PMC 全文抽取** — 从 Abstract 扩展到 Open Access 全文
3. **KGE Link Prediction 闭环** — LLM 抽取 + KGE 推理互相增强

---

## 12. 附录：关键指标汇总

### 12.1 本次 30 篇 A/B 实际执行

| 指标 | 第一代 Pipeline | 第二代 Agent |
|---|---:|---:|
| Run ID | `v1_pipeline_pubmed30_20260629_155656` | `v2_agent_pubmed30_20260629_155656` |
| 输入文章 | 30 | 30 |
| 成功抽取文章 | 30 | 0 |
| LLM/Provider | DeepSeek-compatible | Gemini-compatible proxy |
| 总运行时间 | 168.5s | 120.2s |
| 总实体数 | 200 | 0 |
| 总关系数 | 118 | 0 |
| Schema-valid | 67 (56.8%) | n/a |
| Import-ready | 37 (31.4%) | 0 |
| Review/Discard | 81 review | 0 discard |
| 错误数 | 0 | 30 |
| 结论 | 可作为当前 30 篇 baseline | token 认证失败，需有效 `GEMINI_API_KEY` 后重跑 |

### 12.2 第二代 50 篇离线测试（Gemini 原生，修复后）

| 指标 | 数值 |
|---|---|
| 输入文章 | 50 篇 |
| LLM 模型 | `[按次]gemini-2.5-flash` (原生 Gemini) |
| 总运行时间 | 1,388s (28s/篇) |
| 总实体数 | 992 (19.8/篇) |
| 总关系数 | 688 (13.8/篇) |
| Import-Ready | 595 (86.5%) |
| 丢弃 | 76 (11.0%) |
| 平均质量分 | 0.57 |
| 错误数 | **0** |
| Neo4j 连接 | ❌ (离线模式) |

### 12.3 第二代 10 篇 Neo4j 写入测试（v2.1 含质量防线）

| 指标 | 数值 |
|---|---|
| 输入文章 | 7 篇（3 篇因超时中断，含 2 篇 >120s 长文） |
| LLM 模型 | `[按次]gemini-2.5-flash` (原生 Gemini) |
| 平均速度 | 79s/篇（含 Neo4j 读写往返） |
| 实际写入实体 | **109** (15.6/篇) |
| 实际写入关系 | **43** (6.1/篇) |
| 大小写重复 | **0** ✅ |
| 通用术语泄漏 | **0** ✅ |
| 骨干疾病 UMLS 匹配 | 正常（HCC 等不再重复创建） |
| Neo4j Schema 警告 | **0** ✅ |
| 错误数 | **0** |
| 报告-实际一致性 | **100%** ✅ |

> **速度说明**: 含 Neo4j 写入后从 28s/篇 增至 79s/篇。增量主要来自：(1) 每实体 1 次 `find_entity()` + 1 次 `find_entity_by_name_ci()` + 1-2 次 `MERGE`；(2) 每关系 2 次 `find_entity()` + 1 次 `CREATE`；(3) Context Activation 的 Neo4j 先验查询。1927 篇全量预计 ~42 小时，需考虑分批运行。

### 12.4 历史第一代 500 篇基准（仅作背景）

| 指标 | 数值 |
|---|---|
| 输入文章 | 500 篇 PubMed 肝病相关摘要 |
| LLM 模型 | DeepSeek Chat (temperature=0) |
| 总运行时间 | ~37 分钟 |
| 总实体数 | 3,224 |
| 总关系数 | 2,029 |
| Schema-valid | 1,135 (55.9%) |
| Import-ready | 465 (22.9%) |
| 零产出文章 | 327 (65.4%) |
| 预估最终入库 | 60-120 条 (3-6%) |

### 12.5 系统能力矩阵

| 能力 | 第一代 | 第二代 |
|------|--------|--------|
| PubMed → 结构化三元组 | ✅ 本次 30 篇：3.9 关系/篇 | ⚠️ 架构支持 LangExtract+Gemini；本次 30 篇因 token 认证失败未产出 |
| Schema 强制约束 | ✅ 15+ 规则 | ✅ 原生协议 + 42 对签名 |
| 否定/不确定检测 | ✅ | ✅ |
| 证据溯源 | ✅ 字符级 | ✅ |
| 实体标准化 | ⚠️ 字典驱动 | ✅ + NCBI 外部验证 |
| 实体链接 | ⚠️ Ensemble linker | ✅ KG Verifier + Context Activator |
| 主动创建实体 | ❌ | ✅ 类型特定主键 + MERGE |
| 重试机制 | ❌ | ✅ Exponential backoff |
| 因果推理 | ❌ | ✅ 传递推理 |
| 冲突解决 | 简单标记 | ✅ 4 类冲突 + 决策表 |
| 自我反思 | ❌ | ✅ 元认知循环 |
| 策略自适应 | ❌ | ✅ 探索/聚焦模式 |
| Strategy→Prompt/Examples | ❌ | ✅ v2.2 已闭环 |
| Conflict→Decision | ❌ | ✅ v2.2 已闭环 |
| Update/Dispute 执行 | ❌ | ✅ v2.2 已调用 KGMemory |
| 跨文档记忆 | ❌ | ✅ 三记忆系统 |
| 嵌套属性过滤 | N/A | ✅ RELATION_ATTRIBUTE_KEYS |
| 跨文章实体去重 | N/A | ✅ 不依赖 pmid 的 ID 生成 |
| 写入质量防线 | ❌ | ✅ 黑名单 + CI去重 + 同义词匹配 |
| Neo4j 写入验证 | ❌ | ✅ 5 项 Cypher 审计查询 |
| 报告-实际一致性 | ⚠️ 手动 | ✅ 自动重统计 |

### 12.6 文件清单

| 文件 | 行数 | 说明 |
|---|---|---|
| **第一代系统** | | |
| `multi_stage_extraction_pipeline.py` | 1,628 | 5-Stage 抽取引擎 |
| `entity_linking_preflight.py` | ~1,200 | Ensemble 实体链接 |
| `convert_pubmed_xml_to_jsonl.py` | ~220 | XML→JSONL 转换 |
| **第二代系统** | | |
| `cognitive_agent/agent.py` | 681 | 主循环 + 组件编排 + strategy prompt 注入 |
| `cognitive_agent/extraction_kernel.py` | ~269 | LangExtract 封装 |
| `cognitive_agent/verifier.py` | ~260 | 实体溯源 + Schema 检查 |
| `cognitive_agent/decision_engine.py` | 661 | 决策 + 执行 + NCBI + strategy/conflict 闭环 |
| `cognitive_agent/memory/kg_memory.py` | 687 | Neo4j 记忆体 + 质量防线 |
| `cognitive_agent/schema/` | ~582 | 实体/关系/示例定义 |
| `tests/test_cognitive_agent_loop.py` | 198 | v2.2 闭环回归测试 |

### 12.7 关系 Schema 完整签名矩阵（v2.2：8 谓词 / 42 对签名）

```python
RELATION_SIGNATURES = {
    "ASSOCIATED_WITH": {
        # 经典分子-疾病关联
        ("Gene", "Disease"), ("Protein", "Disease"),
        ("Metabolite", "Disease"), ("Pathway", "Disease"),
        # 疾病影响范围
        ("Disease", "Tissue"), ("Disease", "Pathway"), ("Disease", "Disease"),
        # 逆向关联
        ("Pathway", "Tissue"), ("Tissue", "Disease"), ("CellType", "Disease"),
        # 分子定位
        ("CellType", "Pathway"), ("CellType", "Tissue"),
        ("Gene", "Tissue"), ("Gene", "Pathway"),
        ("Protein", "Pathway"), ("Protein", "Tissue"),
        ("Metabolite", "Pathway"),
        # 功能关联
        ("Gene", "Gene"), ("Gene", "Protein"), ("Protein", "Protein"),
        # 通路/组织互作
        ("Pathway", "Pathway"), ("Tissue", "Pathway"),
    },
    "PROGNOSTIC_IN":     {("Gene", "Disease"), ("Protein", "Disease")},
    "PROGRESSES_TO":     {("Disease", "Disease")},
    "ENCODES":           {("Gene", "Protein")},
    "INTERACTS_WITH":    {("Protein", "Protein"), ("Gene", "Protein"),
                          ("Protein", "Gene"), ("Gene", "Gene"),
                          ("CellType", "CellType"), ("Metabolite", "Protein")},
    "PARTICIPATES_IN":   {("Gene", "Pathway"), ("Protein", "Pathway"),
                          ("CellType", "Pathway"), ("Metabolite", "Pathway")},
    "EXPRESSED_IN":      {("Gene", "Tissue"), ("Gene", "CellType"),
                          ("Protein", "Tissue"), ("Protein", "CellType")},
    "ASSOCIATED_WITH_METABOLITE": {("Gene", "Metabolite"), ("Protein", "Metabolite")},
}
```

---

## 参考文献

1. Elliott, T.J. et al. (2025). Data Overdose? Time for a Quadruple Shot: Knowledge Graph Construction using Enhanced Triple Extraction. *arXiv*.
2. Zong, C. et al. (2026). Building evidence-based knowledge graphs from full-text literature for disease-specific biomedical reasoning. *arXiv*.
3. Zhang, Y. et al. (2024). Liver Cancer Knowledge Graph Construction based on dynamic entity replacement and masking strategies RoBERTa-BiLSTM-CRF model. *arXiv*.
4. Sung, M. et al. (2020). BioSyn: Biomedical Entity Representations using Synonym Marginalization. *ACL*.
5. Liu, F. et al. (2021). SapBERT: Self-alignment pretraining for BERT. *NAACL*.
6. LangExtract GitHub Issues: [#222](https://github.com/google/langextract/issues/222), [#301](https://github.com/google/langextract/issues/301), [#287](https://github.com/google/langextract/issues/287).

---

> **报告版本**: v3.3  
> **涵盖时期**: 2026-06-27 至 2026-06-29  
> **目标数据库**: liver-kg-core-v02 (Neo4j, bolt://100.104.181.96:7687)  
> **更新说明**: v3.3 将主实验口径切换为 30 篇 A/B 实际执行结果：第一代 pipeline 完成 baseline，第二代 agent 因 Gemini token 401 认证失败未形成质量对比；同步更新 §5、§10.2、§10.6 与附录指标，并将 500 篇实验降级为历史背景。
