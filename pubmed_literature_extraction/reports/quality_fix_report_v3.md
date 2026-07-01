# Cognitive Agent V3 质量修复报告

> 生成日期：2026-06-29 | 模型：Gemini 2.5 Flash | 测试集：PubMed 30 篇肝病文献

---

## 一、背景与问题

在 V2 Cognitive Agent 的 30 篇文章 A/B 对比测试中，发现三类关键质量问题：

| # | 问题 | 表现 | 根本原因 |
|---|------|------|----------|
| 1 | **泛化术语泄漏** | "gene symbols"、"chemicals"、"malignancies"、"immune regulation" 等通用术语被当作实体创建 | 黑名单覆盖不足（~65 条），LangExtract 对综述类文章容易提取泛化描述词 |
| 2 | **同概念多节点** | "T2DM" 和 "type 2 diabetes mellitus" 创建两个独立节点；"HCC" 和 "Hepatocellular carcinoma" 重复 | 缺乏缩写消歧机制，LLM 提取时对同一概念使用不同名称变体 |
| 3 | **方法论噪声** | "network pharmacology"、"molecular docking"、"KEGG signaling pathway" 等研究方法/数据库术语混入 KG | 没有区分生物医学实体与研究方法论术语的过滤层 |

---

## 二、解决方案架构：三道防线 + 缩写消歧

```
LLM 提取的实体
      │
      ▼
┌─ [1] 黑名单精确匹配 ─────────────────────────────────────┐
│   GENERIC_TERM_BLACKLIST: ~210 个泛化术语                  │
│   "gene symbols" → DISCARD                                │
│   "malignancies" → DISCARD                                │
│   "immune regulation" → DISCARD                           │
└──────────────────────────────────────────────────────────┘
      │ 通过
      ▼
┌─ [2] 方法论噪声检测 ──────────────────────────────────────┐
│   is_methodology_noise(): 精确 + 子串匹配                  │
│   "network pharmacology" → DISCARD (方法论)                │
│   "molecular docking" → DISCARD (实验技术)                 │
│   白名单例外: "NF-κB signaling pathway" → 通过 ✅          │
└──────────────────────────────────────────────────────────┘
      │ 通过
      ▼
┌─ [3] 名称质量评分 ────────────────────────────────────────┐
│   score_entity_name_quality(): 9 种惩罚模式                │
│   "Gene Ontology" → 含 database 名称 → score < 0.3 → DISCARD │
│   "replication" → 过短+停用词特征 → DISCARD               │
└──────────────────────────────────────────────────────────┘
      │ 通过
      ▼
┌─ [4] 缩写感知消歧 ────────────────────────────────────────┐
│   AbbreviationDetector + AbbreviationMap                  │
│   "ALT" ↔ "alanine aminotransferase" → 合并为一个节点      │
│   "T2DM" ↔ "type 2 diabetes mellitus" → 合并为一个节点     │
│   "HCC" ↔ "Hepatocellular carcinoma" → 合并为一个节点      │
└──────────────────────────────────────────────────────────┘
      │
      ▼
   CREATE 实体
```

### 关键设计决策

- **全部过滤在 `decide()` 阶段执行**：因为离线模式（`--skip-neo4j-write`）跳过 `execute()` 阶段，黑名单检查必须在决策阶段完成
- **缩写消歧在实体创建前完成**：通过 `_deduplicate_entities()` 将实体按规范名称分组，每组仅保留最长/最具描述性的名称
- **方法论过滤白名单机制**：子串匹配会误杀 "NF-κB signaling pathway" 等合法实体，因此从子串列表中移除 "signaling pathway" 和 "metabolic pathway"，仅保留黑名单精确匹配

---

## 三、改动文件清单

| 文件 | 改动类型 | 行数 | 说明 |
|------|----------|------|------|
| `cognitive_agent/abbreviation_detector.py` | **新建** | ~480 | Schwartz-Hearst 算法 + 精选缩写词典 + 方法论黑名单 + 名称质量评分 |
| `cognitive_agent/memory/kg_memory.py` | 修改 | +120 | GENERIC_TERM_BLACKLIST 从 ~65 扩展到 ~210 |
| `cognitive_agent/decision_engine.py` | 修改 | +80 | 集成三道防线 + 缩写消歧 `_deduplicate_entities()` |
| `cognitive_agent/agent.py` | 修改 | +25 | 集成 AbbreviationDetector，Phase 2 前构建缩写映射表 |
| `cognitive_agent/memory/episodic_memory.py` | 修改 | +5 | 添加线程锁，支持并发安全 |

### 3.1 AbbreviationDetector 核心组件

#### Schwartz-Hearst 算法（1999, 2003）
生物医学缩写检测的黄金标准算法，通过匹配括号模式 "长格式 (缩写)" 自动识别缩写：
- 支持词首字符匹配和词内字符匹配
- 候选评分：优先选择内部匹配少、文本更长的候选
- 停用词容忍：允许缩写展开中包含 "of"、"and"、"in" 等停用词

#### CURATED_ABBREVIATIONS（62 条精选词典）
覆盖肝病领域的六大类别，来源于 UMLS/MeSH：

| 类别 | 示例 | 数量 |
|------|------|------|
| 肝病专有名词 | ALT→alanine aminotransferase, AST→aspartate aminotransferase | 12 |
| 糖尿病/代谢 | T2DM→type 2 diabetes mellitus, NAFLD→non-alcoholic fatty liver disease | 10 |
| 临床指标 | BMI→body mass index, HOMA-IR→homeostatic model assessment of insulin resistance | 8 |
| 生物分子 | ROS→reactive oxygen species, TNF-α→tumor necrosis factor alpha | 14 |
| 细胞类型 | HSC→hepatic stellate cell, KC→Kupffer cell | 8 |
| 研究方法论 | GO→Gene Ontology, KEGG→Kyoto Encyclopedia of Genes and Genomes | 10 |

#### METHODOLOGY_BLACKLIST（~120 条方法论噪声词）
覆盖：
- 生物信息学工具：STRING, Cytoscape, AutoDock, PyMOL
- 实验技术：western blot, immunohistochemistry, qRT-PCR, ELISA
- 数据库/资源：Gene Ontology, KEGG, Reactome, UniProt
- 统计术语：meta-analysis, systematic review, sensitivity analysis
- 研究方法：network pharmacology, molecular docking, enrichment analysis

#### 名称质量评分（9 种惩罚模式）
```
1. 太短（≤3 字符）                          → -0.4
2. 全小写且非专有名词                        → -0.3
3. 以泛化类别词开头（"level of", "role of"） → -0.3
4. 包含方法论术语                            → -0.4
5. 数据库/资源名称                           → -0.3
6. 研究人群术语                              → -0.3
7. 停用词类名称                              → -0.3
8. 全大写缩写且无词典匹配                     → -0.2
9. 纯数字/符号                               → -0.5
```

### 3.2 GENERIC_TERM_BLACKLIST 扩展

从 ~65 条扩展到 ~210 条，新增 4 大类别：

| 新增类别 | 示例 | 数量 |
|----------|------|------|
| 方法论噪声 | "network pharmacology", "molecular docking", "enrichment analysis" | ~30 |
| LangExtract 伪影 | "gene symbols", "disease-related targets", "signaling molecules" | ~25 |
| 通用细胞生物学 | "immune regulation", "immunomodulation", "cellular processes" | ~20 |
| 统计/研究术语 | "replication", "confounding factors", "baseline characteristics" | ~15 |

---

## 四、效果评估

### 4.1 V1 vs V2 vs V3 对比（30 篇 PubMed 肝病文献）

| 指标 | V1 Pipeline | V2 Agent | V3 Agent（修复后） | V2→V3 变化 |
|------|:-----------:|:--------:|:------------------:|:----------:|
| 提取实体 | 203 | 708 | 713 | +0.7% |
| 提取关系 | 135 | 486 | 494 | +1.6% |
| 实体创建 | 203 | **602** | **445** | **-26.1%** |
| 关系创建 | 37 | 399 | 412 | +3.3% |
| 丢弃数 | — | 74 | 73 | -1.4% |
| Import-Ready 率 | 27.4% | 82.1% | **83.4%** | +1.3pp |
| 处理耗时 | — | 178.5s | 188.9s | +5.8% |
| API 错误 | 0 | 0 | 0 | — |

### 4.2 噪声拦截效果（V3 相比 V2）

V2 泄漏的 12 个代表性噪声词，V3 全部拦截：

| 噪声词 | V2 状态 | V3 状态 | 拦截手段 |
|--------|:------:|:------:|----------|
| "network pharmacology" | ❌ 泄漏 | ✅ 拦截 | 方法论噪声检测 |
| "molecular docking" | ❌ 泄漏 | ✅ 拦截 | 方法论噪声检测 |
| "gene symbols" | ❌ 泄漏 | ✅ 拦截 | 黑名单精确匹配 |
| "disease-related targets" | ❌ 泄漏 | ✅ 拦截 | 黑名单精确匹配 |
| "chemicals" | ❌ 泄漏 | ✅ 拦截 | 黑名单精确匹配 |
| "Gene Ontology" | ❌ 泄漏 | ✅ 拦截 | 名称质量评分（数据库名） |
| "KEGG signaling pathway" | ❌ 泄漏 | ✅ 拦截 | 名称质量评分（数据库名） |
| "replication" | ❌ 泄漏 | ✅ 拦截 | 黑名单精确匹配 |
| "malignancies" | ❌ 泄漏 | ✅ 拦截 | 黑名单精确匹配 |
| "Immunomodulation" | ❌ 泄漏 | ✅ 拦截 | 黑名单精确匹配 |
| "immune regulation" | ❌ 泄漏 | ✅ 拦截 | 黑名单精确匹配 |
| "immune evasion" | ❌ 泄漏 | ✅ 拦截 | 黑名单精确匹配 |

### 4.3 同概念多节点消歧效果

实体创建从 602 降至 445（-26.1%），其中缩写消歧贡献显著：

| 缩写组 | V2 创建节点 | V3 创建节点 |
|--------|:----------:|:----------:|
| HCC / Hepatocellular carcinoma | 2 | 1 |
| NAFLD / non-alcoholic fatty liver disease | 2 | 1 |
| T2DM / type 2 diabetes mellitus | 2 | 1 |
| ALT / alanine aminotransferase | 2 | 1 |
| HSC / hepatic stellate cell | 2 | 1 |
| MASLD / metabolic dysfunction-associated steatotic liver disease | 2 | 1 |

### 4.4 合法实体零误杀验证

以下合法实体在三道防线中全部正确通过：

- ✅ NF-κB signaling pathway（子串匹配白名单保护）
- ✅ AKT1（基因符号，非泛化术语）
- ✅ Hepatocellular carcinoma（规范疾病名称）
- ✅ Tumor necrosis factor alpha（生物分子全称）
- ✅ Hepatic stellate cells（规范细胞类型）
- ✅ Alanine aminotransferase（规范蛋白名称）

---

## 五、已知局限性

1. **Schwartz-Hearst 相邻缩写干扰**：当文本中出现 "(ALT) and aspartate aminotransferase (AST)" 时，算法可能无法正确匹配 ALT → alanine aminotransferase。**补偿方案**：CURATED_ABBREVIATIONS 词典覆盖高频肝病缩写。

2. **离线模式 Neo4j 去重缺失**：`--skip-neo4j-write` 跳过 `execute()` 阶段的 `find_entity_by_name_ci` 模糊匹配，跨文章的同一缩写变体可能重复。**解决方案**：启用 Neo4j 在线模式可进一步消歧。

3. **黑名单维护成本**：GENERIC_TERM_BLACKLIST 和 METHODOLOGY_BLACKLIST 需要根据新发现的噪声模式持续更新。

4. **语言模型依赖性**：质量仍取决于 LLM 提取的实体名称质量，极端情况下（如 LLM 提取 "liver damage" 而非 "hepatic injury"）可能误分类。

---

## 六、运行指令

### V3 Agent 30 篇测试
```bash
cd /Users/a1234/FYP/liver_disease_kg_project
python3 -m cognitive_agent.agent \
  --input extraction_output/pubmed_liver_30.json \
  --output extraction_output/agent_results_v3_test \
  --model "[按次]gemini-2.5-flash" \
  --max-workers 5 \
  --skip-neo4j-write
```

### V1 Pipeline 30 篇对比测试
```bash
python3 multi_stage_extraction_pipeline.py \
  --input extraction_output/pubmed_liver_30.json \
  --output extraction_output/extraction_results_v1_pipeline_30 \
  --max-workers 5
```

---

## 附录：缩略语对照

| 缩写 | 全称 |
|------|------|
| KG | Knowledge Graph（知识图谱） |
| LLM | Large Language Model（大语言模型） |
| UMLS | Unified Medical Language System |
| MeSH | Medical Subject Headings |
| MASLD | Metabolic Dysfunction-Associated Steatotic Liver Disease |
| MASH | Metabolic Dysfunction-Associated Steatohepatitis |
| HCC | Hepatocellular Carcinoma |
| NAFLD | Non-Alcoholic Fatty Liver Disease |
