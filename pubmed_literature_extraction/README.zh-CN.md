<div align="center">

# PubMed 文献知识抽取 Agent

### 面向 LiverKG 的证据约束型生物医学关系抽取系统

[English](README.md) · [返回 LiverKG](../README.zh-CN.md) · [Agent v3 架构](ARCHITECTURE_V3.md) · [方法创新](METHOD_INNOVATION.md)

</div>

## 离线答辩演示

安装 CLI 后生成审查页，再用浏览器打开。

```bash
liverkg --lang zh review -o defense-review.html
```

依次点击 LiverKG 关系和自定义 schema 关系，比较同样通过语义校验的两条候选为何有不同的图谱写入资格。再切到待复核和拒绝样例，对照原文中缺少关系触发证据与命中排除规则的区别。四条记录均为虚构样例，由项目的验证器现场计算判定，无需模型密钥或 Neo4j。

给命令传入真实的 `agent_results_*.json` 可审查一批已保存的抽取结果。页面提供中英文切换。[CLI 使用说明](LIVERKG_CLI.md#evidence-review)

## 项目目标

本模块从 PubMed 摘要中抽取肝病相关知识，并生成可审计的 Neo4j 候选实体和关系。
项目同时包含最初的提示词流水线、闭环 Cognitive Agent，以及 Central Agent v2：
一个由确定性 Verifier 驱动、只在文章状态确实需要时调用昂贵工具的中央控制器。

Agent v3 在此基础上增加安全规则记忆、DeepSeek/Qwen 双模型批判、最小证据片段
蕴含判断和非参数 Conformal Risk Router。v3 不训练本地 BERT，正式研究路径依赖
冻结开发集和预注册专家 blind cohort。

系统将“提出候选”和“授予写入资格”严格分离：

- LLM、RAG 和 Causal 工具可以提出或注释候选；
- 确定性 Verifier 决定证据与 Schema 是否有效；
- Decision Engine 与 Safe Write 决定是否 import-ready；
- 默认禁止写库，active v2 在验收前也只能 dry-run。

## 主链路

```text
PubMed JSONL
  → 并行预处理
  → 分段切块与主抽取
  → evidence-local 实体对候选图
  → predicate / NO_RELATION 分类
  → 首次确定性验证
  → Central Agent 观察状态
      ├─ cache-first Neo4j 查询
      ├─ 有界第二模型裁决
      ├─ 定向 Debug Reviewer
      ├─ evidence repair / relation recovery
      └─ Causal 与 Conflict 分析
  → 所有修改重新验证
  → Decision Engine
  → dry-run 或 Safe Write
```

## Central Agent v2

Agent v2 为每篇文章维护实体、候选实体对、验证状态、接受/拒绝/争议关系、链接歧义、
模型分歧、假设关系、预算、缓存、调用审计和终止原因。

| 路由 | 动作软预算 | 辅助远程调用 | Neo4j 查询 | 软超时 |
| --- | ---: | ---: | ---: | ---: |
| FAST | 12 | 2 | 2 | 30 秒 |
| STANDARD | 20 | 4 | 4 | 60 秒 |
| DEEP | 28 | 6 | 6 | 120 秒 |

全局硬保护为每篇最多 40 个动作、8 次辅助远程调用、8 次 Neo4j 查询和 180 秒。
缓存命中不消耗远程预算；连续两次远程调用未改变候选、验证或复核状态后立即停止。

执行模式：

| 模式 | 行为 |
| --- | --- |
| `legacy` | 默认兼容模式，原有生产结果保持不变。 |
| `agent-v2-shadow` | 记录反事实动作，不改变生产输出。 |
| `agent-v2` | 执行 v2 路由，但代码层面强制 dry-run。 |

## 证据与写入安全

任何 LLM 或工具都不能覆盖以下硬约束：

- 实体或关系 Schema 不合法；
- 端点缺失、无法解析或无法落到文章实体；
- evidence 不是原文连续片段；
- 否定、背景、研究目标或方法描述；
- 超出范围的非人类证据；
- `import_ready=false`；
- Safe Write 限制。

Causal 结果独立存入 `hypothesis_relations`。没有文章直接证据时，它只能作为研究假设，
不能进入写入路径。Conflict 可以建议 create、keep、update、dispute 或 review，
但最终权限始终属于 Verifier 与 Decision Engine。

## 安装

```bash
python3.12 -m venv .venv-cognitive
source .venv-cognitive/bin/activate
python -m pip install -r requirements-cognitive-agent.txt

cp .env.example .env
# 仅在本地填写密钥，.env 已被 Git 忽略。
set -a && source .env && set +a
```

重要环境变量：

| 变量 | 用途 |
| --- | --- |
| `GEMINI_API_KEY`、`GEMINI_API_BASE`、`GEMINI_MODEL` | 主抽取模型。 |
| `SECOND_LLM_ENABLED` | 开启有界辅助裁决。 |
| `SECOND_LLM_API_KEY`、`SECOND_LLM_API_BASE`、`SECOND_LLM_MODEL_ID` | DeepSeek/Qwen 兼容裁判模型。 |
| `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`、`NEO4J_DATABASE` | 可选图谱访问。 |
| `NEO4J_RAG_ENABLED` | 开启受限只读图谱上下文。 |
| `EXTRACTION_CACHE_MODE`、`EXTRACTION_CACHE_PATH` | 内存或 SQLite 重放缓存。 |
| `AGENT_EXECUTION_MODE`、`AGENT_BUDGET_PROFILE` | Agent 模式及质量/成本档位。 |
| `RULE_MEMORY_MODE`、`RULE_BUNDLE` | 冻结的 Agent v3 软规则记忆。 |
| `EVIDENCE_ENTAILMENT_MODE` | local-first 证据裁决。 |
| `RISK_ROUTER_MODE`、`CONFORMAL_CALIBRATION` | 选择性 conformal 风险路由。 |

## 运行方式

默认 legacy dry-run：

```bash
./run_cognitive_agent.sh 5
```

Agent v2 shadow 与持久化缓存：

```bash
export AGENT_EXECUTION_MODE=agent-v2-shadow
export AGENT_BUDGET_PROFILE=quality
export EXTRACTION_CACHE_MODE=persistent
export EXTRACTION_CACHE_PATH=.cache/agent-v2.sqlite3
./run_cognitive_agent.sh 50
```

开启 DeepSeek 条件裁决：

```bash
export SECOND_LLM_ENABLED=true
export SECOND_LLM_PROVIDER=openai
export SECOND_LLM_API_BASE=https://api.deepseek.com
export SECOND_LLM_MODEL_ID=deepseek-v4-flash
# 可设置 SECOND_LLM_API_KEY；启动器也能使用现有 DEEPSEEK_API_KEY。
./run_cognitive_agent.sh 5 --max-workers=1
```

运行原始 baseline：

```bash
python multi_stage_extraction_pipeline.py \
  --input extraction_output/pubmed_backup_liver_strict_50.jsonl \
  --limit 30 \
  --run-id v1_pipeline_pubmed30 \
  --skip-neo4j
```

## 实体本体与图谱构建

`entity_ontology/` 来自 [liverkg-entity-ontology](https://github.com/Buerchener/liverkg-entity-ontology)（`main@adef5bd`，导入时保留了提交历史）。它把抽取器找到的实体整理成去重后的概念和多父类 `is_a` 层级，现在也接上了抽取器验证过的关系：

```text
PubMed JSONL
 → 抽取（Cognitive Agent，强制 dry-run）            agent_results.json
 → 查漏 → 提及 → 概念化 → 去重 → 排序 → 建树          概念与 is_a 层级
 → 概念对齐（离线读取仓库根目录 data/ 的 v2 数据包）   concept_links.json
 → 关系投影（关系挂到概念上，跨文献合并，统一门禁）    concept_relations.json
 → 图谱导出                                          graph/*.jsonl
 → 加载到本机隔离 Neo4j（单独命令，需要确认）
```

在本目录安装并运行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ontology]'
cp entity_ontology/config.example.json entity-ontology.local.json

export LIVERKG_API_BASE="OpenAI 兼容服务地址，以 /v1 结尾"
read -rs LIVERKG_API_KEY && export LIVERKG_API_KEY

liverkg ontology run --articles inputs/articles.jsonl --run-id first-batch
liverkg ontology serve --open
liverkg ontology load --run-id first-batch --database liverkg-integrated --dry-run
```

`python -m entity_ontology …` 与 `liverkg ontology …` 等价。结果保存在 `runs/entity_ontology/<run-id>/`；同一输入、配置和批次名可以续跑。抽取器代码的哈希是批次身份的一部分，抽取器代码改动后需要换一个批次名。

只有一个 OpenAI 兼容对话模型、没有向量模型时，复制 `entity_ontology/config.single-model.example.json` 并填入模型名：

- 抽取器的第二模型和辅助裁判在没有另配 key 时复用同一个模型，运行清单里记为 `auxiliary_models: single_model`。想用独立的第二模型，把 `SECOND_LLM_API_KEY` 或 `DEEPSEEK_API_KEY` 写进 `LIVERKG_EXTRACTOR_AUX_ENV` 指向的文件。
- 去重候选改用本地字符 n-gram 哈希向量，不需要向量服务。它能找到拼写相近的同义候选，找不到毫无字面重合的同义词；是否合并仍由大模型判断。

概念级关系的门禁只降级、不升级：

| 状态 | 条件 |
| --- | --- |
| `AUTO` | 至少一条证据为 `IMPORT_READY`；没有否定、反证或统计未证实；两端概念都已放入层级；端点类型与概念类型一致。 |
| `REVIEW` | 有可用证据，但上面某一条不满足。原因写在 `gate_reasons`。 |
| `BLOCKED` | 证据全部被抽取器拦截，或谓词不在 LiverKG 范围内。 |

`novelty` 把概念级关系和 v2 数据包对照：`KNOWN_IN_V2` 表示 v2 两端之间已有同类关系，`NEW` 表示两端都在 v2 中但没有这条关系，`UNLINKED` 表示有端点没对齐到 v2。

加载到 Neo4j 的规则：只允许 localhost；库名必须显式给出，拒绝 `liver-kg-core-v02`；只用 MERGE，不删除任何数据；默认只加载 `AUTO` 关系。对齐到 v2 的概念通过 `SAME_AS` 连到 v2 节点：库里已有该节点就直接连上，没有就新建带 `:V2Anchor` 标签的锚点。社区版 Neo4j 不能建多个库，可以单独起一个本机实例，例如用 Docker 跑在 7688 端口。

示例查询：与肝纤维化相关的文献关系、v2 对照结果和证据原句。

```cypher
MATCH (x:Concept)-[r]->(d:Concept)-[:SAME_AS]->(:Disease {disease_id: 'UMLS:C0239946'})
WHERE r.source = 'literature'
MATCH (e:Evidence) WHERE e.uid IN r.evidence_uids
RETURN x.name AS concept, type(r) AS relation, r.novelty AS novelty,
       r.pmids AS pmids, collect(e.text) AS evidence
```

## 缓存与审计

抽取层提供：

- 有界进程内 L1 缓存；
- 可选的有界 SQLite L2 缓存；
- 包含正文、配置、模型与版本的缓存键；
- 并发 single-flight 去重；
- API Key 不进入缓存键或缓存值；
- 仅缓存结构合法、可重放的结果；
- attempted/successful/retried/cached、token 和延迟统计。

在完全填充的 50 篇缓存上，主抽取命中率达到 100%，没有产生新的主抽取请求，
总耗时 4.9 秒。详细结果见 [Agent v2 校准报告](docs/agent_v2_acceptance_20260814.md)。

## 评估与测试

```text
gold_annotations/  # 冻结金标
benchmark_output/  # 固定候选模型与 Router 对比
scripts/           # 评估及基准运行器
docs/              # 架构、证据策略与验收报告
tests/             # 确定性测试与集成测试
```

运行测试：

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s entity_ontology/tests -v
```

真实 Neo4j 集成测试必须显式配置隔离数据库；没有这些配置时测试会安全跳过，
不会回退连接正式数据库。

## 关键文件

| 路径 | 职责 |
| --- | --- |
| `cognitive_agent/agent.py` | 端到端执行与兼容层。 |
| `cognitive_agent/central_agent_v2.py` | 状态、预算、路由与动作审计。 |
| `cognitive_agent/verifier.py` | evidence、端点和 Schema 验证。 |
| `cognitive_agent/relation_pair_classifier.py` | BioRED-style 实体对候选与分类。 |
| `cognitive_agent/rule_memory.py` | 封闭规则 DSL、生命周期和晋升门禁。 |
| `cognitive_agent/evidence_selector.py` | 最小连续证据和蕴含闭环。 |
| `cognitive_agent/conformal_router.py` | 全局/Mondrian 非参数风险路由。 |
| `cognitive_agent/collaborative_extractor.py` | 有界第二模型裁决。 |
| `cognitive_agent/extraction_cache.py` | L1/L2 缓存与 single-flight。 |
| `entity_linking_preflight.py` | 只读实体链接预检查。 |
| `experiment_metrics.py` | 实验指标与比较。 |
| `run_cognitive_agent.sh` | 安全启动器，默认 dry-run。 |
| `entity_ontology/pipeline.py` | 从抽取到图谱导出的整批流程与断点续跑。 |
| `entity_ontology/linking/kg_links.py` | 概念与 v2 图谱节点的离线对齐。 |
| `entity_ontology/relations/project.py` | 关系投影、跨文献合并与概念级门禁。 |
| `entity_ontology/export/neo4j_load.py` | 只写本机隔离库的 Neo4j 加载。 |

## 研究状态

Agent v3 三批代码已经落地。请查看诚实披露的[中文实验状态](RESULTS_V3.zh-CN.md)或
[English results status](RESULTS_V3.md)。最终论文结论仍需专家标注和一次冻结 blind 运行；
预注册 blind cohort 不会用于调参。
