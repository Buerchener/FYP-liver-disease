# PubMed 文章级工具路由 Agent

## 目标

工具路由器让 Cognitive Agent 根据每篇 PubMed 的研究类型、文本复杂度和初次验证结果决定工具调用，而不是对每篇文章执行完全相同的昂贵链路。路由器本身是无状态规则组件，不调用 LLM，适合多线程共享。

所有路径都保留以下安全主链：

`缩写检测 → LangExtract 候选生成 → 确定性验证 → DecisionEngine`

路由只控制可选工具：预抽取记忆、后置 Neo4j RAG、第二 LLM、调试 Reviewer、因果推理和冲突解析。Neo4j RAG 仍然只能用于链接、类型、别名和冲突提示，不能作为当前文章 evidence。

## 四层影子路由（v10）

新版把一次路由拆成四个可审计层次：

1. **安全硬掩码**：证据和写库边界拥有最终否决权；综述、纯预测等文章不能因效用分高而进入文章级因果推理或开放式恢复。
2. **候选工具池**：用便宜画像和多维复杂度缩小工具范围；进入候选池不等于实际调用。
3. **充分性判断**：确定性验证后再读取 import-ready、硬错误、语义风险、明确遗漏，以及实体链接候选的 Top-1/Top-2 分差。
4. **净效用门控**：只有未被掩码、属于候选池且“预期质量增益－延迟－费用－安全风险”达到阈值时才调用工具。

当前效用值是可解释的规则先验，不冒充校准概率。后续可以用冻结金标注和 `tool_marginal_benefit` 轨迹拟合轻量模型。`FAST/STANDARD/DEEP` 仅保留作兼容和审计标签；`DEEP` 不再表示固定调用一整套远程工具。四层策略目前保持影子模式，生产执行仍用 `legacy`。

## 两阶段决策

### 抽取前

路由器先用毫秒级规则识别 review、computational、clinical、human_omics、animal、in_vitro、mechanistic 或 other，并产生三类路径：

| 路径 | 典型文章 | 调用策略 |
|---|---|---|
| FAST | 无结果性陈述的综述、纯计算/预测文章 | 不预查图；先用主抽取器和确定性门控 |
| STANDARD | 普通临床、关联或机制摘要 | 不重复预查图；抽取后再判断是否需要记忆 |
| DEEP | 高实体密度、高机制密度的复杂实验文章 | 最多对 8 个候选提及做预抽取记忆激活 |

路由还向 LangExtract prompt 添加研究类型约束，例如计算文章的预测关系保持 uncertain、动物/体外结果不得静默泛化到人类、临床关联不得改写成分子因果。

### 初次验证后

路由器读取候选的质量标记，再决定昂贵工具：

- `neo4j_rag`：只围绕待复核关系的端点检索小型上下文；图谱上下文只能帮助链接、类型和冲突判断。
- `second_llm_refiner`：只接收端点已落在原文、类型签名合法且仍有语义风险的候选。端点缺失、schema 错误、方法/预测、非连续 evidence 等硬错误直接删除，不让 LLM 猜救。
- `causal_reasoner`：只有存在 import-ready 关系时调用。
- `conflict_resolver`：只有存在关系候选时调用。
- `debug_reviewer`：启用后也只审查 DEEP/RECOVERY 文章。
- `RECOVERY`：主抽取失败时隔离并记录，不允许第二模型从空结果开放生成事实。

第二模型采用非思考模式，输出不设置客户端 `max_tokens` 上限，并且只有
`KEEP / REJECT / CHANGE_PREDICATE / CHANGE_DIRECTION / CHANGE_EVIDENCE` 五种动作。
它不能新增实体、端点或关系；任何修改仍须重新经过原确定性 verifier。
即使第二模型跳过或失败，确定性删除也必须重新验证后才能进入决策层。

## 审计与批处理

每篇结果增加：

- `phases.tool_plan_pre`
- `phases.tool_plan_post`

二者记录 route、study type、reason codes、每个工具的 `CALL/SKIP/DEFER`、原因和成本等级。批报告中的 `tool_router.route_counts` 和 `tool_router.tool_call_counts` 汇总路径和实际计划调用数。

`phases.reader` 还记录结构化 section、句子和保留原字符偏移的 clause evidence
unit；关系 evidence 必须选择这些原文片段，不能由模型改写。

## 配置

- 默认启用工具路由。
- `--disable-tool-router`：恢复兼容模式。
- `--router-pre-context-max-mentions N`：设置预抽取记忆查询上限，默认 8。
- `--neo4j-rag-enabled`：允许路由器在验证后调用只读 RAG。
- `--second-llm-enabled`：允许路由器调用第二模型。
