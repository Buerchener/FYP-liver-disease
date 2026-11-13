# 工作台真实数据试跑

从工作台 Task 1 快照选取 14 个事件概念，涉及 7 篇文献、17 条 Event Instance 和 17 条与原文逐字对齐的证据。原始概念没有定义，试跑使用有证据引用的临时定义。本次选择偏向研究类，用于走通算法。

使用本机 Qwen3-Embedding-4B（256 维）和 Qwen3-4B-Instruct-2507（MLX 4-bit）。Average Linkage 距离阈值为 0.43，Root 权重为 0.15，Embedding 每个概念最多附一个 130 字符的证据摘录。27 次 LLM 请求和 7 批 Embedding 共耗时约 12 分钟。

| 轮次 | 待归纳概念数 | 处理结果 |
|---|---:|---|
| 1 | 14 | 4 个候选组。生成 `statistical processing` 和 `Biomedical Knowledge Integration Event`。另两组的模型回复在重试后仍未满足输入范围或子组约束。 |
| 2 | 11 | 将新概念和剩余概念重新聚类。尝试了一次更小的子组，未新增边或节点，归纳停止。 |

父类判断和归纳共提出 20 条边。全图复核对其中 12 条给出通过判断，6 条给出不通过判断，2 条因模型回复字段错误未完成复核。疑似与 `literature synthesis` 重复的 `Biomedical Knowledge Integration Event` 被隔离，连带移除其 2 条通过边。最终 DAG 保留 **1 个新中间概念、10 条暂定边、8 个可达 Root 的试跑概念**。其中 `assessment` 同时连到 Clinical Care Event 和 Biomedical Research Event，两个归属仍待审核。

未连到 Root 的 6 个概念为 `background statement`、`data processing`、`risk factor identification`、`literature synthesis`、`physiological process` 和 `intervention administration`。

## 核对中发现的问题

- 同一篇摘要的词汇会拉近不同概念。例如 `background statement` 和 `intervention administration` 聚到一组；后续复核删除了把背景陈述归入研究类上位概念的边。
- 复核识别了部分层级错误，例如把 `data collection` 挂在只涵盖已有数据集获取的 `data acquisition` 下。复核仍有误判：把 `physiological process` 对 Biological Event 的边按 Biomedical Research Event 的定义判了无效，又接受了与 `statistical analysis` 定义接近的 `statistical processing`。这些边和概念需要人工复审。
- 模型有时引用组外 ID 或写错 JSON 字段。代码记录失败和重试，对无法核实的结果保留待审状态。

结构校验无环、无悬空引用。源快照的 76 个概念、101 条 Event Instance 到 Concept 的连接、101 个 Event、101 条 Evidence 和 10 篇文献记录均通过保留检查。新增关系均为待人工审核。4 项新增回归测试和 83 项原有测试通过。
