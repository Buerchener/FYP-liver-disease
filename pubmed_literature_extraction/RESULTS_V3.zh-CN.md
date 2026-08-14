# Agent v3 实验状态（中文）

## 可复现性状态

三批代码改造已经完成。179 项确定性测试全部通过；2 项 Neo4j 集成测试只有在显式配置隔离测试库时
才会运行。200 篇开发资源已固定为 120 篇规则归纳、40 篇 validation 和 40 篇
conformal calibration。新 blind-50 已按五种研究类型各选 10 篇，标注模板仍保持
`UNLABELED`，因此没有发生 blind 标签污染。

## 目前可以报告的结果

- 默认行为仍是 legacy 兼容模式；
- 规则 bundle 解析失败时 fail closed，不能绕过 Schema、证据和 Safe Write；
- EvidenceSelector 在测试中保持连续原文和精确 offset；
- DeepSeek 只处理不确定的有界候选，Qwen 只处理冲突；
- Mondrian 分组少于 20 个样本时自动回退全局校准；
- blind PMID 与摘要 hash 有自动泄漏检查；
- 实验脚本强制要求九项消融齐全，并默认执行 10,000 次文章级 bootstrap。

## 5 篇开发集冒烟与缓存重放

这只是系统工程冒烟，不是质量 benchmark。配置为 active evidence-first pair core、active
evidence entailment、DeepSeek/Qwen、shadow conformal、dry-run，且没有 active 学习规则；
整个过程没有读取或运行预注册 blind-50。

| 指标 | 冷启动 | 完全 warm replay |
| --- | ---: | ---: |
| 总延迟 | 82.5 秒 | 0.8 秒 |
| 单篇 P95 | 26.41 秒 | 0.20 秒 |
| 主抽取缓存 | 0% | 100%（8/8 窗口；共享缓存 9/9） |
| 主抽取远程请求 | 8 | 0 |
| Evidence 辅助缓存 | 0% | 100%（5/5 批次） |
| Evidence DeepSeek/Qwen 请求 | 5 / 1 | 0 / 0 |
| 独立关系裁决请求 | 1 | 0 次新增（命中缓存） |
| 连续原文率 | 100% | 100% |
| Zero-change call rate | 0% | 0% |
| 运行错误 / 实际写入 | 0 / 0 | 0 / 0 |

冷启动与 warm replay 的实体、关系候选和最终验证关系完全一致。冷启动共选择 157 个
有界 evidence span，其中 99 个 ENTAILED、54 个 NEI、4 个 CONTRADICTED。冷启动
辅助调用合计 7 次，即 1.4 次/篇，满足内部 ≤2 次/篇的冒烟目标。但这些数字没有使用
专家 gold 评分，因此不能被解释为 semantic F1。

## 尚不能报告的结果

目前不能诚实给出最终 blind F1、precision、置信区间或显著性，因为专家标注尚未完成。
blind-50 不能用于规则学习、Prompt 调整或阈值选择；提前用它调参会直接破坏论文评估。

冻结运行后需要填写以下验收表：

| 指标 | 目标 | 当前状态 |
| --- | ---: | --- |
| Semantic relation F1 | ≥0.45 且 ≥legacy+0.03 | 等待冻结 validation/blind 运行 |
| Strict import-ready precision | ≥0.65 | 等待专家标签 |
| Evidence precision | ≥0.40 | 等待专家标签 |
| 连续原文率 | 1.00 | 代码强制且测试通过；blind 待测 |
| Dangerous writes | 0 | 代码强制且测试通过；blind 待测 |
| 平均辅助调用 | ≤2 次/篇 | 等待路由实验 |
| Zero-change calls | ≤20% | 等待路由实验 |
| Routed P95 | ≤always-call 的 70% | 等待延迟实验 |
| Warm 主抽取缓存 | 100% | 需在冻结 v3 bundle 下重放 |

## 已注册消融

最终表格必须包含 legacy、无规则 Agent v2、DeepSeek always-call、rule memory、无 Qwen、
无 EvidenceSelector、无 conformal router、无 Causal/Conflict、无缓存。若 Causal/Conflict
消融没有质量收益，将从性能贡献中删除，而不会强行包装为创新点。
