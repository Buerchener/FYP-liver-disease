# BioRED type-only 语义路由审计与修复

## 审计口径

本审计只读使用 BioRED Dev Gold 做事后 exact-match 分类。Gold 不进入候选生成、EvidencePack、模型 prompt、阈值或运行时决策。

- 原始 primary → semantic：删除了 180 条 `native_relation_card_type_only` 候选，其中 83 条是 exact TP，97 条是 exact-mismatch FP。
- factual → semantic：由于 Train-only type-signature prior 已先修正部分原始标签，实际进入验证层的 180 条中为 90 TP、90 FP。
- 97 条 primary exact-mismatch FP 又分成：44 条“实体对在 Gold 中，但预测标签错”；53 条“Gold 中没有该实体对”。后者只能称为 official exact mismatch，不能自动解释成事实错误或 Gold 错误。
- 逐项账本：`semantic_routing_audit_v2/primary_type_only_items.csv`、`primary_type_only_items.md` 和 `routing_audit.jsonl`。

## 83 条 TP 的结构

- 标签：Association 49，Positive_Correlation 26，Negative_Correlation 6，Bind 2。
- Evidence：68 条可由单个 OWNER 句闭合端点，13 条需要多个 OWNER 句，2 条端点支持仍不完整。
- 文档集中度很高：23 篇文档贡献全部 83 条；PMID 17397547 占 24 条，15069170 和 19300402 各占 11 条，21903317 占 6 条。
- 正交失败门（可重叠）：54 条存在 judge 标签冲突，63 条低于旧的统一 0.90 置信度，12 条被判为关系歧义，10 条 ABSTAIN，3 条判为 NO，1 条 span 支持不足。

这说明“83 条”不是同一种错误：

1. PMID 17397547 的 24 条大多是“多个基因被 implicated in inflammation”的列表式 Association；旧 matcher 漏掉 BioRED 允许的这种关系表达。
2. PMID 15069170 的 11 条是遗传病—变异关系；局部句只说“变异出现在患者中”，但标题、背景中的遗传机制和结论共同决定标签。只看最小 span 会系统性降成 Association。
3. PMID 19300402 的 11 条涉及基因抑制、受体、疼痛和药物诱导疾病；表面上的 increase/decrease 不是目标实体对的 BioRED sign，两个 LLM 会共享相同的直觉性极性偏差。
4. 两条 Bind 依赖 BioRED 指南中的“同一蛋白复合体成员也标 Bind”，而不是必须出现 `bind` 动词。

## 97 条被删 FP 的结构

- 44 条 wrong-label：同一官方 concept pair 有 Gold 关系，但预测 label 错。judge 推荐标签命中 Gold 的有 28 条，其中 19 条同时明确判断 relation=YES。
- 53 条 extra-pair：常见于一句话中有三个以上实体时把三角形全部补齐、基因/变异归属被误当成自由关系、两个实体共同作用于第三实体、或宽泛疾病/机制概念被当成目标端点。
- 这些 FP 中 70 条单句端点闭合、27 条多句端点闭合，说明“两个端点在证据中出现”只能证明结构闭合，不能证明关系成立。
- 两个模型对 97 条 FP 中 71 条仍给出 YES；因此单纯降低置信度或“两个模型同意就接受”无法解决问题。

## 根因

1. `EvidencePack.support_mode` 同时承担“端点闭合”和“谓词词面触发”两个问题。type-only 被写成 UNRESOLVED，导致事实证据完整却无法进入语义判定。
2. judge 每篇一次处理几十到上百候选，候选之间的 increase/decrease、第三实体和标签先验互相污染。
3. judge 只看最多三个 span；BioRED 明确允许跨句和由遗传/对应基因规则派生的关系，需要 pair-local evidence 与全文信息融合。
4. 一个 `confidence` 同时表示“关系存在”和“标签正确”。旧全局 0.90 阈值既挡住大量 TP，也保留一批高置信 FP。
5. DeepSeek 和 Qwen 使用同一抽象 RelationCard，会共享 Association ↔ signed correlation 的系统偏差；双模型一致不是独立证据。
6. BioRED 的标签边界依赖 endpoint signature。比如 disease-gene、disease-variant、chemical-disease 对同一个 `reduce` 或 `observed in patients` 的标注规则不同。

## 已实现修复

- 为 EvidencePack 增加独立的 `endpoint_support_mode/endpoint_support_span_ids/endpoint_support_closed`，不再用词面 trigger 决定端点是否闭合。
- 只按官方 mention offsets 构造最多八个 pair-local adjudication spans，同时在 prompt 中提供完整标题和摘要；span 仍是精确原文且可审计。
- 每条 review 附带 endpoint-signature 专属官方边界、合法标签和 Train-only label priors。
- label review 默认每 12 条一个微批，要求候选独立判断，避免单篇大批量串扰。
- 模型必须分别输出 `relation_confidence` 和 `label_confidence`；关系真值门默认 0.90，标签门 0.80。
- Train 中某 endpoint signature 的单一标签占比达到 90% 时，它只作为 type-only tie-breaker；共享模型偏见不能在没有 Train-only 转移校准的情况下推翻它。
- 自动模型 relabel 改为显式实验开关，默认关闭。标签冲突保留为可审计 proposal；只有未来 Train-only transition calibration 通过精度门后才允许启用。
- final relation、版本和 lineage 继续按 candidate ID/version 绑定；Neo4j 始终 BLOCKED。

## 验证结果

完整回归：389 tests passed，2 existing skips。

旧 v8 记录离线 replay（不调用模型、不启用 relabel）：

- semantic：504 TP / 360 FP / 658 FN，F1 0.4975
- replay：522 TP / 367 FP / 640 FN，F1 0.5090
- 25 条 REVIEW 被晋升：18 TP、7 FP，promotion precision 72%；0 条原 accepted 被降级。

新 v9 calibration smoke 5 cold：

- raw：16 TP / 16 FP / 23 FN，F1 0.451
- semantic：16 TP / 13 FP / 23 FN，F1 0.471
- TP 保留 100%，删除 3 FP；provider failure=0，lineage/pair accounting=1，Neo4j mutation=0。
- warm 与 cold record hash 完全相同，warm physical attempts=0。

## 从论文和高分方法得到的结论

- BioRED 官方论文和指南把关系真值、关系类型和 novelty 分开；关系类型还依赖实体类型组合及专门标注规则。
- ATLOP 的有效点不是一个更低的全局阈值，而是 entity-pair adaptive threshold 和 localized context。
- EIDER 与 DREEAM 都说明：最小证据帮助去噪，但必须与全文/文档级信息融合；只看局部 evidence 会漏掉跨句和派生关系。
- BioREx 的高分来自 Train 监督和异构数据 harmonization，不是通用 LLM 自由裁决；它适合作为可选 native-label expert，与本框架的 EvidencePack/lineage/selective verification 组合。
- BioREDirect 使用实体对标记、分块和多任务联合学习关系/novelty/方向，也支持“pair-first、task-head 分离”的改造方向。
- HTGRS 的 relation segmentation 消融在 BioRED 上损失明显，说明实体对之间的全局依赖值得保留，但不能把同句所有实体两两连接。

## 下一步硬门

1. 用 Train 构造独立 transition calibration records，校准 Association/Positive/Negative 和 specialized labels 的 edit precision；达不到 0.90 precision 的转移保持 REVIEW。
2. 跑包含 15069170、17397547、19300402、24036311 的 stress smoke，确认新 full-context/microbatch 能解决实际 83 条中的主要簇，而不是只改善普通样本。
3. 再跑 calibration20；只有 TP retention、promotion precision、provider、lineage 和零写入硬门均通过才启动新的 Dev100。
4. 论文版本应将 BioREx/BioREDirect 或同等 supervised native expert 作为强 baseline/可插拔标签头；LLM verification 的贡献应报告为证据审计、选择性接受、错误拦截和 review burden，而不是声称通用 LLM 单独超过监督 SOTA。
