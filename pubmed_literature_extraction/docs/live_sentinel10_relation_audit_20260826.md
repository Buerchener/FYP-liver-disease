# Live Sentinel-10 逐关系 Gold 对照审计（2026-08-26）

## 审计边界

- 审计对象：同一组 10 篇 stability sentinel 的本轮 live 产物。
- 运行产物：`/private/tmp/pubmed_live_sentinel10_20260826_DszEYd/output/agent_results_live_sentinel10_20260826.json`。
- Gold 视图：`gold_annotations/pubmed_200_gold_v2_candidate_view.jsonl`。
- 本报告审计了两层输出：主模型的 13 条 raw relation，以及进入最终 CandidateRelationStore 的 12 条关系（3 条 `ACCEPTED`、9 条 `REVIEW`）。
- 本报告不改 Gold、不把 Gold 文本写入 prompt/few-shot、不据此调阈值，也不启动 Neo4j。

## 先给结论

Gold 不是完全没有问题，但当前低召回主要是系统问题。

1. 10 条 Gold 关系中，最终只有 2 条按现有 typed/directed triple 口径匹配；8 条漏掉。
2. 这 8 条 FN 里，至少 7 条在实体对候选阶段已经出现或出现了非常接近的候选，说明新的跨句/配对逻辑有进步；主要损失已转移到证据选择、Claim Gate、Predicate Judge 和端点规范化。
3. `semantic F1=0.308` 只计算 3 条 `ACCEPTED`，完全忽略 9 条 `REVIEW`。如果把 Candidate Store 保留的 12 条都按当前 Gold 口径计入，结果是 TP=2、FP=10、FN=8，P/R/F1=0.167/0.200/0.182。
4. 6 篇 Gold 零关系论文中，自动接受层是 6/6 正确；但 Candidate Store 层是 5/6，因为肺纤维化论文仍保留了一条肝外关系。因此 `zero-relation specificity=1.0` 只说明自动接受层安全，不能证明 review queue 干净。
5. 3 条 `ACCEPTED` 实际对应两个去重后的有效语义关系：NAFLD-HCC 和 GH-hepatic steatosis。现有 FP 是 NAFLD-HCC 的反向重复和有向评分口径共同造成的，不是一个完全虚假的自动接受关系。
6. 所有最终关系方向均为 `unknown`。即使两条 triple TP 的 Gold 方向分别是 `positive`、`decrease`，方向也没有保留下来。
7. Gold 至少有两条很可能漏标的语义关系，另有三处需要专业标注者复核；但这些问题不足以解释系统 8/10 的 Gold 漏召回。

## 评测口径本身的问题

当前仓库存在两个不一致的评测口径：

- `scripts/run_agent_v3_ablation100.py` 只把 `semantic_status=ACCEPTED` 当正预测，`REVIEW` 是 abstention。
- `scripts/evaluate_gold200_unified.py` 将所有未 `REJECTED` 的关系（包括 `REVIEW`）计入 candidate semantic prediction。
- Pair classifier 和 Pairwise Judge 将 `ASSOCIATED_WITH` 视为对称谓词，但统一评测和协作去重只把 `INTERACTS_WITH` 视为对称谓词。

因此必须同时报告：

| 口径 | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 自动语义接受（仅 `ACCEPTED`） | 2 / 1 / 8 | 0.667 | 0.200 | 0.308 |
| Candidate Store 保留（`ACCEPTED+REVIEW`） | 2 / 10 / 8 | 0.167 | 0.200 | 0.182 |

`REVIEW` 不应等同自动正预测，但它必须有单独的候选覆盖率、review precision 和 review burden 指标，不能从主报告中消失。

## 13 条 raw relation 逐条审计

### PMID 41794448（Gold=0）

1. `Nrf2[Protein] PARTICIPATES_IN Nrf2 pathway`
   - 判定：系统问题，Gold 正确。
   - 原文只说 Nrf2 pathway 被激活，没有直接陈述“Nrf2 蛋白参与以自己命名的 pathway”。这是从路径名称拆出蛋白后制造的近似自环。
   - 结果：后续正确删除。
2. `p38 MAPK[Protein] PARTICIPATES_IN p38 MAPK pathway`
   - 判定：同上，是路径名称自拆造成的伪关系。
   - 结果：后续正确删除。

### PMID 41650163（Gold=6）

3. `OTUD5[Gene] ASSOCIATED_WITH PBC`
   - 判定：正确，对应 Gold G1。
   - 问题：raw 已经有完整 Results 证据，后续却换成残缺的 `PBC. High expression of OTUD5`，最终丢失。
4. `MAVS[Protein] ASSOCIATED_WITH PBC`
   - 判定：正确，对应 Gold G2。
   - 问题：同样在后续证据选择/谓词裁决丢失。
5. `OTUD5[Gene] ASSOCIATED_WITH PBC`（第二条 Conclusion 证据）
   - 判定：语义正确，但与第 3 条重复。
   - 问题：raw 层没有及时按 canonical endpoint+predicate 合并多证据。
6. `OTUD5[Gene] EXPRESSED_IN mononuclear macrophages`
   - 判定：接近 Gold G4，但端点过宽。
   - Gold 是 `macrophage subset 11`；原文是 `subpopulation 11 mononuclear macrophages`。系统丢掉了 subset 11 这一关键限定。
7. `MAVS[Gene] ASSOCIATED_WITH PBC`
   - 判定：关系概念接近 Gold G2，但 MAVS 被错误重复成 Gene；同一篇中已有正确的 Protein 实体。
   - 问题：长名称 `mitochondrial antiviral signalling protein (MAVS)` 没有形成可靠的文章内类型约束。

raw 层还漏了 Gold G3 `OTUD5 INTERACTS_WITH MAVS`、G5 `MAVS EXPRESSED_IN macrophage subset 11`、G6 `PBC ASSOCIATED_WITH NK cell activation`。其中 G3/G5 后来由 pair generation 找到近似候选，G6 因 `NK cell activation` 被抽成 `NK cell[CellType]` 而在实体层就丢了。

### PMID 41902413（Gold=0，肝外论文）

8. `Angio-3 ASSOCIATED_WITH Pulmonary fibrosis`
   - 判定：在原文内部大体成立，但对本肝病项目是 out-of-scope；Gold 的项目范围判定正确。
9. `Angio-3 INTERACTS_WITH PAR-1`
   - 判定：原文有 `PAR-1 antagonist` 支持，但同样是肝外关系；不应进入肝病 candidate 指标或普通 review queue。

### PMID 41581151（Gold=1）

10. `Wnt9b[Protein] EXPRESSED_IN liver endothelial cells`
    - 判定：原文明确说 Endo4 liver endothelial cells 是 Wnt9b 的 source，语义成立。
    - Gold 问题：candidate semantic Gold 很可能漏了这条关系。
    - 系统问题：对象应保留为 Endo4 subtype；而且这条有效 raw relation最终也被丢掉。

### PMID 41719003（Gold=2）

11. `growth hormone ASSOCIATED_WITH MASLD`
    - 判定：原文直接说有 MASLD 的 acromegaly 患者 GH 更低，候选语义成立。
    - Gold 问题：Gold 很可能漏标；至少应由独立标注者复核。
12. `GH ASSOCIATED_WITH MASLD`
    - 判定：与第 11 条是缩写重复，系统应合并。
13. `GH ASSOCIATED_WITH hepatic steatosis`
    - 判定：正确，对应 Gold G2；这是最终保留下来的 TP。

## 12 条最终 Candidate Store 关系逐条审计

### PMID 41650163

1. `MAVS[Protein] EXPRESSED_IN mononuclear macrophages` — `REVIEW`
   - 语义：基本成立，接近 Gold G5。
   - 系统问题：丢失 `subpopulation 11/macrophage subset 11`，方向也从 `increase` 变成 `unknown`。
   - Gold 不是主要问题；这是端点边界/规范化问题。
2. `OTUD5[Gene] EXPRESSED_IN mononuclear macrophages` — `REVIEW`
   - 语义：基本成立，接近 Gold G4。
   - 系统问题：同上。

### PMID 41902413

3. `Angio-3[Protein] PARTICIPATES_IN FXa-PAR-1 signaling axis` — `REVIEW`
   - 语义：在肺纤维化论文内部可作为候选，但不是肝病范围内关系。
   - 系统问题：已有 `article_out_of_scope` 标志却仍写入普通 Candidate Store，污染 candidate specificity 和人工审核队列。
   - 安全层表现尚可：没有自动接受、没有 Neo4j 写入。

### PMID 41581151

4. `Wnt9b[Protein] INTERACTS_WITH Sfrp2[Protein]` — `REVIEW`
   - 原文：`Prediction of cell-cell communication identifies a Wnt9b-Sfrp2 crosstalk`，随后有 perturbation。
   - 判定：候选语义合理，但仅凭 abstract 对 Gene/Protein 类型的证据不足；Gold 的保守排除是可辩护的。
   - 系统表现：留在 `HUMAN_REVIEW` 是合理的，不能自动接受。

### PMID 41475279

5. `HCC[Disease] ASSOCIATED_WITH liver[Tissue]` — `REVIEW`
   - 判定：错误，Gold 正确。
   - 根因：把 `chronic liver conditions` 中的 `liver` 子串当独立 Tissue 节点，再把 `linked to` 错连到 Tissue。
6. `HCC ASSOCIATED_WITH NAFLD` — `ACCEPTED`
   - 判定：关系语义正确，对应 Gold；但与第 10 条是对称重复。
   - 系统问题：文章是 narrative review，候选初始 role 已是 `BACKGROUND`，Pairwise Claim Gate 却提升成 `CURRENT_FINDING`。
7. `HCC ASSOCIATED_WITH viral hepatitis` — `REVIEW`
   - 判定：原文直接支持，Gold 很可能漏标。
   - 应保留为 review/background candidate，不能当本文新发现。
8. `liver[Tissue] ASSOCIATED_WITH NAFLD` — `REVIEW`
   - 判定：错误；由 `liver conditions` 的错误 Tissue 实体和宽泛触发词传播产生。
9. `liver[Tissue] ASSOCIATED_WITH viral hepatitis` — `REVIEW`
   - 判定：错误；同上。
10. `NAFLD ASSOCIATED_WITH HCC` — `ACCEPTED`
    - 判定：正确，是 Gold 的记录方向；与第 6 条重复。
    - 系统问题：仍被错误标成 `CURRENT_FINDING`。
11. `viral hepatitis ASSOCIATED_WITH HCC` — `REVIEW`
    - 判定：与第 7 条是对称重复；语义成立但 Gold 漏标概率高。

### PMID 41719003

12. `growth hormone ASSOCIATED_WITH hepatic steatosis` — `ACCEPTED`
    - 判定：正确，对应 Gold G2。
    - 系统问题：明确的 `inversely associated` 没有转成 `direction=decrease`，最终为 `unknown`。

## 10 篇逐篇 Gold 判定

| PMID | Gold 关系 | 最终保留 | Gold/系统结论 |
|---|---:|---:|---|
| 41810002 | 0 | 0 | Gold 零关系合理；系统正确。 |
| 41799192 | 0 | 0 | 心脏移植论文，Gold 范围判定合理；系统最终正确。 |
| 41620901 | 0 | 0 | 系统与 Gold 一致，但 Gold 的“计算预测全部为零”与 README 中可保留 computational semantic candidate 的政策存在张力，需复核而非直接改标。 |
| 41794448 | 0 | 0 | Gold 合理；raw 的两条 pathway 自拆关系是系统错误，后续正确清除。 |
| 41650163 | 6 | 2（均 near-match review） | Gold 六条大体有证据；系统丢失主关系并过度泛化 cell subset。G3 的 `import_ready=true` 可能过强，因为 Methods 说明 interaction 来自数据库预测，需复核写入标签。 |
| 41573193 | 0 | 0 | Gold 零关系合理；治疗药物/结局不适配当前实体与谓词。 |
| 41902413 | 0 | 1 review | Gold 的肝病范围判定合理；系统应将肝外关系隔离而非放入普通 review queue。 |
| 41581151 | 1 | 1（非 Gold 对） | Gold Endo4-stellate 关系符合项目的 cell-cell interaction 语义；系统错误丢弃。Gold 可能另漏 Wnt9b EXPRESSED_IN Endo4。 |
| 41475279 | 1 | 7 | Gold 漏了 viral hepatitis-HCC；系统同时产生 3 条 Tissue 伪关系、2 组反向重复，并把 review background 提升为 current finding。 |
| 41719003 | 2 | 1 | Gold 两条核心结论合理；系统漏掉 acromegaly-steatosis。Gold 可能另漏 GH-MASLD。 |

## Gold 需要复核的项目

### 高可信漏标

1. PMID 41475279：`viral hepatitis ASSOCIATED_WITH HCC`
   - 与已标的 NAFLD-HCC 出现在同一个 `has been closely linked to ... such as ...` 句子中。
   - 不应因后文 `HBV/HCV` 是 composite mention 而删除前文合法的 `viral hepatitis` Disease 实体。
2. PMID 41581151：`Wnt9b EXPRESSED_IN Endo4 liver endothelial cells`
   - `Endo4 ... as the source of Wnt9b` 是直接、连续、可回源的语义支持。

### 较高可信、需独立复核

3. PMID 41719003：`Growth hormone ASSOCIATED_WITH MASLD`
   - 结果段有显著差异，但需决定 MASLD 与 hepatic steatosis 是否在本 Gold 中作为不同终点重复标注。
4. PMID 41620901：computational prediction 是否应在 candidate semantic view 中保留
   - 当前行以“不是 import-ready fact”为由把 semantic relations 也置空，与 Gold README 的层级政策不完全一致。
5. PMID 41650163：`OTUD5 INTERACTS_WITH MAVS` 的 `import_ready=true`
   - 关系可作为 semantic candidate，但 interaction 的主要来源带数据库 prediction 成分，自动写入标签可能过强。

Gold 仍不是 publication-grade 专家标注；这些项目应由不看系统输出的独立标注者按原文复核。不能因为系统生成了它们就直接改 Gold。

## 系统问题的根因排序

### P0-1：EvidenceSelector 只找“最短端点+任意触发词”，不验证触发词属于当前实体对

`cognitive_agent/evidence_selector.py` 的最短窗口遍历所有 subject/object/trigger 组合并按长度取最小。它没有验证 trigger 的语法主客体。

直接后果：

- acromegaly-steatosis 候选选成 `acromegaly ... GH was inversely associated with hepatic steatosis`，其中 association 实际连接 GH，而不是 acromegaly。
- `liver conditions` 中的 `linked to` 被错误分配给 HCC-liver、liver-NAFLD、liver-viral hepatitis。
- Pairwise Judge 看到被污染的窗口后做出看似合理但针对错误实体对的决定。

对策：

1. 触发词必须与两个端点同 clause，并优先位于两个端点之间或满足 predicate-specific 模板。
2. 若 raw relation 已提供可回源、覆盖端点的证据，应优先使用该证据，不要被更短但语义错误的窗口替换。
3. 一个触发词被分配给实体对时，应排除它在更近的第三实体局部结构中已经形成另一关系的情况。
4. 对无法确定 trigger attachment 的候选标记 `trigger_attachment_ambiguous` 并送 review，不自动 `ENTAILED`。

### P0-2：Pairwise Judge 可以覆盖更可靠的 deterministic role 和 evidence

现象：

- PMID 41475279 的 candidate role 原本是 `BACKGROUND`，Stage A 改成 `CURRENT_FINDING`，最终自动接受。
- PMID 41581151 的 Endo4-stellate candidate 原本有覆盖两个端点的跨句证据，Judge 生成了局部 quote，随后 deterministic endpoint check 以 `endpoint_not_in_evidence` 删除真关系。

对策：

1. Claim role 使用单调规则：模型可以把 `CURRENT_FINDING` 降为非当前角色，但不能把 deterministic `BACKGROUND/PRIOR_WORK/METHOD/PREDICTION` 提升为 `CURRENT_FINDING`。
2. Judge quote 只有在“原文精确可定位 + 两端点/显式别名都在 quote + predicate trigger 在 quote”时才能替换 candidate evidence。
3. Judge quote 不合格时保留原 candidate evidence，并送 `HUMAN_REVIEW`；不要先改坏证据再触发端点硬删除。
4. `endpoint_not_in_evidence` 必须区分“端点根本不在原文”与“局部 quote 边界没覆盖端点”。前者仍是硬门，后者是 evidence repair/review。

### P0-3：`INTERACTS_WITH` 定义没有按端点类型区分

协作裁决 prompt 把 `INTERACTS_WITH` 统一定义为 molecular binding/interaction。于是 reviewer 将 CellType-CellType 的 `juxtaposing/cellular crosstalk` 当作普通共现并拒绝 Endo4-stellate。

对策：

- Protein/Gene 端点：要求 binds/interacts/complex 等分子触发。
- CellType-CellType：允许明确的 cellular crosstalk、cell-cell communication、regulatory interaction、具备功能上下文的 juxtaposition。
- 仍不允许简单同句共现或仅列出两种细胞。

### P0-4：实体边界和文章内规范化仍然不够

已观察到：

- `Endo4` 没被主模型抽为实体，现有 deterministic subtype linker 因“不创建实体”而无法生效。
- `subpopulation 11 mononuclear macrophages` 被缩成过宽的 `mononuclear macrophages`。
- MAVS 同时成为 Gene 和 Protein。
- `NK cell activation` 被缩成 `NK cell[CellType]`，Pathway 终点消失。
- `chronic liver conditions` 内的 `liver` 被误建成 Tissue。

对策：

1. 当原文存在 `specific CellType termed/called "Label"` 时，允许 deterministic recovery 创建带原文 span 的 subtype entity；这不是凭空发明。
2. cell subset canonicalization 必须保留数字/cluster/subpopulation 限定，不得合并到父细胞类型。
3. 用 `long protein name (SYMBOL)` 的父括号定义锁定文章内 Gene/Protein 类型；冲突类型进入 ambiguity/review，不双开普通实体。
4. 对 `X cell activation` 等精确过程短语补充 Pathway recovery。
5. Tissue 单词若只是 `liver disease/condition/cancer` 的内部修饰语，不应独立成 Tissue 实体；必须有解剖语境。

### P1-1：反向 pair 在谓词确定后没有做最终对称去重

Disease-Disease 的 allowed predicates 同时含对称 `ASSOCIATED_WITH` 和有向 `PROGRESSES_TO`，所以生成阶段必须保留双向候选；但当两边最后都预测为 `ASSOCIATED_WITH` 后，没有再按最终谓词去重，造成两组重复。

对策：在 Pairwise Judge 之后、Verifier 之前按“最终 predicate”做一次去重；仅对真正对称的已选谓词生效，不影响 `PROGRESSES_TO`。

同时必须统一项目契约：如果 `ASSOCIATED_WITH` 继续按有向角色评分，就不应在 classifier 中称为 symmetric；如果语义上视为无向，就应统一 evaluator、Candidate Store 和协作去重。

### P1-2：article profile 误分类影响模型裁决

本轮 profile 使用 `legacy_rules` 且 confidence=0.0，出现：

- HCC hydrogel review → `clinical`
- PBC human validation+cell experiment → `computational`
- gut microbiota narrative review → `mechanistic`

对策：

1. 先用标题/`This review`/structured abstract 等高精度规则识别 review。
2. 支持 mixed modalities，不用单一 `computational` 覆盖后续 human validation。
3. confidence=0 的 study type 只能作路由提示，不能覆盖 evidence section 与 deterministic claim role。

### P1-3：模型调用量高但有效产出率低

本轮 150 个 pair candidates 被 Pairwise 两阶段累计请求 234 次，消耗约 80,868 prompt tokens 和 22,589 output tokens、累计 judge latency 281.7 秒。最终只有 3 条 `ACCEPTED`、9 条 `REVIEW`。

对策：

1. out-of-scope article 在 pair judge 前分流到独立 scope bucket。
2. 优先 judge extractor-hinted、trigger-anchored、跨句定义链接三类候选；普通共现不送远程模型。
3. 使用实体去重、context-aware Tissue 过滤和最终谓词对称去重降低重复调用。
4. 已知 deterministic rollback 条件的候选不要先交给 Qwen approve 后再回滚。

### P1-4：方向信息在流水线中丢失

raw relation 中的 `increase/decrease` 和 source direction hints 没有传播到最终关系，全部变成 `unknown`。

对策：当 predicate/endpoints 未改变且证据含 increase/decrease/inverse 等一致 cue 时，保留 raw/source direction；模型只能在有反证时覆盖。

### P1-5：审计字段自身有误导性

Claim Gate 的 `relation_asserted` 可以是 `ASSERTED`，但 `claim_status` 审计字段仍被规范成 `NO_EXPLICIT_RELATION`，导致本轮所有 gate count 都显示 `NO_EXPLICIT_RELATION`。实际控制流用的是 `relation_asserted`，所以主要是 audit bug，但会严重误导诊断。

对策：删除旧 `claim_status` 的双重语义，或从 `relation_asserted|claim_role` 明确派生；报告只显示一致的一套字段。

## 建议实施顺序

1. 先修评测与审计口径：分别报告 ACCEPTED、REVIEW coverage、IMPORT_READY；统一 symmetric contract；修 claim_status audit。
2. 修 EvidenceSelector 的 trigger attachment，并保护 raw exact evidence。
3. 修 Pairwise role/evidence 的单调合并，确保模型不能把 background 提升为 current，也不能用坏 quote 替换好证据。
4. 为 CellType-CellType interaction 使用类型化语义定义。
5. 完成 Endo4/subpopulation/MAVS/NK activation/liver-context 五类实体边界修复。
6. 在最终 predicate 已知后做对称去重，并恢复 direction propagation。
7. 将 out-of-scope candidates 分流，减少 DeepSeek/Qwen 调用和人工队列污染。

## 不使用 Gold 调参的验证方案

每个修复先用人工构造的最小单元测试验证一般规则，而不是复制这 10 篇 Gold 句子：

- 触发词属于第三实体时，不能错配当前 pair。
- deterministic BACKGROUND 不能被模型提升为 CURRENT_FINDING。
- Judge quote 缺一个端点时保留原证据并进入 review。
- CellType-CellType 的 explicit crosstalk 可形成 candidate；普通共现不可。
- `termed/called` subtype recovery 保留 subtype 标识。
- disease phrase 内部的 anatomical modifier 不生成独立 Tissue。
- mixed Disease-Disease shortlist 最终选中 ASSOCIATED_WITH 后去重，选中 PROGRESSES_TO 时保留方向。
- increase/decrease/inverse direction 能贯穿到最终输出。

完成一般化测试后再重跑同一 10 篇 sentinel。下一次门槛建议同时满足：

- ACCEPTED semantic F1 明显高于 0.308；
- Gold pair candidate coverage 不下降；
- Candidate Store 保留层的 FP/review burden 明显下降；
- 背景关系 0 条被标为 CURRENT_FINDING；
- 反向对称重复 0；
- zero-relation specificity：ACCEPTED=1.0，Candidate Store scope-aware=1.0；
- dangerous writes=0、Neo4j writes=0；
- 未达标前不启动 Gold200。

## 2026-08-27 优化后同批复验

复验严格使用相同 10 个 PMID，继续关闭 Neo4j，并复用主抽取缓存；没有把
Gold/BioRED 文本放入 prompt、few-shot、规则学习或阈值调参。

### 结果

| 指标 | 结果 |
|---|---:|
| 单元测试 | 299 passed，2 skipped，0 failed |
| 文章成功率 | 10/10 |
| Candidate semantic（仅 `ACCEPTED`） | TP=4，FP=0，FN=6，P=1.000，R=0.400，F1=0.571 |
| Candidate coverage（`ACCEPTED+REVIEW`） | TP=4，FP=1，FN=6，P=0.800，R=0.400，F1=0.533 |
| Main-KG contract | TP=1，FP=0，FN=1，F1=0.667 |
| Strict import-ready | TP=1，FP=0，FN=1，F1=0.667 |
| Gold 零关系论文 | 6/6 保持零自动接受关系 |
| Pair candidates / Judge requests | 140 / 33 |
| 最终关系 | 5（4 `ACCEPTED`，1 `REVIEW`） |
| Dangerous writes / 实际 Neo4j writes | 0 / 0 |

相对任务起始状态的 candidate semantic F1=0.133，本轮 F1=0.571。唯一
`IMPORT_READY` 关系是 `OTUD5 EXPRESSED_IN macrophage subset 11` 的可回源
等价表述，命中 strict gold；由于 dry-run，实际写入仍为 0。

### 五条最终关系逐条结论

1. `OTUD5 EXPRESSED_IN subpopulation 11 mononuclear macrophages`：正确，命中
   Gold；`high expression` 的 increase 方向现在可确定，且编号亚群等价表述可正确评分。
2. `OTUD5 INTERACTS_WITH MAVS`：正确，命中 Gold；结果/结论中的显式 interaction
   不再被 Claim Gate 错降为 METHOD。该类型不在冻结主库九类签名内，因此仅语义接受，
   不可写入。
3. `Wnt9b EXPRESSED_IN liver endothelial cells`：原文支持但 Gold 未标，继续留在
   `REVIEW`；系统仍未恢复 Gold 的 Endo4 endothelial-stellate 跨句关系。
4. `HCC ASSOCIATED_WITH NAFLD`：正确，命中 Gold；claim role 保持 `BACKGROUND`，
   候选语义可接受但由 `non_current_finding_role` 阻止写入。
5. `growth hormone ASSOCIATED_WITH hepatic steatosis`：正确，命中 Gold，
   `inverse` 方向保留为 decrease；冻结主库签名不匹配，所以不写入。

### API 与剩余风险

- Gemini 主模型 canary 可用；DeepSeek canary 与本轮受限裁决可用。
- 第三方平台的 `qwen3.5-flash` canary 返回 HTTP 503 `model_not_found`；本次最终
  复验没有触发 Qwen critic，不能把它报告为可用。
- 仍漏 6 条 Gold：重点是 PBC 相关的 disease association、MAVS 的第二条
  expression、PBC-NK activation，以及 Endo4-stellate、acromegaly-steatosis。
- 方向一致率仍受 `unknown` 与 Gold `none/positive` 的口径差异影响。
- 当前结果只证明同一 10 篇 development sentinel 过门；尚未授权启动 Gold200。
