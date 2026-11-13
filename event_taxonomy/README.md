# Event Taxonomy 数据准备模块

Task 1 已实现为独立的离线模块：导入保存的抽取和概念化结果，分别保存 Event Instance、Concept Candidate、Unique Concept，定位证据并生成初始 DAG。不会调用模型、重跑抽取或运行递归归纳。

在仓库根目录运行，无需安装新的依赖，Python 3.9+：

```bash
python3 -m event_taxonomy prepare --source-root /path/to/workbench-project
```

默认使用 `configs/task1.workbench.json`。将 `--source-root` 指向包含工作台导出文件的目录，或在配置中填写 `source_root`。每次创建独立的运行目录。

其他已接通的数据源需要将配置中的 `source_root` 和 `input_files` 指向已有的对应批次。

```bash
python3 -m event_taxonomy prepare --config event_taxonomy/configs/task1.official.json
python3 -m event_taxonomy prepare --config event_taxonomy/configs/task1.original_stage2.json
python3 -m event_taxonomy prepare --config event_taxonomy/configs/task1.role_nested.json
python3 -m unittest discover -s event_taxonomy/tests -v
```

## 当前范围

| 部分 | 实现 |
|---|---|
| 数据模型 | dataclass；完整 JSON Schema 位于 `schemas/record.schema.json` |
| 输入适配 | 官方 JSONL、原版 Stage 2、角色／嵌套、工作台 graph、workspace 和单次标注导出 |
| 证据 | Unicode code point 位置；逐字、空白差异、多位置歧义、未定位和摘要级来源分别记录 |
| 概念注册 | 已有概念保留上游身份；新标签仅作 NFC/空白规范化，保留大小写；不做语义相似合并 |
| 初始图 | 复用 BIO/CARE/RESEARCH 定义，只纳入事件类型；主题另存；支持显式多父暂挂决定 |
| DAG 保护 | 环路、自环、根入边、悬空引用、重复边检查；失败回滚；保护已确认祖先关系 |
| 可复现性 | 来源字节快照、输入与代码哈希、独立运行目录、候选到概念映射及完整问题档案 |

不支持的 profile 会明确报错。上游概念 CSV 文本匹配适配器、新概念生成 provider、语义粒度审核、自动根分类与递归中间层算法仍留在后续阶段；没有将它们伪装成已实现功能。

## 输出文件

| 文件 | 内容 |
|---|---|
| `documents.jsonl` | 文档文本、PMID、来源、文本哈希 |
| `events.jsonl` | 每次抽取的事件实例，含原始 ID、运行 ID、证据及异常 |
| `evidence.jsonl` | 保留原始引文与定位结果，不能定位的引文不改写 |
| `concept_candidates.jsonl` | 每次标签关联，保留类型／主题／未知、旧生成信息与 stale 状态 |
| `unique_concepts.jsonl` | 去重登记的全部角色概念；不代表语义已验证 |
| `candidate_concept_mapping.json` | 候选到 Unique Concept 的可追溯映射；过期候选仍有归档记录 |
| `initial_taxonomy.json` | 三根、全部输入事件类型概念、候选实例关联、暂挂边与 unplaced 列表 |
| `prepared_input_snapshot.json` | 内容冻结快照，明确 `eligible_for_induction=false` |
| `concept_review_queue.json` | 定义、粒度、类型有效性及父类归属待审查项 |
| `concept_generation_records.json` | 未生成、明确空标签、已生成与过期分别记录 |
| `import_issues.json` | 原导出拒绝记录及本模块发现的概念问题 |
| `validation.json`, `report.md` | 结构验证与数据准备统计，非算法效果 |
| `manifest.json`, `sources/`, `checksums.json` | 配置、源文件快照、代码与产物哈希 |
| `code_snapshot/`, `replay_config.json` | 当前模块代码和仅依赖本次来源快照的重放配置 |

默认不生成猜测的父类归属。可通过 `assignments_file` 指定 JSON 数组，每项需包含 `concept_id`、`parent_id`、`status`、`decision_id`、`reason`、`validator`；status 为 provisional 或 accepted。accepted 还要求概念已有定义。格式校验不能证明该决定语义正确。

使用 `python3 -m event_taxonomy prepare --config <运行目录>/replay_config.json` 可从该次保存的输入重新生成结果，无需原始网站项目仍在原位置。`code_snapshot` 保留当时模块的完整 Python 源码，避免仅有代码哈希却无法恢复实现。

本批次 76 个类型标签尚未完成定义、粒度和类型有效性审核，初始图为 3 根加 76 个未归属概念，父子边为 0，状态为 partial。全部概念和原始支持关系保留；它是后续审核与归纳的输入准备，不是已经构建完成的分类体系。

根配置、旧实验及网站均只读。根以外的药物反应小试节点不加入 seed。Task 1 的测试数据均标记 Synthetic Data，真实运行使用独立配置与目录。

原检查与后续算法／评测设计见 [TASK1_DESIGN.md](TASK1_DESIGN.md)。

## 本次验证

本地工作台批次 `task1-workbench-final-20261112` 包含 101 条事件、367 条概念候选，登记为 76 个事件类型概念及 176 个主题概念；类型概念全部保留在初始图中。

另外分别重放了官方 5 篇批次的 45 条事件、原版 10 篇的 117 条事件和角色／嵌套版的 97 条事件，保持各方法独立。28 项单元测试通过，1,374 条导出记录通过 JSON Schema 验证；保存代码与来源快照在独立工作目录重放，核心输出逐字节一致。578 个既有源码和实验文件哈希未改变。

这些检查涵盖数据准备与结构保护。真实概念的语义仍待审核。
# Task 2：递归中间概念归纳

已新增独立实现，运行说明、算法顺序、配置和恢复机制见 [TASK2_IMPLEMENTATION.md](TASK2_IMPLEMENTATION.md)。

```bash
python3 -m event_taxonomy induce --run-id task2-demo
```

默认使用 **Synthetic Data + Mock LLM**。新的自底向上实现见 [Root-Constrained Event Taxonomy](../root_event_taxonomy/README.md)。
