# Root-Constrained Event Taxonomy Induction

独立入口 `python -m root_event_taxonomy` 读取 `event_taxonomy` 的 Task 1 导出目录或同结构的 JSON。它沿用项目的 Concept、Event、Evidence 和 Taxonomy 数据结构。原始概念、实例关系和证据映射随结果保留。选择试跑范围时，范围之外的概念也保留原状。

## 事件抽取

终端入口 `python3 -m root_event_taxonomy.extract` 使用 [XML 提示词](prompts/event-extraction.xml)。每次读一篇完整摘要，提取 `Event`、`Evidence` 和 `Trigger`。输入为 JSONL，每行包含字符串 `pmid` 和 `abstract`。

用 API 抽取时，在配置文件的 `llm.provider` 下填写：

```json
{
  "kind": "openai_compatible",
  "base_url": "https://YOUR_ENDPOINT/compatible-mode/v1",
  "api_key_env": "LIVER_EVENT_EXTRACTION_API_KEY",
  "model": "qwen3.7-flash",
  "extra_body": { "enable_thinking": false }
}
```

先在终端环境中提供对应的 API 密钥，再运行：

```sh
python3 -m root_event_taxonomy.extract \
  --input /path/to/abstracts.jsonl \
  --output event_taxonomy/runs/xml-events \
  --model-config /path/to/extraction-api.json \
  --max-tokens 16384
```

`events.json` 保存事件及其 ID。模型选原句编号，程序按编号复制原文到 Evidence，并检查 Trigger 是否是证据中的连续原词。`concept_inputs.json` 给概念化准备事件描述 `source_label`、原句 `context`、触发词 `trigger` 及用于回连的 `source_id`。未通过的记录保存在 `rejected.json`，逐篇处理结果在 `report.json`。完整性和 Trigger 的语义仍可对照摘要审核。

## 事件概念化

沿用抽取时的 API 配置。模型读取 Event、Trigger 和原句，为每条 Event 生成一个概念名和一句定义：

```sh
python3 -m root_event_taxonomy.conceptualize \
  --input event_taxonomy/runs/xml-events/concept_inputs.json \
  --output event_taxonomy/runs/xml-event-concepts \
  --model-config /path/to/extraction-api.json
```

用 `--pmid` 可指定试跑文献，多个文献可重复传入。`concepts.json` 的每条记录包含 `concept_name`、`description` 和可回连抽取结果的 `source_id`；模型无法概念化时，名称和定义均为 `null`。遗漏或无效的记录会单独重试一次。批次统计在 `report.json`，请求和回复在 `llm/`。

## Leaf Concept 归一化

将概念名称和定义发送给 Embedding API，然后用余弦距离和 Complete Linkage 找候选组。用 `--distance-threshold` 调整切分距离，阈值越大，候选组越宽。

```sh
python3 -m root_event_taxonomy.leaf_concepts \
  --concepts event_taxonomy/runs/xml-event-concepts/concepts.json \
  --events event_taxonomy/runs/xml-events/events.json \
  --abstracts /path/to/abstracts.jsonl \
  --model-config /path/to/extraction-api.json \
  --output event_taxonomy/runs/xml-leaf-concepts \
  --distance-threshold 0.35
```

第一轮使用 `prompts/leaf-equivalence.txt` 作为 Baseline。添加 `--equivalence-prompt xml` 可使用 `prompts/leaf-equivalence.xml`，按两个方向检查定义范围。原始判断保存在 `validation.json` 的 `initial` 字段。模型提议合并时，组内每对都须确认等价，并给出共同名称和定义。

`--equivalence-prompt xml-containment` 使用双向包含检查。模型对每个方向返回成立、不成立或不确定，程序据此得出关系。两个方向都成立为等价，恰有一个方向成立且另一个不成立才判上下位。原始检查记录在 `scope_checks`。

`leaf_concepts.json` 保存 Leaf ID、规范名称、定义、原始 Concept 和 Event 映射。Event 保留 Trigger、Evidence 和 PMID。`candidate_clusters.json`、`validation.json`、`canonicalization.json` 保存候选组、逐对判断与合并记录。传入 `--abstracts` 时还会导出下一阶段可读取的 `induction_input.json`，其中包含预定义 Root 和待归纳的 Leaf，父子边为空。

可用 `--reuse-embeddings` 指定已有运行目录，复用相同输入和模型的向量。用 `--review-holds` 传入 `{"holds":[{"source_concept_ids":["原始概念 ID 1","原始概念 ID 2"],"reason":"待复核原因"}]}`，可保留有争议的概念对及其模型判断。

模型归一化结束后，程序会检查全表。名称和定义都相同的节点自动合并，比较时统一 Unicode 和空白，保留大小写。遇到同名异定义，按已确认的决定合并或改名，其余列在 `deduplication.unresolved_names`。合并后的节点连接所有来源 Event。每条 Event 仍保留自己的原文和 PMID。

已有 `leaf_concepts.json` 可在本地直接整理。省略 `--decisions` 可先查看重名。每次运行都会写出结果，发现待复核名称时返回状态码 1。

```sh
python3 -m root_event_taxonomy.leaf_deduplication \
  --input /path/to/leaf_concepts.json \
  --decisions /path/to/reviewed_decisions.json \
  --output /path/to/deduplicated_leaf_concepts.json
```

决定文件的 `merges` 每项提供 `source_concept_ids`、`concept_name`、`description` 和 `reason`。需要保留不同定义时，`renames` 每项提供 `source_concept_ids`、新 `concept_name` 和 `reason`。ID 用叶节点的 `source_concepts` 字段填写。运行 `root_event_taxonomy.leaf_concepts` 时，用 `--dedup-decisions` 应用这些决定。两个入口均可通过 `--review-holds` 保留有争议的概念对。

要用同一批输入重现结果，将原始叶节点文件传给 `--input`，用 `--review-from` 读取已整理的叶节点文件。程序按保存的顺序重放合并、改名和暂缓决定。

第二轮从已保存的运行中读取第一轮原始判断和同一批候选。它用独立的 XML Prompt，补充对应的 Event、Trigger 和 Evidence，要求模型分别判断两个方向的定义包含关系，并给出理由。程序根据两个方向计算关系。只有两个方向都成立才判为等价。两边使用相同的模型参数、分组规则和待复核限制。

```sh
python3 -m root_event_taxonomy.leaf_second_validation \
  --baseline event_taxonomy/runs/xml-leaf-concepts \
  --output event_taxonomy/runs/xml-leaf-second-validation-xml
```

`comparison.json` 汇总各方案的拟合并组。`validation.json` 保留第二轮原始判断和两个方向的理由。`consensus_validation.json` 记录两轮都判为等价时才采纳的合并方案，只有一轮判等价的候选留待复核。对比停在候选合并分组，便于分别观察 XML 验证和两轮确认的影响。

## 自底向上归纳

`python3 -m root_event_taxonomy.upward` 从去重后的叶节点构建层级。`--model-config` 指定提案和审核所用的 API 模型。本次桌面实验用百炼的 `deepseek-v4.1-flash-1017`，新节点的名称和定义仍用 `text-embedding-v4` 向量化。

```sh
python3 -m root_event_taxonomy.upward \
  --leaves /path/to/leaf_concepts.json \
  --leaf-embeddings /path/to/leaf_embeddings.npz \
  --model-config /path/to/api-model.local.json \
  --output /path/to/run --max-rounds 4
```

每轮用当前没有父类的节点聚类，包括上一轮新通过的父类和仍待归属的节点。默认比较余弦距离阈值 0.35、0.45、0.55，按组内接近、组间分离的程度选出本轮切分。用 `--cutoffs` 可指定其他距离。已有第一层提案可通过 `--first-layer-proposals` 和 `--first-cutoff` 接入，先复查定义和关系，再继续往上。

独立审核会检查父类定义、每条拟议父子关系，以及底下叶节点能否纳入新父类。审核要为每个通过的关系给出一个只属于父类的例子。跨组审核通过后才入图。结果写入 `hierarchy.json`，包含各轮切分、审核、未归属节点和停止原因。程序在本轮无新父类通过或达到轮数上限时结束。API 出错时用同一命令加 `--resume` 继续。可用 `--check-input` 预览首轮分组。

本次试跑从 116 个去重后的原始概念开始。首轮沿用距离阈值 0.45，后三轮重新比较后均选用 0.55。各轮通过模型审核的新父类分别为 14、7、4 和 1 个。最终共有 142 个概念、67 条待人工复核的父子关系，48 个原始概念尚未接入任何分支。第四轮仍有新父类通过，实验因达到 4 轮上限结束。

## 运行

在仓库根目录执行。基础依赖为 Python 3.9+、NumPy 和 SciPy。当前本地试跑使用 macOS 上的 MLX。

```sh
python -m pip install -r root_event_taxonomy/requirements.txt
python -m root_event_taxonomy --config root_event_taxonomy/configs/synthetic.local.json --check-input
python -m unittest discover -s root_event_taxonomy/tests -v
```

上面的数据检查和回归测试使用内置合成数据，无需模型或网络。[工作台合成演示](../taxonomy_workbench/README.md)可以查看自顶向下的 A/B/C 流程。

### 本地模型

在运行归纳的 Python 环境安装 PyTorch 和 Sentence Transformers。为 MLX 指定已有模型目录和装有 MLX-LM 的 Python。示例配置使用 Qwen3-Embedding-4B 和 Qwen3-4B-Instruct-2507 的 4-bit MLX 权重。

```sh
export EVENT_TAXONOMY_EMBEDDING_MODEL_PATH=/path/to/Qwen3-Embedding-4B
export EVENT_TAXONOMY_LLM_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507-MLX-4bit
export EVENT_TAXONOMY_MLX_PYTHON=/path/to/mlx-env/bin/python
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m root_event_taxonomy \
  --config root_event_taxonomy/configs/synthetic.local.json --live
```

两种模型也支持 OpenAI-compatible 接口。使用 `kind: "openai_compatible"`、`base_url` 或 `base_url_env`、`api_key_env` 和 `timeout_seconds` 配置，密钥由环境变量提供。

### 真实数据

[试跑摘要](PILOT_RESULTS.md)记录了 14 个概念的实际结果。要重跑这批数据，将自己的 Task 1 导出放到 `event_taxonomy/runs/task1-workbench-final-20261112/`，并放入对应的 `definitions.local.json`，再使用 `configs/workbench-pilot.json`。定义文件以概念 ID 为键，每项提供 `definition` 和该概念已对齐的 `evidence_ids`。对已经有定义的新数据，可从 `configs/synthetic.local.json` 复制配置，将 `dataset` 指向自己的 Task 1 导出。

## 每轮怎样做

### 五步局部归纳（`algorithm: "five_step"`）

保留旧算法作为 `legacy`（默认）。新版本复用相同的输入、Embedding/API 连接和证据结构，使用单独的四份 `induction-*.xml` 提示词，不改 Leaf 归一化的冻结 XML。

1. **Candidate Discovery**：分块计算余弦近邻，结合 Root 相似度的软约束生成可重叠候选组；排除已经有祖先关系的组合。KNN 不生成分类边。配置 `discovery.k`、`block_size`、`max_candidates_per_round`，以及原有距离阈值、Root 权重和组大小。
2. **Intermediate Concept Proposal**：提出名称、定义和可能的子类，允许拒绝、拆组和复用已有概念。先核对重复概念，避免为同义概念重复建点。
3. **Concept Validation**：独立调用逐边检查两个方向的定义包含关系。只有 Child 包含于 Parent、反向不成立才通过；新节点至少需要两个通过的子类。验证输入不包含提议理由。失败成员不进入新节点的证据映射。
4. **Hierarchy Insertion**：对新节点的上位归属再单独验证；整个更新通过无环检查后提交。已有临时直连边仅在另一条路径保留原祖先关系时替换，其他父节点和已确认边保留。每次写入记录新增边、移除边及被排除成员。
5. **Recursive Refinement**：重新处理受影响节点、其子类、父类和相邻候选；原始概念和新中间概念均能参与。被拒邻域按配置重试较小概念对，同一候选不重复送审。未处理的候选保留到下一轮。无新候选、无可接受的局部修改、轮数或调用预算耗尽时停止；达到运行限制不代表语义收敛。

`placement_reserved_calls` 可预留调用预算，用于最后给悬空分支寻找已有父类；这些边同样须在入图前验证。分支无法可靠归属时保持未挂接。所有新边仍为 `provisional`，不等同于人工审核通过。

新算法设置的 LLM stages 为 `group`、`validate_edges`、`placement`、`audit_duplicates`。运行方式仍为 `python3 -m root_event_taxonomy --config ... --live`，输出到新的运行目录。`trace/` 保存五步记录，`semantic_audit.json` 保存入图前的包含检查，`prompts.json` 保存本次提示词快照。

`python3 -m root_event_taxonomy.compare_refinement --baseline OLD_RUN --new NEW_RUN --output NEW_COMPARISON` 对两张已保存图进行相同的定义复查。完全相同的关系只判一次，随机打乱顺序，不提供实验标签。报告同时给出原始 Root 覆盖和仅沿复查支持边的覆盖。该检查使用同一 LLM，并非专家金标准，也不修改任一实验结果。

### 原版本（`legacy`）

1. 对待归纳概念用名称、定义和少量已对齐的证据生成向量。Root 用已有名称和定义。设两个概念对三个 Root 的相似度分别为 `s1` 和 `s2`，候选聚类用 `(1 - w) × cosine_distance + w × mean(abs(s1 - s2))` 做 Average Linkage。距离阈值和组大小只决定送审哪些候选。
2. LLM 逐组判断是否存在严格的共同上位概念。它可提出新概念、引用已有概念、拆成多个子组、只保留部分成员，或拒绝整组。拒绝时可按配置尝试更小的聚类子组，剩下的成员仍可在下一轮参与。代码检验 ID、子组范围、重复命名和 DAG 结构。
3. 每个通过的中间概念进入父类判断。LLM 看到三个 Root、较相似的已有父类候选及各自的祖先路径。它可选择多个父类，也可留空。下一轮处理仍位于各分支最上层的概念，包括新产生的中间概念及尚未归入任何中间概念的成员。没有候选、没有结构进展、待归纳概念少于两个或达到轮次上限时停止归纳。
4. 对还未连到 Root 的分支尝试父类归属，再发起单独的全图语义复核。程序剔除未通过复核的新边，隔离疑似重复或缺乏足够已验证子类的新节点，移除可由其他路径表达的新边，并检查循环、Root 可达性和证据引用。原有边和原始概念的审查状态保留。最终的新边仍标为 `provisional`，等待人工确认。

文件中的 `SUBSUMES` 边从父类指向子类，对应语义上的 `child is-a parent`。Event Instance 仍通过原有 `IS_INSTANCE_OF` 关系连接到概念。中间概念单独记录由哪些子类传递而来的证据。

## 结果文件

- `report.json` 汇总轮次、候选判断、Root 覆盖、待审核项和模型调用量。
- `final_taxonomy.json` 保存全图。试跑范围内的覆盖以 `report.json` 为准。
- `semantic_audit.json` 记录逐边复核、疑似重复及清理结果。
- `trace/`、`llm/`、`embeddings/` 保留每轮候选、模型请求及回复、DAG 更新和向量。模型使用的短 ID 可用 `trace/` 下对应的映射还原。
- `manifest.json` 记录实际参数、模型配置及输入、提示词和模块源码校验值。

精确 Average Linkage 需要存储当前待归纳层的两两距离。阈值、Root 权重和证据截取长度目前用于小样本探索。对数万以上的活跃概念，运行时间、内存和模型调用量需要单独验证。

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s root_event_taxonomy/tests -v
```
