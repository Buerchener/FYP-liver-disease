# Root-Constrained Event Taxonomy Induction

独立入口 `python -m root_event_taxonomy` 读取 `event_taxonomy` 的 Task 1 导出目录或同结构的 JSON。它沿用项目的 Concept、Event、Evidence 和 Taxonomy 数据结构。原始概念、实例关系和证据映射随结果保留。选择试跑范围时，范围之外的概念也保留原状。

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
