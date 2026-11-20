# LiverKG 实体本体流程

给它一批 PubMed 标题和摘要，它会找出疾病、基因、蛋白质、代谢物、通路、组织和细胞类型，再把确认过的概念放进候选层级。工作台可以逐条回看原句、同义判断和父子类决定。

```text
PubMed 摘要 → 实体抽取与补漏 → 概念化 → 向量检索与同义去重
           → 由宽到窄排序 → 逐个放入 is_a 层级 → 网页查看
```

## 安装和运行

在 macOS 或 Linux 上，从仓库根目录使用 Python 3.12 或 3.13。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r entity_ontology/requirements.txt
python -m entity_ontology setup-extractor
cp entity_ontology/config.example.json entity-ontology.local.json
```

按本地模型服务设置 `LIVERKG_API_BASE` 和 `LIVERKG_API_KEY`。API 地址填写 OpenAI 兼容基础地址，例如阿里云百炼的 `https://dashscope.aliyuncs.com/compatible-mode/v1`。示例配置让 Qwen 与 `text-embedding-v4` 共用服务地址和凭据。模型 ID 可在本地配置调整，配置和密钥保存在本地。也可以用 `--config` 指向现有配置。

输入是 UTF-8 JSONL，每行含唯一数字 PMID、标题和摘要。以下一行展示字段格式。

```jsonl
{"pmid":"1","title":"Synthetic example","abstract":"This is a synthetic abstract for illustrating the input format."}
```

```bash
python -m entity_ontology run --articles articles.jsonl --run-id first-batch
python -m entity_ontology serve --open
```

同一个批次名和同一份输入、模型配置可续跑。换论文或配置时用新批次名。运行结果默认在 `runs/entity_ontology/<批次名>/`。如需读取已有试点，可以给 `serve` 指定其批次父目录 `--runs-dir`。

## 每一步在做什么

| 步骤 | 处理方式 | 主要产物 |
|---|---|---|
| 抽取和补漏 | 运行同学的 LiverKG 抽取器，固定到 `fc9ebd7`，以 dry-run 保留原始输出。另查沉积、表达量、活性等短语，补充能回切到原句的实体。 | `agent_results.json`、`entity_coverage.json`、`input_mentions.json` |
| 概念化 | 根据原句识别实体实际所指，再独立核对类型。保留可用概念、待复核项和超出范围项。 | `normalization.json` |
| 检索与去重 | KNN 和共享名称或别名找候选，LLM 判断同义。同一组内的每一对都确认等价才合并。PCA 和 HDBSCAN 用于网页展示。 | `retrieval/retrieved_pairs.json`、`equivalence.json` |
| 排序 | LLM 按实体类型由宽到窄排序，生成逐个摆放的清单。 | `concept_order.json` |
| 构树 | LLM 提议并复核 `is_a` 方向，Python 检查引用、循环和冗余边。允许一个类有多个直接父类。GO Agent 可为仍有疑问的适用候选提供只读参考。 | `incremental-placement/placement.json` |
| 浏览 | 网页加载已完成批次的阶段结果和原文证据。 | 默认地址 `http://127.0.0.1:4320/` |

`--no-coverage` 和 `--no-go-reference` 可分别关闭补漏和 GO 参考。初始层级以本次出现的实体类型作根，再逐个放置概念。文献原句、模型判定和 GO 参考都保存以便审阅。层级是候选结果，运行全程使用本地文件，与 Neo4j 导入流程分开。

## 代码位置

- `__main__.py` 和 `pipeline.py` 提供统一入口和批次管理。
- `extraction.py` 负责获取固定版抽取器、运行 dry-run 和恢复抽取检查点。
- `coverage.py`、`mentions.py`、`normalization.py` 处理原文证据和类型。模型请求、概念化衔接和同义合并在 `stages.py`。
- `retrieve_pairs.py` 负责向量和近邻检索，`order_entities.py` 负责排序。
- `placement.py` 和 `entity_placement_dag.py` 负责层级放置与图校验。内置 GO Agent 的运行代码来自 `542d4c4`，通过 `go_placement.py` 接入。
- `workbench.py` 和 `web/` 提供本地只读工作台。

固定版抽取器首次安装时从 [同学的抽取提交](https://github.com/Buerchener/FYP-liver-disease/commit/fc9ebd72971318a7738f15fc2b86369a552656c2) 取源码，只缓存一份。运行日志、论文样本、检查点、配置和模型缓存均在 Git 忽略的路径内。旧 Event 实验和原版肝病图谱数据可以在各自的 GitHub 分支及目录回查。
