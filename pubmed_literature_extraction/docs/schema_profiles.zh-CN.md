# 可配置数据集 Schema

`SchemaProfile` 贯穿模型提示词、实体和关系解析、候选验证及结果导出。配置中定义的实体类型、谓词、端点组合和证据规则在每次运行时生效。

省略 `--schema-profile` 时，使用内置 LiverKG 流程。自定义数据集可以复制并编辑 `docs/schema_profile.example.json`。在 `pubmed_literature_extraction` 目录下运行以下命令。

```bash
liverkg extract articles.jsonl --schema-profile docs/schema_profile.example.json
```

也可以直接调用 Agent，或在 Python 中使用 `AgentConfig(schema_profile="/path/to/schema.json")`。

```bash
python -m cognitive_agent.agent \
  --input articles.jsonl \
  --schema-profile docs/schema_profile.example.json
```

`liverkg extract` 会在运行目录中保存 schema 快照。后台任务和断点续跑使用同一份快照。结果和报告记录配置名称、版本和 SHA-256。

## 配置字段

- `entity_types` 声明允许的实体类型。省略时，从关系的端点组合推导。
- `predicates` 声明关系标签。`allowed_signatures` 指定每个谓词允许的有序端点组合，`symmetric=true` 允许两端交换。
- `semantic_target` 取 `CURRENT_FINDING` 或 `RELATION_TRUTH`。后者也允许候选引用原文明确陈述的既有发现。
- `entity_validation`：
  - `liverkg_quality` 使用项目原有的 LiverKG 实体质量规则；
  - `source_grounded` 只要求类型已声明且 mention 可回源；
  - `given_entity` 用于 BioRED 这类官方实体已给定的数据集。
- `article_quality_mode`：`liverkg` 或 `none`，避免其他数据集误用肝病范围规则。
- `explicit_patterns`、`weak_patterns`、`exclusion_patterns` 接受正则字符串。明确触发词可为候选提供支持，弱匹配或缺少触发规则的候选进入 `REVIEW`，命中排除规则的候选标记为 `REJECTED`。

验证仍要求两个端点和证据可回溯到原文。自定义运行使用主抽取模型和配置内的证据规则，输出到 `agent_results_*.json` 和 `agent_report_*.json`。结果文件的 `records[].phases.verification` 含验证后的实体和关系。

## 图谱写入

自定义 schema 用于导出候选 JSON，`--schema-profile` 与 `--write-neo4j` 互斥。默认 LiverKG 流程的图谱写入由 `write_contract.py` 管理。

BioRED 的 Train-derived native registry 使用同一 profile 接口，标签和 concept types 沿用 BioRED 的定义。
