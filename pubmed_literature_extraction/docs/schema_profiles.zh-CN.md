# 可配置数据集 Schema

关系验证算法保持固定，数据集自己的实体类型、关系标签、合法端点组合和证据触发规则由只读 `SchemaProfile` 提供。

默认不传参数时使用内置 LiverKG profile。自定义数据集可以复制并编辑 `docs/schema_profile.example.json`，然后运行：

```bash
python -m cognitive_agent.agent \
  --schema-profile /absolute/path/to/dataset-schema.json \
  --skip-neo4j-write
```

## 配置字段

- `semantic_target`：`CURRENT_FINDING` 或 `RELATION_TRUTH`。
- `entity_validation`：
  - `liverkg_quality` 使用项目原有的 LiverKG 实体质量规则；
  - `source_grounded` 只要求类型已声明且 mention 可回源；
  - `given_entity` 用于 BioRED 这类官方实体已给定的数据集。
- `article_quality_mode`：`liverkg` 或 `none`，避免其他数据集误用肝病范围规则。
- `allowed_signatures`：关系允许的有序端点类型；`symmetric=true` 时两端可交换。
- `explicit_patterns`、`weak_patterns`、`exclusion_patterns`：只接受正则字符串，不加载或执行 Python 代码。

每条验证结果会保存 `schema_profile_name`、`schema_profile_version` 和 `schema_profile_sha256`；汇总报告保存完整 profile manifest。

## 安全边界

数据集 schema 只回答“这条候选是否符合当前数据集定义”，不回答“能否写入 LiverKG”。Neo4j 写入仍由独立的 `write_contract.py` 决定；任何自定义 profile 的 manifest 都固定声明 `authorizes_neo4j_write=false`。

BioRED 的 Train-derived native registry 已实现同一个 profile 接口，但仍使用 BioRED 原生标签和官方 concept types，不映射成 LiverKG 谓词，也不读取 Dev/Test Gold 来动态改 schema。
