# PubMed Abstract 抽取与 Neo4j 导入运行说明

本文档用于运行 `multi_stage_extraction_pipeline.py`，目标是从 PubMed abstract 等非结构化文本中抽取候选三元组，并按目标 Neo4j 图数据库 schema 做合规导入。

## 当前结论

已核查目标库 `liver-kg-core-v02` 的实际 schema。当前脚本已改为保守导入模式：

- 不创建新的实体节点。
- 不创建 `Article`、`LLMEntity` 或 `LLM_*` 自定义候选层。
- 只在目标库已有核心节点之间写入允许的关系类型。
- 只写 `validation_status='candidate'`、`evidence_level='llm_extracted_candidate'` 的候选关系。
- 端点实体在 Neo4j 中匹配不到时跳过，不会 `MERGE` 新节点。

因此，当前 `--write-neo4j` 的导入逻辑是按目标 schema 收口后的合规版本。本次修改前的旧写法会创建 `Article` / `LLMEntity` / `LLM_*`，不符合当前目标库 schema。

## 目标 Neo4j Schema

数据库中现有节点标签：

| Label | 当前数量 |
| --- | ---: |
| `Pathway` | 1721 |
| `Gene` | 836 |
| `Protein` | 793 |
| `CellType` | 154 |
| `Metabolite` | 36 |
| `Disease` | 5 |
| `Tissue` | 1 |

允许的关系模式：

| Pattern | 关系类型 | 导入策略 |
| --- | --- | --- |
| `(:Gene)-[:ASSOCIATED_WITH]->(:Disease)` | `ASSOCIATED_WITH` | 可写候选 |
| `(:Metabolite)-[:ASSOCIATED_WITH]->(:Disease)` | `ASSOCIATED_WITH` | 可写候选 |
| `(:Gene)-[:PROGNOSTIC_IN]->(:Disease)` | `PROGNOSTIC_IN` | 可写候选 |
| `(:Protein)-[:INTERACTS_WITH]->(:Protein)` | `INTERACTS_WITH` | 可写候选 |
| `(:Gene)-[:PARTICIPATES_IN]->(:Pathway)` | `PARTICIPATES_IN` | 可写候选 |
| `(:Gene)-[:EXPRESSED_IN]->(:Tissue)` | `EXPRESSED_IN` | 可写候选 |
| `(:Gene)-[:EXPRESSED_IN]->(:CellType)` | `EXPRESSED_IN` | 可写候选 |
| `(:Gene)-[:ASSOCIATED_WITH_METABOLITE]->(:Metabolite)` | `ASSOCIATED_WITH_METABOLITE` | 可写候选 |
| `(:Disease)-[:PROGRESSES_TO]->(:Disease)` | `PROGRESSES_TO` | 只识别，不自动写入 |
| `(:Gene)-[:ENCODES]->(:Protein)` | `ENCODES` | 只识别，不自动写入 |

不属于目标 schema 的关系，例如 `TREATS`、`TARGETS`、`BIOMARKER_OF`，不会被自动导入。

注意：当前库的约束中包含 `Protein.string_protein_id`，但现有 `Protein` 节点主要使用 `protein_id`。导入器匹配 Protein 时会同时尝试 `protein_id`、`string_protein_id` 和 `name`，但仍然不会创建新的 Protein 节点。

## 环境准备

```bash
cd /Users/a1234/FYP/liver_disease_kg_project

set -a
source workstreams/literature_hmdb_kegg/.env
set +a

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-$LLM_API_KEY}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-${LLM_MODEL:-deepseek-chat}}"

export NEO4J_URL="bolt://100.104.181.96:7687"
export NEO4J_DATABASE="liver-kg-core-v02"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="<填入本地密码，不要提交到代码>"
```

如果要执行 `--write-neo4j`，当前 Python 环境还需要 Neo4j driver：

```bash
python3 -m pip install neo4j
```

没有安装 driver 时，脚本会跳过写库并返回 `neo4j driver not installed`。

## 推荐输入文件

优先使用已经整理好的 PubMed JSONL 小样本：

| 文件 | 用途 |
| --- | --- |
| `/Users/a1234/FYP/liver_disease_kg_project/extraction_output/pubmed_backup_liver_strict_50.jsonl` | 推荐，liver 主题更严格 |
| `/Users/a1234/FYP/liver_disease_kg_project/extraction_output/pubmed_backup_sample_50.jsonl` | 较宽泛，噪音更多 |
| `/Users/a1234/FYP/liver_disease_kg_project/extraction_output/biliary_pancreatic_sample_50.jsonl` | 胆道/胰腺相关测试 |

输入记录应至少包含 `pmid`、`title`、`abstract` 字段。

## 先跑 Dry Run

永远先跳过 Neo4j 写入，只看抽取质量：

```bash
python3 multi_stage_extraction_pipeline.py \
  --input extraction_output/pubmed_backup_liver_strict_50.jsonl \
  --limit 10 \
  --run-id pubmed_liver_dryrun_001 \
  --skip-neo4j
```

输出会写到：

```text
extraction_output/extraction_results_<run-id>.json
extraction_output/quality_report_<run-id>.json
extraction_output/provenance_report_<run-id>.html
```

建议先打开 `provenance_report_*.html` 看每条关系的 evidence、flags 和 import 状态。

## 写入 Neo4j

只有当 dry-run 报告质量可接受，并且实体链接 preflight 也通过后，再小批量写入。

实体链接 preflight：

```bash
python3 entity_linking_preflight.py \
  --input extraction_output/extraction_results_pubmed_liver_dryrun_001.json \
  --run-id pubmed_liver_dryrun_001_preflight
```

详细说明见 `ENTITY_LINKING_PREFLIGHT_README.md`。

确认 preflight 报告中存在可接受的 `import_candidate` 后，再写库：

```bash
python3 multi_stage_extraction_pipeline.py \
  --input extraction_output/pubmed_backup_liver_strict_50.jsonl \
  --limit 5 \
  --run-id pubmed_liver_import_001 \
  --write-neo4j
```

写库时脚本会再次检查：

- relation 是否 `import_ready=true`
- predicate 是否在可写白名单中
- subject/object 类型是否符合目标 schema
- subject/object 是否能匹配 Neo4j 现有节点

任何一项失败都会跳过，不会扩展 schema。

## 当前质量观察

使用最新 schema-compliant prompt 和本地校验规则，测试命令：

```bash
python3 multi_stage_extraction_pipeline.py \
  --input extraction_output/pubmed_backup_liver_strict_50.jsonl \
  --limit 3 \
  --run-id schema_compliant_llm_probe_v5_20260627 \
  --skip-neo4j
```

结果摘要：

| 指标 | 数值 |
| --- | ---: |
| records | 3 |
| entities | 25 |
| relations | 14 |
| schema-valid relations | 5 |
| import-ready relations | 0 |
| review relations | 14 |

主要问题：

- LLM 仍会把疗法/佐剂/抑制剂等 broad intervention term 错塞进 `Metabolite`、`Protein` 或 `CellType`。
- 很多 PubMed abstract 是药物、网络药理、机制综述，并不自然落在当前核心 schema 上。
- `Pathway -> Disease`、`Protein -> Disease` 这类关系常被模型抽出，但目标库不允许直接写。
- 带 `uncertain_relation`、`non_human_or_mixed_species`、`unsupported_entity_class` 的关系会进入 review，不自动写库。
- 实体链接是下一步瓶颈；即使关系质量过关，端点不在现有 Neo4j 节点中也会被跳过。

这个结果偏保守，但符合“不能污染目标图数据库 schema”的要求。

## 改进方向

优先级建议：

1. 加实体链接：Gene 用 NCBI/HGNC 映射到 `NCBIGene:*`，Metabolite 用 HMDB，Pathway 用 KEGG/Reactome/WikiPathways。
2. 增加 PubMed 过滤：优先选 gene-disease、prognosis、expression、protein interaction 类型 abstract，减少药物治疗综述。
3. 增加 Neo4j read-only preflight：在 dry-run 阶段提前统计 endpoint match，而不是等 `--write-neo4j` 时才跳过。
4. 若确实需要 `TREATS`、`TARGETS`、`BIOMARKER_OF`，应先正式扩展目标 schema，再改导入器；不要直接让 LLM 输出写进现有库。
