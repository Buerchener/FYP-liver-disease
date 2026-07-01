# FYP Ontology v2：肝病 Backbone 知识图谱数据包

这个仓库保存了一个围绕肝病进展 backbone 构建的知识图谱 v2 数据包。当前版本先聚焦五个核心疾病阶段，并把相关的基因、蛋白、通路、组织表达、细胞类型表达和代谢物信息整理成可审计、可复现的导入材料。

## 疾病 Backbone

```text
NAFLD -> NASH -> Fibrosis -> Cirrhosis -> HCC
```

五个核心疾病节点：

| 阶段 | Disease ID |
|---|---|
| NAFLD | `UMLS:C0400966` |
| NASH | `UMLS:C3241937` |
| Fibrosis | `UMLS:C0239946` |
| Cirrhosis | `UMLS:C0023890` |
| HCC | `UMLS:C2239176` |

## 数据来源

| 数据库 | 用途 |
|---|---|
| DisGeNET | 获取五个疾病对应的 Gene-Disease association |
| STRING | 将 Gene 映射到 Protein，并导入高置信 Protein-Protein interaction |
| KEGG | 获取 Gene 参与的 pathway |
| Reactome | 获取 Gene 参与的 pathway |
| HPA | 获取 liver tissue expression、LIHC prognosis 和泛组织 single-cell type expression |
| HMDB | 获取保守筛选后的疾病相关 metabolite context |
| PubMed | 使用 LLM baseline 与闭环 cognitive agent 从文献 abstract 中抽取候选 KG 关系 |

## 当前图谱规模

| 实体/关系 | 数量 |
|---|---:|
| Disease | 5 |
| Gene | 836 |
| Protein | 793 |
| Pathway | 1721 |
| Tissue | 1 |
| CellType | 154 |
| Metabolite | 36 |
| Gene-Disease association | 1036 |
| STRING PPI | 7154 |
| Gene-Pathway membership | 9947 |
| HPA expression relation | 17745 |
| HPA LIHC prognostic relation | 1384 |
| HMDB Gene-Metabolite relation | 97 |

## 目录说明

```text
.
├── HUMAN_REVIEW.md      # 人工审阅入口
├── review/              # 精简后的当前结论和 schema
├── data/                # v2 导入用 TSV/JSON 数据
├── scripts/             # 抓取、整理和 Neo4j 导入脚本
├── pubmed_literature_extraction/  # Shaopeng Chen 负责的 PubMed 文献抽取模块
├── archive/             # 旧报告、机器审计、字段裁剪前备份
└── MANIFEST.txt         # 当前文件清单
```

建议人工核对时先看：

1. `HUMAN_REVIEW.md`
2. `review/01_current_database_summary.md`
3. `review/02_backbone_scope_audit.md`
4. `review/03_entity_attribute_schema.md`
5. `review/04_caveats_and_next_cleanup.md`

PubMed 文献抽取部分先看：

1. `pubmed_literature_extraction/README.md`
2. `pubmed_literature_extraction/docs/pubmed_extraction_report.md`
3. `pubmed_literature_extraction/docs/cognitive_agent_architecture.md`

## 重要解释边界

- DisGeNET 的 Gene-Disease association 是疾病证据主干。
- STRING PPI 是 backbone Gene 编码蛋白之间的互作背景，不代表直接疾病因果关系。
- KEGG/Reactome pathway 是 Gene 参与的通路背景，不代表该 pathway 只属于某个疾病阶段。
- HPA CellType 数据是泛组织 single-cell type 表达背景，不是 liver-only cell type 证据。
- HMDB metabolite 数据是保守筛选后的疾病相关代谢物背景，不是全量 HMDB。

## Neo4j 版本

当前对应数据库名称：

```text
liver-kg-core-v02
```

脚本默认通过环境变量读取 Neo4j 密码，请不要把密码写入代码或提交到 GitHub：

```bash
export NEO4J_PASSWORD='your_password_here'
```

## 状态

当前 v2 数据已经通过 backbone 范围审计：没有发现 backbone 之外的 Disease 节点，也没有发现无法追溯到 backbone Gene 或 backbone Disease 的下游节点。
