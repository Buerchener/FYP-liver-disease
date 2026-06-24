# Liver KG v2 实体属性人工核对表

用途：这份文件列出当前 Neo4j v2 中节点和关系的实际属性，用于你人工核对哪些字段应该保留、合并或剔除。

检查数据库：`liver-kg-core-v02`

生成依据：

- `2026-06-24` 对 v2 Neo4j 的实际 property key 扫描
- `data/*_v02/` 和 `data/disgenet_liver_backbone/` 下的 v2 TSV 表头
- `scripts/` 下的 v2 导入脚本

审核建议说明：

- `建议保留`：核心 ID、展示字段、证据字段，或后续映射/查询会用到的字段。
- `人工复核`：可能有用，但也可能冗余、过于 source-specific，或更适合只保留在 staging。
- `删除候选`：当前为空、明显重复，或属于导入运行元数据，通常不需要留在最终主图。

## 实体概览

| 实体标签 | 当前数量 | 作用 |
|---|---:|---|
| `Disease` | 5 | 五个 liver disease backbone 阶段：NAFLD、NASH、Fibrosis、Cirrhosis、HCC。 |
| `Gene` | 836 | DisGeNET 中与五个疾病相关的基因。 |
| `Protein` | 793 | 由 Gene 映射得到的 STRING protein 节点。 |
| `Pathway` | 1721 | KEGG 和 Reactome pathway，共用一个 `Pathway` 标签。 |
| `Tissue` | 1 | HPA 的 liver tissue 节点。 |
| `Metabolite` | 36 | HMDB 中保守筛选后的 disease-linked metabolites。 |
| `CellType` | 0 | 当前 v2 没有导入 HPA single-cell 数据。 |

## Disease 节点

推荐主键：`disease_id`。

人工核查结论：Disease 节点最终保留以下字段：

`disease_id`, `name`, `disease_name`, `disease_type`, `external_ids`, `stage_order`, `disease_classes_msh`, `source`, `updated_at`

其中 `external_ids` 由当前字段 `disease_vocabularies` 重命名而来。

应用状态：已于 `2026-06-24` 应用到 Neo4j `liver-kg-core-v02`。更新前备份见 `reports/disease_nodes_before_attribute_prune_20260624.json`。

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `disease_id` | 5/5 | 图谱内部稳定疾病 ID，目前对应五个 backbone disease ID。 | DisGeNET curated table | 建议保留 |
| `project_id` | 5/5 | 项目层面的 ID，目前复制自 `disease_id`。 | 导入派生 | 删除候选；人工核查后不进入 Disease 最终字段。 |
| `umls_id` | 5/5 | UMLS CUI，是最重要的外部疾病 ID。 | DisGeNET | 删除候选；若仍需保留 UMLS，可并入 `external_ids`。 |
| `name` | 5/5 | 展示名，目前设置为 `stage_code`，例如 `NAFLD`。 | 导入派生 | 建议保留 |
| `stage_code` | 5/5 | 标准 backbone 阶段名：`NAFLD`、`NASH`、`Fibrosis`、`Cirrhosis`、`HCC`。 | Backbone 配置 | 删除候选；人工核查后用 `name` 承载短名。 |
| `stage_order` | 5/5 | 疾病进展顺序的数值排序。 | Backbone 配置 | 建议保留 |
| `disease_name` | 5/5 | 疾病完整名称。 | DisGeNET | 建议保留 |
| `disease_type` | 5/5 | DisGeNET 的疾病类型/类别。 | DisGeNET | 建议保留 |
| `disease_vocabularies` | 5/5 | 疾病关联的外部词表来源。 | DisGeNET | 建议保留，但建议在最终 schema 中重命名为 `external_ids`。 |
| `external_ids` | 待迁移 | 建议的新字段名，用于承载原 `disease_vocabularies`。 | 人工核查命名 | 建议保留；替代 `disease_vocabularies`。 |
| `disease_classes_msh` | 5/5 | MeSH 疾病分类。 | DisGeNET | 建议保留 |
| `disease_classes_umls_st` | 5/5 | UMLS semantic type 分类。 | DisGeNET | 删除候选；人工核查后不进入 Disease 最终字段。 |
| `disease_classes_do` | 5/5 | Disease Ontology 分类。 | DisGeNET | 删除候选；人工核查后不进入 Disease 最终字段。 |
| `disease_classes_hpo` | 5/5 | HPO 疾病/表型分类。 | DisGeNET | 删除候选；人工核查后不进入 Disease 最终字段。 |
| `is_progression_stage` | 5/5 | 是否属于 disease progression backbone 的布尔标记。 | 导入派生 | 删除候选；当前最终 Disease 字段不保留。 |
| `source` | 5/5 | 节点来源。 | DisGeNET/导入 | 建议保留 |
| `updated_at` | 5/5 | 导入时间戳。 | 导入派生 | 建议保留 |

## Gene 节点

推荐主键：`gene_id`，即全局 ID，格式为 `NCBIGene:{ncbi_gene_id}`。推荐展示字段：`gene_symbol` 或 `name`。

人工核查结论：Gene 节点最终保留以下字段：

`gene_id`, `ncbi_gene_id`, `gene_symbol`, `name`, `gene_type`, `ensembl_gene_ids`, `gene_dsi`, `gene_dpi`, `gene_pli`, `protein_class_names`, `source`, `updated_at`

字段迁移规则：

- `gene_id`: 从原 NCBI 数字 ID 改为全局 ID，例如 `NCBIGene:50`。
- `ncbi_gene_id`: 新增字段，保存原 NCBI 数字 ID，例如 `50`。
- `gene_type`: 由原 `gene_ncbi_type` 重命名而来。
- `ensembl_gene_ids`: 保留为 canonical Ensembl 字段，删除旧的 `gene_ensembl_ids`。
- `protein_class_names`: 由原 `gene_protein_class_names` 重命名而来。

应用状态：已于 `2026-06-24` 应用到 Neo4j `liver-kg-core-v02`。更新前备份见 `reports/gene_nodes_before_attribute_prune_20260624.json`。

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `gene_id` | 836/836 | Gene 全局 ID，格式为 `NCBIGene:{ncbi_gene_id}`。 | DisGeNET/导入派生 | 建议保留 |
| `ncbi_gene_id` | 836/836 | 原始 NCBI Gene 数字 ID，例如 `50`。 | DisGeNET/导入派生 | 建议保留 |
| `project_id` | 迁移前 836/836 | 原项目层面的 ID，格式为 `NCBIGene:{旧 gene_id}`。 | 导入派生 | 删除候选；已迁移为新的 `gene_id`。 |
| `gene_symbol` | 836/836 | 基因 symbol。 | DisGeNET | 建议保留 |
| `name` | 836/836 | 展示名，目前复制自 `gene_symbol`。 | 导入派生 | 建议保留 |
| `gene_ensembl_ids` | 迁移前 836/836 | DisGeNET 给出的 Ensembl gene ID。 | DisGeNET | 删除候选；已迁移/统一到 `ensembl_gene_ids`。 |
| `ensembl_gene_ids` | 836/836 | canonical Ensembl gene ID 字段。 | DisGeNET/导入派生 | 建议保留 |
| `gene_ncbi_type` | 迁移前 836/836 | NCBI gene type/biotype。 | DisGeNET | 删除候选；已重命名为 `gene_type`。 |
| `gene_type` | 836/836 | Gene type/biotype，例如 `protein-coding`、`ncRNA`、`biological-region`。 | DisGeNET/导入派生 | 建议保留 |
| `gene_protein_str_ids` | 迁移前 836/836 | DisGeNET 中的 STRING protein ID。 | DisGeNET | 删除候选；当前最终 Gene 字段不保留。 |
| `protein_ids_from_disgenet` | 836/836 | `gene_protein_str_ids` 的复制字段。 | 导入派生 | 删除候选；明显重复。 |
| `gene_dsi` | 836/836 | Disease Specificity Index；通常越低表示关联疾病范围越广。 | DisGeNET | 建议保留 |
| `gene_dpi` | 836/836 | Disease Pleiotropy Index；表示关联疾病类别的广度。 | DisGeNET | 建议保留 |
| `gene_pli` | 757/836 | Loss-of-function intolerance 概率。 | DisGeNET | 建议保留 |
| `gene_protein_class_ids` | 迁移前 836/836 | Protein class ID 列表。 | DisGeNET | 删除候选；当前最终 Gene 字段不保留。 |
| `gene_protein_class_names` | 迁移前 836/836 | Protein class 名称列表。 | DisGeNET | 删除候选；已重命名为 `protein_class_names`。 |
| `protein_class_names` | 836/836 | Protein class 名称，例如 `Enzyme`、`Transporter`。 | DisGeNET/导入派生 | 建议保留 |
| `source` | 836/836 | 节点来源。 | DisGeNET/导入 | 建议保留 |
| `updated_at` | 836/836 | 导入时间戳。 | 导入派生 | 建议保留 |

## Protein 节点

推荐主键：`protein_id`。当前 `protein_id` 使用 STRING protein ID，例如 `9606.ENSP00000323929`。

人工核查结论：Protein 节点最终保留以下字段：

`protein_id`, `name`, `annotation`, `source`, `ncbi_taxon_id`, `species_name`, `updated_at`

字段迁移规则：

- `name`: 由原 `preferred_name` 重命名而来。
- `protein_id`: 保留原值；原 `string_protein_id` 与 `protein_id` 完全一致，因此删除 `string_protein_id`。

应用状态：已于 `2026-06-24` 应用到 Neo4j `liver-kg-core-v02`。更新前备份见 `reports/protein_nodes_before_attribute_prune_20260624.json`。

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `protein_id` | 793/793 | Protein ID，目前使用 STRING protein identifier。 | STRING/导入 | 建议保留 |
| `name` | 793/793 | Protein 展示名，由原 `preferred_name` 得到。 | STRING/导入派生 | 建议保留 |
| `string_protein_id` | 迁移前 793/793 | STRING protein identifier。 | STRING | 删除候选；与 `protein_id` 完全重复。 |
| `preferred_name` | 迁移前 793/793 | STRING preferred protein/gene name。 | STRING | 删除候选；已重命名为 `name`。 |
| `ncbi_taxon_id` | 793/793 | Taxon ID，目前为 human `9606`。 | STRING | 建议保留 |
| `species_name` | 793/793 | 物种名，目前为 `Homo sapiens`。 | 导入派生 | 建议保留 |
| `annotation` | 793/793 | STRING protein annotation/description。 | STRING | 建议保留 |
| `source` | 793/793 | 节点来源。 | STRING/导入 | 建议保留 |
| `updated_at` | 793/793 | 导入时间戳。 | 导入派生 | 建议保留 |

## Pathway 节点

推荐主键：`pathway_id`。当前 KEGG 和 Reactome 共用一个 `Pathway` label；`pathway_id` 已经带来源前缀，例如 `KEGG:hsa00010`、`Reactome:R-HSA-1059683`。

人工核查结论：Pathway 节点最终保留以下字段：

`pathway_id`, `name`, `source`, `ncbi_taxon_id`, `species_name`, `updated_at`

应用状态：已于 `2026-06-24` 应用到 Neo4j `liver-kg-core-v02`。更新前备份见 `reports/pathway_nodes_before_attribute_prune_20260624.json`。

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `pathway_id` | 1721/1721 | 图谱内部 pathway ID，带 source 前缀。 | KEGG/Reactome 导入 | 建议保留 |
| `name` | 1721/1721 | Pathway 展示名。 | KEGG/Reactome | 建议保留 |
| `source` | 1721/1721 | 数据库来源：KEGG 或 Reactome。 | 导入 | 建议保留 |
| `ncbi_taxon_id` | 1721/1721 | Taxon ID，目前为 human `9606`。 | KEGG/Reactome/导入 | 建议保留 |
| `species_name` | 1721/1721 | 物种名，例如 `Homo sapiens`。 | KEGG/Reactome/导入 | 建议保留 |
| `kegg_pathway_id` | 迁移前 333/1721 | KEGG pathway accession，例如 `hsa00010`。 | KEGG | 删除候选；已由 `pathway_id=KEGG:hsa...` 承载。 |
| `pathway_name` | 迁移前 333/1721 | KEGG pathway name，通常和 `name` 重复。 | KEGG | 删除候选；统一用 `name`。 |
| `organism` | 迁移前 333/1721 | KEGG organism 信息。 | KEGG | 删除候选；由 `ncbi_taxon_id` 和 `species_name` 承载。 |
| `reactome_stable_id` | 迁移前 1388/1721 | Reactome stable pathway/event ID。 | Reactome | 删除候选；已由 `pathway_id=Reactome:R-HSA-...` 承载。 |
| `schema_class` | 迁移前 1388/1721 | Reactome schema class。 | Reactome | 删除候选；当前最终 Pathway 字段不保留。 |
| `reactome_url` | 迁移前 1388/1721 | Reactome 浏览器 URL。 | Reactome | 删除候选；可由 Reactome ID 重建，或留在 staging/备份中。 |
| `updated_at` | 1721/1721 | 导入时间戳。 | 导入派生 | 建议保留 |

## Tissue 节点

推荐主键：`tissue_id`。

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `tissue_id` | 1/1 | HPA tissue ID，目前是 liver。 | HPA 导入 | 建议保留 |
| `name` | 1/1 | 展示名。 | HPA 导入 | 人工复核；可能和 `tissue_name` 重复。 |
| `tissue_name` | 1/1 | HPA tissue name。 | HPA | 建议保留 |
| `source` | 1/1 | 节点来源。 | HPA/导入 | 建议保留 |
| `updated_at` | 1/1 | 导入时间戳。 | 导入派生 | 删除候选；更适合保留在审计日志或 staging。 |

## CellType 节点

当前 v2 数量为 0。相关文件只有表头，因为本轮没有导入 HPA single-cell expression。

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `cell_type_id` | 0/0 | 未来可能使用的 HPA cell type ID。 | HPA | 当前不建议放入最终图谱，除非后续导入 single-cell。 |
| `name` | 0/0 | 未来可能使用的展示名。 | HPA/导入 | 当前不建议放入最终图谱。 |
| `cell_type_name` | 0/0 | 未来可能使用的 HPA cell type name。 | HPA | 当前不建议放入最终图谱。 |
| `source` | 0/0 | 节点来源。 | HPA/导入 | 当前不建议放入最终图谱。 |
| `updated_at` | 0/0 | 导入时间戳。 | 导入派生 | 当前不建议放入最终图谱。 |

## Metabolite 节点

推荐主键：`metabolite_id`，格式为 `HMDB:{HMDB accession}`，例如 `HMDB:HMDB0000467`。

人工核查结论：Metabolite 节点最终保留以下字段：

`metabolite_id`, `name`, `chemical_formula`, `average_molecular_weight`, `monoisotopic_molecular_weight`, `kingdom`, `super_class`, `class`, `source`, `updated_at`

字段迁移和修复规则：

- `metabolite_id`: 从原始 `HMDB0000467` 改为带来源前缀的 `HMDB:HMDB0000467`。
- `hmdb_id`: 删除；原始 accession 已包含在 `metabolite_id` 中。
- `monoisotopic_molecular_weight`: 修正旧字段拼写 `monisotopic_molecular_weight`。
- `kingdom`, `super_class`, `class`: 原来为空是解析问题；这些字段位于 HMDB XML 的 `taxonomy` 子节点中，已从本地 HMDB XML 回填。

应用状态：已于 `2026-06-24` 应用到 Neo4j `liver-kg-core-v02`。更新前备份见 `reports/metabolite_nodes_before_attribute_prune_20260624.json`，taxonomy 回填表见 `reports/hmdb_metabolite_taxonomy_backfill_20260624.tsv`。

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `metabolite_id` | 36/36 | 图谱内部 metabolite ID，格式为 `HMDB:HMDBxxxx`。 | HMDB/导入派生 | 建议保留 |
| `hmdb_id` | 迁移前 36/36 | HMDB accession。 | HMDB | 删除候选；已并入带前缀的 `metabolite_id`。 |
| `name` | 36/36 | 代谢物名称。 | HMDB | 建议保留 |
| `chemical_formula` | 36/36 | 化学式。 | HMDB | 建议保留 |
| `monisotopic_molecular_weight` | 迁移前 36/36 | 旧拼写的单同位素分子量字段。 | HMDB/旧导入 | 删除候选；已迁移为 `monoisotopic_molecular_weight`。 |
| `monoisotopic_molecular_weight` | 36/36 | 单同位素分子量。 | HMDB | 建议保留 |
| `average_molecular_weight` | 36/36 | 平均分子量。 | HMDB | 建议保留 |
| `kingdom` | 36/36 | 化学分类 kingdom，例如 `Organic compounds`。 | HMDB taxonomy | 建议保留 |
| `super_class` | 36/36 | 化学分类 superclass，例如 `Lipids and lipid-like molecules`。 | HMDB taxonomy | 建议保留 |
| `class` | 36/36 | 化学分类 class，例如 `Steroids and steroid derivatives`。 | HMDB taxonomy | 建议保留 |
| `source` | 36/36 | 节点来源。 | HMDB/导入 | 建议保留 |
| `updated_at` | 36/36 | 导入时间戳。 | 导入派生 | 建议保留 |

## 关系属性

严格来说，以下不是实体 attribute，但当前图谱中大部分证据质量、mapping provenance 和阈值信息都在关系上，所以建议和实体字段一起人工核对。

### `Gene` -[`ASSOCIATED_WITH`]-> `Disease`

DisGeNET GDA 边：1036 条。注意：同一个 `ASSOCIATED_WITH` 类型当前还包含 43 条 HMDB `Metabolite` -[`ASSOCIATED_WITH`]-> `Disease` 边，所以这个关系类型目前存在语义混用。

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `relation_id` | 1036/1079 | DisGeNET GDA relation ID。 | DisGeNET 导入 | 建议保留，用于 GDA 边。 |
| `relationship_id` | 43/1079 | HMDB disease-metabolite relation ID。 | HMDB 导入 | 人工复核；提示当前关系类型被混用。 |
| `source` | 1079/1079 | 来源。 | DisGeNET/HMDB | 建议保留 |
| `source_record_id` | 1036/1079 | source record ID，目前复制自 DisGeNET `assoc_id`。 | DisGeNET 导入 | 人工复核；可能和 `assoc_id` 重复。 |
| `assoc_id` | 1036/1079 | DisGeNET association ID。 | DisGeNET | 建议保留 |
| `score` | 1036/1079 | DisGeNET association score。 | DisGeNET | 建议保留 |
| `normalized_score` | 1036/1079 | normalized score，用于置信度判断。 | DisGeNET/导入 | 建议保留 |
| `confidence_score` | 1036/1079 | `normalized_score` 的复制字段。 | 导入派生 | 删除候选；如果保留 `normalized_score`，此字段冗余。 |
| `num_pmids` | 1036/1079 | 支持该关联的 PubMed 文献数量。 | DisGeNET | 建议保留 |
| `year_initial` | 1026/1079 | 最早证据年份。 | DisGeNET | 人工复核 |
| `year_final` | 1026/1079 | 最新证据年份。 | DisGeNET | 人工复核 |
| `evidence_index` | 1026/1079 | DisGeNET evidence index。 | DisGeNET | 人工复核 |
| `disgenet_evidence_level` | 1/1079 | DisGeNET evidence level；当前几乎为空。 | DisGeNET | 删除候选；除非后续 API 权限提升并补齐。 |
| `evidence_level` | 1036/1079 | 导入时从 `tier` 派生/复制。 | 导入派生 | 人工复核；如果保留 `tier`，此字段可能冗余。 |
| `tier` | 1036/1079 | v2 本地置信度分层：high/medium。 | 本地筛选规则 | 建议保留 |
| `tier_reason` | 1036/1079 | 分层原因，说明为什么进入 high/medium。 | 本地筛选规则 | 建议保留/人工复核 |
| `score_breakdown` | 1036/1079 | DisGeNET score components 的序列化字段。 | DisGeNET | 人工复核；适合审计，可能只留 staging。 |
| `num_db_snp` | 1036/1079 | SNP 证据数量。 | DisGeNET | 人工复核 |
| `num_clinical_trials` | 1036/1079 | clinical trial 证据数量。 | DisGeNET | 人工复核 |
| `num_chemicals` | 1036/1079 | chemical evidence 数量。 | DisGeNET | 人工复核 |
| `num_pmids_with_chemicals` | 1036/1079 | 含 chemical evidence 的 PubMed 数量。 | DisGeNET | 人工复核 |
| `num_trials_with_chemicals` | 1036/1079 | 含 chemical evidence 的 trial 数量。 | DisGeNET | 人工复核 |
| `chemical_evidence` | 1036/1079 | chemical evidence 的序列化字段。 | DisGeNET | 人工复核；如果不查询 chemical evidence，可只留 staging。 |
| `api_endpoint` | 1036/1079 | 抓取记录时使用的 API endpoint。 | 抓取 metadata | 删除候选；建议留 staging/report，不进主图。 |
| `api_query` | 1036/1079 | 抓取记录时使用的 API query。 | 抓取 metadata | 删除候选；建议留 staging/report，不进主图。 |
| `retrieved_at` | 1036/1079 | 抓取时间。 | 抓取 metadata | 人工复核；有 provenance 价值，但也可只留 staging。 |
| `validation_status` | 1036/1079 | 导入/校验状态。 | 导入派生 | 人工复核；如果后续有人工 curation workflow，可保留。 |
| `disease_name` | 43/1079 | HMDB metabolite-disease 边上复制的 disease name。 | HMDB 导入 | 删除候选；Disease 节点已有名称。 |
| `updated_at` | 1079/1079 | 导入时间戳。 | 导入派生 | 删除候选；更适合保留在审计日志或 staging。 |

### `Disease` -[`PROGRESSES_TO`]-> `Disease`

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `progression_id` | 4/4 | backbone transition 的稳定 ID。 | 本地 backbone | 建议保留 |
| `from_stage_code` | 4/4 | 起点疾病阶段。 | 本地 backbone | 人工复核；可由起点 Disease 节点推导，但导出时有用。 |
| `to_stage_code` | 4/4 | 终点疾病阶段。 | 本地 backbone | 人工复核；可由终点 Disease 节点推导，但导出时有用。 |
| `stage_order_delta` | 4/4 | 两个阶段的 order 差值，通常为 1。 | 本地 backbone | 人工复核；如果节点上已有 `stage_order`，这个字段可能冗余。 |
| `source` | 4/4 | progression model 来源。 | 本地 backbone | 建议保留 |
| `evidence_level` | 4/4 | 证据/规范状态。 | 本地 backbone | 建议保留 |
| `validation_status` | 4/4 | curation/import 状态。 | 本地 backbone | 人工复核 |
| `updated_at` | 4/4 | 导入时间戳。 | 导入派生 | 删除候选；更适合保留在审计日志或 staging。 |

### `Gene` -[`ENCODES`]-> `Protein`

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `source` | 793/793 | 来源，目前为 STRING mapping。 | STRING 导入 | 建议保留 |
| `mapping_input` | 793/793 | 用于映射到 STRING 的输入 ID。 | STRING 导入 | 建议保留，用于 mapping 审计。 |
| `mapping_status` | 793/793 | 映射状态/质量。 | STRING 导入 | 建议保留 |
| `updated_at` | 793/793 | 导入时间戳。 | 导入派生 | 删除候选；更适合保留在审计日志或 staging。 |

### `Protein` -[`INTERACTS_WITH`]-> `Protein`

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `interaction_id` | 7154/7154 | STRING interaction edge ID。 | STRING 导入 | 建议保留 |
| `source` | 7154/7154 | 来源。 | STRING | 建议保留 |
| `score` | 7154/7154 | STRING combined interaction score。 | STRING | 建议保留 |
| `required_score` | 7154/7154 | 导入时使用的 STRING score 阈值，目前是 high confidence。 | STRING 导入 | 建议保留 |
| `nscore` | 7154/7154 | STRING neighborhood score component。 | STRING | 人工复核 |
| `fscore` | 7154/7154 | STRING fusion score component。 | STRING | 人工复核 |
| `pscore` | 7154/7154 | STRING co-occurrence/phyletic score component。 | STRING | 人工复核 |
| `ascore` | 7154/7154 | STRING co-expression score component。 | STRING | 人工复核 |
| `escore` | 7154/7154 | STRING experimental score component。 | STRING | 人工复核 |
| `dscore` | 7154/7154 | STRING database score component。 | STRING | 人工复核 |
| `tscore` | 7154/7154 | STRING text-mining score component。 | STRING | 人工复核 |
| `ncbi_taxon_id` | 7154/7154 | Taxon ID。 | STRING | 人工复核；human-only 图谱里可能冗余。 |
| `species_name` | 7154/7154 | 物种名。 | 导入派生 | 人工复核；human-only 图谱里可能冗余。 |
| `updated_at` | 7154/7154 | 导入时间戳。 | 导入派生 | 删除候选；更适合保留在审计日志或 staging。 |

### `Gene` -[`PARTICIPATES_IN`]-> `Pathway`

这个关系类型同时承载 KEGG 和 Reactome pathway membership。

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `relationship_id` | 9947/9947 | pathway membership 的稳定 ID。 | KEGG/Reactome 导入 | 建议保留 |
| `source` | 9947/9947 | 来源：KEGG 或 Reactome。 | 导入 | 建议保留 |
| `kegg_gene_id` | 5310/9947 | KEGG gene identifier。 | KEGG | 建议保留，用于 KEGG membership。 |
| `kegg_pathway_id` | 5310/9947 | KEGG pathway accession。 | KEGG | 人工复核；目标 Pathway 节点已有该字段，但导出时有用。 |
| `mapping_identifier` | 4637/9947 | Reactome 映射使用的 identifier，通常是 Ensembl。 | Reactome | 建议保留，用于 mapping 审计。 |
| `reactome_stable_id` | 4637/9947 | 目标 Reactome stable ID。 | Reactome | 人工复核；目标 Pathway 节点已有该字段，但导出时有用。 |
| `evidence_code` | 4637/9947 | Reactome evidence code。 | Reactome | 建议保留/人工复核 |
| `updated_at` | 9947/9947 | 导入时间戳。 | 导入派生 | 删除候选；更适合保留在审计日志或 staging。 |

### `Gene` -[`EXPRESSED_IN`]-> `Tissue`

当前 v2 只有 tissue expression；cell-type expression 表为空。

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `relationship_id` | 798/798 | HPA expression edge ID。 | HPA 导入 | 建议保留 |
| `source` | 798/798 | 来源。 | HPA | 建议保留 |
| `assay` | 798/798 | assay 或数据模态。 | HPA | 建议保留 |
| `ensembl_gene_id` | 798/798 | HPA 使用的 Ensembl gene ID。 | HPA | 建议保留，用于 mapping 审计。 |
| `n_tpm` | 798/798 | liver 中的 normalized TPM expression。 | HPA | 建议保留 |
| `expression_unit` | 798/798 | 表达量单位，例如 `nTPM`。 | HPA | 建议保留 |
| `updated_at` | 798/798 | 导入时间戳。 | 导入派生 | 删除候选；更适合保留在审计日志或 staging。 |

### `Gene` -[`PROGNOSTIC_IN`]-> `Disease`

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `relationship_id` | 1384/1384 | HPA prognostic edge ID。 | HPA 导入 | 建议保留 |
| `source` | 1384/1384 | 来源。 | HPA | 建议保留 |
| `ensembl_gene_id` | 1384/1384 | HPA Ensembl gene ID。 | HPA | 建议保留，用于 mapping 审计。 |
| `cancer` | 1384/1384 | 癌种，目前是 LIHC。 | HPA | 建议保留 |
| `cohort` | 1384/1384 | survival association 使用的 cohort/dataset。 | HPA | 建议保留 |
| `prognostic_status` | 1384/1384 | 预后关联状态。 | HPA | 建议保留 |
| `prognostic_direction` | 1384/1384 | 预后方向，例如 favorable/unfavorable。 | HPA | 建议保留 |
| `p_value` | 1384/1384 | 统计 p-value。 | HPA | 建议保留 |
| `updated_at` | 1384/1384 | 导入时间戳。 | 导入派生 | 删除候选；更适合保留在审计日志或 staging。 |

### `Gene` -[`ASSOCIATED_WITH_METABOLITE`]-> `Metabolite`

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `relationship_id` | 97/97 | gene-metabolite edge ID。 | HMDB 导入 | 建议保留 |
| `source` | 97/97 | 来源。 | HMDB | 建议保留 |
| `gene_symbol` | 97/97 | 复制到边上的 gene symbol。 | HMDB/导入 | 删除候选；Gene 节点已有 symbol。 |
| `hmdb_protein_accessions` | 97/97 | 与 metabolite 相关的 HMDB protein accession。 | HMDB | 建议保留/人工复核 |
| `uniprot_ids` | 97/97 | HMDB protein entry 中的 UniProt ID。 | HMDB | 建议保留；有助于 protein mapping 审计。 |
| `protein_types` | 97/97 | HMDB protein role/type。 | HMDB | 人工复核 |
| `updated_at` | 97/97 | 导入时间戳。 | 导入派生 | 删除候选；更适合保留在审计日志或 staging。 |

### `Metabolite` -[`ASSOCIATED_WITH`]-> `Disease`

当前这类边和 DisGeNET gene-disease 边共用 `ASSOCIATED_WITH` 关系类型。后续建议改成单独关系类型，避免语义混用，例如 `METABOLITE_ASSOCIATED_WITH_DISEASE`。

| Attribute | 覆盖率 | 解释 | 来源 | 审核建议 |
|---|---:|---|---|---|
| `relationship_id` | HMDB subset 内 43/43 | HMDB disease-metabolite edge ID。 | HMDB 导入 | 建议保留 |
| `source` | HMDB subset 内 43/43 | 来源。 | HMDB | 建议保留 |
| `disease_name` | HMDB subset 内 43/43 | 复制到边上的 disease name。 | HMDB/导入 | 删除候选；Disease 节点已有名称。 |
| `updated_at` | HMDB subset 内 43/43 | 导入时间戳。 | 导入派生 | 删除候选；更适合保留在审计日志或 staging。 |

## 最优先人工核对的字段

1. `Gene.gene_ensembl_ids` vs `Gene.ensembl_gene_ids`：二选一作为 canonical Ensembl 字段。推荐保留 `gene_ensembl_ids`，删除或迁移 `ensembl_gene_ids`。
2. `Gene.gene_protein_str_ids` vs `Gene.protein_ids_from_disgenet`：明显重复。推荐只保留一个。
3. `Disease.name` vs `Disease.stage_code` vs `Disease.disease_name`：需要统一展示规范。推荐保留 `stage_code` 和 `disease_name`；`name` 可作为 UI alias。
4. `Pathway.name` vs `Pathway.pathway_name`：KEGG 节点中基本重复。推荐统一用 `name`。
5. `Protein.protein_id` vs `Protein.string_protein_id`：当前重复。推荐保留 `string_protein_id`；只有计划做 source-agnostic protein 层时才保留 `protein_id`。
6. `Metabolite.metabolite_id` vs `Metabolite.hmdb_id`：当前基本重复。如果 `metabolite_id` 是内部 ID 规范，可以两个都留；否则只留 `hmdb_id` 也足够。
7. `Metabolite.kingdom`、`Metabolite.super_class`、`Metabolite.class`：当前 TSV 值为空。推荐从最终图谱删除，或者先从 HMDB 补齐再保留。
8. `ASSOCIATED_WITH` 关系类型被混用：同时包含 DisGeNET gene-disease 和 HMDB metabolite-disease。推荐 final schema freeze 前把 metabolite-disease 拆成独立关系类型。
9. `updated_at`、`api_endpoint`、`api_query`、`retrieved_at`：有审计/provenance 价值，但会让主图很臃肿。推荐留在 staging/report，除非你需要在 Neo4j 内部直接查 reproducibility。
10. `confidence_score` vs `normalized_score`，以及 `evidence_level` vs `tier`：需要统一命名。推荐保留 `normalized_score` 和 `tier`，删除派生重复字段，除非 UI 或查询已经依赖这些别名。

## 更干净主图的建议保留字段

以下只是建议，不是已经执行的删除方案。

| 实体/关系 | 建议保留的最小字段 |
|---|---|
| `Disease` | `disease_id`, `name`, `disease_name`, `disease_type`, `external_ids`, `stage_order`, `disease_classes_msh`, `source`, `updated_at` |
| `Gene` | `gene_id`, `ncbi_gene_id`, `gene_symbol`, `name`, `gene_type`, `ensembl_gene_ids`, `gene_dsi`, `gene_dpi`, `gene_pli`, `protein_class_names`, `source`, `updated_at` |
| `Protein` | `protein_id`, `name`, `annotation`, `source`, `ncbi_taxon_id`, `species_name`, `updated_at` |
| `Pathway` | `pathway_id`, `name`, `source`, `ncbi_taxon_id`, `species_name`, `updated_at` |
| `Tissue` | `tissue_id`, `tissue_name`, `source` |
| `Metabolite` | `metabolite_id`, `name`, `chemical_formula`, `average_molecular_weight`, `monoisotopic_molecular_weight`, `kingdom`, `super_class`, `class`, `source`, `updated_at` |
| `ASSOCIATED_WITH` DisGeNET | `relation_id`, `assoc_id`, `source`, `score`, `normalized_score`, `num_pmids`, `year_initial`, `year_final`, `tier`, `tier_reason`, `validation_status` |
| `PROGRESSES_TO` | `progression_id`, `source`, `evidence_level`, `validation_status` |
| `ENCODES` | `source`, `mapping_input`, `mapping_status` |
| `INTERACTS_WITH` | `interaction_id`, `source`, `score`, `required_score`，以及可选的 component scores |
| `PARTICIPATES_IN` | `relationship_id`, `source`, `kegg_gene_id`, `mapping_identifier`, `evidence_code` |
| `EXPRESSED_IN` | `relationship_id`, `source`, `assay`, `ensembl_gene_id`, `n_tpm`, `expression_unit` |
| `PROGNOSTIC_IN` | `relationship_id`, `source`, `ensembl_gene_id`, `cancer`, `cohort`, `prognostic_status`, `prognostic_direction`, `p_value` |
| `ASSOCIATED_WITH_METABOLITE` | `relationship_id`, `source`, `hmdb_protein_accessions`, `uniprot_ids`, `protein_types` |
