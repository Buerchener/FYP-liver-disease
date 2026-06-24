# DisGeNET API 字段解释

本文件分成两部分：第一部分解释 DisGeNET `gda/summary` API 返回的原始字段；第二部分列出本项目当前 `ontology/ontology_v1.0.yaml` 中已经定义的 `Gene`、`Disease` 和 `associated_with` relation 字段。

这两部分不是同一件事：DisGeNET API 字段是外部数据库返回的字段，本项目 ontology 字段是我们自己定义的统一 schema。后续数据导入时，需要把 API 字段映射到 ontology 字段。

> 说明：部分评分字段的精确定义应以 DisGeNET 官方文档为准。本文件重点说明它们在本项目 ontology/schema 中的用途。

## 第一部分：DisGeNET API 原始字段

这一部分列出本次 API 请求返回的字段。它们来自 DisGeNET 的一条 GDA record，不等于本项目 ontology 的字段定义。

### Disease 相关字段

| API 字段 | 含义 | 后续映射建议 |
| --- | --- | --- |
| `diseaseName` | Disease 名称。 | `Disease.disease_name` |
| `diseaseType` | Disease 类型，例如 disease、phenotype 等。 | `Disease.disease_type` |
| `diseaseUMLSCUI` | UMLS CUI，是 disease 的重要标准 ID。 | `Disease.umls_id` / `disease_id` |
| `diseaseVocabularies` | Disease 在多个词表里的 cross-reference，例如 MeSH、MONDO、OMIM、DO、HPO、UMLS。 | `Disease.external_ids` |
| `diseaseClasses_MSH` | MeSH disease class 分类。 | `Disease.attributes` |
| `diseaseClasses_UMLS_ST` | UMLS semantic type 分类，例如 Disease or Syndrome。 | `Disease.attributes` |
| `diseaseClasses_DO` | Disease Ontology 分类。 | `Disease.attributes` |
| `diseaseClasses_HPO` | Human Phenotype Ontology 分类。 | `Disease.attributes` |
| `disease_prevalence_class` | Disease prevalence 的类别信息，部分记录为空。 | Optional source-specific attribute |
| `disease_prevalence_geo_area` | Prevalence 对应的地理区域，部分记录为空。 | Optional source-specific attribute |
| `disease_prevalence_type` | Prevalence 类型，部分记录为空。 | Optional source-specific attribute |
| `disease_inheritance` | Disease inheritance 信息，例如遗传模式；很多记录为空。 | Optional source-specific attribute |

### Gene 相关字段

| API 字段 | 含义 | 后续映射建议 |
| --- | --- | --- |
| `symbolOfGene` | Gene symbol，例如 `APP`、`TP53`、`SOD1`。 | `Gene.gene_symbol` |
| `geneNcbiID` | NCBI Gene ID，是 gene 的标准外部 ID。 | `Gene.gene_id` / `primary_external_id` |
| `geneEnsemblIDs` | Ensembl gene ID 列表，例如 `ENSG...`。 | `Gene.external_ids` |
| `geneNcbiType` | Gene 类型，例如 `protein-coding`。 | `Gene.gene_type` |
| `geneDSI` | Disease Specificity Index，表示该 gene 是否更集中地关联少数疾病。值越高通常说明该 gene 更 disease-specific。 | `Gene.attributes` |
| `geneDPI` | Disease Pleiotropy Index，表示该 gene 涉及多少疾病类别。值越高通常说明该 gene 更 pleiotropic。 | `Gene.attributes` |
| `genepLI` | pLI score，与 loss-of-function intolerance 相关，用于描述 gene 对功能缺失突变的耐受性。 | `Gene.attributes` |

### Gene-Disease relation 相关字段

| API 字段 | 含义 | 后续映射建议 |
| --- | --- | --- |
| `assocID` | DisGeNET 给这条 gene-disease association 分配的唯一 ID。适合用于追踪原始来源记录。 | Relation provenance: `source_record_id` |
| `score` | DisGeNET association score，表示这条 gene-disease relation 的证据强度或综合评分。 | Relation: `score` |
| `normalized_score` | 标准化后的 score，适合表示 relation 的置信度。 | Relation provenance: `confidence_score` |
| `scoreBreakdown` | `score` 的组成来源，说明最终 score 由哪些证据部分贡献。 | Relation source-specific attribute |
| `ei` | Evidence Index，与证据一致性或冲突程度相关。 | Relation source-specific attribute |
| `el` | Evidence Level，DisGeNET 自己的证据等级字段；样本中多数为空。 | Relation source-specific attribute |
| `numPMIDs` | 支持这条 association 的 PubMed 文献数量。注意它不是单个 PMID，而是数量统计。 | Relation source-specific attribute |
| `numDBSNPsupportingAssociation` | 支持这条 association 的 dbSNP variant 数量。 | Relation source-specific attribute |
| `numCTsupportingAssociation` | 支持这条 association 的 clinical trial 数量。 | Relation source-specific attribute |
| `numChemsIncludedInEvidences` | 证据中涉及的 chemical/drug 数量。 | Relation source-specific attribute |
| `numPMIDSWithChemsIncludedInEvidences` | 证据中同时涉及 chemical 的 PubMed 文献数量。 | Relation source-specific attribute |
| `numNCTSWithChemsIncludedInEvidences` | 证据中同时涉及 chemical 的 clinical trial 数量。 | Relation source-specific attribute |
| `yearInitial` | 最早支持该 association 的证据年份。 | Relation source-specific attribute |
| `yearFinal` | 最新支持该 association 的证据年份。 | Relation source-specific attribute |

### 其他补充字段

| API 字段 | 含义 | 后续映射建议 |
| --- | --- | --- |
| `geneProteinStrIDs` | STRING protein ID 列表，可作为 gene 与 protein/STRING 网络之间的 cross-reference。它有助于后续把 DisGeNET 的 Gene 和 STRING/HPA 的 Protein 连接起来。 | `Gene.external_ids` / Protein cross-reference |
| `geneProteinClassIDs` | Protein class 的 ID，例如某些蛋白类别 ontology ID。 | Source-specific attribute |
| `geneProteinClassNames` | Protein class 名称，例如 enzyme、transporter 等。 | Source-specific attribute |
| `chemsIncludedInEvidenceBySource` | 证据中出现的 chemical/drug 详细列表，可能包含 chemical ID、名称、cross-reference、相关 PMID 等。这个字段比较复杂，适合作为 nested attribute 保留，暂时不建议扩展成核心实体。 | Nested relation source-specific attribute |

## 第二部分：本项目当前 ontology 字段

这一部分来自本项目当前的 `ontology/ontology_v1.0.yaml`。这些是我们自己定义的统一字段，不要求和 DisGeNET API 字段同名。

### `Gene` entity 字段

| 字段层级 | 当前 ontology 字段 |
| --- | --- |
| confirmed | `gene_id`, `gene_symbol`, `gene_name` |
| recommended | `project_id`, `primary_external_id`, `external_ids`, `synonyms`, `source` |
| planned | `chromosome_location`, `gene_type`, `organism` |

DisGeNET 的 `symbolOfGene`、`geneNcbiID`、`geneEnsemblIDs`、`geneNcbiType` 可以映射到这些字段；`geneDSI`、`geneDPI`、`genepLI` 这类 DisGeNET-specific metrics 更适合作为 `Gene.attributes` 或 `source_specific_attributes` 保存。

### `Disease` entity 字段

| 字段层级 | 当前 ontology 字段 |
| --- | --- |
| confirmed | `disease_id`, `disease_name`, `umls_id` |
| recommended | `project_id`, `mesh_id`, `omim_id`, `synonyms`, `is_progression_stage`, `stage_order` |
| planned | `disease_type`, `clinical_stage_definition`, `description` |

DisGeNET 的 `diseaseName`、`diseaseUMLSCUI`、`diseaseVocabularies`、`diseaseType` 可以映射到这些字段；`diseaseClasses_*`、`disease_prevalence_*` 和 `disease_inheritance` 更适合作为 `Disease.attributes` 或 `source_specific_attributes` 保存。

### `associated_with` relation 字段

当前 ontology 中，DisGeNET 的 GDA 主要对应这条 relation：

```text
Gene -- associated_with --> Disease
```

| 字段层级 | 当前 ontology 字段 |
| --- | --- |
| confirmed | `association_type`, `score`, `evidence_level`, `source`, `publication_id`, `evidence_text`, `date` |
| recommended | `source_record_id`, `validation_status`, `extraction_method`, `confidence_score` |
| planned | `notes` |

DisGeNET 的 `assocID`、`score`、`normalized_score` 最关键，分别可以映射为 `source_record_id`、`score` 和 `confidence_score`。`scoreBreakdown`、`ei`、`el`、support counts、`yearInitial`、`yearFinal` 等字段主要是这条 relation 的 evidence/provenance 补充。

### 全局 relation provenance 字段

当前 ontology 还定义了全局 relation provenance 规则：

| 类型 | 当前 ontology 字段 |
| --- | --- |
| required_relation_provenance | `source`, `evidence_level`, `date` |
| recommended_relation_provenance | `publication_id`, `evidence_text`, `source_record_id`, `extraction_method`, `confidence_score`, `validation_status`, `validated_by`, `notes` |

## 对当前 ontology 的影响

当前 ontology 不需要因为 DisGeNET API 字段而大改。核心结构仍然是：

```text
Gene -- associated_with --> Disease
```

更合理的处理方式是：

- 标准 ID 和名称字段进入 `Gene` / `Disease` 的核心属性。
- `assocID`、`score`、`normalized_score` 进入 relation 的核心或 provenance 属性。
- DisGeNET-specific metrics、support counts、disease class arrays 和 chemical evidence details 进入 `attributes` 或 `source_specific_attributes`。

这样既能保留 DisGeNET 的丰富信息，也不会让 ontology 被某一个数据库的细节字段牵着走。
