# DisGeNET API 字段对齐报告

本报告使用一小批实时 DisGeNET GDA API 样本，检查其返回字段是否能映射到当前的 `ontology/ontology_v1.0.yaml`。
报告不会保存 API key，也不会保存完整 API payload。

## API 样本

- Endpoint: `https://api.disgenet.com/api/v1/gda/summary`
- Query: `gene_ncbi_id=351&page_number=0`
- 本次检查记录数: `30`
- 该 query 可返回的总记录数: `867`
- 顶层 response keys: `status, paging, warnings, requestpar, userinfo, payload, httpStatus`

## 字段映射

| DisGeNET field | 观察到的数据类型 | 样本中为空 | Ontology target | 对齐情况 |
| --- | --- | ---: | --- | --- |
| `assocID` | string x30 | 0/30 | Relation provenance: `source_record_id` | 已覆盖 |
| `chemsIncludedInEvidenceBySource` | array x12, array[object:11] x1, array[object:12] x1, array[object:1] x4, array[object:241] x1, array[object:24] x1, array[object:2] x1, array[object:3] x1, array[object:41] x1, array[object:4] x4, array[object:6] x1, array[object:97] x1, array[object:9] x1 | 12/30 | Relation evidence: `chemical evidence details` | 建议作为嵌套的 source-specific attribute 保存 |
| `diseaseClasses_DO` | array x14, array[string:1] x11, array[string:2] x5 | 14/30 | Disease: `disease classification` | 建议作为 source-specific attribute 保存 |
| `diseaseClasses_HPO` | array x18, array[string:1] x12 | 18/30 | Disease: `disease classification` | 建议作为 source-specific attribute 保存 |
| `diseaseClasses_MSH` | array[string:1] x16, array[string:2] x8, array[string:3] x2, array[string:4] x4 | 0/30 | Disease: `disease classification` | 建议作为 source-specific attribute 保存 |
| `diseaseClasses_UMLS_ST` | array[string:1] x30 | 0/30 | Disease: `disease classification` | 建议作为 source-specific attribute 保存 |
| `diseaseName` | string x30 | 0/30 | Disease: `disease_name` | 已覆盖 |
| `diseaseType` | string x30 | 0/30 | Disease: `disease_type` | 已在 planned 中覆盖 |
| `diseaseUMLSCUI` | string x30 | 0/30 | Disease: `umls_id / disease_id` | 已覆盖 |
| `diseaseVocabularies` | array[string:10] x2, array[string:11] x1, array[string:2] x3, array[string:34] x1, array[string:3] x1, array[string:4] x4, array[string:5] x1, array[string:6] x7, array[string:7] x4, array[string:8] x4, array[string:9] x2 | 0/30 | Disease: `external_ids` | 已覆盖 |
| `disease_inheritance` | string x30 | 25/30 | Disease: `inheritance metadata` | 可选 source-specific attribute |
| `disease_prevalence_class` | string x30 | 25/30 | Disease: `epidemiology metadata` | 可选 source-specific attribute |
| `disease_prevalence_geo_area` | string x30 | 25/30 | Disease: `epidemiology metadata` | 可选 source-specific attribute |
| `disease_prevalence_type` | string x30 | 25/30 | Disease: `epidemiology metadata` | 可选 source-specific attribute |
| `ei` | number x30 | 0/30 | Relation evidence: `evidence index` | 建议作为 source-specific attribute 保存 |
| `el` | null x26, string x4 | 26/30 | Relation evidence: `evidence level metric` | 建议作为 source-specific attribute 保存 |
| `geneDPI` | number x30 | 0/30 | Gene attribute: `DisGeNET-specific metric` | 建议作为 source-specific attribute 保存 |
| `geneDSI` | number x30 | 0/30 | Gene attribute: `DisGeNET-specific metric` | 建议作为 source-specific attribute 保存 |
| `geneEnsemblIDs` | array[string:1] x30 | 0/30 | Gene: `external_ids.Ensembl` | 已覆盖 |
| `geneNcbiID` | integer x30 | 0/30 | Gene: `gene_id / primary_external_id` | 已覆盖 |
| `geneNcbiType` | string x30 | 0/30 | Gene: `gene_type` | 已在 planned 中覆盖 |
| `geneProteinClassIDs` | array[string:1] x30 | 0/30 | Gene/Protein class: `source-specific attribute` | 建议作为 source-specific attribute 保存 |
| `geneProteinClassNames` | array[string:1] x30 | 0/30 | Gene/Protein class: `source-specific attribute` | 建议作为 source-specific attribute 保存 |
| `geneProteinStrIDs` | array[string:1] x30 | 0/30 | Gene/Protein crossref: `external_ids.STRING / protein link candidate` | 可作为 external_ids 保存 |
| `genepLI` | number x30 | 0/30 | Gene attribute: `DisGeNET-specific metric` | 建议作为 source-specific attribute 保存 |
| `normalized_score` | number x30 | 0/30 | Relation provenance: `confidence_score` | 已覆盖 |
| `numCTsupportingAssociation` | integer x30 | 0/30 | Relation evidence: `support count` | 建议作为 source-specific attribute 保存 |
| `numChemsIncludedInEvidences` | integer x30 | 0/30 | Relation evidence: `chemical evidence count` | 建议作为 source-specific attribute 保存 |
| `numDBSNPsupportingAssociation` | integer x30 | 0/30 | Relation evidence: `support count` | 建议作为 source-specific attribute 保存 |
| `numNCTSWithChemsIncludedInEvidences` | integer x30 | 0/30 | Relation evidence: `clinical trial chemical count` | 建议作为 source-specific attribute 保存 |
| `numPMIDSWithChemsIncludedInEvidences` | integer x30 | 0/30 | Relation evidence: `chemical publication count` | 建议作为 source-specific attribute 保存 |
| `numPMIDs` | integer x30 | 0/30 | Relation evidence: `publication support count` | 建议作为 source-specific attribute 保存 |
| `score` | number x30 | 0/30 | Relation: `score` | 已覆盖 |
| `scoreBreakdown` | array[object:1] x30 | 0/30 | Relation evidence: `score components` | 建议作为 source-specific attribute 保存 |
| `symbolOfGene` | string x30 | 0/30 | Gene: `gene_symbol` | 已覆盖 |
| `yearFinal` | integer x30 | 0/30 | Relation provenance: `latest evidence year / date proxy` | 建议作为 source-specific attribute 保存 |
| `yearInitial` | integer x30 | 0/30 | Relation provenance: `first evidence year` | 建议作为 source-specific attribute 保存 |

## 结论

- 当前 ontology 在结构层面与 DisGeNET 对齐：DisGeNET GDA 可以映射为 `Gene associated_with Disease`。
- 核心 identifier 已经覆盖：`symbolOfGene`、`geneNcbiID`、`geneEnsemblIDs`、`diseaseUMLSCUI`、`diseaseName` 和 `diseaseVocabularies`。
- 核心 relation 字段也已经覆盖：`assocID` 可以映射到 `source_record_id`，`score` 可以映射到 `score`，`normalized_score` 可以映射到 `confidence_score`。
- `geneDSI`、`geneDPI`、`genepLI`、`ei`、`el`、`scoreBreakdown`、support counts 以及 disease class arrays 这类 DisGeNET-specific metrics 不需要强行变成新的核心实体。它们更适合作为相关 entity 或 relation 的 source-specific `attributes` 保存。
- 因此，DisGeNET 目前不要求你重新设计 ontology。一个合理的小优化是：在 YAML 中显式加入 `source_specific_attributes` 的说明，或者在 `Gene`、`Disease` 和 `associated_with` 下面补充一条 `DisGeNET metrics` 备注。
