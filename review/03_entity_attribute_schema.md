# 03 实体属性 Schema

这是当前 Neo4j v2 中清理后的实体属性 schema。

## Disease

| 属性 |
|---|
| `disease_id` |
| `name` |
| `disease_name` |
| `disease_type` |
| `external_ids` |
| `stage_order` |
| `disease_classes_msh` |
| `source` |
| `updated_at` |

## Gene

| 属性 |
|---|
| `gene_id` |
| `ncbi_gene_id` |
| `gene_symbol` |
| `name` |
| `gene_type` |
| `ensembl_gene_ids` |
| `gene_dsi` |
| `gene_dpi` |
| `gene_pli` |
| `protein_class_names` |
| `source` |
| `updated_at` |

## Protein

| 属性 |
|---|
| `protein_id` |
| `name` |
| `annotation` |
| `source` |
| `ncbi_taxon_id` |
| `species_name` |
| `updated_at` |

## Pathway

| 属性 |
|---|
| `pathway_id` |
| `name` |
| `source` |
| `ncbi_taxon_id` |
| `species_name` |
| `updated_at` |

## Tissue

| 属性 |
|---|
| `tissue_id` |
| `name` |
| `tissue_name` |
| `source` |
| `updated_at` |

Tissue 还没有进行最终人工精简。目前只有一个 Liver 节点。

## CellType

CellType 已经从 HPA single-cell type expression 数据中启用。

注意：这里的 CellType 是 HPA 泛组织 single-cell type context，连接对象是 backbone Gene；它不是 liver-only cell type 证据。

| 属性 |
|---|
| `cell_type_id` |
| `name` |
| `cell_type_name` |
| `source` |
| `updated_at` |

当前节点数量：`154`

当前表达关系：

```text
Gene -[:EXPRESSED_IN]-> CellType
```

当前关系数量：`16947`

## Metabolite

| 属性 |
|---|
| `metabolite_id` |
| `name` |
| `chemical_formula` |
| `average_molecular_weight` |
| `monoisotopic_molecular_weight` |
| `kingdom` |
| `super_class` |
| `class` |
| `source` |
| `updated_at` |

## 详细属性审阅

完整逐字段审阅文件已归档在：

`archive/final_reports/v2_entity_attribute_review.md`
