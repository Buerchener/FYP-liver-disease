# 01 当前数据库概览

数据库：`liver-kg-core-v02`

## 疾病 Backbone

```text
NAFLD -> NASH -> Fibrosis -> Cirrhosis -> HCC
```

Backbone 疾病 ID：

| 阶段 | Disease ID |
|---|---|
| NAFLD | `UMLS:C0400966` |
| NASH | `UMLS:C3241937` |
| Fibrosis | `UMLS:C0239946` |
| Cirrhosis | `UMLS:C0023890` |
| HCC | `UMLS:C2239176` |

## 节点数量

| 标签 | 数量 |
|---|---:|
| Disease | 5 |
| Gene | 836 |
| Protein | 793 |
| Pathway | 1721 |
| Tissue | 1 |
| CellType | 154 |
| Metabolite | 36 |

## 关系数量

| 关系 | 数量 |
|---|---:|
| `ASSOCIATED_WITH` | 1079 |
| `PROGRESSES_TO` | 4 |
| `ENCODES` | 793 |
| `INTERACTS_WITH` | 7154 |
| `PARTICIPATES_IN` | 9947 |
| `EXPRESSED_IN` | 17745 |
| `PROGNOSTIC_IN` | 1384 |
| `ASSOCIATED_WITH_METABOLITE` | 97 |

## 数据来源层

| 来源 | 导入内容 |
|---|---|
| DisGeNET | Disease、Gene、Gene-Disease 关联 |
| STRING | Gene-Protein 映射、高置信 PPI |
| KEGG | Gene-Pathway 关系 |
| Reactome | Gene-Pathway 关系 |
| HPA | Liver tissue expression、泛组织 single-cell type expression、LIHC prognosis context |
| HMDB | 保守筛选后的疾病相关 metabolite context |

## 分来源数量

| 数据层 | 数量 |
|---|---:|
| DisGeNET Gene-Disease associations | 1036 |
| STRING mapped Proteins | 793 |
| STRING PPI edges | 7154 |
| KEGG pathways | 333 |
| KEGG memberships | 5310 |
| Reactome pathways | 1388 |
| Reactome memberships | 4637 |
| HPA liver tissue expression | 798 |
| HPA cell types | 154 |
| HPA cell type expression | 16947 |
| HPA LIHC prognostic relations | 1384 |
| HMDB disease-linked metabolites | 36 |
| HMDB Gene-Metabolite relations | 97 |
| HMDB Metabolite-Disease relations | 43 |
