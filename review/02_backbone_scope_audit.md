# 02 Backbone 范围审计

## 结论

当前 v2 数据通过 backbone 范围审计。

没有发现无关 Disease 节点、孤立节点，或指向 backbone 之外疾病的关系。

所有非 Disease 数据层都可以追溯到五个 backbone 疾病相关 Gene，或直接追溯到五个 backbone Disease。

## 审计规则

| 数据层 | 必须满足的条件 | 结果 |
|---|---|---|
| Disease | 只能存在五个 backbone Disease 节点 | 通过 |
| Progression | `PROGRESSES_TO` 只能连接 backbone Disease 节点 | 通过 |
| Gene | 每个 Gene 必须连接至少一个 backbone Disease | 通过 |
| Protein | 每个 Protein 必须由 backbone Gene 编码 | 通过 |
| PPI | 每条 PPI 的两端 Protein 都必须由 backbone Gene 编码 | 通过 |
| Pathway | 每个 Pathway 必须至少有一个 backbone Gene 参与 | 通过 |
| HPA expression | 每条 expression 关系都必须从 backbone Gene 出发 | 通过 |
| HPA prognosis | 每条 prognostic 关系必须从 backbone Gene 出发，并指向 backbone Disease | 通过 |
| Metabolite | 每个 Metabolite 必须同时连接 backbone Gene 和 backbone Disease | 通过 |

## 关键审计结果

| 检查项 | 异常数量 |
|---|---:|
| Backbone 之外的 Disease 节点 | 0 |
| Backbone 之外的 progression 关系 | 0 |
| 没有 Disease 关联的 Gene | 0 |
| 连接非 backbone Disease 的 Gene | 0 |
| 没有 Gene 编码的 Protein | 0 |
| 端点未被 Gene 编码的 PPI 关系 | 0 |
| 没有 Gene 参与的 Pathway | 0 |
| 来自无 Disease Gene 的 Pathway 关系 | 0 |
| 来自无 Disease Gene 的 HPA expression | 0 |
| 指向非 backbone Disease 的 HPA prognosis | 0 |
| 没有 Gene 关系的 Metabolite | 0 |
| 没有 Disease 关系的 Metabolite | 0 |
| 连接非 backbone Disease 的 Metabolite | 0 |

## 疾病分布

| 疾病 | Disease ID | GDA 关系数 | 去重 Gene 数 |
|---|---|---:|---:|
| NAFLD | `UMLS:C0400966` | 103 | 103 |
| NASH | `UMLS:C3241937` | 107 | 107 |
| Fibrosis | `UMLS:C0239946` | 2 | 2 |
| Cirrhosis | `UMLS:C0023890` | 169 | 169 |
| HCC | `UMLS:C2239176` | 655 | 655 |

## 解释边界

STRING、KEGG、Reactome、HPA、HMDB 的下游数据属于 backbone-derived context layer。

这些数据被认为在范围内，是因为它们连接到 backbone Gene 或五个 backbone Disease。但除非关系本身带有疾病证据，否则不能把它们解释为直接疾病证据。

HPA CellType expression 是 backbone Gene 的泛组织单细胞表达背景，不是 liver-only cell type 证据。

当前 HPA CellType 层：

- CellType 节点：`154`
- Gene-to-CellType expression 关系：`16947`
- 有 CellType expression 的 Gene：`645`
- 导入阈值：`nCPM >= 100`
- 来自无 Disease Gene 的 CellType expression 关系：`0`

完整原始审计 JSON 已归档在：

`archive/machine_audits/v2_backbone_scope_audit_20260624.json`
