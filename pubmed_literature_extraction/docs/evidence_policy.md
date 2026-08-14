# 证据边界政策

## 直接疾病证据（direct_disease_evidence）

关系本身直接表达疾病、疾病阶段、预后或明确进展关系，并有原文或疾病数据库证据支持：

- DisGeNET `Gene -[:ASSOCIATED_WITH]-> Disease`
- 文献中有明确证据句支持的 `Gene/Protein/Metabolite/Pathway -> Disease`
- `PROGNOSTIC_IN`
- 明确表达疾病阶段进展的 `PROGRESSES_TO`

## 上下文背景（contextual_background）

关系可以帮助解释核心图谱，但不能单独作为直接疾病因果证据：

- STRING `INTERACTS_WITH`
- KEGG / Reactome `PARTICIPATES_IN`
- HPA tissue/cell-type `EXPRESSED_IN`
- 一般 Gene/Protein/Pathway/Tissue/CellType 功能关联
- HMDB 代谢物上下文，包括 `ASSOCIATED_WITH_METABOLITE`

`ASSOCIATED_WITH` 目前同时承载 Gene-Disease 和 Metabolite-Disease；应结合 `source`、端点类型和 `evidence_class` 解读，暂不直接改写既有关系类型。

## 推断或待审核知识（inferred_or_hypothesis）

- 因果推理器生成的传递关系
- 仅由上下文关系推断出的疾病关联
- 证据不确定、存在冲突或需要人工裁决的关系

所有 PubMed 候选关系都不是 curated fact；它们必须保留 PMID、证据句、字符偏移、置信度、物种、方向和 `validation_status`。`source` 表示来源，`evidence_class` 表示解释边界，两者不可混用。
