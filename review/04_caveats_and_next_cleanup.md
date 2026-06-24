# 04 注意事项与下一步清理

## 重要注意事项

1. `ASSOCIATED_WITH` 当前语义混合。
   - `Gene -[:ASSOCIATED_WITH]-> Disease` 来自 DisGeNET。
   - `Metabolite -[:ASSOCIATED_WITH]-> Disease` 来自 HMDB。
   - 范围审计是通过的，但如果要让 schema 语义更干净，建议后续把 HMDB 的 metabolite-disease 关系拆成单独关系类型。

2. STRING PPI 不是直接疾病证据。
   - PPI 表示 backbone Gene 编码 Protein 之间的互作背景。
   - 它应该被描述为 high-confidence STRING context，而不是直接疾病因果证据。

3. Pathway membership 不是疾病特异。
   - Pathway 被纳入，是因为它至少包含一个 backbone Gene。
   - 某些 Pathway 可能非常宽泛，不一定只和某一个疾病阶段相关。

4. HPA expression 是组织表达和预后背景。
   - Liver expression 支持组织相关性。
   - CellType expression 是 HPA 泛组织单细胞表达背景，不是 liver-only cell type 证据。
   - LIHC prognosis 支持 HCC 相关 cancer context。

5. HMDB 是保守筛选后的疾病相关代谢物背景。
   - 这不是全量 HMDB。
   - Metabolite 必须连接 backbone Gene，并且至少连接一个 backbone Disease。

## 建议下一步清理

1. 决定是否拆分 `ASSOCIATED_WITH`。
   - 建议：DisGeNET 继续使用 `ASSOCIATED_WITH`。
   - HMDB metabolite-disease 关系可以改成 `METABOLITE_ASSOCIATED_WITH_DISEASE` 或类似名称。

2. 审阅关系属性。
   - 实体属性已经清理过。
   - 关系属性仍然保留了较多 source-specific evidence 和 provenance 字段。

3. 决定最终 Tissue 和 CellType schema。
   - 当前 Tissue 只有一个 Liver 节点。
   - Tissue 可能可以简化为 `tissue_id`, `name`, `source`, `updated_at`。
   - CellType 当前保留 `cell_type_name`，它和 `name` 重复，后续可以人工决定是否删除。

4. 决定 archive 中旧报告是否长期保留。
   - 它们对审计和恢复有用。
   - 日常人工审阅不需要频繁查看这些旧报告。
