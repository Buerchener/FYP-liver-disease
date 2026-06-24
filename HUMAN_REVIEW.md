# Human Review Entry

这份文件是人工审阅入口。

## 建议阅读顺序

1. `review/01_current_database_summary.md`
   - 看当前数据库包含哪些节点和关系。
   - 看每个数据源导入了多少。

2. `review/02_backbone_scope_audit.md`
   - 看是否严格围绕五个 disease backbone。
   - 看是否有孤儿节点、越界 disease、无关数据。

3. `review/03_entity_attribute_schema.md`
   - 看每类实体最终保留哪些 attribute。
   - 用于继续人工删字段或确认 schema freeze。

4. `review/04_caveats_and_next_cleanup.md`
   - 看还没解决但不影响当前 scope 的 schema 问题。

## 当前结论

当前 v2 数据库基本构件已经完成，并且通过 backbone scope audit。

没有发现：

- 额外 Disease 节点
- 越界 Disease 关系
- 无 Disease 追溯的 Gene
- 无 backbone Gene 追溯的 Protein
- 无 backbone Gene 追溯的 Pathway
- 无 backbone Gene/Disease 追溯的 Metabolite

## 当前需要人工继续决定的重点

1. 是否拆分 `ASSOCIATED_WITH`
   - 当前它同时表示 Gene-Disease 和 Metabolite-Disease。
   - Scope 没问题，但语义上不够干净。

2. 是否继续精简关系属性
   - 目前实体属性已经大量清理。
   - 关系属性里还有很多 source-specific evidence 字段。

3. 是否保留 HPA/STRING/Pathway 的上下文解释边界
   - 它们是 backbone-derived context。
   - 不应被写成每条都是直接 disease evidence。
