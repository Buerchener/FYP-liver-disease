# Archive

这里存放过程文件、旧报告、机器审计结果和 schema 修改前备份。

日常人工审阅不需要看这里；只有在需要追溯、恢复字段或检查原始审计证据时才打开。

目录说明：

- `final_reports/`
  - 旧版较完整的 Markdown 报告。
  - 内容可能和 `review/` 有重复。

- `pre_prune_backups/`
  - 每次实体 attribute 裁剪前的 Neo4j 节点属性 JSON 备份。
  - 如果字段删多了，可以从这里恢复。

- `machine_audits/`
  - 机器审计的原始 JSON 输出。
  - 人工通常看 `review/02_backbone_scope_audit.md` 即可。

- `disgenet_notes/`
  - DisGeNET API 字段解释、字段对齐和抓取报告。
  - 用于追溯数据来源，不是日常入口。

- `old_docs/`
  - 整理前的 docs 文件。

- `misc/`
  - 其他辅助 TSV 或过程文件。
