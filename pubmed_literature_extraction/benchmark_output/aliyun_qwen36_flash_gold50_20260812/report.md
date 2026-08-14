# Fixed-candidate adjudicator benchmark

- Run: `aliyun_qwen36_flash_gold50_20260812`
- Documents: 50
- Workers per model: 10
- Upstream candidates fixed: yes
- Neo4j reads/writes: no/no
- Cost: unavailable for the custom MaaS workspace; tokens are reported.

| Model | Entity P/R/F1 | Gene/Protein | Relation P/R/F1 | Import-ready P/R/F1 | Evidence | Duplicate | p50/p95 | Wall | Tokens in/out | Invalid JSON | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3.6-flash | 0.235/0.846/0.368 | 0.905 | 0.400/0.311/0.350 | 0.400/0.364/0.381 | 0.286 | 0.099 | 0.00/12.19 | 17.3s | 34696/16273 | 0 | 0 |
