# Fixed-candidate adjudicator benchmark

- Run: `aliyun_light_models_gold50_20260812`
- Documents: 50
- Workers per model: 10
- Upstream candidates fixed: yes
- Neo4j reads/writes: no/no
- Cost: unavailable for the custom MaaS workspace; tokens are reported.

| Model | Entity P/R/F1 | Gene/Protein | Relation P/R/F1 | Import-ready P/R/F1 | Evidence | Duplicate | p50/p95 | Wall | Tokens in/out | Invalid JSON | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-v4-flash | 0.235/0.846/0.368 | 0.905 | 0.475/0.422/0.447 | 0.500/0.455/0.476 | 0.350 | 0.099 | 0.00/7.53 | 11.1s | 34933/11074 | 0 | 0 |
| qwen-flash | 0.235/0.846/0.368 | 0.905 | 0.457/0.356/0.400 | 0.435/0.455/0.444 | 0.400 | 0.099 | 0.01/10.25 | 14.4s | 34676/10401 | 0 | 0 |
| qwen3.5-flash | 0.235/0.846/0.368 | 0.905 | 0.405/0.333/0.366 | 0.444/0.364/0.400 | 0.297 | 0.099 | 0.01/17.44 | 45.1s | 34696/19538 | 0 | 0 |
| qwen3-30b-a3b-instruct-2507 | 0.235/0.846/0.368 | 0.905 | 0.394/0.289/0.333 | 0.476/0.455/0.465 | 0.333 | 0.099 | 0.01/11.44 | 20.6s | 34676/10980 | 0 | 0 |
