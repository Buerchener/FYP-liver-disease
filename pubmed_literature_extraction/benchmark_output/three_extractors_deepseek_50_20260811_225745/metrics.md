# Three-extractor benchmark

- Run: `three_extractors_deepseek_50_20260811_225745`
- Documents: 50
- Model: `deepseek-chat`
- Workers: 10
- Gold: relation-centric (45 semantic relations; 22 import-ready relations).
- Cost: reference estimate, not the third-party proxy invoice.

| Arm | Entity P/R/F1* | Gene/Protein type acc. | Semantic relation P/R/F1 | Import-ready relation P/R/F1 | Evidence precision | Duplicate rate | Latency p50/p95 s | Est. cost | Invalid JSON rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original_agent | 0.282/0.821/0.419 | 0.950 | 0.292/0.422/0.345 | 0.333/0.045/0.080 | 0.108 | 0.118 | 2.2/5.9 | $0.0155 | 0.000 |
| langextract_llm_hybrid | 0.266/0.829/0.402 | 0.950 | 0.199/0.644/0.304 | 0.500/0.091/0.154 | 0.112 | 0.106 | 8.4/19.4 | $0.0784 | 0.000 |
| deepseek_direct | 0.277/0.718/0.400 | 0.947 | 0.135/0.689/0.226 | 0.167/0.045/0.071 | 0.083 | 0.003 | 4.1/11.5 | $0.0238 | 0.000 |

* Entity labels contain relation endpoints plus core diseases; they are not exhaustive NER labels.
Evidence precision requires a matched triple, continuous source evidence, and character-span IoU >= 0.5 with gold evidence.
