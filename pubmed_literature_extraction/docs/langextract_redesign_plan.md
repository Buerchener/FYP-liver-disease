# PubMed 抽取系统重设计：基于 Google LangExtract

## 从手写 Pipeline 到 LangExtract 框架

> 日期：2026-06-28
> 设计方案 v1.0

---

## 1. Google LangExtract 是什么

[LangExtract](https://github.com/google/langextract) 是 Google 开源的 Python 库（`pip install langextract`），专为**从非结构化文本中提取结构化信息**设计，核心能力：

| 能力 | 说明 |
|---|---|
| **Schema = Examples** | 通过 Few-shot 示例定义输出结构，无需单独写 schema 文件 |
| **Controlled Generation** | 利用 Gemini 原生结构化输出，保证 JSON 格式匹配示例 |
| **Source Grounding** | 每个实体自动标注在原文中的精确字符偏移量 |
| **Chunking + 并行** | 长文档自动分块，多 worker 并行处理 |
| **交互式可视化** | 内置 HTML 可视化，可探索实体在原文中的位置 |
| **多模型支持** | Gemini（默认）、OpenAI GPT-4o、Ollama 本地模型 |

---

## 2. 当前系统 vs LangExtract

| 维度 | 当前手写 Pipeline | 基于 LangExtract |
|---|---|---|
| **代码量** | 1,628 行单体脚本 | ~400 行编排代码 |
| **LLM 调用** | `urllib.request` 手写 HTTP | `lx.extract()` 一行调用 |
| **Schema 约束** | Prompt 指令（软）+ Stage 2 硬校验 | Few-shot 示例 + Gemini 受控生成 |
| **证据定位** | 手写子串匹配 `_locate_evidence()` | **内置字符级对齐**，自动标注 char_interval |
| **长文档** | 单次 LLM 调用，最多 ~4000 token | 自动分块 + 并行提取 + 聚合 |
| **输出格式** | 自定义 JSON | 标准化 `AnnotatedDocument` |
| **可视化** | 手写 HTML 模板 ~90 行 | `lx.visualize()` 一行 |
| **模型切换** | 硬编码 DeepSeek | 配置切换 Gemini/GPT-4o/Ollama |
| **错误处理** | 15+ 项手写 quality_flags | `alignment_status` 自动标注 EXACT/FUZZY/NO_MATCH |

---

## 3. 重设计架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    新 PubMed 抽取系统                              │
│                   based on Google LangExtract                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  PubMed JSONL ───→ pubmed_extractor.py (新, ~400行)               │
│                    │                                               │
│                    ├─ Phase 1: LangExtract 抽取 (替代 Stage 0+1)  │
│                    │   · lx.extract() with Gemini                 │
│                    │   · extraction_class → KG entity type        │
│                    │   · attributes → relation metadata           │
│                    │   · 自动 char_interval → evidence grounding  │
│                    │                                               │
│                    ├─ Phase 2: KG Schema 适配层 (替代 Stage 2+3)  │
│                    │   · 关系签名校验 (保留 RELATION_SIGNATURES)   │
│                    │   · 实体标准化 (保留 KNOWN_GENE_IDS 等)      │
│                    │   · 物种/否定/不确定检测 (保留正则)           │
│                    │   · unsupported_entity_class 检测 (保留)     │
│                    │                                               │
│                    ├─ Phase 3: 冲突检测 (保留 Stage 4)            │
│                    │                                               │
│                    └─ Phase 4: Neo4j 写入 (保留 Stage 5)          │
│                                                                    │
│                              ↓                                     │
│                    extraction_results.json                         │
│                    provenance_report.html (lx.visualize)           │
│                              ↓                                     │
│              entity_linking_preflight.py (完全不动)                │
│                              ↓                                     │
│                         Neo4j                                      │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 3.1 保留的部分（不动）

- `entity_linking_preflight.py` — **完全不变**，LangExtract 不涉及实体链接
- Stage 2 的校验规则（物种/否定/不确定/unsupported_entity_class）— **逻辑保留**，移植到新架构
- Stage 5 Neo4j 写入模块 — **完全不变**
- `convert_pubmed_xml_to_jsonl.py` — **完全不变**

### 3.2 替换的部分

| 旧代码 | 新方式 |
|---|---|
| `call_deepseek_api()` + `build_stage1_prompt()` (~200行) | `lx.extract()` 一行 |
| `_locate_evidence()` (~20行) | LangExtract 内置 `char_interval` |
| `generate_provenance_report()` (~90行) | `lx.visualize()` 一行 |
| Stage 0 文本分类 (~40行) | 不需要了，PubMed 全是 literature_abstract |
| `coerce_extraction()` (~30行) | LangExtract 输出已是结构化对象 |
| Few-shot 示例散落在代码中 | 集中在 `schema/examples.py` |

---

## 4. 核心设计：KG Schema → LangExtract Mapping

### 4.1 实体类型映射

```python
# 当前: 7 种实体类型硬编码在 ALLOWED_ENTITY_TYPES
# LangExtract: 每种实体类型 = 一个 extraction_class

ENTITY_CLASSES = {
    "gene": {
        "attributes": ["gene_symbol", "species", "normalized_id"],
        "description": "Gene mentioned in the text, e.g. TP53, EGFR, NFE2L2",
    },
    "disease": {
        "attributes": ["disease_name", "disease_stage", "normalized_id"],
        "description": "Disease or disease stage, e.g. HCC, NAFLD, liver fibrosis",
    },
    "protein": {
        "attributes": ["protein_name", "species", "normalized_id"],
        "description": "Protein, e.g. p53, EGFR protein, MAPK14",
    },
    "pathway": {
        "attributes": ["pathway_name", "source_db"],
        "description": "Biological pathway, e.g. ferroptosis, TNF signaling",
    },
    "metabolite": {
        "attributes": ["metabolite_name", "hmdb_id"],
        "description": "Small molecule metabolite, e.g. glucose, ferulic acid",
    },
    "tissue": {
        "attributes": ["tissue_name"],
        "description": "Tissue or organ, e.g. liver, hepatic tissue",
    },
    "cell_type": {
        "attributes": ["cell_type_name"],
        "description": "Cell type, e.g. hepatocyte, Kupffer cell, T cell",
    },
}
```

### 4.2 关系类型映射

LangExtract 没有原生的「关系」概念，但有 **attributes**。我们用两种方式表示关系：

**方案 A（推荐）：实体 + 属性内嵌关系**

```python
# 抽取时 entity 带 relation 属性
lx.data.Extraction(
    extraction_class="gene",
    extraction_text="TP53",
    attributes={
        "gene_symbol": "TP53",
        "associated_with_disease": "hepatocellular carcinoma",  # 关系1
        "association_type": "ASSOCIATED_WITH",
        "prognostic_in_disease": None,                           # 关系2 (无)
        "participates_in_pathway": None,
        "expresses_in_cell_type": None,
        "encodes_protein": "p53",
    }
)
```

**方案 B：实体 + 关系分两次抽取**

```python
# 第一次: 抽取实体
entities = lx.extract(text, prompt=ENTITY_PROMPT, examples=ENTITY_EXAMPLES)

# 第二次: 抽取关系
relations = lx.extract(text, prompt=RELATION_PROMPT, examples=RELATION_EXAMPLES)
```

**推荐方案 A**，因为：
- 一次 API 调用，成本减半
- 关系紧耦合在实体上，避免跨调用对齐
- LangExtract 的 attributes 正好支持这种嵌套结构

### 4.3 Few-shot 示例设计（肝病领域）

```python
import langextract as lx

KG_EXTRACTION_PROMPT = """\
Extract biomedical entities from PubMed abstracts about liver disease.
For each entity, provide its type (gene/disease/protein/pathway/metabolite/tissue/cell_type)
and any relationships to other entities mentioned in the same text.

Rules:
1. Use EXACT text from the source — do not paraphrase
2. Gene symbols should be uppercase (e.g. TP53, not tp53)
3. Disease names should be as they appear in the text
4. Mark uncertain associations with uncertain=true
5. Mark negated associations with negated=true
6. Only extract entities and relations that are directly supported by the text
7. Do NOT classify drugs, therapies, herbal extracts, or adjuvants as Metabolite/Protein
"""

LIVER_DISEASE_EXAMPLES = [
    lx.data.ExampleData(
        text="TP53 mutations are strongly associated with hepatocellular carcinoma progression. "
             "The TP53 gene encodes the p53 tumor suppressor protein.",
        extractions=[
            lx.data.Extraction(
                extraction_class="gene",
                extraction_text="TP53",
                attributes={
                    "gene_symbol": "TP53",
                    "species": "Homo sapiens",
                    "associated_with": [
                        {
                            "target_entity": "hepatocellular carcinoma",
                            "target_type": "disease",
                            "relation": "ASSOCIATED_WITH",
                            "direction": "positive",
                            "negated": False,
                            "uncertain": False,
                            "disease_stage": "HCC",
                            "evidence": "TP53 mutations are strongly associated with hepatocellular carcinoma progression.",
                        }
                    ],
                    "encodes": [{"target_entity": "p53", "target_type": "protein"}],
                }
            ),
            lx.data.Extraction(
                extraction_class="disease",
                extraction_text="hepatocellular carcinoma",
                attributes={
                    "disease_name": "hepatocellular carcinoma",
                    "disease_stage": "HCC",
                }
            ),
            lx.data.Extraction(
                extraction_class="protein",
                extraction_text="p53",
                attributes={
                    "protein_name": "p53",
                    "encoded_by": "TP53",
                }
            ),
        ]
    ),
    lx.data.ExampleData(
        text="SLC7A11 deficiency accelerated MASLD progression via ferroptosis in mice. "
             "Nrf2 activation suppressed this effect.",
        extractions=[
            lx.data.Extraction(
                extraction_class="gene",
                extraction_text="SLC7A11",
                attributes={
                    "gene_symbol": "SLC7A11",
                    "species": "Mus musculus",
                    "associated_with": [
                        {
                            "target_entity": "MASLD",
                            "target_type": "disease",
                            "relation": "ASSOCIATED_WITH",
                            "direction": "increase",
                            "negated": False,
                            "uncertain": False,
                            "disease_stage": "progression",
                            "evidence": "SLC7A11 deficiency accelerated MASLD progression via ferroptosis in mice.",
                        }
                    ],
                    "participates_in": [
                        {"target_entity": "ferroptosis", "target_type": "pathway"}
                    ],
                }
            ),
            lx.data.Extraction(
                extraction_class="disease",
                extraction_text="MASLD",
                attributes={"disease_name": "MASLD", "disease_stage": "MASLD"}
            ),
            lx.data.Extraction(
                extraction_class="pathway",
                extraction_text="ferroptosis",
                attributes={"pathway_name": "ferroptosis"}
            ),
            lx.data.Extraction(
                extraction_class="gene",
                extraction_text="Nrf2",
                attributes={
                    "gene_symbol": "Nrf2",
                    "species": "Mus musculus",
                    "associated_with": [
                        {
                            "target_entity": "MASLD",
                            "target_type": "disease",
                            "relation": "ASSOCIATED_WITH",
                            "direction": "decrease",
                            "negated": False,
                            "uncertain": True,
                            "disease_stage": "progression",
                            "evidence": "Nrf2 activation suppressed this effect.",
                        }
                    ],
                }
            ),
        ]
    ),
]
```

---

## 5. 新 Pipeline 代码结构

```
pubmed_extractor_v2.py          # 主脚本 (~400行)
├── schema/
│   ├── entity_classes.py       # 实体类型定义 + 属性 schema
│   ├── examples.py             # Few-shot 示例库 (肝病领域)
│   └── validators.py           # Stage 2 校验逻辑 (保留)
├── output/
│   ├── extraction_results_{run_id}.json
│   ├── quality_report_{run_id}.json
│   └── provenance_report_{run_id}.html  (lx.visualize)
└── entity_linking_preflight.py # 不动
```

### 主脚本核心代码

```python
#!/usr/bin/env python3
"""pubmed_extractor_v2.py — LangExtract-based PubMed extraction pipeline."""

import json
import argparse
from pathlib import Path
import langextract as lx

from schema.examples import KG_EXTRACTION_PROMPT, LIVER_DISEASE_EXAMPLES
from schema.validators import validate_extraction, normalize_entity, check_conflicts

def extract_from_pubmed(
    input_path: Path,
    limit: int = 500,
    run_id: str = "v2_default",
    model_id: str = "gemini-2.5-flash",
    skip_neo4j: bool = True,
) -> list[dict]:
    """主抽取流程。"""

    # 1. 加载 PubMed JSONL
    records = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    records = records[:limit]

    # 2. 转换为 LangExtract Document 格式
    documents = [
        lx.data.Document(
            document_id=r.get("pmid", f"rec_{i}"),
            text=f"TITLE: {r.get('title','')}\nABSTRACT: {r.get('abstract','')}",
            metadata={"pmid": r.get("pmid", ""), "source": "PubMed"},
        )
        for i, r in enumerate(records)
    ]

    # 3. LangExtract 抽取 (替代 Stage 0+1)  ← 核心变化
    results = lx.extract(
        text_or_documents=documents,
        prompt_description=KG_EXTRACTION_PROMPT,
        examples=LIVER_DISEASE_EXAMPLES,
        model_id=model_id,
        temperature=0,
        max_workers=10,
        use_schema_constraints=True,
        resolver_params={
            "enable_fuzzy_alignment": True,
            "fuzzy_alignment_threshold": 0.75,
            "accept_match_lesser": True,
        },
    )

    # 4. 后处理: KG Schema 适配 (保留 Stage 2+3+4)
    all_outputs = []
    for doc_result in results:
        entities = []
        relations = []

        for ext in doc_result.extractions:
            # LangExtract 自动提供了 char_interval
            if not ext.char_interval:
                continue  # 跳过未锚定的提取 (few-shot bleed)

            entity = {
                "mention": ext.extraction_text,
                "type": ext.extraction_class,
                "char_start": ext.char_interval.start_pos,
                "char_end": ext.char_interval.end_pos,
                "alignment_status": ext.alignment_status.name,
            }
            # 从 attributes 中提取关系
            for rel_data in ext.attributes.get("associated_with", []):
                relations.append({**rel_data, "subject": ext.extraction_text,
                                  "subject_type": ext.extraction_class,
                                  "predicate": "ASSOCIATED_WITH"})
            for rel_data in ext.attributes.get("encodes", []):
                relations.append({**rel_data, "subject": ext.extraction_text,
                                  "subject_type": ext.extraction_class,
                                  "predicate": "ENCODES"})
            # ... 其他关系类型

            entities.append(entity)

        # Stage 2 校验 (保留)
        extraction = {"entities": entities, "relations": relations}
        extraction = validate_extraction(extraction, doc_result.metadata)

        # Stage 3 标准化 (保留)
        for ent in extraction["entities"]:
            normalize_entity(ent)

        # Stage 4 冲突检测 (保留)
        extraction["relations"] = check_conflicts(extraction["relations"], {})

        all_outputs.append(extraction)

    # 5. LangExtract 内置可视化 (替代手写 HTML)
    lx.io.save_annotated_documents(results, f"extraction_output/extraction_results_{run_id}.jsonl")

    # 6. Neo4j 写入 (保留 Stage 5)
    # ... 不变

    return all_outputs
```

---

## 6. 预期改进效果

| 指标 | 当前 Pipeline | 基于 LangExtract | 提升原因 |
|---|---|---|---|
| **代码量** | 1,628 行 | ~400 行 | 框架封装 LLM 调用、分块、并行 |
| **schema_mismatch 率** | ~44% | 预计 ~25% | Gemini 受控生成 + use_schema_constraints |
| **证据锚定准确率** | ~90% (手写匹配) | ~99% (LLM 原生) | LangExtract 内置字符级对齐 |
| **假实体 (few-shot bleed)** | 无法检测 | 自动过滤 | `char_interval=None` → 自动排除 |
| **处理速度** | ~37 分钟/500篇 | ~15 分钟/500篇 | 并行 worker + 更好的 API 吞吐 |
| **可视化质量** | 手写 HTML | 交互式 lx.visualize | 点击实体 → 跳转到原文位置 |
| **模型灵活性** | 仅 DeepSeek | Gemini / GPT-4o / Ollama | 配置切换 |

---

## 7. 迁移步骤

### Phase 1: 安装 + 最小验证（1 天）

```bash
pip install langextract
export LANGEXTRACT_API_KEY="<gemini-api-key>"

# 用 5 篇文章跑原型
python pubmed_extractor_v2.py --limit 5 --run-id v2_prototype
```

验证点：
- [ ] `lx.extract()` 能正常返回
- [ ] extraction_class 映射正确 (gene/disease/protein/...)
- [ ] char_interval 不为 None (source grounding 工作)
- [ ] attributes 中包含关系数据

### Phase 2: Few-shot 示例调优（1-2 天）

- 从当前 500 篇实验结果中挑选 5-8 个高质量 PubMed 摘要
- 手工标注为 LangExtract ExampleData
- 迭代调整 extraction_class + attributes 结构
- 用 20 篇小样本验证 schema_mismatch 率下降

### Phase 3: 全量迁移（1 天）

- 移植 Stage 2 校验规则到 `schema/validators.py`
- 对接 Neo4j 写入模块
- 对接 entity_linking_preflight.py

### Phase 4: 500 篇对比实验（1 天）

- 用相同的 500 篇输入跑 LangExtract 版本
- 对比 import-ready 数量、schema_mismatch 率、处理时间
- 输出对比报告

---

## 8. 风险与注意事项

| 风险 | 缓解措施 |
|---|---|
| **Gemini API 费用** | `gemini-2.5-flash` 比 DeepSeek 贵 ~2x，但并行处理可减少总时间 |
| **Few-shot 示例质量** | 手工标注 5-8 个高质量肝病领域示例，持续迭代 |
| **attributes 嵌套复杂度** | LangExtract 对深度嵌套有限制，关系建议平铺在 attributes 顶层 |
| **中英文混合文本** | 需要 UnicodeTokenizer，当前 PubMed 摘要主要是英文，影响较小 |
| **Gemini 不可用时的回退** | 支持 `model_id="gpt-4o"` 或 `model_id="deepseek-chat"`（通过 OpenAI 兼容接口） |

---

## 9. 结论

**Google LangExtract 解决了当前 Pipeline 最大的三个痛点：**

1. **Source grounding** — 不再需要手写子串匹配，字符级偏移量自动标注
2. **Schema 约束** — Gemini 的受控生成比 prompt 指令更可靠，预计 schema_mismatch 率从 44% 降到 25%
3. **工程质量** — 代码量从 1,628 行降到 ~400 行，维护成本大幅降低

**但它不解决的：**
- 实体链接（仍需 `entity_linking_preflight.py`）
- KG 节点覆盖问题（仍需扩充 Disease/Metabolite 节点）
- 关系语义校验（如 uncertain/negated 检测，Stage 2 规则仍需保留）

最终架构：**LangExtract 做抽取 + 自定义校验层做质量控制 + entity_linking_preflight 做入库决策**。

---

## 参考

- [Google LangExtract GitHub](https://github.com/google/langextract)
- [LangExtract 官方博客](https://developers.googleblog.com/en/introducing-langextract-a-gemini-powered-information-extraction-library/)
- [LangExtract lx.extract() API](https://deepwiki.com/google/langextract/5.1-lx.extract())
- [LangExtract 医疗领域示例](https://deepwiki.com/google/langextract/6.5-domain-specific-examples)
