# AutoSchemaKG-style V0.1 — PubMed event extraction and concept induction

独立、仅文件输出的研究实验。基于 `origin/pubmed-literature-extraction` 的 `5c1e812`，新分支 `codex/autoschema-v01`。
不替换 entity/triple extraction，不导入数据库客户端，不读取或写入 Neo4j，不变更已有 schema。

## 先行审计与复用边界

| 现有组件 | 用途与本实验处理 |
|---|---|
| `convert_pubmed_xml_to_jsonl.py` | 已有 PubMed XML→`pmid/title/abstract` 转换；直接消费已有 JSONL，不重抓取或重写 |
| `cognitive_agent/extraction_kernel.py` / `agent.py` | 原实体/三元组抽取与编排；保持原样，不调用整条 pipeline |
| `cognitive_agent/aux_model_registry.py` | 复用 `AuxModelSpec`、`AuxModelRegistry.call_json`，包括 JSON 解析、用量与重试；使用现有 second LLM 配置 |
| `liverkg_cli/config.py`, `env.py`, `security.py` | 复用配置、环境文件解析和密钥获取；配置仅在 live 模式加载，密钥不序列化 |
| `cognitive_agent/evidence_units.py` | 复用 `ArticleEvidenceReader`，保留原文偏移 |
| `cognitive_agent/abbreviation_detector.py` | 复用文章内缩略语表作为模型上下文，不进行 concept normalization |
| `cognitive_agent/extraction_quality.py` | 复用 `locate_contiguous`；另外要求原始字符串切片严格相等，避免宽松归一化误映射偏移 |
| `article_preprocessing.py` | 现有并行预处理还绑定 profile、golden examples；此处只复用其底层 reader/abbreviation 组件，避免不必要耦合 |
| `evidence_selector.py`, `evidence_pack.py`, `verifier.py` | 原有实体关系约束不覆盖事件关系。只复用通用原文定位，不宣称通过原实体三元组语义验证 |
| `remote_call_broker.py`, central controller | 绑定原 agent 状态与预算，不直接复用；实验保存独立调用账本，不污染原缓存/规则记忆 |
| Python `logging` | 独立 `run.log` 和 manifest；与现有 logging 方式一致 |

### 与 AutoSchemaKG 的差异

上游的 Entity–Entity / Event–Entity / Event–Event 抽取后，对 entity/event/relation 进行概念化；本实验保留原项目实体三元组功能，另加事件句与参与实体，概念化保留原对象及其多个候选概念。
V0.1 增加 PMID、证据原文偏移、拒绝记录和人工审核表，限制在医学摘要小样本，不实现上游全套图构造/检索。
原 pipeline 使用受约束实体类型、谓词、验证与安全入库；新模块的 concepts 只是语义候选，不能据此宣布发现新的医学事实或完成 schema evolution。
不进行 concept normalization、同义合并、ontology alignment，也不把关系概念化解释为旧关系合并或新谓词注册。

参考（2026-09-29 查阅）：
- https://github.com/HKUST-KnowComp/AutoSchemaKG
- https://github.com/HKUST-KnowComp/AutoSchemaKG/blob/main/atlas_rag/llm_generator/prompt/triple_extraction_prompt.py

## 文件范围

所有新增文件均在 `autoschema_experiment/`，现有文件修改数为 0。

```
autoschema_experiment/
  run.py, llm.py, common.py       独立入口、现有客户端适配、读写和稳定 ID
  event_extraction/extract.py     事件/事件关系提示与校验
  concept_induction/induce.py     三类对象的概念生成；可独立消费已抽取对象
  evaluation/review.py            人工审核 CSV 导出
  evaluation/RESULTS.md           实测结果与限制
  data/abstracts.jsonl            3 篇现有摘要，原文保持不变
  data/provenance.json            来源文件哈希与样本选择说明
  outputs/                      每次运行的结果；经检查的 pilot_ludies_configured_v01 四份目标文件、manifest 和原始响应已提交
  tests/test_contracts.py         无网络契约测试
```

## 运行

在仓库根目录，使用 Python 3.11+。已有环境满足依赖即可，不需要安装 AutoSchemaKG、Neo4j 或训练模型。

```bash
python -m venv autoschema_experiment/.venv
autoschema_experiment/.venv/bin/pip install -r autoschema_experiment/requirements.txt
# .env 可来自原 pipeline，读取但不复制到实验或提交到 Git。
autoschema_experiment/.venv/bin/python -m autoschema_experiment.run \
  --env-file /absolute/path/to/pubmed_literature_extraction/.env \
  --output autoschema_experiment/outputs/my_run
```

默认沿用 `load_config()` 的 second_llm_model_id / second_llm_api_base（可设 `SECOND_LLM_MODEL_ID` / `SECOND_LLM_API_BASE`）与 `DEEPSEEK_API_KEY` / `SECOND_LLM_API_KEY`。如该接口不可用，可显式添加 `--model-role extraction`，改用现有 `GEMINI_MODEL`、`GEMINI_API_BASE`、`GEMINI_API_KEY`（配置读取与 API 协议仍走同一个 structured registry）。显式 `--env-file` 优先于进程继承的同名环境变量；未指定时使用现有项目环境与配置；`--model` 仅覆盖模型名，不自动切换服务商。所复用客户端以 OpenAI-compatible JSON API 工作。逻辑调用数为每篇 1 次事件 + 1 次概念；registry 最多 1 次额外重试，SDK 本身可能重试，manifest 的 attempts 不代表精确 HTTP 请求数。无固定货币预算估算。

```bash
# 仅抽事件
python -m autoschema_experiment.run --stage events --output autoschema_experiment/outputs/events_only
# 独立概念模块：可用事件模块的 concept_inputs 或现有 pipeline 对象的显式字段映射
python -m autoschema_experiment.run --stage concepts \
  --objects autoschema_experiment/outputs/my_run/concept_inputs.jsonl \
  --output autoschema_experiment/outputs/concepts_only
# 无密钥、无网络的真实响应重放，要求输入和提示哈希完全相同
python -m autoschema_experiment.run --replay autoschema_experiment/outputs/my_run/raw \
  --output /tmp/autoschema_replay_unique
python -m unittest discover -s autoschema_experiment/tests -v
```

每次指定一个**不存在**的输出目录；拒绝覆盖旧运行。失败返回非零并保留部分结果与 failures。校验拒绝写 rejected.jsonl，不能只看成功条数。
`--input` 支持现有 JSONL 的 `pmid`（数字字符串）、`title`、`abstract`，或 `text` 作为 abstract 缺失时的后备字段。重复 PMID/空摘要直接报错。V0.1 面向摘要，长全文尚未分块。

独立概念输入每行格式：
```json
{"source_id":"existing_entity_001","source_type":"Entity","source_label":"hepatic stellate cell","source_pmid":"41546802","context":"verbatim or explicitly supplied context"}
```
`source_type` 支持 `Entity` / `Event` / `Relation`，保留现有 ID。该接口不运行新的实体或三元组抽取；调用方负责把已有输出映射为这五个字段。在默认小实验中，Entity 是事件内的原文参与实体，Relation 是事件关系。

## 输出 schema（JSONL 每行一个对象）

所有事件和关系包含 `schema_version`, `source_pmid`, `source_hash`, `source_text`, `generation`, `validation_status`。
`source_hash` 是 UTF-8 JSON 序列化的 SHA256，算法见 `common.digest`；ID 是类型前缀 + 内容哈希，稳定但不做语义合并。

| 文件 | 核心字段 |
|---|---|
| `events.jsonl` | `event_id`, `sentence`, `participants:[{entity_id,mention,char_start,char_end,origin}]`, `evidence:{text,char_start,char_end}`, `assertion:asserted/uncertain/negated`, `context` |
| `event_relations.jsonl` | `relation_id`, `head_event_id`, `tail_event_id`, `relation`, `evidence:{text,char_start,char_end}` |
| `concepts.jsonl` | `concept_id`, `source_id`, `source_type`, `source_label`, `source_pmid`, `context`, `concept`, `rank`, `generation`, `validation_status`；每概念一行，每对象 2–5 个候选 |
| `manual_review.csv` | 每事件/关系/概念一行；原文、上下文、两端事件、审核证据/参与实体/方向/抽象质量字段，review_status 初始 pending |
| `concept_inputs.jsonl` | 实际概念输入，可用于独立阶段重跑 |
| `raw/*.json` | 完整提示、请求哈希、原始响应和生成元信息，不含密钥 |
| `rejected.jsonl`, `failures.jsonl` | 校验拒绝与调用/文章失败，保留原因 |
| `manifest.json` | 基线 commit、输入哈希、时间、live/replay、计数、调用用量、审核状态 |

偏移以 `source_text` 的 Python Unicode 字符位置计，0-based、左闭右开；参与实体偏移也针对全文。证据需要原文严格连续匹配。首个重复文本位置被使用，必要时人工审核指明歧义。概念是推断，不能要求其字面出现在原文，不具有 evidence-entailed 状态。

| 关系 | 方向定义 |
|---|---|
| BEFORE | head 早于 tail |
| AFTER | head 晚于 tail |
| AT_THE_SAME_TIME | head 与 tail 明确同时发生 |
| BECAUSE | head 是结果，tail 是原因 |
| AS_A_RESULT | head 是原因，tail 是结果 |

每条关系只引用同一 PMID 已接纳的事件；禁止自环和悬空引用。未检测复杂时间环或因果矛盾；允许无事件关系的摘要，不为凑齐标签强行生成。

## 评估边界

人工检查完整事件句、否定/不确定性、物种/模型、参与实体，关系因果强度与方向，以及概念是否有意义且确属更高抽象层级。结构/原文校验不能证明语义蕴含。`source_grounded_pending_review` 明确不是 `verified` 或 `import_ready`。
此前接口曾返回 HTTP 401；更新本机密钥并修复显式 env 文件优先级后，3 篇在线小实验成功生成候选。详见 `evaluation/RESULTS.md`。
3 篇是目的性选择的开发样本，没有冻结 gold、没有精确率/召回率或医学效能结论。初始 CSV 不预填人工审核结果；raw 和 rejected 保留失败，不能只报告漂亮例子。
