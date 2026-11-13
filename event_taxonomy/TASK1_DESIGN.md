# Event Taxonomy Induction 数据准备与实验设计

检查日期：2026-11-12。此文件保留研究设计；用户随后授权开始 Task 1，离线数据准备、模型、适配器及初始 DAG 已落地，实际入口和范围见 [README.md](README.md)。新的递归归纳算法仍未实现或运行。

研究问题：How can we automatically induce coherent intermediate concept hierarchies between predefined high-level categories and fine-grained event concepts?

结论：已有事件抽取、真实概念化结果、文本规范化、三个高层类别及一个入位小试，但分散在不同项目中。新模块应从这些保存的数据接入，补齐候选概念审核、稳定数据契约及中间层递归归纳。不能把“已有 Concept 标签”视为“已有细粒度概念体系”，也不能把同名去重视为层级归纳。

设计交付后新增了独立的数据准备代码、运行配置、测试与真实数据导出；不修改抽取器、旧实验、网站或浏览器数据，不调用模型。

## 1 检查范围与定位

| 代号 | 实际路径 | 用途 |
|---|---|---|
| A | `<AutoSchemaKG 工作区>` | 当前仓库，上游提交 `d0a1666ae6621806faf814504c73678de4e809f2`，含 `liver_baseline` |
| W | `<生物医学工作台项目>` | 本地工作台、全流程批次和入位小试所在项目 |
| V | `<三类定义验证结果>` | 三类上层定义与覆盖检查 |

端口 4317 的监听进程工作目录为 `W/outputs/biomedical-workbench`。当前浏览器连接超时，直接 HTTP 读取也失败，因此网页功能由前后端代码核实；已运行情况由磁盘上的结果和 API 响应记录核实。未读取浏览器 localStorage，不能断言当前浏览器有没有点击过“生成 Concept”。

### 1.1 当前仓库的抽取输出

| 来源 | 实际结构 | 来源与证据 | 接入限制 |
|---|---|---|---|
| 官方三阶段批次 | 每行 `id, original_text, metadata, entity_relation_dict, event_entity_dict, event_relation_dict`；另有原始输出和 usage | PMID 在 `metadata.pmid`，文档 ID 在 `id` | 文件后缀虽是 `.json`，实际为 JSONL；结果未保存 `chunk_id` |
| 原版 Stage 2 | `Event` 和 `Entity[]` | 只能回到该篇摘要 | 无事件 ID、Trigger、逐事件 Evidence；事件句子不能冒充原文句子 |
| 角色／嵌套 Stage 2 | 文档外层 `pmid,text,events,structural_errors`；事件含 `Event_ID,Event,Trigger,Evidence,Arguments` | Evidence 可通过原文定位；Arguments 有 Role/Type/Value | E1 等仅在文档／运行内唯一；结构通过不等于语义正确 |
| Stage 3 | `Head,Relation,Tail`，端点是独立生成的事件文本 | 摘要级来源 | 尚无可靠的 Stage 2/3 语义对齐，不将其端点自动当作同一事件 |

核实的真实开发数据：

| 保存的运行 | 文档数 | Stage 2 事件数 | 说明 |
|---|---:|---:|---|
| `A/liver_baseline/outputs/20261109T142922.491307Z-official-batch5-live` | 5 | 45 | 官方三阶段执行器；另有 95 条实体关系和 23 条事件关系 |
| `A/liver_baseline/outputs/20261109T151732.781701Z-heldout-stage2-suite-live` 的 original | 10 | 117 | 原版 Stage 2；现已参与开发，不作为未见测试集 |
| `A/liver_baseline/outputs/20261110T090410.447692Z-role-nested-dev-live` | 10 | 97 | 6 篇通过现有文档级结构检查，这 6 篇共 35 条事件；其余原始记录仍在 |

97 与 117 是同批文献的不同抽取结果，不能相加作为独立样本。35 只描述旧严格导出覆盖量，不是新模块应采用的自动删减规则。

### 1.2 概念化确实已经存在

三条路径必须分别标记，不能混作同一个方法。

| 路径 | 实现与输出 | 现状 |
|---|---|---|
| 上游 AutoSchemaKG | `atlas_rag/kg_construction/concept_generation.py`，CSV 为 `node,conceptualized_node,node_type` | 通用实现存在，非肝病示例有产物；A 的肝病批次没有运行概念化 |
| 网页“生成 Concept” | `W/outputs/biomedical-workbench/public/event-concepts.js`、`public/app.js`、`server.mjs` 的 `POST /api/conceptualize-events` | 功能完整实现；只取当前运行未排除事件，相同事件文本共享请求；结果存浏览器并进入 JSON/CSV/工作区导出 |
| 工作台全流程批次 | `W/work/full-pipeline/run.mjs`、`graph.mjs`；输出 `source_id,types[],topics[],status` | 已真实运行；原始调用和解析产物在 `W/outputs/AutoSchema_10篇全流程`，网站有报告副本 |

网页按钮保存结构为：`eventConcepts = {model,promptVersion,createdAt,results:[{eventId,event,labels:[]}]}`。事件修改后标签标为 stale，导出单事件不再附带旧 `Concepts`；但运行级原记录仍可保留，适配器必须重新检查快照是否一致。请求一批最多四条，每条事件各调用一次模型，设置为 temperature 0、max_tokens 512。该路径没有为每个概念保存 definition、type/topic 分类、父节点、token usage 或完整模型响应。

上游和按钮采用“多个不同抽象层次短语”的 Prompt，允许类型及相关主题，因而输出不是固定粒度的叶概念。全流程批次使用本地 adaptation 的 types/topics Prompt，已将二者分开；不是上述按钮的相同 Prompt。

### 1.3 全流程批次的可复用资产

直接检查 `W/outputs/biomedical-workbench/public/pipeline-results/graph.json` 得到：

| 对象 | 数量 | 含义 |
|---|---:|---|
| 文献 | 10 | 已参与开发 |
| Event 节点 | 101 | 工作台 Baseline 运行生成，区别于 A 的官方原版 117 条 |
| Event 类型概念 | 76 | `concept_domain=Event, concept_role=types`，文本规范化后的不同标签 |
| Event 主题概念 | 176 | `concept_domain=Event, concept_role=topics`，不可直接用作事件分类父子节点 |
| Event→类型关联 | 101 | `HAS_TYPE`，仍是待复核的候选关联 |
| Event→主题关联 | 266 | `RELATED_TOPIC` |
| 全部 Concept 节点 | 798 | 混合 Entity/Event/Relation 类型与主题，不能全算作 Event Concepts |
| Concept→Concept 边 | 0 | 该批次尚未构建概念层级 |

101 个 Event 中，96 个的 evidence_status 为 `verified_exact`，5 个为 `unverified`。前者只表示字符串定位成功，不表示论断或归类正确。图仍标记 `graph_ready=false`、`semantic_review_completed=false`。

概念节点已有 ID、label、normalized_label、domain、role、source_assignments；Event 节点已有运行 ID、PMID、证据、字符位置、局部 ID。概念边明确保留 `evidence_entails_concept=false`，不能将原文可定位解释为上位关系已验证。

该批次有 10 个事件概念文件和对应 API 成功响应缓存，可离线导入，无需重新花费模型调用。静态 `workspace-runs.json` 内没有按钮路径的 `eventConcepts` 字段；已有概念实际在 `graph.json` 和 `concepts/*.json` 中，不能只读 workspace-runs 就判断没有生成概念。

现有类型包括 `physiological process`、`diagnostic assessment`、`clinical observation` 等，粒度不一且部分较宽。76 是已有规范化标签数，不是已确认的 76 个细粒度叶概念。还须检查“关联／比较陈述”与真实事件概念的边界，尤其不能把 `statistical association` 自动转成生物学因果过程。

### 1.4 Normalization 与 Deduplication

| 实现 | 已做 | 未做 |
|---|---|---|
| 上游 | 逗号切分、trim/lowercase、set 同名去重；label 哈希作为 Concept ID | 无定义驱动的消歧、同义审核、上下位判断或递归组织 |
| 网页按钮 | NFKC、空白规范化、忽略大小写比较；去掉重复标签及与事件原句相同的标签 | 主要是单次结果清理，无全局 Concept Registry |
| 全流程 `graph.mjs` | 按 `(source_type, types/topics, normalized_label)` 建共享 Concept；保留每次来源分配 | 明确 `semantic_synonym_merging=false`，同名可能异义，近义标签没有语义合并 |

新模块应保留旧 ID、原标签及规范化策略版本。不得用新的大小写处理把有语义差异的生物医学名称静默合并。对已有归并碰撞，保留来源并进入审阅队列；不能宣称已有“Unique Concept”已完成语义消歧。

### 1.5 初始 Taxonomy 与入位小试

确有三类上层及定义：

| ID | 上层名称 | 核心边界 |
|---|---|---|
| BIO | Biological Event | 活体系统中的分子、细胞、生理和病理过程；被研究不自动成为科研行动 |
| CARE | Clinical Care Event | 面向个体的诊断、治疗、康复或预防行动 |
| RESEARCH | Biomedical Research Event | 明确为了产生、检验或分析知识的科研行动 |

完整定义取自 `W/outputs/Event_Ontology_V0.1_入位小试_20261112/protocol.json` 的 seed，与 V 的 `results/ontology_summary.json` 对应。已单独快照到 `configs/roots.v0_1.json`。CARE 与 RESEARCH 允许在定义确实满足时交叉归属；不能因“论文描述了它”就给所有概念加 RESEARCH 父节点。

既往 94.4% 的结果是该验证包中 135/143 个 event-like mentions 被覆盖，另外 24 条 assertion-only 候选不在分母；不是当前 101 条事件的覆盖率，也不是独立专家验证。本实验应重新报告自己的分母与边界案例。

`W/work/event-ontology-placement-pilot.mjs` 已提供 REUSE、ADD_LEAF、INSERT_BETWEEN、DEFER，检查环路、根不变及旧祖先关系保留。它是给定候选逐个入位的小试，不会从大量细粒度子概念归纳新中间概念。

五个小试包含人为准备标签与一个 `constructed_test`。预期 INSERT_BETWEEN 的合成案例实际返回 REUSE，因此不能声称插层能力已实证成功。它们只用于开发检查；新实验不导入其额外中层 `DRUG_RESPONSE`、`SENS_UP` 或结果作为归纳答案。

## 2 独立模块与数据流

新增模块位于 `A/event_taxonomy/`，与 `atlas_rag/` 和 `liver_baseline/` 并列。通过只读文件适配器接入，不让新模块调用旧 runner、修改旧 frozen 文件或写回网站。

流程分为：来源快照与审计 → Event/Evidence 适配 → Concept Candidate 导入或生成 → 事件类型与粒度审核 → 保守规范化及 Unique Concept 注册 → 冻结输入集合 U0 → 根节点暂挂 → 中间层递归归纳 → DAG 验证与评测。

先使用全流程已保存的 Event 类型候选做离线数据准备。网页导出是另一种可选输入。原版 117 条事件用于单独的适配与对照，不能与工作台 101 条混成同一个抽取方法。已有事件概念较宽时，后续在独立 conceptualization 阶段生成细粒度候选，保存模型／Prompt／证据，不修改 Event Extraction。

主题标签单独保留；若从主题生成事件类，必须通过新的概念化与审核记录，不能仅把字段从 topics 改成 types。

以下为完整流程接口契约。导入、规范化、初始图与结构验证已提供离线实现；conceptualize、自动 assign_roots 和 induce_intermediate 仍为后续接口，实际函数签名见对应模块：

| 接口 | 输入 | 输出与责任 |
|---|---|---|
| `load_events(source, profile)` | 官方 JSONL、role_nested、工作台导出或 graph | Document、Event、Evidence、ImportIssue；检查重复、来源版本和局部 ID |
| `load_candidates(source, events)` | 工作台 types/topics、按钮 labels 或上游 CSV | 逐次 ConceptCandidate 与来源映射；无法匹配的项进入问题清单 |
| `conceptualize(events, provider, policy)` | 已登记事件、证据与粒度策略 | 新候选及调用记录；可用已保存候选 provider，默认无在线调用 |
| `normalize_candidates(candidates, policy)` | 候选及审核决定 | UniqueConcept、candidate→unique 映射、冲突记录；不做相似概念压缩 |
| `assign_roots(concepts, roots)` | 概念定义与根定义 | 候选归属、依据与 abstention；不强制选一个根 |
| `build_initial_taxonomy(concepts, roots, assignments)` | 冻结 U0 与暂挂结果 | 初始图、临时边、未归属列表、输入哈希 |
| `induce_intermediate(taxonomy, provider, config)` | 初始图及策略 | 新父概念提案和图修改日志；本阶段仅设计，不实现 |
| `validate_taxonomy(taxonomy, frozen_input)` | DAG 与冻结契约 | 结构报告、概念保留检查、来源链检查 |

CSV 的 `node` 是文本而不是可靠的事件 ID。须在指定 source run 中精确匹配并保留全部匹配项；歧义不猜。未来长文分块时，应通过原始输入及 request_plan 恢复 chunk 身份；无法恢复时用源文件哈希和行号保留身份，offset 标记未知，禁止默认所有 chunk 都是 0。

## 3 数据模型契约

持久化优先 JSON/JSONL，Python dataclass 配合 JSON Schema 作后续实现选择。所有对象含 `schema_version`，所有来源保留文件 SHA-256 和记录定位。不要通过导入大型模型模块来读取 JSON。

### 3.1 Document 与 Evidence

| 模型 | 必要字段 | 规则 |
|---|---|---|
| Document | `document_id,source_pmid,title,text,text_sha256,data_origin,source_ref` | PMID 为字符串，可为空；不能从任意数字型文件名猜 PMID。版本 ID 由来源命名空间、PMID/原始 ID、文本哈希构成 |
| Evidence | `evidence_id,document_id,text,spans,granularity,alignment_status,locator_version,source_ref` | spans 使用 Unicode code point 半开区间 `[start,end)`；保留一段或多段证据 |
| Evidence 状态 | `verified_exact / verified_whitespace / ambiguous / unverified / passage_only / missing` | 原句多次出现时记录所有位置或 ambiguous；无证据句可保留 passage 来源，但不能伪装为 sentence |
| SourceRef | `source_profile,source_run_id,file_sha256,record_locator,original_id` | 原始相对文件路径与数据集根另存 manifest；跨机器可配置 |

`data_origin` 使用 `real_literature / synthetic / mixed`。手工创建的合成例子即使模仿真实 PMID 格式，也必须 source_pmid=null，并标记 synthetic。真实事件配人为准备概念时分别标记事件来源与候选来源，不笼统标为全部真实。

### 3.2 Event Instance

| 字段 | 类型与含义 |
|---|---|
| `event_id` | 全局稳定字符串；由 source namespace/run、document revision、stage、chunk/record locator、local ID、prediction index 构成；不能只哈希事件文本 |
| `event_text` | 当前导入快照的完整事件文本 |
| `source_pmid,document_id` | 指向来源，冗余 PMID 必须与 Document 一致 |
| `evidence_sentence` | 主证据原句，可为 null；由 Evidence 派生而非另造文本 |
| `evidence_ids` | 所有证据片段 ID，支持跨句 |
| `local_event_id,source_run_id,source_profile,source_ref` | 追溯旧格式及原始输出位置 |
| `trigger,arguments,assertion,context` | 可选原始抽取信息；未知断言填 unknown，不能默认 asserted |
| `review_status,validation_issues,data_origin` | 人工状态与机械校验分开保存；排除项保留记录 |

跨阶段近义事件、同文重复抽取和不同模型结果不自动合并。将来需要对齐时用独立 `event_alignment` 表，保留各自 event_id。否定或假设事件归类指文本中被提及的事件类型，不代表它实际发生。

### 3.3 Concept Candidate

一条候选是一次生成／人工提出的概念，不等于全局唯一概念。

| 字段 | 含义 |
|---|---|
| `candidate_id,candidate_label,candidate_description` | 每次输出独立标识；已有标签无定义时 description=null，另记 `description_status=missing` |
| `supporting_event_ids,evidence_ids` | 直接来源，不使用主题相关性替代实例依据 |
| `candidate_role` | `event_type / topic / unknown`；旧按钮及上游 CSV 默认 unknown |
| `granularity_status` | `fine_grained / broad / mixed / unresolved`，需规则和审阅，不凭词数断定 |
| `generation` | provider、model_requested、model_returned、Prompt 哈希、参数、时间、调用 ID；旧数据缺失时为 null |
| `validation_status,validation_reasons` | proposed/accepted/deferred/rejected；任何拒绝或改判均可追溯 |
| `provisional_parent_ids` | 初步根／父概念候选，默认空数组，并关联独立归属依据 |
| `data_origin,source_ref` | existing_model_output、new_model_output、human_prepared 或 synthetic 等来源标记 |

按钮的标签列表拆成多条候选，保留原标签顺序与 eventId；全流程 `types/topics` 按不同 role 拆开。`labels=[]` 是显式无标签，缺失记录是未生成或失败，二者不能合并成一个状态。

### 3.4 Unique Concept

| 字段 | 含义 |
|---|---|
| `concept_id,concept_label,concept_description` | Registry 分配稳定 ID，label 修改不改 ID；保留 upstream_concept_ids |
| `candidate_ids,aliases,normalization_key,normalization_version` | 所有输入候选及归并记录；同名不同定义可分开，通过 disambiguation_key 区分 |
| `supporting_event_ids,evidence` | 去重后的 event_id 和 Evidence 引用；多篇文献支持同一概念必须全部保留 |
| `provisional_parent_ids` | 当前暂挂候选；最终父子关系以 edge 表为权威，不靠此数组替代 DAG |
| `origin_kind,validation_status,granularity_status` | `predefined_root / input_concept / induced_intermediate` 与质量状态分开 |
| `description_provenance,review_history` | 定义是人工、模型生成还是旧来源缺失，不伪装成原文事实 |

规范化只处理有记录的表示重复；“肝星状细胞激活”与“肝细胞激活”等相似但不同概念绝不为减少分叉而合并。必要的同义／同名裁定应在 U0 冻结前完成，并单独报告，不计为归纳算法收益。U0 冻结后 ID、含义和证据映射不可被算法删除或改写。

### 3.5 Taxonomy 与边

Taxonomy 包含 `taxonomy_id,version,nodes,parent_child_edges,predefined_roots,concept_evidence_mapping,instance_concept_links,unplaced_concept_ids,frozen_input_concept_ids,manifest_ref,status`。

| 对象 | 字段与语义 |
|---|---|
| Taxonomy node | 使用 Unique Concept，明确 root/input/intermediate 来源；不把 Event Instance 混进概念子图 |
| ParentChildEdge | `edge_id,parent_id,child_id,relation,status,decision_ref,created_in_run`；统一 parent→child，`relation=SUBSUMES`，即 child IS_A parent |
| RootAssignment | `concept_id,parent_id,status,reason,validator,definition_versions`；provisional 与 accepted 分开 |
| InstanceConceptLink | `event_id,concept_id,relation=IS_INSTANCE_OF,status,evidence_ids,decision_ref`；旧 HAS_TYPE 先以 proposed 导入，审核后再确认 |
| ConceptEvidenceMapping | `concept_id,event_ids,evidence_ids,mapping_kind,via_child_ids`；direct 与 derived_from_descendants 分开 |
| EdgeDecision | 提案、父子定义、支持／反对理由、检查方法、拒绝原因及模型/人工来源；概念证据本身不是父子包含关系的证明 |

中间概念的证据可由后代汇总，但须标记 derived；不得写成原文直接出现了这个抽象类。多父 DAG 汇总按 event_id/evidence_id 去重，不能把同一事件沿不同路径计数多次。

## 4 初始 Taxonomy 与递归算法边界

本次主实验使用三个已有根，固定名称、定义及 ID。入位小试中额外准备的药物反应分支不作为主实验 seed，避免人为提供算法要归纳的中间答案。

1. 导入候选，审查是否能表示事件类型、是否有定义和可靠支持，并记录粒度；不符合条件的输入仍在候选档案和分母中。
2. 规范化后冻结 U0；区分保留的 input concept 与后续生成的 intermediate。输入概念不必永远是拓扑叶子，但必须作为独立概念保留。
3. 依据定义生成零个、一个或多个根归属。初始直接连接可为 provisional；无法归属者留在 unplaced，不能强塞到最相似根或捏造 Other 作为语义类。
4. 形成 `initial_taxonomy.json`，所有 U0 概念均在 nodes；未验证／未归属项导致 status=partial。语义指标仅计算 accepted 边，并同时报告未决覆盖。
5. 后续算法在某父节点下提出相关子集和可解释的中间类；先确认中间类比父节点窄、比每个拟收纳子节点宽，再作为一次图修改提交。
6. 只重连通过检查的子节点；其余保持原位。没有有效提案时允许宽分叉，不能为了树形好看造空泛中间层。

递归原型的运行顺序见 [Task 2 实现](TASK2_IMPLEMENTATION.md)。以下分流方案用于后续设计。

### 4.1 按直接子类数选择归纳流程

以当前父类的直接子类数为准，暂用 20 作为可调试验值。少于 2 个直接子类时跳过当前层的归纳。有 2 至 20 个且输入在 token 预算内时，让 A 直接查看子类的定义和证据样本，判断是否有值得归纳的子群。超过数量阈值或请求超出 token 预算时，对同一父类下的直接子类采用 complete linkage（完全链接法）生成候选组，再让 A 判断哪些组适合归纳。每次提交后，按新的直接子类重新选择流程。

两条流程都把值得归纳的候选组交给 B 命名和写定义。C 检查新类是否属于原父类，以及组中每个孩子是否属于新类。通过检查后提交图变更。

在有初始归属的开发集上，选择不同子类数和输入长度的真实父类，固定模型、输入和证据采样，成对比较直接让 A 判断与先聚类两种流程。人工检查新增上下位边的正确性和分组遗漏，记录 token 与耗时。在切换点附近补测，再冻结数量阈值和 token 预算。

#### 聚类向量与相似度试验

首轮建议把概念名称和一句审定的类型定义拼在一起生成向量。原文证据保留给 A 和 C 核对。已提交的中间类也可用自己的名称和定义生成向量，参与下一轮横向分组。

从现有的余弦相似度试验值 0.82 起步，对应余弦距离上限 0.18。完全链接法要求组内相似度最低的一对也达到阈值。提高阈值会收紧分组。这个阈值控制组员的相近程度，前面的 20 控制何时启用聚类。

固定向量模型和文本格式，在同一批已挂接的开发分支上比较 0.75、0.80、0.82、0.85、0.90。人工核对代表性的正确分组、错分和漏分，再检查候选中间类及其上下位边。记录向量模型版本、文本格式、阈值和分组结果，选择后冻结供独立评测使用。

### 4.2 先横向优化，再纵向递归

**横向**处理同一个父类。插入中间类后，重新查看它现在的直接子类，包括新增中间类和还直接挂在它下面的原有子类。如果其中还有值得归纳的组，就在这个父类再走一轮分流、分组、生成与验证。每轮都按更新后的子类数和 token 预算选择流程。数量阈值决定是否先聚类。

**纵向**依次进入当前父类的各个直接子类。当前层没有值得归纳的组，或达到本层预算时，逐个把这些子类作为新的父类，用同样的顺序处理。进入新增中间类时，再看它自己的直接子类。

合成演示在 `Biological Event` 下接受 `Gene activation` 和 `Gene inhibition` 后，先在同一个父类把它们归到 `Gene regulation`，并为原位的细胞增殖事件插入 `Cell proliferation`。这一层结束后，才分别进入 `Cell proliferation` 和 `Gene regulation`。

DAG 硬约束：

- 加边 p→c 前，p≠c，两个节点存在，且 c 不能已到达 p；批量修改先在副本检查，整批成功后提交。
- 每次提交及完整导出均检查无环、引用完整、根无入边、U0 节点及来源映射保持不变。
- 多父分别需要包含关系依据；“部分重叠”“有关联”“共同出现”不够。
- 插入中层后保留已确认的祖先可达性；原直接边若删除须有等价路径，并记录变更理由。provisional 边的纠正另记审核过程，不要求保留错误祖先。
- 不固定所有叶子的深度，不以最大分叉数作为语义正确性的充分条件。递归预算限制只控制成本，预算耗尽保留部分结果和 stop reason。
- 只有全部要求归属的输入概念得到有效根路径，才能标记 complete；unplaced 不得从覆盖分母消失。

## 5 拟定目录

下表保留完整流程的模块职责规划；当前已实现的数据准备目录和文件见 README。induction 与语义 evaluation 部分尚未实现。

| 路径（相对 event_taxonomy） | 责任 |
|---|---|
| `models.py`, `schemas/` | 数据类型与序列化契约 |
| `adapters/official.py` | A 的官方 JSONL、单篇及 original 文档输出 |
| `adapters/role_nested.py` | 角色／嵌套输出和问题保留 |
| `adapters/workbench.py` | graph、workspace、单次标注导出及 eventConcepts |
| `preparation/evidence.py`, `preparation/snapshots.py` | 原文定位、哈希、不可变输入和 split |
| `concepts/providers.py`, `concepts/validation.py`, `concepts/registry.py` | 已有候选优先、类型/粒度验证、表示规范化 |
| `taxonomy/initialization.py`, `taxonomy/dag.py` | 根归属、初始图、事务与环检测 |
| `induction/proposals.py`, `induction/verification.py`, `induction/recursive.py` | 未来中间概念归纳，后续阶段实现 |
| `evaluation/structural.py`, `evaluation/semantic.py`, `evaluation/reference.py` | 结构、人工审核及参考层级对齐 |
| `configs/`, `prompts/`, `cli.py` | 版本化配置、独立 Prompt、默认离线入口 |
| `tests/fixtures/synthetic/`, `tests/` | 明确 Synthetic Data 的边界与接口测试 |
| `data/snapshots/`, `runs/<run_id>/` | 独立输入快照、manifest、结果和 change log；禁止写到旧 outputs |

基础层仅用标准库及已有 NetworkX/JSON Schema；模型 provider 延迟加载。外部 JS 函数适合参考或通过固定 JSON 契约交换，不把相邻项目的绝对 Python import 路径做成运行时依赖。

可复用重点：工作台 `graph.mjs` 的来源 ID、证据定位与 types/topics 格式；`event-concepts.js` 的 stale 和导入检查；旧入位小试的 DAG 约束；`liver_baseline` 的 hash/manifest 设计。旧 `source_span` 只取首个匹配，需增强重复句歧义处理；旧官方 CSV/DiGraph 导出只在节点首次出现时写 Source，不能作为新模块的唯一来源账本。

上游 `EvaluateKGC/SchemaQuality/eval.py` 主要比较概念标签与参考类型的 BERTScore，不评估中间层 DAG，不能直接作为新算法的主要评测器。

## 6 实验配置与复现

配置草案在 `configs/task1.design.json`，状态为 design_only，API 调用和 induction 均禁用。路径通过数据源根解析；inspection 清单记录本机确切文件及 SHA-256。未来运行必须先解析并验证路径，冻结选择规则、定义、Prompt 和数据版本。

分开配置：source_profile、抽取版本、conceptualization provider、concept role、粒度政策、规范化版本、根版本、未决处理、DAG 约束、算法预算、随机种子、评测分组和模型记录。不得用一个 model 字段替代所有阶段各自的模型快照。

主试点建议先用 workbench batch 已保存概念做接口和质量审计；“原始宽粒度类型集”和“后续独立生成的细粒度集合”作为两个数据配置。只在相同 U0 上比较算法。若尚无足够的有效细粒度输入，只报告数据准备结果，不报告中间层归纳优劣。

所有运行保存：配置、输入哈希、代码版本与新模块源文件哈希、模型请求/返回名称、参数、Prompt 哈希、随机种子、重试、失败、token/耗时、候选/去重/入图数量、拒绝原因及图修改记录。旧记录缺失的 usage/响应不能补造；同名服务别名不等价于固定权重快照。

## 7 实验设计与实施顺序

### 7.1 后续比较设计

比较的对象是中间层构建方法，固定 Event 集、U0、根定义、候选质量政策与模型配置。

| 实验臂 | 定义 | 目的 |
|---|---|---|
| Flat | 根直接挂接相同 U0，无中间概念 | 初始结构基线，不称为当前官方抽取 Baseline |
| One-pass | 只执行一轮同等提案与验证 | 分离递归带来的作用 |
| Recursive | 多轮提案、包含关系验证和 DAG 更新 | 目标方法 |

旧 sequential placement 可作为后续额外比较，但需先统一输入、根、预算及验证政策。不能将五个手工案例与批量新算法直接比较准确率。质量与成本一起报告，并区分相同预算和各方法自然运行成本。

已有 10 篇文献、5 篇批次以及上层验证数据均作为已暴露开发资源。正式测试需后续按 PMID/文档来源分组冻结新的独立集合，重复摘要不跨 split；阈值与 Prompt 只在开发集确定。完整测试概念批次共同参与建图是 transductive hierarchy induction，不能据此声称新概念增量泛化；增量实验另设。

评测至少包含：

- **硬约束**：环路=0，悬空引用=0，根变动=0；U0 节点保留率与原始事件支持映射保留率应为 100%。发现删除即判该运行不符合研究设定。
- **覆盖与结构诊断**：根可达率、unplaced、accepted/provisional 边数、中间节点数、深度和分叉分布。更深、更平衡不是自动更好。
- **语义审核**：新父子边的严格包含正确性、中间概念定义是否清晰、兄弟概念是否围绕同一可解释维度；按根、深度和多父场景抽样，并报告样本量、未决率与审核者一致性。
- **参考层级**：后续建立经人工审核的小型 DAG，允许合理多父和多种组织方式；先冻结节点对齐，再计算父子边及祖先可达关系 P/R/F1。若参考仅部分标注，只在已判定集合内评分，不能把未标注关系全当负例。
- **成本与稳定性**：调用数、token、时间、失败率、不同种子／重复生成的差异。LLM 评分作为辅助，不能称独立专家 Gold。

若采用“删除参考 DAG 中间节点再重建”的恢复实验，应只给算法根和保留的细粒度定义，隐藏被删节点、标签及边。若先用参考层级决定粒度或调整算法，应标注开发使用，不能再作为未见测试答案。

### 7.2 分阶段交付与验收

| 阶段 | 工作 | 验收 |
|---|---|---|
| Task 1 当前 | 检查实际项目、数据契约、根快照、配置与后续步骤 | 事实有文件依据，区分已实现与拟实现，无旧数据改动 |
| Task 2 接入 | 实现 Document/Event/Evidence 和三类适配器，导入真实保存结果 | 每条记录可回到 source run/PMID/文件定位；缺失或歧义显式记录 |
| Task 3 候选准备 | 导入现有 types/topics，检查粒度与类型；必要时独立细粒度概念化；规范化并冻结 U0 | 候选→Unique→Event→Evidence 链完整，原始数据及拒绝项保留 |
| Task 4 初始图 | 使用三根定义做暂挂、支持多父和 abstention，输出初始 DAG | 节点保留、无环、根不变；临时与确认边分开 |
| Task 5 新算法 | 递归提出和验证中间类，事务性插层，记录停止原因 | 不删除或合并 U0 概念；能重放每次结构修改 |
| Task 6 评测 | 冻结参考/测试数据，运行 Flat/One-pass/Recursive 和人工抽样 | 同输入对照、无测试调参、分清结构诊断和语义效果 |

测试优先覆盖稳定 ID、重复 PMID/不同运行、Evidence 缺失／多位置／Unicode 偏移、stale 标签、type/topic 分离、规范化碰撞、多父菱形、成环提案、批量更新回滚、无法归属和预算停止。合成 fixture 全部使用 `SYN-` ID 与 `data_origin=synthetic`，只用于程序验证，绝不进入真实效果表。

## 8 当前交付边界

已查明的网站功能与保存结果可以复用；正式三类根也已找到，不需要重新猜测。未确认的是浏览器内是否另有最新点击生成的记录，以及现有候选中有多少满足本实验的细粒度要求。

当前没有新生成 Concept、没有新建语义分类边、没有声称算法有效。离线适配已经完成，输入按内容冻结但未声称完成语义 U0 冻结；后续从概念定义和粒度审核继续，无需先重新运行 Event Extraction。
