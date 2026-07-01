# 自主认知Agent：基于 LangExtract + Neo4j 的肝病知识管理智能体

## 从信息抽取工具到科研知识管理智能体

> 日期：2026-06-28
> 版本：v1.0 — Architecture Design
> 模型：DeepSeek-V3 (via LangExtract OpenAI-compatible path)

---

## 0. 范式跃迁：Tool → Agent

```
┌──────────────────────────────────────────────────────────────┐
│                    范式对比                                    │
├──────────────────────┬───────────────────────────────────────┤
│  当前 Pipeline (工具) │  Cognitive Agent (智能体)             │
├──────────────────────┼───────────────────────────────────────┤
│  固定 5 阶段流水线    │  动态闭环推理循环                      │
│  单向数据流           │  双向交互：提取 ⇄ 图谱 ⇄ 反思         │
│  被动执行预设规则     │  主动发起验证、推理、决策              │
│  输出 = 三元组列表   │  输出 = 知识状态变更 + 置信度 + 溯源   │
│  错误 → 人工发现     │  错误 → 自我检测 → 自我修正            │
│  每篇独立处理         │  跨文档知识累积 + 策略自适应           │
│  Schema 硬编码        │  Schema 动态演化（从图谱反馈中学习）   │
│  只写不读 Neo4j       │  Neo4j = 外部记忆体的读写循环          │
└──────────────────────┴───────────────────────────────────────┘
```

### 0.1 智能体的"科研思维"

这个 Agent 模拟的是**科学家阅读文献时的认知过程**：

1. **先验知识激活**：读摘要前，先从 Neo4j 检索已知的基因-疾病关联
2. **主动提取**：带着"这个基因和肝病什么关系？"的问题去提取
3. **事实核查**：提取出的实体在知识库中存在吗？关系是否与已有证据矛盾？
4. **因果推理**：A → B 和 B → C 能否推出 A → C？
5. **决策判断**：新发现是补充还是推翻旧知识？应该直接写入还是标记为待验证？
6. **反思调整**：如果提取质量差，是不是摘要类型不适合当前策略？

---

## 1. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Autonomous Cognitive Agent                        │
│                   for Liver Disease Knowledge Graph                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   PubMed Abstract                                                    │
│       │                                                              │
│       ▼                                                              │
│   ┌───────────────────────────────────────────────────────────┐     │
│   │               COGNITIVE REASONING LOOP                     │     │
│   │  ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌─────────┐  │     │
│   │  │PHASE 1  │──▶│PHASE 2  │──▶│PHASE 3   │──▶│PHASE 4  │  │     │
│   │  │Context  │   │Extract  │   │Verify &  │   │Decide & │  │     │
│   │  │Activate │   │+Ground  │   │Reason    │   │Execute  │  │     │
│   │  └─────────┘   └─────────┘   └──────────┘   └─────────┘  │     │
│   │       ▲              │              │              │       │     │
│   │       │              │              │              ▼       │     │
│   │       │    ┌─────────┴──────────────┴──────┬──────────────│     │
│   │       │    │                               │              │     │
│   │       │    ▼                               ▼              │     │
│   │  ┌─────────┐                        ┌──────────┐         │     │
│   │  │PHASE 6  │◀───────────────────────│PHASE 5   │         │     │
│   │  │Reflect &│                        │Conflict  │         │     │
│   │  │Adapt    │                        │Resolve   │         │     │
│   │  └─────────┘                        └──────────┘         │     │
│   └───────────────────────────────────────────────────────────┘     │
│       │         │                │                                   │
│       │         ▼                ▼                                   │
│       │  ┌──────────┐    ┌──────────────┐                           │
│       │  │LangExtract│    │  Neo4j KG    │                           │
│       │  │+ DeepSeek │    │  (Memory)    │                           │
│       │  │(Kernel)   │    │              │                           │
│       │  └──────────┘    └──────────────┘                           │
│       │                                                              │
│       └──── Strategy Adaptation (调整 few-shot, 阈值, 关系偏好)      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.1 核心组件清单

| 组件 | 职责 | 类比 |
|---|---|---|
| **CognitiveAgent** | 主循环编排，管理 Agent 状态 | 大脑皮层 |
| **ExtractionKernel** | LangExtract + DeepSeek 高精度提取 | 视觉系统 |
| **KGMemory** | Neo4j 读写，图谱查询，记忆检索 | 海马体 |
| **CausalReasoner** | 因果链推理，传递关系推导 | 前额叶 |
| **ConflictDetector** | 新旧知识冲突检测 | 前扣带皮层 |
| **DecisionEngine** | 新增/更新/标记争议/丢弃 决策 | 运动皮层 |
| **SelfReflection** | 质量监控，提取策略评估 | 元认知 |
| **StrategyManager** | 动态调整 few-shot 示例、提取提示词 | 学习系统 |

---

## 2. 六阶段认知推理循环

### Phase 1: Context Activation（先验知识激活）

**目标**：提取前先查询 Neo4j，激活相关背景知识。

```
输入: PubMed 摘要
动作:
  1. 快速扫描摘要，识别核心实体提及（基因符号、疾病名称）
  2. 对每个识别出的实体，查询 Neo4j:
     - 该实体是否已存在于 KG？
     - 它有哪些已知关系？
     - 这些关系的置信度如何？
  3. 构建 Context Card（上下文卡片），包含:
     - known_entities: {mention → neo4j_node}
     - existing_relations: [(subject, predicate, object, confidence)]
     - knowledge_gaps: ["Gene X 与 HCC 的关系未知", "Pathway Y 缺少下游靶点"]
  4. 根据 knowledge_gaps 动态生成提取目标：
     - 如果已有大量 gene-disease 关联 → 优先提取 mechanism/pathway
     - 如果 disease 节点信息稀疏 → 优先提取 disease 属性和分期
     - 如果存在矛盾证据 → 提高提取敏感度，标记 uncertainty

输出: ContextCard { known_entities, existing_relations, knowledge_gaps, extraction_goals }
```

**实现要点**：

```python
class ContextActivator:
    """Phase 1: 从 Neo4j 激活先验知识"""
    
    def activate(self, abstract: str, kg_memory: KGMemory) -> ContextCard:
        # 1. 快速实体扫描（规则 + 简单 NER，不调 LLM）
        surface_mentions = self._scan_entities(abstract)
        
        # 2. 查询 Neo4j
        known = {}
        for mention in surface_mentions:
            node = kg_memory.find_entity(mention)
            if node:
                known[mention] = {
                    "node": node,
                    "relations": kg_memory.get_relations(node),
                    "confidence": node.get("confidence", 1.0),
                }
        
        # 3. 识别知识缺口
        gaps = self._identify_gaps(known, surface_mentions)
        
        # 4. 生成提取目标
        goals = self._generate_goals(gaps, known)
        
        return ContextCard(
            known_entities=known,
            existing_relations=...,
            knowledge_gaps=gaps,
            extraction_goals=goals,
        )
```

### Phase 2: Extract + Ground（LangExtract 高精度提取）

**目标**：利用 LangExtract + DeepSeek 从摘要中提取结构化知识。

```
输入: PubMed 摘要 + ContextCard
动作:
  1. 根据 ContextCard 动态组装 few-shot 示例:
     - knowledge_gaps 指向 disease → 加入 disease 相关 few-shot
     - 摘要篇幅长 → 启用 LangExtract 自动分块
     - 已有冲突证据 → 降低 uncertainty 阈值
  2. 调用 lx.extract() 执行提取
  3. 接收结构化输出:
     - entities: 带 extraction_class, char_interval, attributes
     - relations: 带 subject, predicate, object, evidence, confidence
  4. LangExtract 自动完成 source grounding (char_interval)
  5. 初步校验: 实体类型合法性、关系签名匹配

输出: RawExtraction { entities, relations, evidence_spans, confidence_scores }
```

**动态 Few-shot 策略**：

```python
class DynamicExampleSelector:
    """根据 ContextCard 动态选择 few-shot 示例"""
    
    def select_examples(self, context: ContextCard, abstract: str) -> list[ExampleData]:
        examples = []
        
        # 基础示例：总是包含
        examples.append(BASE_EXAMPLE_TP53_HCC)
        
        # 根据知识缺口添加领域示例
        for gap in context.knowledge_gaps:
            if gap.type == "gene_disease":
                examples.append(GENE_DISEASE_EXAMPLE)
            elif gap.type == "pathway_mechanism":
                examples.append(PATHWAY_MECHANISM_EXAMPLE)
            elif gap.type == "protein_interaction":
                examples.append(PROTEIN_INTERACTION_EXAMPLE)
            elif gap.type == "metabolite_marker":
                examples.append(METABOLITE_EXAMPLE)
        
        # 根据冲突状态添加"矛盾检测"示例
        if context.has_conflicts:
            examples.append(CONFLICT_DETECTION_EXAMPLE)
        
        # 控制示例数量（LangExtract 建议 2-5 个）
        return examples[:5]
```

### Phase 3: Verify & Reason（图谱溯源验证 + 因果推理）

**目标**：将提取结果与 Neo4j 已有知识交叉验证，并进行因果推理。

```
输入: RawExtraction + ContextCard
动作:
  1. 实体溯源验证:
     - 每个提取实体在 Neo4j 中是否存在？
     - 如果不存在，能否通过别名/synonym 匹配？
     - 匹配结果: EXACT_MATCH / ALIAS_MATCH / FUZZY_MATCH / NOVEL
  2. 关系验证:
     - 提取的关系在 KG 中是否已存在？
     - 如果存在但方向相反 → 标记 INVERTED
     - 如果存在但证据矛盾 → 标记 CONFLICTING
     - 如果不存在 → 标记 NOVEL
  3. 因果推理:
     - 从提取的三元组中识别因果链: A→B + B→C ⇒ A→C
     - 从 Neo4j 补充缺失的中间节点: A→B 已知, B→C 提取 ⇒ 补全链条
     - 检测因果循环和矛盾
     - 提出传递闭包候选: 如果 A→B 和 B→C 都高置信，建议写入 A→C

输出: VerifiedExtraction { entities_with_status, relations_with_status, inferred_causal_chains, contradiction_flags }
```

**因果推理引擎**：

```python
class CausalReasoner:
    """基于 KG 的因果链推理"""
    
    def infer_causal_chains(
        self, 
        extracted: list[Relation], 
        kg_memory: KGMemory
    ) -> list[CausalChain]:
        chains = []
        
        for rel in extracted:
            # 查找以 rel.object 为起点的已知关系
            downstream = kg_memory.query(f"""
                MATCH (o:Entity)-[r]->(t:Entity)
                WHERE o.name = $object_name
                RETURN t.name, type(r), r.confidence
            """, object_name=rel.object)
            
            for down in downstream:
                # A → B (extracted) + B → C (known) ⇒ A → C (inferred)
                chain = CausalChain(
                    steps=[
                        CausalStep(rel.subject, rel.predicate, rel.object, source="extracted"),
                        CausalStep(rel.object, down.relation_type, down.target, source="known"),
                    ],
                    inferred_relation=Relation(
                        subject=rel.subject,
                        predicate=self._compose_predicate(rel.predicate, down.relation_type),
                        object=down.target,
                        evidence=f"Inferred: {rel.evidence} ∧ {down.evidence}",
                        confidence=min(rel.confidence, down.confidence) * 0.8,
                        derivation="causal_transitivity",
                    ),
                )
                chains.append(chain)
        
        return chains
    
    def _compose_predicate(self, p1: str, p2: str) -> str:
        """组合谓词：ASSOCIATED_WITH + PROGRESSES_TO → ASSOCIATED_WITH"""
        if p2 == "PROGRESSES_TO":
            return "ASSOCIATED_WITH"
        if p1 == "PARTICIPATES_IN" and p2 == "ASSOCIATED_WITH":
            return "ASSOCIATED_WITH"
        return "ASSOCIATED_WITH"
```

### Phase 4: Conflict Resolution（冲突检测与解决）

**目标**：检测新旧知识矛盾，执行冲突解决策略。

```
输入: VerifiedExtraction + Neo4j 现有知识
动作:
  1. 冲突分类:
     - DIRECT_CONTRADICTION: A→促进→D (新) vs A→抑制→D (旧)
     - EVIDENCE_STRENGTH: 新证据质量更高 → 更新旧关系
     - METHODOLOGICAL_DIFF: 不同实验条件导致不同结论 → 都保留，加条件注释
     - TEMPORAL_DRIFT: 旧知识可能已过时 → 标记旧关系 + 添加新关系
  2. 冲突解决策略:
     a. 自动解决（高置信新证据压倒低置信旧知识）
     b. 并存保留（矛盾但都有高质量证据支撑）
     c. 升级标记（无法自动解决 → 标记为 DISPUTED，等待人工审核）
  3. 元数据记录: 每条决策都记录 reasoning trace

输出: ResolutionResult { resolution_type, updated_relations, disputed_flags, reasoning_trace }
```

**冲突解决决策表**：

| 旧证据 | 新证据 | 旧置信 | 新置信 | 决策 |
|---|---|---|---|---|
| 存在 | 同向 | 任意 | 更高 | UPDATE: 合并证据，提升置信度 |
| 存在 | 同向 | 任意 | 更低 | KEEP_OLD: 保留旧证据，记录新证据为 SUPPORTING |
| 存在 | 反向 | <0.6 | >0.8 | UPDATE: 新证据推翻旧结论 |
| 存在 | 反向 | >0.8 | >0.8 | DISPUTE: 两者都高质量，标记学术争议 |
| 存在 | 反向 | >0.8 | <0.6 | KEEP_OLD: 新证据不够强 |
| 不存在 | — | — | >0.7 | CREATE: 直接创建新关系 |
| 不存在 | — | — | 0.4-0.7 | CREATE_WITH_FLAG: 创建但标记 UNCERTAIN |
| 不存在 | — | — | <0.4 | DISCARD: 置信度过低 |

### Phase 5: Decide & Execute（决策执行）

**目标**：根据推理结果，执行 Neo4j 写入操作。

```
输入: ResolutionResult
动作:
  1. 决策类型:
     - CREATE_ENTITY: 在 Neo4j 中创建新实体节点
     - CREATE_RELATION: 创建新关系（带证据和置信度）
     - UPDATE_RELATION: 更新已有关系（追加证据，调整置信度）
     - MARK_DISPUTED: 标记关系为学术争议
     - PROPOSE_HYPOTHESIS: 添加因果推理导出的假设关系
     - NO_ACTION: 信息不足以支持任何操作
  2. 写入 Neo4j:
     - 实体节点: Gene/Protein/Disease/Pathway/Metabolite/CellType/Tissue
     - 关系: 带 provenance（来源PMID、evidence、confidence、creation_date）
     - 争议标记: 使用 DISPUTED 标签 + 双方证据引用
  3. 溯源记录: 每条写入操作记录 agent reasoning trace

输出: ExecutionLog { actions_taken, cypher_statements, affected_nodes, affected_relations }
```

**关键变化**：当前 Pipeline 的保守策略（不创建新节点）被**有条件创建**取代：

```python
class DecisionEngine:
    """Phase 5: 决策执行引擎"""
    
    ENTITY_CREATION_POLICY = {
        "Gene": {"min_confidence": 0.8, "require_external_db": True},      # 需 NCBI/HGNC 验证
        "Protein": {"min_confidence": 0.7, "require_external_db": True},   # 需 UniProt 验证
        "Disease": {"min_confidence": 0.6, "require_external_db": False},  # 肝病领域允许自建
        "Pathway": {"min_confidence": 0.7, "require_external_db": False},
        "Metabolite": {"min_confidence": 0.7, "require_external_db": True}, # 需 HMDB 验证
        "CellType": {"min_confidence": 0.6, "require_external_db": False},
        "Tissue": {"min_confidence": 0.5, "require_external_db": False},
    }
    
    def decide(
        self, 
        verified: VerifiedExtraction, 
        resolution: ResolutionResult,
        kg_memory: KGMemory,
    ) -> ExecutionLog:
        actions = []
        
        for entity in verified.entities:
            if entity.neo4j_status == "NOVEL":
                policy = self.ENTITY_CREATION_POLICY[entity.type]
                if entity.confidence >= policy["min_confidence"]:
                    if policy["require_external_db"]:
                        # 需要外部数据库交叉验证
                        external_validated = self._validate_external(entity)
                        if external_validated:
                            actions.append(Action("CREATE_ENTITY", entity))
                        else:
                            actions.append(Action("FLAG_UNVERIFIED", entity))
                    else:
                        actions.append(Action("CREATE_ENTITY", entity))
        
        for resolution_item in resolution.items:
            if resolution_item.decision == "CREATE":
                actions.append(Action("CREATE_RELATION", resolution_item.relation))
            elif resolution_item.decision == "UPDATE":
                actions.append(Action("UPDATE_RELATION", resolution_item))
            elif resolution_item.decision == "DISPUTE":
                actions.append(Action("MARK_DISPUTED", resolution_item))
        
        # 因果推理的假设关系
        for chain in verified.causal_chains:
            actions.append(Action("PROPOSE_HYPOTHESIS", chain.inferred_relation))
        
        return ExecutionLog(actions=actions, ...)
```

### Phase 6: Reflect & Adapt（元认知反思 + 策略自适应）

**目标**：评估本轮提取质量，动态调整下一轮的提取策略。

```
输入: 本轮 ExecutionLog + 历史统计
动作:
  1. 质量评估:
     - 提取产出率: 提取了多少实体/关系 vs 摘要长度？
     - 实体链接率: 多少实体能在 Neo4j 中找到？
     - 冲突率: 有多少新关系与已有知识冲突？
     - 决策执行率: 多少提取结果最终被写入/更新/标记？
     - 证据质量: char_interval 对齐率如何？
  2. 策略调整:
     - 如果实体链接率低 → 可能是 new entity types in abstracts，放宽链接阈值
     - 如果冲突率高 → 可能是摘要类型不适合当前 schema，调整提取提示词
     - 如果决策执行率低 → 可能是置信度阈值过高，或 KG 节点覆盖不足
     - 如果证据对齐率低 → 可能是 LangExtract char_interval 在 DeepSeek 上不稳定
  3. 跨文档知识累积:
     - 高频出现但不在 KG 中的实体 → 建议扩充 KG 节点
     - 高频出现但 schema 不支持的关系 → 建议扩展 relation signatures
     - 多篇文章指向同一结论 → 提升该结论的置信度
  4. 自我修正:
     - 对决策被标记为 DISPUTED 的关系 → 后续遇到时更谨慎
     - 对高置信 CREATE 操作 → 后续遇到同一实体时提高优先级

输出: StrategyUpdate { adjusted_thresholds, new_examples, schema_proposals, quality_report }
```

**自适应示例**：

```python
class SelfReflection:
    """Phase 6: 元认知反思"""
    
    def reflect(
        self,
        execution_log: ExecutionLog,
        context_card: ContextCard,
        history: AgentHistory,
    ) -> StrategyUpdate:
        update = StrategyUpdate()
        
        # 1. 计算本轮质量指标
        metrics = self._compute_metrics(execution_log, context_card)
        
        # 2. 阈值自适应
        if metrics.entity_link_rate < 0.3:
            # 很多实体不在 KG 中 → 可能是新兴研究方向
            # 策略：降低实体创建门槛，增加 Novel entity 的自动创建
            update.adjust_threshold("entity_creation_confidence", -0.1)
            update.log("检测到低实体链接率，降低 Novel entity 创建阈值")
        
        if metrics.conflict_rate > 0.4:
            # 大量冲突 → 可能是摘要类型与现有知识不一致
            # 策略：增加 DISPUTED 标记比例，减少自动 UPDATE
            update.adjust_strategy("conflict_resolution", "conservative")
            update.log("检测到高冲突率，切换为保守冲突解决策略")
        
        if metrics.decision_execution_rate < 0.1:
            # 几乎所有提取都被丢弃 → 检查是否是 schema 问题
            top_rejected_reasons = self._analyze_rejections(execution_log)
            if "schema_mismatch" in top_rejected_reasons:
                update.propose_schema_extension(top_rejected_reasons["schema_mismatch"])
                update.log(f"检测到高频 schema 不匹配: {top_rejected_reasons['schema_mismatch']}")
        
        # 3. 跨文档知识发现
        if history.total_articles % 20 == 0:
            # 每 20 篇文章做一次宏观反思
            emerging_entities = history.find_emerging_entities(min_frequency=3)
            for entity in emerging_entities:
                update.suggest_node_expansion(entity)
            
            emerging_relations = history.find_emerging_relations(min_frequency=3)
            for rel in emerging_relations:
                update.suggest_schema_extension(rel)
        
        # 4. Few-shot 示例演化
        if metrics.extraction_quality < 0.5:
            # 提取质量差 → 可能 few-shot 示例不匹配当前摘要类型
            # 从历史高质提取中挑选新示例
            new_example = history.select_best_extraction_as_example()
            if new_example:
                update.add_example(new_example)
                update.log("从高质量历史提取中生成新 few-shot 示例")
        
        return update
```

---

## 3. 代码结构

```
liver_disease_kg_project/
│
├── cognitive_agent/
│   ├── __init__.py
│   ├── agent.py                  # CognitiveAgent 主循环 (~300 行)
│   ├── context_activator.py      # Phase 1: 先验知识激活 (~150 行)
│   ├── extraction_kernel.py      # Phase 2: LangExtract 内核包装 (~200 行)
│   ├── verifier.py               # Phase 3: 图谱溯源验证 (~200 行)
│   ├── causal_reasoner.py        # Phase 3: 因果链推理 (~250 行)
│   ├── conflict_resolver.py      # Phase 4: 冲突检测与解决 (~200 行)
│   ├── decision_engine.py        # Phase 5: 决策执行 (~250 行)
│   ├── self_reflection.py        # Phase 6: 元认知反思 (~250 行)
│   ├── strategy_manager.py       # 策略管理器 (~200 行)
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── kg_memory.py          # Neo4j 记忆体接口 (~300 行)
│   │   ├── working_memory.py     # Agent 工作记忆 (~100 行)
│   │   └── episodic_memory.py    # 历史决策记录 (~150 行)
│   │
│   └── schema/
│       ├── __init__.py
│       ├── entity_classes.py     # 实体类型定义 (~100 行)
│       ├── relation_signatures.py # 关系签名 (~80 行)
│       ├── examples.py           # Few-shot 示例库 (~200 行)
│       └── validators.py         # 校验规则（从旧 Pipeline 移植） (~150 行)
│
├── tools/
│   ├── neo4j_client.py           # Neo4j 连接客户端
│   ├── entity_linker.py          # 实体链接器（从 preflight 移植）
│   └── external_validator.py     # NCBI/UniProt/HMDB 外部验证
│
├── multi_stage_extraction_pipeline.py  # [保留] 旧 Pipeline 作为 baseline
├── entity_linking_preflight.py         # [保留] 实体链接预检
├── convert_pubmed_xml_to_jsonl.py      # [保留] XML 转换
│
├── docs/
│   ├── cognitive_agent_architecture.md # 本文档
│   ├── langextract_redesign_plan.md    # 之前的重设计文档
│   └── pubmed_extraction_report.md     # 综合报告
│
└── extraction_output/                  # 输出目录（不变）
```

**总代码量估算**：~2,500 行（Agent 核心 ~2,000 + Schema/Memory ~500）
虽然比旧 Pipeline (1,628 行) 多，但实现了质的飞跃：工具 → 智能体。

---

## 4. 主 Agent 核心代码

```python
#!/usr/bin/env python3
"""
cognitive_agent/agent.py — 自主认知知识管理 Agent 主循环

基于 LangExtract + DeepSeek 的高精度提取，
以 Neo4j 为外部动态记忆，
实现 "提取 → 验证 → 推理 → 决策 → 反思" 的闭环认知循环。
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import langextract as lx
from langextract.factory import ModelConfig

from .context_activator import ContextActivator, ContextCard
from .extraction_kernel import ExtractionKernel, RawExtraction
from .verifier import KGVerifier, VerifiedExtraction
from .causal_reasoner import CausalReasoner, CausalChain
from .conflict_resolver import ConflictResolver, ResolutionResult
from .decision_engine import DecisionEngine, ExecutionLog, Action
from .self_reflection import SelfReflection, StrategyUpdate
from .strategy_manager import StrategyManager
from .memory.kg_memory import KGMemory
from .memory.working_memory import WorkingMemory
from .memory.episodic_memory import EpisodicMemory


@dataclass
class AgentConfig:
    """Agent 配置"""
    model_id: str = "deepseek-chat"
    api_key: str = ""
    api_base: str = "https://api.deepseek.com/v1"
    neo4j_uri: str = "bolt://100.104.181.96:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    
    # 决策阈值
    entity_creation_confidence: float = 0.7
    relation_creation_confidence: float = 0.7
    auto_update_threshold: float = 0.8
    dispute_threshold: float = 0.6
    
    # 反思参数
    reflection_interval: int = 20      # 每 N 篇文章做一次宏观反思
    
    # 跑参数
    max_workers: int = 8
    temperature: float = 0.0


@dataclass
class AgentState:
    """Agent 运行状态"""
    total_articles: int = 0
    total_extractions: int = 0
    total_entities_created: int = 0
    total_relations_created: int = 0
    total_relations_updated: int = 0
    total_disputed: int = 0
    total_discarded: int = 0
    quality_scores: list[float] = field(default_factory=list)
    recent_strategy_updates: list[StrategyUpdate] = field(default_factory=list)


class CognitiveAgent:
    """
    自主认知知识管理 Agent
    
    对每篇 PubMed 摘要执行完整的认知推理循环：
    Context → Extract → Verify → Resolve → Decide → Reflect
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = AgentState()
        
        # 初始化组件
        self.kg_memory = KGMemory(
            uri=config.neo4j_uri,
            user=config.neo4j_user,
            password=config.neo4j_password,
        )
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        
        self.context_activator = ContextActivator(self.kg_memory)
        self.extraction_kernel = ExtractionKernel(self._build_lx_config())
        self.verifier = KGVerifier(self.kg_memory)
        self.causal_reasoner = CausalReasoner(self.kg_memory)
        self.conflict_resolver = ConflictResolver()
        self.decision_engine = DecisionEngine(self.kg_memory)
        self.self_reflection = SelfReflection()
        self.strategy_manager = StrategyManager()
        
        self.history: list[dict] = []  # 本轮所有处理记录
    
    def _build_lx_config(self) -> ModelConfig:
        """构建 LangExtract 的 DeepSeek 模型配置"""
        return ModelConfig(
            provider="openai",
            model_id=self.config.model_id,
            provider_kwargs={
                "api_key": self.config.api_key,
                "base_url": self.config.api_base,
            },
        )
    
    def process_article(self, article: dict) -> dict:
        """
        处理单篇 PubMed 文章 — 完整的认知推理循环
        
        Args:
            article: {"pmid": ..., "title": ..., "abstract": ..., "source": "PubMed"}
        
        Returns:
            processing_record: 包含所有阶段的完整处理记录
        """
        pmid = article["pmid"]
        text = f"TITLE: {article['title']}\nABSTRACT: {article['abstract']}"
        
        record = {
            "pmid": pmid,
            "title": article["title"],
            "timestamp": time.time(),
            "phases": {},
        }
        
        try:
            # ── Phase 1: Context Activation ──
            context_card = self.context_activator.activate(text)
            record["phases"]["context"] = context_card.to_dict()
            
            # 根据 Context 动态调整策略
            strategy = self.strategy_manager.get_strategy(context_card)
            
            # ── Phase 2: Extract + Ground ──
            raw_extraction = self.extraction_kernel.extract(
                text=text,
                context=context_card,
                strategy=strategy,
                document_id=pmid,
            )
            record["phases"]["extraction"] = raw_extraction.to_dict()
            
            # ── Phase 3: Verify & Reason ──
            verified = self.verifier.verify(raw_extraction, context_card)
            causal_chains = self.causal_reasoner.infer_causal_chains(
                verified.relations, self.kg_memory
            )
            verified.causal_chains = causal_chains
            record["phases"]["verification"] = verified.to_dict()
            
            # ── Phase 4: Conflict Resolution ──
            resolution = self.conflict_resolver.resolve(verified)
            record["phases"]["resolution"] = resolution.to_dict()
            
            # ── Phase 5: Decide & Execute ──
            execution_log = self.decision_engine.decide(verified, resolution)
            record["phases"]["execution"] = execution_log.to_dict()
            
            # ── Phase 6: Reflect & Adapt ──
            if self.state.total_articles % self.config.reflection_interval == 0:
                strategy_update = self.self_reflection.reflect(
                    execution_log, context_card, self.state
                )
                self.strategy_manager.apply_update(strategy_update)
                record["phases"]["reflection"] = strategy_update.to_dict()
            
            # 更新状态
            self._update_state(record, execution_log)
            
        except Exception as e:
            record["error"] = str(e)
            # 错误时触发自我修正
            self.strategy_manager.handle_error(e, article)
        
        self.history.append(record)
        self.state.total_articles += 1
        
        return record
    
    def _update_state(self, record: dict, execution_log: ExecutionLog):
        """更新 Agent 全局状态"""
        for action in execution_log.actions:
            if action.type == "CREATE_ENTITY":
                self.state.total_entities_created += 1
            elif action.type == "CREATE_RELATION":
                self.state.total_relations_created += 1
            elif action.type == "UPDATE_RELATION":
                self.state.total_relations_updated += 1
            elif action.type == "MARK_DISPUTED":
                self.state.total_disputed += 1
            elif action.type == "DISCARD":
                self.state.total_discarded += 1
    
    def run_batch(
        self, 
        articles: list[dict],
        run_id: str = "agent_v1",
        output_dir: Path = Path("extraction_output"),
    ) -> dict:
        """
        批量处理 PubMed 文章
        
        Args:
            articles: PubMed JSONL 记录列表
            run_id: 运行标识
            output_dir: 输出目录
        
        Returns:
            batch_report: 批量处理报告
        """
        print(f"\n{'='*60}")
        print(f"  🧠 Cognitive Agent — {run_id}")
        print(f"  Articles: {len(articles)}")
        print(f"  Model: {self.config.model_id} via LangExtract")
        print(f"  Neo4j: {self.config.neo4j_uri}")
        print(f"{'='*60}\n")
        
        for i, article in enumerate(articles):
            print(f"[{i+1}/{len(articles)}] PMID:{article['pmid']} ...", end=" ")
            record = self.process_article(article)
            
            # 简要输出
            exec_log = record.get("phases", {}).get("execution", {})
            n_created = sum(1 for a in exec_log.get("actions", []) if a["type"].startswith("CREATE"))
            n_updated = sum(1 for a in exec_log.get("actions", []) if a["type"] == "UPDATE_RELATION")
            n_disputed = sum(1 for a in exec_log.get("actions", []) if a["type"] == "MARK_DISPUTED")
            print(f"✓ +{n_created} ~{n_updated} ⚡{n_disputed}")
        
        # 生成报告
        report = self._generate_report(run_id, articles)
        
        # 保存
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"agent_results_{run_id}.json", "w") as f:
            json.dump({"report": report, "records": self.history}, f, indent=2, ensure_ascii=False)
        
        self._print_report(report)
        return report
    
    def _generate_report(self, run_id: str, articles: list[dict]) -> dict:
        """生成批量处理报告"""
        total = len(articles)
        return {
            "run_id": run_id,
            "model": self.config.model_id,
            "total_articles": total,
            "agent_state": {
                "total_entities_created": self.state.total_entities_created,
                "total_relations_created": self.state.total_relations_created,
                "total_relations_updated": self.state.total_relations_updated,
                "total_disputed": self.state.total_disputed,
                "total_discarded": self.state.total_discarded,
            },
            "strategy_evolution": [
                u.to_dict() for u in self.state.recent_strategy_updates
            ],
            "quality_metrics": {
                "avg_quality_score": sum(self.state.quality_scores) / len(self.state.quality_scores) if self.state.quality_scores else 0,
                "entity_creation_rate": self.state.total_entities_created / max(total, 1),
                "relation_creation_rate": self.state.total_relations_created / max(total, 1),
                "conflict_rate": self.state.total_disputed / max(self.state.total_relations_created + self.state.total_disputed, 1),
            },
        }
    
    def _print_report(self, report: dict):
        """打印人可读报告"""
        s = report["agent_state"]
        q = report["quality_metrics"]
        print(f"\n{'='*60}")
        print(f"  📊 Agent 运行报告 — {report['run_id']}")
        print(f"{'='*60}")
        print(f"  处理文章:    {report['total_articles']}")
        print(f"  创建实体:    {s['total_entities_created']}")
        print(f"  创建关系:    {s['total_relations_created']}")
        print(f"  更新关系:    {s['total_relations_updated']}")
        print(f"  标记争议:    {s['total_disputed']}")
        print(f"  丢弃:        {s['total_discarded']}")
        print(f"  ─────────────────────────────")
        print(f"  平均质量:    {q['avg_quality_score']:.2f}")
        print(f"  实体创建率:  {q['entity_creation_rate']:.2f}/article")
        print(f"  关系创建率:  {q['relation_creation_rate']:.2f}/article")
        print(f"  冲突率:      {q['conflict_rate']:.1%}")
        print(f"{'='*60}\n")


# ── CLI 入口 ──────────────────────────────────────────────────

def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="🧠 Cognitive Agent — PubMed KG Extraction")
    parser.add_argument("--input", required=True, help="PubMed JSONL 输入文件")
    parser.add_argument("--limit", type=int, default=10, help="处理文章数")
    parser.add_argument("--run-id", default="agent_v1", help="运行标识")
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    args = parser.parse_args()
    
    # 加载文章
    articles = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    articles = articles[:args.limit]
    
    # 初始化 Agent
    config = AgentConfig(
        api_key=args.api_key,
        neo4j_password=args.neo4j_password,
    )
    agent = CognitiveAgent(config)
    
    # 运行
    report = agent.run_batch(articles, run_id=args.run_id)


if __name__ == "__main__":
    main()
```

---

## 5. 与当前 Pipeline 的能力对比

| 维度 | 当前 Pipeline | Cognitive Agent |
|---|---|---|
| **执行模式** | 固定 5 阶段流水线 | 6 阶段动态推理循环 |
| **Neo4j 角色** | 只写不读 (Stage 5) | 读写双向 (Phase 1,3,4,5) |
| **先验知识** | 无 (每篇独立处理) | Phase 1 激活 Neo4j 先验知识 |
| **提取策略** | 固定 prompt + 固定示例 | 动态 few-shot + 策略自适应 |
| **冲突检测** | 仅内部去重 (Stage 4) | 新旧知识全面冲突检测 (Phase 4) |
| **因果推理** | 无 | 因果链传递推理 (Phase 3) |
| **决策粒度** | 二元 (import-ready / reject) | 6 种决策 (Create/Update/Dispute/Discard/Hypothesize/NoAction) |
| **新实体** | 不创建 (保守策略) | 有条件创建 (带外部验证) |
| **错误处理** | 手动修复 | 自我检测 + 自动修正 (Phase 6) |
| **知识演化** | 静态 Schema | Schema 动态演化（从数据中学习） |
| **溯源** | 证据句 + 字符偏移 | 证据句 + 字符偏移 + **推理链** + **决策溯源** |
| **跨文档** | 无 | 跨文档知识累积 + 新实体/关系发现 |

---

## 6. 创新点总结（论文可写）

### 6.1 方法论创新

1. **外部记忆增强的认知循环** (Memory-Augmented Cognitive Loop)
   - Neo4j 不只是存储目标，更是 Agent 的**外部动态记忆体**
   - Agent 在每个推理阶段都主动查询图谱，形成 **提取 ⇄ 记忆** 的双向交互
   - 区别于传统的 "Extract-then-Write" 单向流水线

2. **图谱反馈驱动的策略自适应** (KG-Feedback-Driven Strategy Adaptation)
   - Phase 6 的元认知反思根据图谱反馈动态调整提取策略
   - Few-shot 示例不再是静态的，而是从高质量历史提取中**自动演化**
   - 决策阈值根据实体链接率、冲突率等指标**自动校准**

3. **因果传递推理** (Causal Transitivity Reasoning)
   - 从提取的二元关系 + KG 已知关系中推导新的传递关系
   - 显式记录推理链，区分 direct evidence 和 inferred hypothesis
   - 提出可验证的新假设 (hypothesis generation)

4. **基于冲突证据的知识演化** (Conflict-Driven Knowledge Evolution)
   - 不是简单的去重，而是深度冲突分析 (DIRECT / EVIDENCE / METHODOLOGICAL / TEMPORAL)
   - 学术争议被显式标记为 DISPUTED（保留双方证据），而非硬性选择一方
   - 知识图谱随时间演化，新旧证据共存

### 6.2 工程创新

5. **LangExtract + DeepSeek 的异构模型协作**
   - LangExtract 的高精度提取能力（即使非 Gemini 也能用其框架优势）
   - DeepSeek 的低成本 + 生物医学领域知识
   - 通过 ModelConfig 实现的无缝集成

6. **六阶段推理的可审计性** (Auditable Reasoning)
   - 每个决策都有完整的 reasoning trace
   - 从 Context Activation 到 Execution 的每一步都可回溯
   - 支持按 PMID、实体、关系类型检索决策历史

---

## 7. 实现路线图

### Phase 1: 核心内核（3 天）

```
目标: Agent 能跑通最小完整循环

任务:
  [ ] cognitive_agent/memory/kg_memory.py — Neo4j 读写接口
  [ ] cognitive_agent/agent.py — 主循环骨架
  [ ] cognitive_agent/extraction_kernel.py — LangExtract + DeepSeek 包装
  [ ] cognitive_agent/context_activator.py — Phase 1
  [ ] cognitive_agent/verifier.py — Phase 3 基础版
  [ ] cognitive_agent/decision_engine.py — Phase 5 基础版

验证: 5 篇文章，Agent 能执行 Extract → Verify → Decide 三个核心阶段
```

### Phase 2: 推理与冲突（2 天）

```
目标: 加上因果推理和冲突检测

任务:
  [ ] cognitive_agent/causal_reasoner.py — Phase 3 因果推理
  [ ] cognitive_agent/conflict_resolver.py — Phase 4
  [ ] cognitive_agent/memory/episodic_memory.py — 决策历史

验证: 20 篇文章，能检测冲突并生成因果链
```

### Phase 3: 反思与自适应（2 天）

```
目标: Agent 能自我反思和调整策略

任务:
  [ ] cognitive_agent/self_reflection.py — Phase 6
  [ ] cognitive_agent/strategy_manager.py — 策略管理
  [ ] cognitive_agent/schema/examples.py — 动态 few-shot 示例库

验证: 50 篇文章，观察策略是否随运行而改善
```

### Phase 4: 对比实验（2 天）

```
目标: 500 篇文章完整对比

任务:
  [ ] 相同 500 篇文章跑 Cognitive Agent
  [ ] 对比指标: import-ready 数量、实体创建数、冲突检测数、人工审核通过率
  [ ] 输出对比报告 docs/cognitive_agent_vs_pipeline_comparison.md
```

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| LangExtract char_interval 在 DeepSeek 上不稳定 | Phase 2 证据定位失败 | 回退到手写子串匹配（保留旧 Pipeline 的 `_locate_evidence`） |
| Neo4j 查询延迟影响推理速度 | 处理速度变慢 | 工作内存缓存常用查询结果 |
| DeepSeek API 限流 | 批量运行中断 | 指数退避重试 + 本地队列 |
| 自我修正导致策略漂移 | 提取质量波动 | 保留 baseline 策略快照，偏离超过阈值时回退 |
| 外部验证 API 不可用 | 实体创建受阻 | 降级为仅 Neo4j 内部验证 |

---

## 9. 与实体链接 Preflight 的关系

`entity_linking_preflight.py` **保留但角色升级**：

- 旧角色：Pipeline 的最后一道关卡（Gatekeeper），决定 import/discard
- 新角色：Agent 的一个**验证工具**（Phase 3 调用），为决策提供信息而非做最终决策
- Agent 的 DecisionEngine 参考 Preflight 的链接结果，但拥有最终决策权

```
旧: Extraction → Preflight (Gatekeeper) → Import/Reject
新: Extraction → Preflight (Advisor) → Agent Decision → Create/Update/Dispute/...
```

---

## 10. 结论

这个 Cognitive Agent 实现了一个**质变**：

| 旧范式 | 新范式 |
|---|---|
| 信息抽取工具 | 知识管理智能体 |
| 执行固定流程 | 自主推理循环 |
| 被动适应 Schema | 主动演化 Schema |
| 二元决策 (import/reject) | 六元决策 (create/update/dispute/discard/hypothesize/noop) |
| 独立处理每篇文章 | 跨文档知识累积 |
| 人工发现错误 | 自我检测 + 修正 |

**核心洞见**：LangExtract 提供高精度提取，Neo4j 提供可查询的外部记忆，Agent 在这两者之间建立**认知闭环**——这正是从"工具"到"智能体"的跃迁。

---

## 参考

- [Google LangExtract](https://github.com/google/langextract) — 提取内核
- [LangExtract Provider System](https://deepwiki.com/google/langextract/3.2-provider-system) — 自定义模型配置
- [Instructor](https://python.useinstructor.com/) — 备选结构化提取方案
- [ReAct Prompting](https://arxiv.org/abs/2210.03629) — 推理-行动循环的理论基础
- [DSPy](https://github.com/stanfordnlp/dspy) — 自动 prompt 优化（LangStruct 使用）
