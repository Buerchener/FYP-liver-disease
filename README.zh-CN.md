<div align="center">

# LiverKG 肝病知识图谱

### 一个证据可追溯、面向科研的肝病知识图谱与 PubMed Agent 抽取系统

[English](README.md) · [PubMed Agent](pubmed_literature_extraction/README.zh-CN.md) · [人工审阅入口](HUMAN_REVIEW.md)

</div>

## 项目简介

LiverKG 围绕慢性肝病演进过程构建知识图谱，将权威生物医学数据库与
Verifier 驱动的文献抽取 Agent 结合，把 PubMed 摘要转化为证据可追溯的实体和关系候选。

项目坚持一个核心原则：**模型可以提出知识，但只有 Schema 合法、原文证据充分、
并通过确定性验证的结果才能进入写入路径。**

```text
NAFLD → NASH → Fibrosis → Cirrhosis → HCC
```

## 核心特点

- 以五阶段肝病进展链为主干，保留统一标识符与来源信息；
- 整合基因、蛋白、通路、组织、细胞类型和代谢物背景；
- 同时保留 legacy 多阶段抽取器与完全兼容的 Central Agent v2；
- 采用 evidence-first 验证，严格检查原文连续证据；
- 基于 Schema 构造实体对候选，并显式建模 `NO_RELATION`；
- 支持按需调用 DeepSeek/Qwen 兼容裁判模型，所有修改必须重新验证；
- 支持缓存优先执行、动态远程预算、token/延迟和动作轨迹审计；
- Neo4j 默认 dry-run，并由确定性 Safe Write 门禁控制；
- 提供冻结金标、可复现实验输出和消融研究基础。

## 知识图谱范围

| 层级 | 当前规模 | 主要来源 |
| --- | ---: | --- |
| Disease | 5 | UMLS 对齐主干 |
| Gene | 836 | DisGeNET |
| Protein | 793 | STRING |
| Pathway | 1,721 | KEGG、Reactome |
| Tissue | 1 | HPA |
| CellType | 154 | HPA 单细胞数据 |
| Metabolite | 36 | HMDB |
| Gene–Disease 关系 | 1,036 | DisGeNET |
| Protein–Protein 互作 | 7,154 | STRING |
| Gene–Pathway 关系 | 9,947 | KEGG、Reactome |
| HPA 表达关系 | 17,745 | HPA |
| HPA LIHC 预后关系 | 1,384 | HPA |
| Gene–Metabolite 关系 | 97 | HMDB |

上述规模对应已整理的 v2 数据包。PubMed 抽取结果在通过证据、Schema 和
import-ready 门禁之前，始终只作为候选知识存在。

## 系统架构

```text
结构化数据库 ─────────────────────┐
                                  ├─→ 标准化 TSV/JSON ─→ Neo4j
PubMed 摘要                        │
  └─→ 预处理                       │
      └─→ 主抽取                   │
          └─→ 实体对分类           │
              └─→ 中央 Controller │
                  ├─→ 图谱查询     │
                  ├─→ 第二模型裁决 │
                  ├─→ Reviewer     │
                  └─→ Causal/Conflict
                         ↓
                    确定性 Verifier
                         ↓
                    Safe Write 门禁 ─────┘
```

Central Agent 根据当前文章状态选择工具。Causal 输出被隔离为研究假设，RAG
不能替代文章证据，任何 LLM 都不能覆盖 Verifier 的硬性否决。

## 仓库结构

```text
.
├── data/                         # v2 导入数据
├── scripts/                      # 数据抓取、清洗和导入脚本
├── review/                       # 当前 Schema、范围和限制
├── archive/                      # 历史报告与迁移材料
├── pubmed_literature_extraction/ # PubMed Agent 文献抽取模块
├── HUMAN_REVIEW.md               # 推荐人工审阅入口
└── MANIFEST.txt                  # 数据包清单
```

## 快速开始

```bash
cd pubmed_literature_extraction
python3.12 -m venv .venv-cognitive
source .venv-cognitive/bin/activate
python -m pip install -r requirements-cognitive-agent.txt

cp .env.example .env
# 仅在本地填写凭证，禁止提交 .env。
set -a && source .env && set +a

# 默认兼容模式，禁止写库
./run_cognitive_agent.sh 5

# Central Agent v2 审计模式
AGENT_EXECUTION_MODE=agent-v2-shadow ./run_cognitive_agent.sh 5
```

详细配置、实验方法和安全约束请参阅
[PubMed 抽取说明](pubmed_literature_extraction/README.zh-CN.md)。

## 可复现性与安全

- 默认执行模式为 dry-run，Neo4j 写入必须显式开启；
- active Agent v2 和 active relation classifier 在验收前仅允许 dry-run；
- API Key、密码、虚拟环境、缓存和大批量生成结果不会进入 Git；
- 外部图谱上下文只能辅助链接或冲突判断，不能作为文章直接证据；
- 所有模型修改均重新经过确定性验证；
- Neo4j 集成测试只允许连接显式配置的隔离测试库。

## 证据解释边界

- DisGeNET Gene–Disease association 是疾病证据主干；
- STRING PPI 是分子互作背景，不等于直接疾病因果；
- KEGG/Reactome 通路参与不代表疾病阶段专属性；
- HPA CellType 默认是泛组织背景，除非数据明确为肝脏特异；
- HMDB 数据是保守筛选后的疾病相关背景，不是全量导入。

## 当前状态

v2 数据包已经通过 backbone 范围审计。Central Agent v2 已完成 legacy 兼容、
动态预算、持久化缓存、动作审计和 Safe Write 隔离。下一阶段重点是提升证据精度、
训练生物医学关系分类器，并扩大冻结测试集以形成论文级统计结论。

建议从 [HUMAN_REVIEW.md](HUMAN_REVIEW.md) 和
[当前限制与后续清理](review/04_caveats_and_next_cleanup.md) 开始审阅。
