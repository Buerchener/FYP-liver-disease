# LiverKG：事件分类工作台

首页按「事件抽取 → 概念化 → Leaf 归一化 → 层级归纳」查看已保存的真实结果。默认打开分类图，点击概念逐层查看父类、子类、定义与原文证据；三个预定义顶层和未连顶层的概念分开浏览，多父关系完整保留。连线颜色来自已保存的 Codex 定义复查，仍需人工审定。

首页只读取结果，不会启动模型调用。原有 Task 2 模拟运行、模型连接、参数和历史回放保留在顶栏的「实验记录」中，默认实验使用 **Synthetic Data + Mock LLM**。

自底向上归纳使用独立的 [Root-Constrained CLI](../root_event_taxonomy/README.md)。

## 启动与使用

在仓库根目录双击 **`启动分类体系工作台.command`**。浏览器会打开 <http://127.0.0.1:4318>。保持启动它的终端窗口打开。

也可运行：

```bash
python3 -m taxonomy_workbench --open
```

工作台使用独立的 4318 端口。首页读取 `event_taxonomy/runs/xml-event-only/` 下最新的完整归纳结果，通过保存的 manifest 追溯对应的概念化和 Leaf 归一化记录；不会修改结果、第一轮 XML 提示词或原文证据。没有可用结果时显示空状态，不自动替换为模拟数据。

「实验记录」（`/experiments`）操作顺序：

1. 选择「合成示例」和「模拟模型」。
2. 点击「开始演示」，每次运行会创建独立的实验目录。
3. 在「归纳过程」查看诊断、归纳和验证。
4. 点击下方步骤或使用「回放」，阅读中文解释。回放只读取历史，不会再次调用模型。
5. 在分类图上方切换「当前」「初始」「对比」「候选」。候选视图显示提交状态。
6. 点击图中概念，查看定义、关联 Event、证据原句及 PMID。
7. 运行中可请求暂停；算法完成当前步骤、保存检查点后暂停。之后点击「继续运行」。

默认示例展示一次拒绝与替代候选、接受后继续当前父节点、当前层稳定后递归，以及多父关系保留。概念从 11 个增加到 16 个是合成测试行为，不能解释为真实实验的质量提升。

## 数据与真实模型

数据列表自动读取 Task 1 的最终导出，显示概念数、事件数、未归属概念数和缺失定义数。当前网页来源的 76 个类型概念还不能直接运行；按钮会禁用并解释原因，系统不会自动补写定义或强制分配父节点。

可以导入不超过 20 MB 的完整数据集 JSON：

```text
{ data_origin, taxonomy, events, evidence, documents }
```

导入时复用 Task 2 的完整性与准备状态检查。Mock 仅适用于内置示例；导入数据需要使用真实模型。若需更多 Mock 场景，仍可用 Task 2 的可配置 CLI 与自定义 Mock 计划进行测试。

点击「模型连接」填写 Base URL、模型 ID 和 API Key。选择「真实模型」并点击「开始归纳」后发起模型调用。A/B/C 使用同一指定模型、各自独立的版本化 Prompt。

密钥仅存在于服务内存及工作进程环境，不保存到浏览器、配置文件或实验快照；关闭服务后需要重新配置。也可以启动前设置 `EVENT_TAXONOMY_BASE_URL`、`EVENT_TAXONOMY_MODEL`、`EVENT_TAXONOMY_API_KEY` 环境变量。日志中的数据和证据仍是完整实验记录。

## 工程结构

| 文件 | 用途 |
|---|---|
| `static/index.html`、`pipeline.css`、`pipeline.js`、`taxonomy-graph.js` | 当前流程、交互分类图、概念定义及原文证据 |
| `pipeline.py` | 读取真实结果、追溯上游数据、核对图结构及复查记录 |
| `static/experiments.html`、`style.css`、`app.js` | 原有实验控制、图形比较、逐轮回放及模型记录 |
| `service.py` | 数据导入、运行管理、Python 子进程、暂停恢复、实验读取 |
| `server.py` | 仅监听 loopback 的 HTTP 接口与静态文件服务 |
| `viewmodel.py` | 将结构化决策转换为中文流程解释，不编造模型判断依据 |
| `__main__.py` | 一键启动入口 |
| `tests/` | 接口、真实 Mock 工作进程、暂停恢复、证据、前端逻辑测试 |

服务仅允许 `127.0.0.1`/`localhost` 的正确 Host 与同源请求，写入请求须带页面会话 token。只提供明确列出的静态文件和实验导出，拒绝目录穿越与符号链接目录访问。服务器不提供任意命令执行或任意本地文件读取接口。

工作台自己的数据集、无密钥配置和作业状态在 `taxonomy_workbench/state/`，已加入忽略规则。算法输出仍在 `event_taxonomy/runs/web-*`，包含原有完整日志、Prompt、输入、快照和保留率报告。

本次对 Task 2 核心仅增加了协作暂停接口：`RecursiveRefiner.run(stop_requested=...)` 在步骤之间检查暂停，运行器从 `EVENT_TAXONOMY_PAUSE_FILE` 读取控制请求。不改变归纳判断、接受规则或递归顺序。代码变更前的历史实验可继续查看；若恢复时版本校验不一致，应使用那个实验的 `code_snapshot`，界面不会绕过校验。

## 验证

```bash
python3 -m unittest discover -s event_taxonomy/tests -v
python3 -m unittest taxonomy_workbench.tests.test_workbench taxonomy_workbench.tests.test_pipeline -v
node --test taxonomy_workbench/tests/frontend.test.mjs
```

验证包含：实际启动 Mock Python 工作进程、保存与读取初始/最终图、查看被拒候选、按节点读取证据、导出结果、暂停与恢复、篡改检测、Host/Origin/token 保护、路径限制、密钥不落盘、前端文本转义及真实 ID 保留率计算。

用本机 4318 的浏览器页面检查了桌面和窄屏、四种图视图、缩放、回放、节点键盘操作、原文证据及真实数据的准备提示。模拟实验的启动、暂停与恢复通过临时目录中的后端测试验证。
