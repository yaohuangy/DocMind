# 🧠 DocMind MCP Native Agent — 技术方案

> **目标**：DocMind = MCP Server（对外提供知识库） + MCP Client（对内调用外部工具）
>
> **定位**：从"被 AI 调用的文档查询工具"升级为"能主动调用外部工具的本地知识中枢"

---

## 一、现状

```
当前（只有 Server 端）：

  Claude Code / Copilot
         │
    MCP Tool 调用
         │
    ┌────▼─────┐
    │ DocMind  │  → QAEngine → ChromaDB / Neo4j / SQLite
    └──────────┘
         ↑
    只查本地知识库，没有外部信息

问题：用户问"Q3 营收目标"时，
  如果文档里有 → 能回答
  如果文档里没有 → 回答"找不到"
  无法融合外部信息（行业基准、最新动态）
```

## 二、目标架构

```
                         ┌─────────────────────────┐
                         │   Claude Code / Copilot  │
                         └────────────┬────────────┘
                                      │ MCP 协议
                         ┌────────────▼────────────┐
                         │   DocMind MCP Server     │
                         │   (3 工具，已完成)         │
                         │   + 新增 tool-use 工具    │
                         └──────┬──────────┬───────┘
                                │          │
                    ┌───────────▼──┐   ┌───▼──────────────┐
                    │  QAEngine    │   │ MCPClientManager  │ ← 新增
                    │  (不变)       │   │                   │
                    │  retrieve()  │   │ 连接外部 MCP Server │
                    │  generate()  │   │ 发现/调用外部工具    │
                    │  memory      │   │ 结果融合           │
                    └──────────────┘   └───┬───────────────┘
                                           │ MCP 协议
                              ┌────────────┼────────────┐
                              │            │            │
                         ┌────▼───┐  ┌────▼───┐  ┌────▼───┐
                         │ 联网搜索│  │ 文件系统│  │ 数据库  │  ...
                         │ Brave/  │  │ 读写本地│  │ SQLite │
                         │ Tavily  │  │ 文件    │  │ /PG    │
                         └────────┘  └────────┘  └────────┘
```

### 核心逻辑流程

```
用户问："我们 Q3 营收目标是多少？和行业比怎么样"

1. QAEngine.retrieve("Q3 营收目标")
   → 本地文档找到：会议纪要提到"增长 30%"
   → 复杂度分类器：这个问题需要外部信息

2. MCPClientManager 决策：需要联网搜索
   → 调用 Brave Search MCP 搜 "2026 Q3 行业营收增速"
   → 返回：同类公司 Q3 增速 15-25%

3. 结果融合：
   → 本地结果 + 外部结果 → 去重 → 排序

4. generate() 生成答案：
   "你的文档显示目标是增长 30%，这高于行业平均 15-25%，属于激进目标。
    参考来源：[1]Q3战略会议纪要 [E1]行业基准报告2026"

5. record_interaction() 写入记忆：
   → 外部来源也写入情景记忆（source="mcp:brave-search"）
```

---

## 三、分阶段实施计划

### Phase 0：MCP Server 工具打磨（前置优化）

**目标**：让现有 MCP Server 从"能用"升级到"AI 友好"（只改 `src/mcp_server.py` 一个文件）

#### 0.1 结构化输出 + 工具描述重写（30 min）

**问题**：当前 3 个工具都返回纯 Markdown 文本，AI 客户端收到后需要自行解析 `[1] 📄 xxx | 📍 xxx | 相似度: 0.462` 这种格式，容易出错。

**方案**：`search_documents` 返回结构化 JSON，包含分块数组、置信度、来源页。工具描述按"给 AI 看的 API 文档"重写，包含使用场景和参数示例。

```python
@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> dict:
    """语义检索文档分块。当需要查找文档中的事实、概念或细节时使用。
    优先于 ask_knowledge_base 当：仅需要原文片段而非综合回答时。

    Args:
        query: 自然语言查询。越具体越好，例如"第3章讨论的Transformer架构"
              而非模糊的"Transformer"
        top_k: 返回数量 1-20，默认 5。需要更多上下文时调大。

    Returns:
        {"results": [{"rank": 1, "doc_name": "...", "page": 249,
                      "score": 0.462, "text": "..."}]}
    """
```

#### 0.2 流式回答（1-2 h）

**问题**：`ask_knowledge_base` 等 LLM 生成完才返回，AI 客户端等待 3-5 秒无反馈。

**方案**：复用 Streamlit 已有的 `generate_stream()`，通过 MCP 的 generator/yield 机制逐 token 返回。FastMCP 支持 async generator 作为 tool 返回类型。

```python
@mcp.tool()
async def ask_knowledge_base(question: str, method: str = "auto") -> AsyncGenerator[str, None]:
    """向知识库提问，流式返回答案。..."""
    async for token in engine.generate_stream(question, sources, method=method):
        yield token
```

#### 0.3 暴露检索方法 + 自动路由（30 min）

**问题**：当前硬编码 `method="direct"`，闲置了已有的 MQE/HyDE/复杂度分类器。

**方案**：
- `search_documents` 和 `ask_knowledge_base` 新增 `method` 参数（`"auto"` / `"direct"` / `"mqe"` / `"hyde"` / `"combined"`）
- 默认 `"auto"`：内部调用已有的 `classify_complexity()`（93% 准确率），简单问题走 Direct（零 Token），复杂问题走 HyDE（高精度）
- AI 客户端也可以显式指定方法覆盖自动判断

---

### Phase 1：MCP Client 基础设施（核心）

**目标**：能连接外部 MCP Server，发现和调用工具

#### 1.1 MCPClientManager 类

```python
# src/mcp_client/client_manager.py

class MCPClientManager:
    """管理一个或多个外部 MCP Server 连接"""

    def __init__(self, server_configs: list[MCPServerConfig]):
        self._servers: dict[str, ClientSession] = {}  # name → session
        self._tools: dict[str, ToolInfo] = {}          # tool_name → info
        self._configs = server_configs

    async def connect_all(self) -> None:
        """连接所有配置的外部 MCP Server，发现工具"""

    async def list_tools(self) -> list[ToolInfo]:
        """列出所有外部可用工具（去重 + 来源标注）"""

    async def call_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """调用指定外部工具"""

    async def search_external(self, query: str) -> list[ExternalResult]:
        """在已连接的外部搜索工具中执行搜索（联网搜索优先）"""

    async def close_all(self) -> None:
        """断开所有连接"""
```

#### 1.2 外部 MCP Server 配置

```python
# src/mcp_client/config.py

@dataclass
class MCPServerConfig:
    name: str                    # "brave-search"
    command: str                 # "npx"
    args: list[str]              # ["-y", "@anthropic/mcp-server-brave-search"]
    env: dict[str, str]          # {"BRAVE_API_KEY": "xxx"}
    enabled: bool = True
    category: str = "search"     # search / filesystem / database / ...

@dataclass
class MCPClientConfig:
    enabled: bool = False
    servers: list[MCPServerConfig] = field(default_factory=list)
    external_search_timeout: int = 10   # 外部搜索超时（秒）
    max_external_results: int = 3       # 最多融合几条外部结果
```

#### 1.3 配置来源

外部服务器列表从 `.env` 和 `.mcp.json` 双来源读取：

```bash
# .env 新增
MCP_CLIENT_ENABLED=true
MCP_EXTERNAL_SEARCH_SERVER=brave-search   # 指定默认搜索服务器
MCP_EXTERNAL_SEARCH_TIMEOUT=10
```

`.mcp.json`（已有文件，扩展）负责具体的 server 进程启动配置，`.env` 负责开关和参数。

#### 1.4 文件清单

| 文件 | 说明 |
|------|------|
| `src/mcp_client/__init__.py` | 包入口 |
| `src/mcp_client/client_manager.py` | 连接管理 + 工具发现/调用 |
| `src/mcp_client/config.py` | 配置数据类 |
| `src/mcp_client/models.py` | `ExternalResult` / `ToolInfo` / `ToolResult` DTO |

**改动量**：~300 行新代码，不碰现有文件

---

### Phase 2：智能路由决策

**目标**：AI 自动判断一个问题是否需要外部工具

#### 2.1 决策 Prompt

复用现有的 `LLMClient` + 复杂度分类器思路。新增一个轻量判断：

```
你是一个查询分析器。判断以下问题是否需要调用外部工具（联网搜索、文件系统等）：

问题："{question}"

规则：
- 涉及"最新"、"最近"、"现在"等时效性词 → 需要联网搜索
- 涉及行业对比、市场数据、外部标准 → 需要联网搜索
- 纯粹针对已上传文档的提问 → 不需要
- 概念解释类（文档中应该有）→ 不需要

返回 JSON: {"need_external": true/false, "reason": "...", "suggested_tools": ["web_search"]}
```

#### 2.2 与 QAEngine 集成

```python
# src/engine/qa_engine.py 新增方法（或直接在新模块实现）

async def retrieve_with_external(
    self, question: str, top_k: int = 5, method: str = "auto"
) -> tuple[list[SourceChunk], list[ExternalResult]]:
    """
    检索本地 + 外部，返回融合后的结果。
    内部自动判断是否需要外部搜索。
    """
    # 1. 本地检索（不变）
    local = self.retrieve(question, method=method, top_k=top_k)

    # 2. 判断是否需要外部搜索
    if self.mcp_client and self._need_external(question):
        external = await self.mcp_client.search_external(question)
    else:
        external = []

    return local, external
```

#### 2.3 文件清单

| 文件 | 说明 |
|------|------|
| `src/mcp_client/router.py` | 查询路由决策器（LLM 判断是否需外部工具）|
| `src/mcp_client/__init__.py` | 更新导出 |

**改动量**：~150 行新代码，`qa_engine.py` 新增 1 个方法（~30 行）

---

### Phase 3：结果融合与引用

**目标**：本地 + 外部结果统一格式，区分引用来源

#### 3.1 融合策略

```python
# src/mcp_client/fusion.py

def merge_results(
    local: list[SourceChunk],
    external: list[ExternalResult],
    local_weight: float = 0.6,
    external_weight: float = 0.4,
    max_total: int = 8,
) -> list[MergedResult]:
    """
    本地和外部结果融合：
    1. 本地结果保留前 max_total * local_weight 条（按分数）
    2. 外部结果保留前 max_total * external_weight 条（按分数）
    3. 去重：外部结果与本地结果做 Jaccard 去重（复用已有 dedup 模块）
    4. 交错排列：本地、外部交替，避免外部信息堆在一起被 LLM 忽略
    """
```

#### 3.2 引用格式扩展

```
本地文档引用：  [1] Hello-Agents.pdf 第249页
外部搜索结果：  [E1] brave-search: 行业基准报告2026
外部文件系统：  [F1] filesystem: /home/user/notes/q3.md
```

在 `generation/prompt_templates.py` 中新增 `RAG_QA_WITH_EXTERNAL_SYSTEM` 模板。

#### 3.3 文件清单

| 文件 | 说明 |
|------|------|
| `src/mcp_client/fusion.py` | 本地+外部结果融合 + 去重 |
| `src/generation/prompt_templates.py` | 新增带外部来源的 System Prompt |
| `src/generation/citation_formatter.py` | 新增 [E1] / [F1] 外部引用格式 |

**改动量**：~200 行新代码 + 改 2 个现有文件

---

### Phase 4：MCP Server 工具升级

**目标**：让外部 AI 也能通过 MCP 工具触发 DocMind 的外部搜索能力

#### 4.1 新增 MCP 工具

在 `src/mcp_server.py` 中新增：

```python
@mcp.tool()
async def search_with_web(query: str, top_k: int = 5) -> str:
    """同时搜索本地知识库和互联网，返回融合结果。
    适用于需要最新信息或行业对比的问题。
    """
    engine = _get_engine()
    local, external = await engine.retrieve_with_external(query, top_k=top_k)
    # 格式化返回...

@mcp.tool()
def get_available_tools() -> str:
    """列出 DocMind 当前可用的所有工具（本地 + 已连接的外部 MCP 工具）"""
    ...
```

#### 4.2 已有工具升级

`search_documents` 增加 `include_external: bool = False` 参数。

**改动量**：`src/mcp_server.py` 新增 2 个工具（~80 行）

---

### Phase 5：记忆整合

**目标**：外部搜索结果也写入记忆系统

```python
# 外部结果写入情景记忆
memory_manager.record_external_search(
    query=question,
    source="mcp:brave-search",
    results=external_results,
)

# 外部结果中的关键实体 → 写入语义记忆
memory_manager.extract_concepts_from_external(external_results)
```

**改动量**：`src/memory/` 新增少量方法（~80 行）

---

## 四、完整文件清单

```
新增文件：
  src/mcp_client/
  ├── __init__.py              # 包入口，导出 MCPClientManager
  ├── client_manager.py        # MCP Client 核心：连接/发现/调用
  ├── config.py                # MCPClientConfig 数据类
  ├── models.py                # DTO：ExternalResult / ToolInfo / ToolResult
  ├── router.py                # 查询路由：LLM 判断是否需外部工具
  └── fusion.py                # 结果融合：本地+外部 去重 排序

修改文件：
  src/mcp_server.py            # 新增 search_with_web / get_available_tools
  src/engine/qa_engine.py       # 新增 retrieve_with_external()
  src/generation/prompt_templates.py  # 新增带外部来源的 Prompt
  src/generation/citation_formatter.py # 新增 [E]/[F] 外部引用格式
  src/memory/memory_manager.py  # 新增 record_external_search()
  .env.example                  # 新增 MCP_CLIENT_* 变量
  .mcp.json                     # 示例：连接外部 MCP 的配置模板
  requirements.txt              # 确认 mcp 包已包含（✅ 已加）

测试文件：
  tests/test_mcp_client/
  ├── test_client_manager.py   # Mock 外部 MCP Server 的单元测试
  ├── test_router.py           # 路由决策准确性测试
  └── test_fusion.py           # 融合算法正确性测试
```

---

## 五、推荐的外部 MCP Server

按优先级排列：

| 优先级 | 外部工具 | 价值 | 实现难度 |
|--------|---------|------|---------|
| 🥇 | **Brave Search**（联网搜索）| 解决时效性问题，价值最大 | 低（官方 MCP Server 现成）|
| 🥈 | **Filesystem**（本地文件读写）| 读写笔记、导出报告 | 低（官方 MCP Server 现成）|
| 🥉 | **SQLite**（数据库查询）| 直接查 metadata.db | 中（需自建或找社区实现）|
| 4 | **Tavily Search**（联网搜索备选）| 中文化更好 | 低 |
| 5 | **GitHub**（查 issues/PRs）| 开发场景有价值 | 低（官方现成）|
| 6 | **Postgres**（企业数据库）| 企业场景 | 中 |

起步只需要 Brave Search 一个就够了，就能跑通整个链路。

---

## 六、环境变量扩展

```bash
# .env 新增
# ── MCP Client ─────────────────────────────────
MCP_CLIENT_ENABLED=true                          # 开关
MCP_EXTERNAL_SEARCH_SERVER=brave-search          # 默认搜索服务器名
MCP_EXTERNAL_SEARCH_TIMEOUT=10                   # 外部搜索超时（秒）
MCP_MAX_EXTERNAL_RESULTS=3                       # 外部结果最多融合几条
MCP_LOCAL_WEIGHT=0.6                             # 融合时本地结果权重
```

---

## 七、关键设计决策

| 决策 | 说明 |
|------|------|
| **Client 启动时机** | 懒加载：QAEngine 初始化时不连，`retrieve_with_external()` 首次调用时才 `connect_all()`。避免每次冷启动都连外部服务 |
| **异步模型** | MCP Python SDK 基于 asyncio，`client_manager` 全部 async。但 QAEngine 目前是同步的，需要在 `retrieve_with_external()` 中用 `asyncio.run()` 桥接 |
| **外部结果去重** | 复用已有 `src/retrieval/dedup.py` 的 bigram Jaccard 去重，阈值独立配置 |
| **降级策略** | 外部 MCP 连接失败/超时/返回空 → 自动回退纯本地模式，不影响核心问答 |
| **费用控制** | 联网搜索每次消耗 API 调用费。`router.py` 的 LLM 判断是总闸——判断不通过不调外部搜索 |
| **记忆隔离** | 外部结果记入情景记忆时 `source="mcp:<server_name>"`，可追溯、可过滤、可单独删除 |

---

## 八、风险与应对

| 风险 | 应对 |
|------|------|
| MCP Python SDK 的 Client 端 API 不稳定（目前还在演进）| Phase 1 先做一个最小可用的 Client，封装好接口，后续 SDK 变化只改 `client_manager.py` |
| 外部搜索增加延迟（联网搜索 + LLM 判断 合计 2-5 秒）| 流式返回 + 超时配置 + 用户可见"正在搜索外部信息..." |
| LLM 路由判断不准（该搜不搜、不该搜乱搜）| 先用规则辅助（关键词匹配时效性词），再逐步过渡到 LLM 判断 |
| Brave Search API 有免费额度限制 | 默认不开启，用户自行配置 API Key |

---

## 九、执行顺序

```
Phase 0 (Server 打磨)     ← 今天做，只改一个文件
        ↓
Phase 1 (基础设施)  →  Phase 2 (路由决策)
        ↓                      ↓
   MCPClientManager      router.py + QAEngine
   能连外部 Server        自动判断是否需外部
   （独立可测）            （依赖 Phase 1）
                                ↓
                          Phase 3 (结果融合)
                          fusion.py + 引用格式
                          （依赖 Phase 1+2）
                                ↓
                          Phase 4 (Server 工具升级)
                          MCP Server 新工具
                          （依赖 Phase 1-3）
                                ↓
                          Phase 5 (记忆整合)
                          外部结果写入记忆
                          （可独立做，最后收尾）
```

**预计总改动量**：~900 行新代码 + ~150 行修改现有代码

---

## 十、验证方式

每个 Phase 完成后：

| Phase | 验证 |
|-------|------|
| 1 | 写一个 `scripts/test_mcp_client.py`，连一个测试用的 Echo MCP Server，验证 connect → list_tools → call_tool 全链路 |
| 2 | 用已有评测数据集的路由准确率（类似复杂度分类器的 93% 测试方法） |
| 3 | 单元测试 `test_fusion.py` 覆盖去重/排序/边界 |
| 4 | 在 Claude Code 中实际测试 `search_with_web` 工具 |
| 5 | 查情景记忆/语义记忆确认外部来源已写入 |
