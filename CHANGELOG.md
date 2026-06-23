# 📋 更新日志

## 2026-06-23

### 🧠 MCP 原生智能体 Phase 0-4 完整交付

> **DocMind = MCP Server（对外提供知识库） + MCP Client（对内调用外部工具）**\n> 今日完成 5 个 Phase，将 DocMind 从一个简单的 MCP Server 升级为完整的 MCP 原生智能体。

---

### Phase 0：MCP Server 工具打磨

- ✅ **结构化输出**：`search_documents` / `list_knowledge_base` 返回结构化 JSON 替代纯文本 Markdown，AI 客户端无需自行解析。
- ✅ **流式回答**：`ask_knowledge_base` 改为 async generator，逐 token 流式输出，体验对齐 Streamlit。
- ✅ **自动路由**：所有工具新增 `method` 参数，默认 `"auto"` 模式利用已有动态路由（93% 准确率），AI 客户端也可显式指定。
- ✅ **工具描述重写**：3 个工具的 description 改为面向 AI 的 API 文档风格，含使用场景建议和参数示例。

### Phase 1：MCP Client 基础设施

- ✅ **`src/mcp_client/` 包**：新建 4 个模块（config / models / client_manager / router）+ `__init__.py`。
- ✅ **`MCPClientManager`**：管理外部 MCP Server 的 stdio 连接，支持 `connect_all()` → `list_tools()` → `call_tool()` → `close_all()` 全生命周期。
- ✅ **Tavily 联网搜索**：接入 `tavily-mcp` 官方 MCP Server，5 个工具（search / extract / crawl / map / research），免费 1000 次/月。
- ✅ **自动配置**：从 `.env` 读取 `TAVILY_API_KEY` 自动加载 Tavily，无需手动配置。
- ✅ **测试**：`scripts/test_tavily_search.py` 验证全链路，5 条搜索结果完美解析。

### Phase 2：智能路由决策

- ✅ **`ExternalRouter`**：双模式路由（规则匹配 + LLM 回退），规则模式零 Token 消耗。
- ✅ **路由准确率 100%**（9/9 测试集）：时效性关键词（最新/2026/今天）→ 联网搜索；文档内问题（第X章/根据文档）→ 纯本地；趋势/对比/新闻类 → 联网搜索。
- ✅ **`QAEngine.retrieve_with_external()`**：本地检索 + 可选外部搜索，自动路由判断。内部桥接异步 MCP Client 到同步 QAEngine。

### Phase 3：结果融合与引用

- ✅ **`merge_results()`**：本地文档 + 外部搜索结果融合，bigram Jaccard 去重，交错排列（L1→E1→L2→E2）。
- ✅ **区分引用**：本地来源 `[N]` + 外部来源 `[E1][E2]`，答案中一眼分辨信息来源。
- ✅ **Prompt 扩展**：新增 `RAG_QA_WITH_EXTERNAL_SYSTEM` 模板，引导 LLM 综合本地和网络信息。
- ✅ **`CitationFormatter.format_with_external()`**：解析 `[N]` 和 `[EN]` 双重引用标记。
- ✅ **`QAEngine.generate_with_external()`**：融合问答一站式入口——路由→检索→融合→生成→引用格式化。

### Phase 4：MCP Server 工具升级

- ✅ **新增 `search_with_web` 工具**：同时搜索本地知识库和互联网，返回融合答案（含 `[N]`/`[E]` 引用 + 外部链接）。
- ✅ **新增 `get_available_tools` 工具**：AI 客户端自动发现 DocMind 能力清单（本地检索 / 联网搜索 / 文档管理 / 知识图谱）。
- ✅ **`ask_knowledge_base` 升级**：新增 `include_external=True` 参数，AI 可自行选择是否开启联网搜索。
- ✅ **MCP 工具总数**：3 → 5。全部在 **Claude Code** 和 **VS Code Copilot** 验证通过。

---

### 📊 新增代码统计

| 类别 | 文件数 | 行数 |
|------|:-----:|:----:|
| `src/mcp_client/`（新建包） | 6 | ~750 |
| `src/mcp_server.py`（重写） | 1 | ~350 |
| `mcp_server.py`（同步） | 1 | ~150 |
| `src/engine/qa_engine.py`（新增方法） | 1 | +180 |
| `src/generation/`（prompt + citation） | 2 | +105 |
| `scripts/`（测试脚本） | 4 | ~220 |
| `docs/mcp-agent-plan.md`（方案文档） | 1 | ~500 |
| **合计** | **16** | **~2,255** |

### 📝 文档更新

- 📝 `README.md`：项目定位升级为「MCP 原生知识中枢」；新增 MCP 原生智能体功能区块（7 项）；MCP 协议集成段落全标记为 ✅。
- 📝 `CHANGELOG.md`：本文档。
- 📝 `docs/mcp-agent-plan.md`：完整技术方案（Phase 0-5 + 架构图 + 配置指南）。
- 📝 `.env.example`：新增 MCP Client 配置注释。

### 🔧 配置与依赖

- 🔧 `requirements.txt` / `requirements-dev.txt`：新增 `mcp>=1.0.0`。
- 🔧 `.env`：新增 `TAVILY_API_KEY` + `MCP_CLIENT_ENABLED`。
- 🔧 `pyproject.toml`：mypy `python_version` 3.10 → 3.12（修复 CI numpy 类型存根兼容）。
- 🔧 `.gitignore`：新增 `data/test_chroma/`。

---

## 2026-06-22

### 🔌 MCP 协议集成

- ✅ **DocMind MCP Server**：新增 `src/mcp_server.py`（主入口）和 `mcp_server.py`（备用入口），基于 `FastMCP` 将 DocMind RAG 能力以 MCP 协议暴露。
- ✅ **三个标准工具**：
  - `search_documents(query, top_k)` — 语义检索文档分块，返回来源引用
  - `ask_knowledge_base(question)` — 检索 + LLM 生成带引用答案
  - `list_knowledge_base()` — 列出知识库所有文档
- ✅ **多客户端支持**：
  - **Claude Code**：`.mcp.json` 配置，即插即用，已通过本会话验证
  - **VS Code Copilot**：`.vscode/mcp.json` 配置，Reload Window 后生效，已验证通过
  - **Claude Desktop / Cursor**：配置示例写在 `src/mcp_server.py` 头注释中
- ✅ **用户隔离**：通过 `DOCMIND_USER_ID` 环境变量指定用户，与 Streamlit 共享同一套向量库和记忆。
- 🔧 **VS Code 踩坑**：`.vscode/mcp.json` 不会被工作区设置覆盖，正常编写 JSON 即可。

### 📝 文档更新

- 📝 `README.md` 更新 MCP 协议集成段落（`🚧` → `✅`）、项目结构新增 MCP 文件。
- 📝 `scripts/test_mcp.py` MCP Server 本地验证脚本（无需启动 MCP，直接测 QAEngine）。

---

## 2026-06-18

### 🔀 动态路由（查询复杂度自适应）
- ✅ **复杂度分类器**：`LLMClient` 新增 `classify_complexity()`，单次 LLM 调用（~200 tokens）判断问题复杂度。
- ✅ **自动路由**：`QAEngine.retrieve()` 检索前分类——simple 走 Direct（零 Token），complex 走 HyDE（高精度），覆盖用户手动选择。
- 📈 15 题测试准确率 93%，Token 节省 53%。

### 🎯 检索结果优化
- ✅ **最低相似度过滤**：`RetrievalConfig` 新增 `min_similarity`（默认 0.2），检索后丢弃无关结果，避免 LLM 基于不相关内容硬编。
- ✅ **无上下文回答优化**：`RAG_QA_NO_CONTEXT_SYSTEM` 改为自然回答模式，不再建议上传文档，直接基于知识正常回答。

📝 `eval_data.md` 新增动态路由评测报告；`README.md` 更新关键改进数据表和后续优化方向。

## 2026-06-16

### 🧠 对话摘要压缩
- ✅ **工作记忆升级**：`WorkingMemory` 新增 `_summary` 字段，超过 10 条自动触发 LLM 压缩——旧对话压缩为结构化摘要（关键问题、已确认事实、用户偏好），保留最近 3 轮原始对话。
- ✅ **增量更新**：已有摘要时在基础上增量合并新增对话，避免重复压缩。
- ✅ **生成器注入**：`AnswerGenerator.generate()` / `generate_stream()` 新增 `working_context` 参数，自动将摘要 + 近期对话注入 system prompt。
- ✅ **自动化触发**：`QAEngine.record_interaction()` 后自动调用 `compress_working_memory()`。
- 📈 **效果**：20 轮上下文从 1,992 tokens 降至 497（**-75%**），50 轮预计节省 **89%**。10 轮以内无收益（摘要开销 ≈ 原文）。
- 📝 `eval_data.md` 新增对话摘要压缩评测报告。

## 2026-06-15

### ⚡ 检索性能修复
- 🐛 **修复 Streamlit 检索延迟**：`base_retriever.retrieve()` 在 Streamlit 的 asyncio 环境中走了 `ThreadPoolExecutor + asyncio.run()` 绕路，导致 Direct 检索从 0.2s 膨胀到 30s+。改为纯同步路径 `_retrieve_sync()`，Direct 检索恢复 0.2s，MQE+HyDE 约 3-5s。
- ✅ 四个检索器（Direct / MQE / HyDE / Combined）均实现 `_retrieve_sync()` 方法。

### 🧹 检索结果去重
- ✅ **语义去重模块**：新增 `src/retrieval/dedup.py`，用字符 bigram Jaccard 相似度去重，无需嵌入 API 调用，O(N²) 但 N≤20 实测 < 1ms。
- ✅ **配置**：`RetrievalConfig` 新增 `use_dedup` / `dedup_threshold` + 环境变量 `RETRIEVAL_DEDUP`（默认 true）/ `RETRIEVAL_DEDUP_THRESHOLD`（默认 0.65）。
- ✅ **集成**：`qa_engine.retrieve()` 在检索后、重排序前自动去重。

### 💰 在线 Token 成本看板
- ✅ **Token 累加器**：`LLMClient` 新增 `accumulated_prompt` / `accumulated_completion` 累加器 + `total_token_usage` 属性 + `reset_token_counters()`。
- ✅ **持久化**：`MetadataStore` 新增 `token_usage` 表 + `record_token_usage()` / `get_token_stats()` 方法。
- ✅ **监控页**：新增「💰 Token 成本分析」区域——总 Token、输入/输出比、按方法柱状图、预估费用（¥）、近期记录。

### ⏱️ 文档加载计时
- ✅ **实时计时**：文档管理页加载时后台线程执行摄入，主线程每 0.3s 轮询更新进度条计时器（秒数实时跳动）。
- ✅ **分步耗时**：摄入管线记录 5 步骤耗时（加载/分块/嵌入/入库/元数据），`IngestResult` 传递到 UI。
- ✅ **持久记录**：`documents` 表新增 `total_sec` 列，加载耗时永久存储，文档列表中每条记录旁显示 ⏱ 标记。
- ✅ **文件列表自动清空**：加载完成后 `file_uploader` 动态 key 递增，强制重置为空。

### 🧪 单元测试
- ✅ **173 个测试用例**（之前 31 个），新增 6 个测试文件：
  - `test_citation_formatter.py`（31 用例）— 8 种格式位置描述 + 引用解析 + 编号重映射
  - `test_prompt_templates.py`（11 用例）— 全部 7 个模板占位符验证
  - `test_models.py`（18 用例）— DTO 序列化/反序列化往返
  - `test_fusion_internals.py`（10 用例）— MinMax 归一化边界 + 加权合并
  - `test_chunk_config.py`（16 用例）— 8 种格式预设 + 回退逻辑
  - `test_llm_client.py`（17 用例）— 概念提取解析 + 查询变体 JSON 回退
  - `test_qa_engine_normalize_method.py`（20 用例）— 中英文/大小写/分隔符全路径
  - `test_dedup.py`（19 用例）— bigram/Jaccard/去重边界

### 🔧 监控页改进
- ✅ 反馈记录列表每条加 🗑 单条删除按钮
- ✅ Token 记录列表每条加 🗑 单条删除按钮
- ✅ 新增「🧹 清理异常数据」批量删除（按延迟阈值）
- ✅ 概念提取改为后台线程，不阻塞页面刷新和 👍/👎 按钮

### 🩺 诊断脚本
- ✅ `scripts/diagnose_retrieval.py` — 分别测量嵌入 API / ChromaDB 搜索 / 端到端检索耗时
- ✅ `scripts/diagnose_llm.py` — 分别测量 MQE 变体 / HyDE 假设 / 答案生成 三类 LLM 调用耗时

### 📝 文档
- 📝 更新 `.env.example`：新增 `RETRIEVAL_DEDUP` / `RETRIEVAL_DEDUP_THRESHOLD`
- 📝 更新 `README.md`：去重压缩 ✅、在线 Token 看板 ✅、单元测试覆盖 ✅、加载计时、检索去重、监控改进

## 2026-06-14

### 🔍 重排序（Cross-Encoder Reranker）
- ✅ **重排序模块**：新增 `src/retrieval/reranker.py`，集成 `BAAI/bge-reranker-v2-m3` 交叉编码器，对向量粗筛结果逐对精排。模型通过 ModelScope 下载到本地（2.1GB）。
- ✅ **QAEngine 集成**：`retrieve()` 自动在检索后调用重排序（`.env` 中 `USE_RERANKER=true` 开启），粗筛取 top-20 → 精排 → 返回 top-K。
- ✅ **配置**：`RetrievalConfig` 新增 `use_reranker` / `reranker_top_k` 字段 + 环境变量 `USE_RERANKER` / `RERANKER_TOP_K`。
- 📈 **效果**：Direct MRR 0.58→0.75（+29%），R@5 35.6%→48.9%（+13.3pp），是目前单项改进中收益最大的。
- ⚠️ **延迟代价**：2.1GB 模型 CPU 推理，每次检索增加约 20-24s。后续可换 GPU 或轻量 bge-reranker-base。

### 🧪 RAGAS 生成评测修复

- 🐛 **修复 RAGAS 0.4.x 兼容性**：langchain vertexai 导入顺序修正、旧式 `ragas.metrics` 模块级导入、`EvaluationDataset.from_list()` 适配、用 `LlamaIndexEmbeddingsWrapper` 替代 langchain 嵌入层。
- 🐛 **修复 Faithfulness NaN 聚合**：新增 `_nanmean()` 跳过 NaN 后求均值，避免聚合值被 NaN 传播。
- 📝 `eval_data.md` 新增第四至第八轮评测报告（Token 分块 1024、语义分块、RAGAS 生成质量、4 方法基线、重排序前后对比）。

### 📝 README 与文档
- ✅ 新增 3 个后续优化方向：**多模态输入**（Whisper 语音 + 图片）、**动态路由 RAG**（查询复杂度自适应）、**Agent 记忆升级**（对话摘要压缩 + 独立 MemoryAgent）。
- ✅ RAGAS 生成阶段评测标记为已完成。

### 🔧 配置与代码
- 🔧 `.env` 嵌入模型切回 `text-embedding-v4` API、分块默认值调整为 768、新增语义分块 + 重排序环境变量。
- 🔧 `requirements.txt` 新增 `llama-index-embeddings-openai`。

## 2026-06-13

### ✂️ 分块策略重构
- ✅ **语义分块**：TextChunker 集成 LlamaIndex `SemanticSplitterNodeParser`，按句子嵌入相似度在话题边界切分，替代纯 token 计数分块。超大语义块自动回退 SentenceSplitter。
- ✅ **Markdown 标题分块**：`.md` 文件改为按 `##` / `###` 标题切分，`#` 作文档标题，`####`+ 保留在父级段落内。
- ✅ **自适应分块预设**：`ChunkConfig` 新增 `chunk_presets` 字典，PDF/PPT/Web/Docx/MD/TXT/CSV/XLSX 八种格式各有独立 (chunk_size, chunk_overlap)，`IngestPipeline` 按文档格式自动选择。
- ✅ **小碎片合并**：新增 `min_chunk_tokens` 阈值，低于该值的碎片自动合并到前一块，避免嵌入质量差。
- 🔧 移除文档管理页面中无效的分块参数 UI 输入框（实际从未接线），改用 `.env` 统一配置。

## 2026-06-12

### 🔬 RAG 评测体系
- ✅ **Token 成本分析**：`LLMClient` 新增 `last_usage` 属性追踪每次 API 调用的 token 用量；`InstrumentedLLMClient` 累计统计 prompt/completion tokens；评测报告表格和 JSON 输出均包含 Token 汇总。
- ✅ **RAGAS 生成阶段评测**：`evaluation_runner.py` 新增 `--with-generation` 参数，可选集成 RAGAS 计算 Faithfulness 和 Answer Relevancy；适配 RAGAS 0.4.x API + langchain 兼容补丁。
- ✅ **用户隔离修复**：评测 CLI 新增 `-u/--user-id` 参数，解决 ChromaDB 用户过滤导致的全 0% 召回问题。
- ✅ **本地嵌入支持**：切换为 `BAAI/bge-small-zh-v1.5`（512 维），零费用、零网络延迟。
- 🐛 **修复**：`CombinedRetriever` 并行检索异常时自动降级为分别执行，避免静默返回空结果。
- 🐛 **修复**：MQE/HyDE 的 token 计数遗漏（`generate_query_variants` 等内部调用未累加）。
- 📝 **文档**：`eval_data.md` 新增第三轮评测报告（本地嵌入 + Token 成本分析）；README 更新环境变量、项目结构、评测命令、模型名称。

## 2026-06-11

### ✨ 新增
- 🐳 **Docker 容器化**：新增 `Dockerfile`、`docker-compose.yml`、`.dockerignore`，`docker-compose up -d` 一键启动 Streamlit + Neo4j 全栈服务。
- ☁️ **Railway 云端部署**：项目已部署上线，README 新增部署文档。

### 🔧 CI/CD
- ✅ **GitHub Actions**：新增 `.github/workflows/ci.yml`，Push/PR 自动跑 Ruff + mypy + pytest。
- ✅ **代码质量配置**：新增 `pyproject.toml`、`requirements-dev.txt`。

### ❌ 删除
- 🔄 **移除"重建记忆"功能**：存在 bug，已从回顾页面移除。
