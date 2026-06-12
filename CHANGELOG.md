# 📋 更新日志

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
