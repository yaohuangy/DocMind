# Changelog

本文档记录项目开发过程中遇到的问题及解决方案。

---

## 2026-06-05

### 🔴 app.py 启动报错 `NameError: name 'QAEngine' is not defined`

**现象**：`streamlit run app.py` 后立即崩溃，提示 `NameError: name 'QAEngine' is not defined`。

**原因**：创建 `src/engine/session.py` 把 `get_engine()` 提取出去后，`app.py` 里保留了旧的那份 `get_engine` 函数（使用了 `QAEngine` 类型注解），但对应的 `from src.engine.qa_engine import QAEngine` 已被替换为 `from src.engine.session import get_engine`，导致类型注解引用了未导入的符号。

**解决**：删除 `app.py` 中重复的 `get_engine()` 函数，统一使用 `session.py` 导出的版本。同时清理了不再需要的 `from src.core.config import load_config` 导入。

**涉及文件**：`app.py`

---

### 🔴 DashScope Embedding 报错 `batch size should not be larger than 10`

**现象**：摄入文档时抛出 `BadRequestError: batch size is invalid, it should not be larger than 10`。

**原因**：DashScope 的 `text-embedding-v4` 每次请求最多接受 **10 条**文本，而 `.env.example` 和实际 `.env` 中 `EMBEDDING_BATCH_SIZE=32` 超过了这个限制。当 PDF 分块超过 10 个时，第一批 32 条发送到 DashScope 直接被拒绝。

**解决**：将 `.env.example` 和 `.env` 中的 `EMBEDDING_BATCH_SIZE` 从 32 改为 10。

**涉及文件**：`.env.example`、`.env`

**备注**：不同平台的批量上限不同：

| 平台 | 每次最大条数 |
|------|-------------|
| DashScope text-embedding-v4 | 10 |
| OpenAI text-embedding-3-small | 2048 |
| SiliconFlow bge-large-zh | 32 |

后续切换嵌入服务时需同步调整此参数。

---

### 🔴 ChromaDB 报错 `DuplicateIDError: Expected IDs to be unique`

**现象**：摄入 PDF 文档时，向量存入 ChromaDB 时抛出 `DuplicateIDError: Expected IDs to be unique, found duplicates of: 03f59f160dbbc395`。

**原因**：分块 ID 由 `sha256(source:chunk_index)[:16]` 生成。PDF 解析器按页输出 `LlamaDocument`，**每页的 `chunk_index` 都从 0 开始**。当 PDF 有多页且多页都有有效文本时，不同页面的第 0 个分块产生相同的 ID `sha256(source:0)[:16]`，触发 ChromaDB 唯一性冲突。

**解决**：

1. 将 ID 生成逻辑从使用 `chunk_index`（metadata 中的分页级序号）改为使用**全局枚举序号 `i`**（`enumerate(chunks)`），确保跨页面唯一。

2. 在摄入前增加**幂等处理**——先调用 `delete_by_doc_id(doc_id)` 清理旧分块，避免重复摄入同一文档时残留数据冲突。

**涉及文件**：`src/ingestion/ingest_pipeline.py`

---

### 🟡 Streamlit Pages 导入 app 导致 UI 重复渲染

**现象**：pages 目录下的子页面通过 `from app import engine` 导入引擎时，`app.py` 中的全部 Streamlit UI 代码（`st.set_page_config()`、侧边栏、标题等）也会被执行，导致元素重复渲染或 `st.set_page_config()` 被多次调用。

**原因**：Python 的 `import` 会执行目标模块的全部顶层代码。`app.py` 中包含大量 Streamlit 渲染语句（`st.sidebar`、`st.title`、`st.radio` 等），导入时这些全部在子页面的上下文中执行。

**解决**：

1. 创建独立模块 `src/engine/session.py`，将 `@st.cache_resource` 装饰的 `get_engine()` 函数放在其中。
2. `app.py` 和所有子页面统一通过 `from src.engine.session import get_engine` 获取引擎实例。
3. `app.py` 中的 `st.set_page_config()` 增加 `if __name__ == "__main__":` 守卫。

**涉及文件**：`src/engine/session.py`（新建）、`app.py`、`pages/1_📄_文档管理.py`、`pages/2_📝_笔记.py`、`pages/3_🧠_回顾.py`

---

### 🟡 Neo4j 不可用导致 FAQ 功能无法启动

**现象**：未安装/启动 Neo4j 时，QAEngine 初始化阶段直接崩溃，整个应用无法使用。

**原因**：原始设计中 `SemanticMemory.__init__` 尝试连接 Neo4j，连接失败时异常向上传播至 `QAEngine.__init__`，导致引擎创建失败。

**解决**：在 `QAEngine.__init__` 中增加优雅降级逻辑：

1. `MemoryManager` 初始化包裹在 try/except 中，失败时 `self.memory = None`。
2. Neo4j 连接单独尝试，失败时设置 `_neo4j_available = False` 并打印 warning 日志。
3. 所有记忆相关方法（`record_interaction`、`search_memory`、`get_review_data` 等）增加 `if self.memory is None` 守卫，安全返回空值或跳过操作。
4. `get_stats()` 中 Neo4j 连接状态通过 `Neo4j可用` 字段暴露给前端侧边栏。

**涉及文件**：`src/engine/qa_engine.py`

---

### 🟢 模型切换为免费方案

**需求**：希望使用免费模型替代 OpenAI，降低成本。

**方案**：切换到阿里 DashScope（灵积），注册即送免费额度。

| 组件 | 模型 | 免费额度 |
|------|------|---------|
| LLM | `qwen-turbo` | 200 万 token/天 |
| Embedding | `text-embedding-v4` | 免费期内够用 |

DashScope 提供 OpenAI 兼容接口，只需改 `.env` 中三行配置即可，无需修改代码：

```bash
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-turbo
EMBEDDING_MODEL=text-embedding-v4
```

**涉及文件**：`.env.example`

**注意事项**：

- `text-embedding-v4` 批量上限为 10（见上方相关 issue）
- `text-embedding-v3` 维度和 v4 不同，切换后需重新摄入文档

---

## 已知注意事项

| 事项 | 说明 |
|------|------|
| Neo4j 可选 | 不装也能用，语义记忆自动降级，问答和笔记不受影响 |
| Embedding 批量上限 | 不同平台不同，切换时务必检查 |
| 模型维度不兼容 | 切换嵌入模型后旧向量失效，需重新摄入文档 |
| Streamlit 导入模式 | 子页面不要 `from app import xxx`，走 `src.engine.session` |

---

## 2026-06-05 检索评测报告

### 评测配置

| 参数 | 值 |
|------|---|
| 问题数量 | 30 |
| Top-K | 10 |
| LLM | qwen-turbo (DashScope) |
| Embedding | text-embedding-v4 (DashScope) |
| 评测时间 | 2026-06-05T09:30 |
| 数据集 | dataset_20260605_092545.json |

### 评测结果

| Method | R@5 | R@10 | P@5 | P@10 | MRR | NDCG@10 | Avg Lat | P50 | P95 |
|--------|------|------|------|------|------|----------|---------|------|------|
| 直接检索 | 86.7% | 96.7% | 17.3% | 9.7% | 0.76 | 0.81 | 0.25s | 0.22s | 0.33s |
| MQE | 83.3% | 96.7% | 16.7% | 9.7% | 0.72 | 0.78 | 1.22s | 1.18s | 1.53s |
| HyDE | 86.7% | 96.7% | 17.3% | 9.7% | 0.71 | 0.77 | 1.91s | 1.96s | 2.28s |
| MQE+HyDE | 86.7% | 96.7% | 17.3% | 9.7% | 0.74 | 0.80 | 1.88s | 1.83s | 2.45s |

### 结论

- **召回率持平**：30 个问题中，4 种方法都几乎找到了唯一的 Ground Truth 分块（Recall@10 = 96.7%，30 题中 29 题命中）。原因是每道题只有一个 GT 分块，且问题由 LLM 从该分块生成，原文语义高度匹配，向量检索本身已足够精准。
- **Direct 最快**：0.25s（无 LLM 调用），MQE 慢 5 倍（变体生成），HyDE/MQE+HyDE 慢 7-8 倍（假设答案生成长度远大于变体）。
- **MRR 有差异**：虽然都命中了，但 Direct 的 MRR 最高（0.76），GT 分块的排名最靠前。MQE/HyDE 的查询变体和假设答案引入了噪声，反而把 GT 分块排后了一点。
- **Precision@5 低**：每道题只有 1 个 GT 分块，检索返回 5 个，所以 Precision@5 恒为 ~17%（≈ 1/5.8）。属于正常现象，不影响结论。

### 改进方向（如果要有区分度）

1. **扩展 GT**：每个问题关联多个相关的同文档分块（同一文档的相邻段落），这样 Recall@10 不会都是 100%
2. **跨文档问题**：生成需要综合多个文档才能回答的问题，Direct 单路检索将难以命中
3. **对抗问题**：用不同措辞描述同一个概念，拉开 MQE（多角度重述）和 Direct 的差距

### 指标说明

| 指标 | 全称 | 含义 | 好值 |
|------|------|------|------|
| **R@5** | Recall@5 | 前 5 个结果中命中了多少 GT 分块 | > 0.8 |
| **R@10** | Recall@10 | 前 10 个结果中命中了多少 GT 分块 | > 0.9 |
| **P@5** | Precision@5 | 前 5 个结果中有几个是对的 | 看 GT 密度 |
| **P@10** | Precision@10 | 前 10 个结果中有几个是对的 | 看 GT 密度 |
| **MRR** | Mean Reciprocal Rank | 第一个命中的 GT 排第几位（1/rank） | > 0.7 |
| **NDCG@10** | Normalized Discounted Cumulative Gain | 考虑排名质量的命中评分 | > 0.8 |
| **Avg Lat** | Average Latency | 平均每次检索耗时 | < 1s |
| **P50** | P50 Latency | 50% 的请求延迟低于此值（中位数） | 接近 Avg |
| **P95** | P95 Latency | 95% 的请求延迟低于此值（尾延迟） | < 3× P50 |

---

## 2026-06-05 检索评测报告（第二轮：增强评测集）

### 评测配置

| 参数 | 第一轮（原始） | 第二轮（增强） |
|------|-------------|-------------|
| 问题数量 | 30 | 30 |
| Top-K | 10 | 10 |
| 问题生成方式 | 分块原文→LLM直接生成 | 生成后LLM二次改写（换术语、换句式） |
| GT 分块 | 每个问题 1 个 | 每个问题 3 个（自身 + 前后相邻分块） |
| 评测时间 | 2026-06-05T09:30 | 2026-06-05T10:02 |
| 数据集 | dataset_20260605_092545.json | dataset_20260605_095749.json |

### 第二轮评测结果

| Method | R@5 | R@10 | P@5 | P@10 | MRR | NDCG@10 | Avg Lat | P50 | P95 |
|--------|------|------|------|------|------|----------|---------|------|------|
| 直接检索 | 48.9% | 58.9% | 29.3% | 17.7% | 0.70 | 0.53 | 0.26s | 0.22s | 0.27s |
| MQE | 50.0% | 57.8% | 30.0% | 17.3% | 0.62 | 0.52 | 1.29s | 1.26s | 1.73s |
| **HyDE** | **50.0%** | **61.1%** | **30.0%** | **18.3%** | **0.74** | **0.56** | 1.87s | 1.90s | 2.13s |
| MQE+HyDE | 48.9% | 61.1% | 29.3% | 18.3% | 0.74 | 0.56 | 2.00s | 1.97s | 2.47s |

### 两轮对比

```
                  第一轮（原始）                      第二轮（增强）
               R@10    MRR   延迟              R@10    MRR   延迟
直接检索       96.7%   0.76   0.25s   →       58.9%   0.70   0.26s   (-37.8pp)
MQE           96.7%   0.72   1.22s   →       57.8%   0.62   1.29s   (-38.9pp)
HyDE          96.7%   0.71   1.91s   →       61.1%   0.74   1.87s   (-35.6pp)
MQE+HyDE      96.7%   0.74   1.88s   →       61.1%   0.74   2.00s   (-35.6pp)
```

### 关键发现

1. **改写措辞成功拉开差距**。问题经二次改写后与原文措辞差异很大，单纯向量检索的 Recall@10 从 96.7% 跌到 58.9%。这说明真实场景中，用户不会用文档原话提问，直接检索的精度会大幅下降。

2. **HyDE 是最大赢家**。在语义鸿沟大的场景下，HyDE（假设答案嵌入）的 MRR=0.74 显著优于 MQE（MRR=0.62）和 Direct（MRR=0.70）。原因：假设答案本身就是"像文档一样的文本"，嵌入后天然接近文档；而 MQE 的 4 角度变体只是从不同角度问同一个问题，未必能弥合措辞鸿沟。

3. **MQE 在本轮甚至不如 Direct**。MQE 的 R@10（57.8%）比 Direct（58.9%）还低 1.1pp。原因是改写措辞后，4 个角度的问题变体中可能只有 1-2 个与原问题相关，其余反而引入噪声，RRF 融合后拉低了整体精度。

4. **MQE+HyDE 与纯 HyDE 持平**。两者 Recall@10 和 MRR 完全相同。因为此时 HyDE 的假设答案已经提供了高质量的检索向量，MQE 分支的 RRF 融合没有带来额外增益——反而因噪声轻微拉低了 R@5。

5. **延迟差异稳定**。无论问题难度如何，Direct 始终 0.25s，MQE 约 1.2-1.3s，HyDE 约 1.9s，MQE+HyDE 约 2.0s（两个 LLM 调用并行，耗时 ≈ max(MQE, HyDE) ≈ 1.9s + 合并开销 ≈ 2.0s）。

6. **Precision 翻倍**。增强集每个问题有 3 个 GT 分块，所以 Precision@10 从 ~10% 升到 ~18%。

### 结论：什么时候用哪种策略

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 用户用接近文档原话的关键词搜索 | Direct | 0.25s，精度不输 MQE/HyDE |
| 用户用口语/非术语提问（最常见） | **HyDE** | 假设答案弥合语义鸿沟，MRR 最高 |
| 问题很短/模糊，需要多角度覆盖 | MQE | 4 角度重述，不依赖假设答案质量 |
| 追求最高精度，可接受延迟 | MQE+HyDE | 两者并行，相互补充（虽然本测试未体现增益，但更复杂的交叉问题上会有差异） |

