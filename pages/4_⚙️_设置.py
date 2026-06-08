"""
设置页面。

支持：
- LLM 配置（API 地址、模型名、API Key）
- Embedding 后端切换（API / 本地）
- 检索参数调整（top-k、MQE 变体数、HyDE 权重）
- 应用设置后清空引擎缓存并重新加载
"""

import streamlit as st

from src.core.config import get_config

st.set_page_config(page_title="设置 - Docmind", page_icon="⚙️")

# 刷新保持登录
if "user_id" not in st.session_state:
    qp_user = st.query_params.get("user_id")
    if qp_user:
        st.session_state.user_id = qp_user

st.query_params["user_id"] = st.session_state.get("user_id", "")
st.query_params["page"] = "settings"

st.title("⚙️ 设置")

st.caption("修改配置后，点击「应用设置」将清空引擎缓存并重新加载。")

# 获取当前配置
config = get_config()

# ============================================================================
# LLM 设置
# ============================================================================

st.subheader("🤖 LLM 配置")

col1, col2 = st.columns(2)
with col1:
    llm_base_url = st.text_input(
        "API 地址",
        value=config.llm.base_url,
        help="OpenAI 兼容 API 的基础地址。DeepSeek: https://api.deepseek.com/v1",
        key="llm_base_url",
    )
with col2:
    llm_model = st.text_input(
        "模型名称",
        value=config.llm.model,
        help="例如: gpt-4o, deepseek-chat, glm-4",
        key="llm_model",
    )

llm_api_key = st.text_input(
    "API Key",
    value=config.llm.api_key,
    type="password",
    help="LLM API 密钥。可在 .env 文件中设置 LLM_API_KEY。",
    key="llm_api_key",
)

# ============================================================================
# Embedding 设置
# ============================================================================

st.subheader("🔢 Embedding 配置")

embedding_backend = st.radio(
    "嵌入后端",
    options=["api", "local"],
    index=0 if config.embedding.backend == "api" else 1,
    horizontal=True,
    format_func=lambda x: "API 模式 (OpenAI 兼容)" if x == "api" else "本地模式 (sentence-transformers)",
    help="API 模式需要网络，延迟较高但质量好。本地模式无需网络，延迟低。",
    key="embedding_backend",
)

if embedding_backend == "api":
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        embed_api_url = st.text_input(
            "Embedding API 地址",
            value=config.embedding.base_url,
            help="如不填则使用 LLM 的 API 地址。",
            key="embed_api_url",
        )
    with col_e2:
        embed_model = st.text_input(
            "Embedding 模型",
            value=config.embedding.model,
            help="如 text-embedding-3-small, text-embedding-3-large",
            key="embed_model",
        )
else:
    embed_local_model = st.text_input(
        "本地模型名",
        value=config.embedding.local_model,
        help=(
            "sentence-transformers 模型名称。"
            "推荐: all-MiniLM-L6-v2 (英文), bge-small-zh (中文)"
        ),
        key="embed_local_model",
    )

# ============================================================================
# 检索参数
# ============================================================================

st.subheader("🔍 检索参数")

col_r1, col_r2 = st.columns(2)
with col_r1:
    retrieval_top_k = st.slider(
        "返回结果数 (Top-K)",
        min_value=3,
        max_value=30,
        value=config.retrieval.top_k,
        help="检索返回的最大分块数。",
        key="retrieval_top_k",
    )
with col_r2:
    mqe_variants = st.number_input(
        "MQE 变体数量",
        min_value=2,
        max_value=8,
        value=config.retrieval.mqe_num_variants,
        help="MQE 生成的查询变体数。建议 4。",
        key="mqe_variants",
    )

hyde_weight = st.slider(
    "HyDE 权重",
    min_value=0.1,
    max_value=0.9,
    value=config.retrieval.hyde_weight,
    step=0.05,
    help=(
        "MQE+HyDE 组合模式中 HyDE 分支的权重。"
        "Direct 分支权重 = 1 - HyDE 权重。"
        "默认 0.6 表示 HyDE 结果权重更高。"
    ),
    key="hyde_weight",
)

# ============================================================================
# 应用按钮
# ============================================================================

st.divider()

if st.button("✅ 应用设置", type="primary", use_container_width=True):
    # 更新配置单例
    config.llm.base_url = llm_base_url
    config.llm.model = llm_model
    config.llm.api_key = llm_api_key

    config.embedding.backend = embedding_backend
    if embedding_backend == "api":
        config.embedding.base_url = embed_api_url if 'embed_api_url' in locals() else llm_base_url
        config.embedding.model = embed_model if 'embed_model' in locals() else config.embedding.model
    else:
        config.embedding.local_model = embed_local_model if 'embed_local_model' in locals() else config.embedding.local_model

    config.retrieval.top_k = retrieval_top_k
    config.retrieval.mqe_num_variants = mqe_variants
    config.retrieval.hyde_weight = hyde_weight

    # 清空缓存资源（让 QAEngine 用新配置重新初始化）
    st.cache_resource.clear()

    st.success(
        "✅ 设置已应用！引擎缓存已清空，下次操作将使用新配置。\n\n"
        "页面即将刷新..."
    )

    import time
    time.sleep(1.5)
    st.rerun()

st.divider()
st.caption(
    "💡 提示: LLM API Key 可在此临时设置（仅当前进程有效）。"
    "如需永久配置，请在项目根目录的 `.env` 文件中设置对应环境变量。"
)

# 显示当前 .env 配置状态
with st.expander("📋 当前完整配置"):
    st.json({
        "llm": {
            "base_url": config.llm.base_url,
            "model": config.llm.model,
            "api_key": "***" + config.llm.api_key[-4:] if len(config.llm.api_key) > 4 else "(未设置)",
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
        },
        "embedding": {
            "backend": config.embedding.backend,
            "model": config.embedding.model if config.embedding.backend == "api" else config.embedding.local_model,
            "base_url": config.embedding.base_url,
        },
        "retrieval": {
            "top_k": config.retrieval.top_k,
            "mqe_num_variants": config.retrieval.mqe_num_variants,
            "hyde_weight": config.retrieval.hyde_weight,
            "rrf_k": config.retrieval.rrf_k,
        },
        "chunk": {
            "chunk_size": config.chunk.chunk_size,
            "chunk_overlap": config.chunk.chunk_overlap,
        },
        "chroma": {
            "persist_path": config.chroma.persist_path,
        },
        "neo4j": {
            "uri": config.neo4j.uri,
            "user": config.neo4j.user,
        },
        "sqlite": {
            "path": config.sqlite.path,
        },
    })
