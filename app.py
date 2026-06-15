"""
Docmind - 主页（问答页面）

Streamlit 应用入口。提供聊天式问答界面，集成四种检索策略、
流式答案生成、内联引用展示和三记忆系统。

启动方式:
    streamlit run app.py --server.address 0.0.0.0 --server.port 7860
"""

import streamlit as st

from src.engine.models import SourceChunk
from src.engine.session import get_engine

# ============================================================================
# 页面配置（仅在作为入口时设置）
# ============================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Docmind",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ============================================================================
# 引擎初始化
# ============================================================================

engine = get_engine()

# ============================================================================
# 需求 4：从 query_params 恢复用户身份（刷新后保持登录）
# ============================================================================

if "user_id" not in st.session_state:
    # 优先从 URL 参数恢复
    qp_user = st.query_params.get("user_id")
    if qp_user:
        st.session_state.user_id = qp_user
        engine.set_user_id(qp_user)

# ============================================================================
# 会话状态初始化
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
    # 尝试从持久化存储恢复聊天记录
    if st.session_state.get("user_id"):
        try:
            restored = engine.load_conversation()
            if restored:
                st.session_state.messages = restored
        except Exception:
            pass

if "retrieval_method" not in st.session_state:
    st.session_state.retrieval_method = "MQE+HyDE"

if "session_id" not in st.session_state:
    from datetime import datetime
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    engine.set_session_id(st.session_state.session_id)

if "top_k" not in st.session_state:
    st.session_state.top_k = 10

# ============================================================================
# 需求 1：是否已初始化——判断 user_id 是否已设置
# ============================================================================

initialized = st.session_state.get("user_id") is not None

# 必须在此处同步 user_id，确保侧边栏统计和反馈写入使用正确用户
if initialized:
    engine.set_user_id(st.session_state.user_id)

# ============================================================================
# 侧边栏
# ============================================================================

with st.sidebar:
    st.title("📚 Docmind")

    if initialized:
        # 已登录：显示用户名 + 退出按钮
        col_user, col_out = st.columns([3, 1])
        with col_user:
            st.caption(f"👤 {st.session_state.user_id}")
        with col_out:
            if st.button("🚪", help="退出/切换用户", key="logout_btn"):
                st.session_state.user_id = None
                st.session_state.messages = []
                try:
                    engine.clear_conversation()
                except Exception:
                    pass
                if "user_id" in st.query_params:
                    del st.query_params["user_id"]
                st.rerun()

        st.divider()

        # 统计信息
        try:
            stats = engine.get_stats()
            # 提问次数 = 情景记忆中所有已持久化的问答记录（权威来源）
            try:
                question_count = len([
                    r for r in engine.get_history(limit=500)
                    if r.event_type != "note"
                ])
            except Exception:
                question_count = 0
            col1, col2 = st.columns(2)
            with col1:
                st.metric("文档", stats.get("文档总数", 0))
                st.metric("提问", question_count)
            with col2:
                st.metric("笔记", stats.get("学习笔记", 0))
                st.metric("概念", stats.get("概念数量", 0))
        except Exception as e:
            st.caption(f"统计加载失败: {e}")

        st.divider()

        # 检索设置
        st.caption("⚙️ 检索设置")
        st.session_state.top_k = st.slider(
            "返回结果数", min_value=3, max_value=20, value=st.session_state.top_k
        )

        st.divider()
        st.caption("📄 页面导航见左侧")

# ============================================================================
# 未初始化：显示欢迎/注册页
# ============================================================================

if not initialized:
    # =========================================================================
    # 首页自定义样式
    # =========================================================================
    st.markdown("""
    <style>
        /* 全局容器 —— 顶部归零让 Hero 上移 */
        .main .block-container {
            padding-top: 0;
            padding-bottom: 0;
            max-width: none;
        }

        /* Hero 区域居中 */
        .hero-section {
            text-align: center;
            padding: 0.1rem 0 0.5rem 0;
        }

        /* 欢迎卡片 */
        .welcome-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 1.6rem 2.4rem 1.4rem 2.4rem;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
            max-width: 500px;
            margin: 0 auto;
        }

        /* 输入框 */
        div[data-testid="stTextInput"] input {
            border-radius: 10px;
            padding: 0.6rem 0.85rem;
            font-size: 0.93rem;
            border: 1px solid #d1d5db;
            background: #ffffff;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
        }

        /* 按钮 */
        div[data-testid="stButton"] button {
            border-radius: 10px;
            padding: 0.55rem 1rem;
            font-size: 0.93rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        /* 功能卡片 */
        .feature-card {
            background: #ffffff;
            border: 1px solid #e8ecf1;
            border-radius: 12px;
            padding: 1rem 0.7rem 0.9rem 0.7rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            transition: box-shadow 0.15s, border-color 0.15s;
        }
        .feature-card:hover {
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.08);
            border-color: #93c5fd;
        }
        .feature-icon {
            font-size: 1.8rem;
            margin-bottom: 0.3rem;
        }
        .feature-title {
            font-weight: 600;
            font-size: 0.9rem;
            color: #1e293b;
            margin-bottom: 0.2rem;
        }
        .feature-desc {
            font-size: 0.78rem;
            color: #64748b;
            line-height: 1.45;
        }

        /* 全局压缩 */
        hr {
            margin: 0.4rem 0;
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

    # =========================================================================
    # Hero 区域（整体上移）
    # =========================================================================
    st.markdown("""
    <div style="margin-top: -1.4rem;">
    <div class="hero-section">
        <h1 style="font-size: 2.2rem; font-weight: 700; color: #0f172a; margin: 0 0 0.3rem 0;">
            📚 Docmind
        </h1>
        <p style="font-size: 1.05rem; color: #475569; margin: 0 0 0.15rem 0; font-weight: 500;">
            基于 RAG 的智能文档问答助手
        </p>
        <p style="font-size: 0.8rem; color: #94a3b8; margin: 0;">
            文档解析 · 智能检索 · AI 生成回答
        </p>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # 居中欢迎卡片
    # =========================================================================
    _, center_col, _ = st.columns([1, 1.8, 1])

    with center_col:
        st.markdown("""
        <div class="welcome-card">
            <p style="text-align:center; font-weight:600; font-size:0.85rem;
                      color:#374151; margin:0 0 0.9rem 0;">
                👤 输入用户名开始使用
            </p>
        """, unsafe_allow_html=True)

        username_input = st.text_input(
            "用户名",
            placeholder="请输入您的用户名，例如 user_id",
            key="init_username",
            label_visibility="collapsed",
        )

        confirm_clicked = st.button(
            "✅ 开始使用",
            type="primary",
            use_container_width=True,
            key="init_confirm",
        )

        st.markdown('</div>', unsafe_allow_html=True)  # close .welcome-card

    if confirm_clicked:
        username = username_input.strip()
        if username:
            st.session_state.user_id = username
            st.query_params["user_id"] = username
            engine.set_user_id(username)
            st.rerun()
        else:
            st.warning("请输入一个非空用户名")

    # =========================================================================
    # 功能卡片（三列）
    # =========================================================================
    st.markdown('<div style="margin-top:1.5rem;">', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <div class="feature-title">文档解析</div>
            <div class="feature-desc">PDF · Word · PPT · 网页<br>Excel · CSV · Markdown · TXT<br><small style="color:#94a3b8;">8 种格式自动分块</small></div>
        </div>
        """, unsafe_allow_html=True)

    with fc2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">智能问答</div>
            <div class="feature-desc">基于 RAG 检索增强生成<br>MQE + HyDE 混合检索<br><small style="color:#94a3b8;">支持流式输出与引用定位</small></div>
        </div>
        """, unsafe_allow_html=True)

    with fc3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">学习回顾</div>
            <div class="feature-desc">历史问答与反馈追踪<br>知识图谱自动构建<br><small style="color:#94a3b8;">笔记与概念关联管理</small></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.stop()  # 未初始化时不渲染后续内容

# ============================================================================
# 同步页面标识（user_id 已在侧边栏前设置）
# ============================================================================

st.query_params["user_id"] = st.session_state.user_id
st.query_params["page"] = "qa"

# ============================================================================
# 主区域（已初始化）
# ============================================================================

st.title("📚 Docmind")
st.caption("上传文档后开始提问——支持 8 种格式 | MQE + HyDE 混合检索 | 三记忆学习系统")

# ============================================================================
# 检索模式选择
# ============================================================================

method_map = {"直接检索": "direct", "MQE": "mqe", "HyDE": "hyde", "MQE+HyDE": "mqe+hyde"}
reverse_map = {v: k for k, v in method_map.items()}

current_key = st.session_state.retrieval_method
default_label = reverse_map.get(current_key, "MQE+HyDE")

selected_label = st.radio(
    "检索模式",
    options=list(method_map.keys()),
    index=list(method_map.keys()).index(default_label) if default_label in method_map else 3,
    horizontal=True,
    key="retrieval_method_select",
    help=(
        "**直接检索**: 最快速，直接向量搜索\n"
        "**MQE**: 多查询扩展，4 角度变体 + RRF 融合\n"
        "**HyDE**: 假设文档嵌入，用假设答案替代问题搜索\n"
        "**MQE+HyDE**: 两者并行执行，加权合并（推荐）"
    ),
)

st.session_state.retrieval_method = method_map[selected_label]
method = st.session_state.retrieval_method

# ============================================================================
# 聊天历史展示
# ============================================================================

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 反馈按钮：仅对助手消息，且必须已保存到消息列表后才渲染
        if msg["role"] == "assistant":
            fb_key = f"feedback_{i}"
            if fb_key not in st.session_state:
                st.session_state[fb_key] = None

            if st.session_state[fb_key] is None:
                col_fb, _ = st.columns([1, 9])
                with col_fb:
                    if st.button("👍", key=f"fb_up_{i}", help="有用"):
                        # 从消息中取方法名和延迟
                        fb_method = st.session_state.get("retrieval_method", "mqe+hyde")
                        fb_latency = msg.get("latency_sec", 0.0)
                        engine.record_feedback(
                            st.session_state.messages[i - 1]["content"] if i > 0 else "",
                            msg["content"],
                            fb_method,
                            "useful",
                            fb_latency,
                        )
                        st.session_state[fb_key] = "useful"
                        st.rerun()
                    if st.button("👎", key=f"fb_down_{i}", help="无用"):
                        fb_method = st.session_state.get("retrieval_method", "mqe+hyde")
                        fb_latency = msg.get("latency_sec", 0.0)
                        engine.record_feedback(
                            st.session_state.messages[i - 1]["content"] if i > 0 else "",
                            msg["content"],
                            fb_method,
                            "not_useful",
                            fb_latency,
                        )
                        st.session_state[fb_key] = "not_useful"
                        st.rerun()
            else:
                label = "👍 有用" if st.session_state[fb_key] == "useful" else "👎 无用"
                st.caption(f"已反馈: {label}")

        if msg.get("sources"):
            with st.expander(f"📎 参考来源 ({len(msg['sources'])} 条)"):
                for j, src_dict in enumerate(msg["sources"], 1):
                    src = SourceChunk.from_dict(src_dict)
                    st.caption(
                        f"**[{j}]** 📄 {src.doc_name}  |  "
                        f"📍 {src.location_text}  |  "
                        f"📊 相似度: {src.score:.3f}"
                    )
                    with st.expander(f"查看片段 [{j}]"):
                        st.text(src.text[:800])

# ============================================================================
# 输入处理
# ============================================================================

if prompt := st.chat_input("输入你的问题..."):
    import time as _time

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 检索（计时）
        t0 = _time.perf_counter()
        with st.spinner(f"🔍 检索中 ({selected_label})..."):
            try:
                sources = engine.retrieve(
                    prompt,
                    method=method,
                    top_k=st.session_state.top_k,
                )
            except Exception as e:
                st.error(f"检索失败: {e}")
                sources = []
        retrieval_latency = _time.perf_counter() - t0

        if not sources:
            st.warning("⚠️ 未找到相关文档。请先上传文档到「文档管理」页面。")

        def token_generator():
            for token in engine.generate_stream(prompt, sources, method=method):
                yield token

        try:
            answer = st.write_stream(token_generator())
        except Exception as e:
            answer = f"*[生成失败: {e}]*"
            st.error(answer)

        try:
            formatted_answer, cited_sources = engine.format_answer(answer, sources)
        except Exception:
            formatted_answer = answer
            cited_sources = sources

        if cited_sources:
            with st.expander(f"📎 参考来源 ({len(cited_sources)} 条)"):
                for i, src in enumerate(cited_sources, 1):
                    st.caption(
                        f"**[{i}]** 📄 {src.doc_name}  |  "
                        f"📍 {src.location_text}  |  "
                        f"📊 相似度: {src.score:.3f}"
                    )
                    with st.expander(f"查看片段 [{i}]"):
                        st.text(src.text[:800])

        # 先保存助手消息到聊天历史（反馈按钮和持久化依赖它）
        st.session_state.messages.append({
            "role": "assistant",
            "content": formatted_answer,
            "sources": [s.to_dict() for s in (cited_sources if cited_sources else sources)],
            "latency_sec": retrieval_latency,
        })

        # 持久化聊天记录（快速，不阻塞）
        try:
            engine.save_conversation(st.session_state.messages)
        except Exception:
            pass

        # 记录 Token 用量（快速，SQLite 单行插入）
        try:
            engine.record_token_usage(method=method)
        except Exception:
            pass

        # 概念提取 + 记忆记录放到后台线程，不阻塞页面刷新
        # （LLM 概念提取可能耗时 2-10 秒，用户不需要等它完成）
        import threading
        _qa = prompt
        _ans = formatted_answer
        _srcs = cited_sources if cited_sources else sources

        def _background_memory():
            try:
                engine.record_interaction(
                    question=_qa,
                    answer=_ans,
                    sources=_srcs,
                )
            except Exception:
                pass

        threading.Thread(target=_background_memory, daemon=True).start()

        # 立即刷新页面，让 👍/👎 按钮即刻出现
        st.rerun()
