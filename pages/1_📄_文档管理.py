"""
文档管理页面。

支持：
- 多文件上传（PDF, Word, Markdown, TXT, PPT, CSV, Excel）
- 网页 URL 抓取
- 文档列表、筛选、删除
- 分块参数由 .env 统一配置
"""

import streamlit as st

from src.engine.session import get_engine

st.set_page_config(page_title="文档管理 - Docmind", page_icon="📄")

engine = get_engine()  # 共享 QAEngine 实例

# 刷新保持登录：从 query_params 恢复 user_id
if "user_id" not in st.session_state:
    qp_user = st.query_params.get("user_id")
    if qp_user:
        st.session_state.user_id = qp_user
        engine.set_user_id(qp_user)

# 未登录则跳回首页
if not st.session_state.get("user_id"):
    st.switch_page("app.py")

engine.set_user_id(st.session_state.user_id)
st.query_params["user_id"] = st.session_state.user_id
st.query_params["page"] = "docs"

st.title("📄 文档管理")

# ============================================================================
# 上传区域
# ============================================================================

tab_upload, tab_url = st.tabs(["📁 文件上传", "🌐 网页抓取"])

with tab_upload:
    st.caption("支持格式: PDF, Word (.docx), Markdown (.md), TXT, PPT (.pptx), CSV, Excel (.xlsx)")
    st.caption("⚙️ 分块参数（如分块大小、重叠数）由管理员在 `.env` 文件中统一配置，加载时自动应用。")

    uploaded_files = st.file_uploader(
        "选择文档文件",
        type=["pdf", "docx", "md", "txt", "pptx", "csv", "xlsx"],
        accept_multiple_files=True,
        key="doc_uploader",
        help="可一次选择多个文件。上传后会自动暂存并等待加载。",
    )

    if st.button("🚀 开始加载文件", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("请先选择要上传的文件。")
        else:
            # 暂存上传的文件到本地
            import tempfile
            from pathlib import Path

            temp_dir = Path(tempfile.mkdtemp(prefix="docmind_"))
            saved_paths = []

            for uf in uploaded_files:
                save_path = temp_dir / uf.name
                save_path.write_bytes(uf.getvalue())
                saved_paths.append(str(save_path))

            # 摄入
            progress = st.progress(0, text="正在加载文档...")
            results = []
            total = len(saved_paths)

            for i, path in enumerate(saved_paths):
                progress.progress(
                    (i + 0.5) / total,
                    text=f"处理中 ({i + 1}/{total}): {Path(path).name}",
                )
                try:
                    result = engine.ingest(path)
                    results.append(result)
                except Exception as e:
                    st.error(f"加载失败 [{Path(path).name}]: {e}")

            progress.progress(1.0, text=f"完成! 成功加载 {len(results)}/{total} 个文档")
            if results:
                st.success(f"✅ 成功加载 {len(results)} 个文档")
                st.rerun()

with tab_url:
    st.caption("输入网页 URL，自动提取正文内容（过滤导航、广告等噪声）")

    url_input = st.text_input(
        "网页 URL",
        placeholder="https://example.com/article",
        key="url_input",
    )

    if st.button("🔗 抓取网页", type="primary", key="btn_url"):
        if not url_input:
            st.warning("请输入网页 URL。")
        elif not url_input.startswith(("http://", "https://")):
            st.warning("URL 必须以 http:// 或 https:// 开头。")
        else:
            with st.spinner(f"正在抓取: {url_input}"):
                try:
                    result = engine.ingest(url_input)
                    st.success(
                        f"✅ 网页抓取成功: {result.doc_name} "
                        f"({result.num_chunks} 个分块, {result.char_count} 字符)"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"网页抓取失败: {e}")

# ============================================================================
# 文档列表
# ============================================================================

st.divider()
st.subheader("📋 已加载文档")

# 格式筛选
format_filter = st.selectbox(
    "按格式筛选",
    options=["全部", "pdf", "web", "docx", "md", "txt", "pptx", "csv", "xlsx"],
    key="format_filter",
)

filter_val = None if format_filter == "全部" else format_filter

try:
    documents = engine.list_documents(doc_format=filter_val)
except Exception as e:
    st.error(f"获取文档列表失败: {e}")
    documents = []

if not documents:
    st.info("暂无已加载的文档。请上传文件或抓取网页。")
else:
    st.caption(f"共 {len(documents)} 个文档")

    for doc in documents:
        doc_id = doc["doc_id"]
        name = doc["name"]
        fmt = doc.get("format", "unknown")
        chunks = doc.get("num_chunks", 0)
        chars = doc.get("char_count", 0)
        loaded = doc.get("loaded_at", "")[:19]

        # 格式图标
        fmt_icons = {
            "pdf": "📕", "web": "🌐", "docx": "📘", "md": "📝",
            "txt": "📄", "pptx": "📊", "csv": "📈", "xlsx": "📗",
        }
        icon = fmt_icons.get(fmt, "📎")

        col_info, col_del = st.columns([8, 1])
        with col_info:
            st.markdown(
                f"{icon} **{name}**  "
                f"`{fmt.upper()}` | {chunks} chunks | {chars:,} 字符 | {loaded}"
            )
        with col_del:
            if st.button("🗑️", key=f"del_{doc_id}", help=f"删除 {name}"):
                try:
                    engine.delete_document(doc_id)
                    st.success(f"已删除: {name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"删除失败: {e}")

st.divider()
st.caption("💡 提示: 删除文档会同时移除其所有分块和向量数据。")
