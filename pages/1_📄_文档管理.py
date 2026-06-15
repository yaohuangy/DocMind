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

    # 动态 key：加载完成后 +1，强制 file_uploader 重置为空
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_files = st.file_uploader(
        "选择文档文件",
        type=["pdf", "docx", "md", "txt", "pptx", "csv", "xlsx"],
        accept_multiple_files=True,
        key=f"doc_uploader_{st.session_state.uploader_key}",
        help="可一次选择多个文件。加载完成后列表自动清空。",
    )

    # 展示上次加载结果的耗时分解（跨 rerun 持久化）
    if "ingest_results" in st.session_state and st.session_state.ingest_results:
        results_to_show = st.session_state.ingest_results
        for r in results_to_show:
            timing = r.get("step_timings", {})
            total = r.get("total_sec", 0)
            if timing:
                steps_html = "".join(
                    f"<tr><td>{step}</td><td style='text-align:right'>{sec:.1f}s</td></tr>"
                    for step, sec in timing.items()
                )
                st.markdown(
                    f"**{r['doc_name']}** 总耗时 {total:.1f}s  \n"
                    f"<table style='font-size:0.85rem;margin-top:4px'>"
                    f"<tr><th>步骤</th><th>耗时</th></tr>{steps_html}"
                    f"<tr style='font-weight:600'><td>合计</td>"
                    f"<td style='text-align:right'>{total:.1f}s</td></tr>"
                    f"</table>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(f"**{r['doc_name']}** 总耗时 {total:.1f}s")
        st.divider()
        del st.session_state.ingest_results

    if st.button("🚀 开始加载文件", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("请先选择要上传的文件。")
        else:
            import tempfile
            import threading
            import time as _time
            from pathlib import Path

            temp_dir = Path(tempfile.mkdtemp(prefix="docmind_"))
            saved_paths = []

            for uf in uploaded_files:
                save_path = temp_dir / uf.name
                save_path.write_bytes(uf.getvalue())
                saved_paths.append(str(save_path))

            # 摄入（后台线程 + 主线程实时计时轮询）
            t_total_start = _time.perf_counter()
            progress_bar = st.progress(0, text="准备中...")
            results = []
            total = len(saved_paths)

            for i, path in enumerate(saved_paths):
                # 后台线程执行摄入
                result_holder: list = []
                error_holder: list = []
                done = threading.Event()

                def _ingest(p=path):
                    try:
                        result_holder.append(engine.ingest(p))
                    except Exception as e:
                        error_holder.append(e)
                    done.set()

                thread = threading.Thread(target=_ingest, daemon=True)
                thread.start()

                # 主线程轮询：每 0.3s 更新一次计时器
                while not done.is_set():
                    elapsed = _time.perf_counter() - t_total_start
                    progress_bar.progress(
                        (i + 0.05) / total,
                        text=f"处理中 ({i + 1}/{total}): {Path(path).name}  |  ⏱ {elapsed:.0f}s",
                    )
                    thread.join(timeout=0.3)

                thread.join()  # 确保线程完全结束

                if error_holder:
                    st.error(f"加载失败 [{Path(path).name}]: {error_holder[0]}")
                elif result_holder:
                    results.append(result_holder[0])

                elapsed = _time.perf_counter() - t_total_start
                progress_bar.progress(
                    (i + 1) / total,
                    text=f"完成 ({i + 1}/{total}): {Path(path).name}  |  ⏱ {elapsed:.0f}s",
                )

            total_elapsed = _time.perf_counter() - t_total_start
            progress_bar.progress(
                1.0,
                text=f"✅ 全部完成! {len(results)}/{total} 个文档  |  ⏱ 总耗时 {total_elapsed:.0f}s",
            )

            if results:
                # 存入 session_state 跨 rerun 持久化
                st.session_state.ingest_results = [
                    {
                        "doc_name": r.doc_name,
                        "total_sec": r.total_sec,
                        "step_timings": r.step_timings,
                    }
                    for r in results
                ]

            # 清除文件上传列表：递增 key 强制重建 widget
            st.session_state.uploader_key += 1
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
        total_sec = doc.get("total_sec", 0.0)

        # 格式图标
        fmt_icons = {
            "pdf": "📕", "web": "🌐", "docx": "📘", "md": "📝",
            "txt": "📄", "pptx": "📊", "csv": "📈", "xlsx": "📗",
        }
        icon = fmt_icons.get(fmt, "📎")

        # 耗时显示
        if total_sec > 0:
            if total_sec < 60:
                time_str = f"{total_sec:.0f}s"
            else:
                m, s = divmod(total_sec, 60)
                time_str = f"{int(m)}m{s:.0f}s"
            time_col = f"⏱ {time_str}"
        else:
            time_col = ""

        col_info, col_del = st.columns([8, 1])
        with col_info:
            st.markdown(
                f"{icon} **{name}**  "
                f"`{fmt.upper()}` | {chunks} chunks | {chars:,} 字符 | {loaded}"
                + (f" | {time_col}" if time_col else "")
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
