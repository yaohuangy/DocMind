"""
学习回顾页面。

支持：
- 📜 历史记录：浏览全部 Q&A 和笔记，点击查看完整内容
- 🔍 搜索记忆：按关键词搜索情景/语义/工作记忆
- 🗺️ 知识图谱概览：概念节点统计
- 📊 学习报告：生成 JSON + LLM 建议
"""

import json
import streamlit as st

from src.engine.session import get_engine

st.set_page_config(page_title="学习回顾 - Docmind", page_icon="🧠")

engine = get_engine()

if "user_id" not in st.session_state:
    qp_user = st.query_params.get("user_id")
    if qp_user:
        st.session_state.user_id = qp_user
        engine.set_user_id(qp_user)

if not st.session_state.get("user_id"):
    st.switch_page("app.py")

engine.set_user_id(st.session_state.user_id)
st.query_params["user_id"] = st.session_state.user_id
st.query_params["page"] = "review"

st.title("🧠 学习回顾")

# ============================================================================
# Tab 1: 历史记录（默认显示）
# ============================================================================

tab_history, tab_search, tab_graph, tab_report = st.tabs(
    ["📜 历史记录", "🔍 搜索记忆", "🗺️ 知识图谱", "📊 学习报告"]
)

with tab_history:
    st.caption("所有提问和笔记，按时间倒序。可多选批量删除、编辑、一键清空。")

    # 筛选
    filter_type = st.radio(
        "筛选",
        options=["全部", "仅问答", "仅笔记"],
        horizontal=True,
        key="history_filter",
    )

    try:
        all_records = engine.get_history(limit=200)
    except Exception as e:
        st.warning(f"加载历史记录失败: {e}")
        all_records = []

    if filter_type == "仅问答":
        all_records = [r for r in all_records if r.event_type != "note"]
    elif filter_type == "仅笔记":
        all_records = [r for r in all_records if r.event_type == "note"]

    if not all_records:
        st.info("暂无历史记录。去首页问几个问题或添加笔记吧！")
    else:
        st.caption(f"共 {len(all_records)} 条记录")

        # 记忆管理
        with st.expander("🔧 记忆管理（清空）"):
            col_clear_ep, col_clear_neo = st.columns(2)

            with col_clear_ep:
                if st.button("🗑️ 清空情景记忆", help="删除当前用户所有 Q&A 记录（ChromaDB）", key="clear_episodic"):
                    st.session_state.confirm_clear_episodic = True

            with col_clear_neo:
                if st.button("🧹 清空知识图谱", help="删除当前用户所有 Concept 节点（Neo4j）", key="clear_semantic"):
                    st.session_state.confirm_clear_semantic = True

            # 清空情景记忆确认
            if st.session_state.get("confirm_clear_episodic"):
                st.error("确定要删除当前用户的所有情景记忆记录吗？此操作不可撤销。")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ 确认清空", type="primary", key="confirm_ep_yes"):
                        try:
                            # 直接清空整个 episodic_memory collection 并重建
                            from src.core.vector_store import VectorStore
                            engine._vector_store.delete_collection(VectorStore.EPISODIC_MEMORY)
                            engine._vector_store.ensure_collection(VectorStore.EPISODIC_MEMORY)
                            # 同时清空对话历史持久化
                            engine.clear_conversation()
                            st.session_state.messages = []
                        except Exception as exc:
                            st.warning(f"清空失败: {exc}")
                        st.session_state.confirm_clear_episodic = False
                        st.success("已彻底清空情景记忆")
                        st.rerun()
                with c2:
                    if st.button("❌ 取消", key="confirm_ep_no"):
                        st.session_state.confirm_clear_episodic = False
                        st.rerun()

            # 清空知识图谱确认
            if st.session_state.get("confirm_clear_semantic"):
                user = st.session_state.get("user_id", "")
                st.error(f"确定要清空知识图谱吗？将删除当前用户({user})及所有旧数据(default)的概念节点。")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ 确认清空", type="primary", key="confirm_ne_yes"):
                        try:
                            if engine.memory:
                                engine.memory.semantic.connect()
                                # 删除当前用户 + default + 无 user_id 的所有旧概念
                                engine.memory.semantic._graph.run_query(
                                    "MATCH (c:Concept) "
                                    "WHERE c.user_id = $uid OR c.user_id = 'default' OR c.user_id IS NULL "
                                    "DETACH DELETE c",
                                    {"uid": user},
                                )
                            st.session_state.confirm_clear_semantic = False
                            st.success("知识图谱已彻底清空")
                            st.rerun()
                        except Exception as exc:
                            st.warning(f"清空知识图谱失败: {exc}")
                with c2:
                    if st.button("❌ 取消", key="confirm_ne_no"):
                        st.session_state.confirm_clear_semantic = False
                        st.rerun()

        # ---- 批量操作栏 ----
        col_sel_all, col_sel_none, col_del_sel, col_del_all = st.columns(4)
        with col_sel_all:
            if st.button("☑️ 全选", use_container_width=True, key="btn_sel_all"):
                for rec in all_records:
                    st.session_state[f"hist_select_{rec.record_id}"] = True
                st.rerun()
        with col_sel_none:
            if st.button("◻️ 取消全选", use_container_width=True, key="btn_sel_none"):
                for rec in all_records:
                    st.session_state[f"hist_select_{rec.record_id}"] = False
                st.rerun()
        with col_del_sel:
            if st.button("🗑️ 删除选中", use_container_width=True, key="del_selected"):
                selected_ids = [rec.record_id for rec in all_records
                                if st.session_state.get(f"hist_select_{rec.record_id}")]
                if selected_ids:
                    st.session_state.del_selected_ids = selected_ids
                else:
                    st.warning("未选中任何记录")
        with col_del_all:
            if st.button("⚠️ 清空全部", use_container_width=True, key="del_all"):
                st.session_state.confirm_clear_all = True

        # 删除选中确认弹窗
        if st.session_state.get("del_selected_ids"):
            ids = st.session_state.del_selected_ids
            st.warning(f"确定要删除选中的 {len(ids)} 条历史记录吗？此操作不可撤销。")
            col_y, col_n = st.columns(2)
            with col_y:
                if st.button(f"✅ 确认删除 {len(ids)} 条", type="primary", key="confirm_del_sel"):
                    for rid in ids:
                        try:
                            engine.delete_note(rid)
                        except Exception:
                            pass
                        st.session_state.pop(f"hist_select_{rid}", None)
                    st.session_state.del_selected_ids = None
                    st.success(f"已删除 {len(ids)} 条记录")
                    st.rerun()
            with col_n:
                if st.button("❌ 取消", key="cancel_del_sel"):
                    st.session_state.del_selected_ids = None
                    st.rerun()

        # 清空全部确认弹窗
        if st.session_state.get("confirm_clear_all"):
            st.error(f"确定要删除当前用户的全部 {len(all_records)} 条历史记录吗？此操作不可撤销。")
            col_y, col_n = st.columns(2)
            with col_y:
                if st.button("✅ 确认清空", type="primary", key="confirm_yes"):
                    for rec in all_records:
                        try:
                            engine.delete_note(rec.record_id)
                        except Exception:
                            pass
                    st.session_state.confirm_clear_all = False
                    st.success("已清空全部历史记录")
                    st.rerun()
            with col_n:
                if st.button("❌ 取消", key="confirm_no"):
                    st.session_state.confirm_clear_all = False
                    st.rerun()

        st.divider()

        # ---- 记录列表 ----
        for i, rec in enumerate(all_records):
            rid = rec.record_id
            is_note = rec.event_type == "note"
            icon = "📝" if is_note else "💬"
            ts = rec.timestamp[:19] if rec.timestamp else ""

            if is_note:
                title = (rec.question or rec.answer_summary or "")[:80]
            else:
                title = rec.question[:80] if rec.question else ""
            title = title.split("\n")[0]

            # 每行：复选框 + 展开区
            col_cb, col_exp = st.columns([0.5, 9.5])
            with col_cb:
                st.checkbox("", key=f"hist_select_{rid}", label_visibility="collapsed")
            with col_exp:
                with st.expander(f"{icon} {title}{'...' if len(title) >= 80 else ''}  ({ts[:10]})"):
                    # ---- 编辑模式 ----
                    edit_key = f"hist_edit_{rid}"
                    if st.session_state.get(edit_key):
                        col_q, col_a = st.columns(2)
                        with col_q:
                            new_q = st.text_area("编辑问题", value=rec.question, height=100, key=f"edit_q_{rid}")
                        with col_a:
                            new_a = st.text_area("编辑回答", value=rec.answer_summary, height=100, key=f"edit_a_{rid}")
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button("💾 保存修改", key=f"save_{rid}", type="primary"):
                                # 更新情景记忆中的记录（删除旧+重新记录）
                                try:
                                    engine.memory.episodic.record(
                                        question=new_q,
                                        answer_summary=new_a,
                                        source_chunks=rec.source_chunks,
                                        documents=rec.documents,
                                        concepts_extracted=rec.concepts_extracted,
                                        importance=rec.importance,
                                        session_id=rec.session_id,
                                        user_id=rec.user_id,
                                        event_type=rec.event_type,
                                        timestamp=rec.timestamp,
                                    )
                                    engine.memory.episodic.delete_record(rid)
                                    st.session_state[edit_key] = False
                                    st.success("已保存")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"保存失败: {e}")
                        with col_cancel:
                            if st.button("取消", key=f"cancel_{rid}"):
                                st.session_state[edit_key] = False
                                st.rerun()
                    else:
                        # ---- 查看模式 ----
                        if is_note:
                            st.markdown(rec.answer_summary or rec.question)
                        else:
                            st.markdown(f"**❓ 问题**: {rec.question}")
                            st.markdown(f"**💡 回答**: {rec.answer_summary}")

                        # 元信息 + 操作按钮
                        col_meta, col_act = st.columns([7, 3])
                        with col_meta:
                            parts = []
                            if rec.concepts_extracted:
                                parts.append(f"🏷️ {', '.join(rec.concepts_extracted[:3])}")
                            if rec.documents:
                                parts.append(f"📄 {', '.join(rec.documents[:2])}")
                            parts.append(f"⭐{rec.importance:.2f} | 🕐{ts}")
                            st.caption(" | ".join(parts))
                        with col_act:
                            col_edit, col_del = st.columns(2)
                            with col_edit:
                                if st.button("✏️ 编辑", key=f"editbtn_{rid}"):
                                    st.session_state[edit_key] = True
                                    st.rerun()
                            with col_del:
                                if st.button("🗑️", key=f"delbtn_{rid}", help="删除此条记录"):
                                    try:
                                        engine.delete_note(rid)
                                        st.success("已删除")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"删除失败: {e}")

# ============================================================================
# Tab 2: 搜索记忆
# ============================================================================

with tab_search:
    st.caption("按关键词搜索情景记忆、语义记忆和工作记忆。")

    col_search, col_btn = st.columns([5, 1])
    with col_search:
        search_keyword = st.text_input(
            "搜索关键词",
            placeholder="输入概念名或问题关键词...",
            key="search_keyword_tab",
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("搜索", type="primary", key="btn_search_tab", use_container_width=True)

    if search_clicked and search_keyword.strip():
        with st.spinner(f"搜索中: {search_keyword}"):
            search_results = engine.search_memory(search_keyword.strip())

        # 情景记忆
        st.subheader("📖 情景记忆")
        episodic_results = search_results.get("episodic", [])
        if episodic_results:
            for i, rec in enumerate(episodic_results):
                is_note = rec.event_type == "note"
                icon = "📝" if is_note else "💬"
                q = rec.question[:80]
                ts = rec.timestamp[:10] if rec.timestamp else "?"

                with st.expander(f"{icon} [{i + 1}] {q}{'...' if len(rec.question) > 80 else ''}  ({ts})"):
                    if is_note:
                        st.markdown(rec.answer_summary)
                    else:
                        st.markdown(f"**❓ 问题**: {rec.question}")
                        st.markdown(f"**💡 回答**: {rec.answer_summary}")
                    if rec.concepts_extracted:
                        st.caption(f"🏷️ 概念: {', '.join(rec.concepts_extracted)}")
                    if rec.documents:
                        st.caption(f"📄 来源: {', '.join(rec.documents)}")
                    st.caption(f"⭐ 重要性: {rec.importance:.2f} | 🕐 {rec.timestamp}")
        else:
            st.info("未找到匹配的情景记忆记录。")

        # 语义记忆
        st.subheader("🧩 语义记忆（知识图谱）")
        semantic_results = search_results.get("semantic", [])
        if semantic_results:
            for concept in semantic_results:
                freq = concept.frequency if hasattr(concept, 'frequency') else 0
                ctype = concept.concept_type if hasattr(concept, 'concept_type') else ""
                desc = concept.description if hasattr(concept, 'description') else ""
                st.markdown(f"**{concept.name}**  `{ctype}`  出现 {freq} 次")
                if desc:
                    st.caption(f"  {desc}")
        else:
            st.info("未找到匹配的概念节点。")

        # 工作记忆
        st.subheader("💭 工作记忆（当前会话）")
        working_results = search_results.get("working", [])
        if working_results:
            for i, entry in enumerate(working_results):
                q = entry.question if hasattr(entry, 'question') else str(entry)
                with st.expander(f"[{i + 1}] {q[:80]}"):
                    st.text(q)
        else:
            st.info("当前会话中没有匹配的记录。")

# ============================================================================
# Tab 3: 知识图谱概览
# ============================================================================

with tab_graph:
    st.caption("语义记忆中的概念知识图谱——交互式力导向图。")

    # 检查 Neo4j 是否可用
    neo4j_ok = getattr(engine, '_neo4j_available', False)

    # ---- 统计卡片（始终显示） ----
    try:
        review_data = engine.get_review_data()
    except Exception:
        review_data = {}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("概念总数", review_data.get("concept_count", 0))
    with col2:
        st.metric("情景记忆数", review_data.get("episodic_count", 0))
    with col3:
        st.metric("工作记忆数", review_data.get("working_entries", 0))
    with col4:
        top10_count = min(10, review_data.get("concept_count", 0))
        st.metric("Top 10", top10_count)

    # ---- 交互式图谱 ----
    if not neo4j_ok:
        st.info("⚠️ Neo4j 不可用——知识图谱可视化已降级。启动 Neo4j 后重启应用即可。")
    else:
        # ---- 演示数据按钮 ----
        col_seed, col_hint_seed = st.columns([2, 5])
        with col_seed:
            if st.button("🎲 初始化演示频率", help="为概念节点随机分配 1~15 的频率，模拟多次交互效果", key="seed_demo"):
                with st.spinner("正在生成演示数据..."):
                    try:
                        cnt = engine.seed_demo_frequencies()
                        st.success(f"已更新 {cnt} 个概念的频率！")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"初始化失败: {exc}")
        with col_hint_seed:
            st.caption("💡 如果所有概念频率都是 1，点此按钮模拟真实分布（仅用于演示）")

        graph_data = engine.get_graph_data()
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes:
            st.info("暂无概念数据。去首页问几个问题，系统会自动提取概念并构建知识图谱。")
        else:
            from pyvis.network import Network
            import streamlit.components.v1 as components
            import math

            # ---- 颜色映射 ----
            _type_colors = {
                "technique": "#4CAF50",
                "mechanism": "#2196F3",
                "architecture": "#FF9800",
                "algorithm": "#9C27B0",
                "theory": "#F44336",
                "tool": "#00BCD4",
                "application": "#795548",
                "concept": "#607D8B",
            }

            # ---- 节点大小：按 frequency 映射 16~60 px ----
            freqs = [n.get("frequency", 0) for n in nodes]
            max_freq = max(freqs) if freqs else 1
            min_freq = min(freqs) if freqs else 0

            def _node_size(freq: int) -> int:
                if max_freq <= 1:
                    return 28
                # 对数缩放，避免一两个高频节点撑得过大
                if min_freq > 0:
                    ratio = (math.log(freq) - math.log(min_freq)) / (math.log(max_freq) - math.log(min_freq) + 0.001)
                else:
                    ratio = freq / max_freq
                return max(16, min(60, int(16 + ratio * 44)))

            def _node_color(freq: int, base_color: str) -> str:
                """频率越高颜色越深（混入黑色），让高频节点更突出。"""
                # 在 base_color 基础上按频率加深
                if max_freq <= 1:
                    return base_color
                # 简单的 hex 加深算法
                ratio = freq / max_freq
                r = int(int(base_color[1:3], 16) * (1 - ratio * 0.6))
                g = int(int(base_color[3:5], 16) * (1 - ratio * 0.6))
                b = int(int(base_color[5:7], 16) * (1 - ratio * 0.6))
                return f"#{r:02x}{g:02x}{b:02x}"

            # ---- 构建 pyvis 网络 ----
            net = Network(
                height="650px",
                width="100%",
                bgcolor="#ffffff",
                font_color="#1a1a1a",
                directed=False,
            )

            # ---- 显式设置 vis.js 选项：强制显示标签、配置字体 ----
            net.set_options("""
            {
              "nodes": {
                "font": {
                  "size": 14,
                  "face": "Microsoft YaHei, Arial, sans-serif",
                  "color": "#1a1a1a",
                  "strokeWidth": 2,
                  "strokeColor": "#ffffff",
                  "bold": false
                },
                "borderWidth": 2,
                "borderWidthSelected": 4
              },
              "edges": {
                "font": {
                  "size": 9,
                  "color": "#999999"
                },
                "color": {
                  "color": "#bbbbbb",
                  "highlight": "#666666",
                  "hover": "#888888"
                },
                "smooth": {
                  "type": "continuous"
                }
              },
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -8000,
                  "centralGravity": 0.3,
                  "springLength": 220,
                  "springConstant": 0.005,
                  "damping": 0.09,
                  "avoidOverlap": 0
                },
                "minVelocity": 0.75,
                "solver": "barnesHut"
              },
              "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "navigationButtons": true,
                "keyboard": true
              }
            }
            """)

            # ---- 添加节点 ----
            for n in nodes:
                name = n["name"]
                freq = n.get("frequency", 0)
                ctype = n.get("type", "concept")
                desc = n.get("description", "")[:150]
                base_color = _type_colors.get(ctype, _type_colors["concept"])
                node_color = _node_color(freq, base_color)

                # label：始终显示在节点下方
                label = f"{name}\n({freq})"
                # title：悬停 tooltip
                hover_parts = [f"<b>{name}</b>", f"类型: {ctype}", f"提及: {freq} 次"]
                if desc:
                    hover_parts.append(f"<i>{desc}</i>")
                hover_text = "<br>".join(hover_parts)

                net.add_node(
                    name,
                    label=label,
                    title=hover_text,
                    size=_node_size(freq),
                    color=node_color,
                    shape="dot",
                    font={
                        "size": max(11, min(16, 11 + freq * 2)),
                        "color": "#1a1a1a",
                        "strokeWidth": 1,
                        "strokeColor": "#ffffff",
                    },
                )

            # ---- 添加边 ----
            for e in edges:
                strength = e.get("strength", 0.5)
                width = max(1, strength * 4)
                net.add_edge(
                    e["source"],
                    e["target"],
                    width=width,
                    title=f"关联强度: {strength:.2f}",
                    color={"color": "#bbbbbb", "highlight": "#666666"},
                    smooth=True,
                )

            # 渲染为 HTML 并嵌入
            try:
                import tempfile
                import os

                # pyvis save_graph 要求文件名以 .html 结尾，写入磁盘。
                # 使用 tempfile 避免权限和路径冲突。
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".html", delete=False, encoding="utf-8"
                ) as tmp:
                    tmp_path = tmp.name
                net.save_graph(tmp_path)

                # 读取生成的 HTML
                with open(tmp_path, "r", encoding="utf-8") as f:
                    full_html = f.read()

                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

                # 调试信息：显示节点/边计数
                st.caption(f"📊 {len(nodes)} 个概念节点 · {len(edges)} 条关系连线")

                # 下载按钮：方便手动检查 HTML 内容
                st.download_button(
                    label="📥 下载图谱 HTML",
                    data=full_html,
                    file_name=f"knowledge_graph_{st.session_state.get('user_id', 'user')}.html",
                    mime="text/html",
                )

                # 关键修复：传入完整 HTML（含 <head> 中的 vis.js 脚本），
                # 不能只取 body，否则 js 全丢，图形不可见。
                # st.components.html 用 iframe 渲染，完整 HTML 文档完全没问题。
                components.html(full_html, height=650, scrolling=True)

                # ---- Top 10 高频概念列表 ----
                st.divider()
                st.subheader("🏆 Top 10 高频概念")
                top_nodes = nodes[:10]
                cols = st.columns(min(len(top_nodes), 5))
                for i, n in enumerate(top_nodes):
                    with cols[i % 5]:
                        st.metric(
                            label=n["name"],
                            value=n.get("frequency", 0),
                            delta=n.get("type", ""),
                        )
            except Exception as e:
                import traceback
                st.warning(f"图谱渲染失败: {e}")
                with st.expander("🔍 调试详情"):
                    st.code(traceback.format_exc())
                # 降级：显示文本列表
                st.subheader("🏆 Top 10 高频概念（文本回退）")
                cols = st.columns(min(len(nodes[:10]), 5))
                for i, n in enumerate(nodes[:10]):
                    with cols[i % 5]:
                        st.metric(
                            label=n["name"],
                            value=n.get("frequency", 0),
                            delta=n.get("type", ""),
                        )

# ============================================================================
# Tab 4: 学习报告
# ============================================================================

with tab_report:
    st.caption("聚合所有记忆数据，生成结构化学习报告。")

    # 检测是否有新数据（比较上次生成报告时的记忆数量）
    try:
        current_episodic = engine.memory.episodic.get_record_count() if engine.memory else 0
    except Exception:
        current_episodic = 0

    last_count = st.session_state.get("report_episodic_count", -1)
    report_stale = (last_count >= 0 and current_episodic != last_count)

    # 按钮行
    col_gen, col_hint = st.columns([2, 3])
    with col_gen:
        do_generate = st.button(
            "📝 生成学习报告",
            type="primary",
            use_container_width=True,
        )

    with col_hint:
        if "cached_report" in st.session_state and report_stale:
            st.info(f"⚠️ 新增了 {abs(current_episodic - last_count)} 条记录，建议重新生成报告")

    if do_generate:
        with st.spinner("正在生成学习报告..."):
            try:
                report = engine.generate_report()
                st.session_state.cached_report = report
                st.session_state.report_episodic_count = current_episodic
                st.success("✅ 报告生成完成")
                st.rerun()
            except Exception as e:
                st.error(f"报告生成失败: {e}")
                st.info("请确保 Neo4j 和 ChromaDB 服务可用，且有足够的交互数据。")

    # 显示缓存的报告
    if "cached_report" in st.session_state:
        report = st.session_state.cached_report

        if not report_stale:
            st.caption("⏳ 报告已缓存，新增问答或笔记后会自动提示重新生成")

        # 摘要
        summary = report.get("summary", {})
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("情景记录", summary.get("episodic_count", 0))
        with col_b:
            st.metric("概念数", summary.get("concept_count", 0))
        with col_c:
            st.metric("笔记数", summary.get("note_count", 0))
        with col_d:
            st.metric("工作记忆", summary.get("working_entries", 0))

        # 概念列表
        st.markdown("**核心概念**")
        st.dataframe(
            report.get("concepts", [])[:20],
            use_container_width=True,
            column_config={
                "name": "概念名",
                "type": "类型",
                "frequency": "出现次数",
            },
        )

        # LLM 学习建议
        st.markdown("**💡 学习建议**")
        suggestions = report.get("suggestions", "暂无建议。")
        st.info(suggestions)

        # 下载
        report_json = json.dumps(report, ensure_ascii=False, indent=2, default=str)
        st.download_button(
            label="📥 下载报告 (JSON)",
            data=report_json,
            file_name=f"learning_report_{st.session_state.get('session_id', 'report')}.json",
            mime="application/json",
        )

        # 近期活动
        with st.expander("📜 近期活动详情"):
            for act in report.get("recent_activities", []):
                st.caption(
                    f"[{act.get('timestamp', '?')[:19]}] "
                    f"{act.get('question', '')[:100]} "
                    f"(重要性: {act.get('importance', 0):.2f})"
                )
    else:
        st.info("点击上方按钮生成学习报告。")

st.divider()
st.caption("💡 提示: 每次问答后系统自动记录。在「📜 历史记录」中可浏览和回顾全部内容。")
