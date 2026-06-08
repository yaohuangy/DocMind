"""
笔记管理页面。

支持：
- 添加笔记（含关联概念）
- 查看已有笔记列表
- 删除笔记
"""

import streamlit as st

from src.engine.session import get_engine

st.set_page_config(page_title="学习笔记 - Docmind", page_icon="📝")

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
st.query_params["page"] = "notes"

st.title("📝 学习笔记")

# ============================================================================
# 添加笔记
# ============================================================================

st.subheader("✍️ 添加笔记")

note_content = st.text_area(
    "笔记内容",
    placeholder="在这里写下你的学习笔记...\n\n可以包含从文档中学到的知识点、自己的思考等。",
    height=150,
    key="note_content",
)

related_concepts_input = st.text_input(
    "关联概念（可选，用逗号分隔）",
    placeholder="例如: Transformer, Self-Attention, BERT",
    help="关联的概念将更新到知识图谱中，便于后续回顾。",
    key="related_concepts",
)

if st.button("💾 保存笔记", type="primary", use_container_width=True):
    if not note_content.strip():
        st.warning("请输入笔记内容。")
    else:
        # 解析关联概念
        concepts = []
        if related_concepts_input.strip():
            concepts = [
                c.strip()
                for c in related_concepts_input.split(",")
                if c.strip()
            ]

        try:
            note_id = engine.add_note(
                content=note_content.strip(),
                related_concepts=concepts if concepts else None,
            )
            st.success(f"✅ 笔记已保存 (ID: {note_id[:8]}...)")

            if concepts:
                st.info(f"已关联概念: {', '.join(concepts)}")

            # 清空输入
            st.session_state.note_content = ""
            st.session_state.related_concepts = ""
            st.rerun()
        except Exception as e:
            st.error(f"保存笔记失败: {e}")

# ============================================================================
# 笔记列表
# ============================================================================

st.divider()
st.subheader("📋 已有笔记")

try:
    notes = engine.list_notes(limit=50)
except Exception as e:
    st.error(f"获取笔记列表失败: {e}")
    notes = []

if not notes:
    st.info("暂无笔记。开始记录你的学习心得吧！")
else:
    st.caption(f"共 {len(notes)} 条笔记")

    for note in notes:
        note_id = note.record_id
        # 笔记完整内容（question 和 answer_summary 都存了完整文本）
        content = note.answer_summary or note.question
        if len(content) < 10 and note.question:
            content = note.question

        concepts = note.concepts_extracted
        ts = note.timestamp[:19] if note.timestamp else ""
        importance = note.importance

        # 展开标题：截取第一行或前 60 字
        title = content.split("\n")[0][:60]

        with st.expander(
            f"📌 {title}{'...' if len(content) > 60 else ''}  "
            f"({ts}{', ⭐' + str(importance) if importance > 0.7 else ''})"
        ):
            st.markdown(content)

            if concepts:
                st.caption(f"关联概念: {', '.join(concepts)}")

            col_meta, col_del = st.columns([6, 1])
            with col_meta:
                st.caption(f"ID: {note_id} | 重要性: {importance:.2f}")
            with col_del:
                if st.button("🗑️", key=f"del_note_{note_id}", help="删除此笔记"):
                    try:
                        engine.delete_note(note_id)
                        st.success("笔记已删除")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")

st.divider()
st.caption("💡 提示: 笔记会存入情景记忆，关联概念会更新语义知识图谱。可在「回顾」页面搜索和查看。")
