"""
RAG 效果评估 + 监控页面。

直接读写 SQLite，不依赖 engine 的 user_id 状态。
"""

import os as _os
import sqlite3
from pathlib import Path

import streamlit as st

from src.core.config import get_config
from src.engine.session import get_engine

st.set_page_config(page_title="监控 - Docmind", page_icon="📊", layout="wide")

# ============================================================================
# 自定义 CSS —— 紧凑 Dashboard 风格（参考 Grafana / Datadog 面板）
# ============================================================================
st.markdown("""
<style>
    /* 全局容器紧凑 */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        max-width: none;
    }

    /* KPI 指标卡片 */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetric"] label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* H1/H3 间距压缩 */
    h1 {
        margin-top: 0 !important;
        padding-top: 0 !important;
        font-size: 1.8rem !important;
    }
    h3 {
        margin-top: 0.4rem !important;
        margin-bottom: 0.2rem !important;
        font-size: 0.95rem !important;
        color: #334155;
    }

    /* 分割线紧凑 */
    hr {
        margin: 0.4rem 0;
    }

    /* 图表外层 div 间距 */
    .stElementContainer {
        margin-bottom: 0 !important;
    }

    /* caption 紧凑 */
    [data-testid="stCaptionContainer"] {
        margin: 2px 0;
    }
    [data-testid="stCaptionContainer"] p {
        font-size: 0.78rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 引擎初始化 & 鉴权
# ============================================================================

engine = get_engine()
engine._metadata_store.ensure_tables()

if "user_id" not in st.session_state:
    qp_user = st.query_params.get("user_id")
    if qp_user:
        st.session_state.user_id = qp_user
        engine.set_user_id(qp_user)

if not st.session_state.get("user_id"):
    st.switch_page("app.py")

engine.set_user_id(st.session_state.user_id)
st.query_params["user_id"] = st.session_state.user_id
st.query_params["page"] = "monitor"

user = st.session_state.user_id
db_path = get_config().sqlite.path
Path(db_path).parent.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 数据加载（逻辑完全不变）
# ============================================================================

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

conn.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        question TEXT NOT NULL,
        answer_preview TEXT NOT NULL,
        method TEXT NOT NULL,
        rating TEXT NOT NULL,
        latency_sec REAL DEFAULT 0,
        created_at TEXT NOT NULL
    )
""")
conn.commit()

rows = conn.execute(
    "SELECT * FROM feedback WHERE user_id = ? ORDER BY created_at DESC",
    (user,),
).fetchall()
conn.close()

records = [dict(r) for r in rows]
total = len(records)
useful = sum(1 for r in records if r["rating"] == "useful")
not_useful = total - useful
satisfaction = useful / total if total > 0 else 0.0

by_method = {}
for r in records:
    m = r["method"]
    if m not in by_method:
        by_method[m] = {"total": 0, "useful": 0, "latencies": []}
    by_method[m]["total"] += 1
    if r["rating"] == "useful":
        by_method[m]["useful"] += 1
    if r.get("latency_sec"):
        by_method[m]["latencies"].append(r["latency_sec"])
for m in by_method:
    bm = by_method[m]
    bm["rate"] = bm["useful"] / bm["total"] if bm["total"] > 0 else 0.0
    lats = bm.pop("latencies")
    bm["avg_latency"] = sum(lats) / len(lats) if lats else 0.0

all_latencies = [r["latency_sec"] for r in records if r.get("latency_sec")]
avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

display_names = {
    "direct": "直接检索", "mqe": "MQE", "hyde": "HyDE", "mqe+hyde": "MQE+HyDE",
}

# ============================================================================
# 页面标题
# ============================================================================

st.title("📊 RAG 效果监控")

# ============================================================================
# KPI 概览行 —— 4 卡片紧凑布局
# ============================================================================

st.markdown('<div style="margin-top: -0.5rem;">', unsafe_allow_html=True)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("📊 总反馈数", total)
with kpi2:
    st.metric("✅ 满意率", f"{satisfaction:.0%}")
with kpi3:
    st.metric("⏱️ 平均检索延迟", f"{avg_latency:.2f}s")
with kpi4:
    st.metric("👍 / 👎", f"{useful} / {not_useful}")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# 图表区域 —— 左右并排，高度充足
# ============================================================================

if by_method:
    st.markdown("### 📈 检索方法对比")

    chart_left, chart_right = st.columns(2, gap="small")

    with chart_left:
        # 满意率柱状图
        chart_data_sat = {
            display_names.get(m, m): by_method[m].get("rate", 0.0)
            for m in by_method
        }
        st.caption("各方法满意率")
        st.bar_chart(
            chart_data_sat,
            horizontal=True,
            height=350,
            use_container_width=True,
        )

    with chart_right:
        # 平均检索延迟柱状图
        chart_data_lat = {
            display_names.get(m, m): by_method[m].get("avg_latency", 0.0)
            for m in by_method
        }
        st.caption("各方法平均检索延迟 (s)")
        st.bar_chart(
            chart_data_lat,
            horizontal=True,
            height=350,
            use_container_width=True,
        )

    # =========================================================================
    # 各方法明细表 —— 紧凑展示
    # =========================================================================

    st.markdown("### 📋 方法明细")

    detail_cols = st.columns(len(by_method))
    for idx, m in enumerate(by_method):
        name = display_names.get(m, m)
        d = by_method[m]
        with detail_cols[idx]:
            st.metric(
                label=name,
                value=f"{d.get('rate', 0):.0%}",
                delta=f"{d.get('total', 0)} 次 · {d.get('avg_latency', 0):.2f}s",
            )

# ============================================================================
# 最近反馈 —— 紧凑列表（每条可单独删除）
# ============================================================================

if records:
    st.markdown("### 📜 最近反馈")
    for r in records[:10]:
        rid = r["id"]
        icon = "👍" if r["rating"] == "useful" else "👎"
        mname = display_names.get(r.get("method", ""), r.get("method", "?"))
        ts = r["created_at"][:19] if r.get("created_at") else "?"
        lat = r.get("latency_sec", 0)

        col_info, col_del = st.columns([20, 1])
        with col_info:
            st.caption(
                f"{icon} &nbsp;|&nbsp; **{mname}** &nbsp;|&nbsp; "
                f"{r['question'][:80]} &nbsp;|&nbsp; "
                f"`{lat:.2f}s` &nbsp;|&nbsp; "
                f"_{ts}_"
            )
        with col_del:
            if st.button("🗑", key=f"del_fb_{rid}", help=f"删除此反馈 (ID={rid})"):
                conn_del = sqlite3.connect(db_path)
                conn_del.execute("DELETE FROM feedback WHERE id = ?", (rid,))
                conn_del.commit()
                conn_del.close()
                st.rerun()

    # 批量清理异常延迟记录
    st.markdown("---")
    with st.expander("🧹 清理异常数据"):
        threshold = st.number_input(
            "删除延迟 >= (秒) 的反馈记录",
            min_value=1.0, max_value=300.0, value=5.0, step=1.0,
            key="clean_threshold",
        )
        if st.button("🗑 清理异常延迟记录", key="clean_slow"):
            count = engine._metadata_store.delete_feedback_by_latency(user, threshold)
            st.success(f"已删除 {count} 条记录（延迟 >= {threshold}s）")
            st.rerun()
else:
    st.info("暂无反馈数据。在问答页点赞/踩后这里会出现统计。")

# ============================================================================
# Token 成本看板
# ============================================================================

# 模型定价参考（人民币/1M tokens）
MODEL_PRICING = {
    "qwen3.7-max":   {"input": 2.0,  "output": 8.0},
    "qwen3.7-plus":  {"input": 1.0,  "output": 4.0},
    "qwen3.7-flash": {"input": 0.3,  "output": 1.2},
    "qwen-turbo":    {"input": 0.3,  "output": 1.2},
    "default":       {"input": 2.0,  "output": 8.0},  # 兜底
}

# 获取当前模型名（从环境变量）
current_model = _os.getenv("LLM_MODEL", "default")
pricing = MODEL_PRICING.get(current_model, MODEL_PRICING["default"])

token_stats = engine.get_token_stats()

st.markdown("---")
st.markdown("### 💰 Token 成本分析")

if token_stats["total_calls"] > 0:
    # ---- KPI 行 ----
    tk1, tk2, tk3, tk4 = st.columns(4)

    input_cost = token_stats["total_prompt"] / 1_000_000 * pricing["input"]
    output_cost = token_stats["total_completion"] / 1_000_000 * pricing["output"]
    total_cost = input_cost + output_cost

    with tk1:
        st.metric("🔤 总 Token", f"{token_stats['total_tokens']:,}")
    with tk2:
        st.metric("📥 输入 Token", f"{token_stats['total_prompt']:,}")
    with tk3:
        st.metric("📤 输出 Token", f"{token_stats['total_completion']:,}")
    with tk4:
        st.metric("💵 预估费用", f"¥{total_cost:.4f}",
                  help=f"模型: {current_model} | 输入 ¥{pricing['input']}/M | 输出 ¥{pricing['output']}/M")

    # ---- 按方法图表 ----
    if token_stats["by_method"]:
        st.markdown("#### 按检索方法")
        chart_col, table_col = st.columns([3, 2])

        with chart_col:
            # Token 用量柱状图
            chart_tokens = {
                display_names.get(m, m): d["total"]
                for m, d in token_stats["by_method"].items()
            }
            st.caption("各方法 Token 消耗")
            st.bar_chart(chart_tokens, horizontal=True, height=280, use_container_width=True)

            # 成本柱状图
            chart_cost = {}
            for m, d in token_stats["by_method"].items():
                method_cost = (d["prompt"] * pricing["input"] + d["completion"] * pricing["output"]) / 1_000_000
                chart_cost[display_names.get(m, m)] = round(method_cost, 4)
            st.caption("各方法预估费用 (¥)")
            st.bar_chart(chart_cost, horizontal=True, height=280, use_container_width=True)

        with table_col:
            # 明细表
            st.caption("方法明细")
            for m, d in token_stats["by_method"].items():
                mname = display_names.get(m, m)
                mcost = (d["prompt"] * pricing["input"] + d["completion"] * pricing["output"]) / 1_000_000
                st.markdown(
                    f"**{mname}**  \n"
                    f"调用 {d['calls']} 次 · {d['total']:,} token  \n"
                    f"输入 {d['prompt']:,} / 输出 {d['completion']:,}  \n"
                    f"约 ¥{mcost:.4f}"
                )

    # ---- 近期记录 ----
    st.markdown("#### 📜 近期 Token 记录")
    for r in token_stats["recent"][:10]:
        tid = r["id"]
        mname = display_names.get(r["method"], r["method"])
        ts = r["created_at"][:19] if r.get("created_at") else "?"

        col_tinfo, col_tdel = st.columns([20, 1])
        with col_tinfo:
            st.caption(
                f"**{mname}** &nbsp;|&nbsp; "
                f"📥 {r['prompt_tokens']:,} + 📤 {r['completion_tokens']:,} "
                f"= {r['total_tokens']:,} token &nbsp;|&nbsp; "
                f"_{ts}_"
            )
        with col_tdel:
            if st.button("🗑", key=f"del_tk_{tid}", help=f"删除此 Token 记录 (ID={tid})"):
                engine._metadata_store.delete_token_usage_record(tid)
                st.rerun()
else:
    if token_stats["total_calls"] == 0:
        st.info("暂无 Token 用量数据。进行一次问答后这里会出现统计。")
    else:
        # total_calls 为 0 时的特殊处理（仅首次加载可能）
        st.info("暂无 Token 用量数据。")

st.divider()
st.caption("💡 每次问答后点击 👍 或 👎 即为系统提供反馈。")

# ============================================================================
# 调试信息（收起到页面底部）
# ============================================================================

with st.expander("🔧 调试信息"):
    st.write(f"当前用户: `{user}`")
    st.write(f"数据库: `{db_path}`")
    st.write(f"feedback 表行数 (该用户): {total}")
    if records:
        st.write("最近 3 条:", records[:3])
    else:
        conn2 = sqlite3.connect(db_path)
        conn2.row_factory = sqlite3.Row
        all_rows = conn2.execute("SELECT * FROM feedback").fetchall()
        conn2.close()
        st.write(f"feedback 表总行数 (不限用户): {len(all_rows)}")
        if all_rows:
            st.write("示例:", [dict(r) for r in all_rows[:3]])
