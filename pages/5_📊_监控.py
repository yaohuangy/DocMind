"""
RAG 效果评估 + 监控页面。

直接读写 SQLite，不依赖 engine 的 user_id 状态。
"""

import sqlite3
import streamlit as st
from pathlib import Path

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
# 最近反馈 —— 紧凑列表
# ============================================================================

if records:
    st.markdown("### 📜 最近反馈")
    for r in records[:10]:
        icon = "👍" if r["rating"] == "useful" else "👎"
        mname = display_names.get(r.get("method", ""), r.get("method", "?"))
        ts = r["created_at"][:19] if r.get("created_at") else "?"
        st.caption(
            f"{icon} &nbsp;|&nbsp; **{mname}** &nbsp;|&nbsp; "
            f"{r['question'][:80]} &nbsp;|&nbsp; "
            f"`{r.get('latency_sec', 0):.2f}s` &nbsp;|&nbsp; "
            f"_{ts}_"
        )
else:
    st.info("暂无反馈数据。在问答页点赞/踩后这里会出现统计。")

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
