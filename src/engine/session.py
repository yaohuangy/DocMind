"""
Streamlit 会话管理——引擎单例工厂。

将 QAEngine 创建逻辑从 app.py 中分离，
使 pages/ 下的页面可以直接导入而不会触发 app.py 的 UI 渲染。
"""

import streamlit as st

from src.core.config import load_config
from src.engine.qa_engine import QAEngine


@st.cache_resource
def get_engine() -> QAEngine:
    """创建全局 QAEngine 单例（进程内缓存）。

    所有页面通过此函数获取同一个引擎实例，
    由 ``@st.cache_resource`` 保证进程内唯一。

    Returns:
        QAEngine 实例。
    """
    config = load_config()
    return QAEngine(config)
