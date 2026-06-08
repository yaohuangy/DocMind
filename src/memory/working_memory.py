"""
工作记忆模块。

维护当前会话的最近 N 轮对话上下文（默认 10 轮）。
数据仅存于进程内存中（`st.session_state`），浏览器刷新即丢失。

设计：实例自身存储所有状态，可直接放入 `st.session_state` 中。
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from src.memory.models import WorkingMemoryEntry

logger = logging.getLogger(__name__)

# 默认保留的最近对话轮数
_DEFAULT_MAX_ENTRIES = 10


class WorkingMemory:
    """工作记忆——管理当前会话的短期对话上下文。

    存储最近 N 轮 Q&A 的摘要，为 LLM 提供近期对话上下文。
    实例可直接存入 Streamlit ``st.session_state``。

    Usage::

        wm = WorkingMemory(session_id="session_001", max_entries=10)
        wm.add("什么是Transformer？", "Transformer是一种...", sources=[...], concepts=[...])
        context = wm.get_context()  # 格式化后的近期对话文本
    """

    def __init__(
        self,
        session_id: str = "default",
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> None:
        """
        Args:
            session_id: 会话唯一标识。
            max_entries: 最大保留条目数，默认 10。
        """
        self.session_id = session_id
        self.max_entries = max_entries
        self._entries: List[WorkingMemoryEntry] = []
        self._active_concepts: List[str] = []

    # ------------------------------------------------------------------
    # 基本操作
    # ------------------------------------------------------------------

    def add(
        self,
        question: str,
        answer: str,
        sources: Optional[List[str]] = None,
        concepts: Optional[List[str]] = None,
    ) -> None:
        """添加一轮 Q&A 到工作记忆。

        超过 max_entries 时自动淘汰最早记录（FIFO）。

        Args:
            question: 用户问题。
            answer: 模型答案（会自动截取前 200 字符作为摘要）。
            sources: 引用的源文档列表。
            concepts: 提取的关键概念名称。
        """
        entry = WorkingMemoryEntry(
            question=question,
            answer=answer[:200] + ("..." if len(answer) > 200 else ""),
            sources=sources or [],
            concepts=concepts or [],
            timestamp=datetime.now().isoformat(),
        )

        self._entries.append(entry)

        # 淘汰
        while len(self._entries) > self.max_entries:
            removed = self._entries.pop(0)
            logger.debug("工作记忆淘汰: %s", removed.question[:40])

        # 更新活跃概念
        if concepts:
            self._active_concepts = list(
                dict.fromkeys(self._active_concepts + concepts)
            )[-20:]  # 最近 20 个去重

        logger.info("工作记忆: +1 条目 (总计 %d/%d)", len(self._entries), self.max_entries)

    def get_context(self, last_n: int = 5) -> str:
        """获取最近的对话上下文文本，供 LLM 作为附加上下文。

        Args:
            last_n: 取最近 N 轮。

        Returns:
            格式化的多轮对话文本，如::

                用户: 什么是Transformer？
                助手: Transformer是一种基于自注意力...
                ---
                用户: 自注意力机制如何工作？
                助手: 自注意力通过...

            若无记录则返回空字符串。
        """
        if not self._entries:
            return ""

        recent = self._entries[-last_n:]
        lines: List[str] = []

        for entry in recent:
            lines.append(f"用户: {entry.question}")
            lines.append(f"助手: {entry.answer}")

        return "\n---\n".join(lines)

    def get_recent(self, n: int = 5) -> List[WorkingMemoryEntry]:
        """获取最近 N 条记录。

        Args:
            n: 返回条数。

        Returns:
            WorkingMemoryEntry 列表。
        """
        return self._entries[-n:]

    def get_active_concepts(self) -> List[str]:
        """获取当前会话中活跃的概念列表。

        Returns:
            概念名称列表。
        """
        return list(self._active_concepts)

    def clear(self) -> None:
        """清空工作记忆。"""
        self._entries.clear()
        self._active_concepts.clear()
        logger.info("工作记忆已清空")

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """当前条目数。"""
        return len(self._entries)

    @property
    def is_empty(self) -> bool:
        """是否为空。"""
        return len(self._entries) == 0

    @property
    def entries(self) -> List[WorkingMemoryEntry]:
        """所有条目（只读）。"""
        return list(self._entries)
