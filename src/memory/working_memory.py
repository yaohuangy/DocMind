"""
工作记忆模块。

维护当前会话的最近 N 轮对话上下文（默认 10 轮）。
数据仅存于进程内存中（`st.session_state`），浏览器刷新即丢失。

设计：实例自身存储所有状态，可直接放入 `st.session_state` 中。
"""

import logging
from datetime import datetime

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
        self._entries: list[WorkingMemoryEntry] = []
        self._active_concepts: list[str] = []
        self._summary: str = ""  # 旧对话 LLM 压缩摘要

    # ------------------------------------------------------------------
    # 基本操作
    # ------------------------------------------------------------------

    def add(
        self,
        question: str,
        answer: str,
        sources: list[str] | None = None,
        concepts: list[str] | None = None,
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
        """获取对话上下文，供 LLM 作为附加上下文。

        优先返回压缩摘要（历史对话） + 最近 N 轮原始对话。

        Args:
            last_n: 取最近 N 轮原始对话。

        Returns:
            格式化的上下文文本。
        """
        parts: list[str] = []

        # 压缩摘要（更早的对话）
        if self._summary:
            parts.append(
                "## 历史对话摘要\n"
                "以下是此前对话的关键信息摘要，请基于此了解用户已讨论过的内容：\n"
                f"{self._summary}"
            )

        # 最近 N 轮原始对话
        if self._entries:
            recent = self._entries[-last_n:]
            lines: list[str] = []
            for entry in recent:
                lines.append(f"用户: {entry.question}")
                lines.append(f"助手: {entry.answer}")
            if lines:
                parts.append("## 最近对话\n" + "\n---\n".join(lines))

        return "\n\n".join(parts) if parts else ""

    def get_recent(self, n: int = 5) -> list[WorkingMemoryEntry]:
        """获取最近 N 条记录。

        Args:
            n: 返回条数。

        Returns:
            WorkingMemoryEntry 列表。
        """
        return self._entries[-n:]

    def get_active_concepts(self) -> list[str]:
        """获取当前会话中活跃的概念列表。

        Returns:
            概念名称列表。
        """
        return list(self._active_concepts)

    def clear(self) -> None:
        """清空工作记忆。"""
        self._entries.clear()
        self._active_concepts.clear()
        self._summary = ""
        logger.info("工作记忆已清空")

    # ------------------------------------------------------------------
    # 摘要
    # ------------------------------------------------------------------

    @property
    def summary(self) -> str:
        """当前压缩摘要。"""
        return self._summary

    def set_summary(self, text: str) -> None:
        """设置压缩摘要（由 MemoryManager 调用）。

        Args:
            text: LLM 生成的摘要文本。
        """
        self._summary = text

    def should_compress(self, threshold: int = 10) -> bool:
        """是否需要压缩——条目数超过阈值。

        Args:
            threshold: 触发压缩的条目数阈值（默认 10 条）。

        Returns:
            是否需要压缩。
        """
        return len(self._entries) > threshold

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
    def entries(self) -> list[WorkingMemoryEntry]:
        """所有条目（只读）。"""
        return list(self._entries)
