"""
答案生成模块。

基于 LLM 的 RAG 答案生成，支持：
- 同步生成：返回完整答案字符串
- 流式生成：生成器逐 token yield，适配 Streamlit st.write_stream()

两种模式共享相同的 prompt 构建和上下文拼接逻辑。
"""

import logging
from collections.abc import Generator

from src.core.llm_client import LLMClient
from src.engine.models import SourceChunk
from src.generation.prompt_templates import (
    RAG_QA_NO_CONTEXT_SYSTEM,
    RAG_QA_SYSTEM,
)

logger = logging.getLogger(__name__)

# 每个分块在上下文中最多使用的字符数
_MAX_CHUNK_CHARS = 800


class AnswerGenerator:
    """RAG 答案生成器。

    基于检索到的源文档分块和用户问题，通过 LLM 生成带引用的答案。
    支持同步和流式两种模式。

    Usage::

        gen = AnswerGenerator(llm_client)
        # 流式
        for token in gen.generate_stream(question, sources):
            print(token, end="", flush=True)
        # 同步
        answer = gen.generate(question, sources)
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """
        Args:
            llm_client: LLM 客户端。None 则自动创建。
        """
        self._llm_client = llm_client or LLMClient()

    # ------------------------------------------------------------------
    # 流式生成（★ 核心方法）
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        question: str,
        sources: list[SourceChunk],
        method: str = "mqe+hyde",
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        """流式生成带引用的答案。

        每次 yield 一个 token，供 Streamlit ``st.write_stream()`` 消费。

        Args:
            question: 用户问题。
            sources: 检索到的 SourceChunk 列表。
            method: 检索策略名称（仅用于日志）。
            system_prompt: 自定义系统提示，None 则使用默认 RAG 模板。

        Yields:
            str: 每个增量 token。
        """
        messages = self._build_messages(question, sources, system_prompt)

        logger.info(
            "开始流式生成: method=%s, sources=%d, question=%s",
            method, len(sources), question[:60],
        )

        try:
            for token in self._llm_client.chat_stream(messages):
                yield token
        except Exception as e:
            logger.error("LLM 流式生成失败: %s", e)
            yield f"\n\n*[生成失败: {e}]*"

    # ------------------------------------------------------------------
    # 同步生成
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        sources: list[SourceChunk],
        method: str = "mqe+hyde",
        system_prompt: str | None = None,
    ) -> str:
        """同步生成完整答案（非流式）。

        Args:
            question: 用户问题。
            sources: SourceChunk 列表。
            method: 检索策略名称。
            system_prompt: 自定义系统提示。

        Returns:
            完整答案字符串。
        """
        messages = self._build_messages(question, sources, system_prompt)

        logger.info(
            "开始同步生成: method=%s, sources=%d", method, len(sources)
        )

        try:
            answer = self._llm_client.chat(messages)
            return answer
        except Exception as e:
            logger.error("LLM 同步生成失败: %s", e)
            return f"*[生成失败: {e}]*"

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        question: str,
        sources: list[SourceChunk],
        system_prompt: str | None = None,
    ) -> list[dict]:
        """构建 LLM 消息列表（system + user）。

        Args:
            question: 用户问题。
            sources: 源文档分块。
            system_prompt: 自定义系统提示。

        Returns:
            消息列表 [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        if sources:
            context = self._build_context(sources)
            system = system_prompt or RAG_QA_SYSTEM
            system_content = system.format(context=context, question=question)
        else:
            # 无检索结果时的回退 prompt
            system_content = RAG_QA_NO_CONTEXT_SYSTEM.format(question=question)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]

    @staticmethod
    def _build_context(sources: list[SourceChunk]) -> str:
        """构建上下文字符串（编号的文档片段）。

        每个分块格式：:

            [N] (来源: 文档名, 位置: xxx)
            分块文本内容...

        Args:
            sources: SourceChunk 列表。

        Returns:
            格式化的上下文字符串。
        """
        blocks: list[str] = []

        for i, src in enumerate(sources, 1):
            location = src.location_text or "未知位置"
            # 截断过长文本
            text = src.text
            if len(text) > _MAX_CHUNK_CHARS:
                text = text[:_MAX_CHUNK_CHARS] + "..."

            block = (
                f"[{i}] (来源: {src.doc_name}, 位置: {location})\n"
                f"{text}\n"
            )
            blocks.append(block)

        return "\n---\n".join(blocks)
