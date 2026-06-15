"""
检索器抽象基类。

所有检索策略（Direct / MQE / HyDE / Combined）必须实现此接口。
提供同步和异步两个检索方法，子类至少实现 ``_retrieve_async``。
"""

import asyncio
import logging
from abc import ABC, abstractmethod

from src.core.vector_store import SearchResult

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """检索器抽象基类。

    所有检索策略必须实现 ``_retrieve_async()``。
    框架自动提供同步包装 ``retrieve()``。

    Usage::

        retriever = DirectRetriever(embedder, vector_store)
        results = retriever.retrieve("什么是Transformer？", top_k=10)
        # 或异步
        results = await retriever.retrieve_async("什么是Transformer？", top_k=10)
    """

    @abstractmethod
    async def retrieve_async(
        self,
        question: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """异步检索——子类实现。

        Args:
            question: 用户问题。
            top_k: 返回结果数。

        Returns:
            SearchResult 列表，按分数降序排列。
        """
        ...

    def retrieve(
        self,
        question: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """同步检索（自动管理事件循环）。

        智能路由：
        - 无运行中事件循环 → asyncio.run（标准路径）
        - 有运行中事件循环（如 Streamlit）→ 直接同步执行，避免
          ThreadPoolExecutor + asyncio.run 的额外开销

        Args:
            question: 用户问题。
            top_k: 返回结果数。

        Returns:
            SearchResult 列表，按分数降序排列。
        """
        try:
            _ = asyncio.get_running_loop()
            # Streamlit / Jupyter 等有事件循环的环境：
            # 不用 ThreadPoolExecutor + asyncio.run()，直接走同步路径
            return self._retrieve_sync(question, top_k)
        except RuntimeError:
            # 无事件循环：标准 asyncio 路径
            return asyncio.run(self.retrieve_async(question, top_k))

    def _retrieve_sync(
        self,
        question: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """同步检索——子类可选实现。

        默认回退到 asyncio.run()。子类应覆盖此方法以避免 asyncio 开销。
        """
        return asyncio.run(self.retrieve_async(question, top_k))
