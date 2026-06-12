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

        Args:
            question: 用户问题。
            top_k: 返回结果数。

        Returns:
            SearchResult 列表，按分数降序排列。
        """
        try:
            _ = asyncio.get_running_loop()  # 仅检测是否有运行中的事件循环
            # 已有运行中的事件循环（如在 Jupyter/Streamlit 中）
            # 使用 nest_asyncio 或新建线程中的 loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run, self.retrieve_async(question, top_k)
                )
                return future.result()
        except RuntimeError:
            # 无运行中事件循环，直接 asyncio.run
            return asyncio.run(self.retrieve_async(question, top_k))
