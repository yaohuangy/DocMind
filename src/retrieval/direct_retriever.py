"""
直接向量检索器。

将用户问题嵌入后，直接在 ChromaDB 中检索最相似的分块。
不涉及 LLM 调用，是检索策略中最简单、延迟最低的模式。
"""

import logging

from src.core.embedder import BaseEmbedder, create_embedder
from src.core.vector_store import SearchResult, VectorStore
from src.retrieval.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class DirectRetriever(BaseRetriever):
    """直接向量检索——embed 问题 → ChromaDB 搜索。

    无 LLM 参与，适合对延迟敏感的快速检索场景。

    Usage::

        retriever = DirectRetriever(embedder, vector_store)
        results = retriever.retrieve("什么是Transformer？", top_k=10)
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        vector_store: VectorStore | None = None,
        where_filter: dict | None = None,
    ) -> None:
        """
        Args:
            embedder: 嵌入器。None 则自动创建。
            vector_store: 向量存储。None 则自动创建。
            where_filter: ChromaDB where 过滤条件（如 {"user_id": "Yao"}）。
        """
        self._embedder = embedder or create_embedder()
        self._vector_store = vector_store or VectorStore()
        self._where_filter = where_filter

    async def retrieve_async(
        self,
        question: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """直接向量检索。

        Args:
            question: 用户问题。
            top_k: 返回结果数。

        Returns:
            SearchResult 列表，按相似度分数降序排列。
        """
        import asyncio

        # 嵌入查询（在线程池中执行，不阻塞事件循环）
        query_embedding = await asyncio.to_thread(
            self._embedder.embed_query, question
        )

        # 检索（带用户过滤）
        results = self._vector_store.search(
            collection_name=VectorStore.DOCUMENT_CHUNKS,
            query_embedding=query_embedding,
            limit=top_k,
            where=self._where_filter,
        )

        logger.info(
            "直接检索完成: \"%s\" → %d 结果", question[:50], len(results)
        )
        return results
