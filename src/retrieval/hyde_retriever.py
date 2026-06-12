"""
HyDE（假设文档嵌入）检索器。

流程：
1. LLM 生成 200-400 字假设答案（无检索上下文）
2. 将假设答案嵌入作为查询向量
3. 在 ChromaDB 中检索 top-10

核心思路：假设答案的嵌入向量比问题本身更接近真实文档，
因为假设答案使用了与文档相似的表述方式。
"""

import asyncio
import logging

from src.core.embedder import BaseEmbedder, create_embedder
from src.core.llm_client import LLMClient
from src.core.vector_store import SearchResult, VectorStore
from src.retrieval.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class HyDERetriever(BaseRetriever):
    """HyDE 假设文档嵌入检索器。

    先生成假设答案，再用假设答案的嵌入去检索真实文档。
    适用于问题与文档表述风格差异较大的场景。

    Usage::

        retriever = HyDERetriever(llm_client, embedder, vector_store)
        results = retriever.retrieve("什么是Transformer？", top_k=10)
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        embedder: BaseEmbedder | None = None,
        vector_store: VectorStore | None = None,
        where_filter: dict | None = None,
    ) -> None:
        """
        Args:
            llm_client: LLM 客户端（用于生成假设答案）。
            embedder: 嵌入器。
            vector_store: 向量存储。
            where_filter: ChromaDB where 过滤条件。
        """
        self._llm_client = llm_client or LLMClient()
        self._embedder = embedder or create_embedder()
        self._vector_store = vector_store or VectorStore()
        self._where_filter = where_filter

    async def retrieve_async(
        self,
        question: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """HyDE 检索——生成假设答案 → 嵌入 → 检索。

        Args:
            question: 用户问题。
            top_k: 返回结果数。

        Returns:
            SearchResult 列表。
        """
        # ---- 1. 生成假设答案 ----
        hypothetical_answer = await asyncio.to_thread(
            self._llm_client.generate_hypothetical_answer, question
        )

        if not hypothetical_answer:
            logger.warning("HyDE: 假设答案生成为空，回退到直接嵌入原问题")
            query_text = question
        else:
            query_text = hypothetical_answer
            logger.info(
                "HyDE: 假设答案生成完成 (%d 字符)", len(hypothetical_answer)
            )

        # ---- 2. 嵌入假设答案 ----
        query_embedding = await asyncio.to_thread(
            self._embedder.embed_query, query_text
        )

        # ---- 3. 检索 ----
        results = self._vector_store.search(
            collection_name=VectorStore.DOCUMENT_CHUNKS,
            query_embedding=query_embedding,
            limit=top_k,
            where=self._where_filter,
        )

        logger.info("HyDE 检索完成: %d 结果", len(results))
        return results
