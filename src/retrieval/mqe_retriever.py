"""
MQE（多查询扩展）检索器。

流程：
1. LLM 生成 4 个查询变体（定义 / 机制 / 对比 / 应用角度）
2. 原问题 + 4 变体 → 5 路并行检索，每路 top-10
3. Reciprocal Rank Fusion (k=60) 融合排序
4. 去重，返回 top-10
"""

import asyncio
import logging

from src.core.embedder import BaseEmbedder, create_embedder
from src.core.llm_client import LLMClient
from src.core.vector_store import SearchResult, VectorStore
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class MQERetriever(BaseRetriever):
    """MQE 多查询扩展检索器。

    使用 LLM 从 4 个角度生成查询变体，
    并行检索后通过 RRF 融合多路结果。

    Usage::

        retriever = MQERetriever(llm_client, embedder, vector_store)
        results = retriever.retrieve("什么是Transformer？", top_k=10)
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        embedder: BaseEmbedder | None = None,
        vector_store: VectorStore | None = None,
        num_variants: int = 4,
        where_filter: dict | None = None,
    ) -> None:
        """
        Args:
            llm_client: LLM 客户端（用于生成查询变体）。
            embedder: 嵌入器。
            vector_store: 向量存储。
            num_variants: 生成的变体数，默认 4。
            where_filter: ChromaDB where 过滤条件。
        """
        self._llm_client = llm_client or LLMClient()
        self._embedder = embedder or create_embedder()
        self._vector_store = vector_store or VectorStore()
        self._num_variants = num_variants
        self._where_filter = where_filter

    async def retrieve_async(
        self,
        question: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """MQE 检索——变体生成 → 多路并行检索 → RRF 融合。

        Args:
            question: 用户问题。
            top_k: 返回结果数。

        Returns:
            RRF 融合后的 SearchResult 列表。
        """
        # ---- 1. 生成查询变体 ----
        variants = await asyncio.to_thread(
            self._llm_client.generate_query_variants,
            question,
            self._num_variants,
        )

        # 原问题 + 变体
        all_queries = [question] + variants
        logger.info(
            "MQE: 生成 %d 个变体, 共 %d 路查询",
            len(variants), len(all_queries),
        )

        # ---- 2. 嵌入所有查询 ----
        query_embeddings = await asyncio.to_thread(
            self._embedder.embed, all_queries
        )

        # ---- 3. 并行检索（asyncio.gather） ----
        per_query_limit = top_k  # 每路返回 top_k 条，给 RRF 足够的候选
        batch_results = await self._vector_store.search_batch(
            collection_name=VectorStore.DOCUMENT_CHUNKS,
            query_embeddings=query_embeddings,
            limit=per_query_limit,
            where=self._where_filter,
        )

        logger.info(
            "MQE: %d 路检索完成, 各路结果数: %s",
            len(batch_results),
            [len(r) for r in batch_results],
        )

        # ---- 4. RRF 融合 ----
        fused = reciprocal_rank_fusion(
            result_lists=batch_results,
            k=60,
            top_k=top_k,
        )

        logger.info("MQE 检索完成: %d 路 → RRF → %d 结果", len(batch_results), len(fused))
        return fused
