"""
MQE + HyDE 组合检索器。

两个 LLM 调用（变体生成 + 假设文档生成）互不依赖，
通过 ``asyncio.gather`` 并行执行，降低总延迟。

流程（见 spec §9.3）::

    原问题 ─┬→ LLM生成变体(4个) → 并行检索 → RRF ─┐
            │                                      ├→ 加权合并 → top-10
            └→ LLM生成假设文档 → embed → search ──┘

    asyncio.gather: 两个 LLM 调用并行，互不依赖

加权比例：MQE(direct) 权重 0.4, HyDE 权重 0.6。
"""

import asyncio
import logging

from src.core.config import RetrievalConfig, get_config
from src.core.embedder import BaseEmbedder, create_embedder
from src.core.llm_client import LLMClient
from src.core.vector_store import SearchResult, VectorStore
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.fusion import weighted_merge
from src.retrieval.hyde_retriever import HyDERetriever
from src.retrieval.mqe_retriever import MQERetriever

logger = logging.getLogger(__name__)


class CombinedRetriever(BaseRetriever):
    """MQE + HyDE 组合检索器。

    并行执行 MQE 和 HyDE 两个分支，通过加权合并融合结果。
    MQE 分支负责提升召回覆盖，HyDE 分支负责语义对齐。

    Usage::

        retriever = CombinedRetriever(llm_client, embedder, vector_store)
        results = retriever.retrieve("什么是Transformer？", top_k=10)
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        embedder: BaseEmbedder | None = None,
        vector_store: VectorStore | None = None,
        config: RetrievalConfig | None = None,
        where_filter: dict | None = None,
    ) -> None:
        """
        Args:
            llm_client: LLM 客户端（共享给 MQE 和 HyDE 分支）。
            embedder: 嵌入器。
            vector_store: 向量存储。
            config: 检索配置（含 hyde_weight, mqe_num_variants 等）。
            where_filter: ChromaDB where 过滤条件。
        """
        if llm_client is None:
            llm_client = LLMClient()
        if embedder is None:
            embedder = create_embedder()
        if vector_store is None:
            vector_store = VectorStore()
        if config is None:
            config = get_config().retrieval

        self._llm_client = llm_client
        self._embedder = embedder
        self._vector_store = vector_store
        self._config = config

        # 创建子检索器（传递 user_id 过滤）
        self._mqe_retriever = MQERetriever(
            llm_client=llm_client,
            embedder=embedder,
            vector_store=vector_store,
            num_variants=config.mqe_num_variants,
            where_filter=where_filter,
        )
        self._hyde_retriever = HyDERetriever(
            llm_client=llm_client,
            embedder=embedder,
            vector_store=vector_store,
            where_filter=where_filter,
        )

    async def retrieve_async(
        self,
        question: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """MQE + HyDE 组合检索——两个分支并行执行。

        Args:
            question: 用户问题。
            top_k: 返回结果数。

        Returns:
            加权合并后的 SearchResult 列表。
        """
        hyde_weight = self._config.hyde_weight
        direct_weight = 1.0 - hyde_weight  # 默认 0.4

        logger.info(
            "Combined 检索开始: MQE(w=%.1f) || HyDE(w=%.1f)",
            direct_weight, hyde_weight,
        )

        # ---- 并行执行两个分支 ----
        # asyncio.gather: 两个 LLM 调用互不依赖，同时发出
        mqe_task = self._mqe_retriever.retrieve_async(question, top_k=top_k)
        hyde_task = self._hyde_retriever.retrieve_async(question, top_k=top_k)

        mqe_results, hyde_results = await asyncio.gather(
            mqe_task, hyde_task
        )

        logger.info(
            "Combined 分支完成: MQE=%d 结果, HyDE=%d 结果",
            len(mqe_results), len(hyde_results),
        )

        # ---- 加权合并 ----
        merged = weighted_merge(
            results_a=mqe_results,
            results_b=hyde_results,
            weight_a=direct_weight,
            weight_b=hyde_weight,
            top_k=top_k,
            normalize=True,
        )

        logger.info(
            "Combined 检索完成: MQE(%d) + HyDE(%d) → 合并 %d 结果",
            len(mqe_results), len(hyde_results), len(merged),
        )
        return merged
