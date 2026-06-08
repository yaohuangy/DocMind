"""
情景记忆模块。

使用 ChromaDB ``episodic_memory`` collection 持久化存储 Q&A 交互记录。
每条记录包含问题、答案摘要、时间戳、重要性评分、来源引用等。

支持关键词搜索（通过嵌入相似度）和时间范围过滤。
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.embedder import BaseEmbedder, create_embedder
from src.core.vector_store import SearchResult, VectorStore
from src.memory.models import EpisodicMemoryRecord

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """情景记忆——基于 ChromaDB 持久化存储 Q&A 记录。

    每次问答后自动记录，支持按关键词（语义搜索）和时间检索。
    Embedding 由「问题 + 答案摘要」拼接生成。

    Usage::

        em = EpisodicMemory(embedder, vector_store)
        em.record(question="什么是Transformer？", answer_summary="Transformer是...",
                  source_chunks=[...], concepts=["Self-Attention"], importance=0.85)
        results = em.search("Transformer 自注意力", limit=5)
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        """
        Args:
            embedder: 嵌入器。None 则自动创建。
            vector_store: 向量存储。None 则自动创建。
        """
        self._embedder = embedder or create_embedder()
        self._vector_store = vector_store or VectorStore()

        # 确保集合存在
        self._collection_name = VectorStore.EPISODIC_MEMORY
        self._vector_store.ensure_collection(self._collection_name)

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------

    def record(
        self,
        question: str,
        answer_summary: str,
        source_chunks: Optional[List[str]] = None,
        documents: Optional[List[str]] = None,
        concepts_extracted: Optional[List[str]] = None,
        importance: float = 0.5,
        session_id: str = "",
        user_id: str = "default_user",
        event_type: str = "qa_interaction",
        timestamp: Optional[str] = None,
    ) -> str:
        """记录一条 Q&A 交互到情景记忆。

        自动生成嵌入向量并存入 ChromaDB。

        Args:
            question: 用户问题全文。
            answer_summary: 答案摘要。
            source_chunks: 引用的 chunk_id 列表。
            documents: 涉及的文档名列表。
            concepts_extracted: 提取的概念名称列表。
            importance: 重要性评分（0~1）。
            session_id: 会话 ID。
            user_id: 用户 ID。
            event_type: 事件类型。
            timestamp: 时间戳，None 则取当前时间。

        Returns:
            记录的 UUID。
        """
        record_id = uuid.uuid4().hex[:16]
        ts = timestamp or datetime.now().isoformat()

        record = EpisodicMemoryRecord(
            record_id=record_id,
            question=question,
            answer_summary=answer_summary,
            source_chunks=source_chunks or [],
            documents=documents or [],
            concepts_extracted=concepts_extracted or [],
            importance=importance,
            session_id=session_id,
            user_id=user_id,
            timestamp=ts,
            event_type=event_type,
        )

        # 嵌入
        embed_text = record.to_embedding_text()
        embeddings = self._embedder.embed([embed_text])

        # 存储
        self._vector_store.add_chunks(
            collection_name=self._collection_name,
            ids=[record_id],
            texts=[embed_text],
            embeddings=embeddings,
            metadatas=[record.to_metadata()],
        )

        logger.info(
            "情景记忆记录: id=%s, importance=%.2f, concepts=%d",
            record_id, importance, len(record.concepts_extracted),
        )
        return record_id

    # ------------------------------------------------------------------
    # 搜索
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 10,
        min_importance: float = 0.0,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[EpisodicMemoryRecord]:
        """按关键词搜索情景记忆（语义搜索）。

        Args:
            query: 搜索关键词。
            limit: 返回结果数上限。
            min_importance: 最低重要性过滤（0~1）。
            event_type: 按事件类型过滤，None 则不过滤。
            user_id: 按用户过滤，None 则不过滤。

        Returns:
            EpisodicMemoryRecord 列表，按相似度降序。
        """
        query_embedding = self._embedder.embed_query(query)

        # 构建 ChromaDB where 过滤条件
        conditions: List[Dict[str, Any]] = []
        if event_type:
            conditions.append({"event_type": event_type})
        if user_id:
            conditions.append({"user_id": user_id})

        where: Optional[Dict[str, Any]] = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        results = self._vector_store.search(
            collection_name=self._collection_name,
            query_embedding=query_embedding,
            limit=limit,
            where=where,
        )

        records = [self._to_record(r) for r in results]

        # 客户端侧重要性过滤（ChromaDB 的 where 不支持浮点比较）
        if min_importance > 0:
            records = [r for r in records if r.importance >= min_importance]

        logger.info("情景记忆搜索: \"%s\" → %d 条记录", query[:50], len(records))
        return records

    def search_by_time(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50,
        user_id: Optional[str] = None,
    ) -> List[EpisodicMemoryRecord]:
        """按时间范围搜索情景记忆。

        通过 ChromaDB 的 metadata 过滤实现。
        注意：ChromaDB 的 where 仅支持精确匹配，时间范围过滤在客户端侧完成。

        Args:
            start_time: 开始时间 ISO 字符串。
            end_time: 结束时间 ISO 字符串。
            limit: 返回结果数上限。
            user_id: 按用户过滤。

        Returns:
            EpisodicMemoryRecord 列表。
        """
        # 获取所有记录（ChromaDB 不支持按字符串比较范围过滤）
        # 客户端侧过滤
        try:
            collection = self._vector_store.get_collection(self._collection_name)
            data = collection.get(
                include=["metadatas", "documents"],
                limit=limit,
            )
        except ValueError:
            return []

        if not data["ids"]:
            return []

        records: List[EpisodicMemoryRecord] = []
        for i, cid in enumerate(data["ids"]):
            meta = data["metadatas"][i] if data.get("metadatas") else {}
            record = EpisodicMemoryRecord.from_metadata(meta, record_id=cid)

            # 按用户过滤
            if user_id and record.user_id != user_id:
                continue

            # 时间范围过滤
            ts = record.timestamp
            if start_time and ts < start_time:
                continue
            if end_time and ts > end_time:
                continue

            records.append(record)

        # 按时间降序排列
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    # ------------------------------------------------------------------
    # 管理
    # ------------------------------------------------------------------

    def delete_record(self, record_id: str) -> bool:
        """删除一条记录。

        Args:
            record_id: 记录 ID。

        Returns:
            是否删除了数据（当前总是返回 True，ChromaDB delete 不返回计数）。
        """
        self._vector_store.delete_by_ids(self._collection_name, [record_id])
        logger.info("情景记忆删除: %s", record_id)
        return True

    def get_record_count(self) -> int:
        """获取记录总数。"""
        stats = self._vector_store.collection_stats(self._collection_name)
        return stats.get("count", 0)

    def clear_all(self) -> None:
        """清空所有情景记忆（危险操作）。"""
        self._vector_store.delete_collection(self._collection_name)
        self._vector_store.ensure_collection(self._collection_name)
        logger.warning("情景记忆已全部清空")

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _to_record(result: SearchResult) -> EpisodicMemoryRecord:
        """将 SearchResult 转为 EpisodicMemoryRecord。"""
        return EpisodicMemoryRecord.from_metadata(
            result.metadata,
            record_id=result.chunk_id,
        )
