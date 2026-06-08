"""
ChromaDB 向量存储模块。

封装 ChromaDB 嵌入式 PersistentClient，提供分块存储、检索和管理的
统一接口。所有数据持久化到本地磁盘，无需独立服务进程。

Collections 设计（见 spec §8.2）：
- ``document_chunks`` — 所有文档的分块（格式无关）
- ``episodic_memory`` — 情景记忆（Q&A 记录）
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.types import Embedding, Metadata

from src.core.config import ChromaConfig, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """单条检索结果。

    Attributes:
        chunk_id: 分块 ID。
        text: 分块文本内容。
        score: 相似度分数（0~1，由 ChromaDB distance 转换）。
        metadata: 分块元数据（doc_id, format, page_number 等）。
    """

    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """ChromaDB 嵌入式向量存储封装。

    基于 ``chromadb.PersistentClient``，数据持久化到本地路径。
    所有集合按需自动创建。

    Usage::

        store = VectorStore()
        store.ensure_collection("document_chunks")
        store.add_chunks("document_chunks", ids, texts, embeddings, metadatas)
        results = store.search("document_chunks", query_vec, limit=10)
    """

    # 默认集合名称
    DOCUMENT_CHUNKS = "document_chunks"
    EPISODIC_MEMORY = "episodic_memory"

    def __init__(self, config: Optional[ChromaConfig] = None):
        """
        Args:
            config: ChromaDB 配置。为 None 时自动从全局配置加载。
        """
        if config is None:
            config = get_config().chroma
        self._config = config
        self._persist_path = config.persist_path

        import os
        os.makedirs(self._persist_path, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._persist_path,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB 已连接，持久化路径: %s", self._persist_path)

    # ------------------------------------------------------------------
    # Collection 管理
    # ------------------------------------------------------------------

    def ensure_collection(self, name: str) -> None:
        """确保指定集合存在，不存在则创建。

        Args:
            name: 集合名称。
        """
        try:
            self._client.get_collection(name)
        except Exception:
            self._client.create_collection(name)
            logger.info("创建 ChromaDB 集合: %s", name)

    def get_collection(self, name: str):
        """获取已存在的集合。

        Args:
            name: 集合名称。

        Returns:
            chromadb.Collection 实例。

        Raises:
            ValueError: 集合不存在时。
        """
        try:
            return self._client.get_collection(name)
        except Exception:
            raise ValueError(f"集合不存在: {name}")

    def delete_collection(self, name: str) -> None:
        """删除指定集合及其所有数据。

        Args:
            name: 集合名称。
        """
        try:
            self._client.delete_collection(name)
            logger.info("删除 ChromaDB 集合: %s", name)
        except Exception:
            logger.warning("尝试删除不存在的集合: %s", name)

    # ------------------------------------------------------------------
    # 数据操作
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        collection_name: str,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """批量添加分块到指定集合。

        Args:
            collection_name: 集合名称。
            ids: 分块 ID 列表（唯一标识，建议 sha256 前 16 字符）。
            texts: 分块文本列表。
            embeddings: 嵌入向量列表，与 texts 等长。
            metadatas: 元数据列表，可包含 doc_id, format, page_number 等。
        """
        if not ids:
            return

        self.ensure_collection(collection_name)
        collection = self.get_collection(collection_name)

        # ChromaDB 要求 metadatas 中的值都是基本类型
        clean_metadatas: List[Metadata] = []
        if metadatas:
            for m in metadatas:
                clean: Dict[str, Any] = {}
                for k, v in m.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean[k] = v
                    else:
                        clean[k] = str(v)
                clean_metadatas.append(clean)

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=clean_metadatas if clean_metadatas else None,  # type: ignore[arg-type]
        )
        logger.info("向 %s 添加 %d 条分块", collection_name, len(ids))

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        limit: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """单查询向量检索。

        Args:
            collection_name: 集合名称。
            query_embedding: 查询嵌入向量。
            limit: 返回结果数上限（top-k）。
            where: ChromaDB 过滤器，如 {"format": "pdf"}。

        Returns:
            SearchResult 列表，按相似度降序排列。
        """
        collection = self.get_collection(collection_name)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,  # type: ignore[arg-type]
            include=["documents", "metadatas", "distances"],
        )

        return self._parse_results(results)

    async def search_batch(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        limit: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[List[SearchResult]]:
        """批量检索（MQE 多路查询用）。

        使用 ``asyncio.to_thread`` 将 ChromaDB 同步调用转为异步，
        以 ``asyncio.gather`` 实现并行查询，减少总延迟。

        Args:
            collection_name: 集合名称。
            query_embeddings: 多组查询嵌入向量。
            limit: 每组返回结果数上限。
            where: ChromaDB 过滤条件。

        Returns:
            嵌套列表，外层与 query_embeddings 等长。
        """

        async def _search_one(embedding: List[float]) -> List[SearchResult]:
            return await asyncio.to_thread(
                self.search, collection_name, embedding, limit, where
            )

        tasks = [_search_one(emb) for emb in query_embeddings]
        results = await asyncio.gather(*tasks)
        return list(results)

    # ------------------------------------------------------------------
    # 删除与统计
    # ------------------------------------------------------------------

    def delete_by_ids(self, collection_name: str, ids: List[str]) -> None:
        """按 ID 删除分块。

        Args:
            collection_name: 集合名称。
            ids: 要删除的分块 ID 列表。
        """
        if not ids:
            return
        try:
            collection = self.get_collection(collection_name)
            collection.delete(ids=ids)
            logger.info("从 %s 删除 %d 条分块", collection_name, len(ids))
        except ValueError:
            logger.warning("集合 %s 不存在，跳过删除", collection_name)

    def delete_by_doc_id(self, collection_name: str, doc_id: str) -> None:
        """按文档 ID 删除所有关联分块。

        通过 metadata 过滤找到属于该 doc_id 的所有分块并删除。

        Args:
            collection_name: 集合名称。
            doc_id: 文档唯一 ID。
        """
        try:
            collection = self.get_collection(collection_name)
            # 先查出所有匹配 doc_id 的分块
            existing = collection.get(
                where={"doc_id": doc_id},
                include=[],
            )
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
                logger.info("删除文档 %s 的 %d 条分块", doc_id, len(existing["ids"]))
        except ValueError:
            logger.warning("集合 %s 不存在", collection_name)

    def collection_stats(self, name: str) -> Dict[str, Any]:
        """获取集合统计信息。

        Args:
            name: 集合名称。

        Returns:
            包含 count, name 等字段的字典。
        """
        try:
            collection = self.get_collection(name)
            data = collection.get(include=[])
            return {
                "name": name,
                "count": len(data["ids"]),
            }
        except ValueError:
            return {"name": name, "count": 0}

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _parse_results(self, raw: Dict) -> List[SearchResult]:
        """将 ChromaDB 原始查询结果转为 SearchResult 列表。"""
        results: List[SearchResult] = []
        if not raw["ids"] or not raw["ids"][0]:
            return results

        ids = raw["ids"][0]
        documents = raw["documents"][0] if raw.get("documents") else [""] * len(ids)
        distances = raw["distances"][0] if raw.get("distances") else [0.0] * len(ids)
        metadatas = raw["metadatas"][0] if raw.get("metadatas") else [{}] * len(ids)

        for chunk_id, text, dist, meta in zip(ids, documents, distances, metadatas):
            # ChromaDB 返回 L2 distance，转换为余弦相似度近似分数
            # 公式: score = 1 / (1 + distance)
            score = 1.0 / (1.0 + float(dist))
            results.append(SearchResult(
                chunk_id=chunk_id,
                text=text or "",
                score=round(score, 4),
                metadata=meta or {},
            ))

        return results
