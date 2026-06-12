"""
文档摄入流水线。

端到端流程：加载 → 分块 → 嵌入 → 向量存储 + 元数据记录。

单个方法 ``ingest()`` 完成全流程，使用者无需关心中间步骤。
"""

import hashlib
import logging
from datetime import datetime

from src.core.chunker import TextChunker
from src.core.config import ChunkConfig, get_config
from src.core.embedder import BaseEmbedder, create_embedder
from src.core.metadata_store import MetadataStore
from src.core.vector_store import VectorStore
from src.ingestion.document_loader import MultiFormatLoader

logger = logging.getLogger(__name__)


def _create_li_embed_model():
    """创建 LlamaIndex 兼容的 BaseEmbedding（从 Docmind 配置）。

    用于 SemanticSplitterNodeParser 的语义分块。
    通过 llama-index-embeddings-openai 的 OpenAIEmbedding 封装
    DashScope（或其他 OpenAI 兼容 API）。

    Returns:
        OpenAIEmbedding 实例，导入失败或创建异常时返回 None。
    """
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding  # noqa: F811
    except ImportError:
        logger.warning(
            "llama-index-embeddings-openai 未安装，语义分块不可用。"
            "安装：pip install llama-index-embeddings-openai"
        )
        return None

    cfg = get_config().embedding
    api_key = cfg.api_key or get_config().llm.api_key

    try:
        # model_name= 绕过 LlamaIndex 的 OpenAIEmbeddingModelType 枚举校验，
        # 允许 DashScope 等第三方兼容 API 的自定义模型名（如 text-embedding-v4）
        return OpenAIEmbedding(
            model_name=cfg.model,
            api_key=api_key,
            api_base=cfg.base_url.rstrip("/") if cfg.base_url else None,
            embed_batch_size=cfg.batch_size,
        )
    except Exception as e:
        logger.warning("创建 LlamaIndex OpenAIEmbedding 失败: %s", e)
        return None


class IngestPipeline:
    """端到端文档摄入流水线。

    封装了加载 → 分块 → 嵌入 → 入库的完整流程。
    调用者只需提供文件路径或 URL。

    Usage::

        pipeline = IngestPipeline()
        result = pipeline.ingest("/path/to/paper.pdf")
        print(f"文档 {result.doc_id}: {result.num_chunks} chunks")

        # 或批量
        results = pipeline.ingest_batch([
            "/path/a.pdf",
            "https://example.com/article",
            "/path/b.docx",
        ])
    """

    def __init__(
        self,
        loader: MultiFormatLoader | None = None,
        chunker: TextChunker | None = None,
        embedder: BaseEmbedder | None = None,
        vector_store: VectorStore | None = None,
        metadata_store: MetadataStore | None = None,
    ) -> None:
        """
        所有参数均可选，None 时使用默认实例。

        Args:
            loader: 多格式加载器。
            chunker: 文本分块器。
            embedder: 嵌入器。
            vector_store: 向量存储（ChromaDB）。
            metadata_store: 元数据存储（SQLite）。
        """
        self._loader = loader or MultiFormatLoader()

        # 语义分块用的 LlamaIndex 嵌入模型（与检索嵌入共用同一 API）
        self._li_embed_model = _create_li_embed_model()

        if chunker:
            self._chunker = chunker
        else:
            self._chunker = TextChunker(
                embed_model=self._li_embed_model,
            )

        self._embedder = embedder or create_embedder()
        self._vector_store = vector_store or VectorStore()
        self._metadata_store = metadata_store or MetadataStore()

        # 确保表就绪
        self._metadata_store.ensure_tables()
        self._vector_store.ensure_collection(VectorStore.DOCUMENT_CHUNKS)

    # ------------------------------------------------------------------
    # 摄入入口
    # ------------------------------------------------------------------

    def ingest(self, source: str, user_id: str = "default") -> "IngestResult":
        """摄入单个文档（文件路径或 URL）。

        流程:
        1. MultiFormatLoader 加载 → List[LlamaDocument]
        2. TextChunker 分块 → List[LlamaDocument]
        3. Embedder 嵌入 → List[List[float]]
        4. VectorStore 存入 document_chunks
        5. MetadataStore 记录文档信息

        Args:
            source: 文件路径或 URL。

        Returns:
            IngestResult，包含 doc_id, num_chunks, 来源信息等。

        Raises:
            ValueError: 格式不支持。
        """
        logger.info("=== 开始摄入: %s ===", source)

        # ---- 1. 加载 ----
        documents = self._loader.load(source)
        if not documents:
            raise RuntimeError(f"文档未提取到有效内容: {source}")

        # 从第一个文档的 metadata 提取文档元信息
        doc_id = documents[0].metadata.get("doc_id", self._make_doc_id(source))
        doc_name = documents[0].metadata.get("doc_name", source)
        doc_format = documents[0].metadata.get("format", "unknown")
        total_chars = sum(len(d.text) for d in documents)
        num_pages = self._extract_page_count(documents)

        logger.info("步骤1/4 加载完成: %d 个文档段落, %d 字符", len(documents), total_chars)

        # 如果之前摄入过同一文档，先清理旧数据（幂等）
        self._vector_store.delete_by_doc_id(
            VectorStore.DOCUMENT_CHUNKS, doc_id
        )

        # ---- 2. 分块（按文档格式自适应参数） ----
        chunk_size, chunk_overlap = get_config().chunk.get_chunk_params(doc_format)
        global_chunk = get_config().chunk
        fmt_chunk_config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_semantic_chunking=global_chunk.use_semantic_chunking,
            semantic_buffer_size=global_chunk.semantic_buffer_size,
            semantic_breakpoint_percentile=global_chunk.semantic_breakpoint_percentile,
            semantic_max_chunk_multiplier=global_chunk.semantic_max_chunk_multiplier,
            min_chunk_tokens=global_chunk.min_chunk_tokens,
        )
        fmt_chunker = TextChunker(
            config=fmt_chunk_config,
            embed_model=self._li_embed_model,
        )
        chunks = fmt_chunker.split(documents)
        if not chunks:
            raise RuntimeError(f"文档分块后无内容: {source}")

        logger.info(
            "步骤2/4 分块完成: %d → %d chunks "
            "(fmt=%s, size=%d, overlap=%d, semantic=%s)",
            len(documents), len(chunks), doc_format,
            chunk_size, chunk_overlap, global_chunk.use_semantic_chunking,
        )

        # ---- 3. 嵌入 ----
        chunk_texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(chunk_texts)
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"嵌入向量数 ({len(embeddings)}) 与 chunk 数 ({len(chunks)}) 不匹配"
            )

        logger.info("步骤3/4 嵌入完成: %d 个向量, 维度=%d", len(embeddings), len(embeddings[0]) if embeddings else 0)

        # ---- 4. 向量入库 ----
        chunk_ids: list[str] = []
        for i, chunk in enumerate(chunks):
            # 用全局序号确保唯一（PDF 等多页文档每页 chunk_index 从 0 开始会重复）
            cid = hashlib.sha256(
                f"{source}:{i}".encode()
            ).hexdigest()[:16]
            chunk_ids.append(cid)

        # 补全 chunk metadata（确保 vector_store 中有 doc_id 和 user_id）
        metadatas = []
        for chunk in chunks:
            meta = dict(chunk.metadata)
            meta["doc_id"] = doc_id
            meta["user_id"] = user_id
            metadatas.append(meta)

        self._vector_store.add_chunks(
            collection_name=VectorStore.DOCUMENT_CHUNKS,
            ids=chunk_ids,
            texts=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info("步骤4/4 向量入库完成: %d 条", len(chunk_ids))

        # ---- 5. 元数据记录 ----
        self._metadata_store.add_document(
            doc_id=doc_id,
            name=doc_name,
            source=source,
            doc_format=doc_format,
            user_id=user_id,
            num_chunks=len(chunks),
            num_pages=num_pages,
            char_count=total_chars,
            loaded_at=datetime.now().isoformat(),
        )

        result = IngestResult(
            doc_id=doc_id,
            doc_name=doc_name,
            source=source,
            doc_format=doc_format,
            num_chunks=len(chunks),
            num_pages=num_pages,
            char_count=total_chars,
            loaded_at=datetime.now().isoformat(),
        )

        logger.info("=== 摄入完成: %s ===", result)
        return result

    def ingest_batch(self, sources: list[str], user_id: str = "default") -> list["IngestResult"]:
        """批量摄入多个文档。

        每个文档独立处理，一个失败不影响后续。

        Args:
            sources: 文件路径或 URL 列表。

        Returns:
            IngestResult 列表（仅包含成功的）。
        """
        results: list[IngestResult] = []
        for i, source in enumerate(sources, 1):
            logger.info("批量摄入 [%d/%d]: %s", i, len(sources), source)
            try:
                result = self.ingest(source, user_id=user_id)
                results.append(result)
            except ValueError as e:
                logger.error("格式不支持 [%s]: %s", source, e)
            except RuntimeError as e:
                logger.error("摄入失败 [%s]: %s", source, e)
            except Exception as e:
                logger.exception("摄入异常 [%s]: %s", source, e)

        logger.info("批量摄入完成: %d/%d 成功", len(results), len(sources))
        return results

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> bool:
        """删除文档及其所有分块和元数据。

        Args:
            doc_id: 文档 ID。

        Returns:
            是否实际删除了数据。
        """
        deleted_vec = False
        try:
            self._vector_store.delete_by_doc_id(
                VectorStore.DOCUMENT_CHUNKS, doc_id
            )
            deleted_vec = True
        except Exception as e:
            logger.error("删除向量数据失败 %s: %s", doc_id, e)

        deleted_meta = self._metadata_store.delete_document(doc_id)
        return deleted_vec or deleted_meta

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _make_doc_id(source: str) -> str:
        """生成稳定的文档 ID。"""
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _extract_page_count(documents: list) -> int:
        """从 metadata 推断页数/幻灯片数。

        Args:
            documents: 解析后的文档列表。

        Returns:
            页数/幻灯片数/行数 等。
        """
        if not documents:
            return 0
        meta = documents[0].metadata
        # 各格式的页数/幻灯片数/行数
        for key in ("total_pages", "total_slides", "total_rows"):
            if key in meta:
                return int(meta[key])
        return len(documents)  # 默认返回段落数作为参考

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def loader(self) -> MultiFormatLoader:
        return self._loader

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def metadata_store(self) -> MetadataStore:
        return self._metadata_store


# ---------------------------------------------------------------------------
# 数据传输对象
# ---------------------------------------------------------------------------

class IngestResult:
    """单次摄入的结果描述。

    Attributes:
        doc_id: 文档唯一 ID。
        doc_name: 文档名称。
        source: 原始来源（路径或 URL）。
        doc_format: 格式（pdf / web / docx / md / txt / pptx / csv / xlsx）。
        num_chunks: 生成的分块总数。
        num_pages: 页数/幻灯片数（如适用）。
        char_count: 总字符数。
        loaded_at: 加载时间 ISO 字符串。
    """

    def __init__(
        self,
        doc_id: str,
        doc_name: str,
        source: str,
        doc_format: str,
        num_chunks: int,
        num_pages: int,
        char_count: int,
        loaded_at: str,
    ) -> None:
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.source = source
        self.doc_format = doc_format
        self.num_chunks = num_chunks
        self.num_pages = num_pages
        self.char_count = char_count
        self.loaded_at = loaded_at

    def __repr__(self) -> str:
        return (
            f"IngestResult(doc_id={self.doc_id!r}, name={self.doc_name!r}, "
            f"fmt={self.doc_format}, chunks={self.num_chunks}, "
            f"chars={self.char_count})"
        )

    def to_dict(self) -> dict:
        """转为可序列化字典。"""
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "source": self.source,
            "doc_format": self.doc_format,
            "num_chunks": self.num_chunks,
            "num_pages": self.num_pages,
            "char_count": self.char_count,
            "loaded_at": self.loaded_at,
        }
