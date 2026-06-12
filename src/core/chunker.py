"""
文本分块模块。

封装 LlamaIndex SentenceSplitter，提供语义感知的文本分块能力。
默认参数：chunk_size=1024 tokens, chunk_overlap=128 tokens（可配置）。
"""

import logging

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.schema import TextNode

from src.core.config import ChunkConfig, get_config

logger = logging.getLogger(__name__)


class TextChunker:
    """文本分块器——封装 LlamaIndex SentenceSplitter。

    对 LlamaDocument 列表进行语义分块，保留原始 metadata 并追加
    分块位置信息（chunk_index, total_chunks）。

    Usage::

        chunker = TextChunker()
        chunks = chunker.split(documents)
        for chunk in chunks:
            print(chunk.text, chunk.metadata["chunk_index"])
    """

    def __init__(self, config: ChunkConfig | None = None):
        """
        Args:
            config: 分块配置。为 None 时自动从全局配置加载。
        """
        if config is None:
            config = get_config().chunk
        self._config = config

        self._splitter = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;。；]+[,.;。；]?",
            tokenizer=self._default_tokenizer,
        )

    @staticmethod
    def _default_tokenizer(text: str) -> list[str]:
        """默认 tokenizer——按空白字符分词，作为 token 计数的近似。

        SentenceSplitter 通过 tokenizer 的返回长度估算 token 数。
        在生产环境中可替换为 tiktoken 精确计数。
        """
        return text.split()

    # ------------------------------------------------------------------
    # 分块入口
    # ------------------------------------------------------------------

    def split(self, documents: list[LlamaDocument]) -> list[LlamaDocument]:
        """对一个或多个 LlamaDocument 进行分块。

        每个文档独立分块，metadata 继承自源文档并追加：
        - ``chunk_index``: 分块在文档中的序号（从 0 开始）
        - ``total_chunks``: 文档总分块数
        - ``char_count``: 分块字符数

        Args:
            documents: 待分块的 LlamaDocument 列表。

        Returns:
            分块后的 LlamaDocument 列表（平铺，顺序保持）。
        """
        all_chunks: list[LlamaDocument] = []

        for doc in documents:
            doc_chunks = self._split_one(doc)
            all_chunks.extend(doc_chunks)

        logger.info(
            "分块完成: %d 个文档 → %d 个分块 (chunk_size=%d, overlap=%d)",
            len(documents),
            len(all_chunks),
            self._config.chunk_size,
            self._config.chunk_overlap,
        )
        return all_chunks

    def _split_one(self, document: LlamaDocument) -> list[LlamaDocument]:
        """对单个文档执行分块。

        Args:
            document: 单个 LlamaDocument。

        Returns:
            分块列表。
        """
        text = document.text
        if not text or not text.strip():
            return []

        # LlamaIndex SentenceSplitter 的 get_nodes_from_documents 方法
        nodes: list[TextNode] = self._splitter.get_nodes_from_documents(
            [document]
        )

        chunks: list[LlamaDocument] = []
        total = len(nodes)

        for idx, node in enumerate(nodes):
            chunk_doc = LlamaDocument(
                text=node.text,
                metadata={
                    **document.metadata,
                    "chunk_index": idx,
                    "total_chunks": total,
                    "char_count": len(node.text),
                },
            )
            chunks.append(chunk_doc)

        return chunks

    # ------------------------------------------------------------------
    # 单文本便捷方法
    # ------------------------------------------------------------------

    def split_text(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[LlamaDocument]:
        """对单段文本进行分块（便捷方法）。

        Args:
            text: 原始文本。
            metadata: 基础元数据。

        Returns:
            分块列表。
        """
        doc = LlamaDocument(text=text, metadata=metadata or {})
        return self.split([doc])

    # ------------------------------------------------------------------
    # 配置访问
    # ------------------------------------------------------------------

    @property
    def chunk_size(self) -> int:
        """当前 chunk_size 配置。"""
        return self._config.chunk_size

    @property
    def chunk_overlap(self) -> int:
        """当前 chunk_overlap 配置。"""
        return self._config.chunk_overlap
