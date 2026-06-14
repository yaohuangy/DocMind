"""
文本分块模块。

支持两种分块策略：
- **语义分块** (SemanticSplitterNodeParser)：按句子嵌入相似度在话题边界切分，
  超大块回退到 token 分块。
- **Token 分块** (SentenceSplitter)：按 token 数切分，优先在句子边界断。
- **标题分块**：Markdown 文档按 ## / ### 标题切分。

使用 tiktoken 精确计数（支持中英文）。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument

from src.core.config import ChunkConfig, get_config

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.node_parser import SemanticSplitterNodeParser

logger = logging.getLogger(__name__)

# tiktoken 编码器缓存
_tokenizer_cache: dict[str, object] = {}


def _get_encoding():
    """获取 tiktoken cl100k_base 编码器（兼容多数嵌入模型）。"""
    if "cl100k_base" not in _tokenizer_cache:
        import tiktoken
        _tokenizer_cache["cl100k_base"] = tiktoken.get_encoding("cl100k_base")
    return _tokenizer_cache["cl100k_base"]


class TextChunker:
    """文本分块器。

    支持三种分块路径：
    - Markdown → 按 ## / ### 标题切分
    - 语义模式 → SemanticSplitterNodeParser（超大块回退 token 分块）
    - Token 模式 → SentenceSplitter

    Usage::

        # Token 模式（默认）
        chunker = TextChunker()
        chunks = chunker.split(documents)

        # 语义模式
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(...)
        chunker = TextChunker(embed_model=embed_model)
        chunks = chunker.split(documents)
    """

    def __init__(
        self,
        config: ChunkConfig | None = None,
        embed_model: BaseEmbedding | None = None,
    ):
        """
        Args:
            config: 分块配置。为 None 时自动从全局配置加载。
            embed_model: LlamaIndex 兼容的嵌入模型。提供时启用语义分块。
        """
        if config is None:
            config = get_config().chunk
        self._config = config
        self._embed_model = embed_model
        self._semantic_splitter: SemanticSplitterNodeParser | None = None

        # Token 分块器始终创建（作为兜底和 md 超大段落回退）
        self._splitter = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;。；]+[,.;。；]?",
            tokenizer=self._tokenizer,
        )

        self._use_semantic = (
            config.use_semantic_chunking and embed_model is not None
        )

    @staticmethod
    def _tokenizer(text: str) -> list[str]:
        """tiktoken 精确 tokenizer——支持中英文混合文本。

        将文本编码为 token ID 列表，SentenceSplitter 通过列表长度
        估算真实 token 数，避免中文无空格导致的计数偏差。
        """
        enc = _get_encoding()
        tokens = enc.encode(text, disallowed_special=())
        # 返回占位列表，长度 = token 数量（SentenceSplitter 只用 len()）
        return [""] * len(tokens)

    # ------------------------------------------------------------------
    # 分块入口
    # ------------------------------------------------------------------

    def split(self, documents: list[LlamaDocument]) -> list[LlamaDocument]:
        """对一个或多个 LlamaDocument 进行分块。

        - **Markdown**：已由 MarkdownParser 按 ## / ### 标题预切分，
          直接作为分块（超大段落回退 SentenceSplitter）。
        - **语义模式**：非 md 文档使用 SemanticSplitterNodeParser
          按话题边界切分，超大块回退 SentenceSplitter。
        - **Token 模式**：使用 SentenceSplitter 按 token 数分块。

        分块后合并过小碎片（< min_chunk_tokens）。

        Args:
            documents: 待分块的 LlamaDocument 列表。

        Returns:
            分块后的 LlamaDocument 列表（平铺，顺序保持）。
        """
        md_docs = [d for d in documents if d.metadata.get("format") == "md"]
        other_docs = [d for d in documents if d.metadata.get("format") != "md"]

        all_chunks: list[LlamaDocument] = []

        # ---- Markdown：按标题已预切分 ----
        total_md = len(md_docs)
        for idx, doc in enumerate(md_docs):
            text = doc.text
            if not text or not text.strip():
                continue
            token_count = len(self._tokenizer(text))
            if token_count > self._config.chunk_size * 2:
                sub_chunks = self._split_one(doc)
                all_chunks.extend(sub_chunks)
            else:
                chunk_doc = LlamaDocument(
                    text=text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "total_chunks": total_md,
                        "char_count": len(text),
                        "chunker": "heading",
                    },
                )
                all_chunks.append(chunk_doc)

        # ---- 其他格式 ----
        for doc in other_docs:
            if self._use_semantic:
                doc_chunks = self._split_semantic(doc)
            else:
                doc_chunks = self._split_one(doc)
            all_chunks.extend(doc_chunks)

        # ---- 后处理：合并过小碎片 ----
        all_chunks = self._merge_tiny_chunks(all_chunks)

        logger.info(
            "分块完成: %d 个文档 → %d 个分块 "
            "(size=%d, overlap=%d, md=%d, semantic=%s)",
            len(documents), len(all_chunks),
            self._config.chunk_size, self._config.chunk_overlap,
            len(md_docs), self._use_semantic,
        )
        return all_chunks

    # ------------------------------------------------------------------
    # 语义分块
    # ------------------------------------------------------------------

    def _get_semantic_splitter(self) -> SemanticSplitterNodeParser:
        """懒加载 SemanticSplitterNodeParser。"""
        if self._semantic_splitter is None:
            from llama_index.core.node_parser import (
                SemanticSplitterNodeParser,
            )
            assert self._embed_model is not None, "语义分块需要 embed_model"
            self._semantic_splitter = SemanticSplitterNodeParser(
                embed_model=self._embed_model,
                buffer_size=self._config.semantic_buffer_size,
                breakpoint_percentile_threshold=(
                    self._config.semantic_breakpoint_percentile
                ),
            )
        return self._semantic_splitter

    def _split_semantic(
        self, document: LlamaDocument
    ) -> list[LlamaDocument]:
        """语义分块——在话题边界处切分，超大块回退 token 分块。

        Args:
            document: 单个 LlamaDocument。

        Returns:
            分块列表。
        """
        text = document.text
        if not text or not text.strip():
            return []

        splitter = self._get_semantic_splitter()
        try:
            nodes = splitter.get_nodes_from_documents([document])
        except Exception as e:
            logger.warning(
                "语义分块失败，回退到 token 分块: %s", e
            )
            return self._split_one(document)

        chunks: list[LlamaDocument] = []
        max_allowed = int(
            self._config.chunk_size * self._config.semantic_max_chunk_multiplier
        )

        for node in nodes:
            node_text = node.get_content() or getattr(node, "text", "")
            if not node_text.strip():
                continue
            token_count = len(self._tokenizer(node_text))

            if token_count <= max_allowed:
                chunks.append(LlamaDocument(
                    text=node_text,
                    metadata={**document.metadata, "chunker": "semantic"},
                ))
            else:
                # 超大块 → token 分块兜底
                sub_doc = LlamaDocument(
                    text=node_text, metadata=document.metadata
                )
                chunks.extend(self._split_one(sub_doc))

        # 重编索引
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk.metadata.update(
                chunk_index=idx, total_chunks=total,
                char_count=len(chunk.text),
            )

        return chunks

    # ------------------------------------------------------------------
    # 后处理：小碎片合并
    # ------------------------------------------------------------------

    def _merge_tiny_chunks(
        self, chunks: list[LlamaDocument]
    ) -> list[LlamaDocument]:
        """合并 token 数低于 min_chunk_tokens 的碎片到前一块。

        分块后可能出现尾部的短碎片（如文档最后一行），
        合并到相邻块可避免嵌入质量差的孤立小块。

        Args:
            chunks: 分块列表。

        Returns:
            合并后的分块列表。
        """
        min_tokens = self._config.min_chunk_tokens
        if min_tokens <= 0 or len(chunks) <= 1:
            return chunks

        merged: list[LlamaDocument] = []
        for chunk in chunks:
            token_count = len(self._tokenizer(chunk.text))
            if token_count < min_tokens and merged:
                # 合并到前一块
                prev = merged[-1]
                merged_text = prev.text + "\n\n" + chunk.text
                prev.set_content(merged_text)
                prev.metadata["char_count"] = len(merged_text)
                prev.metadata["total_chunks"] = max(
                    1, prev.metadata.get("total_chunks", 1) - 1
                )
            else:
                merged.append(chunk)

        # 重编索引
        for idx, chunk in enumerate(merged):
            chunk.metadata.update(
                chunk_index=idx, total_chunks=len(merged),
            )

        if len(merged) < len(chunks):
            logger.debug(
                "小碎片合并: %d → %d (阈值=%d tokens)",
                len(chunks), len(merged), min_tokens,
            )

        return merged

    # ------------------------------------------------------------------
    # Token 分块（SentenceSplitter）
    # ------------------------------------------------------------------

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
        nodes: list = self._splitter.get_nodes_from_documents(
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
