"""
分块器单元测试。
"""

import pytest

from src.core.chunker import TextChunker
from src.core.config import ChunkConfig


class TestTextChunker:
    """TextChunker 测试。"""

    def test_split_single_document(self, sample_llama_document):
        """测试单文档分块。"""
        config = ChunkConfig(chunk_size=1024, chunk_overlap=128)
        chunker = TextChunker(config)
        chunks = chunker.split([sample_llama_document])

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.text
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert chunk.metadata["doc_id"] == "abc123def456"

    def test_split_text_convenience(self):
        """测试 split_text 便捷方法。"""
        config = ChunkConfig(chunk_size=1024, chunk_overlap=128)
        chunker = TextChunker(config)
        chunks = chunker.split_text("Hello world test text. " * 100)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata

    def test_empty_document(self):
        """测试空文档。"""
        config = ChunkConfig(chunk_size=1024, chunk_overlap=128)
        chunker = TextChunker(config)
        from llama_index.core.schema import Document as LlamaDocument
        empty_doc = LlamaDocument(text="", metadata={})
        chunks = chunker.split([empty_doc])
        assert len(chunks) == 0

    def test_chunk_config_properties(self):
        """测试配置属性访问。"""
        config = ChunkConfig(chunk_size=512, chunk_overlap=64)
        chunker = TextChunker(config)
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 64
