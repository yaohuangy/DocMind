"""
引擎 DTO 单元测试。

覆盖 SourceChunk、AnswerResult、IngestResult 的
序列化/反序列化、计算属性等。
"""

import pytest

from src.engine.models import AnswerResult, IngestResult, SourceChunk


# ============================================================================
# SourceChunk
# ============================================================================


class TestSourceChunk:
    """SourceChunk DTO 测试。"""

    def test_basic_creation(self):
        chunk = SourceChunk(
            chunk_id="abc123",
            doc_name="test.pdf",
            text="Transformer 架构",
            score=0.95,
            metadata={"format": "pdf", "doc_id": "doc_001", "page_number": 3},
            location_text="第3页",
        )
        assert chunk.chunk_id == "abc123"
        assert chunk.doc_name == "test.pdf"
        assert chunk.text == "Transformer 架构"
        assert chunk.score == 0.95

    def test_doc_format_property(self):
        chunk = SourceChunk(
            chunk_id="c1", doc_name="a.docx", text="text",
            score=0.8, metadata={"format": "docx"},
        )
        assert chunk.doc_format == "docx"

    def test_doc_format_unknown_default(self):
        chunk = SourceChunk(
            chunk_id="c1", doc_name="a.xyz", text="text",
            score=0.8, metadata={},
        )
        assert chunk.doc_format == "unknown"

    def test_doc_id_property(self):
        chunk = SourceChunk(
            chunk_id="c1", doc_name="a.pdf", text="text",
            score=0.5, metadata={"doc_id": "doc_abc"},
        )
        assert chunk.doc_id == "doc_abc"

    def test_doc_id_empty_default(self):
        chunk = SourceChunk(
            chunk_id="c1", doc_name="a.pdf", text="text",
            score=0.5, metadata={},
        )
        assert chunk.doc_id == ""

    def test_to_dict(self):
        chunk = SourceChunk(
            chunk_id="c1", doc_name="a.pdf", text="hello",
            score=0.9, metadata={"format": "pdf"}, location_text="第1页",
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "c1"
        assert d["doc_name"] == "a.pdf"
        assert d["text"] == "hello"
        assert d["score"] == 0.9
        assert d["metadata"] == {"format": "pdf"}
        assert d["location_text"] == "第1页"
        assert d["doc_format"] == "pdf"
        assert d["doc_id"] == ""

    def test_from_dict_roundtrip(self):
        """to_dict → from_dict 往返一致性。"""
        original = SourceChunk(
            chunk_id="xyz", doc_name="doc.pdf", text="content",
            score=0.75, metadata={"k": "v"}, location_text="loc",
        )
        restored = SourceChunk.from_dict(original.to_dict())
        assert restored.chunk_id == original.chunk_id
        assert restored.doc_name == original.doc_name
        assert restored.text == original.text
        assert restored.score == original.score
        assert restored.metadata == original.metadata
        assert restored.location_text == original.location_text

    def test_from_dict_partial(self):
        """部分字段缺失时的回退值。"""
        chunk = SourceChunk.from_dict({})
        assert chunk.chunk_id == ""
        assert chunk.doc_name == ""
        assert chunk.text == ""
        assert chunk.score == 0.0
        assert chunk.metadata == {}
        assert chunk.location_text == ""

    def test_default_values(self):
        chunk = SourceChunk(chunk_id="c1", doc_name="d", text="t", score=0.1)
        assert chunk.metadata == {}
        assert chunk.location_text == ""


# ============================================================================
# AnswerResult
# ============================================================================


class TestAnswerResult:
    """AnswerResult DTO 测试。"""

    def test_basic_creation(self):
        result = AnswerResult(
            question="什么是 RAG？",
            answer="RAG 是检索增强生成。",
            method="mqe+hyde",
            concepts_extracted=["RAG", "Retrieval"],
        )
        assert result.question == "什么是 RAG？"
        assert result.answer == "RAG 是检索增强生成。"
        assert result.method == "mqe+hyde"
        assert result.concepts_extracted == ["RAG", "Retrieval"]

    def test_source_count_property(self):
        sources = [
            SourceChunk(chunk_id=f"c{i}", doc_name="d", text="t", score=0.5)
            for i in range(3)
        ]
        result = AnswerResult(
            question="q", answer="a", sources=sources,
        )
        assert result.source_count == 3

    def test_source_count_empty(self):
        result = AnswerResult(question="q", answer="a")
        assert result.source_count == 0

    def test_default_values(self):
        result = AnswerResult(question="q", answer="a")
        assert result.sources == []
        assert result.method == ""
        assert result.concepts_extracted == []

    def test_to_dict(self):
        chunk = SourceChunk(
            chunk_id="c1", doc_name="d.pdf", text="hello",
            score=0.9, metadata={"format": "pdf"}, location_text="p1",
        )
        result = AnswerResult(
            question="q?", answer="ans",
            sources=[chunk], method="direct",
            concepts_extracted=["C1"],
        )
        d = result.to_dict()
        assert d["question"] == "q?"
        assert d["answer"] == "ans"
        assert d["method"] == "direct"
        assert len(d["sources"]) == 1
        assert d["sources"][0]["chunk_id"] == "c1"
        assert d["concepts_extracted"] == ["C1"]


# ============================================================================
# IngestResult
# ============================================================================


class TestIngestResult:
    """IngestResult DTO 测试。"""

    def test_basic_creation(self):
        result = IngestResult(
            doc_id="doc_001",
            doc_name="paper.pdf",
            source="/path/to/paper.pdf",
            doc_format="pdf",
            num_chunks=42,
            num_pages=10,
            char_count=32000,
            loaded_at="2025-01-01T00:00:00",
        )
        assert result.doc_id == "doc_001"
        assert result.doc_name == "paper.pdf"
        assert result.num_chunks == 42

    def test_to_dict(self):
        result = IngestResult(
            doc_id="d1", doc_name="a.pdf", source="/a.pdf",
            doc_format="pdf", num_chunks=5, num_pages=2,
            char_count=1000, loaded_at="t",
        )
        d = result.to_dict()
        assert d["doc_id"] == "d1"
        assert d["num_chunks"] == 5
        assert d["loaded_at"] == "t"

    def test_default_values(self):
        result = IngestResult(
            doc_id="d1", doc_name="a", source="s", doc_format="txt",
        )
        assert result.num_chunks == 0
        assert result.num_pages == 0
        assert result.char_count == 0
        assert result.loaded_at == ""
