"""
引擎数据传输对象（DTO）。

定义引擎层对外暴露的数据结构，确保上下游接口清晰。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceChunk:
    """检索到的源文档分块——包含完整信息供前端展示。

    Attributes:
        chunk_id: 分块唯一 ID。
        doc_name: 所属文档名称。
        text: 分块文本内容。
        score: 相似度/融合分数。
        metadata: 原始元数据（含 format, page_number, url 等）。
        location_text: 人类可读的位置描述（如"第3页"、"段落: 架构概述"）。
    """

    chunk_id: str
    doc_name: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    location_text: str = ""

    @property
    def doc_format(self) -> str:
        """文档格式。"""
        return self.metadata.get("format", "unknown")

    @property
    def doc_id(self) -> str:
        """所属文档 ID。"""
        return self.metadata.get("doc_id", "")

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典（供 Streamlit st.session_state 存储）。"""
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "location_text": self.location_text,
            "doc_format": self.doc_format,
            "doc_id": self.doc_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SourceChunk":
        """从字典恢复。"""
        return cls(
            chunk_id=d.get("chunk_id", ""),
            doc_name=d.get("doc_name", ""),
            text=d.get("text", ""),
            score=d.get("score", 0.0),
            metadata=d.get("metadata", {}),
            location_text=d.get("location_text", ""),
        )


@dataclass
class AnswerResult:
    """一次问答的完整结果。

    Attributes:
        question: 用户问题。
        answer: 模型生成的完整答案（含引用标记）。
        sources: 引用的源文档分块列表。
        method: 使用的检索策略（direct / mqe / hyde / mqe+hyde）。
        concepts_extracted: 从问答中提取的概念名称列表（可选）。
    """

    question: str
    answer: str
    sources: list[SourceChunk] = field(default_factory=list)
    method: str = ""
    concepts_extracted: list[str] = field(default_factory=list)

    @property
    def source_count(self) -> int:
        """引用来源数。"""
        return len(self.sources)

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "method": self.method,
            "concepts_extracted": self.concepts_extracted,
        }


@dataclass
class IngestResult:
    """单次文档摄入结果。

    Attributes:
        doc_id: 文档唯一 ID。
        doc_name: 文档名称。
        source: 原始来源（路径或 URL）。
        doc_format: 格式。
        num_chunks: 生成的分块总数。
        num_pages: 页数/幻灯片数/行数。
        char_count: 总字符数。
        loaded_at: 加载时间 ISO 字符串。
    """

    doc_id: str
    doc_name: str
    source: str
    doc_format: str
    num_chunks: int = 0
    num_pages: int = 0
    char_count: int = 0
    loaded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
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
