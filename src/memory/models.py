"""
记忆系统数据模型。

定义三类记忆的数据结构：
- WorkingMemoryEntry: 工作记忆中的单轮对话
- EpisodicMemoryRecord: 情景记忆中的 Q&A 记录
- ConceptNode / Relation: 语义记忆中的概念和关系
"""

from dataclasses import dataclass, field
from typing import Any

# ============================================================================
# 工作记忆
# ============================================================================

@dataclass
class WorkingMemoryEntry:
    """工作记忆中的一条记录——代表一轮问答交互。

    Attributes:
        question: 用户问题。
        answer: 模型生成的答案摘要（完整答案的前 200 字符）。
        sources: 引用的源文档列表（doc_name + location）。
        concepts: 提取的关键概念名称。
        timestamp: ISO 格式时间戳。
    """

    question: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "concepts": self.concepts,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkingMemoryEntry":
        return cls(
            question=d.get("question", ""),
            answer=d.get("answer", ""),
            sources=d.get("sources", []),
            concepts=d.get("concepts", []),
            timestamp=d.get("timestamp", ""),
        )


# ============================================================================
# 情景记忆
# ============================================================================

@dataclass
class EpisodicMemoryRecord:
    """情景记忆中的一条 Q&A 记录——持久化到 ChromaDB。

    对应 spec §10.2 的 episodic_memory collection schema。

    Attributes:
        record_id: 唯一 ID（UUID4）。
        question: 用户问题全文。
        answer_summary: 答案摘要。
        source_chunks: 引用的 chunk_id 列表。
        documents: 涉及的文档名列表。
        concepts_extracted: 提取的概念名称列表。
        importance: 重要性评分（0~1）。
        session_id: 所属会话 ID。
        user_id: 用户 ID。
        timestamp: ISO 格式时间戳。
        event_type: 事件类型（qa_interaction / note / review）。
    """

    record_id: str = ""
    question: str = ""
    answer_summary: str = ""
    source_chunks: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    concepts_extracted: list[str] = field(default_factory=list)
    importance: float = 0.5
    session_id: str = ""
    user_id: str = "default_user"
    timestamp: str = ""
    event_type: str = "qa_interaction"

    def to_embedding_text(self) -> str:
        """生成用于嵌入的文本——问题 + 答案摘要拼接。

        Returns:
            嵌入文本。
        """
        return f"问题: {self.question}\n答案摘要: {self.answer_summary}"

    def to_metadata(self) -> dict[str, Any]:
        """转为 ChromaDB metadata 字典（仅基本类型值）。"""
        return {
            "memory_type": "episodic",
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "question": self.question,
            "answer_summary": self.answer_summary,
            "source_chunks": str(self.source_chunks),
            "documents": str(self.documents),
            "concepts_extracted": str(self.concepts_extracted),
            "importance": self.importance,
            "event_type": self.event_type,
        }

    @classmethod
    def from_metadata(cls, meta: dict[str, Any], record_id: str = "") -> "EpisodicMemoryRecord":
        """从 ChromaDB metadata 字典还原。

        Args:
            meta: metadata 字典。
            record_id: 记录 ID。

        Returns:
            EpisodicMemoryRecord 实例。
        """
        import json

        def _parse_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    return [val] if val else []
            return []

        return cls(
            record_id=record_id,
            question=meta.get("question", ""),
            answer_summary=meta.get("answer_summary", ""),
            source_chunks=_parse_list(meta.get("source_chunks", "[]")),
            documents=_parse_list(meta.get("documents", "[]")),
            concepts_extracted=_parse_list(meta.get("concepts_extracted", "[]")),
            importance=float(meta.get("importance", 0.5)),
            session_id=meta.get("session_id", ""),
            user_id=meta.get("user_id", "default_user"),
            timestamp=meta.get("timestamp", ""),
            event_type=meta.get("event_type", "qa_interaction"),
        )


# ============================================================================
# 语义记忆
# ============================================================================

@dataclass
class ConceptNode:
    """语义记忆中的概念节点。

    Attributes:
        name: 概念名称（唯一标识）。
        concept_type: 概念类型（mechanism / architecture / algorithm / theory / technique / tool / application）。
        description: 详细描述。
        first_encountered: 首次遇到的时间戳。
        frequency: 出现次数。
    """

    name: str
    concept_type: str = "concept"
    description: str = ""
    first_encountered: str = ""
    frequency: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.concept_type,
            "description": self.description,
            "first_encountered": self.first_encountered,
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConceptNode":
        return cls(
            name=d.get("name", ""),
            concept_type=d.get("type", "concept"),
            description=d.get("description", ""),
            first_encountered=d.get("first_encountered", ""),
            frequency=int(d.get("frequency", 0)),
        )


@dataclass
class Relation:
    """语义记忆中概念间的关系。

    Attributes:
        source: 源概念名称。
        target: 目标概念名称。
        rel_type: 关系类型（RELATES_TO / MENTIONED_IN / FOUND_IN）。
        strength: 关联强度（0~1）。
        description: 关系描述。
    """

    source: str
    target: str
    rel_type: str = "RELATES_TO"
    strength: float = 0.5
    description: str = ""
