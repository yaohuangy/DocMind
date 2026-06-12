"""
三记忆编排器。

整合工作记忆、情景记忆和语义记忆，在问答流程中自动执行：
- 问答前：从工作记忆获取近期对话上下文
- 问答后：记录到情景记忆，提取概念存入语义记忆
- 回顾时：跨记忆搜索，聚合学习数据

此模块是记忆系统对外的唯一入口。
"""

import logging
from datetime import datetime
from typing import Any

from src.core.embedder import BaseEmbedder
from src.core.graph_store import GraphStore
from src.core.llm_client import LLMClient
from src.core.vector_store import VectorStore
from src.engine.models import SourceChunk
from src.memory.episodic_memory import EpisodicMemory
from src.memory.models import (
    ConceptNode,
    EpisodicMemoryRecord,
)
from src.memory.semantic_memory import SemanticMemory
from src.memory.working_memory import WorkingMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """三记忆编排器——对外统一的记忆接口。

    协调三类记忆的读写：
    - **工作记忆**：当前会话上下文（`WorkingMemory`）
    - **情景记忆**：持久化 Q&A 记录（`EpisodicMemory`）
    - **语义记忆**：知识图谱（`SemanticMemory`）

    Usage::

        mm = MemoryManager(session_id="session_001")
        # 问答后记录
        mm.record_interaction(
            question="什么是Transformer？",
            answer="Transformer是一种基于自注意力...",
            sources=source_chunks,
            concepts=["Self-Attention", "Transformer"],
        )
        # 搜索
        episodic, semantic = mm.search_memory("Transformer")
        # 获取回顾数据
        review = mm.get_review_data()
    """

    def __init__(
        self,
        session_id: str = "default",
        working_memory: WorkingMemory | None = None,
        episodic_memory: EpisodicMemory | None = None,
        semantic_memory: SemanticMemory | None = None,
        llm_client: LLMClient | None = None,
        embedder: BaseEmbedder | None = None,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        """
        Args:
            session_id: 会话唯一标识。
            working_memory: 工作记忆实例。None 则自动创建。
            episodic_memory: 情景记忆实例。None 则自动创建。
            semantic_memory: 语义记忆实例。None 则自动创建。
            llm_client: LLM 客户端（用于概念提取和重要性评估）。
            embedder: 嵌入器。
            vector_store: 向量存储。
            graph_store: 图数据库。
        """
        self.session_id = session_id
        self._user_id: str = "default"
        self._llm_client = llm_client or LLMClient()

        # 记忆实例
        self.working = working_memory or WorkingMemory(session_id=session_id)
        self.episodic = episodic_memory or EpisodicMemory(
            embedder=embedder, vector_store=vector_store
        )
        self.semantic = semantic_memory or SemanticMemory(graph_store=graph_store)

        # 统计
        self._note_count: int = 0

    def set_user_id(self, user_id: str) -> None:
        """设置当前用户 ID，同步到情景记忆和语义记忆。

        Args:
            user_id: 用户名。
        """
        self._user_id = user_id

    # ==================================================================
    # 核心：问答交互记录
    # ==================================================================

    def record_interaction(
        self,
        question: str,
        answer: str,
        sources: list[SourceChunk] | None = None,
        concepts: list[str] | None = None,
        auto_extract_concepts: bool = True,
    ) -> None:
        """记录一次完整的问答交互到三记忆系统。

        1. **工作记忆**：添加 Q&A 摘要
        2. **情景记忆**：持久化完整记录
        3. **语义记忆**：提取概念并更新知识图谱

        Args:
            question: 用户问题。
            answer: 模型生成答案。
            sources: 引用的 SourceChunk 列表。
            concepts: 已知的概念名称列表。
            auto_extract_concepts: 是否自动用 LLM 提取新概念。
        """
        sources = sources or []
        doc_names = list(set(s.metadata.get("doc_name", "") for s in sources if s.metadata.get("doc_name")))
        chunk_ids = [s.chunk_id for s in sources]
        concepts = concepts or []

        # ---- 自动提取概念 ----
        print(f"[MEMORY] record_interaction called: question='{question[:50]}...', auto_extract={auto_extract_concepts}, has_answer={bool(answer)}")
        if auto_extract_concepts and answer:
            try:
                extracted = self._extract_concepts(question, answer)
                all_concepts = list(dict.fromkeys(concepts + extracted))
                print(f"[MEMORY] 概念总数: {len(all_concepts)}")
            except Exception as e:
                print(f"[MEMORY] 概念提取失败: {e}")
                all_concepts = concepts
        else:
            all_concepts = concepts

        # ---- 评估重要性 ----
        importance = self._estimate_importance(question, answer, len(sources))

        # ---- 1. 工作记忆 ----
        self.working.add(
            question=question,
            answer=answer,
            sources=doc_names,
            concepts=all_concepts,
        )

        # ---- 2. 情景记忆 ----
        answer_summary = self._summarize_answer(answer)
        self.episodic.record(
            question=question,
            answer_summary=answer_summary,
            source_chunks=chunk_ids,
            documents=doc_names,
            concepts_extracted=all_concepts,
            importance=importance,
            session_id=self.session_id,
            user_id=self._user_id,
            event_type="qa_interaction",
        )

        # ---- 3. 语义记忆 ----
        print(f"[MEMORY] 语义记忆阶段: all_concepts={all_concepts}")
        if all_concepts:
            try:
                self.semantic.connect()
                print("[MEMORY] Neo4j 连接成功")
            except Exception as e:
                print(f"[MEMORY] Neo4j 连接失败: {e}")
                return

            try:
                for concept_name in all_concepts:
                    node = self.semantic.add_concept(name=concept_name, user_id=self._user_id)
                    print(f"[MEMORY] 写入概念: {concept_name}, user_id={self._user_id}")
                # 立即验证
                verify = self.semantic.get_all_concepts(limit=100, user_id=self._user_id)
                print(f"[MEMORY] 写入后验证: Neo4j 中 user_id='{self._user_id}' 的概念数={len(verify)}")
                all_concepts_in_db = self.semantic._graph.run_query(
                    "MATCH (c:Concept) RETURN c.name, c.user_id LIMIT 5"
                )
                print(f"[MEMORY] Neo4j 最近5个节点: {all_concepts_in_db}")
                print(f"[MEMORY] 语义记忆: 已写入 {len(all_concepts)} 个概念")

                # 为新概念两两建立弱关系
                if len(all_concepts) >= 2:
                    for i in range(len(all_concepts)):
                        for j in range(i + 1, len(all_concepts)):
                            try:
                                self.semantic.add_relation(
                                    all_concepts[i],
                                    all_concepts[j],
                                    rel_type="RELATES_TO",
                                    strength=0.3,
                                    description=f"来自问答: {question[:50]}",
                                    user_id=self._user_id,
                                )
                            except Exception:
                                pass  # 关系创建失败不影响主流程
            except Exception as e:
                print(f"[MEMORY] 语义记忆写入失败: {e}")

        print("[MEMORY] record_interaction 完成")

    # ==================================================================
    # 检索前：获取附加上下文
    # ==================================================================

    def get_working_context(self, last_n: int = 5) -> str:
        """获取工作记忆中的近期对话上下文。

        在 QA 检索前调用，为 LLM 提供多轮对话背景。

        Args:
            last_n: 取最近 N 轮。

        Returns:
            格式化的上下文文本。
        """
        return self.working.get_context(last_n=last_n)

    def get_active_concepts(self) -> list[str]:
        """获取当前会话中的活跃概念列表。

        Returns:
            概念名称列表。
        """
        return self.working.get_active_concepts()

    # ==================================================================
    # 搜索记忆
    # ==================================================================

    def search_memory(
        self,
        keyword: str,
        episodic_limit: int = 10,
        semantic_limit: int = 20,
    ) -> dict[str, Any]:
        """跨记忆搜索——同时搜索情景记忆和语义记忆。

        Args:
            keyword: 搜索关键词。
            episodic_limit: 情景记忆返回数上限。
            semantic_limit: 语义记忆返回数上限。

        Returns:
            {
                "episodic": List[EpisodicMemoryRecord],
                "semantic": List[ConceptNode],
                "working": List[WorkingMemoryEntry],
            }
        """
        # 情景记忆搜索
        episodic_results: list[EpisodicMemoryRecord] = []
        try:
            episodic_results = self.episodic.search(keyword, limit=episodic_limit, user_id=self._user_id)
        except Exception as e:
            logger.error("情景记忆搜索失败: %s", e)

        # 语义记忆搜索
        semantic_results: list[ConceptNode] = []
        try:
            semantic_results = self.semantic.search_concepts(keyword, limit=semantic_limit, user_id=self._user_id)
        except Exception as e:
            logger.error("语义记忆搜索失败: %s", e)

        # 工作记忆中匹配
        working_matches = [
            e for e in self.working.entries
            if keyword.lower() in e.question.lower()
            or keyword.lower() in e.answer.lower()
        ]

        return {
            "episodic": episodic_results,
            "semantic": semantic_results,
            "working": working_matches,
        }

    # ==================================================================
    # 回顾数据
    # ==================================================================

    def get_review_data(self) -> dict[str, Any]:
        """获取学习回顾所需的聚合数据。

        Returns:
            适合生成学习报告的字典：
            - episodic_count: 情景记忆总数
            - concept_count: 概念总数
            - recent_episodes: 最近 20 条情景记忆
            - top_concepts: 频率最高的 20 个概念
            - working_entries: 当前会话记录数
        """
        episodic_count = self.episodic.get_record_count()

        recent_episodes: list[EpisodicMemoryRecord] = []
        try:
            recent_episodes = self.episodic.search_by_time(limit=20, user_id=self._user_id)
        except Exception as e:
            logger.error("获取情景记忆失败: %s", e)

        concept_count = 0
        top_concepts: list[ConceptNode] = []
        try:
            top_concepts = self.semantic.get_all_concepts(limit=20, user_id=self._user_id)
            concept_count = self.semantic.get_concept_count(user_id=self._user_id)
        except Exception as e:
            logger.error("获取语义记忆失败: %s", e)

        return {
            "episodic_count": episodic_count,
            "concept_count": concept_count,
            "recent_episodes": recent_episodes,
            "top_concepts": [
                {"name": c.name, "type": c.concept_type, "frequency": c.frequency}
                for c in top_concepts
            ],
            "working_entries": self.working.entry_count,
        }

    # ==================================================================
    # 笔记管理
    # ==================================================================

    def add_note(
        self,
        content: str,
        related_concepts: list[str] | None = None,
    ) -> str:
        """添加一条学习笔记。

        笔记会存入情景记忆（event_type="note"），
        相关概念会更新语义记忆。

        Args:
            content: 笔记内容。
            related_concepts: 关联的概念名称列表。

        Returns:
            笔记记录 ID。
        """
        self._note_count += 1

        # 记入情景记忆（存完整内容，UI 层按需截断）
        note_id = self.episodic.record(
            question=content,
            answer_summary=content,
            concepts_extracted=related_concepts or [],
            importance=0.7,
            session_id=self.session_id,
            user_id=self._user_id,
            event_type="note",
        )

        # 更新语义记忆
        if related_concepts:
            try:
                self.semantic.connect()
                for name in related_concepts:
                    self.semantic.add_concept(name=name, concept_type="concept", user_id=self._user_id)
            except Exception as e:
                logger.warning("笔记概念关联失败: %s", e)

        logger.info("笔记已添加: id=%s, concepts=%s", note_id, related_concepts)
        return note_id

    # ==================================================================
    # 内部辅助
    # ==================================================================

    def _extract_concepts(self, question: str, answer: str) -> list[str]:
        """使用 LLM 从问答中提取概念名称。

        Args:
            question: 用户问题。
            answer: 模型答案。

        Returns:
            概念名称列表。
        """
        try:
            text = f"问题: {question}\n答案: {answer}"
            print(f"[CONCEPT] 调用 LLM 提取概念... 文本长度={len(text)}")
            raw_concepts = self._llm_client.extract_concepts(text)
            print(f"[CONCEPT] LLM 返回 {len(raw_concepts)} 个概念: {raw_concepts}")
            names = [c.get("name", "") for c in raw_concepts if c.get("name")]
            print(f"[CONCEPT] 最终概念: {names}")
            return names
        except Exception as e:
            print(f"[CONCEPT] 概念提取异常: {e}")
            return []

    @staticmethod
    def _estimate_importance(
        question: str,
        answer: str,
        num_sources: int,
    ) -> float:
        """基于启发式规则估算交互重要性。

        评分因素：
        - 答案长度（越长越重要）
        - 来源数量（来源越多越重要）
        - 问题长度（问题越长越可能是深度问题）

        Args:
            question: 问题。
            answer: 答案。
            num_sources: 来源数量。

        Returns:
            重要性评分（0~1）。
        """
        score = 0.3  # 基础分

        # 答案长度贡献（max 0.3）
        ans_len = len(answer)
        if ans_len > 800:
            score += 0.3
        elif ans_len > 400:
            score += 0.2
        elif ans_len > 200:
            score += 0.1

        # 来源数量贡献（max 0.25）
        if num_sources >= 5:
            score += 0.25
        elif num_sources >= 3:
            score += 0.15
        elif num_sources >= 1:
            score += 0.05

        # 问题长度贡献（max 0.15）
        q_len = len(question)
        if q_len > 100:
            score += 0.15
        elif q_len > 50:
            score += 0.1
        elif q_len > 20:
            score += 0.05

        return min(score, 1.0)

    @staticmethod
    def _summarize_answer(answer: str, max_chars: int = 2000) -> str:
        """生成答案摘要（截断）。

        Args:
            answer: 完整答案。
            max_chars: 最大字符数。

        Returns:
            摘要文本。
        """
        if len(answer) <= max_chars:
            return answer
        # 尝试在句子边界截断
        truncated = answer[:max_chars]
        last_period = max(
            truncated.rfind("。"),
            truncated.rfind("."),
            truncated.rfind("\n"),
        )
        if last_period > max_chars // 2:
            return truncated[: last_period + 1]
        return truncated + "..."

    # ==================================================================
    # 笔记列表与删除
    # ==================================================================

    def list_notes(self, limit: int = 50) -> list[EpisodicMemoryRecord]:
        """列出所有笔记（按时间降序）。

        Args:
            limit: 返回数量上限。

        Returns:
            EpisodicMemoryRecord 列表。
        """
        try:
            # 获取所有 episodic 记录，客户端过滤 event_type="note"
            all_records = self.episodic.search_by_time(limit=limit * 2, user_id=self._user_id)
            notes = [r for r in all_records if r.event_type == "note"]
            return notes[:limit]
        except Exception as e:
            logger.error("列出笔记失败: %s", e)
            return []

    def list_notes_count(self) -> int:
        """获取笔记总数。

        Returns:
            笔记数量。
        """
        try:
            notes = self.list_notes(limit=1000)
            return len(notes)
        except Exception:
            return 0

    def get_history(self, limit: int = 200) -> list:
        """获取当前用户的所有历史记录（Q&A + 笔记）。

        Args:
            limit: 返回数量上限。

        Returns:
            EpisodicMemoryRecord 列表。
        """
        try:
            return self.episodic.search_by_time(limit=limit, user_id=self._user_id)
        except Exception as e:
            logger.error("获取历史记录失败: %s", e)
            return []

    def delete_note(self, note_id: str) -> bool:
        """删除一条笔记。

        Args:
            note_id: 笔记记录 ID。

        Returns:
            是否成功。
        """
        try:
            self.episodic.delete_record(note_id)
            logger.info("笔记已删除: %s", note_id)
            return True
        except Exception as e:
            logger.error("删除笔记失败: %s", e)
            return False

    # ==================================================================
    # 学习报告生成
    # ==================================================================

    def generate_report(self) -> dict:
        """生成 JSON 格式的学习报告。

        聚合情景记忆 + 语义记忆数据，可选调用 LLM 生成学习建议。

        Returns:
            报告字典：
            {
                "generated_at": str,
                "session_id": str,
                "summary": {"episodic_count", "concept_count", "note_count", ...},
                "concepts": [...],
                "recent_activities": [...],
                "suggestions": str  (LLM 生成的学习建议),
            }
        """
        review = self.get_review_data()

        # 构建概念摘要
        concepts_summary = [
            {"name": c["name"], "type": c["type"], "frequency": c["frequency"]}
            for c in review.get("top_concepts", [])
        ]

        # 构建近期活动
        recent: list[dict] = []
        for ep in review.get("recent_episodes", []):
            recent.append({
                "question": ep.question[:120] if hasattr(ep, 'question') else "",
                "timestamp": ep.timestamp if hasattr(ep, 'timestamp') else "",
                "importance": ep.importance if hasattr(ep, 'importance') else 0.5,
            })

        # 生成学习建议（如果 LLM 可用）
        suggestions = "暂无建议（LLM 不可用或数据不足）"
        if recent and concepts_summary:
            try:
                prompt = (
                    f"用户最近进行了 {len(recent)} 次问答交互，"
                    f"涉及 {len(concepts_summary)} 个核心概念。"
                    f"最常出现的概念: {', '.join(c['name'] for c in concepts_summary[:5])}。"
                    f"请给出 3-5 条简短的学习建议（每条不超过 50 字）。"
                )
                suggestions = self._llm_client.chat(
                    messages=[
                        {"role": "system", "content": "你是一个学习分析助手。请给出具体、可操作的学习建议。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=400,
                )
            except Exception as e:
                logger.warning("LLM 学习建议生成失败: %s", e)

        report = {
            "generated_at": datetime.now().isoformat(),
            "session_id": self.session_id,
            "summary": {
                "episodic_count": review.get("episodic_count", 0),
                "concept_count": review.get("concept_count", 0),
                "note_count": self.list_notes_count(),
                "working_entries": review.get("working_entries", 0),
            },
            "concepts": concepts_summary,
            "recent_activities": recent[:20],
            "suggestions": suggestions,
        }

        logger.info("学习报告已生成: concepts=%d, activities=%d",
                    len(concepts_summary), len(recent))
        return report

    # ==================================================================
    # 演示数据
    # ==================================================================

    def seed_demo_frequencies(self) -> int:
        """为语义记忆中的概念分配模拟随机频率（仅用于演示/开发）。

        Returns:
            更新的概念节点数量。
        """
        try:
            self.semantic.connect()
            count = self.semantic.seed_demo_frequencies(user_id=self._user_id)
            logger.info("演示频率已分配: %d 个概念", count)
            return count
        except Exception as e:
            logger.error("演示频率分配失败: %s", e)
            raise
