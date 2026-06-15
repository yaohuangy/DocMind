"""
QA 引擎——应用编排层。

整合所有子系统（摄入、检索、生成、记忆），对外提供统一接口。
Streamlit UI 通过此引擎完成所有操作，无需直接接触底层模块。

Usage::

    engine = QAEngine()
    engine.ingest("/path/to/paper.pdf")
    sources = engine.retrieve("什么是Transformer？", method="mqe+hyde")
    for token in engine.generate_stream("什么是Transformer？", sources):
        print(token, end="", flush=True)
"""

import logging
from collections.abc import Generator

from src.core.config import Settings, get_config
from src.core.embedder import create_embedder
from src.core.llm_client import LLMClient
from src.core.metadata_store import MetadataStore
from src.core.vector_store import SearchResult, VectorStore
from src.engine.models import AnswerResult, IngestResult, SourceChunk
from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_formatter import CitationFormatter
from src.ingestion.ingest_pipeline import IngestPipeline
from src.memory.memory_manager import MemoryManager
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.combined_retriever import CombinedRetriever
from src.retrieval.direct_retriever import DirectRetriever
from src.retrieval.hyde_retriever import HyDERetriever
from src.retrieval.mqe_retriever import MQERetriever

logger = logging.getLogger(__name__)


class QAEngine:
    """Document QA 主引擎——应用编排层。

    提供四大核心能力：
    - **摄入**：多格式文档加载、分块、嵌入、入库
    - **检索**：Direct / MQE / HyDE / MQE+HyDE 四种策略
    - **生成**：流式 RAG 答案生成（带引用）
    - **管理**：文档删除、统计查询

    所有方法均为同步（内部异步操作通过 asyncio.run 处理）。
    """

    def __init__(self, config: Settings | None = None) -> None:
        """
        Args:
            config: 全局配置。None 则自动加载。
        """
        if config is None:
            config = get_config()
        self._config = config

        # ---- 基础设施 ----
        self._llm_client = LLMClient(config.llm)
        self._embedder = create_embedder(config.embedding)
        self._vector_store = VectorStore(config.chroma)
        self._metadata_store = MetadataStore(config.sqlite)

        # ---- 重排序器（懒加载） ----
        self._reranker: Reranker | None = None  # type: ignore[name-defined]  # noqa: F821

        # ---- 子模块 ----
        self._ingest_pipeline = IngestPipeline(
            embedder=self._embedder,
            vector_store=self._vector_store,
            metadata_store=self._metadata_store,
        )
        self._answer_generator = AnswerGenerator(self._llm_client)
        self._citation_formatter = CitationFormatter()

        # ---- 记忆系统 ----
        self._neo4j_available = False
        try:
            self.memory = MemoryManager(
                session_id="default",
                llm_client=self._llm_client,
                embedder=self._embedder,
                vector_store=self._vector_store,
            )
            # 尝试连接 Neo4j（失败不影响主功能）
            try:
                self.memory.semantic.connect()
                self._neo4j_available = True
                logger.info("Neo4j 连接成功，语义记忆可用")
            except Exception:
                logger.warning(
                    "Neo4j 不可用——语义记忆功能将禁用。"
                    "启动 Neo4j 后重启应用以启用知识图谱功能。"
                )
        except Exception as e:
            logger.warning("记忆系统初始化失败（将降级运行）: %s", e)
            self.memory = None  # type: ignore[assignment]

        # ---- 检索器（延迟实例化） ----
        self._retrievers: dict = {}

        # ---- 用户 ----
        self._user_id: str = "default"

        # ---- 统计 ----
        self._question_count: int = 0

        # 初始化元数据表
        self._metadata_store.ensure_tables()

        logger.info("QAEngine 初始化完成 (Neo4j=%s)", self._neo4j_available)

    # ==================================================================
    # 文档摄入
    # ==================================================================

    def ingest(self, source: str) -> IngestResult:
        """摄入单个文档（文件路径或 URL）。

        完成 加载→分块→嵌入→入库 全流程。

        Args:
            source: 文件路径或 URL。

        Returns:
            IngestResult 描述摄入结果。
        """
        try:
            pipe_result = self._ingest_pipeline.ingest(source, user_id=self._user_id)
            result = IngestResult(
                doc_id=pipe_result.doc_id,
                doc_name=pipe_result.doc_name,
                source=pipe_result.source,
                doc_format=pipe_result.doc_format,
                num_chunks=pipe_result.num_chunks,
                num_pages=pipe_result.num_pages,
                char_count=pipe_result.char_count,
                loaded_at=pipe_result.loaded_at,
                step_timings=pipe_result.step_timings,
                total_sec=pipe_result.total_sec,
            )
            return result
        except ValueError as e:
            logger.error("格式不支持: %s", e)
            raise
        except RuntimeError as e:
            logger.error("摄入失败: %s", e)
            raise
        except Exception as e:
            logger.exception("摄入异常: %s", e)
            raise

    def ingest_batch(self, sources: list[str]) -> list[IngestResult]:
        """批量摄入文档。

        Args:
            sources: 文件路径或 URL 列表。

        Returns:
            成功摄入的结果列表。
        """
        pipe_results = self._ingest_pipeline.ingest_batch(sources, user_id=self._user_id)
        return [
            IngestResult(
                doc_id=r.doc_id,
                doc_name=r.doc_name,
                source=r.source,
                doc_format=r.doc_format,
                num_chunks=r.num_chunks,
                num_pages=r.num_pages,
                char_count=r.char_count,
                loaded_at=r.loaded_at,
                step_timings=r.step_timings,
                total_sec=r.total_sec,
            )
            for r in pipe_results
        ]

    def delete_document(self, doc_id: str) -> bool:
        """删除文档（向量数据 + 元数据）。

        Args:
            doc_id: 文档唯一 ID。

        Returns:
            是否实际删除了数据。
        """
        return self._ingest_pipeline.delete_document(doc_id)

    # ==================================================================
    # 检索
    # ==================================================================

    def retrieve(
        self,
        question: str,
        method: str = "mqe+hyde",
        top_k: int = 10,
    ) -> list[SourceChunk]:
        """根据问题检索相关文档分块。

        Args:
            question: 用户问题。
            method: 检索策略，可选 "direct" / "mqe" / "hyde" / "mqe+hyde"
                    （也支持中文 "直接检索" 等和下划线分隔）。
            top_k: 返回结果数。

        Returns:
            SourceChunk 列表，含位置描述。

        Raises:
            ValueError: 无效的检索方法。
        """
        method_key = self._normalize_method(method)

        # 检查是否有文档
        stats = self._vector_store.collection_stats(VectorStore.DOCUMENT_CHUNKS)
        if stats.get("count", 0) == 0:
            logger.warning("检索时无可用文档")
            return []

        retriever = self._get_retriever(method_key)

        # 重排序模式：粗筛取更多，精排后截断
        fetch_k = top_k
        if self._config.retrieval.use_reranker:
            fetch_k = max(self._config.retrieval.reranker_top_k, top_k * 2)

        try:
            search_results = retriever.retrieve(question, top_k=fetch_k)
        except Exception as e:
            logger.error("检索失败 [%s]: %s", method_key, e)
            return []

        # 去重（文本相似度去重，减少冗余信息送入 LLM）
        if self._config.retrieval.use_dedup and len(search_results) > 1:
            from src.retrieval.dedup import deduplicate_chunks

            before = len(search_results)
            search_results = deduplicate_chunks(
                search_results,
                threshold=self._config.retrieval.dedup_threshold,
            )
            logger.info(
                "去重: %d → %d 结果 (threshold=%.2f)",
                before, len(search_results), self._config.retrieval.dedup_threshold,
            )

        # 重排序（Cross-Encoder 精排）
        if self._config.retrieval.use_reranker and len(search_results) > 1:
            search_results = self._apply_rerank(question, search_results, top_k)

        # SearchResult → SourceChunk（含位置描述）
        sources = self._to_source_chunks(search_results[:top_k])

        self._question_count += 1
        logger.info(
            "检索完成 [%s]: \"%s\" → %d sources (rerank=%s)",
            method_key, question[:50], len(sources),
            self._config.retrieval.use_reranker,
        )
        return sources

    # ==================================================================
    # 生成
    # ==================================================================

    def generate_stream(
        self,
        question: str,
        sources: list[SourceChunk],
        method: str = "mqe+hyde",
    ) -> Generator[str, None, None]:
        """流式生成带引用的答案。

        每次 yield 一个 token，供 ``st.write_stream()`` 逐字渲染。

        Args:
            question: 用户问题。
            sources: 检索到的源文档分块。
            method: 检索策略名称。

        Yields:
            str: 增量 token。
        """
        yield from self._answer_generator.generate_stream(
            question=question,
            sources=sources,
            method=method,
        )

    def generate(
        self,
        question: str,
        sources: list[SourceChunk],
        method: str = "mqe+hyde",
    ) -> str:
        """同步生成完整答案。

        Args:
            question: 用户问题。
            sources: 源文档分块。
            method: 检索策略名称。

        Returns:
            完整答案字符串。
        """
        return self._answer_generator.generate(
            question=question,
            sources=sources,
            method=method,
        )

    # ==================================================================
    # 引用格式化
    # ==================================================================

    def format_answer(
        self,
        raw_answer: str,
        sources: list[SourceChunk],
    ) -> tuple:
        """格式化答案中的引用标记，并重新映射编号。

        将 LLM 输出中的 [N] 引用标记替换为紧凑的 [1][2][3] 序列，
        并返回对应的 SourceChunk 列表。

        Args:
            raw_answer: LLM 原始答案。
            sources: 检索到的源文档分块。

        Returns:
            (formatted_answer, cited_sources) 元组。
        """
        # 转为 SearchResult 以复用 CitationFormatter
        search_results = [
            SearchResult(
                chunk_id=s.chunk_id,
                text=s.text,
                score=s.score,
                metadata=s.metadata,
            )
            for s in sources
        ]

        formatted, cited = self._citation_formatter.format_with_remap(
            raw_answer, search_results
        )

        # 转回 SourceChunk
        cited_sources = [
            SourceChunk(
                chunk_id=sr.chunk_id,
                doc_name=sr.metadata.get("doc_name", ""),
                text=sr.text,
                score=sr.score,
                metadata=sr.metadata,
                location_text=self._citation_formatter._format_location(sr.metadata),
            )
            for sr in cited
        ]

        return formatted, cited_sources

    # ==================================================================
    # 端到端问答（便捷方法）
    # ==================================================================

    def ask(
        self,
        question: str,
        method: str = "mqe+hyde",
        top_k: int = 10,
    ) -> AnswerResult:
        """端到端问答——检索 + 生成（非流式）。

        适合需要一次性获取完整答案的场景。

        Args:
            question: 用户问题。
            method: 检索策略。
            top_k: 返回结果数。

        Returns:
            AnswerResult，含答案和引用源。
        """
        # 检索
        sources = self.retrieve(question, method=method, top_k=top_k)

        # 生成
        answer = self.generate(question, sources, method=method)

        # 格式化
        formatted_answer, cited_sources = self.format_answer(answer, sources)

        return AnswerResult(
            question=question,
            answer=formatted_answer,
            sources=cited_sources if cited_sources else sources,
            method=method,
        )

    # ==================================================================
    # 统计与管理
    # ==================================================================

    @property
    def neo4j_available(self) -> bool:
        """Neo4j 是否可用。"""
        return self._neo4j_available

    def get_stats(self) -> dict:
        """获取全局统计快照。

        Returns:
            包含文档数、提问次数、笔记数等统计信息。
        """
        doc_count = self._metadata_store.get_document_count(user_id=self._user_id)
        session_count = self._metadata_store.get_session_count()

        # 记忆统计（安全获取，Neo4j/ChromaDB 不可用时回退）
        concept_count = 0
        note_count = 0
        if self.memory is not None:
            try:
                concept_count = self.memory.semantic.get_concept_count(user_id=self._user_id)
                print(f"[STATS] user={self._user_id}, concept_count={concept_count}")
            except Exception as e:
                print(f"[STATS] get_concept_count 异常: {e}")
            try:
                note_count = self.memory.list_notes_count()
            except Exception:
                pass

        return {
            "文档总数": doc_count,
            "会话总数": session_count,
            "提问次数": self._question_count,
            "学习笔记": note_count,
            "概念数量": concept_count,
            "Neo4j可用": "✅" if self._neo4j_available else "⚠️ 不可用",
        }

    @property
    def user_id(self) -> str:
        """当前用户 ID。"""
        return self._user_id

    def set_user_id(self, user_id: str) -> None:
        """设置当前用户 ID，同步到所有子系统。

        Args:
            user_id: 用户名。
        """
        self._user_id = user_id
        # 清空检索器缓存（不同用户的 where 过滤不同）
        self._retrievers = {}
        if self.memory is not None:
            self.memory.set_user_id(user_id)
        logger.info("用户切换: %s", user_id)

    def set_session_id(self, session_id: str) -> None:
        """更新当前会话 ID（工作记忆绑定到新会话）。

        Args:
            session_id: 新会话 ID。
        """
        if self.memory is not None:
            self.memory.session_id = session_id
            self.memory.working.session_id = session_id

    # ==================================================================
    # 记忆系统便捷方法（委托给 MemoryManager，None-safe）
    # ==================================================================

    def record_interaction(
        self, question: str, answer: str,
        sources: list[SourceChunk] | None = None,
        concepts: list[str] | None = None,
    ) -> None:
        """记录一次问答交互到三记忆系统。"""
        if self.memory is None:
            import streamlit as st
            st.warning("记忆系统未初始化，无法记录。请检查 ChromaDB 是否可用。")
            return
        try:
            self.memory.record_interaction(question, answer, sources, concepts)
        except Exception as e:
            logger.warning("记忆记录失败: %s", e)
            import streamlit as st
            st.warning(f"记忆记录失败: {e}")

    def add_note(
        self, content: str, related_concepts: list[str] | None = None
    ) -> str:
        """添加学习笔记。"""
        if self.memory is None:
            raise RuntimeError("记忆系统未初始化（Neo4j/ChromaDB 不可用）")
        return self.memory.add_note(content, related_concepts)

    def search_memory(self, keyword: str) -> dict:
        """跨记忆搜索。"""
        if self.memory is None:
            return {"episodic": [], "semantic": [], "working": []}
        try:
            return self.memory.search_memory(keyword)
        except Exception as e:
            logger.warning("记忆搜索失败: %s", e)
            return {"episodic": [], "semantic": [], "working": []}

    def get_review_data(self) -> dict:
        """获取学习回顾数据。"""
        if self.memory is None:
            return {"episodic_count": 0, "concept_count": 0, "top_concepts": [], "recent_episodes": [], "working_entries": 0}
        try:
            return self.memory.get_review_data()
        except Exception as e:
            logger.warning("回顾数据获取失败: %s", e)
            return {"episodic_count": 0, "concept_count": 0, "top_concepts": [], "recent_episodes": [], "working_entries": 0}

    def get_graph_data(self, top_n: int = 0) -> dict:
        """获取当前用户的知识图谱数据（节点 + 关系）。

        Args:
            top_n: 仅返回频率最高的 top_n 个概念，0 表示返回全部。

        Returns:
            {"nodes": [...], "edges": [...]}，Neo4j 不可用时返回空列表。
        """
        if self.memory is None or not self._neo4j_available:
            return {"nodes": [], "edges": []}
        try:
            return self.memory.semantic.get_user_graph_data(
                user_id=self._user_id, top_n=top_n,
            )
        except Exception as e:
            logger.warning("图谱数据获取失败: %s", e)
            return {"nodes": [], "edges": []}

    def seed_demo_frequencies(self) -> int:
        """为语义记忆中的概念分配模拟随机频率（仅用于演示/开发）。

        Returns:
            更新的概念节点数量。
        """
        if self.memory is None or not self._neo4j_available:
            raise RuntimeError("记忆系统/Neo4j 不可用")
        return self.memory.seed_demo_frequencies()

    def generate_report(self) -> dict:
        """生成 JSON 格式学习报告。"""
        if self.memory is None:
            raise RuntimeError("记忆系统未初始化（Neo4j/ChromaDB 不可用）")
        return self.memory.generate_report()

    def list_notes(self, limit: int = 50) -> list:
        """列出所有笔记。"""
        if self.memory is None:
            return []
        try:
            return self.memory.list_notes(limit)
        except Exception as e:
            logger.warning("笔记列表获取失败: %s", e)
            return []

    def get_history(self, limit: int = 200) -> list:
        """获取当前用户所有历史记录。"""
        if self.memory is None:
            return []
        return self.memory.get_history(limit)

    def record_token_usage(self, method: str) -> None:
        """记录当前问答的 Token 消耗到 SQLite（从 LLMClient 累加器读取并重置）。

        Args:
            method: 检索方法。
        """
        usage = self._llm_client.total_token_usage
        if usage["total_tokens"] > 0:
            self._metadata_store.record_token_usage(
                user_id=self._user_id,
                method=method,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
            )
            logger.info(
                "Token 用量已记录: method=%s, prompt=%d, completion=%d, total=%d",
                method, usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"],
            )
            self._llm_client.reset_token_counters()

    def get_token_stats(self) -> dict:
        """获取当前用户的 Token 用量统计。

        Returns:
            统计字典，见 MetadataStore.get_token_stats。
        """
        return self._metadata_store.get_token_stats(user_id=self._user_id)

    def record_feedback(
        self,
        question: str,
        answer_preview: str,
        method: str,
        rating: str,
        latency_sec: float = 0.0,
    ) -> None:
        """记录用户反馈到 SQLite。

        Args:
            question: 问题文本。
            answer_preview: 答案前 200 字。
            method: 检索方法。
            rating: 'useful' 或 'not_useful'。
            latency_sec: 检索延迟秒数。
        """
        self._metadata_store.add_feedback(
            user_id=self._user_id,
            question=question,
            answer_preview=answer_preview,
            method=method,
            rating=rating,
            latency_sec=latency_sec,
        )

    def save_conversation(self, messages: list) -> None:
        """持久化聊天记录。"""
        self._metadata_store.save_conversation(self._user_id, messages)

    def load_conversation(self) -> list:
        """加载持久化的聊天记录。"""
        return self._metadata_store.load_conversation(self._user_id)

    def clear_conversation(self) -> None:
        """清除持久化的聊天记录。"""
        self._metadata_store.clear_conversation(self._user_id)

    def get_feedback_stats(self, method: str | None = None) -> dict:
        """获取当前用户的反馈统计。

        Args:
            method: 按检索方法过滤。

        Returns:
            反馈统计字典。
        """
        return self._metadata_store.get_feedback_stats(
            user_id=self._user_id,
            method=method,
        )

    def delete_note(self, note_id: str) -> bool:
        """删除一条笔记。"""
        if self.memory is None:
            return False
        return self.memory.delete_note(note_id)

    @property
    def working_memory(self):
        """工作记忆实例（可注入到 session_state）。"""
        if self.memory is None:
            return None
        return self.memory.working

    def list_documents(self, doc_format: str | None = None) -> list[dict]:
        """列出已加载的文档（仅当前用户）。

        Args:
            doc_format: 按格式过滤，None 返回全部。

        Returns:
            文档信息字典列表。
        """
        return self._metadata_store.list_documents(doc_format=doc_format, user_id=self._user_id)

    def get_collection_stats(self) -> dict:
        """获取向量库统计信息。"""
        return self._vector_store.collection_stats(VectorStore.DOCUMENT_CHUNKS)

    # ==================================================================
    # 内部方法
    # ==================================================================

    def _apply_rerank(
        self,
        question: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """对检索结果用 Cross-Encoder 重排序。

        Args:
            question: 用户问题。
            results: 粗筛结果列表。
            top_k: 返回前 K 个。

        Returns:
            重排序后的结果列表。
        """
        if self._reranker is None:
            from src.retrieval.reranker import Reranker
            self._reranker = Reranker()

        docs = [{"text": r.text, "score": r.score, "meta": r} for r in results]
        reranked = self._reranker.rerank(question, docs, top_k=top_k)

        # 更新原始 SearchResult 的 score 为精排分数
        out: list[SearchResult] = []
        for doc in reranked:
            original: SearchResult = doc["meta"]
            original.score = doc.get("rerank_score", original.score)
            out.append(original)
        return out

    @staticmethod
    def _normalize_method(method: str) -> str:
        """规范化检索方法名。

        支持多种输入格式，统一转为内部 key：
        - "直接检索" / "direct" → "direct"
        - "MQE" / "mqe" → "mqe"
        - "HyDE" / "hyde" → "hyde"
        - "MQE+HyDE" / "mqe_hyde" / "mqe+hyde" → "mqe+hyde"

        Args:
            method: 原始方法名。

        Returns:
            规范化的内部 key。

        Raises:
            ValueError: 无法识别的方法名。
        """
        m = method.lower().strip().replace(" ", "_").replace("+", "_").replace("-", "_")

        mapping = {
            "direct": "direct",
            "直接检索": "direct",
            "mqe": "mqe",
            "hyde": "hyde",
            "mqe_hyde": "mqe+hyde",
            "mqehyde": "mqe+hyde",
            "combined": "mqe+hyde",
        }

        if m in mapping:
            return mapping[m]
        if m in ("direct", "mqe", "hyde", "mqe+hyde"):
            return m

        raise ValueError(
            f"无效的检索方法: '{method}'。"
            f"可选: direct / mqe / hyde / mqe+hyde"
        )

    def _get_retriever(self, method_key: str) -> BaseRetriever:
        """获取或创建检索器实例（延迟实例化 + 缓存）。

        Args:
            method_key: 规范化的方法 key。

        Returns:
            BaseRetriever 实例。
        """
        where = {"user_id": self._user_id}

        if method_key not in self._retrievers:
            if method_key == "direct":
                self._retrievers[method_key] = DirectRetriever(
                    embedder=self._embedder,
                    vector_store=self._vector_store,
                    where_filter=where,
                )
            elif method_key == "mqe":
                self._retrievers[method_key] = MQERetriever(
                    llm_client=self._llm_client,
                    embedder=self._embedder,
                    vector_store=self._vector_store,
                    num_variants=self._config.retrieval.mqe_num_variants,
                    where_filter=where,
                )
            elif method_key == "hyde":
                self._retrievers[method_key] = HyDERetriever(
                    llm_client=self._llm_client,
                    embedder=self._embedder,
                    vector_store=self._vector_store,
                    where_filter=where,
                )
            elif method_key == "mqe+hyde":
                self._retrievers[method_key] = CombinedRetriever(
                    llm_client=self._llm_client,
                    embedder=self._embedder,
                    vector_store=self._vector_store,
                    config=self._config.retrieval,
                    where_filter=where,
                )
            else:
                raise ValueError(f"未知检索方法: {method_key}")

        return self._retrievers[method_key]

    @staticmethod
    def _to_source_chunks(results: list[SearchResult]) -> list[SourceChunk]:
        """将 SearchResult 列表转为 SourceChunk 列表（含位置描述）。

        Args:
            results: 检索结果列表。

        Returns:
            SourceChunk 列表。
        """
        fmt = CitationFormatter()
        return [
            SourceChunk(
                chunk_id=r.chunk_id,
                doc_name=r.metadata.get("doc_name", "未知文档"),
                text=r.text,
                score=r.score,
                metadata=r.metadata,
                location_text=fmt._format_location(r.metadata),
            )
            for r in results
        ]
