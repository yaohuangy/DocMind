"""
Ground Truth 评测数据集生成器。

从 ChromaDB 已入库的分块中随机采样，使用 LLM 为每个分块
生成一个自然的问题，构建 {问题 → 相关分块ID} 的评测集。

支持两种增强模式（拉开 Direct vs MQE+HyDE 差距）：
- **改写** (rewrite): LLM 用完全不同的措辞重述问题，制造语义鸿沟
- **扩展 GT** (expand_gt): 将同文档相邻分块也标记为 GT，让指标更细腻
"""

import json
import logging
import random
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from src.core.llm_client import LLMClient
from src.core.vector_store import SearchResult, VectorStore
from src.evaluation.models import EvalQuestion

logger = logging.getLogger(__name__)

# 每个分块最少字符数（太短的不适合生成问题）
_MIN_CHUNK_CHARS = 100

# 相似度阈值（超过此值视为重复问题）
_DUPLICATE_SIMILARITY = 0.85

# 问题生成 System Prompt（强化版：必须用口语化、非术语方式提问）
QUESTION_GEN_SYSTEM = """\
你是一个检索评测数据集生成专家。给定一段文档内容，请你生成一个用户可能用来检索这段内容的问题。

核心要求：
1. 问题的答案必须能够在这段文档内容中找到
2. **禁止直接使用文档中的专业术语**——用日常语言、口语化的方式描述
3. **用完全不同的句式**——不要照搬文档原文的任何完整短语
4. 像一个初学者或普通用户会问的问题，而非专家查询
5. 避免太宽泛的问题（如"这篇文章讲了什么？"）
6. 问题语言与文档内容语言一致
7. 直接输出问题，不要加任何前缀或解释

文档内容：
{chunk_text}

请生成一个从措辞和句式上**完全不同于原文**的用户问题：
"""

# 问题改写 Prompt（对已生成的问题做二次改写，进一步拉开语义距离）
REWRITE_SYSTEM = """\
你是一个语言表达专家。请将以下问题用**完全不同的话**重新说一遍，但保持原意不变。

改写要求：
1. 换掉所有关键词汇（用同义词、日常说法替代专业术语）
2. 换掉句式（陈述式变疑问式、长句拆短句、被动变主动等）
3. 可以增减背景铺垫，但不能改变问题的核心意图
4. 像一个普通人在论坛/搜索引擎里会打出来的话
5. 直接输出改写后的问题，不要加前缀或解释

原问题：{question}
文档内容参考：{chunk_text}

改写后的问题："""



class GroundTruthGenerator:
    """Ground Truth 评测数据集生成器。

    从已入库的 ChromaDB 分块采样，用 LLM 为每个分块生成问题，
    输出 JSON 评测集。

    Usage::

        gen = GroundTruthGenerator()
        gen.generate_dataset(num_questions=30, output_path="data/evaluation/dataset.json")
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """
        Args:
            llm_client: LLM 客户端。None 则自动创建。
            vector_store: 向量存储。None 则自动创建。
        """
        self._llm_client = llm_client or LLMClient()
        self._vector_store = vector_store or VectorStore()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        num_questions: int = 30,
        output_path: str = "",
        delay_between_calls: float = 0.3,
        rewrite: bool = False,
        expand_gt: bool = False,
    ) -> list[EvalQuestion]:
        """生成评测数据集。

        Args:
            num_questions: 目标问题数。
            output_path: JSON 输出路径。为空则不保存。
            delay_between_calls: LLM 调用间隔秒数。
            rewrite: 是否对问题做二次改写（制造语义鸿沟，拉开 Direct vs MQE+HyDE 差距）。
            expand_gt: 是否将同文档相邻分块也标记为 GT（让指标更细腻，不再非 0 即 1）。

        Returns:
            EvalQuestion 列表。
        """
        # 1. 检查是否有文档
        stats = self._vector_store.collection_stats(VectorStore.DOCUMENT_CHUNKS)
        if stats.get("count", 0) == 0:
            raise RuntimeError(
                "ChromaDB 中没有文档分块。请先通过应用上传并摄入文档。"
            )

        total_chunks = stats["count"]
        sample_size = min(num_questions * 3, total_chunks)  # 多采样以应对跳过的

        logger.info(
            "开始生成评测集: 目标 %d 问题, 从 %d 个分块中采样 %d 个, 改写=%s, 扩展GT=%s",
            num_questions, total_chunks, sample_size, rewrite, expand_gt,
        )

        # 2. 采样分块
        sampled = self._sample_chunks(sample_size)
        if not sampled:
            raise RuntimeError("采样失败，未能获取任何分块。")

        # 3. 为每个分块生成问题
        questions: list[EvalQuestion] = []
        seen_questions: list[str] = []

        for i, chunk in enumerate(sampled):
            if len(questions) >= num_questions:
                break

            # 过滤太短的分块
            text = chunk.text.strip()
            if len(text) < _MIN_CHUNK_CHARS:
                continue

            # 生成问题
            q_text = self._generate_question(text)
            if not q_text:
                logger.warning("第 %d 个分块问题生成失败，跳过", i + 1)
                continue

            # 改写（制造语义鸿沟）
            if rewrite:
                rewritten = self._rewrite_question(q_text, text)
                if rewritten:
                    q_text = rewritten
                    if delay_between_calls > 0:
                        import time
                        time.sleep(delay_between_calls)
                else:
                    logger.warning("问题改写失败，使用原始问题: %s", q_text[:60])

            # 去重
            if self._is_duplicate(q_text, seen_questions):
                logger.debug("跳过重复问题: %s", q_text[:60])
                continue

            seen_questions.append(q_text)

            # GT 分块：扩展为同文档相邻分块
            gt_ids = [chunk.chunk_id]
            if expand_gt:
                gt_ids = self._expand_gt_chunks(chunk)

            eq = EvalQuestion(
                question=q_text,
                relevant_chunk_ids=gt_ids,
                doc_id=chunk.metadata.get("doc_id", ""),
                source_chunk_text=text[:500],
            )
            questions.append(eq)

            logger.info(
                "[%d/%d] 生成问题: %s",
                len(questions), num_questions, q_text[:80],
            )

            # API 限流保护
            if delay_between_calls > 0 and len(questions) < num_questions:
                import time
                time.sleep(delay_between_calls)

        logger.info("评测集生成完成: %d 个问题", len(questions))

        # 4. 保存
        if output_path:
            self._save(questions, output_path)

        return questions

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _sample_chunks(self, n: int) -> list[SearchResult]:
        """从 ChromaDB 随机采样 n 个分块。

        Args:
            n: 采样数量。

        Returns:
            SearchResult 列表。
        """
        try:
            collection = self._vector_store.get_collection(
                VectorStore.DOCUMENT_CHUNKS
            )
            # 获取所有分块
            data = collection.get(
                include=["documents", "metadatas"],
            )

            if not data["ids"]:
                return []

            total = len(data["ids"])
            n = min(n, total)
            indices = random.sample(range(total), n)

            results: list[SearchResult] = []
            for idx in indices:
                cid = data["ids"][idx]
                text = (data["documents"][idx] if data.get("documents") else "")
                meta = (data["metadatas"][idx] if data.get("metadatas") else {})

                results.append(SearchResult(
                    chunk_id=cid,
                    text=text or "",
                    score=0.0,
                    metadata=meta or {},
                ))

            return results
        except Exception as e:
            logger.error("采样分块失败: %s", e)
            return []

    def _generate_question(self, chunk_text: str) -> str:
        """调用 LLM 为分块生成一个问题。

        Args:
            chunk_text: 分块文本。

        Returns:
            问题字符串，失败返回空。
        """
        system = QUESTION_GEN_SYSTEM.format(chunk_text=chunk_text)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "请根据上述文档内容生成一个问题。"},
        ]

        try:
            result = self._llm_client.chat(messages, temperature=0.7, max_tokens=128)
            question = result.strip()
            # 清理 LLM 可能的输出前缀
            for prefix in ["问题：", "问题:", "Q:", "Q："]:
                if question.startswith(prefix):
                    question = question[len(prefix):].strip()
            return question
        except Exception as e:
            logger.error("LLM 问题生成失败: %s", e)
            return ""

    def _rewrite_question(self, question: str, chunk_text: str) -> str:
        """对已生成的问题做二次改写——替换术语和句式。

        Args:
            question: 原始生成的问题。
            chunk_text: 源分块文本（提供上下文）。

        Returns:
            改写后的问题，失败返回空。
        """
        system = REWRITE_SYSTEM.format(question=question, chunk_text=chunk_text)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "请改写上述问题。"},
        ]

        try:
            result = self._llm_client.chat(messages, temperature=0.8, max_tokens=200)
            rewritten = result.strip()
            for prefix in ["改写后的问题：", "改写：", "问题：", "问题:"]:
                if rewritten.startswith(prefix):
                    rewritten = rewritten[len(prefix):].strip()
            # 去括号说明
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            return rewritten
        except Exception as e:
            logger.warning("LLM 问题改写失败: %s", e)
            return ""

    def _expand_gt_chunks(self, source_chunk: SearchResult) -> list[str]:
        """将 GT 从单个分块扩展为同文档的相邻分块。

        找到与 source_chunk 属于同一 doc_id 的分块，
        取前一个和后一个（按 chunk_index 排序），
        返回 [prev_id, source_id, next_id]（去重去空）。

        Args:
            source_chunk: 源分块。

        Returns:
            扩展后的分块 ID 列表。
        """
        doc_id = source_chunk.metadata.get("doc_id", "")
        if not doc_id:
            return [source_chunk.chunk_id]

        src_idx = source_chunk.metadata.get("chunk_index", -1)
        if src_idx < 0:
            return [source_chunk.chunk_id]

        try:
            collection = self._vector_store.get_collection(
                VectorStore.DOCUMENT_CHUNKS
            )
            # 获取同一 doc_id 的所有分块
            data = collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"],
            )
        except Exception:
            return [source_chunk.chunk_id]

        if not data["ids"] or not data.get("metadatas"):
            return [source_chunk.chunk_id]

        # 构建 (chunk_id, chunk_index) 列表，按 chunk_index 排序
        chunks_with_idx = []
        for cid, meta in zip(data["ids"], data["metadatas"]):
            idx = meta.get("chunk_index", -1) if meta else -1
            chunks_with_idx.append((cid, idx))

        chunks_with_idx.sort(key=lambda x: x[1])

        # 找到源分块位置及相邻分块
        gt_ids = [source_chunk.chunk_id]

        for pos, (cid, idx) in enumerate(chunks_with_idx):
            if cid == source_chunk.chunk_id:
                # 前一个
                if pos > 0:
                    gt_ids.insert(0, chunks_with_idx[pos - 1][0])
                # 后一个
                if pos < len(chunks_with_idx) - 1:
                    gt_ids.append(chunks_with_idx[pos + 1][0])
                break

        return gt_ids

    @staticmethod
    def _is_duplicate(question: str, existing: list[str]) -> bool:
        """检查问题是否与已有问题过于相似。

        Args:
            question: 新问题。
            existing: 已有问题列表。

        Returns:
            是否重复。
        """
        for prev in existing:
            similarity = SequenceMatcher(None, question, prev).ratio()
            if similarity >= _DUPLICATE_SIMILARITY:
                return True
        return False

    @staticmethod
    def _save(questions: list[EvalQuestion], output_path: str) -> None:
        """保存数据集到 JSON。

        Args:
            questions: EvalQuestion 列表。
            output_path: 输出路径。
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for q in questions:
            data.append({
                "question": q.question,
                "relevant_chunk_ids": q.relevant_chunk_ids,
                "doc_id": q.doc_id,
                "source_chunk_text": q.source_chunk_text,
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "num_questions": len(data),
                    "questions": data,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info("评测集已保存: %s (%d 条)", output_path, len(data))
