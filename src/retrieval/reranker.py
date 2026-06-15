"""
重排序模块。

使用交叉编码器（Cross-Encoder）对向量检索的粗筛结果进行精排，
在不重新摄入文档的前提下提升检索精度。

默认模型：BAAI/bge-reranker-v2-m3（通过 ModelScope 下载到本地）。

Usage::

    reranker = Reranker()
    reranked = reranker.rerank("什么是RAG？", documents)
    # documents = [{"text": "...", "score": 0.8, ...}, ...]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# 模型缓存（进程级单例）
_reranker_model: CrossEncoder | None = None
_model_path: str | None = None


def _get_model_path() -> str:
    """获取模型路径（ModelScope 缓存 > HuggingFace 缓存）。"""
    from pathlib import Path

    # 检查 ModelScope 缓存
    ms_cache = Path("data/models/BAAI/bge-reranker-v2-m3")
    if ms_cache.exists():
        return str(ms_cache.resolve())

    # 回退到 HuggingFace 在线下载
    return "BAAI/bge-reranker-v2-m3"


class Reranker:
    """交叉编码器重排序器。

    加载 BAAI/bge-reranker-v2-m3 模型，
    对 (query, document) 对逐一打分，按相关性降序排列。

    Usage::

        reranker = Reranker()
        results = reranker.rerank("问题", [
            {"text": "文档A", "score": 0.8},
            {"text": "文档B", "score": 0.7},
        ])
    """

    def __init__(self, model_path: str | None = None):
        """
        Args:
            model_path: 模型路径，None 则自动检测（ModelScope > HF）。
        """
        global _reranker_model, _model_path  # noqa: PLW0603

        if model_path is None:
            model_path = _get_model_path()

        if _reranker_model is not None and _model_path == model_path:
            self._model = _reranker_model
        else:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("加载重排序模型: %s", model_path)
                self._model = CrossEncoder(model_path, trust_remote_code=True)
                _reranker_model = self._model
                _model_path = model_path
            except Exception as e:
                logger.warning(
                    "重排序模型加载失败（%s），重排序将禁用。可设置 USE_RERANKER=false 跳过。",
                    e,
                )
                self._model = None

    # ------------------------------------------------------------------
    # 重排序入口
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """对文档列表重排序。

        Args:
            query: 用户查询。
            documents: 文档列表，每项至少含 ``text`` 字段。
            top_k: 返回前 K 个结果。

        Returns:
            重排序后的文档列表，每项追加 ``rerank_score`` 字段。
        """
        if not documents or self._model is None:
            return documents[:top_k]

        # 构建 (query, doc_text) 对
        pairs = [(query, d.get("text", "")) for d in documents]

        # 批量打分
        scores = self._model.predict(
            pairs,  # type: ignore[arg-type]
            batch_size=32,
            show_progress_bar=False,
        )

        # 单值转列表
        scores_list: list[float]
        if not hasattr(scores, "__iter__"):
            scores_list = [float(scores)]  # type: ignore[arg-type]
        else:
            scores_list = [float(s) for s in scores]  # type: ignore[arg-type]

        # 合并分数并排序
        for doc, score in zip(documents, scores_list):
            doc["rerank_score"] = score

        documents.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
        return documents[:top_k]
