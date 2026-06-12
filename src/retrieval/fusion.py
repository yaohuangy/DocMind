"""
融合算法模块。

实现两种核心融合策略：
1. **Reciprocal Rank Fusion (RRF)** — 多路检索结果融合，不依赖绝对相似度分数
2. **加权合并 (Weighted Merge)** — 两组结果按权重合并，含 Min-Max 归一化
"""

import logging

from src.core.vector_store import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    top_k: int = 10,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion 算法。

    将多路排序结果列表融合为一个排序列表，不依赖各路的绝对分数。
    核心公式：score(d) = Σ 1 / (k + rank_i(d))
    其中 rank_i(d) 是文档 d 在第 i 路结果中的排名（0-based）。

    参考：Cormack et al. (2009) "Reciprocal Rank Fusion outperforms Condorcet..."

    Args:
        result_lists: 多路检索结果列表，每个子列表已按分数降序排列。
        k: 平滑常数，默认 60。k 越大，高排名 vs 低排名的权重差异越小。
        top_k: 最终返回的结果数。

    Returns:
        RRF 融合后的 SearchResult 列表，score 字段为 RRF 分数。
    """
    if not result_lists:
        return []

    # chunk_id → (累积 RRF 分数, 最优 text, 最优 metadata)
    rrf_scores: dict[str, float] = {}
    best_text: dict[str, str] = {}
    best_metadata: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            cid = result.chunk_id
            contribution = 1.0 / (k + rank)

            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contribution

            # 保留首次出现的 text 和 metadata（或可改用分数最高的那个）
            if cid not in best_text:
                best_text[cid] = result.text
                best_metadata[cid] = result.metadata

    # 按 RRF 分数降序排列
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused: list[SearchResult] = []
    for cid in sorted_ids[:top_k]:
        fused.append(SearchResult(
            chunk_id=cid,
            text=best_text.get(cid, ""),
            score=round(rrf_scores[cid], 4),
            metadata=best_metadata.get(cid, {}),
        ))

    logger.debug("RRF 融合: %d 路 → %d 结果 (k=%d)", len(result_lists), len(fused), k)
    return fused


# ---------------------------------------------------------------------------
# 加权合并 (Weighted Merge)
# ---------------------------------------------------------------------------

def weighted_merge(
    results_a: list[SearchResult],
    results_b: list[SearchResult],
    weight_a: float = 0.4,
    weight_b: float = 0.6,
    top_k: int = 10,
    normalize: bool = True,
) -> list[SearchResult]:
    """两组检索结果加权合并。

    对每组结果做 Min-Max 归一化后，按权重计算最终分数：
        final_score(d) = weight_a × norm_score_a(d) + weight_b × norm_score_b(d)

    如果某 chunk 只在一组中出现，另一组贡献 0。

    Args:
        results_a: 第一组检索结果（如 MQE 结果）。
        results_b: 第二组检索结果（如 HyDE 结果）。
        weight_a: 第一组权重，默认 0.4。
        weight_b: 第二组权重，默认 0.6。
        top_k: 最终返回的结果数。
        normalize: 是否对组内分数做 Min-Max 归一化。关闭则直接用原始分数。

    Returns:
        加权合并后的 SearchResult 列表，score 字段为加权后分数。
    """
    if not results_a and not results_b:
        return []

    # 构建 chunk_id → 原始分数 的映射
    scores_a = _to_score_map(results_a)
    scores_b = _to_score_map(results_b)

    # Min-Max 归一化（组内）
    if normalize and scores_a:
        scores_a = _minmax_normalize(scores_a)
    if normalize and scores_b:
        scores_b = _minmax_normalize(scores_b)

    # 加权合并
    all_ids = set(scores_a.keys()) | set(scores_b.keys())
    merged_scores: dict[str, float] = {}

    for cid in all_ids:
        sa = scores_a.get(cid, 0.0)
        sb = scores_b.get(cid, 0.0)
        merged_scores[cid] = weight_a * sa + weight_b * sb

    # 排序
    sorted_ids = sorted(merged_scores.keys(), key=lambda cid: merged_scores[cid], reverse=True)

    # 保留最佳 text/metadata（优先 scores_a 或分数高的那个）
    best_text: dict[str, str] = {}
    best_metadata: dict[str, dict] = {}

    # 收集所有结果用于回溯 text/metadata
    for r in results_a + results_b:
        if r.chunk_id not in best_text:
            best_text[r.chunk_id] = r.text
            best_metadata[r.chunk_id] = r.metadata

    merged: list[SearchResult] = []
    for cid in sorted_ids[:top_k]:
        merged.append(SearchResult(
            chunk_id=cid,
            text=best_text.get(cid, ""),
            score=round(merged_scores[cid], 4),
            metadata=best_metadata.get(cid, {}),
        ))

    logger.debug(
        "加权合并: |A|=%d × %.1f + |B|=%d × %.1f → %d 结果",
        len(results_a), weight_a, len(results_b), weight_b, len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# 去重辅助
# ---------------------------------------------------------------------------

def deduplicate(
    results: list[SearchResult],
    key_fn=None,
) -> list[SearchResult]:
    """按 chunk_id 去重，保留分数最高的那个。

    Args:
        results: 可能含重复 chunk_id 的列表。
        key_fn: 可选的自定义去重键函数，默认按 chunk_id。

    Returns:
        去重后的列表，按分数降序排列。
    """
    if key_fn is None:
        key_fn = lambda r: r.chunk_id

    seen: dict[str, SearchResult] = {}
    for r in results:
        k = key_fn(r)
        if k not in seen or r.score > seen[k].score:
            seen[k] = r

    deduped = sorted(seen.values(), key=lambda r: r.score, reverse=True)
    return deduped


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _to_score_map(results: list[SearchResult]) -> dict[str, float]:
    """将 SearchResult 列表转为 chunk_id → score 映射。"""
    return {r.chunk_id: r.score for r in results}


def _minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-Max 归一化：将所有分数映射到 [0, 1] 区间。

    Args:
        scores: chunk_id → 原始分数。

    Returns:
        chunk_id → 归一化后分数。
    """
    if not scores:
        return scores

    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        # 所有分数相同，全部置为 1.0
        return {k: 1.0 for k in scores}

    return {
        k: (v - min_val) / (max_val - min_val)
        for k, v in scores.items()
    }
