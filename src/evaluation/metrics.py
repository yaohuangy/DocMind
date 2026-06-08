"""
评测指标计算。

纯函数模块，不依赖任何外部服务。可直接单元测试。
"""

import math
from typing import Dict, List, Set, Tuple


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """计算 Recall@k。

    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Args:
        retrieved_ids: 检索结果 ID 列表（按分数降序）。
        relevant_ids: Ground Truth 相关分块 ID 集合。
        k: 截断位置。

    Returns:
        Recall@k (0.0 ~ 1.0)。relevant_ids 为空时返回 0.0。
    """
    if not relevant_ids:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    hits = len(retrieved_set & relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """计算 Precision@k。

    Precision@k = |retrieved[:k] ∩ relevant| / k

    Args:
        retrieved_ids: 检索结果 ID 列表。
        relevant_ids: Ground Truth 相关分块 ID 集合。
        k: 截断位置。

    Returns:
        Precision@k (0.0 ~ 1.0)。
    """
    if k <= 0:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    hits = len(retrieved_set & relevant_ids)
    return hits / k


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """计算 Mean Reciprocal Rank。

    MRR = 1 / rank_of_first_relevant
    rank 从 1 开始计数。未命中返回 0.0。

    Args:
        retrieved_ids: 检索结果 ID 列表。
        relevant_ids: Ground Truth 相关分块 ID 集合。

    Returns:
        MRR 值。
    """
    if not relevant_ids:
        return 0.0

    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """计算 NDCG@k（Normalized Discounted Cumulative Gain）。

    使用二元相关性（在 relevant_ids 中为 1，否则为 0）。

    DCG@k = Σ rel_i / log₂(i + 2)
    IDCG@k = 理想 DCG（前 |relevant| 个全命中）。
    NDCG@k = DCG / IDCG。

    Args:
        retrieved_ids: 检索结果 ID 列表。
        relevant_ids: Ground Truth 相关分块 ID 集合。
        k: 截断位置。

    Returns:
        NDCG@k (0.0 ~ 1.0)。
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    # DCG
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k]):
        rel = 1.0 if cid in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i 从 0 开始，log₂(i+2)

    # IDCG: 理想情况下前 min(k, |relevant|) 个全命中
    idcg = 0.0
    ideal_hits = min(k, len(relevant_ids))
    for i in range(ideal_hits):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_all_metrics(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k_values: Tuple[int, ...] = (5, 10),
) -> Dict[str, float]:
    """计算全套评测指标。

    Args:
        retrieved_ids: 检索结果 ID 列表。
        relevant_ids: Ground Truth 相关分块 ID 集合。
        k_values: 截断位置元组。

    Returns:
        指标字典，键如 "recall@5", "recall@10", "precision@5",
        "precision@10", "mrr", "ndcg@5", "ndcg@10"。
    """
    metrics: Dict[str, float] = {}

    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

    metrics["mrr"] = mrr(retrieved_ids, relevant_ids)

    return metrics


def percentile(values: List[float], p: float) -> float:
    """计算分位数（线性插值）。

    Args:
        values: 数值列表。
        p: 百分位（0~100）。

    Returns:
        分位数值。
    """
    if not values:
        return 0.0

    sorted_vals = sorted(values)
    k = (p / 100.0) * (len(sorted_vals) - 1)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_vals[int(k)]

    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1
