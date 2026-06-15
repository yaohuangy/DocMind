"""
检索结果去重模块。

对向量检索返回的多个分块做文本相似度去重，
合并高度重叠的片段，减少送入 LLM 的冗余信息。

算法：字符 bigram Jaccard 相似度——快速、语言无关、无需嵌入 API 调用。
复杂度 O(N²) for N chunks，但 N 通常 ≤20，实际耗时可忽略。
"""

from __future__ import annotations

import re
from collections.abc import Callable

from src.core.vector_store import SearchResult


def _normalize(text: str) -> str:
    """文本归一化：去空白、标点，保留字母数字和中日韩字符。"""
    # 合并连续空白
    text = re.sub(r"\s+", " ", text.lower())
    # 保留字母、数字、中文、日文、韩文
    text = re.sub(r"[^\w一-鿿぀-ゟ゠-ヿ가-힯]", "", text)
    return text


def _bigrams(text: str) -> set[str]:
    """提取字符 bigram 集合。"""
    return {text[i:i + 2] for i in range(len(text) - 1)} if len(text) >= 2 else set()


def _jaccard(a: str, b: str) -> float:
    """字符 bigram Jaccard 相似度 [0, 1]。"""
    ba = _bigrams(a)
    bb = _bigrams(b)
    if not ba or not bb:
        return 0.0
    intersection = len(ba & bb)
    union = len(ba | bb)
    return intersection / union if union > 0 else 0.0


def deduplicate_chunks(
    results: list[SearchResult],
    threshold: float = 0.65,
    key_fn: Callable[[SearchResult], str] | None = None,
) -> list[SearchResult]:
    """对检索结果做文本相似度去重。

    将相似度 >= threshold 的 chunk 归为一组，每组保留分数最高的那个。
    不改变相对顺序（高分 chunk 保持在前）。

    Args:
        results: 检索结果列表。
        threshold: 相似度阈值 [0, 1]。默认 0.65。越低去重越激进。
        key_fn: 提取去重文本的函数。默认直接用 result.text。

    Returns:
        去重后的 SearchResult 列表（保留原始 score 和顺序）。
    """
    if len(results) <= 1:
        return results

    if key_fn is None:
        key_fn = lambda r: r.text

    # 归一化文本
    norms = [(_normalize(key_fn(r)), r) for r in results]

    # 贪心去重：按原始顺序遍历，跳过与已保留 chunk 高度相似的
    kept: list[SearchResult] = []
    kept_norms: list[str] = []

    for norm, result in norms:
        if not norm:
            kept.append(result)
            kept_norms.append(norm)
            continue

        is_dup = False
        for kn in kept_norms:
            if _jaccard(norm, kn) >= threshold:
                # 如果当前分数更高，替换已保留的 (罕见情况)
                if result.score > kept[kept_norms.index(kn)].score:
                    kept[kept_norms.index(kn)] = result
                    kept_norms[kept_norms.index(kn)] = norm
                is_dup = True
                break

        if not is_dup:
            kept.append(result)
            kept_norms.append(norm)

    # 按分数重新排序（保持高质量结果靠前）
    kept.sort(key=lambda r: r.score, reverse=True)

    return kept
