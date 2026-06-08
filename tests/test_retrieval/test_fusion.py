"""
融合算法单元测试。
"""

from src.core.vector_store import SearchResult
from src.retrieval.fusion import (
    deduplicate,
    reciprocal_rank_fusion,
    weighted_merge,
)


def _make_result(chunk_id: str, score: float, text: str = "") -> SearchResult:
    return SearchResult(chunk_id=chunk_id, text=text or f"text_{chunk_id}", score=score, metadata={})


class TestReciprocalRankFusion:
    """RRF 算法测试。"""

    def test_basic_fusion(self):
        """基本 RRF 融合——两路结果合并。"""
        list_a = [
            _make_result("a", 0.9),
            _make_result("b", 0.8),
            _make_result("c", 0.7),
        ]
        list_b = [
            _make_result("b", 0.95),  # b 在两路中都排第一
            _make_result("d", 0.85),
            _make_result("e", 0.6),
        ]

        fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_k=5)
        assert len(fused) <= 5
        # b 在 list_a 排第 2 (rank=1) + list_b 排第 1 (rank=0) → 总分最高
        assert fused[0].chunk_id == "b"

    def test_empty_input(self):
        """空输入。"""
        assert reciprocal_rank_fusion([], k=60) == []

    def test_single_list(self):
        """单路输入——直接返回原顺序（截断到 top_k）。"""
        items = [_make_result(f"x{i}", 1.0 - i * 0.1) for i in range(5)]
        fused = reciprocal_rank_fusion([items], k=60, top_k=3)
        assert len(fused) == 3
        assert fused[0].chunk_id == "x0"


class TestWeightedMerge:
    """加权合并测试。"""

    def test_basic_merge(self):
        list_a = [_make_result("a", 0.9), _make_result("b", 0.5)]
        list_b = [_make_result("b", 0.8), _make_result("c", 0.7)]

        merged = weighted_merge(list_a, list_b, weight_a=0.4, weight_b=0.6, top_k=3)
        assert len(merged) == 3

    def test_one_empty(self):
        """一组为空时直接返回另一组。"""
        items = [_make_result("a", 0.9)]
        merged = weighted_merge(items, [], weight_a=0.4, weight_b=0.6, top_k=5)
        assert len(merged) == 1
        assert merged[0].chunk_id == "a"


class TestDeduplicate:
    """去重测试。"""

    def test_dedup_keeps_highest_score(self):
        items = [
            _make_result("a", 0.5),
            _make_result("a", 0.9),
            _make_result("b", 0.7),
        ]
        deduped = deduplicate(items)
        assert len(deduped) == 2
        assert deduped[0].chunk_id == "a"
        assert deduped[0].score == 0.9
