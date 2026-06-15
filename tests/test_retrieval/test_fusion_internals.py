"""
融合算法内部函数补充测试。

覆盖 _minmax_normalize 边界情况和 _to_score_map。
"""

import pytest

from src.core.vector_store import SearchResult
from src.retrieval.fusion import (
    _minmax_normalize,  # type: ignore[attr-defined]
    _to_score_map,  # type: ignore[attr-defined]
    weighted_merge,
)


class TestMinMaxNormalize:
    """Min-Max 归一化边界情况测试。"""

    def test_normal_range(self):
        scores = {"a": 0.5, "b": 1.0, "c": 0.0}
        normalized = _minmax_normalize(scores)
        assert normalized["c"] == 0.0
        assert normalized["b"] == 1.0
        assert 0.0 < normalized["a"] < 1.0

    def test_all_equal_scores(self):
        """所有分数相同时全部归一化为 1.0。"""
        scores = {"a": 0.5, "b": 0.5, "c": 0.5}
        normalized = _minmax_normalize(scores)
        for v in normalized.values():
            assert v == 1.0

    def test_empty_dict(self):
        normalized = _minmax_normalize({})
        assert normalized == {}

    def test_single_value(self):
        normalized = _minmax_normalize({"a": 0.7})
        assert normalized["a"] == 1.0

    def test_negative_scores(self):
        scores = {"a": -0.5, "b": 0.5}
        normalized = _minmax_normalize(scores)
        assert normalized["a"] == 0.0
        assert normalized["b"] == 1.0


class TestToScoreMap:
    """_to_score_map 测试。"""

    def test_basic(self):
        results = [
            SearchResult(chunk_id="a", text="", score=0.9, metadata={}),
            SearchResult(chunk_id="b", text="", score=0.7, metadata={}),
        ]
        score_map = _to_score_map(results)
        assert score_map == {"a": 0.9, "b": 0.7}

    def test_empty_list(self):
        assert _to_score_map([]) == {}


class TestWeightedMergeEdgeCases:
    """加权合并边界情况补充。"""

    def test_both_empty(self):
        merged = weighted_merge([], [], top_k=5)
        assert merged == []

    def test_no_normalize(self):
        """不归一化时直接用原始分数合并。"""
        a = [
            SearchResult(chunk_id="a", text="t", score=10.0, metadata={}),
            SearchResult(chunk_id="b", text="t", score=5.0, metadata={}),
        ]
        b = [
            SearchResult(chunk_id="a", text="t", score=8.0, metadata={}),
            SearchResult(chunk_id="c", text="t", score=4.0, metadata={}),
        ]
        merged = weighted_merge(a, b, weight_a=0.5, weight_b=0.5, normalize=False)
        assert len(merged) == 3
        # a: 0.5*10 + 0.5*8 = 9.0 → 应排第一
        assert merged[0].chunk_id == "a"
        assert merged[0].score == 9.0

    def test_top_k_truncation(self):
        a = [
            SearchResult(chunk_id=f"x{i}", text="t", score=1.0 - i * 0.1, metadata={})
            for i in range(10)
        ]
        merged = weighted_merge(a, [], weight_a=1.0, weight_b=0.0, top_k=3)
        assert len(merged) == 3
