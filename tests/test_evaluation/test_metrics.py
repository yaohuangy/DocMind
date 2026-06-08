"""
评测指标单元测试。
"""

import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    mrr,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        relevant = {"a", "b", "c"}
        retrieved = ["a", "b", "c", "d", "e"]
        assert recall_at_k(retrieved, relevant, k=3) == 1.0
        assert recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_partial_recall(self):
        relevant = {"a", "b", "c"}
        retrieved = ["a", "d", "e", "b", "f"]
        assert recall_at_k(retrieved, relevant, k=5) == 2 / 3

    def test_zero_recall(self):
        relevant = {"a", "b"}
        retrieved = ["c", "d", "e"]
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), k=3) == 0.0

    def test_k_larger_than_retrieved(self):
        relevant = {"a"}
        retrieved = ["a", "b"]
        assert recall_at_k(retrieved, relevant, k=10) == 1.0


class TestPrecisionAtK:
    def test_perfect_precision(self):
        relevant = {"a", "b"}
        retrieved = ["a", "b"]
        assert precision_at_k(retrieved, relevant, k=2) == 1.0

    def test_partial_precision(self):
        relevant = {"a", "b"}
        retrieved = ["a", "c", "d", "b", "e"]
        assert precision_at_k(retrieved, relevant, k=5) == 2 / 5

    def test_zero_k(self):
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_third_position(self):
        assert mrr(["x", "y", "a", "z"], {"a"}) == 1.0 / 3

    def test_not_found(self):
        assert mrr(["x", "y"], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert mrr(["a"], set()) == 0.0


class TestNDCG:
    def test_perfect_ndcg(self):
        relevant = {"a", "b"}
        retrieved = ["a", "b", "c"]
        score = ndcg_at_k(retrieved, relevant, k=3)
        assert score == 1.0  # 理想顺序就是 a, b 在最前

    def test_imperfect_order(self):
        relevant = {"a", "b"}
        retrieved = ["c", "a", "b"]
        score = ndcg_at_k(retrieved, relevant, k=3)
        assert 0.0 < score < 1.0

    def test_empty(self):
        assert ndcg_at_k([], {"a"}, k=5) == 0.0


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        m = compute_all_metrics(["a", "b", "c", "d", "e"], {"b", "d"})
        assert "recall@5" in m
        assert "recall@10" in m
        assert "precision@5" in m
        assert "precision@10" in m
        assert "mrr" in m
        assert "ndcg@5" in m
        assert "ndcg@10" in m

    def test_perfect_match(self):
        m = compute_all_metrics(["a", "b", "c"], {"a", "b", "c"}, k_values=(3, 5))
        assert m["recall@3"] == 1.0
        assert m["precision@3"] == 1.0
        assert m["mrr"] == 1.0
        assert m["ndcg@3"] == 1.0


class TestPercentile:
    def test_median(self):
        assert percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_p95(self):
        values = list(range(100))
        assert percentile(values, 95) == 94.05

    def test_empty(self):
        assert percentile([], 50) == 0.0

    def test_single(self):
        assert percentile([5.0], 50) == 5.0
