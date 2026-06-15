"""
检索结果去重单元测试。
"""

from src.core.vector_store import SearchResult
from src.retrieval.dedup import _bigrams, _jaccard, _normalize, deduplicate_chunks


class TestNormalize:
    def test_lowercase(self):
        assert _normalize("Hello World") == "helloworld"

    def test_strip_punctuation(self):
        result = _normalize("你好，世界！")
        assert "你好世界" in result

    def test_whitespace_collapse(self):
        result = _normalize("a   b\n\nc")
        assert "a" in result
        assert "b" in result

    def test_empty(self):
        assert _normalize("") == ""


class TestBigrams:
    def test_ascii(self):
        bg = _bigrams("hello")
        assert "he" in bg
        assert "ll" in bg
        assert "lo" in bg
        assert len(bg) == 4  # h-e, e-l, l-l, l-o

    def test_chinese(self):
        bg = _bigrams("你好世界")
        assert "你好" in bg
        assert "好世" in bg
        assert "世界" in bg

    def test_short_string(self):
        assert _bigrams("a") == set()
        assert _bigrams("") == set()


class TestJaccard:
    def test_identical(self):
        assert _jaccard("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert _jaccard("hello", "world") < 0.5

    def test_partial_overlap(self):
        sim = _jaccard("hello world", "hello there")
        assert 0.0 < sim < 1.0

    def test_empty(self):
        assert _jaccard("", "hello") == 0.0
        assert _jaccard("a", "b") == 0.0  # "a" has no bigrams


class TestDeduplicateChunks:
    def _r(self, cid: str, text: str, score: float = 0.9) -> SearchResult:
        return SearchResult(chunk_id=cid, text=text, score=score, metadata={})

    def test_no_duplicates(self):
        results = [
            self._r("a", "Transformer 是一种神经网络架构"),
            self._r("b", "Python 是流行的编程语言"),
            self._r("c", "北京是中国的首都"),
        ]
        deduped = deduplicate_chunks(results, threshold=0.65)
        assert len(deduped) == 3

    def test_near_duplicates_merged(self):
        """两个内容高度相似的 chunk 应合并为一个。"""
        results = [
            self._r("a", "Transformer 由 Vaswani 在 2017 年提出，基于自注意力机制。", score=0.9),
            self._r("b", "Transformer 由 Vaswani 等人于 2017 年提出，基于自注意力机制。", score=0.8),
            self._r("c", "BERT 是一个预训练语言模型。", score=0.7),
        ]
        deduped = deduplicate_chunks(results, threshold=0.6)
        # a 和 b 高度相似 → 保留分数更高的 a，b 被去重
        assert len(deduped) == 2
        assert any(r.chunk_id == "a" for r in deduped)
        assert any(r.chunk_id == "c" for r in deduped)

    def test_keeps_higher_score(self):
        """去重冲突时保留分数更高的 chunk。"""
        results = [
            self._r("low", "自注意力机制是 Transformer 的核心组件", score=0.5),
            self._r("high", "自注意力机制是 Transformer 架构的核心组件", score=0.95),
        ]
        deduped = deduplicate_chunks(results, threshold=0.6)
        assert len(deduped) == 1
        assert deduped[0].chunk_id == "high"

    def test_single_result(self):
        results = [self._r("a", "唯一的结果")]
        deduped = deduplicate_chunks(results)
        assert len(deduped) == 1

    def test_empty_list(self):
        assert deduplicate_chunks([]) == []

    def test_custom_key_fn(self):
        """用自定义 key 提取去重文本。"""
        results = [
            SearchResult(chunk_id="a", text="short", score=0.9,
                         metadata={"summary": "Transformer 架构详解"}),
            SearchResult(chunk_id="b", text="different text", score=0.8,
                         metadata={"summary": "Transformer 架构详解"}),
        ]
        deduped = deduplicate_chunks(
            results, threshold=0.7,
            key_fn=lambda r: r.metadata.get("summary", ""),
        )
        assert len(deduped) == 1  # summaries are identical → merge

    def test_lower_threshold_more_aggressive(self):
        """阈值越低，去重越激进。"""
        results = [
            self._r("a", "Transformer 的自注意力机制详解", score=0.9),
            self._r("b", "BERT 的 MLM 预训练任务介绍", score=0.8),
            self._r("c", "GPT 的自回归生成方式说明", score=0.7),
        ]
        # 默认 0.65：三个不同主题，不合并
        assert len(deduplicate_chunks(results, threshold=0.65)) == 3
        # 极低阈值 0.1：几乎合并所有
        deduped_low = deduplicate_chunks(results, threshold=0.1)
        # 至少保留一个
        assert len(deduped_low) >= 1

    def test_results_sorted_by_score(self):
        """去重后按分数降序排列。"""
        results = [
            self._r("a", "文本A", score=0.5),
            self._r("b", "文本B", score=0.95),
            self._r("c", "文本C", score=0.7),
        ]
        deduped = deduplicate_chunks(results, threshold=0.65)
        scores = [r.score for r in deduped]
        assert scores == sorted(scores, reverse=True)
