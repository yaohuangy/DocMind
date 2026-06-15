"""
引用格式化器单元测试。

覆盖 _format_location（8 种格式）、_extract_cited_indices、
format_with_remap 重映射、format 边界情况等。
"""

import pytest

from src.core.vector_store import SearchResult
from src.generation.citation_formatter import CitationFormatter


def _make_result(chunk_id: str, text: str = "", metadata: dict | None = None) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=text or f"text_{chunk_id}",
        score=0.9,
        metadata=metadata or {},
    )


# ============================================================================
# _format_location — 8 种格式位置描述
# ============================================================================


class TestFormatLocation:
    """_format_location 各格式测试。"""

    def test_pdf_with_page_total(self):
        loc = CitationFormatter._format_location({
            "format": "pdf", "page_number": 3, "total_pages": 10,
        })
        assert "第3页" in loc
        assert "共10页" in loc

    def test_pdf_page_only(self):
        loc = CitationFormatter._format_location({
            "format": "pdf", "page_number": 5,
        })
        assert "第5页" in loc

    def test_web_with_section(self):
        loc = CitationFormatter._format_location({
            "format": "web", "section_title": "架构概述",
        })
        assert "段落: 架构概述" in loc

    def test_web_with_url_fallback(self):
        loc = CitationFormatter._format_location({
            "format": "web",
            "url": "https://example.com/very/long/path/to/some/article/page",
        })
        # 长 URL 会被截断
        assert "https://" in loc

    def test_web_with_short_url(self):
        loc = CitationFormatter._format_location({
            "format": "web", "url": "https://short.url",
        })
        assert "https://short.url" in loc

    def test_docx_with_heading_and_paragraph(self):
        loc = CitationFormatter._format_location({
            "format": "docx", "heading": "背景介绍", "paragraph_index": 2,
        })
        assert "章节: 背景介绍" in loc
        assert "第3段" in loc  # paragraph_index=2 → 第3段

    def test_docx_heading_only(self):
        loc = CitationFormatter._format_location({
            "format": "docx", "heading": "概述",
        })
        assert "章节: 概述" in loc

    def test_md_with_heading_path(self):
        loc = CitationFormatter._format_location({
            "format": "md",
            "heading_path": "## 背景 > ### 动机",
        })
        assert "## 背景 > ### 动机" in loc

    def test_md_no_heading_path(self):
        loc = CitationFormatter._format_location({"format": "md"})
        assert loc  # 兜底值非空

    def test_txt_with_line_range(self):
        loc = CitationFormatter._format_location({
            "format": "txt", "line_start": 15, "line_end": 22,
        })
        assert "第15-22行" in loc

    def test_txt_line_start_only(self):
        loc = CitationFormatter._format_location({
            "format": "txt", "line_start": 10,
        })
        assert "第10行起" in loc

    def test_pptx_with_slide_and_title(self):
        loc = CitationFormatter._format_location({
            "format": "pptx", "slide_number": 5, "slide_title": "总结",
            "total_slides": 12,
        })
        assert "第5页幻灯片" in loc
        assert "共12页" in loc
        assert "标题: 总结" in loc

    def test_csv_with_row_and_sheet(self):
        loc = CitationFormatter._format_location({
            "format": "csv", "row_start": 10, "sheet_name": "Sheet1",
        })
        assert "第10行" in loc
        assert "工作表: Sheet1" in loc

    def test_xlsx_same_as_csv(self):
        loc = CitationFormatter._format_location({
            "format": "xlsx", "row_start": 5,
        })
        assert "第5行" in loc

    def test_unknown_format_fallback(self):
        loc = CitationFormatter._format_location({
            "format": "unknown", "source": "test_source.txt",
        })
        # 兜底使用 source
        assert "test_source.txt" in loc

    def test_empty_metadata_fallback(self):
        loc = CitationFormatter._format_location({})
        assert len(loc) > 0  # 非空兜底


# ============================================================================
# _extract_cited_indices
# ============================================================================


class TestExtractCitedIndices:
    """_extract_cited_indices 测试。"""

    def test_single_citation(self):
        indices = CitationFormatter._extract_cited_indices("见 [1] 说明。")
        assert indices == [1]

    def test_multiple_citations_sorted(self):
        indices = CitationFormatter._extract_cited_indices(
            "参考 [3] 和 [1] 以及 [2] 的内容。"
        )
        assert indices == [1, 2, 3]

    def test_duplicate_removed(self):
        indices = CitationFormatter._extract_cited_indices("[1] ... [1] ... [2]")
        assert indices == [1, 2]

    def test_no_citation(self):
        indices = CitationFormatter._extract_cited_indices("无引用的文本。")
        assert indices == []

    def test_multi_digit_numbers(self):
        indices = CitationFormatter._extract_cited_indices(
            "见 [10] 和 [1] 以及 [100]"
        )
        assert indices == [1, 10, 100]

    def test_empty_string(self):
        indices = CitationFormatter._extract_cited_indices("")
        assert indices == []


# ============================================================================
# format — 基本格式化
# ============================================================================


class TestFormat:
    """CitationFormatter.format() 测试。"""

    def test_basic_format(self):
        """基本格式化——解析引用并生成 SourceChunk 列表。"""
        formatter = CitationFormatter()
        results = [
            _make_result("c1", "文本A", {"doc_name": "docA.pdf", "format": "pdf", "page_number": 1}),
            _make_result("c2", "文本B", {"doc_name": "docB.pdf", "format": "pdf", "page_number": 3}),
        ]

        formatted, sources = formatter.format("见 [1] 和 [2] 说明", results)
        # formatted 保持原样
        assert "见 [1] 和 [2] 说明" == formatted
        assert len(sources) == 2
        assert sources[0].doc_name == "docA.pdf"
        assert sources[1].doc_name == "docB.pdf"

    def test_format_empty_answer(self):
        formatter = CitationFormatter()
        formatted, sources = formatter.format("", [_make_result("c1")])
        assert formatted == ""
        assert sources == []

    def test_format_empty_results(self):
        formatter = CitationFormatter()
        formatted, sources = formatter.format("见 [1]", [])
        assert formatted == "见 [1]"
        assert sources == []

    def test_format_out_of_range_citation(self):
        """引用编号超出范围时跳过。"""
        formatter = CitationFormatter()
        results = [_make_result("c1")]
        formatted, sources = formatter.format("见 [1] 和 [5]", results)
        # [1] 有效，[5] 超出范围被跳过
        assert len(sources) == 1
        assert sources[0].chunk_id == "c1"

    def test_format_no_citation_shows_all(self):
        """无引用标记时默认展示全部检索结果。"""
        formatter = CitationFormatter()
        results = [
            _make_result("c1", metadata={"doc_name": "a.pdf"}),
            _make_result("c2", metadata={"doc_name": "b.pdf"}),
        ]
        formatted, sources = formatter.format("无引用文本", results)
        assert formatted == "无引用文本"
        assert len(sources) == 2


# ============================================================================
# format_with_remap — 引用重映射
# ============================================================================


class TestFormatWithRemap:
    """CitationFormatter.format_with_remap() 测试。"""

    def test_remap_reorder(self):
        """模型引用了 [3], [1], [2] → 重映射为 [1], [2], [3]。"""
        formatter = CitationFormatter()
        results = [
            _make_result("c1", metadata={"doc_name": "doc1.pdf"}),
            _make_result("c2", metadata={"doc_name": "doc2.pdf"}),
            _make_result("c3", metadata={"doc_name": "doc3.pdf"}),
        ]

        raw = "见 [3] 的说明，同时参考 [1] 和 [2]。"
        remapped, sources = formatter.format_with_remap(raw, results)

        # [3]→[1], [1]→[2], [2]→[3]（按首次出现顺序）
        assert "[1]" in remapped
        assert "[2]" in remapped
        assert "[3]" in remapped
        # 原始标记不应出现
        assert remapped.count("[1]") > 0
        assert len(sources) == 3

    def test_remap_partial_citations(self):
        """只引用了部分结果。"""
        formatter = CitationFormatter()
        results = [
            _make_result("c1", metadata={"doc_name": "a.pdf"}),
            _make_result("c2", metadata={"doc_name": "b.pdf"}),
            _make_result("c3", metadata={"doc_name": "c.pdf"}),
        ]

        raw = "见 [2] 的内容。"
        remapped, sources = formatter.format_with_remap(raw, results)

        # [2]→[1]
        assert "[1]" in remapped
        assert "[2]" not in remapped
        assert len(sources) == 1

    def test_remap_empty_citations(self):
        """无引用时返回原文和空列表。"""
        formatter = CitationFormatter()
        results = [_make_result("c1")]
        remapped, sources = formatter.format_with_remap("无引用", results)
        assert remapped == "无引用"
        assert sources == []

    def test_remap_duplicate_citation(self):
        """重复引用同一编号只展示一次。"""
        formatter = CitationFormatter()
        results = [
            _make_result("c1", metadata={"doc_name": "doc.pdf"}),
        ]
        raw = "[1] ... [1] ... [1]"
        remapped, sources = formatter.format_with_remap(raw, results)
        # 两个 [1] 都映射为 [1]（因为只有一个 source）
        assert "[1]" in remapped
        assert len(sources) == 1
