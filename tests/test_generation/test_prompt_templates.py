"""
Prompt 模板单元测试。

验证所有模板的占位符变量正确，可被 .format() 正常渲染。
"""

from src.generation.prompt_templates import (
    CONCEPT_EXTRACTION_SYSTEM,
    HYPOTHETICAL_ANSWER_SYSTEM,
    LEARNING_REPORT_SYSTEM,
    NOTE_SUMMARY_SYSTEM,
    QUERY_VARIANTS_SYSTEM,
    RAG_QA_NO_CONTEXT_SYSTEM,
    RAG_QA_SYSTEM,
)


# ============================================================================
# RAG 问答模板
# ============================================================================


class TestRagQaSystem:
    """RAG_QA_SYSTEM 模板测试。"""

    def test_format_all_placeholders(self):
        result = RAG_QA_SYSTEM.format(
            context="[1] Transformer 由 Vaswani 提出",
            question="什么是 Transformer？",
        )
        assert "[1] Transformer 由 Vaswani 提出" in result
        assert "什么是 Transformer？" in result
        assert "内联引用" in result

    def test_missing_placeholder_raises_keyerror(self):
        """缺少占位符变量应抛出 KeyError。"""
        with pytest.raises(KeyError):
            RAG_QA_SYSTEM.format(context="...")  # 缺少 question

    def test_empty_context(self):
        result = RAG_QA_SYSTEM.format(context="", question="test")
        assert "test" in result


class TestRagQaNoContextSystem:
    """RAG_QA_NO_CONTEXT_SYSTEM 模板测试。"""

    def test_format_with_question(self):
        result = RAG_QA_NO_CONTEXT_SYSTEM.format(question="什么是 AI？")
        assert "什么是 AI？" in result
        assert "没有可用的参考文档" in result

    def test_missing_question_raises_keyerror(self):
        with pytest.raises(KeyError):
            RAG_QA_NO_CONTEXT_SYSTEM.format()


# ============================================================================
# MQE 查询变体模板
# ============================================================================


class TestQueryVariantsSystem:
    """QUERY_VARIANTS_SYSTEM 模板测试。"""

    def test_format_with_num_variants(self):
        for n in [2, 4, 8]:
            result = QUERY_VARIANTS_SYSTEM.format(num_variants=n)
            assert str(n) in result
            assert "JSON" in result

    def test_missing_num_variants_raises_keyerror(self):
        with pytest.raises(KeyError):
            QUERY_VARIANTS_SYSTEM.format()


# ============================================================================
# HyDE 模板
# ============================================================================


class TestHypotheticalAnswerSystem:
    """HYPOTHETICAL_ANSWER_SYSTEM 模板测试。"""

    def test_no_placeholders(self):
        """该模板无占位符，直接使用即可。"""
        result = HYPOTHETICAL_ANSWER_SYSTEM
        assert "假设性的答案" in result
        assert "200-400字" in result


# ============================================================================
# 概念提取模板
# ============================================================================


class TestConceptExtractionSystem:
    """CONCEPT_EXTRACTION_SYSTEM 模板测试。"""

    def test_no_placeholders(self):
        assert "知识图谱" in CONCEPT_EXTRACTION_SYSTEM
        assert "JSON" in CONCEPT_EXTRACTION_SYSTEM


# ============================================================================
# 学习报告模板
# ============================================================================


class TestLearningReportSystem:
    """LEARNING_REPORT_SYSTEM 模板测试。"""

    def test_format_all_placeholders(self):
        result = LEARNING_REPORT_SYSTEM.format(
            qa_records="3 次问答",
            concepts="Transformer, BERT",
            notes="学习了 NLP 基础",
        )
        assert "3 次问答" in result
        assert "Transformer, BERT" in result
        assert "学习了 NLP 基础" in result
        assert "学习概览" in result

    def test_empty_inputs(self):
        result = LEARNING_REPORT_SYSTEM.format(
            qa_records="无", concepts="无", notes="无",
        )
        assert "无" in result


# ============================================================================
# 笔记摘要模板
# ============================================================================


class TestNoteSummarySystem:
    """NOTE_SUMMARY_SYSTEM 模板测试。"""

    def test_no_placeholders(self):
        assert "摘要" in NOTE_SUMMARY_SYSTEM
        assert "JSON" in NOTE_SUMMARY_SYSTEM


import pytest
