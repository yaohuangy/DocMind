"""
ChunkConfig 补充测试。

覆盖 get_chunk_params 的各格式预设查询和回退逻辑。
"""

import pytest

from src.core.config import ChunkConfig


class TestChunkConfigGetChunkParams:
    """ChunkConfig.get_chunk_params() 测试。"""

    def test_known_format_pdf(self):
        config = ChunkConfig()
        size, overlap = config.get_chunk_params("pdf")
        assert size == 1024
        assert overlap == 256

    def test_known_format_csv(self):
        config = ChunkConfig()
        size, overlap = config.get_chunk_params("csv")
        assert size == 384
        assert overlap == 64

    def test_known_format_xlsx(self):
        config = ChunkConfig()
        size, overlap = config.get_chunk_params("xlsx")
        assert size == 384
        assert overlap == 64

    def test_known_format_pptx(self):
        config = ChunkConfig()
        size, overlap = config.get_chunk_params("pptx")
        assert size == 768
        assert overlap == 192

    def test_known_format_web(self):
        config = ChunkConfig()
        size, overlap = config.get_chunk_params("web")
        assert size == 768
        assert overlap == 192

    def test_known_format_docx(self):
        config = ChunkConfig()
        size, overlap = config.get_chunk_params("docx")
        assert size == 768
        assert overlap == 192

    def test_known_format_md(self):
        config = ChunkConfig()
        size, overlap = config.get_chunk_params("md")
        assert size == 768
        assert overlap == 192

    def test_known_format_txt(self):
        config = ChunkConfig()
        size, overlap = config.get_chunk_params("txt")
        assert size == 768
        assert overlap == 192

    def test_unknown_format_falls_back(self):
        """未配置的格式回退到全局默认值。"""
        config = ChunkConfig(chunk_size=512, chunk_overlap=64)
        size, overlap = config.get_chunk_params("unknown_format")
        assert size == 512
        assert overlap == 64

    def test_custom_presets(self):
        """自定义预设覆盖默认。"""
        config = ChunkConfig(
            chunk_size=100,
            chunk_overlap=20,
            chunk_presets={"custom_fmt": (200, 50)},
        )
        size, overlap = config.get_chunk_params("custom_fmt")
        assert size == 200
        assert overlap == 50

    def test_case_sensitive_match(self):
        """格式名大小写必须完全匹配。"""
        config = ChunkConfig()
        size, overlap = config.get_chunk_params("PDF")  # 大写
        assert size == 1024  # 回退到默认值，因为 "PDF" != "pdf"
        assert overlap == 128

    def test_all_defined_formats_have_valid_params(self):
        """所有预设格式的参数必须为正整数。"""
        config = ChunkConfig()
        for fmt, (size, overlap) in config.chunk_presets.items():
            assert size > 0, f"{fmt}: chunk_size={size} 必须 > 0"
            assert overlap >= 0, f"{fmt}: chunk_overlap={overlap} 必须 >= 0"
            assert overlap < size, f"{fmt}: overlap={overlap} 必须 < size={size}"


class TestChunkConfigDefaults:
    """ChunkConfig 默认值测试。"""

    def test_default_chunk_size(self):
        config = ChunkConfig()
        assert config.chunk_size == 1024

    def test_default_chunk_overlap(self):
        config = ChunkConfig()
        assert config.chunk_overlap == 128

    def test_default_use_semantic(self):
        config = ChunkConfig()
        assert config.use_semantic_chunking is False

    def test_default_min_chunk_tokens(self):
        config = ChunkConfig()
        assert config.min_chunk_tokens == 50
