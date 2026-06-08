"""
配置模块单元测试。
"""

import os

import pytest

from src.core.config import (
    LLMConfig,
    Settings,
    get_config,
    load_config,
)


class TestLLMConfig:
    """LLMConfig 数据类测试。"""

    def test_default_values(self):
        config = LLMConfig()
        assert config.model == "gpt-4o"
        assert config.temperature == 0.0
        assert config.max_tokens == 2048

    def test_override_values(self):
        config = LLMConfig(model="custom-model", temperature=0.5)
        assert config.model == "custom-model"
        assert config.temperature == 0.5


class TestSettings:
    """Settings 聚合配置测试。"""

    def test_default_creation(self):
        settings = Settings()
        assert settings.llm.model == "gpt-4o"
        assert settings.retrieval.top_k == 10
        assert settings.chunk.chunk_size == 1024

    def test_config_singleton(self, reset_config_singleton):
        """测试配置单例。"""
        cfg1 = load_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_env_override(self, reset_config_singleton):
        """测试环境变量覆盖。"""
        os.environ["RETRIEVAL_TOP_K"] = "15"
        cfg = load_config()
        assert cfg.retrieval.top_k == 15
        # 清理
        del os.environ["RETRIEVAL_TOP_K"]
