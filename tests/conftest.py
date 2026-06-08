"""
Pytest 全局 fixtures 和配置。

提供 mock 对象和测试工具函数，
避免测试依赖真实的 LLM / ChromaDB / Neo4j 服务。
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 将项目根目录加入 sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# 设置测试环境变量（确保不依赖 .env 文件）
os.environ.setdefault("LLM_API_KEY", "test-api-key")
os.environ.setdefault("LLM_BASE_URL", "https://api.test.com/v1")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("EMBEDDING_BACKEND", "api")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("CHROMA_PERSIST_PATH", "./data/test_chroma")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "password")
os.environ.setdefault("SQLITE_PATH", "./data/test_metadata.db")
os.environ.setdefault("DEBUG", "true")


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_client():
    """返回模拟的 LLMClient。"""
    with patch("src.core.llm_client.OpenAI") as mock_openai:
        client_mock = MagicMock()
        # 模拟 chat
        client_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))]
        )
        mock_openai.return_value = client_mock

        from src.core.llm_client import LLMClient
        from src.core.config import LLMConfig

        config = LLMConfig(
            api_key="test-key",
            base_url="https://api.test.com/v1",
            model="test-model",
        )
        yield LLMClient(config)


@pytest.fixture
def mock_embedder():
    """返回模拟的 Embedder（返回固定维度向量）。"""
    with patch("src.core.embedder.OpenAI") as mock_openai:
        embed_mock = MagicMock()
        embed_mock.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )
        mock_openai.return_value = embed_mock

        from src.core.embedder import APIEmbedder
        from src.core.config import EmbeddingConfig

        config = EmbeddingConfig(backend="api", model="text-embedding-3-small")
        yield APIEmbedder(config)


@pytest.fixture
def mock_config():
    """返回测试用配置。"""
    from src.core.config import Settings

    return Settings()


@pytest.fixture
def sample_llama_document():
    """返回一个示例 LlamaDocument。"""
    from llama_index.core.schema import Document as LlamaDocument

    return LlamaDocument(
        text="Transformer 架构由 Vaswani 等人在 2017 年提出。"
             "其核心是自注意力机制（Self-Attention），"
             "允许模型在处理序列时关注不同位置的信息。",
        metadata={
            "doc_id": "abc123def456",
            "doc_name": "transformer_intro.pdf",
            "source": "/path/to/transformer_intro.pdf",
            "format": "pdf",
            "page_number": 3,
            "total_pages": 10,
        },
    )


@pytest.fixture
def sample_search_results():
    """返回示例检索结果列表。"""
    from src.core.vector_store import SearchResult

    return [
        SearchResult(
            chunk_id="chunk_001",
            text="Transformer 架构由 Vaswani 等人在 2017 年提出。",
            score=0.95,
            metadata={
                "doc_id": "abc123",
                "doc_name": "transformer_paper.pdf",
                "format": "pdf",
                "page_number": 3,
            },
        ),
        SearchResult(
            chunk_id="chunk_002",
            text="自注意力机制允许模型关注序列中不同位置的信息。",
            score=0.87,
            metadata={
                "doc_id": "abc123",
                "doc_name": "transformer_paper.pdf",
                "format": "pdf",
                "page_number": 5,
            },
        ),
    ]


# ============================================================================
# 通用工具
# ============================================================================

@pytest.fixture(autouse=True)
def reset_config_singleton():
    """每个测试前重置配置单例。"""
    from src.core import config as cfg_mod
    cfg_mod._config_singleton = None
    yield
    cfg_mod._config_singleton = None
