"""
嵌入接口模块。

提供双后端嵌入能力：
- API 模式：通过 OpenAI 兼容 API 获取嵌入（如 text-embedding-3-small）
- 本地模式：通过 sentence-transformers 在本地计算嵌入（如 all-MiniLM-L6-v2）

通过配置中的 ``EMBEDDING_BACKEND`` 切换。
"""

import logging
from abc import ABC, abstractmethod

from openai import OpenAI

from src.core.config import EmbeddingConfig, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 抽象接口
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """嵌入器抽象基类。

    所有嵌入后端实现必须实现 embed() 和 embed_query()。
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """将多个文本批量转为嵌入向量。

        Args:
            texts: 文本列表。

        Returns:
            等长的嵌入向量列表，每个向量为 float 列表。
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """将单个查询文本转为嵌入向量。

        Args:
            text: 单个查询文本。

        Returns:
            嵌入向量。
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """嵌入向量的维度。"""
        ...


# ---------------------------------------------------------------------------
# API 模式嵌入
# ---------------------------------------------------------------------------

class APIEmbedder(BaseEmbedder):
    """基于 OpenAI 兼容 Embedding API 的嵌入器。

    使用 /v1/embeddings 接口获取嵌入向量。
    支持批量请求，自动处理空文本。
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        """
        Args:
            config: 嵌入配置。为 None 时自动从全局配置加载。
        """
        if config is None:
            config = get_config().embedding
        self._config = config

        # 使用独立 API key 或回退到 LLM API key
        api_key = config.api_key or get_config().llm.api_key
        base_url = config.base_url

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = config.model
        self._batch_size = config.batch_size or 32

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文本。

        Args:
            texts: 文本列表。

        Returns:
            等长的嵌入向量列表。
        """
        if not texts:
            return []

        # 去掉首尾空白，但保留空字符串（后续会处理）
        cleaned = [t.strip() for t in texts]

        all_embeddings: list[list[float]] = []

        # 分批请求
        for i in range(0, len(cleaned), self._batch_size):
            batch = cleaned[i : i + self._batch_size]
            logger.debug("嵌入请求: 批次 %d/%d (%d 条文本)", i // self._batch_size + 1,
                         (len(cleaned) + self._batch_size - 1) // self._batch_size,
                         len(batch))

            resp = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            batch_embeddings = [d.embedding for d in resp.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询文本。

        Args:
            text: 查询文本。

        Returns:
            嵌入向量。
        """
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """返回 API 模型的嵌入维度（首次调用会发一次请求探测）。"""
        # 各常见模型的维度映射，避免额外请求
        _KNOWN_DIMS = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        if self._model in _KNOWN_DIMS:
            return _KNOWN_DIMS[self._model]
        # 回退：发送简单文本获取维度
        try:
            vec = self.embed_query("dimension probe")
            return len(vec)
        except Exception:
            logger.warning("无法探测嵌入维度，返回默认 1536")
            return 1536


# ---------------------------------------------------------------------------
# 本地模式嵌入
# ---------------------------------------------------------------------------

class LocalEmbedder(BaseEmbedder):
    """基于 sentence-transformers 的本地嵌入器。

    模型首次加载后缓存在进程内存中，后续调用即时响应。
    推荐模型: all-MiniLM-L6-v2（384维）、bge-small-zh（512维）等。
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        """
        Args:
            config: 嵌入配置。为 None 时自动从全局配置加载。
        """
        if config is None:
            config = get_config().embedding
        self._config = config
        self._model_name = config.local_model
        self._batch_size = config.batch_size or 32
        self._model = None  # 延迟加载

    def _lazy_load(self):
        """延迟加载 sentence-transformers 模型。"""
        if self._model is not None:
            return
        logger.info("加载本地嵌入模型: %s", self._model_name)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        logger.info("模型加载完成，维度=%d", self.dimension)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文本。

        Args:
            texts: 文本列表。

        Returns:
            等长的嵌入向量列表。
        """
        if not texts:
            return []
        self._lazy_load()
        assert self._model is not None

        cleaned = [t.strip() for t in texts]
        # sentence-transformers encode 自动处理批处理
        embeddings = self._model.encode(
            cleaned,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询文本。"""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """返回本地模型的嵌入维度。"""
        self._lazy_load()
        assert self._model is not None
        return self._model.get_sentence_embedding_dimension()


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_embedder(config: EmbeddingConfig | None = None) -> BaseEmbedder:
    """根据配置创建嵌入器实例。

    Args:
        config: 嵌入配置。为 None 时自动从全局配置加载。

    Returns:
        APIEmbedder 或 LocalEmbedder 实例。

    Raises:
        ValueError: 当 backend 值不合法时。
    """
    if config is None:
        config = get_config().embedding

    if config.backend == "api":
        logger.info("使用 API 嵌入后端: %s", config.model)
        return APIEmbedder(config)
    elif config.backend == "local":
        logger.info("使用本地嵌入后端: %s", config.local_model)
        return LocalEmbedder(config)
    else:
        raise ValueError(
            f"不支持的嵌入后端 '{config.backend}'，可选: 'api' / 'local'"
        )
