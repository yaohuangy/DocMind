"""
全局配置模块。

从 .env 文件和环境变量中加载配置，提供单例 Settings 实例。
所有模块通过 `get_config()` 获取统一配置，确保全项目配置一致性。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 加载项目根目录 .env 文件（向上查找 src/core → src → 项目根）
_project_root = Path(__file__).resolve().parent.parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()  # 回退：从当前工作目录加载


@dataclass
class LLMConfig:
    """LLM 相关配置。"""

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 2048
    stream_timeout: int = 60  # 流式请求超时（秒）


@dataclass
class EmbeddingConfig:
    """嵌入模型相关配置。"""

    backend: str = "api"  # "api" 或 "local"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "text-embedding-3-small"
    local_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32  # 本地模型批处理大小


@dataclass
class ChromaConfig:
    """ChromaDB 配置。"""

    persist_path: str = "./data/chroma"


@dataclass
class Neo4jConfig:
    """Neo4j 图数据库配置。"""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"  # 默认数据库名（Neo4j 4.x+）


@dataclass
class SQLiteConfig:
    """SQLite 元数据库配置。"""

    path: str = "./data/metadata.db"


@dataclass
class RetrievalConfig:
    """检索参数配置。"""

    top_k: int = 10
    mqe_num_variants: int = 4
    hyde_weight: float = 0.6  # HyDE 结果在融合中的权重（direct 权重=0.4）
    rrf_k: int = 60  # RRF 平滑常数


@dataclass
class ChunkConfig:
    """分块参数配置。"""

    chunk_size: int = 1024
    chunk_overlap: int = 128


@dataclass
class Settings:
    """全局配置单例，聚合所有子配置。

    通过 load_config() 获取实例，所有字段由环境变量填充。
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    sqlite: SQLiteConfig = field(default_factory=SQLiteConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)

    # 应用级
    app_name: str = "Docmind"
    app_port: int = 7860
    debug: bool = False


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_config_singleton: Optional[Settings] = None


def _env(key: str, default: str = "") -> str:
    """读取环境变量，去首尾空白。"""
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    """读取整型环境变量。"""
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """读取浮点环境变量。"""
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    """读取布尔环境变量（1/true/yes 视为 True）。"""
    val = _env(key, str(default)).lower()
    return val in ("1", "true", "yes")


def load_config() -> Settings:
    """加载并返回全局配置单例。

    首次调用从环境变量构建 Settings 实例，后续调用返回缓存。
    环境变量约定见 .env.example。
    """
    global _config_singleton
    if _config_singleton is not None:
        return _config_singleton

    _config_singleton = Settings(
        llm=LLMConfig(
            api_key=_env("LLM_API_KEY"),
            base_url=_env("LLM_BASE_URL", "https://api.openai.com/v1"),
            model=_env("LLM_MODEL", "gpt-4o"),
            temperature=_env_float("LLM_TEMPERATURE", 0.0),
            max_tokens=_env_int("LLM_MAX_TOKENS", 2048),
            stream_timeout=_env_int("LLM_STREAM_TIMEOUT", 60),
        ),
        embedding=EmbeddingConfig(
            backend=_env("EMBEDDING_BACKEND", "api"),
            api_key=_env("EMBEDDING_API_KEY", _env("LLM_API_KEY")),
            base_url=_env("EMBEDDING_BASE_URL", _env("LLM_BASE_URL", "https://api.openai.com/v1")),
            model=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
            local_model=_env("EMBEDDING_LOCAL_MODEL", "all-MiniLM-L6-v2"),
            batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32),
        ),
        chroma=ChromaConfig(
            persist_path=_env("CHROMA_PERSIST_PATH", "./data/chroma"),
        ),
        neo4j=Neo4jConfig(
            uri=_env("NEO4J_URI", "bolt://localhost:7687"),
            user=_env("NEO4J_USER", "neo4j"),
            password=_env("NEO4J_PASSWORD", "password"),
            database=_env("NEO4J_DATABASE", "neo4j"),
        ),
        sqlite=SQLiteConfig(
            path=_env("SQLITE_PATH", "./data/metadata.db"),
        ),
        retrieval=RetrievalConfig(
            top_k=_env_int("RETRIEVAL_TOP_K", 10),
            mqe_num_variants=_env_int("MQE_NUM_VARIANTS", 4),
            hyde_weight=_env_float("HYDE_WEIGHT", 0.6),
            rrf_k=_env_int("RRF_K", 60),
        ),
        chunk=ChunkConfig(
            chunk_size=_env_int("CHUNK_SIZE", 1024),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 128),
        ),
        app_name=_env("APP_NAME", "Docmind"),
        app_port=_env_int("APP_PORT", 7860),
        debug=_env_bool("DEBUG", False),
    )

    return _config_singleton


def get_config() -> Settings:
    """load_config() 的别名，语义更清晰。"""
    return load_config()
