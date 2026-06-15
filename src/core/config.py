"""
全局配置模块。

从 .env 文件和环境变量中加载配置，提供单例 Settings 实例。
所有模块通过 `get_config()` 获取统一配置，确保全项目配置一致性。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

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

    # 重排序（Cross-Encoder 精排）
    use_reranker: bool = False     # 启用 BAAI/bge-reranker-v2-m3 重排序
    reranker_top_k: int = 20       # 粗筛取 top-N 后再精排

    # 去重（文本相似度去重）
    use_dedup: bool = True          # 启用检索结果语义去重
    dedup_threshold: float = 0.65   # bigram Jaccard 相似度阈值（0-1）


@dataclass
class ChunkConfig:
    """分块参数配置。

    支持按文档格式自适应分块：``chunk_presets`` 字典为每种格式指定
    独立的 (chunk_size, chunk_overlap)。未配置的格式回退到全局默认值。
    Markdown 文档按 ## / ### 标题切分，此处的 preset 仅用于超大段落的
    token 分块回退。

    Usage::

        config = ChunkConfig()
        size, overlap = config.get_chunk_params("pdf")   # → (1024, 256)
        size, overlap = config.get_chunk_params("unknown") # → (1024, 128)
    """

    chunk_size: int = 1024
    chunk_overlap: int = 128

    # 按文档格式的自适应分块预设 → {格式: (chunk_size, chunk_overlap)}
    chunk_presets: dict[str, tuple[int, int]] = field(default_factory=lambda: {
        "pdf": (1024, 256),    # 技术文档/论文，大分块保证概念完整
        "pptx": (768, 192),    # PPT 幻灯片，单页内容稀疏
        "web": (768, 192),     # 网页正文，噪声过滤后中等密度
        "docx": (768, 192),    # Word 文档
        "md": (768, 192),      # Markdown（超大段落回退用）
        "txt": (768, 192),     # 纯文本
        "csv": (384, 64),      # 表格数据，每行独立
        "xlsx": (384, 64),     # 电子表格
    })

    # ------------------------------------------------------------------
    # 语义分块（SemanticSplitterNodeParser）
    # ------------------------------------------------------------------
    use_semantic_chunking: bool = False
    """启用语义分块：按句子嵌入相似度在话题边界切分。
    启用后非 md 文档优先使用 SemanticSplitterNodeParser，
    token 分块（SentenceSplitter）仅作为超大块的兜底。
    需要 llama-index-embeddings-openai 包。"""

    semantic_buffer_size: int = 1
    """语义比较的上下文窗口（相邻句子数）。
    1 = 逐句比较；2-5 = 平滑短离题但粗糙边界。"""

    semantic_breakpoint_percentile: int = 95
    """断点分位数阈值（百分位）。越小切分越激进、块越多。
    95 = 仅最不相似的 5% 句子对触发切分。"""

    semantic_max_chunk_multiplier: float = 2.0
    """语义块最大 token 倍数（相对于 chunk_size）。
    超过此倍数的语义块回退到 SentenceSplitter token 分块。
    例如 chunk_size=512 × 2.0 = 最大 1024 token 的语义块。"""

    min_chunk_tokens: int = 50
    """最小分块 token 数。低于此阈值的碎片合并到前一块。"""

    def get_chunk_params(self, doc_format: str) -> tuple[int, int]:
        """获取指定文档格式的分块参数。

        Args:
            doc_format: 文档格式标识（如 pdf / web / docx）。

        Returns:
            (chunk_size, chunk_overlap) 元组。
        """
        preset = self.chunk_presets.get(doc_format)
        if preset is not None:
            return preset
        return (self.chunk_size, self.chunk_overlap)


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

_config_singleton: Settings | None = None


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
            use_reranker=_env_bool("USE_RERANKER", False),
            reranker_top_k=_env_int("RERANKER_TOP_K", 20),
            use_dedup=_env_bool("RETRIEVAL_DEDUP", True),
            dedup_threshold=_env_float("RETRIEVAL_DEDUP_THRESHOLD", 0.65),
        ),
        chunk=ChunkConfig(
            chunk_size=_env_int("CHUNK_SIZE", 1024),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 128),
            use_semantic_chunking=_env_bool("CHUNK_SEMANTIC", False),
            semantic_buffer_size=_env_int("CHUNK_SEMANTIC_BUFFER_SIZE", 1),
            semantic_breakpoint_percentile=_env_int(
                "CHUNK_SEMANTIC_BREAKPOINT_PERCENTILE", 95
            ),
            semantic_max_chunk_multiplier=_env_float(
                "CHUNK_SEMANTIC_MAX_MULTIPLIER", 2.0
            ),
            min_chunk_tokens=_env_int("CHUNK_MIN_TOKENS", 50),
        ),
        app_name=_env("APP_NAME", "Docmind"),
        app_port=_env_int("APP_PORT", 7860),
        debug=_env_bool("DEBUG", False),
    )

    return _config_singleton


def get_config() -> Settings:
    """load_config() 的别名，语义更清晰。"""
    return load_config()
