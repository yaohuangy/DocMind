"""
多格式文档加载器——分发器模式。

根据输入类型（文件扩展名或 URL 前缀）自动路由到对应的 Parser，
统一返回 List[LlamaDocument]，下游管线完全格式无关。

支持的格式：PDF, WEB(URL), DOCX, MD, TXT, PPTX, CSV, XLSX
"""

import logging
import os
from enum import Enum

from llama_index.core.schema import Document as LlamaDocument

from src.ingestion.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)


class DocFormat(Enum):
    """支持的文档格式枚举。"""

    PDF = "pdf"
    WEB = "web"       # HTTP/HTTPS URL
    DOCX = "docx"
    MARKDOWN = "md"
    TEXT = "txt"
    PPTX = "pptx"
    CSV = "csv"
    EXCEL = "xlsx"


# 扩展名 → DocFormat 映射
_EXTENSION_MAP: dict[str, DocFormat] = {
    ".pdf": DocFormat.PDF,
    ".docx": DocFormat.DOCX,
    ".md": DocFormat.MARKDOWN,
    ".txt": DocFormat.TEXT,
    ".pptx": DocFormat.PPTX,
    ".csv": DocFormat.CSV,
    ".xlsx": DocFormat.EXCEL,
}


class MultiFormatLoader:
    """多格式文档加载器——分发器模式。

    根据输入类型自动检测格式，路由到对应 Parser，
    统一返回 ``List[LlamaDocument]``。
    下游管线（分块→嵌入→入库）完全格式无关。

    Usage::

        loader = MultiFormatLoader()
        docs = loader.load("/path/to/paper.pdf")
        docs = loader.load("https://example.com/article")
        all_docs = loader.load_batch(["/path/a.pdf", "/path/b.docx", "https://..."])
    """

    def __init__(self) -> None:
        """初始化分发器，注册所有 8 个格式解析器。"""
        self._parsers: dict[DocFormat, BaseParser] = {}

        # 延迟导入各 parser，避免一次性加载所有依赖
        from src.ingestion.parsers.csv_parser import CSVParser
        from src.ingestion.parsers.docx_parser import DocxParser
        from src.ingestion.parsers.excel_parser import ExcelParser
        from src.ingestion.parsers.markdown_parser import MarkdownParser
        from src.ingestion.parsers.pdf_parser import PDFParser
        from src.ingestion.parsers.pptx_parser import PptxParser
        from src.ingestion.parsers.text_parser import TextParser
        from src.ingestion.parsers.web_parser import WebParser

        self.register_parser(DocFormat.PDF, PDFParser())
        self.register_parser(DocFormat.WEB, WebParser())
        self.register_parser(DocFormat.DOCX, DocxParser())
        self.register_parser(DocFormat.MARKDOWN, MarkdownParser())
        self.register_parser(DocFormat.TEXT, TextParser())
        self.register_parser(DocFormat.PPTX, PptxParser())
        self.register_parser(DocFormat.CSV, CSVParser())
        self.register_parser(DocFormat.EXCEL, ExcelParser())

        logger.info("MultiFormatLoader 初始化: 已注册 %d 个解析器", len(self._parsers))

    # ------------------------------------------------------------------
    # Parser 注册
    # ------------------------------------------------------------------

    def register_parser(self, fmt: DocFormat, parser: BaseParser) -> None:
        """注册格式解析器。

        Args:
            fmt: 文档格式枚举值。
            parser: 对应的解析器实例。
        """
        self._parsers[fmt] = parser
        logger.debug("注册解析器: %s → %s", fmt.value, type(parser).__name__)

    # ------------------------------------------------------------------
    # 格式检测
    # ------------------------------------------------------------------

    @staticmethod
    def detect_format(source: str) -> DocFormat | None:
        """自动检测文档格式。

        规则：
        - http:// 或 https:// 开头 → WEB
        - 常见文件扩展名 → 对应格式
        - 其他 → None

        Args:
            source: 文档来源（文件路径或 URL）。

        Returns:
            DocFormat 枚举值，无法识别返回 None。
        """
        if not source:
            return None

        # URL 检测
        if source.startswith(("http://", "https://")):
            return DocFormat.WEB

        # 扩展名检测（大小写不敏感）
        ext = os.path.splitext(source)[1].lower()
        return _EXTENSION_MAP.get(ext)

    @staticmethod
    def get_format_name(source: str) -> str:
        """获取格式的显示名称。

        Args:
            source: 文档来源。

        Returns:
            格式名称字符串，未知返回 "unknown"。
        """
        fmt = MultiFormatLoader.detect_format(source)
        return fmt.value if fmt else "unknown"

    # ------------------------------------------------------------------
    # 加载入口
    # ------------------------------------------------------------------

    def load(self, source: str) -> list[LlamaDocument]:
        """统一加载入口——根据 source 自动选择 Parser。

        Args:
            source: 文件路径（如 /path/to/doc.pdf）或网页 URL。

        Returns:
            LlamaDocument 列表。解析失败返回空列表。

        Raises:
            ValueError: 无法识别的格式。
        """
        fmt = self.detect_format(source)
        if fmt is None:
            raise ValueError(
                f"无法识别文档格式: {source}。"
                f"支持的扩展名: {list(_EXTENSION_MAP.keys())}"
            )

        parser = self._parsers.get(fmt)
        if parser is None:
            raise RuntimeError(f"格式 {fmt.value} 的解析器未注册")

        logger.info("加载文档 [%s]: %s", fmt.value, source)

        try:
            documents = parser.parse(source)
            if not documents:
                logger.warning("文档 %s 未提取到有效内容", source)
            return documents
        except Exception as e:
            logger.error("文档加载失败 [%s] %s: %s", fmt.value, source, e)
            return []

    def load_batch(self, sources: list[str]) -> list[LlamaDocument]:
        """批量加载多个文档。

        每个 source 独立解析，一个失败不影响其他。
        所有结果平铺返回。

        Args:
            sources: 文档来源列表。

        Returns:
            所有文档的 LlamaDocument 列表（平铺）。
        """
        all_docs: list[LlamaDocument] = []
        for source in sources:
            try:
                docs = self.load(source)
                if docs:
                    all_docs.extend(docs)
            except ValueError as e:
                logger.warning("跳过不支持格式: %s", e)
            except Exception as e:
                logger.error("批量加载 %s 失败: %s", source, e)

        logger.info("批量加载完成: %d 个来源 → %d 个文档", len(sources), len(all_docs))
        return all_docs

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    @property
    def supported_formats(self) -> list[str]:
        """返回所有支持的格式标识列表。"""
        return [fmt.value for fmt in self._parsers.keys()]

    def has_parser(self, fmt: DocFormat) -> bool:
        """检查是否已注册指定格式的解析器。

        Args:
            fmt: 文档格式。

        Returns:
            是否已注册。
        """
        return fmt in self._parsers
