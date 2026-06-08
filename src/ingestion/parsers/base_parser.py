"""
文档解析器抽象基类。

所有格式解析器必须实现此接口：
- ``parse(source) -> List[LlamaDocument]``
- ``supported_format`` 属性返回对应 DocFormat 枚举值。
"""

import hashlib
from abc import ABC, abstractmethod
from typing import List

from llama_index.core.schema import Document as LlamaDocument


def make_doc_id(source: str) -> str:
    """从来源路径/URL 生成稳定的文档 ID（SHA256 前 16 字符）。

    Args:
        source: 文档来源（文件路径或 URL）。

    Returns:
        16 字符的十六进制字符串。
    """
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


class BaseParser(ABC):
    """文档解析器基类——所有 Parser 遵循此接口。

    子类只需实现 ``parse()`` 和 ``supported_format``。
    """

    @abstractmethod
    def parse(self, source: str) -> List[LlamaDocument]:
        """解析文档，返回 LlamaDocument 列表。

        每个 LlamaDocument 包含：
        - ``text``: 文本内容
        - ``metadata``: 字典，至少包含 doc_id, doc_name, source, format，
          以及格式特定的位置信息（如 page_number, url, slide_number 等）。

        Args:
            source: 文档来源（文件路径或 URL）。

        Returns:
            LlamaDocument 列表。
        """
        ...

    @property
    @abstractmethod
    def supported_format(self) -> str:
        """返回此解析器支持的格式标识（如 "pdf", "docx" 等）。"""
        ...

    # ------------------------------------------------------------------
    # 辅助方法（子类可复用）
    # ------------------------------------------------------------------

    @staticmethod
    def _make_doc_id(source: str) -> str:
        """生成文档 ID。"""
        return make_doc_id(source)

    @staticmethod
    def _source_name(source: str) -> str:
        """从完整路径或 URL 提取文档名称。"""
        import os
        from urllib.parse import urlparse

        if source.startswith(("http://", "https://")):
            parsed = urlparse(source)
            name = parsed.path.strip("/").split("/")[-1] or parsed.netloc
            return name
        return os.path.basename(source)
