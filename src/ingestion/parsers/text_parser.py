"""
纯文本 (.txt) 解析器。

使用 chardet 自动检测文件编码，按段落分割文本。
记录每个段落在原文件中的起止行号。
"""

import logging
from pathlib import Path
from typing import List

from llama_index.core.schema import Document as LlamaDocument

from src.ingestion.parsers.base_parser import BaseParser, make_doc_id

logger = logging.getLogger(__name__)


class TextParser(BaseParser):
    """纯文本解析器——chardet 编码检测 + 按段落分割。

    Usage::

        parser = TextParser()
        docs = parser.parse("/path/to/notes.txt")
    """

    @property
    def supported_format(self) -> str:
        return "txt"

    def parse(self, source: str) -> List[LlamaDocument]:
        """解析 TXT 文件。

        Args:
            source: .txt 文件路径。

        Returns:
            LlamaDocument 列表，每段一个文档。
        """
        doc_id = make_doc_id(source)
        doc_name = self._source_name(source)
        file_path = Path(source)

        # 读取原始字节
        try:
            raw_bytes = file_path.read_bytes()
        except Exception as e:
            logger.error("无法读取 TXT 文件 %s: %s", source, e)
            return []

        if not raw_bytes:
            return []

        # 编码检测
        encoding, confidence = self._detect_encoding(raw_bytes)
        logger.info("TXT 编码检测: %s (置信度 %.2f)", encoding, confidence)

        # 解码
        try:
            text = raw_bytes.decode(encoding, errors="replace")
        except Exception:
            text = raw_bytes.decode("utf-8", errors="replace")

        if not text.strip():
            return []

        # 按段落分割（空行分隔）
        lines = text.split("\n")
        paragraphs = self._group_paragraphs(lines)

        documents: List[LlamaDocument] = []

        for para_text, line_start, line_end in paragraphs:
            if not para_text.strip():
                continue

            doc = LlamaDocument(
                text=para_text.strip(),
                metadata={
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "source": source,
                    "format": "txt",
                    "line_start": line_start,  # 1-based
                    "line_end": line_end,
                    "encoding": encoding,
                },
            )
            documents.append(doc)

        logger.info(
            "TXT 解析完成: %s (%d 段, %d 字符, 编码=%s)",
            doc_name, len(documents), sum(len(d.text) for d in documents), encoding,
        )
        return documents

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_encoding(raw_bytes: bytes) -> tuple:
        """使用 chardet 检测编码。

        Args:
            raw_bytes: 原始字节。

        Returns:
            (encoding, confidence) 元组。
        """
        try:
            import chardet
            result = chardet.detect(raw_bytes)
            encoding = result.get("encoding") or "utf-8"
            confidence = result.get("confidence", 0.0)
            return str(encoding), float(confidence)
        except ImportError:
            logger.warning("chardet 未安装，回退到 UTF-8")
            return "utf-8", 0.0

    @staticmethod
    def _group_paragraphs(
        lines: List[str], max_para_lines: int = 50
    ) -> List[tuple]:
        """将行列表按空行分组为段落。

        超长段落会被拆分为多个子段落。

        Args:
            lines: 文本行列表。
            max_para_lines: 单个段落最大行数，超过则拆分。

        Returns:
            (text, line_start, line_end) 元组列表，均为 1-based。
        """
        paragraphs: List[tuple] = []
        current: List[str] = []
        current_start = 1  # 1-based

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()

            if stripped == "" and current:
                # 空行 → 段落结束
                text = "\n".join(current)
                paragraphs.append((text, current_start, idx - 1))
                current = []
                # 空行后下一行是新的开始
                current_start = idx + 1
            elif stripped != "":
                current.append(line)

            # 超长段落拆分
            if len(current) >= max_para_lines:
                text = "\n".join(current)
                paragraphs.append((text, current_start, idx))
                current = []
                current_start = idx + 1

        # 最后一段
        if current:
            paragraphs.append((("\n".join(current)), current_start, len(lines)))

        return paragraphs
