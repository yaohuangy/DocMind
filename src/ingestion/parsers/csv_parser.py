"""
CSV 文档解析器。

将 CSV 每行作为一个 chunk，自动检测分隔符和编码。
每行数据转为 "列名: 值" 格式的文本表示。
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional

from llama_index.core.schema import Document as LlamaDocument

from src.ingestion.parsers.base_parser import BaseParser, make_doc_id

logger = logging.getLogger(__name__)


class CSVParser(BaseParser):
    """CSV 解析器——每行数据作为一个 LlamaDocument。

    Usage::

        parser = CSVParser()
        docs = parser.parse("/path/to/data.csv")
        for d in docs:
            print(d.metadata["row_start"], d.text[:100])
    """

    @property
    def supported_format(self) -> str:
        return "csv"

    def parse(self, source: str) -> List[LlamaDocument]:
        """解析 CSV 文件。

        Args:
            source: .csv 文件路径。

        Returns:
            LlamaDocument 列表，每行数据一个文档。
        """
        doc_id = make_doc_id(source)
        doc_name = self._source_name(source)
        file_path = Path(source)

        # 编码检测
        try:
            raw_bytes = file_path.read_bytes()
        except Exception as e:
            logger.error("无法读取 CSV 文件 %s: %s", source, e)
            return []

        encoding = self._detect_encoding(raw_bytes)
        text_content = raw_bytes.decode(encoding, errors="replace")

        # 尝试读取
        rows = self._parse_csv(text_content)
        if not rows or len(rows) < 2:
            logger.warning("CSV 文件 %s 无有效数据行", source)
            return []

        # 第一行作为列名
        headers = rows[0]
        data_rows = rows[1:]

        documents: List[LlamaDocument] = []

        for row_idx, row in enumerate(data_rows):
            # 构建 "col: value" 格式文本
            parts: List[str] = []
            for col_idx, value in enumerate(row):
                col_name = headers[col_idx] if col_idx < len(headers) else f"Column{col_idx}"
                parts.append(f"{col_name}: {value}")

            row_text = "\n".join(parts)

            if not row_text.strip():
                continue

            doc = LlamaDocument(
                text=row_text,
                metadata={
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "source": source,
                    "format": "csv",
                    "row_start": row_idx + 2,  # 1-based, 跳过表头
                    "row_end": row_idx + 2,
                    "total_rows": len(data_rows),
                    "columns": str(headers),
                },
            )
            documents.append(doc)

        logger.info(
            "CSV 解析完成: %s (%d 行数据, %d 列)",
            doc_name, len(documents), len(headers),
        )
        return documents

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_encoding(raw_bytes: bytes) -> str:
        """检测 CSV 文件编码。"""
        try:
            import chardet
            result = chardet.detect(raw_bytes)
            return result.get("encoding") or "utf-8"
        except ImportError:
            return "utf-8"

    @staticmethod
    def _parse_csv(text: str) -> List[List[str]]:
        """解析 CSV 文本，自动检测分隔符。

        优先尝试 CSV sniffing，失败则回退到逗号分隔。

        Args:
            text: CSV 文本内容。

        Returns:
            二维列表，[行][列]。
        """
        try:
            # 尝试嗅探方言
            sample = text[:8192]
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
            logger.debug("CSV 检测: delimiter=%r, has_header=%s",
                         getattr(dialect, 'delimiter', ','), has_header)
        except Exception:
            # 回退到默认逗号分隔
            dialect = csv.excel

        reader = csv.reader(text.splitlines(), dialect)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        return rows
