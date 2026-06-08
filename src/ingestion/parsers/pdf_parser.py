"""
PDF 文档解析器。

使用 PyMuPDF (fitz) 提取文本，保留页码元数据。
每页生成一个 LlamaDocument，metadata 中包含 page_number。
"""

import logging
from typing import List

from llama_index.core.schema import Document as LlamaDocument

from src.ingestion.parsers.base_parser import BaseParser, make_doc_id

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """PDF 解析器——基于 PyMuPDF，逐页提取文本。

    Usage::

        parser = PDFParser()
        docs = parser.parse("/path/to/paper.pdf")
        for d in docs:
            print(d.metadata["page_number"], d.text[:100])
    """

    @property
    def supported_format(self) -> str:
        return "pdf"

    def parse(self, source: str) -> List[LlamaDocument]:
        """解析 PDF 文件，每页生成一个 LlamaDocument。

        Args:
            source: PDF 文件路径。

        Returns:
            LlamaDocument 列表，每页一个文档。
        """
        import fitz  # PyMuPDF

        doc_id = self._make_doc_id(source)
        doc_name = self._source_name(source)
        documents: List[LlamaDocument] = []

        try:
            pdf = fitz.open(source)
        except Exception as e:
            logger.error("无法打开 PDF 文件 %s: %s", source, e)
            return []

        total_pages = pdf.page_count
        total_chars = 0

        for page_idx in range(total_pages):
            try:
                page = pdf[page_idx]
                text = page.get_text("text")  # type: ignore[arg-type]
                if not text or not text.strip():
                    logger.debug("PDF 第 %d 页为空，跳过", page_idx + 1)
                    continue

                total_chars += len(text)
                doc = LlamaDocument(
                    text=text.strip(),
                    metadata={
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "source": source,
                        "format": "pdf",
                        "page_number": page_idx + 1,  # 1-based
                        "total_pages": total_pages,
                    },
                )
                documents.append(doc)
            except Exception as e:
                logger.warning("PDF 第 %d 页提取失败: %s", page_idx + 1, e)

        pdf.close()
        logger.info(
            "PDF 解析完成: %s (%d 页, %d 非空页, %d 字符)",
            doc_name, total_pages, len(documents), total_chars,
        )
        return documents

    # ------------------------------------------------------------------
    # 可选：使用 LlamaIndex PDFReader 作为回退
    # ------------------------------------------------------------------

    @staticmethod
    def parse_via_llamaindex(source: str) -> List[LlamaDocument]:
        """使用 LlamaIndex PDFReader 作为备选解析方案。

        Args:
            source: PDF 文件路径。

        Returns:
            LlamaDocument 列表。
        """
        try:
            from llama_index.readers.file import PDFReader

            reader = PDFReader()
            docs = reader.load_data(file_path=source)  # type: ignore[arg-type]
            logger.info("LlamaIndex PDFReader 解析完成: %s (%d 页)", source, len(docs))
            return list(docs)
        except ImportError:
            logger.error("llama-index-readers-file 未安装")
            return []
