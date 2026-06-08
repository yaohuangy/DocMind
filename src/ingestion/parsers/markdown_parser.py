"""
Markdown (.md) 文档解析器。

按章节（以 # 为界）分割内容，保留标题层级路径信息。
每个顶级标题（# ) 或次级标题（## ）段落作为一个 LlamaDocument。
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple

from llama_index.core.schema import Document as LlamaDocument

from src.ingestion.parsers.base_parser import BaseParser, make_doc_id

logger = logging.getLogger(__name__)


class MarkdownParser(BaseParser):
    """Markdown 解析器——按标题层级分割，保留标题路径。

    Usage::

        parser = MarkdownParser()
        docs = parser.parse("/path/to/doc.md")
        for d in docs:
            print(d.metadata["heading_path"], d.text[:100])
    """

    @property
    def supported_format(self) -> str:
        return "md"

    def parse(self, source: str) -> List[LlamaDocument]:
        """解析 Markdown 文件。

        Args:
            source: .md 文件路径。

        Returns:
            LlamaDocument 列表，按标题分节。
        """
        doc_id = make_doc_id(source)
        doc_name = self._source_name(source)

        try:
            text = Path(source).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # 尝试 chardet
            try:
                import chardet
                raw = Path(source).read_bytes()
                encoding = chardet.detect(raw)["encoding"] or "utf-8"
                text = raw.decode(encoding, errors="replace")
            except ImportError:
                text = Path(source).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error("无法读取 Markdown 文件 %s: %s", source, e)
            return []

        if not text.strip():
            logger.warning("Markdown 文件为空: %s", source)
            return []

        # 按标题分割
        sections = self._split_by_headings(text)
        documents: List[LlamaDocument] = []

        for heading_path, content in sections:
            if not content.strip():
                continue

            doc = LlamaDocument(
                text=content.strip(),
                metadata={
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "source": source,
                    "format": "md",
                    "heading_path": heading_path or "",
                    "char_count": len(content),
                },
            )
            documents.append(doc)

        logger.info(
            "Markdown 解析完成: %s (%d 节, %d 字符)",
            doc_name, len(documents), sum(len(d.text) for d in documents),
        )
        return documents

    # ------------------------------------------------------------------
    # 内部：标题分割
    # ------------------------------------------------------------------

    @staticmethod
    def _split_by_headings(text: str) -> List[Tuple[str, str]]:
        """按 Markdown 标题（# ~ ######）分割文本。

        构建每个段落对应的 heading_path，例如：
        "## 背景 > ### 动机" 表示该段落位于 背景 下的 动机 小节。

        Args:
            text: Markdown 原始文本。

        Returns:
            (heading_path, content) 元组列表。
        """
        lines = text.split("\n")
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        # 构建标题层级栈: [(level, title), ...]
        heading_stack: List[Tuple[int, str]] = []
        sections: List[Tuple[str, str]] = []
        current_lines: List[str] = []

        for line in lines:
            match = heading_pattern.match(line)
            if match:
                # 保存前一个段落
                if current_lines:
                    path = " > ".join(title for _, title in heading_stack)
                    sections.append((path, "\n".join(current_lines)))
                    current_lines = []

                level = len(match.group(1))
                title = match.group(2).strip()

                # 弹出层级 >= 当前级别的旧标题
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
            else:
                current_lines.append(line)

        # 最后一个段落
        if current_lines:
            path = " > ".join(title for _, title in heading_stack)
            sections.append((path, "\n".join(current_lines)))

        return sections
