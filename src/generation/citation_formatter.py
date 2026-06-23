"""
引用格式化模块。

职责：
1. 从 LLM 生成的答案中解析引用标记（如 [1]、[Chunk 3]）
2. 验证引用索引与源列表的一致性
3. 为每个引用生成人类可读的位置描述（适配 8 种格式）
4. 输出前端可直接渲染的 SourceChunk 列表

适配的格式位置描述（见 spec §7.3）：
- PDF: "第3页"
- 网页: "段落: 架构概述"
- Word: "章节: 背景"
- Markdown: "## 背景 > ### 动机"
- TXT: "第15-22行"
- PPT: "第5页幻灯片"
- CSV/Excel: "第10行"
"""

from __future__ import annotations

import logging
import re

from src.core.vector_store import SearchResult
from src.engine.models import SourceChunk

logger = logging.getLogger(__name__)

# 匹配引用标记：[1], [2], [3] 等
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
# 匹配外部引用标记：[E1], [E2] 等
_EXTERNAL_CITATION_PATTERN = re.compile(r"\[E(\d+)\]")


class CitationFormatter:
    """引用格式化器。

    将 LLM 输出中的 [N] 标记与检索到的源文档分块关联，
    生成带有位置描述的 SourceChunk 列表。

    Usage::

        formatter = CitationFormatter()
        formatted_answer, sources = formatter.format(
            raw_answer="Transformer 由 Vaswani 提出 [1]，核心是自注意力 [2]。",
            search_results=retrieved_chunks,
        )
    """

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def format(
        self,
        raw_answer: str,
        search_results: list[SearchResult],
    ) -> tuple[str, list[SourceChunk]]:
        """格式化答案中的引用。

        解析 [N] 标记，验证有效性，生成 SourceChunk 列表。
        不修改答案文本（保留原始引用标记）。

        Args:
            raw_answer: LLM 生成的原始答案（含 [N] 引用标记）。
            search_results: 检索到的源文档分块列表。

        Returns:
            (formatted_answer, sources) 元组：
            - formatted_answer: 保留引用标记的答案（当前与 raw_answer 相同）。
            - sources: 按引用序号排列的 SourceChunk 列表（[1] 在 index 0）。
        """
        if not raw_answer or not search_results:
            return raw_answer or "", []

        # 解析答案中引用了哪些编号
        cited_indices = self._extract_cited_indices(raw_answer)

        # 为每个被引用的编号创建 SourceChunk
        sources: list[SourceChunk] = []
        for idx in cited_indices:
            # 编号从 1 开始，列表索引从 0 开始
            result_idx = idx - 1
            if 0 <= result_idx < len(search_results):
                sr = search_results[result_idx]
                source = self._build_source_chunk(sr, idx)
                sources.append(source)
            else:
                logger.warning("引用 [%d] 超出源列表范围 (共 %d 条)", idx, len(search_results))

        # 如果没有解析到引用，默认展示全部检索结果
        if not sources and search_results:
            sources = [
                self._build_source_chunk(sr, i + 1)
                for i, sr in enumerate(search_results)
            ]
            logger.info("未检测到引用标记，默认展示全部 %d 条结果", len(sources))

        return raw_answer, sources

    def format_with_remap(
        self,
        raw_answer: str,
        search_results: list[SearchResult],
    ) -> tuple[str, list[SourceChunk]]:
        """格式化并重新映射引用编号。

        当答案中的引用编号与最终展示顺序不一致时使用。
        例如模型引用了 [3], [7], [2]，重映射为 [1], [2], [3]。

        Args:
            raw_answer: LLM 生成的原始答案。
            search_results: 检索结果列表。

        Returns:
            (remapped_answer, remapped_sources) 元组。
        """
        cited_indices = sorted(set(self._extract_cited_indices(raw_answer)))

        if not cited_indices:
            return raw_answer, []

        # 构建旧编号 → 新编号的映射
        remap: dict[int, int] = {}
        new_sources: list[SourceChunk] = []
        new_idx = 1

        for old_idx in cited_indices:
            result_idx = old_idx - 1
            if 0 <= result_idx < len(search_results):
                remap[old_idx] = new_idx
                source = self._build_source_chunk(search_results[result_idx], new_idx)
                new_sources.append(source)
                new_idx += 1

        # 替换答案中的引用编号
        def _replace(match: re.Match) -> str:
            old_num = int(match.group(1))
            new_num = remap.get(old_num, old_num)
            return f"[{new_num}]"

        remapped_answer = _CITATION_PATTERN.sub(_replace, raw_answer)

        return remapped_answer, new_sources

    def format_with_external(
        self,
        raw_answer: str,
        search_results: list[SearchResult],
        external_results: list[dict] | None = None,
    ) -> dict:
        """格式化包含本地和外部来源的答案。

        解析 [N]（本地）和 [EN]（外部）引用标记。

        Args:
            raw_answer: LLM 生成的答案。
            search_results: 本地检索结果。
            external_results: 外部搜索结果列表，每项为
                {"title": str, "url": str, "snippet": str}。

        Returns:
            {
                "formatted_answer": str,
                "local_sources": [SourceChunk, ...],
                "external_sources": [dict, ...],
            }
        """
        if not raw_answer:
            return {
                "formatted_answer": "",
                "local_sources": [],
                "external_sources": [],
            }

        # 解析本地引用
        cited_indices = self._extract_cited_indices(raw_answer)
        local_sources: list[SourceChunk] = []
        for idx in cited_indices:
            result_idx = idx - 1
            if 0 <= result_idx < len(search_results):
                local_sources.append(
                    self._build_source_chunk(search_results[result_idx], idx)
                )

        # 解析外部引用
        external_sources: list[dict] = []
        if external_results:
            ext_indices = self._extract_external_indices(raw_answer)
            for ext_idx in ext_indices:
                result_idx = ext_idx - 1
                if 0 <= result_idx < len(external_results):
                    ext = external_results[result_idx]
                    external_sources.append({
                        "index": ext_idx,
                        "citation": f"[E{ext_idx}]",
                        "title": ext.get("title", ""),
                        "url": ext.get("url", ""),
                        "snippet": ext.get("snippet", "")[:300],
                    })

        return {
            "formatted_answer": raw_answer,
            "local_sources": local_sources,
            "external_sources": external_sources,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cited_indices(text: str) -> list[int]:
        """从答案中提取 [N] 引用编号（排除 [EN] 类外部引用）。

        Args:
            text: 含 [N] 标记的答案文本。

        Returns:
            去重后的编号列表，按升序排列。
        """
        # 先剔除所有 [E\d+] 标记，避免误匹配
        cleaned = _EXTERNAL_CITATION_PATTERN.sub("", text)
        matches = _CITATION_PATTERN.findall(cleaned)
        indices = sorted(set(int(m) for m in matches))
        return indices

    @staticmethod
    def _extract_external_indices(text: str) -> list[int]:
        """从答案中提取 [EN] 外部引用编号。

        Args:
            text: 含 [E1][E2] 标记的答案文本。

        Returns:
            去重后的编号列表，按升序排列。
        """
        matches = _EXTERNAL_CITATION_PATTERN.findall(text)
        indices = sorted(set(int(m) for m in matches))
        return indices

    def _build_source_chunk(
        self,
        sr: SearchResult,
        display_index: int,
    ) -> SourceChunk:
        """将 SearchResult 转为 SourceChunk（含位置描述）。

        Args:
            sr: 检索结果。
            display_index: 展示编号（1-based）。

        Returns:
            填充了 location_text 的 SourceChunk。
        """
        location = self._format_location(sr.metadata)

        return SourceChunk(
            chunk_id=sr.chunk_id,
            doc_name=sr.metadata.get("doc_name", "未知文档"),
            text=sr.text,
            score=sr.score,
            metadata=sr.metadata,
            location_text=location,
        )

    @staticmethod
    def _format_location(metadata: dict) -> str:
        """根据文档格式生成人类可读的位置描述。

        适配 8 种格式（见 spec §7.3）。

        Args:
            metadata: 分块元数据字典。

        Returns:
            位置描述字符串。
        """
        fmt = metadata.get("format", "").lower()
        parts: list[str] = []

        if fmt == "pdf":
            page = metadata.get("page_number")
            total = metadata.get("total_pages")
            if page:
                parts.append(f"第{page}页" + (f"/共{total}页" if total else ""))

        elif fmt == "web":
            section = metadata.get("section_title")
            url = metadata.get("url", "")
            if section:
                parts.append(f"段落: {section}")
            elif url:
                # 截短 URL
                short_url = url[:60] + "..." if len(url) > 60 else url
                parts.append(short_url)

        elif fmt == "docx":
            heading = metadata.get("heading")
            para_idx = metadata.get("paragraph_index")
            if heading:
                parts.append(f"章节: {heading}")
            if para_idx is not None:
                parts.append(f"第{para_idx + 1}段")

        elif fmt == "md":
            heading_path = metadata.get("heading_path")
            if heading_path:
                parts.append(heading_path)

        elif fmt == "txt":
            line_start = metadata.get("line_start")
            line_end = metadata.get("line_end")
            if line_start and line_end:
                parts.append(f"第{line_start}-{line_end}行")
            elif line_start:
                parts.append(f"第{line_start}行起")

        elif fmt == "pptx":
            slide = metadata.get("slide_number")
            title = metadata.get("slide_title")
            total = metadata.get("total_slides")
            if slide:
                parts.append(f"第{slide}页幻灯片" + (f"/共{total}页" if total else ""))
            if title and title not in "".join(parts):
                parts.append(f"标题: {title}")

        elif fmt in ("csv", "xlsx"):
            row = metadata.get("row_start")
            sheet = metadata.get("sheet_name")
            if row:
                parts.append(f"第{row}行")
            if sheet:
                parts.append(f"工作表: {sheet}")

        # 兜底
        if not parts:
            parts.append(metadata.get("source", "未知位置"))

        return ", ".join(parts)
