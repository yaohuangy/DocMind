"""本地 + 外部结果融合模块。

提供 merge_results() 函数，将 QAEngine 检索到的本地文档分块
与 MCPClientManager 获取的外部搜索结果融合为一个统一列表，
支持去重、交错排列和权重控制。

使用示例::

    from src.mcp_client.fusion import merge_results, MergedResult

    merged = merge_results(local_chunks, external_results)
    for m in merged:
        print(f"{m.citation} [{m.source_type}] {m.title}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.engine.models import SourceChunk
from src.mcp_client.models import ExternalResult

logger = logging.getLogger(__name__)


@dataclass
class MergedResult:
    """融合后的统一结果条目。"""

    source_type: str  # "local" 或 "external"
    citation: str     # 引用标记，如 "[1]"、"[E1]"
    title: str        # 来源标题（文档名 或 搜索标题）
    text: str         # 内容摘要
    url: str = ""     # 外部链接（仅 external）
    score: float = 0.0
    chunk: SourceChunk | None = None       # 本地结果
    external: ExternalResult | None = None # 外部结果


def merge_results(
    local: list[SourceChunk],
    external: list[ExternalResult],
    local_weight: float = 0.6,
    external_weight: float = 0.4,
    max_total: int = 8,
) -> list[MergedResult]:
    """融合本地检索与外部搜索结果。

    策略：
    1. 本地保留前 max_local = max_total * local_weight 条（按分数）
    2. 外部保留前 max_external = max_total * external_weight 条
    3. 外部 vs 本地做 bigram Jaccard 去重（复用已有算法）
    4. 交错排列：L1 → E1 → L2 → E2 → ...（避免外部堆在一起被 LLM 忽略）

    Args:
        local: 本地检索结果（SourceChunk 列表，已按分数降序）。
        external: 外部搜索结果（ExternalResult 列表）。
        local_weight: 本地结果权重占比，默认 0.6。
        external_weight: 外部结果权重占比，默认 0.4。
        max_total: 融合后最大条数，默认 8。

    Returns:
        MergedResult 列表，citation 已分配好（[1][2]...[E1][E2]...）。
    """
    max_local = max(1, round(max_total * local_weight))
    max_external = max(1, round(max_total * external_weight))

    # 截断
    local_top = local[:max_local]
    external_top = _dedup_external(external, local_top)[:max_external]

    # 交错排列
    merged: list[MergedResult] = []
    local_idx = 0
    ext_idx = 0
    local_cite = 1
    ext_cite = 1

    for _ in range(max(len(local_top), len(external_top))):
        if local_idx < len(local_top):
            src = local_top[local_idx]
            merged.append(MergedResult(
                source_type="local",
                citation=f"[{local_cite}]",
                title=src.doc_name,
                text=src.text[:600],
                score=src.score,
                chunk=src,
            ))
            local_idx += 1
            local_cite += 1

        if ext_idx < len(external_top):
            ext = external_top[ext_idx]
            merged.append(MergedResult(
                source_type="external",
                citation=f"[E{ext_cite}]",
                title=ext.title or ext.source,
                text=ext.snippet[:600],
                url=ext.url,
                external=ext,
            ))
            ext_idx += 1
            ext_cite += 1

    logger.info(
        "融合完成: local=%d/%d, external=%d/%d → total=%d",
        len(local_top), len(local),
        len(external_top), len(external),
        len(merged),
    )

    return merged


def build_context(merged: list[MergedResult]) -> str:
    """将融合结果构建为 LLM context 字符串。

    本地结果用 [N] 引用，外部结果用 [E1][E2] 引用。

    Args:
        merged: merge_results() 的输出。

    Returns:
        格式化的 context 文本，供注入 System Prompt。
    """
    parts: list[str] = []

    local_items = [m for m in merged if m.source_type == "local"]
    external_items = [m for m in merged if m.source_type == "external"]

    if local_items:
        parts.append("## 📄 本地文档片段\n")
        for m in local_items:
            parts.append(f"**{m.citation}** — {m.title}")
            parts.append(f"{m.text}\n")

    if external_items:
        parts.append("## 🌐 网络搜索结果\n")
        for m in external_items:
            parts.append(f"**{m.citation}** — {m.title}")
            if m.url:
                parts.append(f"来源: {m.url}")
            parts.append(f"{m.text}\n")

    return "\n".join(parts)


# ------------------------------------------------------------------
# 去重辅助
# ------------------------------------------------------------------


def _bigram_jaccard(text1: str, text2: str) -> float:
    """计算两个文本的 bigram Jaccard 相似度。"""
    def bigrams(s: str) -> set[str]:
        # 取每两个相邻字符
        return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}

    b1 = bigrams(text1)
    b2 = bigrams(text2)
    if not b1 or not b2:
        return 0.0

    intersection = len(b1 & b2)
    union = len(b1 | b2)
    return intersection / union if union > 0 else 0.0


def _dedup_external(
    external: list[ExternalResult],
    local: list[SourceChunk],
    threshold: float = 0.5,
) -> list[ExternalResult]:
    """过滤与本地结果高度重复的外部结果。"""
    kept: list[ExternalResult] = []
    for ext in external:
        is_dup = False
        ext_text = f"{ext.title} {ext.snippet}"
        for loc in local:
            loc_text = f"{loc.doc_name} {loc.text[:300]}"
            if _bigram_jaccard(ext_text, loc_text) >= threshold:
                is_dup = True
                logger.debug("去重外部结果 [%s] vs 本地 [%s]", ext.title, loc.doc_name)
                break
        if not is_dup:
            kept.append(ext)
    return kept
