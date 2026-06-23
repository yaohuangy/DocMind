"""DocMind MCP Server —— 将 RAG 能力以 MCP 协议暴露给外部 AI 客户端。

支持的 MCP 客户端：Claude Desktop、Cursor、VS Code Copilot、Continue、Claude Code 等。

提供五个工具：

- ``search_documents``：语义检索文档分块，返回结构化 JSON（含分数、来源页）。
- ``ask_knowledge_base``：检索 + 流式生成答案，可选联网搜索（include_external=True）。
- ``search_with_web``：同时搜索本地知识库和互联网，返回融合答案。
- ``list_knowledge_base``：列出知识库文档清单。
- ``get_available_tools``：列出 DocMind 当前可用能力。

启动方式（stdio 模式，客户端自动管理进程）：:

    # Claude Desktop 配置示例
    {
      "mcpServers": {
        "docmind": {
          "command": "python",
          "args": ["-m", "src.mcp_server"],
          "cwd": "/path/to/docmind",
          "env": { "DOCMIND_USER_ID": "default" }
        }
      }
    }

    # Cursor 配置示例 (.cursor/mcp.json)
    {
      "mcpServers": {
        "docmind": {
          "command": "python",
          "args": ["-m", "src.mcp_server"],
          "cwd": "/path/to/docmind"
        }
      }
    }

    # VS Code Copilot (.vscode/mcp.json)
    {
      "servers": {
        "docmind": {
          "type": "stdio",
          "command": "${workspaceFolder}/venv/Scripts/python",
          "args": ["-m", "src.mcp_server"],
          "env": { "DOCMIND_USER_ID": "Yao" }
        }
      }
    }

    # Claude Code（项目根目录 .mcp.json，自动识别）
    {
      "mcpServers": {
        "docmind": {
          "type": "stdio",
          "command": "venv/Scripts/python",
          "args": ["mcp_server.py"],
          "env": { "DOCMIND_USER_ID": "Yao" }
        }
      }
    }
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 确保项目根在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from src.engine.qa_engine import QAEngine

# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

mcp = FastMCP("DocMind")
_engine: QAEngine | None = None


def _get_engine() -> QAEngine:
    """懒加载 QAEngine 单例。"""
    global _engine
    if _engine is None:
        import os

        _engine = QAEngine()
        user_id = os.getenv("DOCMIND_USER_ID", "default")
        _engine.set_user_id(user_id)
    return _engine


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


@mcp.tool()
def search_documents(
    query: str,
    top_k: int = 5,
    method: str = "auto",
) -> dict[str, Any]:
    """语义检索知识库文档分块。当你需要查找文档中的具体事实、概念、
    代码片段或细节时使用此工具。返回结构化 JSON 含分数与来源位置。

    使用建议：query 越具体越好。例如"第三章讨论的 Transformer 架构
    的三个变体"而非模糊的"Transformer"。需要更多上下文时调大 top_k。

    区别于 ask_knowledge_base：本工具只检索不生成，适合你自行判断
    和综合多个分块时使用；ask_knowledge_base 直接生成完整答案。

    Args:
        query: 自然语言搜索查询，越具体越好。
        top_k: 返回数量，1-20，默认 5。需要更多上下文时调大。
        method: 检索策略。默认 "auto" 自动路由（简单→Direct 省 Token，
               复杂→HyDE 高精度）。可显式指定：
               "direct" — 最快，直接向量检索
               "mqe"    — 多查询扩展，提高召回率
               "hyde"   — 假设文档嵌入，改善精度
               "mqe+hyde" — 两者并行融合，最全面但最慢

    Returns:
        {"results": [{"rank": 1, "doc": "...", "page": "...",
                      "score": 0.462, "text": "..."}],
         "count": 5, "method_used": "direct"}
        知识库为空时返回 {"results": [], "count": 0, "hint": "..."}
    """
    engine = _get_engine()
    resolved_method = _resolve_method(engine, query, method)
    sources = engine.retrieve(query, method=resolved_method, top_k=top_k)

    if not sources:
        return {
            "results": [],
            "count": 0,
            "hint": "知识库中未找到相关文档。请尝试换个关键词，或确认已上传相关文档。",
            "method_used": resolved_method,
        }

    results = []
    for i, src in enumerate(sources, 1):
        results.append({
            "rank": i,
            "doc": src.doc_name,
            "page": src.location_text or "未知位置",
            "score": round(src.score, 4),
            "text": src.text[:800],
        })

    return {
        "results": results,
        "count": len(results),
        "method_used": resolved_method,
    }


@mcp.tool()
async def ask_knowledge_base(
    question: str,
    method: str = "auto",
    include_external: bool = False,
):
    """向知识库提问，流式返回基于文档内容生成的答案。先检索相关分块，
    再调用 LLM 生成带引用（[N] 标记）的综合回答。适合需要综合分析
    多个文档片段的场景。

    区别于 search_documents：本工具直接返回整理好的答案，而非原始分块。
    区别于 search_with_web：本工具默认仅本地检索；设置 include_external=True
    时可联网搜索并融合外部结果。

    Args:
        question: 你的问题，越具体越好。
        method: 检索策略，默认 "auto" 自动路由。可选值同 search_documents。
        include_external: 是否也搜索互联网，默认 False。
                          True 时自动融合本地+外部结果，引用格式为 [N] 本地 + [E] 外部。

    Yields:
        逐 token 流式输出生成的答案文本，引用用 [N] / [E1][E2] 标记。
    """
    engine = _get_engine()

    # 联网搜索模式：走 generate_with_external，非流式
    if include_external and _has_external():
        result = engine.generate_with_external(question, method=method, top_k=5)
        yield result["answer"]
        return

    # 纯本地模式（流式）
    resolved_method = _resolve_method(engine, question, method)
    sources = engine.retrieve(question, method=resolved_method, top_k=5)

    if not sources:
        answer = engine.generate(question, [], method=resolved_method)
        yield answer
        return

    for token in engine.generate_stream(question, sources, method=resolved_method):
        yield token


@mcp.tool()
def list_knowledge_base() -> dict[str, Any]:
    """列出知识库中已加载的文档清单。当你需要了解有哪些文档可用、
    文档数量、格式分布时使用。通常在 search_documents 或
    ask_knowledge_base 之前调用，以便了解可检索的范围。

    Returns:
        {"count": 2, "documents": [{"name": "xxx.pdf", "format": "pdf",
         "chunks": 756, "chars": 123456}, ...]}
        知识库为空时返回 {"count": 0, "documents": [], "hint": "..."}
    """
    engine = _get_engine()
    try:
        docs = engine.list_documents()
    except Exception:
        return {
            "count": 0,
            "documents": [],
            "hint": "获取文档列表失败，请检查 ChromaDB 是否运行正常。",
        }

    if not docs:
        return {
            "count": 0,
            "documents": [],
            "hint": "知识库为空，请先通过 DocMind 上传文档。",
        }

    documents = []
    for d in docs:
        documents.append({
            "name": d.get("name", "?"),
            "format": d.get("format", "?"),
            "chunks": d.get("num_chunks", 0),
            "chars": d.get("char_count", 0),
        })

    return {
        "count": len(documents),
        "documents": documents,
    }


@mcp.tool()
def search_with_web(query: str, top_k: int = 5) -> dict[str, Any]:
    """同时搜索本地知识库和互联网，返回基于两类来源生成的融合答案。

    当用户的问题涉及最新进展、行业趋势、实时事件等时效性内容时，
    优先使用此工具。内部自动判断是否需要联网，不需要时退化为纯本地检索。

    区别于 search_documents：本工具直接返回整理好的答案；
    search_documents 返回原始分块供你自行分析。

    Args:
        query: 搜索查询，越具体越好。
        top_k: 本地检索返回数，默认 5。

    Returns:
        {
            "answer": "融合答案文本（含 [N] 和 [E1][E2] 引用标记）",
            "local_sources": [{"doc": "...", "page": "...", "text": "..."}],
            "external_sources": [{"citation": "[E1]", "title": "...", "url": "..."}],
            "route_decision": {"need_external": true, "reason": "..."},
            "merged_count": 8
        }
    """
    engine = _get_engine()

    if not _has_external():
        # 无外部搜索能力，回退纯本地
        sources = engine.retrieve(query, method="direct", top_k=top_k)
        answer = engine.generate(query, sources, method="direct")
        return {
            "answer": answer,
            "local_sources": [
                {"doc": s.doc_name, "page": s.location_text or "?", "text": s.text[:300]}
                for s in (sources or [])
            ],
            "external_sources": [],
            "route_decision": {"need_external": False, "reason": "外部搜索未配置"},
            "merged_count": len(sources),
        }

    result = engine.generate_with_external(query, method="direct", top_k=top_k)
    return {
        "answer": result["answer"],
        "local_sources": [
            {"doc": s.doc_name if hasattr(s, 'doc_name') else s.get('doc', '?'),
             "page": s.location_text if hasattr(s, 'location_text') else s.get('page', '?'),
             "text": (s.text if hasattr(s, 'text') else s.get('text', ''))[:300]}
            for s in result.get("local_sources", [])
        ],
        "external_sources": result.get("external_sources", []),
        "route_decision": {
            "need_external": result["route_decision"].need_external,
            "reason": result["route_decision"].reason,
        },
        "merged_count": len(result.get("merged", [])),
    }


@mcp.tool()
def get_available_tools() -> dict[str, Any]:
    """列出 DocMind 当前所有可用能力，包括本地检索和可选的外部搜索。

    当你刚连接 DocMind、不确定它能做什么时调用此工具。
    返回的工具列表和描述可帮助你决定用哪个工具来满足用户需求。

    Returns:
        {
            "capabilities": ["local_search", "external_search", ...],
            "tools": [{"name": "search_documents", "description": "..."}, ...],
            "external_available": true/false
        }
    """
    capabilities = ["local_search", "qa_generation", "document_listing"]
    if _has_external():
        capabilities.append("external_search")

    tools = [
        {
            "name": "search_documents",
            "description": "语义检索本地文档分块，返回结构化结果（分数+来源页）。适合查找事实/概念/细节。",
            "method": "local",
        },
        {
            "name": "ask_knowledge_base",
            "description": "向知识库提问，流式返回生成答案（带 [N] 引用）。支持 include_external=True 开启联网搜索。",
            "method": "local+generation",
        },
        {
            "name": "search_with_web",
            "description": "同时搜索本地文档和互联网，返回融合答案（[N] 本地 + [E] 外部引用）。适合时效性问题。",
            "method": "local+external+generation" if _has_external() else "local+generation",
            "available": _has_external(),
        },
        {
            "name": "list_knowledge_base",
            "description": "列出知识库文档清单（名称/格式/分块数/字符数）。",
            "method": "local",
        },
        {
            "name": "get_available_tools",
            "description": "列出 DocMind 当前可用能力和工具列表（本工具）。",
            "method": "meta",
        },
    ]

    return {
        "capabilities": capabilities,
        "tools": tools,
        "external_available": _has_external(),
    }


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

_external_checked: bool = False
_external_available: bool = False


def _has_external() -> bool:
    """检查外部搜索是否可用（结果缓存）。"""
    global _external_checked, _external_available
    if not _external_checked:
        _external_available = _get_engine().has_external_search()
        _external_checked = True
    return _external_available


def _resolve_method(engine: QAEngine, question: str, method: str) -> str:
    """解析检索方法。

    - "auto" 模式：启用 QAEngine 已有的动态路由（classify_complexity），
      简单问题→Direct（零额外 Token），复杂问题→HyDE（高精度）。
    - 显式指定：直接透传给 engine.retrieve()。
    """
    if method == "auto":
        engine.set_dynamic_routing(True)
        return "direct"  # 引擎会在内部路由到 hyde 如果需要
    engine.set_dynamic_routing(False)
    return method


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
