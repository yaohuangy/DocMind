"""DocMind MCP Server —— 简化版，放在项目根目录直接运行。

启动方式：python mcp_server.py

这是 src/mcp_server.py 的镜像副本，供 Claude Code 的 .mcp.json 直接引用。
完整版（含注释、配置示例、结构化输出）见 src/mcp_server.py。
"""
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

from src.engine.qa_engine import QAEngine

mcp = FastMCP("DocMind")
_engine: QAEngine | None = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = QAEngine()
        _engine.set_user_id(os.getenv("DOCMIND_USER_ID", "Yao"))
    return _engine


def _resolve_method(engine: QAEngine, question: str, method: str) -> str:
    if method == "auto":
        engine.set_dynamic_routing(True)
        return "direct"
    engine.set_dynamic_routing(False)
    return method


@mcp.tool()
def search_documents(query: str, top_k: int = 5, method: str = "auto") -> dict[str, Any]:
    """语义检索文档分块。返回结构化 JSON 含分数与来源位置。

    Args:
        query: 自然语言搜索查询。
        top_k: 返回数量，默认 5。
        method: 检索策略。"auto"自动路由 / "direct" / "mqe" / "hyde" / "mqe+hyde"
    """
    engine = _get_engine()
    resolved = _resolve_method(engine, query, method)
    sources = engine.retrieve(query, method=resolved, top_k=top_k)
    if not sources:
        return {"results": [], "count": 0, "hint": "未找到相关文档。"}
    results = [{
        "rank": i, "doc": s.doc_name, "page": s.location_text or "?",
        "score": round(s.score, 4), "text": s.text[:800],
    } for i, s in enumerate(sources, 1)]
    return {"results": results, "count": len(results), "method_used": resolved}


@mcp.tool()
async def ask_knowledge_base(question: str, method: str = "auto", include_external: bool = False):
    """向知识库提问，流式返回带引用的答案。

    Args:
        question: 你的问题。
        method: 检索策略，默认 "auto"。
        include_external: 是否联网搜索，默认 False。
    """
    engine = _get_engine()
    if include_external and _has_external():
        result = engine.generate_with_external(question, method=method, top_k=5)
        yield result["answer"]
        return
    resolved = _resolve_method(engine, question, method)
    sources = engine.retrieve(question, method=resolved, top_k=5)
    if not sources:
        yield engine.generate(question, [], method=resolved)
        return
    for token in engine.generate_stream(question, sources, method=resolved):
        yield token


@mcp.tool()
def list_knowledge_base() -> dict[str, Any]:
    """列出知识库文档清单，返回结构化 JSON。"""
    engine = _get_engine()
    try:
        docs = engine.list_documents()
    except Exception:
        return {"count": 0, "documents": [], "hint": "获取失败，请检查 ChromaDB。"}
    if not docs:
        return {"count": 0, "documents": [], "hint": "知识库为空。"}
    documents = [{"name": d.get("name","?"), "format": d.get("format","?"),
                  "chunks": d.get("num_chunks",0), "chars": d.get("char_count",0)} for d in docs]
    return {"count": len(documents), "documents": documents}


_external_checked = False
_external_available = False


def _has_external():
    global _external_checked, _external_available
    if not _external_checked:
        try:
            _external_available = _get_engine().has_external_search()
        except Exception:
            _external_available = False
        _external_checked = True
    return _external_available


@mcp.tool()
def search_with_web(query: str, top_k: int = 5) -> dict[str, Any]:
    """同时搜索本地知识库和互联网，返回融合答案（[N]本地 + [E]外部引用）。"""
    engine = _get_engine()
    if not _has_external():
        sources = engine.retrieve(query, method="direct", top_k=top_k)
        answer = engine.generate(query, sources, method="direct")
        return {"answer": answer, "local_sources": [], "external_sources": [],
                "route_decision": {"need_external": False, "reason": "外部搜索未配置"},
                "merged_count": len(sources)}
    result = engine.generate_with_external(query, method="direct", top_k=top_k)
    return {
        "answer": result["answer"],
        "local_sources": result.get("local_sources", []),
        "external_sources": result.get("external_sources", []),
        "route_decision": {"need_external": result["route_decision"].need_external,
                           "reason": result["route_decision"].reason},
        "merged_count": len(result.get("merged", [])),
    }


@mcp.tool()
def get_available_tools() -> dict[str, Any]:
    """列出 DocMind 当前可用能力和工具列表。"""
    caps = ["local_search", "qa_generation", "document_listing"]
    if _has_external():
        caps.append("external_search")
    return {
        "capabilities": caps,
        "external_available": _has_external(),
        "tools": [
            {"name": "search_documents", "description": "语义检索本地文档分块", "method": "local"},
            {"name": "ask_knowledge_base", "description": "向知识库提问，流式生成答案。支持 include_external=True", "method": "local+generation"},
            {"name": "search_with_web", "description": "本地+联网融合搜索", "method": "local+external+generation", "available": _has_external()},
            {"name": "list_knowledge_base", "description": "列出知识库文档清单", "method": "local"},
            {"name": "get_available_tools", "description": "列出可用能力（本工具）", "method": "meta"},
        ],
    }


if __name__ == "__main__":
    mcp.run()
