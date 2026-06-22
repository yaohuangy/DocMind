"""DocMind MCP Server —— 将 RAG 能力以 MCP 协议暴露给外部 AI 客户端。

支持的 MCP 客户端：Claude Desktop、Cursor、VS Code Copilot、Continue 等。

启动方式（stdio 模式，客户端自动管理进程）：:

    # Claude Desktop 配置示例 (~/Library/Application Support/Claude/claude_desktop_config.json):
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

    # Cursor 配置示例 (.cursor/mcp.json):
    {
      "mcpServers": {
        "docmind": {
          "command": "python",
          "args": ["-m", "src.mcp_server"],
          "cwd": "/path/to/docmind"
        }
      }
    }

提供三个工具：
- ``search_documents``：语义检索文档分块
- ``ask_knowledge_base``：检索 + LLM 生成答案
"""

from __future__ import annotations

import sys
from pathlib import Path

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
def search_documents(query: str, top_k: int = 5) -> str:
    """从 DocMind 知识库中语义检索相关文档分块。

    返回最相关的文档片段及其来源信息（文档名、页码等）。
    用于需要查找文档中某个事实、概念或细节的场景。

    Args:
        query: 搜索查询，用自然语言描述你要找的内容。
        top_k: 返回结果数量，默认 5。

    Returns:
        格式化的检索结果，包含分块文本和来源引用。
    """
    engine = _get_engine()
    sources = engine.retrieve(query, method="direct", top_k=top_k)
    if not sources:
        return "未在知识库中找到相关文档。请尝试换个关键词，或确认已上传相关文档。"

    lines = []
    for i, src in enumerate(sources, 1):
        loc = src.location_text or "未知位置"
        lines.append(
            f"[{i}] 📄 {src.doc_name} | 📍 {loc} | 相似度: {src.score:.3f}\n"
            f"{src.text[:800]}"
        )
    return "\n\n---\n\n".join(lines)


@mcp.tool()
def ask_knowledge_base(question: str) -> str:
    """向 DocMind 知识库提问，返回基于文档内容生成的答案。

    先从文档库检索相关内容，再调用 LLM 生成带引用的答案。
    适合需要综合多个文档片段的提问场景。

    Args:
        question: 你的问题，越具体越好。

    Returns:
        基于知识库文档生成的答案，引用用 [N] 标记对应文档分块。
    """
    engine = _get_engine()
    sources = engine.retrieve(question, method="direct", top_k=5)
    if not sources:
        return engine.generate(question, [], method="direct")

    answer = engine.generate(question, sources, method="direct")
    return answer


@mcp.tool()
def list_knowledge_base() -> str:
    """列出 DocMind 知识库中已加载的文档清单。

    Returns:
        文档列表，含名称、格式、分块数、字符数。
    """
    engine = _get_engine()
    try:
        docs = engine.list_documents()
    except Exception:
        return "获取文档列表失败，请检查 ChromaDB 是否运行正常。"

    if not docs:
        return "知识库为空，请先通过 DocMind 上传文档。"

    lines = [f"知识库共 {len(docs)} 个文档：\n"]
    for d in docs:
        lines.append(
            f"- 📄 {d['name']} | {d.get('format', '?')} | "
            f"{d.get('num_chunks', 0)} chunks | {d.get('char_count', 0):,} 字符"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
