"""DocMind MCP Server —— 简化版，放在项目根目录直接运行。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

from src.engine.qa_engine import QAEngine

mcp = FastMCP("DocMind")
_engine: QAEngine | None = None

def _get_engine():
    global _engine
    if _engine is None:
        import os
        _engine = QAEngine()
        _engine.set_user_id(os.getenv("DOCMIND_USER_ID", "Yao"))
    return _engine

@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> str:
    engine = _get_engine()
    sources = engine.retrieve(query, method="direct", top_k=top_k)
    if not sources:
        return "未在知识库中找到相关文档。"
    lines = []
    for i, src in enumerate(sources, 1):
        lines.append(f"[{i}] {src.doc_name} | {src.location_text or '?'} | {src.score:.3f}\n{src.text[:800]}")
    return "\n\n---\n\n".join(lines)

@mcp.tool()
def ask_knowledge_base(question: str) -> str:
    engine = _get_engine()
    sources = engine.retrieve(question, method="direct", top_k=5)
    return engine.generate(question, sources, method="direct")

@mcp.tool()
def list_knowledge_base() -> str:
    engine = _get_engine()
    docs = engine.list_documents()
    if not docs:
        return "知识库为空。"
    lines = [f"共 {len(docs)} 个文档："]
    for d in docs:
        lines.append(f"- {d['name']} | {d.get('format','?')} | {d.get('num_chunks',0)} chunks")
    return "\n".join(lines)

if __name__ == "__main__":
    mcp.run()
