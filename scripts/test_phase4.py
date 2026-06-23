"""Phase 4 验证 —— MCP Server 新增工具。

用法：venv/Scripts/python scripts/test_phase4.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()


def test_get_available_tools():
    """验证 get_available_tools。"""
    print("=" * 60)
    print("  Phase 4.1: get_available_tools()")
    print("=" * 60)

    from src.mcp_server import get_available_tools

    result = get_available_tools()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def test_search_with_web():
    """验证 search_with_web。"""
    print("\n" + "=" * 60)
    print("  Phase 4.2: search_with_web()")
    print("=" * 60)

    from src.mcp_server import search_with_web

    result = search_with_web("2026年AI Agent趋势", top_k=3)
    print(f"answer 长度: {len(result['answer'])} 字符")
    print(f"merged_count: {result['merged_count']}")
    print(f"local_sources: {len(result['local_sources'])} 条")
    print(f"external_sources: {len(result['external_sources'])} 条")
    print(f"route_decision: {result['route_decision']}")

    if result["external_sources"]:
        print(f"\n外部来源:")
        for e in result["external_sources"][:3]:
            print(f"  [{e.get('citation','?')}] {e.get('title','?')[:80]}")

    print(f"\n答案预览:")
    print(result["answer"][:300])
    if len(result["answer"]) > 300:
        print("...")


if __name__ == "__main__":
    test_get_available_tools()
    test_search_with_web()

    print("\n" + "=" * 60)
    print("  ✅ Phase 4 验证完成")
    print("=" * 60)
