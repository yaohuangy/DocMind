"""测试 Tavily 联网搜索 —— 验证 MCPClientManager 全链路。

用法：venv/Scripts/python scripts/test_tavily_search.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.mcp_client import MCPClientManager  # noqa: E402


async def main():
    print("=" * 60)
    print("  Phase 1: Tavily 联网搜索验证")
    print("=" * 60)

    # 从 .env 自动加载 Tavily 配置
    manager = MCPClientManager()  # server_configs=None → 自动从 .env 读取

    if not manager._configs:
        print("\n❌ 未找到 TAVILY_API_KEY，请在 .env 中配置")
        return

    print(f"\n📡 连接 Tavily MCP Server...")
    print(f"   servers: {[c.name for c in manager._configs]}")

    await manager.connect_all()

    if not manager.connected_servers:
        print("\n❌ 连接失败！请检查网络和 npx 是否可用")
        await manager.close_all()
        return

    print(f"   ✅ 已连接: {manager.connected_servers}")

    # 列出工具
    tools = manager.list_tools()
    print(f"\n🔧 发现 {len(tools)} 个工具:")
    for t in tools:
        desc = t.description[:80].replace("\n", " ")
        print(f"   • {t.name}: {desc}...")

    # 搜索
    print("\n🔍 搜索: 2026年AI Agent最新进展")
    results = await manager.search_external("2026年AI Agent最新进展")

    if not results:
        print("   ⚠️ 搜索结果为空（可能是工具名不匹配，尝试手动调用...）")
        # 尝试直接调用 tavily_search
        for t in tools:
            if "search" in t.name.lower():
                print(f"\n   尝试直接调用: {t.name}")
                r = await manager.call_tool(t.name, {"query": "AI Agent 2026"})
                print(f"   success={r.success}, text_len={len(r.text)}")
                if r.success:
                    print(f"   text preview: {r.text[:300]}...")
    else:
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r.title}")
            print(f"    {r.snippet[:120]}...")
            if r.url:
                print(f"    🔗 {r.url}")

    # 关闭
    await manager.close_all()
    print(f"\n✅ 测试完成")


if __name__ == "__main__":
    asyncio.run(main())
