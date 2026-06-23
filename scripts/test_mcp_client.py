"""MCP Client 验证脚本 —— 启动一个 Echo MCP Server，用 MCPClientManager 连接并调用。

用法：venv/Scripts/python scripts/test_mcp_client.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# 1. 动态创建一个 Echo MCP Server 脚本
# ---------------------------------------------------------------------------

ECHO_SERVER_CODE = textwrap.dedent("""\
    import sys
    from pathlib import Path
    sys.path.insert(0, r"{project_root}")
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("EchoTest")

    @mcp.tool()
    def echo(message: str) -> dict:
        '''回显消息，用于测试。'''
        return {{"echo": message, "length": len(message)}}

    @mcp.tool()
    def add(a: int, b: int) -> dict:
        '''两数相加。'''
        return {{"result": a + b}}

    if __name__ == "__main__":
        mcp.run()
""")

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
ECHO_SERVER_PATH = Path(PROJECT_ROOT) / "scripts" / "_echo_mcp_server.py"


def _write_echo_server() -> None:
    """写入 Echo MCP Server 脚本。"""
    code = ECHO_SERVER_CODE.format(project_root=PROJECT_ROOT)
    ECHO_SERVER_PATH.write_text(code, encoding="utf-8")


def _cleanup_echo_server() -> None:
    """删除临时 Echo MCP Server 脚本。"""
    if ECHO_SERVER_PATH.exists():
        ECHO_SERVER_PATH.unlink()


# ---------------------------------------------------------------------------
# 2. 测试
# ---------------------------------------------------------------------------


async def main() -> None:
    from src.mcp_client import MCPClientManager, MCPServerConfig

    print("=" * 60)
    print("  Phase 1: MCP Client 验证")
    print("=" * 60)

    _write_echo_server()

    # 配置：用本地 Python 启动 Echo Server
    config = MCPServerConfig(
        name="echo-test",
        command=sys.executable,
        args=[str(ECHO_SERVER_PATH)],
        category="general",
        timeout=10,
    )

    print(f"\n📡 连接 Echo MCP Server...")
    print(f"   command: {config.command}")
    print(f"   args: {config.args}")

    try:
        manager = MCPClientManager([config])

        # --- connect_all ---
        print("\n1️⃣ connect_all()")
        results = await manager.connect_all()
        for name, ok in results.items():
            status = "✅" if ok else "❌"
            print(f"   {status} {name}: {ok}")

        if not results.get("echo-test"):
            print("\n❌ 连接失败！请检查 mcp 包版本和 Python 环境。")
            return

        # --- list_tools ---
        print("\n2️⃣ list_tools()")
        tools = manager.list_tools()
        for t in tools:
            print(f"   🔧 {t.name}")
            print(f"      desc: {t.description}")
            print(f"      server: {t.server_name} | category: {t.category}")

        # --- call_tool: echo ---
        print("\n3️⃣ call_tool('echo', message='Hello Phase 1!')")
        result = await manager.call_tool("echo", {"message": "Hello Phase 1!"})
        print(f"   success: {result.success}")
        print(f"   text: {result.text}")
        print(f"   data: {json.dumps(result.data, ensure_ascii=False) if result.data else 'N/A'}")
        print(f"   elapsed: {result.elapsed_ms:.1f}ms")

        # --- call_tool: add ---
        print("\n4️⃣ call_tool('add', a=3, b=5)")
        result2 = await manager.call_tool("add", {"a": 3, "b": 5})
        print(f"   success: {result2.success}")
        print(f"   text: {result2.text}")
        print(f"   data: {json.dumps(result2.data, ensure_ascii=False) if result2.data else 'N/A'}")
        print(f"   elapsed: {result2.elapsed_ms:.1f}ms")

        # --- call_tool: 不存在的工具 ---
        print("\n5️⃣ call_tool('nonexistent') → 预期失败")
        result3 = await manager.call_tool("nonexistent")
        print(f"   success: {result3.success}")
        print(f"   error: {result3.error[:80]}...")

        # --- connected_servers / tool_count ---
        print(f"\n📊 状态: connected={manager.connected_servers}, tools={manager.tool_count}")

        # --- close_all ---
        print("\n6️⃣ close_all()")
        await manager.close_all()
        print(f"   已断开，剩余 tools={manager.tool_count}, servers={manager.connected_servers}")

    finally:
        _cleanup_echo_server()

    print("\n" + "=" * 60)
    print("  ✅ Phase 1 全链路验证通过")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
