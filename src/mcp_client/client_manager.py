"""MCP Client Manager —— 管理外部 MCP Server 的连接、工具发现和调用。

使用示例::

    import asyncio
    from src.mcp_client import MCPClientManager, MCPServerConfig

    async def main():
        configs = [
            MCPServerConfig(
                name="brave-search",
                command="npx",
                args=["-y", "@anthropic/mcp-server-brave-search"],
                env={"BRAVE_API_KEY": "xxx"},
                category="search",
            ),
        ]
        async with MCPClientManager(configs) as manager:
            # 搜索外部
            results = await manager.search_external("AI trends 2026")
            for r in results:
                print(f"[{r.source}] {r.title}: {r.snippet}")

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from src.mcp_client.config import MCPServerConfig
from src.mcp_client.models import ExternalResult, ToolInfo, ToolResult

logger = logging.getLogger(__name__)


class MCPClientManager:
    """管理多个外部 MCP Server 的连接。

    支持 async context manager 协议::

        async with MCPClientManager(configs) as manager:
            tools = await manager.list_tools()
            result = await manager.call_tool("brave_web_search", query="hello")
    """

    def __init__(self, server_configs: list[MCPServerConfig] | None = None) -> None:
        """初始化管理器。

        Args:
            server_configs: 外部 MCP Server 配置列表。传 None 时从 .env 读取。
        """
        if server_configs is None:
            server_configs = _load_configs_from_env()

        # 只取 enabled=True 的
        self._configs: list[MCPServerConfig] = [c for c in server_configs if c.enabled]
        # 连接状态：server_name → (read, write, session)
        self._sessions: dict[str, tuple[Any, Any, ClientSession]] = {}
        # 工具注册表：tool_name → ToolInfo
        self._tools: dict[str, ToolInfo] = {}
        # 上下文管理器追踪
        self._contexts: list[Any] = []

    # ------------------------------------------------------------------
    # async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MCPClientManager:
        await self.connect_all()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close_all()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    async def connect_all(self) -> dict[str, bool]:
        """连接所有已配置的外部 MCP Server，发现它们提供的工具。

        Returns:
            {server_name: success} 字典。
        """
        results: dict[str, bool] = {}

        for config in self._configs:
            try:
                await self._connect_one(config)
                results[config.name] = True
                logger.info("已连接外部 MCP Server: %s", config.name)
            except Exception as exc:
                results[config.name] = False
                logger.warning("连接外部 MCP Server 失败 [%s]: %s", config.name, exc)

        return results

    async def _connect_one(self, config: MCPServerConfig) -> None:
        """连接单个外部 MCP Server，发现其工具并注册。"""
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )

        # stdio_client 是 async context manager
        ctx = stdio_client(server_params)
        read, write = await ctx.__aenter__()
        self._contexts.append(ctx)

        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()

        self._sessions[config.name] = (read, write, session)

        # 发现工具
        try:
            list_result = await asyncio.wait_for(
                session.list_tools(),
                timeout=config.timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("发现工具超时 [%s]", config.name)
            return

        for tool in list_result.tools:
            qualified_name = f"{config.name}:{tool.name}"
            self._tools[qualified_name] = ToolInfo(
                name=qualified_name,
                description=tool.description or "",
                server_name=config.name,
                category=config.category,
                input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
            )

        logger.info(
            "发现 %d 个工具 [%s]: %s",
            len(list_result.tools),
            config.name,
            [t.name for t in list_result.tools],
        )

    async def close_all(self) -> None:
        """断开所有外部 MCP Server 连接。"""
        for name in list(self._sessions.keys()):
            try:
                _, _, session = self._sessions[name]
                await session.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("关闭 session 异常 [%s]: %s", name, exc)

        self._sessions.clear()

        for ctx in reversed(self._contexts):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("关闭 stdio 异常: %s", exc)

        self._contexts.clear()
        self._tools.clear()
        logger.info("已断开所有外部 MCP Server 连接")

    # ------------------------------------------------------------------
    # 工具发现
    # ------------------------------------------------------------------

    def list_tools(self, category: str | None = None) -> list[ToolInfo]:
        """列出所有已发现的外部工具。

        Args:
            category: 按类别过滤（"search" / "filesystem" / ...），
                      None 返回全部。

        Returns:
            ToolInfo 列表（按 server_name 排序）。
        """
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        tools.sort(key=lambda t: (t.server_name, t.name))
        return tools

    def get_tool(self, name: str) -> ToolInfo | None:
        """根据完整名称查找工具。

        Args:
            name: 工具名，支持 "server:tool" 或仅 "tool"。
                  仅 "tool" 时匹配第一个同名的。

        Returns:
            ToolInfo 或 None。
        """
        # 精确匹配
        if name in self._tools:
            return self._tools[name]
        # 模糊匹配（找第一个同名工具）
        for tool in self._tools.values():
            if tool.name.endswith(f":{name}"):
                return tool
        return None

    # ------------------------------------------------------------------
    # 工具调用
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> ToolResult:
        """调用指定外部工具。

        Args:
            tool_name: 工具名，支持 "server:tool" 或仅 "tool"。
            arguments: 工具参数。
            timeout: 超时（秒），None 使用配置默认值。

        Returns:
            ToolResult（success=False 时检查 error 字段）。
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                server_name="?",
                success=False,
                error=f"未找到工具: {tool_name}。可用工具: {list(self._tools.keys())}",
            )

        server_name = tool.server_name
        if server_name not in self._sessions:
            return ToolResult(
                tool_name=tool.name,
                server_name=server_name,
                success=False,
                error=f"Server 未连接: {server_name}",
            )

        config = next((c for c in self._configs if c.name == server_name), None)
        effective_timeout = timeout or (config.timeout if config else 15)

        _, _, session = self._sessions[server_name]
        t0 = time.perf_counter()

        try:
            call_result = await asyncio.wait_for(
                session.call_tool(tool.name.split(":", 1)[1], arguments=arguments or {}),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=tool.name,
                server_name=server_name,
                success=False,
                error=f"调用超时（{effective_timeout}s）",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool.name,
                server_name=server_name,
                success=False,
                error=str(exc),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 提取文本内容和结构化数据
        text_parts: list[str] = []
        for block in call_result.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        return ToolResult(
            tool_name=tool.name,
            server_name=server_name,
            success=not call_result.isError,
            text="\n".join(text_parts),
            data=call_result.structuredContent,
            error="Tool returned isError=True" if call_result.isError else "",
            elapsed_ms=round(elapsed_ms, 1),
        )

    # ------------------------------------------------------------------
    # 高级搜索
    # ------------------------------------------------------------------

    async def search_external(self, query: str) -> list[ExternalResult]:
        """在外部搜索工具中执行查询。

        自动选择第一个可用的 search 类别工具，调用它进行搜索。
        如果无 search 类别工具，尝试所有工具中名称包含 "search" 的。

        Args:
            query: 搜索查询。

        Returns:
            ExternalResult 列表。
        """
        # 优先找 search 类别的工具，再按名称过滤真正的搜索工具
        all_search = self.list_tools(category="search")
        # 优先：名称中含有 "search" 的工具（排除 crawl/extract/map）
        search_tools = [t for t in all_search if "search" in t.name.lower().rsplit(":", 1)[-1]]
        if not search_tools:
            # 回退：所有 search 类别的工具
            search_tools = all_search

        if not search_tools:
            logger.warning("无可用搜索工具")
            return []

        results: list[ExternalResult] = []
        for tool in search_tools[:2]:  # 最多试 2 个搜索工具
            try:
                result = await self.call_tool(
                    tool.name,
                    arguments={"query": query},
                )
                if result.success:
                    # 尝试解析为结构化结果
                    results.extend(_parse_search_result(result, tool.server_name))
            except Exception as exc:
                logger.warning("搜索工具 %s 失败: %s", tool.name, exc)

        return results

    @property
    def connected_servers(self) -> list[str]:
        """已连接的 Server 名称列表。"""
        return list(self._sessions.keys())

    @property
    def tool_count(self) -> int:
        """已发现的外部工具总数。"""
        return len(self._tools)


# ------------------------------------------------------------------
# 辅助
# ------------------------------------------------------------------


def _load_configs_from_env() -> list[MCPServerConfig]:
    """从环境变量加载外部 MCP Server 配置。

    支持三种方式（优先级从高到低）：
    1. MCP_EXTERNAL_SERVERS JSON 数组
    2. 特定 Server 的环境变量（TAVILY_API_KEY 等）
    """
    import json
    import os

    configs: list[MCPServerConfig] = []

    # 方式 1：MCP_EXTERNAL_SERVERS JSON
    raw = os.getenv("MCP_EXTERNAL_SERVERS", "")
    if raw:
        try:
            data = json.loads(raw)
            for item in data:
                configs.append(MCPServerConfig(
                    name=item.get("name", "?"),
                    command=item.get("command", ""),
                    args=item.get("args", []),
                    env=item.get("env"),
                    cwd=item.get("cwd"),
                    enabled=item.get("enabled", True),
                    category=item.get("category", "general"),
                    timeout=item.get("timeout", 15),
                ))
            return configs
        except json.JSONDecodeError:
            logger.warning("MCP_EXTERNAL_SERVERS 解析失败，跳过")

    # 方式 2：Tavily Search
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        configs.append(MCPServerConfig(
            name="tavily-search",
            command="npx",
            args=["-y", "tavily-mcp@latest"],
            env={"TAVILY_API_KEY": tavily_key},
            category="search",
            timeout=20,
        ))

    return configs


def _parse_search_result(result: ToolResult, source: str) -> list[ExternalResult]:
    """将 ToolResult 解析为 ExternalResult 列表。

    支持：
    - structuredContent（JSON 格式）
    - Tavily 文本输出格式（"Title: ...\\nURL: ...\\nContent: ..."）
    - 纯文本回退
    """
    results: list[ExternalResult] = []

    # 尝试 structuredContent
    if result.data:
        if isinstance(result.data, list):
            for item in result.data:
                results.append(ExternalResult(
                    source=source,
                    title=item.get("title", ""),
                    snippet=item.get("snippet", item.get("description", "")),
                    url=item.get("url", ""),
                    raw=item,
                ))
            return results
        if isinstance(result.data, dict):
            items = result.data.get("results", result.data.get("items", []))
            if isinstance(items, list):
                for item in items:
                    results.append(ExternalResult(
                        source=source,
                        title=item.get("title", ""),
                        snippet=item.get("snippet", item.get("description", "")),
                        url=item.get("url", ""),
                        raw=item,
                    ))
                return results

    # 尝试解析 Tavily 文本格式
    text = result.text.strip()
    if text:
        parsed = _parse_tavily_text(text, source)
        if parsed:
            return parsed

    # 回退：返回纯文本作为一个结果
    if text:
        results.append(ExternalResult(
            source=source,
            snippet=text[:2000],
        ))

    return results


def _parse_tavily_text(text: str, source: str) -> list[ExternalResult]:
    """解析 Tavily 返回的文本格式结果。

    格式示例::

        Detailed Results:

        Title: xxx
        URL: https://...
        Content: ...

        Title: yyy
        URL: https://...
        Content: ...
    """
    import re

    results: list[ExternalResult] = []

    # 按 "Title:" 分割，每个 block 是一条结果
    # 先去掉开头的 "Detailed Results:" 等
    text = re.sub(r"^Detailed Results:\s*", "", text.strip())

    blocks = re.split(r"\n(?=Title:)", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        title_match = re.search(r"^Title:\s*(.+)$", block, re.MULTILINE)
        url_match = re.search(r"^URL:\s*(.+)$", block, re.MULTILINE)
        content_match = re.search(r"^Content:\s*(.+)$", block, re.MULTILINE | re.DOTALL)

        title = title_match.group(1).strip() if title_match else ""
        url = url_match.group(1).strip() if url_match else ""
        snippet = content_match.group(1).strip() if content_match else block[:500]

        if title or snippet:
            results.append(ExternalResult(
                source=source,
                title=title,
                snippet=snippet[:500],
                url=url,
            ))

    return results
