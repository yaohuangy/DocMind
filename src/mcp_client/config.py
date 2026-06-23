"""MCP Client 配置数据类。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MCPServerConfig:
    """单个外部 MCP Server 的连接配置。

    Examples:
        # Brave Search（通过 npx 启动）
        MCPServerConfig(
            name="brave-search",
            command="npx",
            args=["-y", "@anthropic/mcp-server-brave-search"],
            env={"BRAVE_API_KEY": "xxx"},
            category="search",
        )

        # 本地 Python MCP Server
        MCPServerConfig(
            name="local-notes",
            command="python",
            args=["-m", "my_notes_server"],
            cwd="/path/to/project",
            category="filesystem",
        )
    """

    name: str
    """唯一名称，如 "brave-search"、\"local-files\"。"""

    command: str
    """启动命令，如 "npx"、\"python\"、\"node\"。"""

    args: list[str] = field(default_factory=list)
    """命令行参数。"""

    env: dict[str, str] | None = None
    """额外的环境变量（API Key 等）。"""

    cwd: str | None = None
    """工作目录。"""

    enabled: bool = True
    """是否启用。False 时 connect_all() 跳过此 Server。"""

    category: str = "general"
    """工具类别：search / filesystem / database / general。用于 search_external() 路由。"""

    timeout: int = 15
    """单次工具调用超时（秒）。"""


@dataclass
class MCPClientConfig:
    """MCP Client 全局配置。

    从 .env 读取：
        MCP_CLIENT_ENABLED=true
        MCP_EXTERNAL_SEARCH_TIMEOUT=10
        MCP_MAX_EXTERNAL_RESULTS=3
    """

    enabled: bool = False
    """总开关。False 时 MCPClientManager 不初始化。"""

    servers: list[MCPServerConfig] = field(default_factory=list)
    """已配置的外部 MCP Server 列表。"""

    external_search_timeout: int = 10
    """search_external() 超时（秒）。"""

    max_external_results: int = 3
    """search_external() 最多返回几条结果。"""
