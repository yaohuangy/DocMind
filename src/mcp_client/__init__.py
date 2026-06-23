"""MCP Client 包 —— DocMind 作为 MCP Client 调用外部工具。

提供：

- ``MCPClientManager``：管理外部 MCP Server 连接，发现/调用工具
- ``MCPServerConfig`` / ``MCPClientConfig``：配置数据类
- ``ToolInfo`` / ``ToolResult`` / ``ExternalResult``：数据模型
- ``ExternalRouter`` / ``RouteDecision``：查询路由，判断是否需外部工具
"""

from src.mcp_client.client_manager import MCPClientManager
from src.mcp_client.config import MCPClientConfig, MCPServerConfig
from src.mcp_client.fusion import MergedResult, merge_results
from src.mcp_client.models import ExternalResult, ToolInfo, ToolResult
from src.mcp_client.router import ExternalRouter, RouteDecision

__all__ = [
    "MCPClientManager",
    "MCPServerConfig",
    "MCPClientConfig",
    "ToolInfo",
    "ToolResult",
    "ExternalResult",
    "ExternalRouter",
    "RouteDecision",
    "merge_results",
    "MergedResult",
]
