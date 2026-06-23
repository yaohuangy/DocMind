"""MCP Client 数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolInfo:
    """外部 MCP Server 提供的工具元信息。"""

    name: str
    """工具完整名称，格式 "server_name:tool_name"，如 "brave-search:brave_web_search"。"""

    description: str
    """工具描述文本。"""

    server_name: str
    """所属 MCP Server 名称。"""

    category: str = "general"
    """工具类别。"""

    input_schema: dict[str, Any] = field(default_factory=dict)
    """工具的 JSON Schema 输入定义。"""


@dataclass
class ToolResult:
    """工具调用结果。"""

    tool_name: str
    """调用的工具名。"""

    server_name: str
    """所属 MCP Server 名称。"""

    success: bool
    """调用是否成功。"""

    text: str = ""
    """返回的文本内容。"""

    data: dict[str, Any] | None = None
    """结构化数据（如果有）。"""

    error: str = ""
    """错误信息（success=False 时）。"""

    elapsed_ms: float = 0.0
    """调用耗时（毫秒）。"""


@dataclass
class ExternalResult:
    """外部搜索结果（供融合模块使用）。"""

    source: str
    """来源标识，如 "brave-search"。"""

    title: str = ""
    """结果标题。"""

    snippet: str = ""
    """结果摘要/片段。"""

    url: str = ""
    """来源 URL（搜索结果有）。"""

    score: float = 0.0
    """相关度分数。"""

    raw: dict[str, Any] | None = None
    """原始返回数据。"""
