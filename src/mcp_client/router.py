"""查询路由决策器 —— 判断一个问题是否需要调用外部工具（联网搜索等）。

设计：规则优先（零成本、零延迟），LLM 兜底（复杂/模糊问题）。

使用示例::

    from src.mcp_client.router import ExternalRouter, RouteDecision

    router = ExternalRouter()
    decision = router.decide("2026年AI最新趋势是什么")
    # → RouteDecision(need_external=True, reason="含时效性关键词：2026,最新",
    #                  suggested_tools=["web_search"], confidence=0.9)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    """路由决策结果。"""

    need_external: bool
    """是否需要调用外部工具。"""

    reason: str = ""
    """决策理由。"""

    suggested_tools: list[str] = field(default_factory=list)
    """建议使用的外部工具列表，如 ["web_search"]。"""

    confidence: float = 0.0
    """置信度 0-1（规则匹配=0.9+，LLM 按输出）。"""

    method: str = "rule"
    """决策方式：rule / llm。"""


# ---------------------------------------------------------------------------
# 规则定义
# ---------------------------------------------------------------------------

# 时效性关键词 → 几乎肯定需要联网搜索
TEMPORAL_KEYWORDS = [
    "最新", "最近", "今天", "昨天", "本周", "本月", "今年",
    "现在", "当前", "目前", "刚刚", "刚刚发布",
    "2025", "2026", "2027", "2024",
]

# 趋势/对比类 → 可能需要外部基准数据
TREND_KEYWORDS = [
    "趋势", "行业", "对比", "市场", "排名", "排行",
    "超过", "领先", "落后", "基准", "benchmark",
    "预测", "展望", "前景", "走向",
]

# 新闻/事件类 → 通常需要实时信息
NEWS_KEYWORDS = [
    "新闻", "事件", "发生了什么", "股价", "天气",
    "发布会", "上市", "融资", "收购",
]

# 纯文档内概念 → 不需要外部搜索（减弱误判）
DOCUMENT_ONLY_PATTERNS = [
    r"第[一二三四五六七八九十\d]+章",
    r"第[一二三四五六七八九十\d]+节",
    r"根据文档",
    r"文档中[的提及说到]",
    r"上面(提到|说到|写的)",
    r"这本书",
    r"本文",
    r"作者",
]


class ExternalRouter:
    """判断问题是否需外部工具的路由器。

    双模式：
    - **rule**：正则+关键词匹配（默认，零延迟）
    - **llm**：调用 LLM 判断（高准确率，~200 tokens）

    优先规则，规则不确定时可选 LLM 兜底。
    """

    def __init__(self, use_llm_fallback: bool = False) -> None:
        """初始化路由器。

        Args:
            use_llm_fallback: True 时规则不确定则调 LLM。
        """
        self._use_llm_fallback = use_llm_fallback
        self._llm_client: Any = None  # 延迟加载

    def decide(self, question: str) -> RouteDecision:
        """判断一个问题是否需要外部工具。

        Args:
            question: 用户问题。

        Returns:
            RouteDecision，含 need_external / reason / suggested_tools。
        """
        # 1. 规则匹配
        decision = self._rule_match(question)
        if decision.confidence >= 0.65:
            return decision

        # 2. 规则不确定 → LLM（如果启用）
        if self._use_llm_fallback:
            return self._llm_decide(question)

        # 3. 规则不确定且无 LLM → 保守：不调外部
        return RouteDecision(
            need_external=False,
            reason="规则无法确定，保守跳过外部搜索",
            confidence=0.3,
            method="rule",
        )

    def _rule_match(self, question: str) -> RouteDecision:
        """基于关键词和正则的快速匹配。"""
        q_lower = question.lower()

        # 先检查：是否明确是文档内问题（降低误判）
        for pattern in DOCUMENT_ONLY_PATTERNS:
            if re.search(pattern, question):
                return RouteDecision(
                    need_external=False,
                    reason=f"匹配文档内问题模式: {pattern}",
                    confidence=0.9,
                )

        reasons: list[str] = []

        # 时效性关键词
        temporal_matches = [kw for kw in TEMPORAL_KEYWORDS if kw in question]
        if temporal_matches:
            reasons.append(f"时效性关键词：{','.join(temporal_matches)}")

        # 趋势/对比
        trend_matches = [kw for kw in TREND_KEYWORDS if kw in q_lower]
        if trend_matches:
            reasons.append(f"趋势/对比关键词：{','.join(trend_matches)}")

        # 新闻事件
        news_matches = [kw for kw in NEWS_KEYWORDS if kw in q_lower]
        if news_matches:
            reasons.append(f"新闻事件关键词：{','.join(news_matches)}")

        if reasons:
            # 时效性关键词权重最高（每条 0.25），趋势/新闻次之（每条 0.18）
            confidence = 0.5
            confidence += 0.25 * len([r for r in reasons if "时效性" in r])
            confidence += 0.18 * len([r for r in reasons if "趋势" in r or "新闻" in r])
            confidence = min(0.95, confidence)
            return RouteDecision(
                need_external=True,
                reason="; ".join(reasons),
                suggested_tools=["web_search"],
                confidence=confidence,
            )

        # 短问题（< 8 字）且无明确文档指向 → 可能是实时查询
        if len(question) < 8 and not any(kw in q_lower for kw in ["什么", "如何", "定义", "概念"]):
            return RouteDecision(
                need_external=True,
                reason="短问题，可能为实时查询",
                suggested_tools=["web_search"],
                confidence=0.5,
            )

        # 默认不需要
        return RouteDecision(
            need_external=False,
            reason="未匹配任何外部搜索规则",
            confidence=0.6,
        )

    def _llm_decide(self, question: str) -> RouteDecision:
        """调用 LLM 做路由判断。"""
        try:
            from src.core.llm_client import LLMClient

            if self._llm_client is None:
                self._llm_client = LLMClient()

            prompt = f"""你是一个查询分析器。判断以下问题是否需要调用外部工具（联网搜索）来回答。

问题："{question}"

判断标准：
- 涉及时效性（最新、现在、今年、2026 等）→ 需要联网搜索
- 涉及行业对比、市场数据、外部标准、新闻事件 → 需要联网搜索
- 纯粹针对已上传文档的提问（解释概念、查找章节内容）→ 不需要
- 从文档内部能找到答案的事实性问题 → 不需要

请只返回 JSON：
{{"need_external": true/false, "reason": "一句话理由"}}"""

            raw = self._llm_client.chat([
                {"role": "user", "content": prompt}
            ])
            # 尝试解析 JSON
            import json
            # 提取 JSON 部分（LLM 可能包裹在 markdown 中）
            json_match = re.search(r'\{[^}]+\}', raw)
            if json_match:
                data = json.loads(json_match.group())
                return RouteDecision(
                    need_external=data.get("need_external", False),
                    reason=data.get("reason", "LLM 判断"),
                    suggested_tools=["web_search"] if data.get("need_external") else [],
                    confidence=0.7,
                    method="llm",
                )
        except Exception as exc:
            logger.warning("LLM 路由判断失败: %s", exc)

        return RouteDecision(
            need_external=False,
            reason="LLM 判断异常，回退保守策略",
            confidence=0.3,
            method="llm",
        )
