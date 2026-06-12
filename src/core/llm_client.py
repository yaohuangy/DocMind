"""
LLM 客户端模块。

基于 OpenAI 兼容 SDK，提供对话、流式生成、查询变体生成、
假设文档生成和概念提取等高层方法。支持所有 OpenAI 兼容的 API 后端
（DeepSeek / 智谱 / DashScope / OpenAI 等）。
"""

import logging
from collections.abc import Generator

from openai import OpenAI

from src.core.config import LLMConfig, get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt 模板（内联，正式版迁移至 generation/prompt_templates.py）
# ---------------------------------------------------------------------------

QUERY_VARIANTS_SYSTEM = """\
你是一个查询扩展专家。给定一个用户问题，生成 {num_variants} 个不同角度的查询变体，
用于提高检索召回率。每个变体应从不同角度重新表述原问题：

1. **定义角度**：从概念定义和基本解释的角度提问
2. **机制角度**：从内部原理和工作机制的角度提问
3. **对比角度**：从与相关概念对比和区分的角度提问
4. **应用角度**：从实际应用和使用场景的角度提问

直接输出 JSON 字符串列表，不要包含任何其他文本。
示例输出格式：["变体1", "变体2", "变体3", "变体4"]
"""

HYPOTHETICAL_ANSWER_SYSTEM = """\
你是一个知识渊博的助手。请根据一个问题，生成一个假设性的答案（200-400字），
就好像你已经知道答案一样。不要说你不知道——大胆猜测，构建一个看似合理的回答段落。
这个假设答案将用于通过嵌入向量检索真正的相关文档。

直接输出假设答案文本，不要包含前缀或后记。
"""

CONCEPT_EXTRACTION_SYSTEM = """\
从文本中提取 3-8 个核心概念。每个概念一行，格式：
名称,类型,一句话描述

类型可选: mechanism,architecture,algorithm,theory,technique,tool,application,concept

示例:
Transformer,architecture,基于自注意力机制的神经网络架构
Self-Attention,mechanism,计算序列中元素间相关性的核心机制

只输出概念行，不要其他内容。"""


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """OpenAI 兼容 LLM 客户端。

    封装 OpenAI Python SDK，提供同步对话、流式生成以及
    高层语义方法（查询变体、假设文档、概念提取）。

    Usage::

        config = get_config()
        client = LLMClient(config.llm)
        answer = client.chat([{"role": "user", "content": "Hello"}])
        for token in client.chat_stream(messages):
            print(token, end="", flush=True)
    """

    def __init__(self, llm_config: LLMConfig | None = None):
        """
        Args:
            llm_config: LLM 配置。为 None 时自动从全局配置加载。
        """
        if llm_config is None:
            llm_config = get_config().llm
        self._config = llm_config

        self._client = OpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
        )

    # ------------------------------------------------------------------
    # 底层调用
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """同步对话，返回完整回复文本。

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 采样温度，None 则使用配置默认值。
            max_tokens: 最大生成 token 数，None 则使用配置默认值。

        Returns:
            LLM 生成的完整文本。
        """
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature if temperature is not None else self._config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self._config.max_tokens,
            stream=False,
        )
        content = response.choices[0].message.content  # type: ignore[union-attr]
        return content.strip() if content else ""

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """流式对话，逐 token yield。

        Args:
            messages: 消息列表。
            temperature: 采样温度。
            max_tokens: 最大生成 token 数。

        Yields:
            str: 每个增量 token 的文本片段。
        """
        stream = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature if temperature is not None else self._config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self._config.max_tokens,
            stream=True,
            timeout=self._config.stream_timeout,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta  # type: ignore[union-attr]
            if delta.content:
                yield delta.content

    # ------------------------------------------------------------------
    # 高层方法
    # ------------------------------------------------------------------

    def generate_query_variants(
        self,
        question: str,
        num_variants: int = 4,
    ) -> list[str]:
        """生成查询变体（MQE 用），从 4 个角度重述原问题。

        角度：定义角度 / 机制角度 / 对比角度 / 应用角度。

        Args:
            question: 原始用户问题。
            num_variants: 生成的变体数量，默认 4。

        Returns:
            查询变体字符串列表（包含原问题在内共 num_variants+1 条变体）。
        """
        import json as _json

        system = QUERY_VARIANTS_SYSTEM.format(num_variants=num_variants)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"原问题：{question}"},
        ]

        response = self.chat(messages, temperature=0.7, max_tokens=512)

        try:
            # 尝试从回复中提取 JSON 列表
            # 处理模型可能在 JSON 前后加 markdown 代码块的情况
            text = response.strip()
            if text.startswith("```"):
                # 移除 ```json ... ``` 包裹
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            variants: list[str] = _json.loads(text)
            # 确保至少返回 num_variants 条
            variants = variants[:num_variants]
            logger.info("生成 %d 条查询变体", len(variants))
            return variants
        except (_json.JSONDecodeError, TypeError):
            logger.warning("查询变体 JSON 解析失败，回退到行分割: %s", response[:100])
            # 回退：按行分割
            lines = [l.strip(" -•0123456789. ") for l in response.split("\n") if l.strip()]
            return lines[:num_variants] if lines else [question]

    def generate_hypothetical_answer(self, question: str) -> str:
        """生成假设答案文档（HyDE 用），200-400 字。

        Args:
            question: 用户问题。

        Returns:
            假设答案文本。
        """
        messages = [
            {"role": "system", "content": HYPOTHETICAL_ANSWER_SYSTEM},
            {"role": "user", "content": f"问题：{question}"},
        ]
        answer = self.chat(messages, temperature=0.5, max_tokens=600)
        logger.info("生成假设答案，长度 %d 字符", len(answer))
        return answer

    def extract_concepts(
        self,
        text: str,
    ) -> list[dict]:
        """从文本中提取关键概念（语义记忆构建用）。

        使用逗号分隔的纯文本格式，兼容所有模型。

        Args:
            text: 输入文本。

        Returns:
            概念字典列表，每个包含: name, type, description。
        """
        messages = [
            {"role": "system", "content": CONCEPT_EXTRACTION_SYSTEM},
            {"role": "user", "content": f"文本：\n{text}"},
        ]

        response = self.chat(messages, temperature=0.3, max_tokens=1024)
        print(f"[EXTRACT] LLM 原始响应 ({len(response)} 字符):\n{response[:400]}")

        concepts: list[dict] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            # 去除行首序号和符号
            for strip_prefix in ["- ", "• ", "* ", "· "]:
                if line.startswith(strip_prefix):
                    line = line[len(strip_prefix):]
                    break
            if line and line[0].isdigit():
                for sep in [". ", ") ", "、"]:
                    idx = line.find(sep, 0, 4)
                    if idx > 0:
                        line = line[idx + len(sep):]
                        break

            # 按逗号分割（最多3段：名称,类型,描述）
            parts = [p.strip() for p in line.split(",", 2)]
            if parts and parts[0]:
                concepts.append({
                    "name": parts[0],
                    "type": parts[1] if len(parts) > 1 and parts[1] else "concept",
                    "description": parts[2] if len(parts) > 2 and parts[2] else "",
                })

        logger.info("提取到 %d 个概念: %s", len(concepts),
                    [c["name"] for c in concepts])
        return concepts

    @property
    def model(self) -> str:
        """当前使用的模型名称。"""
        return self._config.model
