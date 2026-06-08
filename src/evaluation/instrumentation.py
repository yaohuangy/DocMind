"""
评测计时仪表层。

提供对 LLMClient 和 Embedder 的非侵入式计时包装器。
不改动原有代码，通过委托模式记录每次调用的耗时。
"""

import time
from typing import List

from src.core.embedder import BaseEmbedder
from src.core.llm_client import LLMClient


class InstrumentedLLMClient:
    """LLMClient 计时包装器。

    记录 generate_query_variants 和 generate_hypothetical_answer
    的最近一次调用耗时。所有其他方法透明委托给原始客户端。

    Usage::

        inst_llm = InstrumentedLLMClient(real_llm_client)
        variants = inst_llm.generate_query_variants("什么是Transformer？")
        print(f"变体生成耗时: {inst_llm.variant_gen_time:.2f}s")
    """

    def __init__(self, wrapped: LLMClient):
        self._wrapped = wrapped

        # 最近一次耗时记录
        self.variant_gen_time: float = 0.0
        self.hypothetical_answer_time: float = 0.0
        self.chat_time: float = 0.0

    def generate_query_variants(
        self, question: str, num_variants: int = 4
    ) -> List[str]:
        t0 = time.perf_counter()
        result = self._wrapped.generate_query_variants(question, num_variants)
        self.variant_gen_time = time.perf_counter() - t0
        return result

    def generate_hypothetical_answer(self, question: str) -> str:
        t0 = time.perf_counter()
        result = self._wrapped.generate_hypothetical_answer(question)
        self.hypothetical_answer_time = time.perf_counter() - t0
        return result

    def chat(self, messages, temperature=None, max_tokens=None) -> str:
        t0 = time.perf_counter()
        result = self._wrapped.chat(messages, temperature, max_tokens)
        self.chat_time = time.perf_counter() - t0
        return result

    def __getattr__(self, name):
        """透明委托所有其他属性和方法。"""
        return getattr(self._wrapped, name)


class InstrumentedEmbedder:
    """Embedder 计时包装器。

    记录 embed 和 embed_query 的最近一次调用耗时。

    Usage::

        inst_emb = InstrumentedEmbedder(real_embedder)
        vec = inst_emb.embed_query("hello")
        print(f"嵌入耗时: {inst_emb.last_embed_time:.2f}s")
    """

    def __init__(self, wrapped: BaseEmbedder):
        self._wrapped = wrapped

        self.last_embed_time: float = 0.0
        self.last_embed_query_time: float = 0.0
        self._embed_count: int = 0

    def embed(self, texts: List[str]) -> List[List[float]]:
        t0 = time.perf_counter()
        result = self._wrapped.embed(texts)
        self.last_embed_time = time.perf_counter() - t0
        self._embed_count += 1
        return result

    def embed_query(self, text: str) -> List[float]:
        t0 = time.perf_counter()
        result = self._wrapped.embed_query(text)
        self.last_embed_query_time = time.perf_counter() - t0
        return result

    @property
    def embed_count(self) -> int:
        """embed 调用次数。"""
        return self._embed_count

    @property
    def dimension(self) -> int:
        return self._wrapped.dimension

    def __getattr__(self, name):
        """透明委托所有其他属性和方法。"""
        return getattr(self._wrapped, name)
