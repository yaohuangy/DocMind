"""
LLM 调用耗时诊断。

直接测量 qwen-turbo 的三类 LLM 调用耗时：
1. 生成查询变体（MQE 用）
2. 生成假设答案（HyDE 用）
3. 生成最终答案（带上下文的 RAG 生成）
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import load_config
from src.core.llm_client import LLMClient
from src.core.embedder import create_embedder
from src.core.vector_store import VectorStore
from src.retrieval.direct_retriever import DirectRetriever


def main():
    config = load_config()
    print(f"LLM 模型: {config.llm.model}")
    print(f"LLM Base URL: {config.llm.base_url}")

    llm = LLMClient(config.llm)

    # ---- 1. 测量查询变体生成 ----
    print("\n[1] 测量 MQE 查询变体生成...")
    t0 = time.perf_counter()
    try:
        variants = llm.generate_query_variants("什么是 Transformer？", num_variants=4)
        elapsed = time.perf_counter() - t0
        print(f"  生成 {len(variants)} 个变体, 耗时 {elapsed:.1f}s")
        for v in variants:
            print(f"    - {v[:60]}")
        if elapsed > 5:
            print(f"  ⚠️ 慢！>5s，这会导致 MQE 检索延迟很高")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  ❌ 失败 ({elapsed:.1f}s): {e}")

    # ---- 2. 测量假设答案生成 ----
    print("\n[2] 测量 HyDE 假设答案生成...")
    t0 = time.perf_counter()
    try:
        hypo = llm.generate_hypothetical_answer("什么是 Transformer？")
        elapsed = time.perf_counter() - t0
        print(f"  生成 {len(hypo)} 字符, 耗时 {elapsed:.1f}s")
        if elapsed > 5:
            print(f"  ⚠️ 慢！>5s，这会导致 HyDE 检索延迟很高")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  ❌ 失败 ({elapsed:.1f}s): {e}")

    # ---- 3. 测量 RAG 答案生成（无上下文） ----
    print("\n[3] 测量答案生成（无上下文，200 token 上限）...")
    t0 = time.perf_counter()
    try:
        answer = llm.chat(
            messages=[
                {"role": "user", "content": "请用一句话解释什么是 Transformer 架构。"}
            ],
            max_tokens=200,
        )
        elapsed = time.perf_counter() - t0
        print(f"  生成 {len(answer)} 字符, 耗时 {elapsed:.1f}s")
        print(f"  答案: {answer[:100]}...")
        usage = llm.last_usage
        print(f"  Token: 输入={usage.get('prompt_tokens','?')}, 输出={usage.get('completion_tokens','?')}")
        if elapsed > 5:
            print(f"  ⚠️ 慢！>5s")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  ❌ 失败 ({elapsed:.1f}s): {e}")

    # ---- 4. 测量 RAG 答案生成（带上下文） ----
    print("\n[4] 测量答案生成（带 10 段上下文，模拟真实 RAG）...")
    # 从 ChromaDB 取真实上下文
    embedder = create_embedder(config.embedding)
    vs = VectorStore(config.chroma)
    retriever = DirectRetriever(embedder=embedder, vector_store=vs)
    sources = retriever.retrieve("什么是 Transformer？", top_k=10)

    if sources:
        context_blocks = []
        for i, s in enumerate(sources, 1):
            text = s.text[:800]
            context_blocks.append(f"[{i}] {text}")
        context = "\n---\n".join(context_blocks)

        prompt = f"""基于以下文档片段回答问题：什么是 Transformer？

文档片段：
{context}

请用一段话回答（不超过 300 字）。"""

        t0 = time.perf_counter()
        try:
            answer = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            elapsed = time.perf_counter() - t0
            usage = llm.last_usage
            print(f"  生成 {len(answer)} 字符, 耗时 {elapsed:.1f}s")
            print(f"  Token: 输入={usage.get('prompt_tokens','?')}, 输出={usage.get('completion_tokens','?')}")
            print(f"  答案: {answer[:120]}...")
            if elapsed > 5:
                print(f"  ⚠️ 慢！>5s")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  ❌ 失败 ({elapsed:.1f}s): {e}")

    # ---- 总结 ----
    print("\n" + "=" * 50)
    print("总结：MQE+HyDE 问答 = MQE变体(LLM) + HyDE假设(LLM) + 答案生成(LLM)")
    print("如果单个 LLM 调用 >5s，累计延迟就会 >15s")
    print("qwen-turbo 是免费模型，高峰期可能很慢，考虑换 qwen-plus 或 deepseek-chat")


if __name__ == "__main__":
    main()
