"""
检索性能诊断脚本。

分别测量嵌入 API 调用和 ChromaDB 搜索的耗时，
帮助定位检索慢的瓶颈。
"""

import time
import sys
from pathlib import Path

# 确保项目根在 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import load_config
from src.core.embedder import create_embedder
from src.core.vector_store import VectorStore


def main():
    config = load_config()
    print(f"嵌入后端: {config.embedding.backend}")
    print(f"嵌入模型: {config.embedding.model}")
    print(f"嵌入批量: {config.embedding.batch_size}")

    # ---- 1. 测量嵌入 API 调用 ----
    print("\n[1] 测量嵌入 API 调用...")
    embedder = create_embedder(config.embedding)

    test_queries = [
        "什么是 Transformer？",
        "请解释自注意力机制的原理",
        "BERT 和 GPT 有什么区别？",
    ]

    for i, q in enumerate(test_queries):
        t0 = time.perf_counter()
        vec = embedder.embed_query(q)
        elapsed = time.perf_counter() - t0
        print(f"  查询{i+1}: '{q[:30]}...' → {len(vec)}维, 耗时 {elapsed:.3f}s")

    # ---- 2. 测量 ChromaDB 搜索 ----
    print("\n[2] 测量 ChromaDB 搜索...")
    vs = VectorStore(config.chroma)
    stats = vs.collection_stats(VectorStore.DOCUMENT_CHUNKS)
    print(f"  document_chunks 集合: {stats['count']} 条")

    if stats["count"] > 0:
        # 先用一个嵌入向量
        query_vec = embedder.embed_query("test query for benchmarking")

        for top_k in [5, 10, 20]:
            t0 = time.perf_counter()
            results = vs.search(
                collection_name=VectorStore.DOCUMENT_CHUNKS,
                query_embedding=query_vec,
                limit=top_k,
            )
            elapsed = time.perf_counter() - t0
            print(f"  top_k={top_k}: {len(results)} 结果, 耗时 {elapsed:.3f}s")

        # 带 user_id 过滤
        print("\n[3] 带 user_id 过滤的搜索...")
        for user in ["Yao", "default"]:
            t0 = time.perf_counter()
            results = vs.search(
                collection_name=VectorStore.DOCUMENT_CHUNKS,
                query_embedding=query_vec,
                limit=10,
                where={"user_id": user},
            )
            elapsed = time.perf_counter() - t0
            print(f"  user_id='{user}': {len(results)} 结果, 耗时 {elapsed:.3f}s")

    # ---- 4. 测量端到端 Direct 检索（包括 base_retriever 开销） ----
    print("\n[4] 测量端到端 Direct 检索（含 asyncio 开销）...")
    from src.retrieval.direct_retriever import DirectRetriever

    retriever = DirectRetriever(
        embedder=embedder,
        vector_store=vs,
        where_filter={"user_id": "Yao"},
    )

    for i, q in enumerate(test_queries):
        t0 = time.perf_counter()
        results = retriever.retrieve(q, top_k=5)
        elapsed = time.perf_counter() - t0
        print(f"  查询{i+1}: '{q[:30]}...' → {len(results)} 结果, 耗时 {elapsed:.3f}s")

    print("\n✅ 诊断完成")


if __name__ == "__main__":
    main()
