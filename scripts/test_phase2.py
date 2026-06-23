"""Phase 2 验证 —— 路由决策 + QAEngine 本地+外部融合搜索。

用法：venv/Scripts/python scripts/test_phase2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()


def test_router():
    """测试路由决策器。"""
    from src.mcp_client.router import ExternalRouter

    router = ExternalRouter()

    test_cases = [
        # (问题, 预期 need_external)
        ("什么是Transformer架构", False),
        ("请解释第三章讨论的注意力机制", False),
        ("2026年AI Agent最新进展是什么", True),
        ("今年大模型行业有什么趋势", True),
        ("今天全球股市怎么样", True),
        ("英伟达最新发布的GPU性能如何", True),
        ("对比GPT-5和Claude Opus的差异", True),
        ("根据文档，作者对RAG的观点是什么", False),
        ("这本书第5章讲了什么", False),
    ]

    print("=" * 60)
    print("  Phase 2.1: 路由决策测试")
    print("=" * 60)

    correct = 0
    for question, expected in test_cases:
        decision = router.decide(question)
        match = "✅" if decision.need_external == expected else "❌"
        if decision.need_external == expected:
            correct += 1
        print(f"\n{match} Q: {question[:50]}")
        print(f"   need_external={decision.need_external} (expected={expected})")
        print(f"   reason: {decision.reason}")
        print(f"   confidence: {decision.confidence}")

    print(f"\n📊 准确率: {correct}/{len(test_cases)} ({correct*100//len(test_cases)}%)")

    return correct == len(test_cases)


def test_retrieve_with_external():
    """测试 QAEngine.retrieve_with_external()。"""
    from src.engine.qa_engine import QAEngine

    engine = QAEngine()
    engine.set_user_id("Yao")

    print("\n" + "=" * 60)
    print("  Phase 2.2: retrieve_with_external()")
    print("=" * 60)

    # 测试1：不需要外部搜索的问题
    print("\n📌 问题1: 什么是智能体（预期：纯本地）")
    result = engine.retrieve_with_external("什么是智能体", method="direct", top_k=3)

    local = result["local"]
    external = result["external"]
    decision = result["route_decision"]

    print(f"   路由: need_external={decision.need_external}, reason={decision.reason}")
    print(f"   本地结果: {len(local)} 条")
    print(f"   外部结果: {len(external)} 条")
    if local:
        print(f"   [{local[0].doc_name}] {local[0].text[:80]}...")

    # 测试2：需要外部搜索的问题
    print("\n📌 问题2: 2026年AI Agent最新进展（预期：本地+外部）")
    result2 = engine.retrieve_with_external("2026年AI Agent最新进展", method="direct", top_k=3)

    local2 = result2["local"]
    external2 = result2["external"]
    decision2 = result2["route_decision"]

    print(f"   路由: need_external={decision2.need_external}, reason={decision2.reason}")
    print(f"   本地结果: {len(local2)} 条")
    print(f"   外部结果: {len(external2)} 条")
    for i, ext in enumerate(external2[:3], 1):
        print(f"   [E{i}] {ext.title}")
        print(f"       {ext.snippet[:100]}...")
        if ext.url:
            print(f"       🔗 {ext.url[:80]}...")

    print(f"\n✅ retrieve_with_external() 完成")


def test_has_external_search():
    """测试 has_external_search()。"""
    from src.engine.qa_engine import QAEngine

    engine = QAEngine()

    print("\n" + "=" * 60)
    print("  Phase 2.2: has_external_search()")
    print("=" * 60)

    available = engine.has_external_search()
    print(f"   外部搜索可用: {available}")

    return True


if __name__ == "__main__":
    test_has_external_search()
    test_router()
    test_retrieve_with_external()

    print("\n" + "=" * 60)
    print("  ✅ Phase 2 全链路验证完成")
    print("=" * 60)
