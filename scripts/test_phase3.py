"""Phase 3 验证 —— 本地+外部融合问答，带区分引用。

用法：venv/Scripts/python scripts/test_phase3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()


def test_generate_with_external():
    """测试 generate_with_external() —— 本地+外部融合问答。"""
    from src.engine.qa_engine import QAEngine

    engine = QAEngine()
    engine.set_user_id("Yao")

    print("=" * 60)
    print("  Phase 3: generate_with_external() 验证")
    print("=" * 60)

    # 测试：需要外部搜索的时效性问题
    question = "2026年AI Agent最新进展"
    print(f"\n❓ 问题: {question}")
    print(f"📡 正在检索本地+联网...\n")

    result = engine.generate_with_external(question, method="direct", top_k=3)

    # 路由决策
    decision = result["route_decision"]
    print(f"🧭 路由决策: need_external={decision.need_external}")
    print(f"   理由: {decision.reason}")

    # 融合结果
    merged = result["merged"]
    print(f"\n📊 融合结果: {len(merged)} 条")
    for m in merged:
        icon = "📄" if m.source_type == "local" else "🌐"
        print(f"   {icon} {m.citation} — {m.title[:60]}")

    # 答案
    answer = result["answer"]
    print(f"\n💬 答案 ({len(answer)} 字符):")
    print(answer[:600])
    if len(answer) > 600:
        print("...")

    # 本地来源
    local = result["local_sources"]
    print(f"\n📎 本地来源: {len(local)} 条")
    for s in local:
        print(f"   [{s.location_text}] {s.doc_name}")

    # 外部来源
    external = result["external_sources"]
    print(f"\n🔗 外部来源: {len(external)} 条")
    for e in external:
        print(f"   [{e['citation']}] {e['title'][:80]}")
        if e.get("url"):
            print(f"       🔗 {e['url'][:80]}")

    print("\n" + "=" * 60)
    print("  ✅ Phase 3 验证完成")
    print("=" * 60)


if __name__ == "__main__":
    test_generate_with_external()
