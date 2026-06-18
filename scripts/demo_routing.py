"""动态路由演示——直观感受查询复杂度分类效果。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.llm_client import LLMClient

llm = LLMClient()

questions = [
    # 简单 → Direct
    ("三文鱼是日料吗？", "simple"),
    ("什么是检索增强生成？", "simple"),
    ("今天星期几？", "simple"),
    ("Python的作者是谁？", "simple"),
    ("RAG是什么的缩写？", "simple"),
    ("文档管理支持哪些格式？", "simple"),
    ("什么是嵌入向量？", "simple"),
    ("重排序用什么模型？", "simple"),

    # 复杂 → HyDE
    ("Agent和普通程序在设计哲学上有什么根本区别？", "complex"),
    ("为什么语义分块比固定token分块效果更好？", "complex"),
    ("比较MQE、HyDE和Direct三种检索策略在不同场景下的适用性", "complex"),
    ("如果换了一个嵌入模型，对整个RAG系统会产生哪些连锁影响？", "complex"),
    ("如何设计一个评测体系来持续验证RAG系统的改进效果？", "complex"),
    ("RNN在处理长序列时为什么会出问题，Transformer又是如何解决的？", "complex"),
    ("动态路由应该如何平衡LLM调用开销和检索精度？", "complex"),
]

print("=" * 72)
print(f"{'问题':<45s} {'分类':>7s} {'预期':>7s} {'匹配'}")
print("-" * 72)

correct = 0
for q, expected in questions:
    result = llm.classify_complexity(q)
    match = "✓" if result == expected else "✗"
    if result == expected:
        correct += 1
    print(f"{q[:44]:<44s} {result:>7s} {expected:>7s} {match}")

print("-" * 72)
print(f"准确率: {correct}/{len(questions)} ({correct/len(questions)*100:.0f}%)")
print()

# Token 节省估算
simple_count = sum(1 for _, e in questions if e == "simple")
complex_count = sum(1 for _, e in questions if e == "complex")
print(f"简单→Direct: {simple_count} 个 | 复杂→HyDE: {complex_count} 个")
print(f"若全用 HyDE: ~{len(questions) * 2000} tokens")
print(f"动态路由后:  ~{complex_count * 2000} tokens (省 {simple_count * 2000} tokens)")
