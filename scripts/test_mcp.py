"""验证 MCP Server 的三个工具是否正常工作。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.qa_engine import QAEngine

engine = QAEngine()
engine.set_user_id("Yao")

print("=" * 60)
print("  1. search_documents  —— 语义检索文档")
print("=" * 60)
results = engine.retrieve("什么是智能体", method="direct", top_k=4)
if results:
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.doc_name} | 相似度: {r.score:.3f}")
        print(f"    {r.text[:100]}...\n")
else:
    print("未找到相关文档")

print("=" * 60)
print("  2. ask_knowledge_base  —— 检索 + 生成答案")
print("=" * 60)
answer = engine.generate("智能体的定义是什么", results[:3] if results else [], method="direct")
print(f"{answer}\n")

print("=" * 60)
print("  3. list_knowledge_base  —— 列出已加载文档")
print("=" * 60)
try:
    docs = engine.list_documents()
    print(f"共 {len(docs)} 个文档：\n")
    for d in docs:
        print(f"  📄 {d['name']} | {d.get('format','?')} | "
              f"{d.get('num_chunks',0)} chunks | {d.get('char_count',0):,} 字符")
except Exception as e:
    print(f"获取失败: {e}")

print("\n" + "=" * 60)
print("  验证完成 —— 如果上面都正常，MCP Server 即可用")
print("=" * 60)
