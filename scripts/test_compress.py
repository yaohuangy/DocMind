"""对话摘要压缩——模拟测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.llm_client import LLMClient
from src.memory.memory_manager import MemoryManager

mm = MemoryManager()
mm._llm_client = LLMClient()

# 生成 20 轮模拟对话
TOPICS = [
    "RAG技术原理",
    "语义分块策略",
    "重排序实现细节",
    "HyDE vs MQE对比",
    "Token优化方法",
    "嵌入模型选型",
    "后续优化方向",
    "文档摄入流程",
    "评测框架设计",
    "多模态输入支持",
    "动态路由机制",
    "Agent记忆架构",
    "Neo4j知识图谱",
    "ChromaDB向量库",
    "CI/CD自动化",
    "Docker部署方案",
    "反馈闭环设计",
    "引用格式化逻辑",
    "Streamlit前端",
    "Python类型注解",
]
qa = [
    (f"关于{t}，请详细介绍一下？", f"{t}是DocMind项目的核心模块之一，经过多轮迭代和评测验证，已经在生产环境中稳定运行。具体实现细节和优化参数可参考eval_data.md中的评测报告。")
    for t in TOPICS
]

print("=" * 60)
print("对话摘要压缩测试")
print("=" * 60)

for i, (q, a) in enumerate(qa, 1):
    mm.working.add(q, a)

    if mm.working.should_compress():
        old_count = mm.working.entry_count
        summary = mm.compress_working_memory(force=True)
        new_count = mm.working.entry_count
        ctx_len = len(mm.get_working_context())
        print(f"\n[第{i}轮] 触发压缩: {old_count}条 -> {new_count}条 | 上下文 {ctx_len} 字符")
        if summary:
            print(f"  摘要预览: {summary[:150]}...")
    else:
        ctx_len = len(mm.get_working_context())
        print(f"\n[第{i}轮] 上下文 {ctx_len} 字符 (未触发)")

print("\n" + "=" * 60)
print("最终上下文")
print("=" * 60)
ctx = mm.get_working_context()
print(ctx[:300])

# ---- Token 对比 ----
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# 无压缩：全 8 轮原话
raw = "\n---\n".join(
    f"用户: {q}\n助手: {a}" for q, a in qa
)
raw_tokens = len(enc.encode(raw))

# 有压缩：摘要 + 最近 3 轮
ctx_tokens = len(enc.encode(ctx))

print(f"\n{'='*60}")
print(f"Token 对比（tiktoken cl100k_base）")
print(f"{'='*60}")
print(f"无压缩（{len(qa)}轮原话）: {raw_tokens} tokens")
print(f"有压缩（摘要+3轮）: {ctx_tokens} tokens")
print(f"节省: {raw_tokens - ctx_tokens} tokens ({(1 - ctx_tokens/raw_tokens)*100:.0f}%)")
