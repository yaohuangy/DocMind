"""
evaluation package - 检索评测框架

提供 Ground Truth 生成、延迟/精度评测和报告输出。

Usage:
    # 生成评测集
    python scripts/generate_eval_dataset.py --num 30

    # 跑评测
    python scripts/run_evaluation.py --dataset data/evaluation/dataset_xxx.json

注意：为避免导入重依赖（llama_index, chromadb, neo4j），
本包不在此处做 eager import。请按需从子模块导入。
"""

# 轻量导入（纯函数，无外部依赖）
from src.evaluation.metrics import (
    compute_all_metrics,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.evaluation.models import (
    EvalQuestion,
    EvalReport,
    MethodMetrics,
    PerQuestionResult,
)

__all__ = [
    # Metrics (纯函数)
    "compute_all_metrics",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    # Models
    "EvalQuestion",
    "PerQuestionResult",
    "MethodMetrics",
    "EvalReport",
]
