"""
评测执行器。

加载 Ground Truth 数据集，对每种检索方法逐一评测，
收集延迟和精度数据，聚合为 MethodMetrics 输出对比报告。
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from src.core.config import Settings, get_config
from src.core.embedder import create_embedder
from src.core.llm_client import LLMClient
from src.core.vector_store import VectorStore
from src.engine.qa_engine import QAEngine
from src.evaluation.instrumentation import (
    InstrumentedEmbedder,
    InstrumentedLLMClient,
)
from src.evaluation.metrics import compute_all_metrics, percentile
from src.evaluation.models import (
    EvalQuestion,
    EvalReport,
    MethodMetrics,
    PerQuestionResult,
)

logger = logging.getLogger(__name__)

# 评测方法列表
DEFAULT_METHODS = ["direct", "mqe", "hyde", "mqe+hyde"]

# 方法显示名称
_METHOD_DISPLAY: dict[str, str] = {
    "direct": "直接检索",
    "mqe": "MQE",
    "hyde": "HyDE",
    "mqe+hyde": "MQE+HyDE",
}


class EvaluationRunner:
    """检索评测执行器。

    加载数据集，对每种方法跑全部问题，计算指标，输出报告。

    Usage::

        runner = EvaluationRunner()
        report = runner.run("data/evaluation/dataset.json", top_k=10)
        runner.print_table(report)
    """

    def __init__(self, config: Settings | None = None) -> None:
        """
        Args:
            config: 全局配置。None 则自动加载。
        """
        if config is None:
            config = get_config()
        self._config = config

        # 共享基础设施
        self._llm_client = LLMClient(config.llm)
        self._embedder = create_embedder(config.embedding)
        self._vector_store = VectorStore(config.chroma)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_path: str,
        methods: list[str] | None = None,
        top_k: int = 10,
    ) -> EvalReport:
        """执行完整评测。

        Args:
            dataset_path: Ground Truth 数据集 JSON 路径。
            methods: 评测方法列表。None 则使用全部 4 种。
            top_k: 检索结果截断数。

        Returns:
            EvalReport 包含各方法聚合指标和明细。

        Raises:
            FileNotFoundError: 数据集文件不存在。
            ValueError: 数据集为空。
        """
        # 加载数据集
        questions = self._load_dataset(dataset_path)
        if not questions:
            raise ValueError(f"数据集为空: {dataset_path}")

        methods = methods or DEFAULT_METHODS

        logger.info(
            "开始评测: %d 个问题, %d 种方法, top_k=%d",
            len(questions), len(methods), top_k,
        )

        all_per_question: list[PerQuestionResult] = []
        method_metrics_list: list[MethodMetrics] = []

        for method in methods:
            logger.info("--- 评测方法: %s ---", method)

            # 创建独立的计时基础设施（每个方法重置计时器）
            inst_llm = InstrumentedLLMClient(self._llm_client)
            inst_embedder = InstrumentedEmbedder(self._embedder)

            # 创建 QAEngine（使用计时组件）
            engine = QAEngine(self._config)
            # 替换为计时版本
            engine._llm_client = inst_llm
            engine._embedder = inst_embedder
            # 重建检索器缓存（使用新的计时组件）
            engine._retrievers = {}

            method_results: list[PerQuestionResult] = []

            for i, eq in enumerate(questions):
                pq = self._eval_one(
                    engine, eq, method, top_k, inst_llm, inst_embedder
                )
                method_results.append(pq)

                if (i + 1) % 5 == 0:
                    logger.info(
                        "  [%s] %d/%d 完成", method, i + 1, len(questions)
                    )

            all_per_question.extend(method_results)

            # 聚合指标（使用 GT 数据集计算精度）
            metrics = self._aggregate_with_dataset(method, method_results, questions)
            method_metrics_list.append(metrics)

            logger.info(
                "  %s: recall@5=%.2f recall@10=%.2f mrr=%.2f avg_lat=%.2fs",
                method, metrics.recall_at_5, metrics.recall_at_10,
                metrics.mrr, metrics.avg_latency_sec,
            )

        report = EvalReport(
            dataset_path=dataset_path,
            num_questions=len(questions),
            methods=method_metrics_list,
            per_question=all_per_question,
            generated_at=datetime.now().isoformat(),
        )

        logger.info("评测完成")
        return report

    # ------------------------------------------------------------------
    # 单问题评测
    # ------------------------------------------------------------------

    def _eval_one(
        self,
        engine: QAEngine,
        eq: EvalQuestion,
        method: str,
        top_k: int,
        inst_llm: InstrumentedLLMClient,
        inst_embedder: InstrumentedEmbedder,
    ) -> PerQuestionResult:
        """对单个问题执行一次检索评测。

        Args:
            engine: QAEngine 实例。
            eq: 评测问题。
            method: 检索方法。
            top_k: 截断数。
            inst_llm: 计时 LLM 客户端。
            inst_embedder: 计时嵌入器。

        Returns:
            PerQuestionResult。
        """
        # 重置计时器
        inst_llm.variant_gen_time = 0.0
        inst_llm.hypothetical_answer_time = 0.0
        inst_embedder.last_embed_time = 0.0
        inst_embedder.last_embed_query_time = 0.0

        # 执行检索（计时）
        t0 = time.perf_counter()

        try:
            sources = engine.retrieve(eq.question, method=method, top_k=top_k)
        except Exception as e:
            logger.error("检索失败 [%s] %s: %s", method, eq.question[:50], e)
            return PerQuestionResult(
                question=eq.question,
                method=method,
                retrieved_chunk_ids=[],
                retrieved_scores=[],
                total_latency_sec=time.perf_counter() - t0,
                phase_timings={"error": str(e)[:80]},
            )

        wall_time = time.perf_counter() - t0

        # 收集阶段耗时
        phase_timings = {
            "llm_variants": inst_llm.variant_gen_time,
            "llm_hypothetical": inst_llm.hypothetical_answer_time,
            "embedding": inst_embedder.last_embed_time + inst_embedder.last_embed_query_time,
            "total_wall": wall_time,
        }

        return PerQuestionResult(
            question=eq.question,
            method=method,
            retrieved_chunk_ids=[s.chunk_id for s in sources],
            retrieved_scores=[s.score for s in sources],
            total_latency_sec=wall_time,
            phase_timings=phase_timings,
        )

    # ------------------------------------------------------------------
    # 聚合
    # ------------------------------------------------------------------

    def _aggregate_with_dataset(
        self,
        method: str,
        results: list[PerQuestionResult],
        questions: list[EvalQuestion],
    ) -> MethodMetrics:
        """使用数据集中的 GT 信息计算聚合指标。

        Args:
            method: 方法名。
            results: PerQuestionResult 列表（顺序与 questions 一致）。
            questions: EvalQuestion 列表。

        Returns:
            MethodMetrics。
        """
        # 构建 question_text → GT IDs 映射
        gt_map: dict[str, set[str]] = {}
        for eq in questions:
            gt_map[eq.question] = set(eq.relevant_chunk_ids)

        recall5_vals: list[float] = []
        recall10_vals: list[float] = []
        prec5_vals: list[float] = []
        prec10_vals: list[float] = []
        mrr_vals: list[float] = []
        ndcg5_vals: list[float] = []
        ndcg10_vals: list[float] = []
        latencies: list[float] = []

        for r in results:
            gt_ids = gt_map.get(r.question, set())
            m = compute_all_metrics(r.retrieved_chunk_ids, gt_ids, k_values=(5, 10))

            recall5_vals.append(m["recall@5"])
            recall10_vals.append(m["recall@10"])
            prec5_vals.append(m["precision@5"])
            prec10_vals.append(m["precision@10"])
            mrr_vals.append(m["mrr"])
            ndcg5_vals.append(m["ndcg@5"])
            ndcg10_vals.append(m["ndcg@10"])
            latencies.append(r.total_latency_sec)

        return MethodMetrics(
            method=method,
            recall_at_5=sum(recall5_vals) / len(recall5_vals) if recall5_vals else 0.0,
            recall_at_10=sum(recall10_vals) / len(recall10_vals) if recall10_vals else 0.0,
            precision_at_5=sum(prec5_vals) / len(prec5_vals) if prec5_vals else 0.0,
            precision_at_10=sum(prec10_vals) / len(prec10_vals) if prec10_vals else 0.0,
            mrr=sum(mrr_vals) / len(mrr_vals) if mrr_vals else 0.0,
            ndcg_at_5=sum(ndcg5_vals) / len(ndcg5_vals) if ndcg5_vals else 0.0,
            ndcg_at_10=sum(ndcg10_vals) / len(ndcg10_vals) if ndcg10_vals else 0.0,
            avg_latency_sec=sum(latencies) / len(latencies) if latencies else 0.0,
            p50_latency_sec=percentile(latencies, 50),
            p95_latency_sec=percentile(latencies, 95),
            num_questions=len(results),
        )

    # ------------------------------------------------------------------
    # 报告输出
    # ------------------------------------------------------------------

    def print_table(self, report: EvalReport) -> None:
        """打印对比表格到终端。

        Args:
            report: 评测报告。
        """
        methods = report.methods
        if not methods:
            print("无评测数据")
            return

        print()
        print("=" * 100)
        print(f"  Retrieval Benchmark Results ({report.num_questions} questions, k=10)")
        print(f"  Dataset: {report.dataset_path}")
        print(f"  Time: {report.generated_at}")
        print("=" * 100)

        # 表头
        header = (
            f"{'Method':<14} {'R@5':>6} {'R@10':>6} {'P@5':>6} {'P@10':>6} "
            f"{'MRR':>6} {'NDCG@10':>8} {'Avg Lat':>8} {'P50':>8} {'P95':>8}"
        )
        print(header)
        print("-" * 100)

        # 找到延迟最低的方法（用于对比）
        min_lat = min(m.avg_latency_sec for m in methods) if methods else 1.0

        for m in methods:
            display = _METHOD_DISPLAY.get(m.method, m.method)
            row = (
                f"{display:<14} "
                f"{m.recall_at_5:>6.1%} "
                f"{m.recall_at_10:>6.1%} "
                f"{m.precision_at_5:>6.1%} "
                f"{m.precision_at_10:>6.1%} "
                f"{m.mrr:>6.2f} "
                f"{m.ndcg_at_10:>8.2f} "
                f"{m.avg_latency_sec:>6.2f}s "
                f"{m.p50_latency_sec:>6.2f}s "
                f"{m.p95_latency_sec:>6.2f}s"
            )
            print(row)

        print("=" * 100)
        print()

        # 简单解读
        best_recall = max(methods, key=lambda m: m.recall_at_10)
        fastest = min(methods, key=lambda m: m.avg_latency_sec)

        print(f"  最高 Recall@10: {_METHOD_DISPLAY.get(best_recall.method, best_recall.method)} ({best_recall.recall_at_10:.1%})")
        print(f"  最低延迟:       {_METHOD_DISPLAY.get(fastest.method, fastest.method)} ({fastest.avg_latency_sec:.2f}s)")
        print()

    def save_report(self, report: EvalReport, output_path: str) -> None:
        """保存完整评测报告到 JSON。

        Args:
            report: 评测报告。
            output_path: 输出路径。
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dataset_path": report.dataset_path,
            "num_questions": report.num_questions,
            "generated_at": report.generated_at,
            "methods": [
                {
                    "method": m.method,
                    "display": _METHOD_DISPLAY.get(m.method, m.method),
                    "recall_at_5": m.recall_at_5,
                    "recall_at_10": m.recall_at_10,
                    "precision_at_5": m.precision_at_5,
                    "precision_at_10": m.precision_at_10,
                    "mrr": m.mrr,
                    "ndcg_at_5": m.ndcg_at_5,
                    "ndcg_at_10": m.ndcg_at_10,
                    "avg_latency_sec": m.avg_latency_sec,
                    "p50_latency_sec": m.p50_latency_sec,
                    "p95_latency_sec": m.p95_latency_sec,
                    "num_questions": m.num_questions,
                }
                for m in report.methods
            ],
            "per_question": [
                {
                    "question": pq.question[:120],
                    "method": pq.method,
                    "retrieved_chunk_ids": pq.retrieved_chunk_ids[:10],
                    "total_latency_sec": pq.total_latency_sec,
                    "phase_timings": pq.phase_timings,
                }
                for pq in report.per_question
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info("评测报告已保存: %s", output_path)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dataset(dataset_path: str) -> list[EvalQuestion]:
        """从 JSON 加载评测数据集。

        Args:
            dataset_path: JSON 文件路径。

        Returns:
            EvalQuestion 列表。
        """
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        questions: list[EvalQuestion] = []
        for item in data.get("questions", []):
            questions.append(EvalQuestion(
                question=item["question"],
                relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                doc_id=item.get("doc_id", ""),
                source_chunk_text=item.get("source_chunk_text", ""),
            ))

        return questions
