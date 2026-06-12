"""
评测数据模型。

定义评测框架中使用的所有数据传输对象。
"""

from dataclasses import dataclass, field


@dataclass
class EvalQuestion:
    """一条 Ground Truth 评测问题。

    Attributes:
        question: 用户问题文本。
        relevant_chunk_ids: Ground Truth——应被检索到的分块 ID 列表。
        doc_id: 来源文档 ID。
        source_chunk_text: 用于生成此问题的原始分块文本。
    """

    question: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    doc_id: str = ""
    source_chunk_text: str = ""


@dataclass
class PerQuestionResult:
    """单个问题 × 单个检索方法的评测结果。

    Attributes:
        question: 问题文本。
        method: 检索方法名。
        retrieved_chunk_ids: 检索到的分块 ID（按分数降序）。
        retrieved_scores: 对应分数。
        total_latency_sec: 总耗时（秒）。
        phase_timings: 各阶段耗时明细。
        answer_text: LLM 生成的答案（仅 --with-generation 模式下填充）。
        faithfulness: RAGAS 忠实度评分（0~1）。
        answer_relevancy: RAGAS 答案相关性评分（0~1）。
        input_tokens: 输入 prompt token 数。
        output_tokens: 输出 completion token 数。
    """

    question: str
    method: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    total_latency_sec: float = 0.0
    phase_timings: dict[str, float] = field(default_factory=dict)
    answer_text: str = ""
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class MethodMetrics:
    """单个检索方法在所有问题上的聚合指标。

    Attributes:
        method: 检索方法名。
        recall_at_5 / recall_at_10: Recall@k。
        precision_at_5 / precision_at_10: Precision@k。
        mrr: Mean Reciprocal Rank。
        ndcg_at_5 / ndcg_at_10: NDCG@k。
        avg_latency_sec: 平均延迟。
        p50_latency_sec / p95_latency_sec: 分位延迟。
        num_questions: 问题数。
    """

    method: str = ""
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    avg_latency_sec: float = 0.0
    p50_latency_sec: float = 0.0
    p95_latency_sec: float = 0.0
    num_questions: int = 0
    # 生成质量（RAGAS）
    avg_faithfulness: float | None = None
    avg_answer_relevancy: float | None = None
    # Token 成本
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0


@dataclass
class EvalReport:
    """完整评测报告。

    Attributes:
        dataset_path: 评测数据集路径。
        num_questions: 问题总数。
        methods: 各方法聚合指标列表。
        per_question: 逐问题×方法明细。
        generated_at: 报告生成时间。
    """

    dataset_path: str = ""
    num_questions: int = 0
    methods: list[MethodMetrics] = field(default_factory=list)
    per_question: list[PerQuestionResult] = field(default_factory=list)
    generated_at: str = ""
