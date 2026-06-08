#!/usr/bin/env python
"""
运行检索评测 Benchmark。

加载 Ground Truth 数据集，对 4 种检索方法逐一评测，
输出精度和延迟对比表格，保存完整 JSON 报告。

Usage:
    python scripts/run_evaluation.py --dataset data/evaluation/dataset_xxx.json
    python scripts/run_evaluation.py --dataset data/evaluation/dataset.json --methods direct,mqe,hyde,mqe+hyde --top-k 10
    python scripts/run_evaluation.py --dataset data/evaluation/dataset.json --output data/evaluation/results.json

Requirements:
    - .env 已配置 LLM_API_KEY
    - 已运行 generate_eval_dataset.py 生成数据集
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluation_runner import EvaluationRunner


def main():
    parser = argparse.ArgumentParser(
        description="运行检索评测 Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_evaluation.py --dataset data/evaluation/dataset_20240605_120000.json
  python scripts/run_evaluation.py --dataset data/evaluation/dataset.json --methods direct,mqe+hyde --top-k 5
        """,
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Ground Truth 数据集 JSON 路径",
    )
    parser.add_argument(
        "--methods", "-m",
        type=str,
        default="direct,mqe,hyde,mqe+hyde",
        help="评测方法，逗号分隔 (default: direct,mqe,hyde,mqe+hyde)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="检索截断数 (default: 10)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="评测报告 JSON 输出路径 (default: data/evaluation/results_<timestamp>.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志",
    )

    args = parser.parse_args()

    # 日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 检查数据集存在
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"错误: 数据集文件不存在: {args.dataset}")
        print("请先运行: python scripts/generate_eval_dataset.py --num 30")
        sys.exit(1)

    # 解析方法
    methods = [m.strip() for m in args.methods.split(",")]

    # 输出路径
    output = args.output
    if not output:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"data/evaluation/results_{ts}.json"

    print(f"=== 检索评测 Benchmark ===")
    print(f"  数据集:   {args.dataset}")
    print(f"  方法:     {', '.join(methods)}")
    print(f"  Top-K:    {args.top_k}")
    print(f"  输出:     {output}")
    print()

    # 跑评测
    runner = EvaluationRunner()

    try:
        report = runner.run(
            dataset_path=args.dataset,
            methods=methods,
            top_k=args.top_k,
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 打印表格
    runner.print_table(report)

    # 保存报告
    runner.save_report(report, output)
    print(f"完整报告已保存: {output}")


if __name__ == "__main__":
    main()
