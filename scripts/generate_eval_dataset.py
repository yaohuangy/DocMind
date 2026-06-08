#!/usr/bin/env python
"""
生成检索评测 Ground Truth 数据集。

从已入库的 ChromaDB 文档分块中采样，使用 LLM 为每个分块
生成自然问题，构建 {问题 → 相关分块ID} 的评测集。

Usage:
    python scripts/generate_eval_dataset.py --num 30
    python scripts/generate_eval_dataset.py --num 30 --output data/evaluation/my_dataset.json
    python scripts/generate_eval_dataset.py --num 20 --delay 0.5

Requirements:
    - .env 已配置 LLM_API_KEY
    - 至少有一个文档已摄入 ChromaDB
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ground_truth_generator import GroundTruthGenerator


def main():
    parser = argparse.ArgumentParser(
        description="生成检索评测 Ground Truth 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/generate_eval_dataset.py --num 30
  python scripts/generate_eval_dataset.py --num 20 --output data/evaluation/my_dataset.json
        """,
    )

    parser.add_argument(
        "--num", "-n",
        type=int,
        default=30,
        help="目标问题数量 (default: 30)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="输出 JSON 路径 (default: data/evaluation/dataset_<timestamp>.json)",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.3,
        help="LLM 调用间隔秒数，避免 API 限流 (default: 0.3)",
    )
    parser.add_argument(
        "--rewrite", "-r",
        action="store_true",
        help="对问题做二次改写，用不同措辞重述（制造语义鸿沟，拉开 Direct vs MQE+HyDE 差距）",
    )
    parser.add_argument(
        "--expand-gt", "-e",
        action="store_true",
        help="将同文档相邻分块也标记为 GT，让指标更细腻（不再非0即1）",
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

    # 输出路径
    output = args.output
    if not output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"data/evaluation/dataset_{ts}.json"

    print(f"=== 生成评测数据集 ===")
    print(f"  目标问题数: {args.num}")
    print(f"  输出路径:   {output}")
    print(f"  API 间隔:   {args.delay}s")
    print(f"  改写措辞:   {'是' if args.rewrite else '否'}")
    print(f"  扩展GT:     {'是' if args.expand_gt else '否'}")
    print()

    # 生成
    generator = GroundTruthGenerator()

    try:
        questions = generator.generate_dataset(
            num_questions=args.num,
            output_path=output,
            delay_between_calls=args.delay,
            rewrite=args.rewrite,
            expand_gt=args.expand_gt,
        )
    except RuntimeError as e:
        print(f"\n错误: {e}")
        print("请先通过 Streamlit 应用上传并摄入至少一个文档。")
        sys.exit(1)

    print(f"\n✅ 评测集已生成: {len(questions)} 个问题")
    print(f"   文件: {output}")
    print(f"\n下一步: python scripts/run_evaluation.py --dataset {output}")


if __name__ == "__main__":
    main()
