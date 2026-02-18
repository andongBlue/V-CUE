#!/usr/bin/env python3
"""
Run CulturalBench evaluation with V-CUE.

Usage:
    python scripts/run_culturalbench.py --config config/default.yaml
    python scripts/run_culturalbench.py --system_type text --model deepseek-v3
    python scripts/run_culturalbench.py --system_type vl --max_samples 100
"""

import argparse
import os
import sys

import yaml
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.culturalbench_eval import CulturalBenchEvaluator
from vcue.pipeline import VCUEPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run CulturalBench evaluation")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--system_type", type=str, choices=["text", "reasoning", "vl"],
        default=None, help="Override system type"
    )
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--task", type=str, choices=["mc", "tf", "both"], default="both",
        help="Task type: mc (multiple-choice), tf (true/false), both"
    )
    parser.add_argument("--output_dir", type=str, default="output/results/culturalbench")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.system_type:
        config["pipeline"]["system_type"] = args.system_type
    if args.model:
        system_type = config["pipeline"]["system_type"]
        if system_type == "vl":
            config["vlm"]["model_name"] = args.model
        elif system_type == "reasoning":
            config["llm_reasoning"]["model_name"] = args.model
        else:
            config["llm"]["model_name"] = args.model

    pipeline = VCUEPipeline(config=config)
    evaluator = CulturalBenchEvaluator(
        pipeline=pipeline,
        output_dir=args.output_dir,
    )

    dataset = evaluator.load_data()

    if args.task in ("mc", "both"):
        logger.info("=" * 60)
        logger.info("Evaluating Multiple-Choice Task")
        logger.info("=" * 60)
        mc_results = evaluator.evaluate_multiple_choice(
            dataset=dataset,
            num_runs=args.num_runs,
            max_samples=args.max_samples,
        )
        logger.info(f"MC Average Accuracy: {mc_results['average_accuracy']:.4f}")

    if args.task in ("tf", "both"):
        logger.info("=" * 60)
        logger.info("Evaluating True/False Task")
        logger.info("=" * 60)
        tf_results = evaluator.evaluate_true_false(
            dataset=dataset,
            num_runs=args.num_runs,
            max_samples=args.max_samples,
        )
        logger.info(f"TF Average Accuracy: {tf_results['average_accuracy']:.4f}")


if __name__ == "__main__":
    main()
