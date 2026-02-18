#!/usr/bin/env python3
"""
Run CARE-eval benchmark evaluation with V-CUE.

Usage:
    python scripts/run_care.py --config config/default.yaml
    python scripts/run_care.py --system_type text --cultures chinese japanese
    python scripts/run_care.py --num_runs 3 --max_samples 50
"""

import argparse
import os
import sys

import yaml
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.care_eval import CAREEvaluator
from vcue.pipeline import VCUEPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run CARE-eval evaluation")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--system_type", type=str, choices=["text", "reasoning", "vl"],
        default=None,
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--cultures", nargs="+",
        choices=["chinese", "japanese", "arabic"],
        default=None,
    )
    parser.add_argument("--output_dir", type=str, default="output/results/care")
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

    judge_config = config.get("care_judge", {})
    evaluator = CAREEvaluator(
        pipeline=pipeline,
        judge_config=judge_config,
        output_dir=args.output_dir,
    )

    results = evaluator.evaluate(
        num_runs=args.num_runs,
        max_samples=args.max_samples,
        cultures=args.cultures,
    )

    logger.info("=" * 60)
    logger.info("CARE-eval Results Summary")
    logger.info("=" * 60)
    for culture, data in results.items():
        logger.info(
            f"  {culture}: base={data['avg_base_score']:.2f}, "
            f"vcue={data['avg_vcue_score']:.2f}, "
            f"gain=+{data['improvement']:.2f}"
        )


if __name__ == "__main__":
    main()
