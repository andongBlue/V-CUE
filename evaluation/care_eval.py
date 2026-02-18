"""
CARE-eval benchmark evaluation script.
Evaluates V-CUE on culture-specific generation tasks
(Chinese, Japanese, Arabic) using GPT-4o as judge.
"""

import json
import os
from typing import Optional

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from evaluation.metrics import CAREJudge
from vcue.pipeline import VCUEPipeline


class CAREEvaluator:
    """Evaluator for the CARE-eval benchmark."""

    CULTURES = ["chinese", "japanese", "arabic"]
    CATEGORIES_PER_CULTURE = 5
    SAMPLES_PER_CATEGORY = 30

    def __init__(
        self,
        pipeline: VCUEPipeline,
        dataset_name: str = "geyang627/CARE-eval",
        judge_config: Optional[dict] = None,
        output_dir: str = "output/results/care",
    ):
        self.pipeline = pipeline
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        judge_cfg = judge_config or {}
        self.judge = CAREJudge(
            provider=judge_cfg.get("provider", "openai"),
            model_name=judge_cfg.get("model_name", "gpt-4o-2024-05-13"),
            api_key=judge_cfg.get("api_key"),
            base_url=judge_cfg.get("base_url"),
        )

    def load_data(self, split: str = "test"):
        """Load CARE-eval dataset from HuggingFace."""
        logger.info(f"Loading CARE-eval dataset: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, split=split)
        return dataset

    def evaluate(
        self,
        dataset=None,
        num_runs: int = 3,
        max_samples: Optional[int] = None,
        cultures: Optional[list] = None,
    ) -> dict:
        """
        Evaluate on CARE-eval benchmark.

        Args:
            dataset: Pre-loaded dataset
            num_runs: Number of evaluation runs
            max_samples: Limit samples per culture
            cultures: List of cultures to evaluate (default: all)

        Returns:
            Dict with scores per culture
        """
        if dataset is None:
            dataset = self.load_data()

        target_cultures = cultures or self.CULTURES
        all_results = {}

        for culture in target_cultures:
            culture_data = [
                ex for ex in dataset
                if ex.get("culture", "").lower() == culture
            ]
            if max_samples:
                culture_data = culture_data[:max_samples]

            logger.info(f"Evaluating CARE [{culture}]: {len(culture_data)} samples")

            run_scores = []
            for run_idx in range(num_runs):
                scores = self._evaluate_culture(culture_data, culture, run_idx)
                run_scores.append(scores)
                logger.info(
                    f"CARE [{culture}] Run {run_idx + 1}: "
                    f"base_avg={scores['base_average']:.2f}, "
                    f"vcue_avg={scores['vcue_average']:.2f}"
                )

            avg_base = sum(r["base_average"] for r in run_scores) / num_runs
            avg_vcue = sum(r["vcue_average"] for r in run_scores) / num_runs

            all_results[culture] = {
                "runs": run_scores,
                "avg_base_score": avg_base,
                "avg_vcue_score": avg_vcue,
                "improvement": avg_vcue - avg_base,
            }

        self._save_results(all_results, "care_results.json")
        return all_results

    def _evaluate_culture(
        self, culture_data: list, culture: str, run_idx: int
    ) -> dict:
        """Evaluate a single culture in a single run."""
        base_scores, vcue_scores = [], []

        for i, sample in enumerate(tqdm(
            culture_data, desc=f"CARE {culture} Run {run_idx + 1}"
        )):
            question = sample["question"]

            # Base model answer
            base_answer = self.pipeline._generate_base_answer(question)
            base_eval = self.judge.evaluate(question, base_answer)
            base_scores.append(base_eval["average"])

            # V-CUE enhanced answer
            result = self.pipeline.run(
                question=question,
                sample_id=f"care_{culture}_{i}",
                seed=42 + i,
            )
            vcue_eval = self.judge.evaluate(question, result.final_answer)
            vcue_scores.append(vcue_eval["average"])

        return {
            "base_average": sum(base_scores) / len(base_scores) if base_scores else 0,
            "vcue_average": sum(vcue_scores) / len(vcue_scores) if vcue_scores else 0,
            "base_scores": base_scores,
            "vcue_scores": vcue_scores,
        }

    def _save_results(self, results: dict, filename: str):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {path}")
