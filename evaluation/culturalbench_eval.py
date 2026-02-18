"""
CulturalBench evaluation script.
Evaluates V-CUE on Multiple-Choice and True/False cultural knowledge tasks.
"""

import json
import os
from typing import Optional

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from evaluation.metrics import compute_accuracy, compute_correction_rate, compute_regression_rate
from vcue.pipeline import VCUEPipeline
from vcue.prompts import CULTURALBENCH_MC_PROMPT, CULTURALBENCH_TF_PROMPT


class CulturalBenchEvaluator:
    """Evaluator for the CulturalBench benchmark."""

    def __init__(
        self,
        pipeline: VCUEPipeline,
        dataset_name: str = "kellycyy/CulturalBench",
        output_dir: str = "output/results/culturalbench",
    ):
        self.pipeline = pipeline
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self, split: str = "test"):
        """Load CulturalBench dataset from HuggingFace."""
        logger.info(f"Loading CulturalBench dataset: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, split=split)
        return dataset

    def evaluate_multiple_choice(
        self,
        dataset=None,
        num_runs: int = 5,
        max_samples: Optional[int] = None,
        ablation: Optional[str] = None,
    ) -> dict:
        """
        Evaluate on Multiple-Choice task.

        Args:
            dataset: Pre-loaded dataset (if None, loads from HF)
            num_runs: Number of evaluation runs for statistical significance
            max_samples: Limit number of samples (for debugging)
            ablation: None, "no_ud", "no_vg", or "no_vr"

        Returns:
            Dict with accuracy results
        """
        if dataset is None:
            dataset = self.load_data()

        mc_data = [ex for ex in dataset if ex.get("task_type") == "multiple_choice"]
        if max_samples:
            mc_data = mc_data[:max_samples]

        logger.info(f"Evaluating Multiple-Choice: {len(mc_data)} samples, {num_runs} runs")

        all_accuracies = []
        for run_idx in range(num_runs):
            base_preds, enhanced_preds, labels = [], [], []

            for i, sample in enumerate(tqdm(mc_data, desc=f"MC Run {run_idx + 1}")):
                question = self._format_mc_question(sample)
                label = sample["answer"]

                skip_flags = self._get_ablation_flags(ablation)
                result = self.pipeline.run(
                    question=question,
                    sample_id=f"mc_{i}",
                    seed=42 + i,
                    **skip_flags,
                )

                base_preds.append(result.base_answer)
                enhanced_preds.append(result.final_answer)
                labels.append(label)

            acc = compute_accuracy(enhanced_preds, labels)
            base_acc = compute_accuracy(base_preds, labels)
            correction = compute_correction_rate(base_preds, enhanced_preds, labels)

            all_accuracies.append(acc)
            logger.info(
                f"MC Run {run_idx + 1}: base_acc={base_acc:.4f}, "
                f"vcue_acc={acc:.4f}, correction_rate={correction:.4f}"
            )

        avg_acc = sum(all_accuracies) / len(all_accuracies)
        results = {
            "task": "multiple_choice",
            "num_runs": num_runs,
            "num_samples": len(mc_data),
            "accuracies": all_accuracies,
            "average_accuracy": avg_acc,
            "ablation": ablation,
        }

        self._save_results(results, f"mc_{ablation or 'full'}.json")
        return results

    def evaluate_true_false(
        self,
        dataset=None,
        num_runs: int = 5,
        max_samples: Optional[int] = None,
        ablation: Optional[str] = None,
    ) -> dict:
        """
        Evaluate on True/False task.

        Args:
            dataset: Pre-loaded dataset (if None, loads from HF)
            num_runs: Number of evaluation runs
            max_samples: Limit number of samples
            ablation: None, "no_ud", "no_vg", or "no_vr"

        Returns:
            Dict with accuracy results
        """
        if dataset is None:
            dataset = self.load_data()

        tf_data = [ex for ex in dataset if ex.get("task_type") == "true_false"]
        if max_samples:
            tf_data = tf_data[:max_samples]

        logger.info(f"Evaluating True/False: {len(tf_data)} samples, {num_runs} runs")

        all_accuracies = []
        for run_idx in range(num_runs):
            base_preds, enhanced_preds, labels = [], [], []

            for i, sample in enumerate(tqdm(tf_data, desc=f"TF Run {run_idx + 1}")):
                question = CULTURALBENCH_TF_PROMPT.format(question=sample["question"])
                label = sample["answer"]

                skip_flags = self._get_ablation_flags(ablation)
                result = self.pipeline.run(
                    question=question,
                    sample_id=f"tf_{i}",
                    seed=42 + i,
                    **skip_flags,
                )

                base_preds.append(result.base_answer)
                enhanced_preds.append(result.final_answer)
                labels.append(label)

            acc = compute_accuracy(enhanced_preds, labels)
            base_acc = compute_accuracy(base_preds, labels)

            all_accuracies.append(acc)
            logger.info(
                f"TF Run {run_idx + 1}: base_acc={base_acc:.4f}, vcue_acc={acc:.4f}"
            )

        avg_acc = sum(all_accuracies) / len(all_accuracies)
        results = {
            "task": "true_false",
            "num_runs": num_runs,
            "num_samples": len(tf_data),
            "accuracies": all_accuracies,
            "average_accuracy": avg_acc,
            "ablation": ablation,
        }

        self._save_results(results, f"tf_{ablation or 'full'}.json")
        return results

    @staticmethod
    def _format_mc_question(sample: dict) -> str:
        options_text = "\n".join(
            f"({chr(65 + i)}) {opt}"
            for i, opt in enumerate(sample.get("options", []))
        )
        return CULTURALBENCH_MC_PROMPT.format(
            question=sample["question"],
            options=options_text,
        )

    @staticmethod
    def _get_ablation_flags(ablation: Optional[str]) -> dict:
        flags = {"skip_ud": False, "skip_vg": False, "skip_vr": False}
        if ablation == "no_ud":
            flags["skip_ud"] = True
        elif ablation == "no_vg":
            flags["skip_vg"] = True
        elif ablation == "no_vr":
            flags["skip_vr"] = True
        return flags

    def _save_results(self, results: dict, filename: str):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {path}")
