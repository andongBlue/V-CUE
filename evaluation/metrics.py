"""
Evaluation metrics for V-CUE experiments.
"""

import json
import re
from typing import Optional

from loguru import logger

from vcue.models.llm_client import LLMClient
from vcue.prompts import CARE_JUDGE_SYSTEM_PROMPT, CARE_JUDGE_USER_PROMPT


def compute_accuracy(predictions: list, labels: list) -> float:
    """Compute accuracy for CulturalBench tasks."""
    assert len(predictions) == len(labels)
    correct = sum(
        1 for pred, label in zip(predictions, labels)
        if normalize_answer(pred) == normalize_answer(label)
    )
    return correct / len(predictions) if predictions else 0.0


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().lower()
    answer = re.sub(r"[^a-z0-9\s]", "", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def compute_correction_rate(
    base_predictions: list,
    enhanced_predictions: list,
    labels: list,
) -> float:
    """
    Compute correction rate (Eq. 6):
    Correction Rate = #(Incorrect -> Correct) / #(Original Incorrect)
    """
    incorrect_indices = [
        i for i, (pred, label) in enumerate(zip(base_predictions, labels))
        if normalize_answer(pred) != normalize_answer(label)
    ]

    if not incorrect_indices:
        return 0.0

    corrected = sum(
        1 for i in incorrect_indices
        if normalize_answer(enhanced_predictions[i]) == normalize_answer(labels[i])
    )

    return corrected / len(incorrect_indices)


def compute_regression_rate(
    base_predictions: list,
    enhanced_predictions: list,
    labels: list,
) -> float:
    """
    Compute regression rate (correct -> incorrect after enhancement).
    """
    correct_indices = [
        i for i, (pred, label) in enumerate(zip(base_predictions, labels))
        if normalize_answer(pred) == normalize_answer(label)
    ]

    if not correct_indices:
        return 0.0

    regressed = sum(
        1 for i in correct_indices
        if normalize_answer(enhanced_predictions[i]) != normalize_answer(labels[i])
    )

    return regressed / len(correct_indices)


class CAREJudge:
    """
    LLM-as-a-Judge evaluator for CARE benchmark.
    Scores responses on generality, cultural relevance, and literary quality.
    """

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o-2024-05-13",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.judge = LLMClient(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=256,
        )

    def evaluate(self, question: str, response: str) -> dict:
        """
        Score a response on the CARE evaluation dimensions.

        Returns:
            Dict with keys: generality, cultural_relevance, literary_quality, average
        """
        user_prompt = CARE_JUDGE_USER_PROMPT.format(
            question=question, response=response
        )
        try:
            raw = self.judge.chat(
                system_prompt=CARE_JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            scores = self._parse_scores(raw)
            scores["average"] = sum(
                scores[k] for k in ("generality", "cultural_relevance", "literary_quality")
            ) / 3.0
            return scores
        except Exception as e:
            logger.warning(f"CARE judge evaluation failed: {e}")
            return {
                "generality": 0, "cultural_relevance": 0,
                "literary_quality": 0, "average": 0,
            }

    @staticmethod
    def _parse_scores(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(text)
