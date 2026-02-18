"""
Uncertainty Detection (UD) module.
LLM-based judge that evaluates whether a generated response
is culturally reliable, following the LLM-as-Judge paradigm.

If UD = False, the Visual Generation module is activated.
If UD = True, the response is returned directly.
"""

from typing import Optional

from loguru import logger

from vcue.models.llm_client import LLMClient
from vcue.prompts import UD_SYSTEM_PROMPT, UD_USER_PROMPT


class UncertaintyDetector:
    """Estimates response uncertainty for cultural contexts."""

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.judge = LLMClient(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=16,
        )
        logger.info(f"UncertaintyDetector initialized with {model_name}")

    def detect(self, question: str, answer: str) -> bool:
        """
        Evaluate whether the generated answer is culturally reliable.

        Args:
            question: The original cultural question
            answer: The model-generated answer

        Returns:
            True if the answer is reliable (no enhancement needed),
            False if uncertain (Visual Generation should be triggered)
        """
        user_prompt = UD_USER_PROMPT.format(question=question, answer=answer)

        try:
            response = self.judge.chat(
                system_prompt=UD_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            result = self._parse_response(response)
            logger.debug(
                f"UD result: {result} | raw='{response}' | "
                f"question='{question[:60]}...'"
            )
            return result
        except Exception as e:
            logger.warning(f"UD detection failed: {e}. Defaulting to uncertain.")
            return False

    @staticmethod
    def _parse_response(response: str) -> bool:
        cleaned = response.strip().lower().rstrip(".")
        if cleaned == "true":
            return True
        if cleaned == "false":
            return False
        # Fallback: treat ambiguous as uncertain to trigger enhancement
        logger.warning(f"Ambiguous UD response: '{response}'. Treating as uncertain.")
        return False
