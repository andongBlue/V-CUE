"""
Visual Generation (VG) module.
Extracts culturally salient elements from input text and constructs
a culture-aware condition unit: c = [region] + [object] + [symbol].
Then generates a culturally relevant image: V = Text-to-Image(c).
"""

import json
import os
from typing import Dict, Optional

from loguru import logger

from vcue.models.llm_client import LLMClient
from vcue.models.image_generator import ImageGenerator
from vcue.prompts import (
    CUE_EXTRACTION_SYSTEM_PROMPT,
    CUE_EXTRACTION_USER_PROMPT,
    NEUTRAL_CUE_SYSTEM_PROMPT,
    NEUTRAL_CUE_USER_PROMPT,
)


class VisualGenerator:
    """Extracts cultural cues and generates culturally relevant images."""

    def __init__(
        self,
        cue_extractor_provider: str = "openai",
        cue_extractor_model: str = "gpt-4o",
        cue_extractor_api_key: Optional[str] = None,
        cue_extractor_base_url: Optional[str] = None,
        image_gen_config: Optional[dict] = None,
    ):
        self.cue_extractor = LLMClient(
            provider=cue_extractor_provider,
            model_name=cue_extractor_model,
            api_key=cue_extractor_api_key,
            base_url=cue_extractor_base_url,
            temperature=0.0,
            max_tokens=256,
        )

        gen_cfg = image_gen_config or {}
        self.image_generator = ImageGenerator(**gen_cfg)
        self.output_dir = gen_cfg.get("output_dir", "output/images")

        logger.info("VisualGenerator initialized")

    def extract_cultural_cues(self, question: str) -> Dict[str, str]:
        """
        Extract cultural cue elements from the question text.
        Parsing is restricted to question-only input to prevent label leakage.

        Args:
            question: The cultural question text

        Returns:
            Dict with keys: region, object, symbol
        """
        user_prompt = CUE_EXTRACTION_USER_PROMPT.format(question=question)

        try:
            response = self.cue_extractor.chat(
                system_prompt=CUE_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            cues = self._parse_json(response)
            logger.debug(f"Extracted cultural cues: {cues}")
            return cues
        except Exception as e:
            logger.warning(f"Cultural cue extraction failed: {e}")
            return {"region": "None", "object": "None", "symbol": "None"}

    def build_condition_unit(self, cues: Dict[str, str]) -> str:
        """
        Construct the conditioning text: c = [region] + [object] + [symbol].

        Args:
            cues: Dict with extracted cultural elements

        Returns:
            Concatenated condition string for image generation
        """
        parts = []
        for key in ["region", "object", "symbol"]:
            val = cues.get(key, "None")
            if val and val.lower() != "none":
                parts.append(val)

        condition = ", ".join(parts) if parts else "cultural scene"
        logger.debug(f"Condition unit: {condition}")
        return condition

    def generate_image(
        self,
        question: str,
        sample_id: str = "0",
        seed: Optional[int] = None,
        mode: str = "cultural",
    ) -> tuple:
        """
        Full VG pipeline: extract cues -> build condition -> generate image.

        Args:
            question: The cultural question text
            sample_id: Unique identifier for naming the output file
            seed: Random seed for reproducibility
            mode: "cultural" (default), "neutral", or "unrelated"

        Returns:
            Tuple of (image_path, cultural_cues_dict)
        """
        cues = self.extract_cultural_cues(question)

        if mode == "neutral":
            cues = self._neutralize_cues(cues)
        elif mode == "unrelated":
            cues = {"region": "random location", "object": "random item", "symbol": "random pattern"}

        condition = self.build_condition_unit(cues)

        output_path = os.path.join(self.output_dir, f"{mode}_{sample_id}.png")
        image_path = self.image_generator.generate(
            prompt=condition,
            output_path=output_path,
            seed=seed,
        )

        return image_path, cues

    def _neutralize_cues(self, cues: Dict[str, str]) -> Dict[str, str]:
        """Remove culturally specific elements (Appendix C)."""
        user_prompt = NEUTRAL_CUE_USER_PROMPT.format(
            region=cues.get("region", "None"),
            object=cues.get("object", "None"),
            symbol=cues.get("symbol", "None"),
        )
        try:
            response = self.cue_extractor.chat(
                system_prompt=NEUTRAL_CUE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            return self._parse_json(response)
        except Exception as e:
            logger.warning(f"Cue neutralization failed: {e}")
            return {"region": "urban area", "object": "generic item", "symbol": "geometric symbol"}

    @staticmethod
    def _parse_json(text: str) -> Dict[str, str]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(text)
