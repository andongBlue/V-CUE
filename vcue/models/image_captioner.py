"""
Image captioning module for the Image-to-Text Mediation strategy.
Converts generated images into textual descriptions: Cap(V).
Used in the LLM-text pathway of V-CUE (Eq. 4).
"""

from typing import Optional

from loguru import logger

from vcue.models.llm_client import LLMClient
from vcue.prompts import CAPTION_SYSTEM_PROMPT, CAPTION_USER_PROMPT


class ImageCaptioner:
    """Image-to-text captioner using a vision-language model."""

    def __init__(
        self,
        provider: str = "dashscope",
        model_name: str = "qwen2.5-vl-72b-instruct",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.vlm = LLMClient(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=1024,
        )
        logger.info(f"ImageCaptioner initialized with {model_name}")

    def caption(self, image_path: str) -> str:
        """
        Generate a culturally-aware textual description of the image.

        Args:
            image_path: Path to the generated cultural image

        Returns:
            Textual description encoding cultural visual cues (Cap(V))
        """
        description = self.vlm.chat_with_image(
            system_prompt=CAPTION_SYSTEM_PROMPT,
            user_prompt=CAPTION_USER_PROMPT,
            image_path=image_path,
        )
        logger.debug(f"Caption generated for {image_path}: {description[:100]}...")
        return description
