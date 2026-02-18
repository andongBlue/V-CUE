"""
Visual-based Generation Reconstruction (VR) module.
Integrates generated visual cues into the LLM generation process.

Two pathways:
  - LLM-text: Image -> Caption -> Text concat (Eq. 4)
  - LLM-VL:   Image fed directly to multimodal LLM (Eq. 5)
"""

from typing import Optional

from loguru import logger

from vcue.models.llm_client import LLMClient
from vcue.models.image_captioner import ImageCaptioner
from vcue.prompts import (
    VR_TEXT_SYSTEM_PROMPT,
    VR_TEXT_USER_PROMPT,
    VR_VL_SYSTEM_PROMPT,
    VR_VL_USER_PROMPT,
)


class VisualReconstructor:
    """Reconstructs LLM output using visual cultural cues."""

    def __init__(
        self,
        system_type: str = "text",
        llm_config: Optional[dict] = None,
        captioner_config: Optional[dict] = None,
        vlm_config: Optional[dict] = None,
    ):
        """
        Args:
            system_type: "text" for LLM-text/reasoning, "vl" for LLM-VL
            llm_config: Config for the text LLM
            captioner_config: Config for the image captioner (text pathway)
            vlm_config: Config for the vision-language model (VL pathway)
        """
        self.system_type = system_type

        if system_type in ("text", "reasoning"):
            llm_cfg = llm_config or {}
            self.llm = LLMClient(**llm_cfg)
            cap_cfg = captioner_config or {}
            self.captioner = ImageCaptioner(**cap_cfg)
            logger.info(f"VR initialized for {system_type} pathway (with captioner)")
        elif system_type == "vl":
            vlm_cfg = vlm_config or {}
            self.vlm = LLMClient(**vlm_cfg)
            logger.info("VR initialized for VL pathway (direct image input)")
        else:
            raise ValueError(f"Unknown system_type: {system_type}")

    def reconstruct(
        self,
        question: str,
        image_path: str,
    ) -> str:
        """
        Reconstruct the answer using visual cultural cues.
        Answer = F(C, V) where C is the textual context and V is the image.

        Args:
            question: The original question (textual context C)
            image_path: Path to the generated cultural image (V)

        Returns:
            The reconstructed answer
        """
        if self.system_type in ("text", "reasoning"):
            return self._text_pathway(question, image_path)
        else:
            return self._vl_pathway(question, image_path)

    def _text_pathway(self, question: str, image_path: str) -> str:
        """
        LLM-text pathway (Eq. 4):
        F(C, V) = LLM_text(C ⊕ Cap(V))
        """
        caption = self.captioner.caption(image_path)
        logger.debug(f"Image caption: {caption[:100]}...")

        user_prompt = VR_TEXT_USER_PROMPT.format(
            question=question,
            caption=caption,
        )
        answer = self.llm.chat(
            system_prompt=VR_TEXT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        return answer

    def _vl_pathway(self, question: str, image_path: str) -> str:
        """
        LLM-VL pathway (Eq. 5):
        F(C, V) = LLM_VL(C, V)
        """
        answer = self.vlm.chat_with_image(
            system_prompt=VR_VL_SYSTEM_PROMPT,
            user_prompt=VR_VL_USER_PROMPT.format(question=question),
            image_path=image_path,
        )
        return answer
