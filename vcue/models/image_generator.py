"""
Image generation module using Stable Diffusion or other text-to-image models.
Implements the Visual Generation (VG) component: V = Text-to-Image(c).
"""

import os
from typing import Optional

import torch
from loguru import logger


class ImageGenerator:
    """Text-to-image generation using Stable Diffusion."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        height: int = 512,
        width: int = 512,
        use_ema: bool = True,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.model_id = model_id
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.use_ema = use_ema
        self.device = device
        self.seed = seed
        self.pipe = None

    def load_model(self):
        """Lazy-load the Stable Diffusion pipeline."""
        if self.pipe is not None:
            return

        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

        logger.info(f"Loading image generation model: {self.model_id}")

        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16,
        )

        if self.use_ema:
            self.pipe.unet.load_attn_procs(self.model_id)
            logger.info("EMA weights loaded for higher visual quality")

        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        logger.info("Image generation model loaded successfully")

    def generate(
        self,
        prompt: str,
        output_path: str,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate a culturally relevant image from the condition unit.

        Args:
            prompt: Cultural condition text (c = [region] + [object] + [symbol])
            output_path: Path to save the generated image
            seed: Random seed for reproducibility

        Returns:
            Path to the generated image
        """
        self.load_model()

        actual_seed = seed if seed is not None else self.seed
        generator = torch.Generator(device=self.device).manual_seed(actual_seed)

        image = self.pipe(
            prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            height=self.height,
            width=self.width,
            generator=generator,
        ).images[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        logger.debug(f"Generated image saved to {output_path}")
        return output_path
