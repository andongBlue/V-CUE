"""
V-CUE Main Pipeline.
Orchestrates Uncertainty Detection (UD), Visual Generation (VG),
and Visual-based Reconstruction (VR) for culturally enhanced generation.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml
from loguru import logger

from vcue.models.llm_client import LLMClient
from vcue.uncertainty_detection import UncertaintyDetector
from vcue.visual_generation import VisualGenerator
from vcue.visual_reconstruction import VisualReconstructor


@dataclass
class VCUEResult:
    """Result container for a single V-CUE inference."""
    question: str
    base_answer: str
    enhanced_answer: Optional[str] = None
    ud_reliable: Optional[bool] = None
    cultural_cues: Optional[dict] = None
    image_path: Optional[str] = None
    final_answer: str = ""

    def __post_init__(self):
        if not self.final_answer:
            self.final_answer = self.enhanced_answer or self.base_answer


class VCUEPipeline:
    """
    Complete V-CUE pipeline.

    Flow:
    1. Generate base answer with LLM
    2. Uncertainty Detection: UD(Query, Answer)
       - If UD = True:  return base answer
       - If UD = False: proceed to step 3
    3. Visual Generation: extract cues and generate image
    4. Visual-based Reconstruction: F(C, V)
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[dict] = None):
        if config is None:
            config_path = config_path or os.path.join(
                os.path.dirname(__file__), "..", "config", "default.yaml"
            )
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

        self.config = config
        self.system_type = config.get("pipeline", {}).get("system_type", "text")

        self._init_base_llm(config)
        self._init_ud(config)
        self._init_vg(config)
        self._init_vr(config)

        logger.info(f"V-CUE Pipeline initialized (system_type={self.system_type})")

    def _init_base_llm(self, config: dict):
        """Initialize the base LLM for initial answer generation."""
        if self.system_type == "reasoning":
            llm_cfg = config.get("llm_reasoning", config.get("llm", {}))
        elif self.system_type == "vl":
            llm_cfg = config.get("vlm", {})
        else:
            llm_cfg = config.get("llm", {})

        self.base_llm = LLMClient(
            provider=llm_cfg.get("provider", "openai"),
            model_name=llm_cfg.get("model_name", "gpt-3.5-turbo"),
            api_key=llm_cfg.get("api_key"),
            base_url=llm_cfg.get("base_url"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 2048),
        )

    def _init_ud(self, config: dict):
        """Initialize Uncertainty Detection module."""
        ud_cfg = config.get("uncertainty_detection", {})
        self.ud_enabled = ud_cfg.get("enabled", True)
        if self.ud_enabled:
            self.ud = UncertaintyDetector(
                provider=ud_cfg.get("judge_provider", "openai"),
                model_name=ud_cfg.get("judge_model", "gpt-4o"),
                api_key=ud_cfg.get("judge_api_key"),
                base_url=ud_cfg.get("judge_base_url"),
            )
        else:
            self.ud = None

    def _init_vg(self, config: dict):
        """Initialize Visual Generation module."""
        img_cfg = config.get("image_generation", {})
        llm_cfg = config.get("llm", {})

        output_dir = config.get("pipeline", {}).get(
            "image_output_dir", "output/images"
        )
        img_cfg["output_dir"] = output_dir

        gen_config = {
            k: v for k, v in img_cfg.items()
            if k in (
                "model_id", "guidance_scale", "num_inference_steps",
                "height", "width", "use_ema", "device", "seed",
            )
        }
        self.vg = VisualGenerator(
            cue_extractor_provider=llm_cfg.get("provider", "openai"),
            cue_extractor_model=llm_cfg.get("model_name", "gpt-4o"),
            cue_extractor_api_key=llm_cfg.get("api_key"),
            cue_extractor_base_url=llm_cfg.get("base_url"),
            image_gen_config=gen_config,
        )
        self.vg.output_dir = output_dir

    def _init_vr(self, config: dict):
        """Initialize Visual Reconstruction module."""
        if self.system_type in ("text", "reasoning"):
            llm_key = "llm_reasoning" if self.system_type == "reasoning" else "llm"
            llm_cfg = config.get(llm_key, config.get("llm", {}))
            cap_cfg = config.get("image_captioner", {})
            self.vr = VisualReconstructor(
                system_type=self.system_type,
                llm_config={
                    "provider": llm_cfg.get("provider", "openai"),
                    "model_name": llm_cfg.get("model_name"),
                    "api_key": llm_cfg.get("api_key"),
                    "base_url": llm_cfg.get("base_url"),
                    "temperature": llm_cfg.get("temperature", 0.0),
                    "max_tokens": llm_cfg.get("max_tokens", 2048),
                },
                captioner_config={
                    "provider": cap_cfg.get("provider", "dashscope"),
                    "model_name": cap_cfg.get("model_name", "qwen2.5-vl-72b-instruct"),
                    "api_key": cap_cfg.get("api_key"),
                    "base_url": cap_cfg.get("base_url"),
                },
            )
        else:
            vlm_cfg = config.get("vlm", {})
            self.vr = VisualReconstructor(
                system_type="vl",
                vlm_config={
                    "provider": vlm_cfg.get("provider", "dashscope"),
                    "model_name": vlm_cfg.get("model_name"),
                    "api_key": vlm_cfg.get("api_key"),
                    "base_url": vlm_cfg.get("base_url"),
                    "temperature": vlm_cfg.get("temperature", 0.0),
                    "max_tokens": vlm_cfg.get("max_tokens", 2048),
                },
            )

    def run(
        self,
        question: str,
        sample_id: str = "0",
        seed: Optional[int] = None,
        system_prompt: str = "",
        skip_ud: bool = False,
        skip_vg: bool = False,
        skip_vr: bool = False,
    ) -> VCUEResult:
        """
        Run the full V-CUE pipeline on a single question.

        Args:
            question: The cultural question
            sample_id: Unique ID for image file naming
            seed: Random seed for image generation
            system_prompt: Optional system prompt for the base LLM
            skip_ud: Ablation flag - skip uncertainty detection
            skip_vg: Ablation flag - skip visual generation
            skip_vr: Ablation flag - skip visual reconstruction

        Returns:
            VCUEResult with all intermediate and final outputs
        """
        # Step 1: Generate base answer
        base_answer = self._generate_base_answer(question, system_prompt)
        result = VCUEResult(question=question, base_answer=base_answer)

        # Step 2: Uncertainty Detection
        if self.ud_enabled and not skip_ud:
            reliable = self.ud.detect(question, base_answer)
            result.ud_reliable = reliable
            if reliable:
                result.final_answer = base_answer
                logger.info(f"UD=True, returning base answer for sample {sample_id}")
                return result
        else:
            result.ud_reliable = False

        # Step 3: Visual Generation (if not skipped)
        if not skip_vg:
            image_path, cues = self.vg.generate_image(
                question=question,
                sample_id=sample_id,
                seed=seed,
            )
            result.image_path = image_path
            result.cultural_cues = cues
        else:
            logger.info("VG skipped (ablation)")
            result.final_answer = base_answer
            return result

        # Step 4: Visual-based Reconstruction (if not skipped)
        if not skip_vr:
            enhanced = self.vr.reconstruct(
                question=question,
                image_path=image_path,
            )
            result.enhanced_answer = enhanced
            result.final_answer = enhanced
        else:
            logger.info("VR skipped (ablation)")
            result.final_answer = base_answer

        return result

    def _generate_base_answer(self, question: str, system_prompt: str = "") -> str:
        """Generate the initial answer using the base LLM."""
        sys_prompt = system_prompt or "You are a helpful assistant."
        return self.base_llm.chat(
            system_prompt=sys_prompt,
            user_prompt=question,
        )

    def run_always_on(
        self,
        question: str,
        sample_id: str = "0",
        seed: Optional[int] = None,
        system_prompt: str = "",
    ) -> VCUEResult:
        """Run V-CUE without UD gating (always-on mode for analysis)."""
        return self.run(
            question=question,
            sample_id=sample_id,
            seed=seed,
            system_prompt=system_prompt,
            skip_ud=True,
        )
