"""
Unified LLM client supporting OpenAI-compatible APIs (OpenAI, DeepSeek, DashScope, etc.).
"""

import base64
import os
from typing import Optional

from loguru import logger
from openai import OpenAI


class LLMClient:
    """Unified LLM client for text and vision-language models."""

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_key = api_key or self._resolve_api_key(provider)
        resolved_url = base_url or self._resolve_base_url(provider)

        self.client = OpenAI(api_key=resolved_key, base_url=resolved_url)
        logger.info(f"LLMClient initialized: provider={provider}, model={model_name}")

    @staticmethod
    def _resolve_api_key(provider: str) -> str:
        env_map = {
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
        }
        key = os.getenv(env_map.get(provider, "OPENAI_API_KEY"), "")
        if not key:
            logger.warning(f"No API key found for provider '{provider}'")
        return key

    @staticmethod
    def _resolve_base_url(provider: str) -> Optional[str]:
        url_map = {
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        return url_map.get(provider)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a text chat request."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def chat_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a vision-language chat request with a local image."""
        base64_image = self._encode_image(image_path)
        ext = os.path.splitext(image_path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
            ext, "image/png"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{base64_image}"
                        },
                    },
                ],
            },
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
