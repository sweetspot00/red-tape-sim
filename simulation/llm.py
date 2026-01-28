import json
import logging
import os
import random
import re
from typing import Optional, Tuple

from openai import OpenAI, OpenAIError
logger = logging.getLogger(__name__)

class LLMClient:
    """
    Minimal LLM client wrapper (OpenAI-compatible). Configure with env vars:
    - OPENAI_API_KEY (required for OpenAI-compatible APIs)
    - OPENAI_BASE_URL (optional, for self-hosted/compatible APIs)
    - OPENAI_MODEL (defaults to gpt-4o-mini or gpt-3.5-turbo fallback)
    - DEEPSEEK_API_KEY (used automatically when model name indicates DeepSeek)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = "https://aikey-gateway.ivia.ch",
        model: Optional[str] = "azure/gpt-4o",
    ):
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o"
        self.disable_seed = os.getenv("LLM_DISABLE_SEED", "").lower() in {"1", "true", "yes"}
        if self._is_deepseek_model(self.model):
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeek models")
            if base_url in (None, "https://aikey-gateway.ivia.ch"):
                self.base_url = "https://api.deepseek.com"
            else:
                self.base_url = base_url
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise RuntimeError("OPENAI_API_KEY is required for LLM calls")
            self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _model_blocks_seed(self) -> bool:
        return "gemini" in (self.model or "").lower()

    def _resolve_seed(self) -> Optional[int]:
        if self.disable_seed or self._model_blocks_seed():
            return None
        return random.randint(1, 1_000_000_000)

    def generate(self, prompt: str, *, temperature: float = 1) -> str:  # this is not used
        try:
            seed = self._resolve_seed()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are simulating human reactions to public policies."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,# fully sample
                **({} if seed is None else {"seed": seed}),
            )
            return response.choices[0].message.content or ""
        except OpenAIError as exc:
            # Surface a readable error without crashing the caller.
            return f"[LLM error: {exc}]"

    def generate_reaction(
        self,
        prompt: str,
        emotions: Optional[list[str]] = None,
        *,
        temperature: float = 1.0, # fully sample
    ) -> Tuple[dict, str, Optional[int]]:
        """
        Ask the LLM for a single-word emotion from a fixed set.
        Returns (comment, emotion) where both are the chosen word.
        """
        system = (
            "You are simulating a citizen from a country reactions to a region policy. "
            "Your reaction is a mixture of emotions, each feeling should be a probability between 0 and 1."
            "Return a vector of emotions in Json format including all the emotions listed: "
            + ", ".join(emotions)
            + ". No punctuation or extra text."
        )
        logger.info("Prompt for reaction generation: %s", prompt)
        logger.info("System Prompt for reaction generation: %s", system)

        content = ""
        try:
            seed = self._resolve_seed()
            if seed is not None:
                logger.info("Using seed %d for LLM call", seed)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,  # some models (e.g., azure/gpt-5) only support temperature=1
                **({} if seed is None else {"seed": seed}),
            )
            content = (response.choices[0].message.content or "").strip().lower()
            emotion_dict = self._parse_emotion_json(content)
            return emotion_dict, "success", seed

        except OpenAIError as exc:
            logger.warning("LLM request failed: %s", exc)
            return {"error": str(exc)}, "error", seed
        except Exception as exc:
            logger.warning("Failed to parse emotion JSON: %s | content=%r", exc, content)
            return {"raw": content, "error": str(exc)}, "parse_error", seed

    @staticmethod
    def _normalize_emotion(content: str, allowed: list[str]) -> str:
        token = content.split()[0] if content else ""
        if token in allowed:
            return token
        # Try to strip punctuation
        token = "".join(ch for ch in token if ch.isalpha())
        return token if token in allowed else "confusion"

    @staticmethod
    def _parse_emotion_json(content: str) -> dict:
        """
        Be tolerant of Gemini/GPT returning code fences or extra text; extract a JSON object if present.
        """
        if not content:
            return {}
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.lstrip("`")
            cleaned = cleaned.replace("json", "", 1).strip("`\n ")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    @staticmethod
    def _is_deepseek_model(model: str) -> bool:
        lowered = (model or "").lower()
        return "deepseek" in lowered
