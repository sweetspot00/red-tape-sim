import json
import os
from typing import Optional, Tuple

from openai import OpenAI, OpenAIError


class LLMClient:
    """
    Minimal LLM client wrapper (OpenAI-compatible). Configure with env vars:
    - OPENAI_API_KEY (required)
    - OPENAI_BASE_URL (optional, for self-hosted/compatible APIs)
    - OPENAI_MODEL (defaults to gpt-4o-mini or gpt-3.5-turbo fallback)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = "https://aikey-gateway.ivia.ch",
        model: Optional[str] = "azure/gpt-5",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM calls")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, *, temperature: float = 1) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are simulating human reactions to public policies."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except OpenAIError as exc:
            # Surface a readable error without crashing the caller.
            return f"[LLM error: {exc}]"

    def generate_reaction(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
    ) -> Tuple[str, str]:
        """
        Ask the LLM for a single-word emotion from a fixed set.
        Returns (comment, emotion) where both are the chosen word.
        """
        emotions = [
            "peace",
            "anger",
            "contempt",
            "fear",
            "disgust",
            "joy",
            "sadness",
            "surprise",
            "confusion",
            "frustration",
        ]
        system = (
            "You are simulating persona reactions to events. "
            "Reply with ONLY ONE WORD, the emotion, lowercase, from this list: "
            + ", ".join(emotions)
            + ". No punctuation or extra text."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,  # some models (e.g., azure/gpt-5) only support temperature=1
            )
            content = (response.choices[0].message.content or "").strip().lower()
            emotion = self._normalize_emotion(content, emotions)
            return emotion, emotion
        except OpenAIError as exc:
            return f"[LLM error: {exc}]", "error"

    @staticmethod
    def _normalize_emotion(content: str, allowed: list[str]) -> str:
        token = content.split()[0] if content else ""
        if token in allowed:
            return token
        # Try to strip punctuation
        token = "".join(ch for ch in token if ch.isalpha())
        return token if token in allowed else "confusion"
