"""Small google-genai wrapper with call accounting and 429 backoff."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any

from config.settings import GOOGLE_API_KEY, MAX_LLM_CALLS_PER_QUESTION, MODEL_ID


class LLMError(RuntimeError):
    pass


class LLMRateLimitError(LLMError):
    pass


@dataclass
class QuestionLLM:
    api_key: str = GOOGLE_API_KEY
    model: str = MODEL_ID
    max_calls: int = MAX_LLM_CALLS_PER_QUESTION
    calls: int = 0

    def __post_init__(self) -> None:
        if not self.api_key:
            self.client = None
            return
        from google import genai

        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, *, json_schema: dict[str, Any] | None = None) -> str:
        if self.calls >= self.max_calls:
            raise LLMError(f"Per-question LLM call budget ({self.max_calls}) exceeded")
        if self.client is None:
            raise LLMError("GOOGLE_API_KEY is not configured")
        self.calls += 1
        config: dict[str, Any] = {"temperature": 0, "max_output_tokens": 4096}
        if json_schema:
            config.update({"response_mime_type": "application/json", "response_json_schema": json_schema})
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
                if not response.text:
                    raise LLMError("Gemini returned an empty response")
                return response.text
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                is_rate_limit = status == 429 or "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                if not is_rate_limit:
                    raise LLMError(f"Gemini request failed: {exc}") from exc
                if attempt < 3:
                    time.sleep((2**attempt) + random.random())
        raise LLMRateLimitError(
            "Gemini is rate-limited after automatic retries. Wait about a minute and try again."
        ) from last_error

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> Any:
        try:
            return json.loads(self.generate(prompt, json_schema=schema))
        except json.JSONDecodeError as exc:
            raise LLMError("Gemini returned invalid structured output") from exc

