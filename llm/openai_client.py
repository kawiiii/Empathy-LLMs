from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
from openai import OpenAI


@dataclass
class LLMResponse:
    text: str
    raw: object | None = None


class OpenAIResponsesClient:
    """
    Minimal wrapper around OpenAI Responses API.

    Future extension:
    - Add other providers / open-source models behind the same interface.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_output_tokens: int = 300,
    ) -> LLMResponse:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = self.client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return LLMResponse(text=resp.output_text, raw=resp)

    def judge(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 64,
    ) -> LLMResponse:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = self.client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return LLMResponse(text=resp.output_text.strip(), raw=resp)