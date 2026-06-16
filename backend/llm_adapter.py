import httpx
import json
import logging
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse
from config import settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.base_url = {
            "ollama": f"{settings.ollama_base_url}/v1",
            "lmstudio": f"{settings.lmstudio_base_url}/v1",
            "openai": "https://api.openai.com/v1",
        }.get(self.provider)
        self.api_key = settings.llm_api_key or "not-needed"

    async def chat(self, messages: list[dict]) -> str:
        """Standard blocking chat request"""
        if self.provider == "anthropic":
            return await self._chat_anthropic(messages)

        async with httpx.AsyncClient(timeout=120) as client:
            try:
                r = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": messages, "stream": False},
                )
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                return self._strip_reasoning(content)
            except Exception as e:
                logger.error(f"LLM API request failed: {e}")
                raise

    async def chat_stream(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """Streaming chat response via async generator"""
        if self.provider == "anthropic":
            async for chunk in self._chat_stream_anthropic(messages):
                yield chunk
            return

        async with httpx.AsyncClient(timeout=120) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": messages, "stream": True},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or line.strip() == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    yield f"data: {json.dumps({'content': delta['content']})}\n\n"
                            except Exception as parse_err:
                                logger.warning(f"Error parsing SSE line: {line} - {parse_err}")
            except Exception as e:
                logger.error(f"Streaming LLM API request failed: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

    async def _chat_anthropic(self, messages: list[dict]) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self.model,
                    "max_tokens": 1024,
                    "system": system,
                    "messages": user_messages,
                },
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]
            
    async def _chat_stream_anthropic(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]
        
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self.model,
                    "max_tokens": 1024,
                    "system": system,
                    "messages": user_messages,
                    "stream": True
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            delta = data["delta"]
                            if delta.get("type") == "text_delta":
                                yield f"data: {json.dumps({'content': delta['text']})}\n\n"
                    except:
                        pass
                        
    @staticmethod
    def _strip_reasoning(content: str) -> str:
        if "<think>" in content and "</think>" in content:
            return content.split("</think>")[-1].strip()
        return content