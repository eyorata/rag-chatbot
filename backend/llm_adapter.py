import httpx
from config import settings

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
        if self.provider == "anthropic":
            return await self._chat_anthropic(messages)

        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages, "stream": False},
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return self._strip_reasoning(content)

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

    @staticmethod
    def _strip_reasoning(content: str) -> str:
        if "<think>" in content and "</think>" in content:
            return content.split("</think>")[-1].strip()
        return content