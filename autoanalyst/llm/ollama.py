import httpx

from autoanalyst.llm.client import LLMResponse


class OllamaProvider:
    name = "ollama"

    def __init__(self, host: str, model: str, timeout: int = 180, seed: int = 42, num_ctx: int = 8192):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.seed = seed
        self.num_ctx = num_ctx

    def complete(self, system: str, user: str, schema: dict, temperature: float, agent: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "options": {"temperature": temperature, "seed": self.seed, "num_ctx": self.num_ctx},
        }
        reply = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        reply.raise_for_status()
        data = reply.json()
        return LLMResponse(
            text=data["message"]["content"],
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    def available(self) -> bool:
        try:
            tags = httpx.get(f"{self.host}/api/tags", timeout=5).json()
        except httpx.HTTPError:
            return False
        return any(m["name"] in (self.model, f"{self.model}:latest") for m in tags.get("models", []))
