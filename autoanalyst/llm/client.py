"""The single LLM interface. Agents call `LLMClient.structured`; the provider is chosen by config."""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from autoanalyst.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class Provider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, schema: dict, temperature: float, agent: str) -> LLMResponse:
        ...


@dataclass
class CallRecord:
    agent: str
    attempt: int
    ok: bool
    error: str | None
    input_tokens: int
    output_tokens: int
    seconds: float
    provider: str
    model: str


def inline_schema(model: type[BaseModel]) -> dict:
    """JSON schema with $refs resolved and titles removed: the subset every provider accepts."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


@dataclass
class LLMClient:
    provider: Provider
    temperature: float = 0.1
    max_retries: int = 2
    calls: list[CallRecord] = field(default_factory=list)

    def structured(self, system: str, user: str, output: type[T], agent: str) -> T:
        """Ask for JSON matching `output`. Invalid replies are rejected and retried with the error."""
        schema = inline_schema(output)
        prompt = user
        last_error = ""
        for attempt in range(1, self.max_retries + 2):
            started = time.perf_counter()
            response, error = None, None
            try:
                response = self.provider.complete(system, prompt, schema, self.temperature, agent)
                parsed = output.model_validate(json.loads(_strip_fences(response.text)))
            except (json.JSONDecodeError, ValidationError) as exc:
                error = f"invalid output: {exc}"
            except Exception as exc:
                error = f"provider error: {type(exc).__name__}: {exc}"
            self.calls.append(CallRecord(
                agent=agent, attempt=attempt, ok=error is None, error=error,
                input_tokens=response.input_tokens if response else 0,
                output_tokens=response.output_tokens if response else 0,
                seconds=round(time.perf_counter() - started, 3),
                provider=self.provider.name, model=self.provider.model,
            ))
            if error is None:
                return parsed
            last_error = error
            prompt = (
                f"{user}\n\nYour previous reply was rejected ({error[:500]}). "
                "Reply with a single JSON object that matches the schema exactly."
            )
        raise LLMError(f"{agent}: no valid reply after {self.max_retries + 1} attempts; {last_error[:300]}")

    def stats(self) -> dict:
        return {
            "provider": self.provider.name,
            "model": self.provider.model,
            "calls": len(self.calls),
            "failed_calls": sum(not c.ok for c in self.calls),
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "seconds": round(sum(c.seconds for c in self.calls), 2),
        }

    def call_log(self) -> list[dict]:
        return [asdict(c) for c in self.calls]


def make_provider(settings: Settings) -> Provider:
    if settings.llm_provider == "mock":
        from autoanalyst.llm.mock import MockProvider
        return MockProvider()
    if settings.llm_provider == "ollama":
        from autoanalyst.llm.ollama import OllamaProvider
        return OllamaProvider(settings.ollama_host, settings.ollama_model, settings.llm_timeout,
                              settings.seed, settings.ollama_num_ctx)
    if settings.gemini_api_key is None:
        raise LLMError("GEMINI_API_KEY is not set; add it to .env or choose another provider")
    from autoanalyst.llm.gemini import GeminiProvider
    return GeminiProvider(settings.gemini_api_key.get_secret_value(), settings.hosted_model,
                          settings.llm_timeout, settings.seed)


def make_client(settings: Settings | None = None, provider: Provider | None = None) -> LLMClient:
    """Build the client for the configured provider. Raises LLMError if it cannot be set up."""
    settings = settings or get_settings()
    return LLMClient(
        provider=provider or make_provider(settings),
        temperature=settings.llm_temperature,
        max_retries=settings.llm_max_retries,
    )
