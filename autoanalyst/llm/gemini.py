from autoanalyst.llm.client import LLMResponse


class GeminiProvider:
    name = "hosted"

    def __init__(self, api_key: str, model: str, timeout: int = 180, seed: int = 42, client=None):
        from google import genai
        from google.genai import types

        self.model = model
        self.seed = seed
        self._types = types
        self._client = client or genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=timeout * 1000)
        )

    def complete(self, system: str, user: str, schema: dict, temperature: float, agent: str) -> LLMResponse:
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            seed=self.seed,
            response_mime_type="application/json",
            response_json_schema=schema,
        )
        reply = self._client.models.generate_content(model=self.model, contents=user, config=config)
        usage = reply.usage_metadata
        output_tokens = 0
        input_tokens = 0
        if usage is not None:
            input_tokens = usage.prompt_token_count or 0
            # Thinking tokens are billed as output.
            output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        return LLMResponse(reply.text or "", input_tokens, output_tokens)

    def list_models(self) -> list[str]:
        return [m.name.removeprefix("models/") for m in self._client.models.list()]
