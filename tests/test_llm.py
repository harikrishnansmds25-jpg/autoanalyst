import json
from types import SimpleNamespace

import pytest

from autoanalyst.agents.messages import CriticOutput, PlannerOutput
from autoanalyst.config import Settings
from autoanalyst.llm import LLMClient, LLMError, MockProvider, inline_schema, make_client
from autoanalyst.llm.gemini import GeminiProvider
from autoanalyst.llm.ollama import OllamaProvider

APPROVE = '{"verdict": "approve", "issues": [], "summary": "fine"}'


def _client(*replies, retries=2):
    return LLMClient(MockProvider({"critic": list(replies)}), max_retries=retries)


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


@pytest.mark.parametrize("model", [PlannerOutput, CriticOutput])
def test_inline_schema_has_no_refs_or_titles(model):
    schema = inline_schema(model)
    assert "$defs" not in schema
    assert not any("$ref" in n or "title" in n for n in _walk(schema))
    assert schema["type"] == "object"


def test_column_is_required_so_decoders_cannot_drop_it():
    step = inline_schema(PlannerOutput)["properties"]["steps"]["items"]
    issue = inline_schema(CriticOutput)["properties"]["issues"]["items"]
    assert "column" in step["required"] and "column" in issue["required"]


def test_valid_reply_first_time():
    client = _client(APPROVE)
    assert client.structured("s", "u", CriticOutput, "critic").verdict == "approve"
    assert client.stats()["calls"] == 1 and client.stats()["failed_calls"] == 0


def test_invalid_json_is_retried_with_the_error():
    client = _client("not json", APPROVE)
    assert client.structured("s", "u", CriticOutput, "critic").verdict == "approve"
    assert [c.ok for c in client.calls] == [False, True]
    retry_prompt = client.provider.prompts[1][2]
    assert "previous reply was rejected" in retry_prompt and "invalid output" in retry_prompt


def test_schema_violation_is_retried():
    blocker_but_approved = json.dumps({
        "verdict": "approve", "summary": "looks fine",
        "issues": [{"column": None, "problem": "bad", "suggestion": "fix it", "severity": "blocker"}],
    })
    client = _client(blocker_but_approved, APPROVE)
    client.structured("s", "u", CriticOutput, "critic")
    assert "blocker" in client.calls[0].error


def test_code_fences_are_accepted():
    client = _client(f"```json\n{APPROVE}\n```")
    assert client.structured("s", "u", CriticOutput, "critic").summary == "fine"


def test_gives_up_after_max_retries():
    client = _client("x", "y", "z", APPROVE)
    with pytest.raises(LLMError, match="3 attempts"):
        client.structured("s", "u", CriticOutput, "critic")
    assert len(client.calls) == 3


def test_provider_errors_are_retried_then_raised():
    client = _client(ConnectionError("down"), ConnectionError("down"), ConnectionError("down"))
    with pytest.raises(LLMError, match="ConnectionError"):
        client.structured("s", "u", CriticOutput, "critic")
    assert all(not c.ok for c in client.calls)


def test_make_client_selects_provider_by_config():
    assert make_client(Settings(_env_file=None, llm_provider="mock")).provider.name == "mock"
    ollama = make_client(Settings(_env_file=None, llm_provider="ollama", ollama_model="qwen2.5:7b"))
    assert ollama.provider.name == "ollama" and ollama.provider.model == "qwen2.5:7b"
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        make_client(Settings(_env_file=None, llm_provider="hosted"))
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        make_client(Settings(_env_file=None, llm_provider="hosted", gemini_api_key=""))
    hosted = make_client(Settings(_env_file=None, llm_provider="hosted", gemini_api_key="k",
                                  hosted_model="gemini-x"))
    assert hosted.provider.name == "hosted" and hosted.provider.model == "gemini-x"


def test_api_key_from_env_is_secret(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(settings)


def test_gemini_request_and_token_counting():
    captured = {}

    def generate_content(model, contents, config):
        captured.update(model=model, contents=contents, config=config)
        usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=30)
        return SimpleNamespace(text=APPROVE, usage_metadata=usage)

    fake = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    provider = GeminiProvider("k", "gemini-x", seed=7, client=fake)
    reply = provider.complete("system text", "user text", {"type": "object"}, 0.1, "critic")
    config = captured["config"]
    assert captured["model"] == "gemini-x" and captured["contents"] == "user text"
    assert config.system_instruction == "system text" and config.temperature == 0.1 and config.seed == 7
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == {"type": "object"}
    assert (reply.input_tokens, reply.output_tokens) == (100, 50)


def test_ollama_request(monkeypatch):
    captured = {}

    def post(url, json, timeout):
        captured.update(url=url, payload=json, timeout=timeout)
        body = {"message": {"content": APPROVE}, "prompt_eval_count": 80, "eval_count": 12}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    monkeypatch.setattr("autoanalyst.llm.ollama.httpx.post", post)
    provider = OllamaProvider("http://host:11434/", "qwen2.5:7b", timeout=30, seed=3, num_ctx=4096)
    reply = provider.complete("sys", "usr", {"type": "object"}, 0.2, "critic")
    payload = captured["payload"]
    assert captured["url"] == "http://host:11434/api/chat"
    assert payload["format"] == {"type": "object"} and payload["stream"] is False
    assert payload["options"] == {"temperature": 0.2, "seed": 3, "num_ctx": 4096}
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert (reply.text, reply.input_tokens, reply.output_tokens) == (APPROVE, 80, 12)
