from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.llm_clients import (
    BedrockConverseClient,
    OpenAIResponsesClient,
    create_llm_client_from_env,
)


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps({"output_text": "API_OK"}).encode("utf-8")


def test_public_openai_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAIResponsesClient("secret", "test-model")

    assert client.generate_text("same prompt") == "API_OK"
    payload = json.loads(captured["request"].data)
    assert payload["model"] == "test-model"
    assert payload["input"] == "same prompt"
    assert captured["request"].get_header("Authorization") == "Bearer secret"


def test_public_openai_refreshes_api_key_once_after_401(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                hdrs=None,
                fp=None,
            )
        return _FakeResponse()

    def fake_run(args, **kwargs):
        assert args == ["refresh-token", "--region", "us-east-1"]
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(args, 0, stdout="fresh-token\n", stderr="")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv(
        "OPENAI_API_KEY_REFRESH_COMMAND",
        "refresh-token --region us-east-1",
    )
    client = OpenAIResponsesClient("expired-token", "test-model")

    assert client.generate_text("same prompt") == "API_OK"
    assert len(requests) == 2
    assert requests[0].get_header("Authorization") == "Bearer expired-token"
    assert requests[1].get_header("Authorization") == "Bearer fresh-token"


def test_public_openai_retries_timeout(monkeypatch):
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary timeout")
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    client = OpenAIResponsesClient("secret", "test-model")

    assert client.generate_text("same prompt") == "API_OK"
    assert calls == 2


def test_factory_selects_public_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    client = create_llm_client_from_env()
    assert isinstance(client, OpenAIResponsesClient)


def test_factory_selects_deepseek_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    client = create_llm_client_from_env()
    assert isinstance(client, OpenAIResponsesClient)
    assert client.model == "deepseek-v4-flash"
    assert client.base_url == "https://api.deepseek.com"


class _FakeBedrockClient:
    def __init__(self, text):
        self.text = text
        self.requests = []

    def converse(self, **request):
        self.requests.append(request)
        return {
            "output": {
                "message": {
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": "hidden"}}},
                        {"text": self.text},
                    ]
                }
            }
        }


def test_bedrock_converse_request_and_json_parsing(monkeypatch):
    monkeypatch.delenv("BEDROCK_TEMPERATURE", raising=False)
    fake_client = _FakeBedrockClient('```json\n{"status": "ok"}\n```')
    client = BedrockConverseClient(
        model="us.openai.gpt-5.6-sol",
        max_tokens=1234,
        client=fake_client,
    )

    assert client.generate_json("same prompt") == {"status": "ok"}
    request = fake_client.requests[0]
    assert request["modelId"] == "us.openai.gpt-5.6-sol"
    assert request["messages"][0]["content"] == [{"text": "same prompt"}]
    assert request["inferenceConfig"] == {"maxTokens": 1234}


def test_factory_selects_bedrock(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setattr(
        BedrockConverseClient,
        "from_env",
        lambda timeout=60: sentinel,
    )

    assert create_llm_client_from_env() is sentinel
