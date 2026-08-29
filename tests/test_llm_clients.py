from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.llm_clients import OpenAIResponsesClient, create_llm_client_from_env


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
