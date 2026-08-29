from __future__ import annotations

import json
import os
import urllib.parse
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class OpenAIResponsesClient:
    """Minimal client for OpenAI-format Responses APIs."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(
        cls,
        timeout: int = 60,
        provider: str = "openai",
    ) -> "OpenAIResponsesClient":
        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            api_key_name = "DEEPSEEK_API_KEY"
            model_name = "DEEPSEEK_MODEL"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_MODEL")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            api_key_name = "OPENAI_API_KEY"
            model_name = "OPENAI_MODEL"
        if not api_key or not model:
            missing = []
            if not api_key:
                missing.append(api_key_name)
            if not model:
                missing.append(model_name)
            raise ValueError("Missing OpenAI-compatible settings: " + ", ".join(missing))
        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        raw_text = self.generate_json_text(prompt)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenAI did not return valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI JSON response was not an object")
        return parsed

    def generate_json_text(self, prompt: str) -> str:
        return self._completion(prompt, json_mode=True)

    def generate_text(self, prompt: str) -> str:
        return self._completion(prompt, json_mode=False)

    def _completion(self, prompt: str, json_mode: bool) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "store": False,
            "text": {
                "format": {"type": "json_object" if json_mode else "text"}
            },
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return _extract_responses_text(result, provider="OpenAI-compatible")


class AzureOpenAIChatClient:
    """
    Minimal Azure OpenAI chat client that matches the simulator's injected
    `generate_json` / `generate_text` interface.

    Expected environment variables:
    - AZURE_OPENAI_API_KEY
    - Either:
      - AZURE_OPENAI_RESPONSES_ENDPOINT
      - AZURE_OPENAI_DEPLOYMENT
    - Or:
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_CHAT_DEPLOYMENT
    - AZURE_OPENAI_API_VERSION (optional)
    """

    def __init__(
        self,
        api_key: str,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        responses_endpoint: Optional[str] = None,
        api_version: str = "2024-10-21",
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/") if endpoint else None
        self.deployment = deployment
        self.responses_endpoint = responses_endpoint.strip() if responses_endpoint else None
        self.api_version = api_version
        self.timeout = timeout

        if not self.responses_endpoint and (not self.endpoint or not self.deployment):
            raise ValueError(
                "AzureOpenAIChatClient requires either "
                "`responses_endpoint + deployment` or `endpoint + deployment`."
            )

    @classmethod
    def from_env(
        cls,
        api_version: Optional[str] = None,
        timeout: int = 60,
    ) -> "AzureOpenAIChatClient":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        responses_endpoint = os.getenv("AZURE_OPENAI_RESPONSES_ENDPOINT")
        deployment = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        )
        resolved_api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"

        missing = []
        if not api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not deployment:
            missing.append("AZURE_OPENAI_DEPLOYMENT or AZURE_OPENAI_CHAT_DEPLOYMENT")
        if not responses_endpoint and not endpoint:
            missing.append("AZURE_OPENAI_RESPONSES_ENDPOINT or AZURE_OPENAI_ENDPOINT")
        if missing:
            raise ValueError(
                "Missing Azure OpenAI settings: " + ", ".join(missing)
            )

        return cls(
            api_key=api_key,
            endpoint=endpoint,
            deployment=deployment,
            responses_endpoint=responses_endpoint,
            api_version=resolved_api_version,
            timeout=timeout,
        )

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        raw_text = self.generate_json_text(prompt)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError("Azure OpenAI did not return valid JSON") from exc

        if not isinstance(parsed, dict):
            raise ValueError("Azure OpenAI JSON response was not an object")
        return parsed

    def generate_json_text(self, prompt: str) -> str:
        if self.responses_endpoint:
            raw_text = self._responses_completion(
                prompt=prompt,
                temperature=0.1,
                json_mode=True,
            )
        else:
            raw_text = self._chat_completion(
                prompt=prompt,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        return raw_text

    def generate_text(self, prompt: str) -> str:
        if self.responses_endpoint:
            return self._responses_completion(
                prompt=prompt,
                temperature=0.7,
                json_mode=False,
            )
        return self._chat_completion(
            prompt=prompt,
            temperature=0.7,
            response_format=None,
        )

    def _responses_completion(
        self,
        prompt: str,
        temperature: float,
        json_mode: bool,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.deployment,
            "input": prompt,
            "temperature": temperature,
            "store": False,
        }
        if json_mode:
            payload["text"] = {"format": {"type": "json_object"}}
        else:
            payload["text"] = {"format": {"type": "text"}}

        response = self._post_to_url(self._build_responses_url(), payload)
        return self._extract_responses_text(response)

    def _chat_completion(
        self,
        prompt: str,
        temperature: float,
        response_format: Optional[Dict[str, Any]],
    ) -> str:
        payload: Dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            response = self._post(payload)
        except urllib.error.HTTPError as exc:
            if response_format is not None and exc.code in {400, 404}:
                fallback_payload = dict(payload)
                fallback_payload.pop("response_format", None)
                response = self._post(fallback_payload)
            else:
                raise

        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return self._normalize_content(content)

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )
        return self._post_to_url(url, payload)

    def _post_to_url(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "api-key": self.api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _build_responses_url(self) -> str:
        if not self.responses_endpoint:
            raise ValueError("responses_endpoint is not configured")

        parsed = urllib.parse.urlsplit(self.responses_endpoint)
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if not any(key == "api-version" for key, _ in query):
            query.append(("api-version", self.api_version))
        rebuilt = parsed._replace(query=urllib.parse.urlencode(query))
        return urllib.parse.urlunsplit(rebuilt)

    def _extract_responses_text(self, response: Dict[str, Any]) -> str:
        return _extract_responses_text(response, provider="Azure OpenAI")

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(part.strip() for part in parts if part and part.strip()).strip()

        return str(content).strip()


def _extract_responses_text(response: Dict[str, Any], provider: str) -> str:
    if isinstance(response.get("output_text"), str) and response["output_text"].strip():
        return response["output_text"].strip()

    text_parts = []
    output = response.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") == "output_text":
                    text_value = content_item.get("text")
                elif content_item.get("type") == "text":
                    text_value = content_item.get("value")
                else:
                    continue
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value.strip())

    if text_parts:
        return "\n".join(text_parts).strip()
    raise ValueError(f"{provider} Responses API returned no text output")


def create_llm_client_from_env(
    *,
    azure_api_version: Optional[str] = None,
    timeout: int = 60,
) -> Any:
    """Create the configured LLM client without changing caller prompts.

    LLM_PROVIDER may be ``deepseek``, ``openai``, or ``azure``. In auto mode,
    DeepSeek/OpenAI keys take priority over the existing Azure configuration.
    """
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    if provider not in {"auto", "deepseek", "openai", "azure"}:
        raise ValueError("LLM_PROVIDER must be one of: auto, deepseek, openai, azure")
    if provider == "deepseek" or (provider == "auto" and os.getenv("DEEPSEEK_API_KEY")):
        return OpenAIResponsesClient.from_env(timeout=timeout, provider="deepseek")
    if provider == "openai" or (provider == "auto" and os.getenv("OPENAI_API_KEY")):
        return OpenAIResponsesClient.from_env(timeout=timeout, provider="openai")
    return AzureOpenAIChatClient.from_env(
        api_version=azure_api_version,
        timeout=timeout,
    )


__all__ = [
    "AzureOpenAIChatClient",
    "OpenAIResponsesClient",
    "create_llm_client_from_env",
]
