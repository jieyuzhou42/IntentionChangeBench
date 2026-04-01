from __future__ import annotations

import json
import os
import urllib.parse
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


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
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError("Azure OpenAI did not return valid JSON") from exc

        if not isinstance(parsed, dict):
            raise ValueError("Azure OpenAI JSON response was not an object")
        return parsed

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
        if isinstance(response.get("output_text"), str) and response["output_text"].strip():
            return response["output_text"].strip()

        output = response.get("output", [])
        text_parts = []
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "message":
                    continue
                content = item.get("content", [])
                if not isinstance(content, list):
                    continue
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    item_type = content_item.get("type")
                    if item_type == "output_text":
                        text_value = content_item.get("text")
                        if isinstance(text_value, str) and text_value.strip():
                            text_parts.append(text_value.strip())
                    elif item_type == "text":
                        text_value = content_item.get("value")
                        if isinstance(text_value, str) and text_value.strip():
                            text_parts.append(text_value.strip())

        if text_parts:
            return "\n".join(text_parts).strip()

        raise ValueError("Azure OpenAI Responses API returned no text output")

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


__all__ = ["AzureOpenAIChatClient"]
