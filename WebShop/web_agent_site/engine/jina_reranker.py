from __future__ import annotations

import json
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}


@dataclass(frozen=True)
class JinaRerankConfig:
    api_key: str
    model: str = "jina-reranker-v3"
    endpoint: str = "https://api.jina.ai/v1/rerank"
    top_n: int = 50
    timeout: float = 30.0
    truncate: bool = True
    max_doc_chars: int = 1200
    cache_size: int = 256
    max_retries: int = 3
    retry_backoff: float = 1.0
    min_interval: float = 0.0
    log_path: str = ""
    verbose: bool = False


def build_jina_rerank_config_from_env() -> Optional[JinaRerankConfig]:
    mode = os.getenv("WEBSHOP_JINA_RERANK", "auto").strip().lower()
    if mode in FALSE_VALUES:
        return None

    api_key = os.getenv("JINA_API_KEY", "").strip()
    if not api_key:
        if mode in TRUE_VALUES:
            raise ValueError("WEBSHOP_JINA_RERANK is enabled but JINA_API_KEY is not set.")
        return None

    return JinaRerankConfig(
        api_key=api_key,
        model=(
            os.getenv("WEBSHOP_JINA_RERANK_MODEL")
            or os.getenv("JINA_RERANKER_MODEL")
            or "jina-reranker-v3"
        ).strip(),
        endpoint=os.getenv("WEBSHOP_JINA_RERANK_ENDPOINT", "https://api.jina.ai/v1/rerank").strip(),
        top_n=_env_int("WEBSHOP_JINA_RERANK_TOP_N", 50),
        timeout=_env_float("WEBSHOP_JINA_RERANK_TIMEOUT", 30.0),
        truncate=_env_bool("WEBSHOP_JINA_RERANK_TRUNCATE", True),
        max_doc_chars=_env_int("WEBSHOP_JINA_RERANK_MAX_DOC_CHARS", 1200),
        cache_size=_env_int("WEBSHOP_JINA_RERANK_CACHE_SIZE", 256),
        max_retries=_env_int("WEBSHOP_JINA_RERANK_MAX_RETRIES", 3),
        retry_backoff=_env_float("WEBSHOP_JINA_RERANK_RETRY_BACKOFF", 1.0),
        min_interval=_env_float("WEBSHOP_JINA_RERANK_MIN_INTERVAL", 0.0),
        log_path=os.getenv("WEBSHOP_JINA_RERANK_LOG_PATH", "").strip(),
        verbose=_env_bool("WEBSHOP_JINA_RERANK_VERBOSE", False),
    )


def maybe_wrap_with_jina_reranker(searcher: Any) -> Any:
    config = build_jina_rerank_config_from_env()
    if config is None:
        return searcher
    return JinaRerankSearcher(searcher, config=config)


class JinaRerankSearcher:
    """
    Drop-in wrapper for Pyserini's LuceneSearcher.

    BM25 still performs the first-stage retrieval. Only the first configured
    number of hits are sent to Jina and reordered; .doc() and all other searcher
    methods are delegated to the wrapped LuceneSearcher.
    """

    def __init__(self, searcher: Any, config: JinaRerankConfig, session: Optional[requests.Session] = None):
        self._searcher = searcher
        self.config = config
        self._session = session or requests.Session()
        self._cache: Dict[Tuple[str, int, Tuple[str, ...]], List[Any]] = {}
        self._cache_order: List[Tuple[str, int, Tuple[str, ...]]] = []
        self._lock = threading.Lock()
        self._warned_failure = False
        self._request_count = 0
        self._cache_hit_count = 0
        self._last_request_time = 0.0

    def search(self, query: str, k: int = 10, **kwargs: Any) -> List[Any]:
        hits = list(self._searcher.search(query, k=k, **kwargs))
        if not hits:
            return hits

        rerank_count = min(len(hits), max(0, self.config.top_n), k)
        if rerank_count <= 1:
            return hits

        docids = tuple(str(hit.docid) for hit in hits[:rerank_count])
        cache_key = (str(query), rerank_count, docids)
        cached = self._get_cached(cache_key)
        if cached is not None:
            with self._lock:
                self._cache_hit_count += 1
                cache_hit_count = self._cache_hit_count
            self._log_event(
                {
                    "event": "cache_hit",
                    "cache_hit_count": cache_hit_count,
                    "query": str(query),
                    "k": k,
                    "rerank_count": rerank_count,
                    "caller": _caller_summary(),
                }
            )
            return list(cached) + hits[rerank_count:]

        with self._lock:
            self._request_count += 1
            request_count = self._request_count
        self._log_event(
            {
                "event": "request",
                "request_count": request_count,
                "query": str(query),
                "k": k,
                "rerank_count": rerank_count,
                "estimated_input_tokens": _estimate_tokens(str(query), hits[:rerank_count], self._document_text),
                "caller": _caller_summary(),
            }
        )
        try:
            reranked = self._rerank_hits(str(query), hits[:rerank_count])
        except Exception as exc:
            fallback_hits = list(hits[:rerank_count])
            self._set_cached(cache_key, fallback_hits)
            self._log_event(
                {
                    "event": "fallback",
                    "request_count": request_count,
                    "query": str(query),
                    "k": k,
                    "rerank_count": rerank_count,
                    "error": repr(exc),
                    "caller": _caller_summary(),
                }
            )
            self._warn_once(f"Jina rerank failed; falling back to BM25 order: {exc}")
            return fallback_hits + hits[rerank_count:]

        self._set_cached(cache_key, reranked)
        self._log_event(
            {
                "event": "success",
                "request_count": request_count,
                "query": str(query),
                "k": k,
                "rerank_count": rerank_count,
            }
        )
        return reranked + hits[rerank_count:]

    def doc(self, docid: Any) -> Any:
        return self._searcher.doc(docid)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._searcher, name)

    def _rerank_hits(self, query: str, hits: Sequence[Any]) -> List[Any]:
        documents = [self._document_text(hit) for hit in hits]
        payload = {
            "model": self.config.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
            "truncate": self.config.truncate,
        }
        response = self._post_with_retries(payload)
        response.raise_for_status()
        data = response.json()
        results = data.get("results")
        if not isinstance(results, list):
            raise ValueError("Jina response did not include a results list.")

        reranked: List[Any] = []
        used_indices = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if not isinstance(index, int):
                continue
            if 0 <= index < len(hits) and index not in used_indices:
                reranked.append(hits[index])
                used_indices.add(index)

        if not reranked:
            raise ValueError("Jina response did not include usable result indices.")

        for index, hit in enumerate(hits):
            if index not in used_indices:
                reranked.append(hit)
        return reranked

    def _post_with_retries(self, payload: Dict[str, Any]) -> requests.Response:
        last_error: Optional[Exception] = None
        attempts = max(0, self.config.max_retries) + 1
        for attempt in range(attempts):
            try:
                self._wait_for_rate_limit_slot()
                return self._session.post(
                    self.config.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                        "Connection": "close",
                    },
                    json=payload,
                    timeout=self.config.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                self._reset_session()
                if attempt == attempts - 1:
                    break
                time.sleep(self.config.retry_backoff * (2 ** attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Jina rerank request failed before receiving a response.")

    def _wait_for_rate_limit_slot(self) -> None:
        if self.config.min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            wait_s = self.config.min_interval - (now - self._last_request_time)
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_request_time = time.time()

    def _reset_session(self) -> None:
        with self._lock:
            try:
                self._session.close()
            finally:
                self._session = requests.Session()

    def _document_text(self, hit: Any) -> str:
        doc = self._searcher.doc(hit.docid)
        raw = ""
        if doc is not None and hasattr(doc, "raw"):
            raw = doc.raw() or ""

        text = _text_from_raw_doc(raw)
        if not text:
            text = raw or str(doc or hit)
        return text[: self.config.max_doc_chars]

    def _get_cached(self, key: Tuple[str, int, Tuple[str, ...]]) -> Optional[List[Any]]:
        with self._lock:
            cached = self._cache.get(key)
            return list(cached) if cached is not None else None

    def _set_cached(self, key: Tuple[str, int, Tuple[str, ...]], value: List[Any]) -> None:
        if self.config.cache_size <= 0:
            return
        with self._lock:
            if key not in self._cache:
                self._cache_order.append(key)
            self._cache[key] = list(value)
            while len(self._cache_order) > self.config.cache_size:
                old_key = self._cache_order.pop(0)
                self._cache.pop(old_key, None)

    def _warn_once(self, message: str) -> None:
        with self._lock:
            if self._warned_failure:
                return
            self._warned_failure = True
        print(message)

    def _log_event(self, event: Dict[str, Any]) -> None:
        if not self.config.log_path and not self.config.verbose:
            return

        payload = {
            "ts": time.time(),
            **event,
        }
        line = json.dumps(payload, ensure_ascii=False)
        if self.config.verbose:
            print(f"[JinaRerank] {line}")
        if self.config.log_path:
            path = Path(self.config.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")


def _text_from_raw_doc(raw: str) -> str:
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    if isinstance(payload, dict):
        text_parts = _dict_text_parts(payload)
        if text_parts:
            return " ".join(text_parts)
        return json.dumps(payload, ensure_ascii=False)
    return str(payload)


def _dict_text_parts(payload: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    for key in (
        "contents",
        "text",
        "title",
        "Title",
        "name",
        "Description",
        "description",
        "full_description",
        "small_description",
    ):
        value = payload.get(key)
        if value:
            parts.append(_normalize_text(value))

    product = payload.get("product")
    if isinstance(product, dict):
        parts.extend(_dict_text_parts(product))
    return [part for part in parts if part]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_normalize_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_normalize_text(item) for item in value.values())
    return str(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be true or false.")


def _estimate_tokens(query: str, hits: Sequence[Any], document_text) -> int:
    char_count = len(query or "")
    for hit in hits:
        char_count += len(document_text(hit))
    return max(1, char_count // 4)


def _caller_summary() -> List[str]:
    frames = []
    for frame in traceback.extract_stack(limit=12)[:-2]:
        filename = frame.filename.replace("\\", "/")
        if "jina_reranker.py" in filename:
            continue
        if "web_agent_site" not in filename and "IntentionChangeBench" not in filename:
            continue
        frames.append(f"{Path(filename).name}:{frame.lineno}:{frame.name}")
    return frames[-4:]
