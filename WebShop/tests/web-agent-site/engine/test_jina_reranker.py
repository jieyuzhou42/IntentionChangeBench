import json

from web_agent_site.engine.jina_reranker import (
    JinaRerankConfig,
    JinaRerankSearcher,
    build_jina_rerank_config_from_env,
)


class FakeHit:
    def __init__(self, docid):
        self.docid = docid


class FakeDoc:
    def __init__(self, raw):
        self._raw = raw

    def raw(self):
        return self._raw


class FakeSearcher:
    def __init__(self):
        self.hits = [FakeHit("d0"), FakeHit("d1"), FakeHit("d2")]
        self.docs = {
            "d0": FakeDoc(json.dumps({"id": "A", "contents": "red shoes"})),
            "d1": FakeDoc(json.dumps({"id": "B", "contents": "black office chair"})),
            "d2": FakeDoc(json.dumps({"id": "C", "contents": "blue mug"})),
        }

    def search(self, query, k=10, **kwargs):
        return self.hits[:k]

    def doc(self, docid):
        return self.docs[docid]


class FakeResponse:
    def __init__(self, payload=None, should_raise=False):
        self.payload = payload or {}
        self.should_raise = should_raise

    def raise_for_status(self):
        if self.should_raise:
            raise requests_error()

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def post(self, *args, **kwargs):
        self.requests.append((args, kwargs))
        return self.response


def requests_error():
    return RuntimeError("network down")


def test_build_config_auto_requires_api_key(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.delenv("WEBSHOP_JINA_RERANK", raising=False)

    assert build_jina_rerank_config_from_env() is None


def test_reranks_hits_by_jina_indices():
    session = FakeSession(
        FakeResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.4},
                    {"index": 2, "relevance_score": 0.1},
                ]
            }
        )
    )
    searcher = JinaRerankSearcher(
        FakeSearcher(),
        config=JinaRerankConfig(api_key="key", top_n=50),
        session=session,
    )

    hits = searcher.search("office chair", k=3)

    assert [hit.docid for hit in hits] == ["d1", "d0", "d2"]
    payload = session.requests[0][1]["json"]
    assert payload["query"] == "office chair"
    assert payload["documents"] == ["red shoes", "black office chair", "blue mug"]
    assert payload["top_n"] == 3


def test_falls_back_to_bm25_order_on_jina_failure():
    session = FakeSession(FakeResponse(should_raise=True))
    searcher = JinaRerankSearcher(
        FakeSearcher(),
        config=JinaRerankConfig(api_key="key", top_n=50),
        session=session,
    )

    hits = searcher.search("office chair", k=3)

    assert [hit.docid for hit in hits] == ["d0", "d1", "d2"]
