

## WebShop Dataset Size

The simulation and benchmark runners default to the full WebShop product dataset:

```powershell
python .\src\run_simulation.py
```

Make sure these files exist:

- `WebShop/data/items_shuffle.json`
- `WebShop/data/items_ins_v2_1000.json`
- `WebShop/search_engine/indexes/`

Then run:

```powershell
python .\src\run_simulation.py --webshop_num_products all
```

For faster local smoke tests, pass `--webshop_num_products 1000`.
`--webshop_num_products 100000` is still supported if you built `WebShop/search_engine/indexes_100k/`.

The product catalog and instruction/attribute file can be scaled separately.
By default, `--webshop_num_products all` uses the full product catalog with the
existing 1k instruction/attribute file. Set `WEBSHOP_ATTR_DATASET=all` only if
you also downloaded `WebShop/data/items_ins_v2.json`.

## WebShop Search Reranking

WebShop can keep its BM25/Pyserini first-stage search and rerank the top 50
candidates with Jina AI before results are shown to the agent.

Add these variables to `IntentionChangeBench/.env` or your shell:

```powershell
JINA_API_KEY=your-jina-api-key
WEBSHOP_JINA_RERANK=1
WEBSHOP_JINA_RERANK_MODEL=jina-reranker-v3
WEBSHOP_JINA_RERANK_TOP_N=50
WEBSHOP_JINA_RERANK_MAX_DOC_CHARS=1200
WEBSHOP_JINA_RERANK_MIN_INTERVAL=8
WEBSHOP_JINA_RERANK_MAX_RETRIES=3
WEBSHOP_JINA_RERANK_RETRY_BACKOFF=1
```

If `JINA_API_KEY` is set, reranking is enabled by default unless
`WEBSHOP_JINA_RERANK=0`. The wrapper falls back to the original BM25 order if a
Jina request fails, so long simulations can continue. If you see transient
connection resets, increase `WEBSHOP_JINA_RERANK_MAX_RETRIES` or lower
`--parallelism`.

Jina's free reranker limit can be reached by TPM before RPM because one WebShop
request sends many product documents. `WEBSHOP_JINA_RERANK_MAX_DOC_CHARS`
shortens each candidate document before reranking, and
`WEBSHOP_JINA_RERANK_MIN_INTERVAL` spaces requests out.

To see exactly which query/code path is calling Jina, enable diagnostics:

```powershell
WEBSHOP_JINA_RERANK_VERBOSE=1
WEBSHOP_JINA_RERANK_LOG_PATH=IntentionChangeBench/data/jina_rerank_calls.jsonl
```
