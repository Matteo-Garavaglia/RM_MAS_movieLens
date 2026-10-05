# Steps 3–5 integration: GTE → Qdrant → generic resource service → MCP

This is an **additive overlay** for the existing `movie_rag_qdrant` project. Copy the
`src/movie_rag/*.py`, `notebooks/*.ipynb` and `tests/*.py` files into the corresponding
project directories. Existing movie-specific classes remain untouched, so notebooks 05–09
continue to work.

## Runtime choice

- First-stage retrieval: `Alibaba-NLP/gte-multilingual-base`, 768 dimensions, normalized dense vectors.
- Movie text representation: `document__D_full_enriched_v2`; if missing, it is rebuilt with `tmdb_title`.
- Qdrant collection: `movies_d_full_enriched_v2_gte_multilingual_base_v1`.
- `fast` profile: GTE/Qdrant top-10, no reranking.
- `quality` profile: GTE/Qdrant top-20 → optional reranker → top-10.

The reranker is intentionally outside Qdrant and outside the MCP contract. A local scorer,
a Colab/GPU scorer, or a future remote scorer can be wrapped by `ScoreRerankerAdapter`
without changing agents or tools.

## Why this is WhoTeach-ready

The stable contract is `ResourceDocument(resource_id, title, text, metadata, domain, source)`.
The Qdrant layer never refers to movies. `DataFrameResourceAdapter` can map future WhoTeach
chunks to the same representation by changing column configuration only.

## Dependencies

Use the project's existing extras for Qdrant/MCP plus Sentence Transformers, e.g. the same
environment used by notebooks 08–09. The new code imports heavy dependencies lazily.

## MCP contract

- `search_resources(query, filters, limit, profile)`
- `get_resources(resource_ids)`

The public tools do not expose Qdrant collection names, embedding models, vector sizes, or GPU choices.
