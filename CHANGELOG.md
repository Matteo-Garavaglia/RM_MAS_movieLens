# Changelog

## 0.5.0-dev — retrieval ablation 2k / multimodello

- applicate al workspace le modifiche dei notebook 02/04 per il corpus ~2.000 film e `tmdb_title`;
- introdotta `BASELINE_V2` come configurazione **sperimentale**, lasciando `BASELINE` v1 come default runtime;
- aggiunti adapter sperimentali per BM25, BGE-M3 dense+sparse, Qwen3 Embedding e GTE multilingual;
- aggiunte fusioni weighted RRF basate sui rank e non sugli score grezzi;
- aggiunto il notebook `09_retrieval_ablation_2000_multimodel.ipynb`;
- aggiunto relevance pooling top-20, Recall@20 e controllo appaiato IT/EN;
- predisposta una seconda fase con reranker BGE, Qwen3 e GTE su candidate pool 10/20/30;
- aggiunto extra `experiments` e test unitari per le nuove primitive di retrieval.

## 0.4.0 — LLM orchestration

- aggiunti adapter asincroni e provider-neutral per Gemini e OpenAI;
- aggiunti Planner, Critic opzionale e Answer agent con output Pydantic;
- aggiunto orchestratore deterministico con stato, trace e retry limitati;
- aggiunto gateway tipizzato verso i tool MCP esistenti;
- aggiunti controlli di grounding su `movie_id`, titolo e `no_results`;
- aggiunti prompt e profili provider/agente/workflow versionati;
- aggiunto il notebook `08_llm_mcp_integration.ipynb`;
- aggiunta la guida `docs/llm_orchestration_v1.md`;
- mantenuto `dense_v1` come profilo MCP predefinito e Critic disabilitato;
- test complessivi: 47, senza chiamate API reali o download dei modelli BGE.

La chiave Gemini non è inclusa nel pacchetto. Il notebook carica esplicitamente
`GEMINI_API_KEY` dal `.env` locale, escluso da Git.

## 0.3.0 — MCP

- introdotto il server MCP read-only;
- fissato il contratto pubblico `search_movies` / `get_movie_details`;
- aggiunti profili server-side dense e dense+reranker;
- mantenuto il reranker come esperimento opzionale.
