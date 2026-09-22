# Changelog

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
