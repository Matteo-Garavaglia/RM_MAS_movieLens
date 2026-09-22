# Movie RAG — Qdrant, MCP e orchestrazione LLM

Questo pacchetto porta in Qdrant la configurazione risultata migliore nel notebook
`04_retrieval_ablation_50_movies.ipynb` e mantiene due profili separati:

```text
dense_v1:        query + filtri → BGE-M3/Qdrant → top-k
dense_rerank_v1: query + filtri → BGE-M3/Qdrant top-n → reranker → top-k
```

La baseline dense rimane congelata e riproducibile:

- rappresentazione: `D_full_enriched`;
- retriever: `BAAI/bge-m3` dense;
- vettori: 1024 dimensioni, normalizzati L2;
- distanza Qdrant: cosine;
- metriche medie osservate nel notebook: NDCG@10 `0.806716`,
  Recall@10 `0.793732`, Precision@10 `0.500000`.

Il test successivo non ha mostrato un vantaggio complessivo del reranker: perciò
`dense_v1` resta il profilo predefinito. Il reranker viene conservato come
profilo sperimentale, utile per analisi per-query e future ablation.

La versione 0.4 aggiunge un primo workflow LLM provider-neutral:

```text
Planner → MCP/Qdrant → Answer
              ↑
       Critic opzionale
```

Il grafo è governato da Python. Gemini o OpenAI producono esclusivamente output
strutturati e non possono cambiare profilo di retrieval, collection o vincoli.

## Struttura

```text
notebooks/05_qdrant_dense_integration.ipynb  migrazione e equivalenza dense
notebooks/06_qdrant_dense_reranking.ipynb    dense vs dense+reranker e top-n
notebooks/07_mcp_server_validation.ipynb     contratto MCP e round-trip in-process
notebooks/08_llm_mcp_integration.ipynb       Gemini/OpenAI, agenti e flusso end-to-end
src/movie_rag/config.py                      configurazioni versionate
src/movie_rag/documents.py                   documento D, payload e fingerprint
src/movie_rag/embeddings.py                  adapter BGE-M3 e cache validata
src/movie_rag/filters.py                     stessi filtri in Pandas e Qdrant
src/movie_rag/qdrant_store.py                collection, upsert e dense retriever
src/movie_rag/reranking.py                   adapter cross-encoder e pipeline completa
src/movie_rag/mcp_profiles.py                profili tecnici versionati e whitelist
src/movie_rag/mcp_contract.py                schemi pubblici input/output
src/movie_rag/mcp_service.py                 logica applicativa indipendente dal trasporto
src/movie_rag/mcp_server.py                  adapter MCP e tool read-only
src/movie_rag/llm_contract.py                piani, risposte, stato e trace
src/movie_rag/llm_config.py                  profili LLM/agenti/workflow
src/movie_rag/llm_providers.py               adapter Gemini e OpenAI
src/movie_rag/llm_agents.py                  Planner, Critic e Answer
src/movie_rag/llm_mcp.py                     gateway tipizzato verso MCP
src/movie_rag/llm_workflow.py                orchestratore deterministico
src/movie_rag/validation.py                  equivalenza NumPy ↔ Qdrant
src/movie_rag/evaluation.py                  NDCG, Recall e Precision
configs/baseline_manifest_v1.json            provenienza della baseline dense
configs/reranker_manifest_v1.json            contratto del reranker sperimentale
configs/mcp_profiles_v1.json                 dense default e dense+rerank opzionale
configs/mcp.env.example                      variabili d'ambiente commentate
configs/mcp_client.example.json              configurazione client stdio di esempio
docs/mcp_contract_v1.md                      contratto MCP motivato e completo
docs/llm_orchestration_v1.md                 architettura LLM ed estensioni future
tests/                                       test senza download dei modelli BGE
CHANGELOG.md                                 modifiche per versione
```

## Avvio rapido

Dalla cartella radice:

```bash
docker compose up -d
python -m pip install -e ".[bge,mcp,test]"
python -m pytest -q
```

Eseguire poi in ordine:

1. `notebooks/05_qdrant_dense_integration.ipynb`;
2. `notebooks/06_qdrant_dense_reranking.ipynb`;
3. `notebooks/07_mcp_server_validation.ipynb`;
4. `notebooks/08_llm_mcp_integration.ipynb`.

Il notebook 06 presume che la collection sia già popolata. Non la elimina, non
ricalcola gli embedding dei documenti e non esegue un nuovo upsert.

Per il notebook 08 con Gemini:

```bash
python -m pip install -e ".[bge,mcp,llm-gemini,test]"
```

Creare un `.env` locale non versionato:

```text
GEMINI_API_KEY=...
```

Per installare anche OpenAI usare l’extra `llm`; i profili e i modelli sono in
`configs/llm_profiles_v1.json`. Il package non carica `.env` implicitamente: lo
fa in modo esplicito il notebook/applicazione.

Qdrant è esposto dal Docker Compose solo su `127.0.0.1` e conserva i dati nel
volume `qdrant_storage`.

## Dati salvati in Qdrant

Ogni punto usa `movieId` come ID e contiene:

- il named vector `dense`;
- il documento completo `D_full_enriched`;
- titoli, sinossi, persone e metadata per la futura risposta grounded;
- campi normalizzati per filtri esatti su anno, durata, lingua, paese, genere e
  contenuto adulto;
- versione della rappresentazione, modello e fingerprint del catalogo.

Il cross-encoder non genera un vettore riutilizzabile. Il suo score dipende dalla
coppia query-film e viene calcolato a runtime, quindi non viene salvato in Qdrant.

## MCP: contratto stabile e profili interni

Il tool destinato alla LLM è volutamente piccolo:

```text
search_movies(query, filters, limit)
get_movie_details(movie_ids)
```

Il client non passa modello, collection, flag di reranking, candidate pool o
soglia. Il processo seleziona un profilo coerente tramite
`MOVIE_RAG_PROFILE`; il default è `dense_v1`.

Per confronti controllati per singola query è disponibile un terzo tool,
disabilitato di default:

```bash
export MOVIE_RAG_ENABLE_EXPERIMENTAL_TOOLS=true
```

`search_movies_experimental` accetta soltanto un `profile_id` presente nel
registry e override limitati alla chiamata corrente. Non esiste un tool
`set_config`, quindi non c'è stato globale modificabile da una sessione MCP.

Avvio locale con trasporto stdio:

```bash
export QDRANT_URL=http://localhost:6333
export MOVIE_RAG_PROFILE=dense_v1
movie-rag-mcp
```

Il modello viene caricato al primo uso. Profili con la stessa baseline dense
condividono encoder e retriever nel processo, evitando di caricare BGE-M3 due
volte durante i confronti.

Il contratto completo, inclusi filtri, output, no-results ed errori, è in
`docs/mcp_contract_v1.md`. I file `configs/mcp.env.example` e
`configs/mcp_client.example.json` raccolgono le impostazioni senza incorporare
segreti.

## LLM: agenti logici e workflow applicativo

Il contratto MCP non cambia con l’introduzione della LLM. Il Planner traduce la
richiesta in `RetrievalPlan`, il codice chiama i tool MCP e l’Answer agent usa
soltanto i film restituiti. Un controllo applicativo rifiuta ID inventati e
titoli alterati.

Il Critic è disponibile ma disabilitato nel workflow iniziale. Se attivato, può
proporre soltanto una nuova query semantica e non può cambiare filtri; i retry
sono limitati. Ogni ruolo può essere associato a Gemini, OpenAI o a un endpoint
OpenAI-compatible tramite profili server-side, senza esporre la scelta nei tool.

La spiegazione completa, le estensioni multi-agente e la matrice di valutazione
sono in `docs/llm_orchestration_v1.md`.

## Contratto interno del reranking

Il confine applicativo interno usato dal profilo MCP reranked è:

```python
results = reranked_retriever.search(
    query="un film malinconico su memoria e identità",
    filters={"max_runtime": 120},
    candidate_pool_size=20,
    top_k=10,
)
```

L'output conserva separatamente:

- `rank`: rango finale;
- `score` e `reranker_score`: score finale del cross-encoder;
- `dense_rank`: posizione originale tra i candidati Qdrant;
- `dense_score`: similarità cosine del primo stadio;
- payload del film, incluso `movieId`, titolo e documento.

I filtri sono applicati prima del dense retrieval. Il reranker può soltanto
riordinare un sottoinsieme dei candidati già validi.

## Criteri di accettazione

### Notebook 05 — Qdrant dense

Con ricerca `exact=True`, tutte le query devono mostrare:

- `same_order=True`;
- `overlap@10=1.0`;
- differenze di score prossime alla precisione numerica.

### Notebook 06 — reranker

Il notebook verifica e salva:

- appartenenza di ogni risultato finale al candidate pool dense;
- ranghi contigui e tie-break stabile sul rango dense;
- estensione incrementale del pool di giudizi con gli eventuali nuovi film;
- candidate Recall@10/20/30;
- NDCG@10, Recall@10 e Precision@10 dopo reranking;
- delta NDCG per query rispetto alla baseline;
- latenza separata di retrieval e reranking.

Per questo confronto viene attivata anche la modalità di valutazione esplicita
delle run vuote: una query prevista ma senza risultati riceve valori zero e non
scompare silenziosamente dalla media. Il comportamento legacy resta disponibile
per riprodurre i numeri già salvati nel notebook di ablation.

I film nuovi introdotti dal candidate pool massimo vengono esportati in
`artifacts/reranking_v1/reranker_annotation_extension.csv`. Finché resta una
`relevance` vuota, il notebook non calcola metriche definitive: assegnarle
implicitamente zero penalizzerebbe proprio il sistema appena aggiunto.

La decisione corrente è mantenere dense senza reranking come comportamento
normale. `dense_rerank_v1` usa candidate pool 20 per riprodurre l'esperimento;
pool differenti possono essere provati soltanto con il tool sperimentale.

## Nota sul notebook di ablation

Il notebook originale non viene modificato, così gli output che hanno portato
alla baseline restano immutati. Durante la review erano emersi due punti:

- `from __future__ import annotations` non era nella posizione corretta per una
  riesecuzione pulita;
- la funzione dimostrativa `search_movies` passava al reranker candidati BM25,
  benché la sezione parlasse di candidati hybrid.

La nuova pipeline non copia quella funzione: passa esplicitamente al reranker i
candidati dense restituiti da Qdrant, coerentemente con il sistema vincente.

## Riferimenti

- https://qdrant.tech/documentation/quickstart/
- https://qdrant.tech/documentation/search/filtering/
- https://huggingface.co/BAAI/bge-m3
- https://huggingface.co/BAAI/bge-reranker-v2-m3
- https://github.com/FlagOpen/FlagEmbedding
- https://modelcontextprotocol.io/docs/develop/build-server
- https://modelcontextprotocol.io/specification/2026-07-28/server/tools
- https://ai.google.dev/gemini-api/docs/structured-output
- https://ai.google.dev/gemini-api/docs/function-calling
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/guides/function-calling
