# Contratto MCP Movie RAG — `movie-search/v1`

Questo documento congela il confine tra una LLM e il retriever cinematografico.
Il contratto descrive **capacità applicative** e non dettagli di implementazione:
il client chiede film compatibili con una richiesta, mentre il server decide quale
pipeline versionata eseguire.

## Decisioni di progetto

1. `search_movies` è il tool destinato all'uso normale da parte della LLM.
2. Il profilo predefinito è scelto una volta all'avvio del server.
3. Modello, collection, reranker, candidate pool, ricerca esatta e soglia non
   compaiono nell'input del tool normale.
4. `search_movies_experimental` permette confronti per singola query, ma viene
   registrato soltanto se abilitato esplicitamente.
5. Il tool sperimentale accetta un `profile_id` presente nella whitelist del
   server, non un nome modello libero.
6. Nessuna chiamata modifica la configurazione delle chiamate successive.
7. Una ricerca senza risultati è una risposta valida con `status=no_results`,
   non un errore MCP.

Questa separazione evita che una LLM scelga casualmente parametri tecnici e
mantiene riproducibili gli esperimenti. Un profilo lega sempre:

- encoder dense;
- dimensione, nome vettore e collection Qdrant compatibili;
- rappresentazione documentale;
- eventuale reranker e suo candidate pool;
- impostazioni di esecuzione predefinite.

Il server verifica inoltre che tutti i punti nella collection dichiarino lo
stesso modello, la stessa rappresentazione e lo stesso schema payload attesi dal
profilo.

## Tool 1 — `search_movies`

### Input

```json
{
  "query": "un film malinconico su memoria e identità",
  "filters": {
    "max_runtime": 120,
    "min_year": 1990,
    "languages": ["en"],
    "exclude_genres": ["Horror"],
    "adult": false
  },
  "limit": 10
}
```

| Campo | Semantica |
|---|---|
| `query` | Preferenze morbide: tema, trama, tono, atmosfera, entità testuali |
| `filters` | Vincoli obbligatori applicati da Qdrant prima del ranking |
| `limit` | Numero massimo di risultati, da 1 a 20; default 10 |

I filtri supportati sono:

| Filtro | Tipo | Semantica |
|---|---:|---|
| `min_runtime`, `max_runtime` | numero | intervallo inclusivo in minuti |
| `min_year`, `max_year` | intero | intervallo inclusivo |
| `languages` | lista | almeno una lingua ammessa |
| `countries` | lista | almeno un paese ammesso |
| `include_genres` | lista | almeno un genere richiesto |
| `exclude_genres` | lista | nessun genere escluso |
| `adult` | booleano | corrispondenza esatta |

Le alternative dentro la stessa lista sono in OR; categorie di filtro diverse
sono combinate in AND. Le stringhe sono confrontate senza distinzione tra
maiuscole e minuscole.

### Output

```json
{
  "contract_version": "movie-search/v1",
  "status": "ok",
  "query": "un film malinconico su memoria e identità",
  "applied_filters": {
    "max_runtime": 120,
    "min_runtime": null,
    "min_year": 1990,
    "max_year": null,
    "languages": ["en"],
    "countries": [],
    "include_genres": [],
    "exclude_genres": ["Horror"],
    "adult": false
  },
  "requested_limit": 10,
  "returned_count": 1,
  "pipeline": {
    "profile_id": "dense_v1",
    "pipeline": "dense",
    "vector_store": "qdrant",
    "collection_name": "movies_d_full_enriched_dense_v1",
    "representation_version": "D_full_enriched_v1",
    "encoder_model": "BAAI/bge-m3",
    "reranker_enabled": false,
    "reranker_model": null,
    "candidate_pool_size": null,
    "exact": true,
    "score_threshold": null
  },
  "results": [
    {
      "rank": 1,
      "dense_rank": 1,
      "movie_id": 1,
      "title": "Example Movie",
      "overview": "...",
      "year": 1999,
      "runtime_minutes": 110,
      "genres": ["Drama"],
      "original_language": "en",
      "countries": ["US"],
      "directors": ["Example Director"],
      "main_cast": ["Example Actor"],
      "scores": {
        "final": 0.81,
        "dense": 0.81,
        "reranker": null
      }
    }
  ],
  "message": null
}
```

La provenienza della pipeline è restituita per audit, ma non è modificabile dal
tool normale. Con un profilo reranked, `rank` è la posizione finale,
`dense_rank` conserva la posizione del primo stadio e i tre score restano
separati.

### Nessun risultato

Filtri legittimi possono produrre una lista vuota, soprattutto nel catalogo da
50 film. La risposta mantiene lo stesso schema:

```json
{
  "contract_version": "movie-search/v1",
  "status": "no_results",
  "returned_count": 0,
  "results": [],
  "message": "No catalog movies satisfy the query and hard filters."
}
```

Sarà l'orchestratore, non il retriever, a decidere se chiedere un chiarimento,
proporre di rilassare un vincolo o astenersi. Il server non rilassa mai un filtro
in modo implicito.

## Tool 2 — `get_movie_details`

Input:

```json
{"movie_ids": [1, 2, 3]}
```

Restituisce metadata grounded per un massimo di 50 ID unici. L'ordine richiesto
viene preservato e gli ID non presenti sono elencati in `missing_movie_ids`.
Questo tool serve ad approfondire risultati già ottenuti con `search_movies`,
non a fare retrieval.

## Tool opzionale — `search_movies_experimental`

Il tool è assente per default. Si abilita all'avvio con:

```bash
export MOVIE_RAG_ENABLE_EXPERIMENTAL_TOOLS=true
```

Input aggiuntivo rispetto a `search_movies`:

```json
{
  "profile_id": "dense_rerank_v1",
  "execution_overrides": {
    "candidate_pool_size": 30,
    "exact": true,
    "score_threshold": 0.15
  }
}
```

Vincoli:

- `profile_id` deve esistere in `configs/mcp_profiles_v1.json`;
- `candidate_pool_size` vale soltanto per profili con reranker e deve essere
  almeno pari a `limit`;
- gli override valgono solo per la chiamata corrente;
- encoder, dimensione e collection non sono mai valori liberi;
- il profilo effettivo e gli override applicati compaiono in `pipeline`.

Il tool serve ad ablation, confronto per-query e validazione. Non dovrebbe essere
mostrato alla LLM applicativa. Se in seguito l'orchestratore dovrà scegliere tra
workflow, esporremo strategie semantiche controllate, non nomi di modelli.

## Profili inclusi

| Profilo | Pipeline | Stato |
|---|---|---|
| `dense_v1` | BGE-M3 dense → Qdrant | predefinito; migliore risultato attuale |
| `dense_rerank_v1` | dense top-n → BGE reranker v2-m3 | sperimentale; conservato per confronti |

Nuovi modelli richiedono un nuovo profilo versionato e una collection indicizzata
con i vettori compatibili. Non basta cambiare il nome del modello a runtime.

## Errori

Sono errori MCP:

- schema input non valido o campo sconosciuto;
- intervallo invertito;
- profilo non autorizzato;
- candidate pool incompatibile;
- collection assente, vuota o con provenienza incompatibile;
- Qdrant o modello non disponibili.

Non sono errori MCP:

- zero risultati dopo filtri validi;
- alcuni ID mancanti in `get_movie_details`.

## Configurazione del processo

| Variabile | Default | Scopo |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | endpoint Qdrant |
| `QDRANT_API_KEY` | vuota | credenziale per Qdrant remoto |
| `QDRANT_TIMEOUT_SECONDS` | `30` | timeout del client |
| `MOVIE_RAG_PROFILE` | `dense_v1` | profilo del tool normale |
| `MOVIE_RAG_PROFILES_PATH` | file incluso | registry JSON alternativo |
| `MOVIE_RAG_USE_FP16` | auto | forza/nega FP16 per BGE |
| `MOVIE_RAG_ENABLE_EXPERIMENTAL_TOOLS` | `false` | registra il terzo tool |
| `MOVIE_RAG_MCP_TRANSPORT` | `stdio` | `stdio` o `streamable-http` |

Con trasporto `stdio`, il processo non deve scrivere messaggi applicativi su
stdout perché quello stream trasporta il protocollo MCP. I log vanno su stderr.
