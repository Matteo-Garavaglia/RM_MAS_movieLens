# Baseline BGE-M3 dense su Qdrant (notebook 05)

> **Provenienza:** `README_QDRANT.md` · **Tipo:** documentazione tecnica del progetto.
> **Collocazione nel percorso:** Prima integrazione e verifica di equivalenza tra ranking NumPy e Qdrant.
> **Nota:** i contenuti tecnici sottostanti sono stati conservati dalla documentazione fornita;
> non costituiscono una nuova verifica dell’esecuzione del codice nel repository corrente.

Questo pacchetto porta in Qdrant la configurazione risultata migliore nel notebook
`04_retrieval_ablation_50_movies.ipynb`:

- rappresentazione: `D_full_enriched`;
- retriever: dense `BAAI/bge-m3`;
- vettori: 1024 dimensioni, normalizzati L2;
- distanza Qdrant: cosine;
- metriche medie nel notebook: NDCG@10 `0.806716`, Recall@10 `0.793732`,
  Precision@10 `0.500000`.

È una baseline provvisoria e versionata (`v1`). BM25, hybrid, reranker, MCP e LLM
sono volutamente fuori da questo passaggio: prima si verifica che Qdrant replichi
il ranking dense già misurato.

## Struttura

```text
notebooks/05_qdrant_dense_integration.ipynb  esperimento guidato e validazione
src/movie_rag/config.py                      scelte congelate e nomi di versione
src/movie_rag/documents.py                   documento D, payload e fingerprint
src/movie_rag/embeddings.py                  adapter BGE-M3 e cache validata
src/movie_rag/filters.py                     stessi filtri in Pandas e Qdrant
src/movie_rag/qdrant_store.py                collection, upsert e retriever
src/movie_rag/validation.py                  equivalenza NumPy ↔ Qdrant
src/movie_rag/evaluation.py                  metriche estratte dal notebook
configs/baseline_manifest_v1.json            provenienza e risultati della baseline
tests/                                       test senza download del modello BGE-M3
```

## Avvio rapido

Da questa cartella:

```bash
docker compose up -d
python -m pip install -e ".[bge,test]"
```

Poi aprire `notebooks/05_qdrant_dense_integration.ipynb` ed eseguire le celle in
ordine. Il notebook usa `df_test_enriched` se è già in memoria; altrimenti carica
il CSV configurato nella prima sezione.

Qdrant è esposto solo su `127.0.0.1` e conserva i dati nel volume Docker
`qdrant_storage`. La collection non viene cancellata automaticamente. Il flag
`RECREATE_COLLECTION` è `False`; abilitarlo elimina e ricrea soltanto la collection
indicata.

## Cosa viene salvato

Ogni punto Qdrant usa `movieId` come ID e contiene:

- il named vector `dense`;
- il documento completo `D_full_enriched`;
- titoli, sinossi, persone e altri metadati utili alla futura risposta grounded;
- campi normalizzati per filtri esatti su anno, durata, lingua, paese, genere e
  contenuto adulto;
- versione della rappresentazione, modello e fingerprint del catalogo.

Il reranker, quando verrà reintrodotto, resterà un calcolo a runtime: non produce
un vettore da salvare in Qdrant.

## Criterio di accettazione

Con ricerca `exact=True`, la validazione deve mostrare per le query disponibili:

- `same_order=True`;
- `overlap@10=1.0`;
- differenze di score prossime alla precisione numerica.

Se questo controllo fallisce, non si procede ancora con MCP: si confrontano ordine
del DataFrame, normalizzazione, testo documentale, filtri e versione del modello.

## Nota sul notebook di ablation

Il notebook originale non viene modificato, così gli output che hanno portato alla
scelta della baseline restano un artefatto di provenienza immutato. Durante la review
sono emersi due punti da non copiare nel nuovo codice:

- nella cella degli import, `from __future__ import annotations` compare dopo istruzioni
  eseguibili; in una riesecuzione pulita deve essere la prima istruzione Python;
- la funzione dimostrativa `search_movies` passa al reranker candidati BM25, benché il
  testo della sezione parli di candidati hybrid. La griglia salvata usa invece
  `include_reranker=False`, quindi questo non determina il risultato dense scelto.

Entrambi i punti sono risolti nei moduli del presente pacchetto; il reranker non viene
ancora reintrodotto.

## Riferimenti

- https://qdrant.tech/documentation/quickstart/
- https://qdrant.tech/documentation/search/filtering/
- https://qdrant.tech/documentation/manage-data/collections/
- https://huggingface.co/BAAI/bge-m3
