# Refactor: reranker caricato dal profilo MCP

Obiettivo: usare `movielens_gte_quality_v2` senza costruire o passare manualmente un reranker al registry.

## Uso dopo il refactor

```python
registry = ResourceSearchServiceRegistry.from_environment(
    MCP_CONFIG_PATH,
    profile_id="movielens_gte_quality_v2",
)
```

Il registry:

1. legge il blocco `encoder`;
2. costruisce il retriever GTE + Qdrant;
3. legge il blocco top-level `reranker`;
4. costruisce `SentenceTransformerCrossEncoderScorer` + `ScoreRerankerAdapter`;
5. costruisce il `RetrievalService` con il reranker;
6. il modello CrossEncoder resta lazy e viene caricato solo alla prima query che usa `rerank=true`.

Il parametro Python `reranker=` resta solo come override opzionale per test/esperimenti e non è necessario nel normale startup MCP/LLM.

## Configurazione quality

```json
"reranker": {
  "name": "gte_multilingual_reranker",
  "backend": "sentence_transformers_cross_encoder",
  "model_name": "Alibaba-NLP/gte-multilingual-reranker-base",
  "trust_remote_code": true,
  "batch_size": 8,
  "max_length": 1024,
  "device": "auto",
  "precision": "auto"
}
```

`device=auto` seleziona CUDA, poi MPS, poi CPU. `precision=auto` usa FP16 su CUDA e lascia la precisione standard su MPS/CPU.

## Validazione

Il profilo viene ora rifiutato già in fase di parsing se il `search_profile_id` selezionato ha `rerank=true` ma manca il blocco top-level `reranker`.

Test: `14 passed`.
