# Evoluzione del progetto — dalla retrieval ablation alla recommendation cold-start

[← README generale](../README.md) · [Indice](INDEX.md) · [Avvertenze metodologiche](RESEARCH_LIMITATIONS.md)

> Ricostruzione documentale basata sui dieci README forniti e sul contesto del progetto di tesi. I documenti descrivono fasi diverse e talvolta sono **fotografie storiche**: non sostituiscono una verifica delle versioni del codice e dei risultati effettivamente disponibili.

## 1. Due traiettorie parallele

Nel percorso si riconoscono due linee collegate ma non identiche:

1. **Infrastruttura e orchestrazione:** costruzione del catalogo, retrieval vettoriale, MCP, LLM e controller iterativo.
2. **Sperimentazione recommender cold-start:** definizione di scenari controllati (rating rivelati/nascosti), benchmark di metodi classici e confronto con LLM e MCP.

La prima produce componenti riutilizzabili; la seconda definisce **che cosa** misurare. I notebook possono condividere MovieLens e i vettori GTE senza appartenere alla stessa sequenza numerica.

## 2. Fasi di sviluppo

### Fase A — Dataset e retrieval offline (notebook 01–04)

- MovieLens fornisce film, rating, tag e ID; TMDb ne arricchisce le descrizioni.
- Le prove iniziali sono su un sottoinsieme di **50 film**. Sono confrontati retrieval lessicale, dense, fusioni e reranking, con giudizi/pooling manuale e metriche ranking.
- La documentazione del notebook 05 dichiara come scelta iniziale `D_full_enriched` + BGE-M3 dense, con risultati precedenti NDCG@10 `0.806716`, Recall@10 `0.793732` e Precision@10 `0.500000` **relativi al test iniziale documentato**, non al successivo catalogo completo.

### Fase B — Qdrant e tool MCP specifici per MovieLens (notebook 05–08)

- Il notebook 05 trasporta la baseline dense da NumPy a Qdrant e richiede equivalenza del ranking.
- Il notebook 06 confronta dense con dense+reranker; il vantaggio del reranker **non è confermato in aggregato** nel test descritto, quindi dense rimane default in questa fase.
- Il notebook 07 introduce l’interfaccia MCP `search_movies` / `get_movie_details`.
- Il notebook 08 aggiunge agenti LLM Planner/Answer e un Critic opzionale governato da Python.

**Documento di riferimento storico:** [architettura MovieLens legacy](history/movielens-bge-m3-legacy-architecture.md); [baseline BGE-M3/Qdrant](retrieval/bge-m3-qdrant-dense-baseline-v1.md).

### Fase C — Aumento di scala e confronto multimodello (notebook 09)

- Ablation documentata su circa **2.000 film**, distinta dalle prime prove su 50.
- Confronto di BGE-M3 dense/sparse, BM25, fusioni, Qwen3-Embedding, GTE multilingual e reranker.
- Introduzione e valutazione separata della rappresentazione `D_full_enriched_v2` con `tmdb_title`, evitando di invalidare retroattivamente la baseline BGE v1.

Non si deve applicare indistintamente il risultato sui 50 film ai 2.000 film o all’intero catalogo; i relativi giudizi e manifest restano diversi.

### Fase D — Astrazione resource-oriented (notebook 10)

- Scelta di `Alibaba-NLP/gte-multilingual-base` (768 dimensioni) e collection `movies_d_full_enriched_v2_gte_multilingual_base_v1` per la nuova integrazione.
- Contratto generico `ResourceDocument(resource_id, title, text, metadata, domain, source)`.
- Nuova API MCP: `search_resources` / `get_resources`, separata dalla configurazione interna di encoder, collection e reranker.
- Profili tecnici `fast` (top-10 senza rerank) e `quality` (top-20, reranking opzionale, top-10 finale).

**Documento:** [GTE → Qdrant → MCP generico](architecture/gte-qdrant-resource-mcp-integration.md).

### Fase E — Workflow LLM generico e configurato (notebook 11)

- Baseline lineare: **ResourcePlannerAgent → MCP resource retrieval → ResourceAnswerAgent**.
- Separazione configurazioni LLM, dominio e retrieval nei rispettivi JSON.
- Stato condiviso, trace LLM/tool, gestione dei chiarimenti obbligatori, assenza risultati e whitelisting degli ID raccomandati.
- La documentazione di questa baseline indica il workflow come **prevalentemente fisso**, non ancora una libera orchestrazione dinamica.

**Documento:** [guida architetturale resource LLM/MCP](architecture/resource-llm-mcp-workflow.md).

### Fase F — Reranking e controller iterativo (estensioni successive)

- Un refactor gestisce il reranker **tramite profilo MCP server-side** e ne carica il CrossEncoder alla prima richiesta effettiva; documento di [configurazione del GTE reranker](retrieval/gte-reranker-mcp-profile-refactor.md).
- `resource_iterative_orchestration.py` introduce un controller additivo (lineare o iterativo), con budget, tracce, verifiche, ripetizione controllata dei retrieval e pause `WAITING_USER`.
- **Grado di prova dichiarato:** smoke test offline/mock. Il documento specifica che la versione non è stata validata sul vero Qdrant e sulle API reali del repository attuale.

**Documento:** [orchestrazione iterativa v1](orchestration/iterative-orchestration-v1-installation.md).

## 3. Filone sperimentale recommendation (notebook dedicati)

| Esperimento | Confronto | Note da preservare |
| --- | --- | --- |
| [`recommendation_01`](experiments/recommendation-01-coldstart-evaluation-protocol.md) | Definizione split, scenari e metriche | `rs_coldstart_v1`, A/B, ground truth nascosta |
| [`recommendation_02`](experiments/recommendation-02-popularity-mf-gte-baselines.md) | Popularity vs MF fold-in vs GTE diretto | Selezione iperparametri su validation; possibili tag leakati |
| [`recommendation_04`](experiments/recommendation-04-llm-semantic-profile-gte.md) | GTE diretto vs profilo in linguaggio naturale generato da LLM e poi codificato GTE | Nessun dialogo tra due LLM, nessun reranker in fase 2A |
| [`recommendation_05`](experiments/recommendation-05-linear-llm-mcp-benchmark.md) | Pipeline Planner → MCP → Answer contro baseline | Metriche top-10 sugli ID del **retriever**, non sull’Answer |

La numerazione di questi esperimenti non implica che le loro cartelle siano già installate nel repo né che `recommendation_03` sia assente dal progetto: **manca soltanto tra i README consegnati**.

## 4. I cambi di contratto che contano

| Componente | Prima (fase B/C) | Dopo (fase D/E) |
| --- | --- | --- |
| Unità di dominio | Film con `movieId` | Risorsa con `resource_id` e `domain` |
| Tool MCP | `search_movies`, `get_movie_details` | `search_resources`, `get_resources` |
| Encoder della baseline | BGE-M3 dense (1024 dimensioni) | GTE multilingual base (768 dimensioni) |
| Persistenza | Collection MovieLens BGE | Collection GTE dedicata, documentazione v2 |
| Pipeline LLM | Agenti movie-specific | `ResourcePlannerAgent` / `ResourceAnswerAgent` |
| Configurazione | Profilo Movie RAG interno | Profili LLM, dominio e retrieval distinti |
| Strategia di esecuzione | Lineare con Critic opzionale | Baseline lineare generica; variante iterativa additiva |

Non cancellare il passato: ogni baseline ha un proprio corpus, una rappresentazione e dei manifest necessari a interpretare i risultati.

## 5. Questioni ancora aperte secondo i documenti

- **Allineamento live:** convalida del gateway MCP per l’iterativo e degli output del server aggiornato.
- **Confronto equo:** stesso modello, prompt Answer, parametri, vincoli e budget in linear/iterative.
- **Sperimentazione attendibile:** evitare leakage dagli embedding e non riutilizzare come test confermativo uno split già consultato.
- **Grounding:** whitelist degli ID restituiti non certifica da sola la veridicità del testo generato.
- **WhoTeach:** disponibilità di risorse, metadati didattici, indexing, filtri e valutazione nel dominio reale.
- **Orchestrazione:** il ciclo iterativo non equivale a topologie multi-agente arbitrarie né a un sistema che apprende da una experience library.

Vedi [RESEARCH_LIMITATIONS.md](RESEARCH_LIMITATIONS.md).
