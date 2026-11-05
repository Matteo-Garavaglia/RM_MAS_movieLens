# Limiti sperimentali, compatibilità e verifiche aperte

[← README generale](../README.md) · [Indice](INDEX.md) · [Cronologia](PROJECT_EVOLUTION.md)

Questa pagina raccoglie i vincoli che ricorrono nei documenti. Non modifica il protocollo né colma dati mancanti: serve a evitare inferenze improprie quando si scrive la tesi o si confrontano le run.

## 1. Cosa viene effettivamente misurato

| Percorso | Ranking/risultato primario | Che cosa **non** si può concludere automaticamente |
| --- | --- | --- |
| Retrieval 50 film (notebook 04) | Pertinenza su query e giudizi di quel campione | Generalizzazione ai 2.000 o ~9.000 film |
| Recommendation 02 | Ranking Popularity/MF/GTE su scenario `k` | Che il test sia leakage-free usando vecchi embedding |
| Recommendation 04 | Delta appaiato tra GTE diretto e LLM→GTE | Miglioramento attribuibile al solo ragionamento LLM senza altre differenze di rappresentazione |
| Recommendation 05 | Primi 10 ID ammissibili in `state.search_response.results` | Qualità semantica, testuale o pedagogica dell’Answer LLM |
| Orchestrazione iterativa v1 | Smoke test di controllo flusso e contratti con mock | Efficienza/qualità live o superiorità rispetto alla lineare |

I documenti di `recommendation_02`, `04` e `05` dichiarano test sintetici e/o mock. Le metriche reali si ottengono solo dalle relative esecuzioni nel repository e dai CSV esportati.

## 2. Protocollo cold-start `rs_coldstart_v1`

- Livelli `k = 0, 1, 3, 5, 10` rating visibili (nel benchmark `recommendation_04` il default usa l’LLM solo con feedback informativo, non a `k=0`).
- **A:** 20 target nascosti fissi. **B:** tutti i rating ancora non rivelati. Il ranking è lo stesso; cambia il set di valutazione.
- Train globale limitato agli utenti previsti dal protocollo; durante la predizione, ogni sistema accede solo alle interazioni visibili di quello scenario.
- I film senza un rating non devono essere assimilati a film non graditi. Valori assoluti A/B e in particolare Recall@10 non sono comparabili direttamente se cambia il numero dei target rilevanti.
- Il test descritto come già consultato non deve essere riutilizzato per nuove ottimizzazioni confermative: servirà uno split davvero non consultato o un disegno di cross-validation appropriato.

Fonte: [protocollo](experiments/recommendation-01-coldstart-evaluation-protocol.md), [04](experiments/recommendation-04-llm-semantic-profile-gte.md), [05](experiments/recommendation-05-linear-llm-mcp-benchmark.md).

## 3. Leakage e rappresentazioni dei film

La documentazione segnala che i vettori GTE precomputati possono incorporare tag MovieLens e statistiche aggregate potenzialmente riferibili agli utenti di test. Non basta escludere i rating di test dalla query se l’informazione è già nel documento indicizzato. Un confronto pienamente leakage-free richiede una nuova rappresentazione del catalogo costruita senza interazioni degli utenti di test (per esempio solo metadata esterni idonei) e la ripetizione della valutazione.

Le rappresentazioni BGE v1, `D_full_enriched_v2` e GTE non sono intercambiabili: vanno confrontati modello, dimensioni, normalizzazione, generazione del testo, collection e fingerprint. Una buona self-similarity dell’encoder è solo un test diagnostico, non la prova di identico preprocessing.

## 4. Reranker vs LLM vs Answer

- Il reranker tecnico è un CrossEncoder applicato ai **candidati recuperati**; non è un agente LLM e non viene eseguito internamente da Qdrant.
- L’Answer LLM può restituire meno risorse rispetto al retrieval e produce spiegazioni. Nel benchmark `recommendation_05` si registrano separatamente i suoi ID, mentre la metrica principale viene dal MCP.
- `recommendation_04` genera con LLM **query/profili semantici** poi rappresentati da GTE: non implementa due LLM in dialogo, né Critic/Answer, né reranking.

Per attribuzioni causali servono controlli coerenti su candidate set, rappresentazioni, modelli e scelte di scoring.

## 5. Confronto lineare vs iterativo

La baseline del notebook 11 è prevalentemente `Planner → MCP Retrieval → Answer`. La variante iterativa v1 introduce un ciclo decisionale condizionale, ma è **additiva** e usa un prompt Answer distinto nella versione documentata. Prima di un confronto scientifico corretto allineare:

1. Input/scenari, corpus, filtri, candidate pool e ground truth;
2. Modelli e provider, prompt/versioni, output richiesti e temperatura dove supportata;
3. Budget dei retrieval, delle chiamate LLM, dei token e del tempo;
4. Regole di grounding e valutazione del ranking/risposta;
5. Logging di costi, latenza, trigger, terminate reason e run fallite.

I documenti dichiarano pausa/ripresa nel **kernel corrente** con `WAITING_USER`, non un checkpoint persistente con ripresa autonoma fra sessioni. La Validation LLM è un giudizio semantico senza ground truth; il controllo di ID ammissibili è deterministico ma non valida automaticamente le spiegazioni.

## 6. Stato della verifica e passaggio a WhoTeach

I README non documentano una validazione end-to-end live della nuova orchestrazione iterativa con la tua istanza Qdrant, il gateway MCP e le API LLM reali. I test offline sono evidenze utili di robustezza delle componenti, non risultati empirici finali della tesi.

Il porting reale a WhoTeach richiede più di una modifica dei prompt: **schema delle risorse educative, ingestion/chunking, embedding, collection, adapter di filtri, profilo di dominio e giudizi di valutazione**. Il `whoteach_template` descritto nella guida resource LLM/MCP non è ancora una integrazione operativa verificata.

Il progetto di tesi può proporre un’architettura multi-agente più ampia, ma il prototipo documentato **non** dimostra ancora l’esecuzione di topologie libere o l’apprendimento progressivo di strategie da una experience library.
