# Architettura Resource LLM/MCP: Planner, Retrieval, Answer (notebook 11)

> **Provenienza:** `LLM_RESOURCE_INTEGRATION_README(1).md` · **Tipo:** documentazione tecnica del progetto.
> **Collocazione nel percorso:** Riferimento più completo sulla baseline generica resource_* e i profili configurabili.
> **Nota:** i contenuti tecnici sottostanti sono stati conservati dalla documentazione fornita;
> non costituiscono una nuova verifica dell’esecuzione del codice nel repository corrente.

## Architettura, componenti e guida ai file del progetto

### 1. Obiettivo dell'integrazione

L'integrazione introduce un'architettura di retrieval e interazione LLM riutilizzabile su domini differenti, evolvendo il precedente flusso specifico per MovieLens.

**Pipeline precedente:**

`Planner → search_movies → Answer`

**Pipeline attuale:**

`Planner → search_resources → Answer`

La nuova architettura separa:

- **Core applicativo:** gestione degli agenti, workflow, modelli strutturati e stato di esecuzione.
- **Retrieval:** embedding, ricerca vettoriale, reranking ed esposizione tramite MCP.
- **Dominio:** filtri, vocabolario, metadata e istruzioni specifiche di MovieLens o WhoTeach.
- **Configurazione tecnica:** provider LLM, modelli di embedding, collection Qdrant e profili di ricerca.

L'obiettivo non è eliminare completamente le differenze tra domini, ma rendere riutilizzabili le componenti infrastrutturali e di orchestrazione.

---

### 2. Architettura attuale

Il notebook 11 valida la seguente pipeline:

**1. User Request**

L'utente fornisce una richiesta in linguaggio naturale.

**2. ResourcePlannerAgent**

Il Planner LLM interpreta la richiesta e produce un piano strutturato contenente:

- `semantic_query`
- filtri nativi del dominio
- indicazione di eventuale necessità di chiarimento
- motivazione delle scelte

Il Planner non sceglie il modello di embedding, la collection Qdrant, il reranker o il candidate pool.

**3. Resource Workflow**

Il codice Python riceve e valida il piano, mantiene lo stato della richiesta e decide se procedere alla ricerca oppure interrompersi per un chiarimento.

**4. MCPResourceToolGateway**

Espone al workflow un'interfaccia per invocare e validare i risultati degli strumenti MCP.

**5. search_resources**

Effettua la ricerca tramite il `RetrievalService`, utilizzando la configurazione tecnica selezionata lato server.

**6. ResourceAnswerAgent**

Riceve le risorse recuperate e produce una risposta motivata. Le raccomandazioni devono riferirsi esclusivamente a risorse realmente restituite dal retrieval.

**7. Output**

Il workflow restituisce una risposta strutturata e uno stato contenente il piano, i risultati della ricerca, la risposta finale e le trace.

---

### 3. Mappa dei file Python

#### 3.1 Agenti, workflow e modelli LLM

| File | Responsabilità |
|---|---|
| `resource_llm_agents.py` | Definizione degli agenti LLM generici, tra cui `ResourcePlannerAgent`. Gestisce i ruoli degli agenti e la produzione dei relativi output. |
| `resource_llm_models.py` | Modelli strutturati per piani, risposte e stato del workflow. Definisce i contratti utilizzati per scambiare e validare informazioni. |
| `resource_llm_workflow.py` | Gestione della pipeline applicativa. Coordina Planner, MCP e Answer e gestisce le transizioni di esecuzione. |
| `resource_domain.py` | Definizione e caricamento dei profili di dominio, con terminologia, filtri ammessi e istruzioni specifiche. |

**Dove intervenire:**

- Per modificare il comportamento generale del Planner o dell'Answer Agent: `resource_llm_agents.py`.
- Per modificare gli output strutturati o lo stato condiviso: `resource_llm_models.py`.
- Per modificare l'ordine delle operazioni, aggiungere agenti o introdurre nuove condizioni di esecuzione: `resource_llm_workflow.py`.
- Per modificare la semantica specifica di MovieLens o WhoTeach: configurazione del dominio.

La distinzione fondamentale è che **l'agente definisce il comportamento di una singola operazione LLM, mentre il workflow decide quando eseguirla e come utilizzare il risultato**.

#### 3.2 MCP e retrieval

| File | Responsabilità |
|---|---|
| `resource_mcp.py` | Definizione del server MCP e degli strumenti generici di accesso alle risorse. |
| `resource_mcp_gateway.py` | Client/gateway utilizzato dal workflow LLM per invocare gli strumenti MCP e validarne le risposte. |
| `resource_mcp_profiles.py` | Configurazione e costruzione del servizio di retrieval. Contiene `ResourceSearchServiceRegistry` e i modelli dei profili tecnici. |
| `resource_service.py` | Logica applicativa del retrieval, profili di ricerca e integrazione del reranker tramite adapter. |
| `resource_qdrant.py` | Integrazione con Qdrant, gestione della collection e ricerca vettoriale. |
| `resource_embeddings.py` | Encoder e rappresentazione vettoriale di documenti e query. |
| `resource_rerankers.py` | Implementazione del componente di scoring per il reranking, tra cui il CrossEncoder. |
| `resource_adapters.py` | Traduzione tra strutture specifiche del dominio e contratti generici, inclusi i filtri MovieLens. |

Il flusso di retrieval è:

`search_resources → RetrievalService → QdrantResourceRetriever → Qdrant`

Con profilo quality:

`Qdrant → top-20 → reranker → top-10`

Il reranker non è un agente LLM e non è eseguito direttamente da Qdrant: è un componente separato gestito dal servizio di retrieval.

---

### 4. Configurazioni JSON

L'architettura utilizza tre principali livelli di configurazione.

| File | Contenuto |
|---|---|
| `configs/llm_profiles_v1.json` | Provider e configurazioni dei modelli LLM: modello, temperatura e altri parametri di generazione. |
| `configs/resource_domains_v1.json` | Configurazione logica del dominio: terminologia, filtri consentiti, istruzioni specifiche per Planner e Answer. |
| `configs/resource_mcp_profiles_v1.json` | Configurazione tecnica del retrieval: collection Qdrant, encoder, profili fast/quality e reranker. |

#### 4.1 Profili LLM

Gli agenti dipendono da un'interfaccia comune di generazione, mentre il provider concreto viene configurato separatamente.

Nel notebook 11 vengono riutilizzati `GeminiProvider` e `load_llm_config` dell'infrastruttura LLM precedente.

Questo permette di cambiare provider o modello senza modificare direttamente MCP e retrieval.

#### 4.2 Profili di dominio

`resource_domains_v1.json` contiene istruzioni specifiche del dominio, tra cui `planner_instructions`.

Per esempio, MovieLens utilizza concetti come genere cinematografico, anno e durata. WhoTeach utilizzerà disciplina, livello scolastico, obiettivi e prerequisiti.

Le istruzioni generali degli agenti e quelle specifiche del dominio vengono mantenute logicamente distinte.

**Nota sui prompt:** nella struttura attuale le istruzioni specifiche sono esternalizzate nel JSON, mentre le istruzioni generali degli agenti appartengono alla loro implementazione. La composizione testuale esatta va verificata nel sorgente di `resource_llm_agents.py`.

Una possibile evoluzione futura è esternalizzare anche i prompt generali in file versionati. Non è necessaria per completare l'attuale integrazione.

#### 4.3 Profili MCP/retrieval

`resource_mcp_profiles_v1.json` contiene le configurazioni utilizzate dal server MCP.

**`movielens_gte_fast_v2`**

- GTE multilingual dense;
- ricerca vettoriale Qdrant;
- top-10 risultati;
- nessun reranking.

**`movielens_gte_quality_v2`**

- GTE multilingual dense;
- top-20 candidati;
- GTE multilingual reranker;
- top-10 risultati finali.

Nella versione rifattorizzata, `ResourceSearchServiceRegistry` costruisce il reranker direttamente dalla configurazione JSON. Non è necessario istanziarlo manualmente nel notebook.

Il CrossEncoder viene caricato in maniera lazy, al primo utilizzo effettivo.

Attualmente il profilo tecnico viene selezionato lato server e non dal Planner LLM. La possibilità di far scegliere fast/quality all'orchestratore è una possibile estensione sperimentale.

---

### 5. Come vengono creati e utilizzati gli agenti

Gli agenti sono componenti applicativi che utilizzano un provider LLM, istruzioni e schemi di output.

Nel notebook 11 il Planner viene costruito attraverso `ResourcePlannerAgent`, fornendogli:

- provider LLM;
- profilo di dominio;
- temperatura;
- strategia di riscrittura della query.

Il Planner viene eseguito tramite `run(...)` e restituisce un piano strutturato e una trace.

Il workflow completo viene invece creato tramite:

`build_resource_assistant_workflow(...)`

definito in `resource_llm_workflow.py`.

La factory riceve, tra le altre informazioni:

- profilo di dominio;
- provider del Planner;
- provider dell'Answer Agent;
- gateway MCP;
- identificativo del workflow;
- limite dei risultati;
- strategia di riscrittura della query.

Una volta costruito il workflow, una richiesta viene elaborata tramite:

`await workflow.run(query)`

Gli agenti non devono necessariamente comunicare direttamente tra loro. Il codice Python mantiene lo stato e passa a ciascun agente le informazioni necessarie.

---

### 6. Stato condiviso e comunicazione tra componenti

Il notebook 11 dispone già di uno stato di esecuzione che conserva le informazioni prodotte durante il workflow.

Tra i campi osservabili:

| Campo | Contenuto |
|---|---|
| `state.plan` | Piano strutturato prodotto dal Planner |
| `state.search_response` | Risultati e metadata del retrieval MCP |
| `state.final_answer` | Risposta finale e raccomandazioni |
| `state.llm_traces` | Trace delle chiamate ai modelli LLM |
| `state.tool_traces` | Trace delle chiamate agli strumenti |
| `state.grounding_violations` | Eventuali violazioni dei vincoli di grounding |
| `state.status` | Stato dell'esecuzione |

Lo stato è gestito dal workflow e può essere esteso in futuro.

Per esempio, l'introduzione di una fase di elicitation più articolata potrebbe richiedere un `RequestProfile`, mentre un agente di validazione potrebbe produrre un `validation_result`.

Questi elementi non sono ancora componenti obbligatori della baseline corrente.

**Differenza tra piano e stato:**

- Il piano descrive come effettuare una determinata operazione.
- Lo stato conserva il piano insieme a tutti gli altri risultati intermedi e finali.

---

### 7. Gestione delle casistiche già implementate

La pipeline è prevalentemente fissa, ma contiene già alcuni rami condizionali.

**Clarification**

Se il Planner segnala che manca un'informazione indispensabile, il workflow può interrompere la ricerca senza invocare MCP.

Non è ancora un ciclo conversazionale completo che riceve una risposta dell'utente e aggiorna il profilo.

**No results**

Se il retrieval non restituisce risultati, il sistema non deve inventare risorse esterne al catalogo.

**Grounding**

L'Answer Agent lavora sui candidati recuperati. Il workflow controlla le raccomandazioni restituite e registra eventuali identificativi fuori dall'insieme consentito.

Queste funzionalità costituiscono una baseline verificabile su cui costruire l'orchestrazione futura.

---

### 8. Componenti legacy e compatibilità

Il package contiene ancora classi e moduli sviluppati per il precedente workflow MovieLens.

La nuova architettura `resource_*` è quella da utilizzare per i prossimi sviluppi, ma non sostituisce necessariamente tutte le infrastrutture esistenti.

In particolare:

- i precedenti agenti/workflow MovieLens rimangono utili per compatibilità con i notebook più vecchi;
- l'infrastruttura dei provider LLM viene ancora riutilizzata;
- i vecchi file sotto `src/movie_rag/prompts/` appartengono alla precedente organizzazione dei prompt; prima di eliminarli occorre verificare eventuali riferimenti ancora attivi;
- gli adapter specifici MovieLens rimangono necessari finché i test utilizzano quel dataset.

Non è prevista, in questa fase, la rimozione dei componenti legacy.

---

### 9. Notebook di riferimento e test

| File | Scopo |
|---|---|
| `10_qdrant_gte_resource_mcp.ipynb` | Integrazione di GTE, Qdrant, `RetrievalService` e MCP generico. |
| `11_resource_llm_mcp_integration_refactored.ipynb` | Test end-to-end della pipeline Planner → MCP → Answer, incluse configurazioni fast/quality, stato e trace. |
| `tests/test_resource_llm_integration.py` | Test dell'integrazione LLM/resource e dei relativi contratti. |

Il notebook 11 è attualmente il riferimento principale per verificare il comportamento della baseline.

**Controlli utili durante lo sviluppo:**

- coerenza tra richiesta originale, query semantica e filtri generati;
- correttezza dei risultati recuperati;
- assenza di raccomandazioni fuori catalogo;
- gestione di richieste incomplete e assenza di risultati;
- correttezza delle trace;
- confronto fra configurazioni con e senza reranking;
- eventuali differenze di latenza e token.

---

### 10. Evoluzione prevista: orchestrazione dinamica

La pipeline corrente può essere sintetizzata come:

`Planner → Retrieval → Answer`

Questa è una baseline fissa, domain-agnostic e già utilizzabile.

Il prossimo contributo progettuale riguarderà la possibilità di adattare il workflow alla richiesta.

Le estensioni possibili comprendono:

**Elicitation più articolata:** estrarre bisogni, preferenze, vincoli e ambiguità, eventualmente richiedendo informazioni aggiuntive.

**RequestProfile:** rappresentare in forma strutturata le informazioni disponibili sull'utente, indipendentemente dalla specifica operazione di retrieval.

**Orchestratore condizionale:** decidere quali agenti attivare, in quale ordine e quando interrompere il processo.

**Validation:** controllare la qualità e la coerenza delle raccomandazioni rispetto ai requisiti.

**Refinement:** eseguire eventuali nuovi tentativi di retrieval in presenza di risultati insufficienti.

Queste funzionalità dovranno essere introdotte mantenendo quanto possibile invariata l'infrastruttura MCP/Qdrant.

Dal punto di vista sperimentale, la baseline attuale rappresenta anche un riferimento per un futuro confronto fra pipeline fissa e orchestrazione condizionale.

---

### 11. Porting a WhoTeach

Il progetto include un profilo `whoteach_template`, ma non rappresenta ancora un dominio operativo.

Il trasferimento richiederà:

1. definizione dello schema effettivo delle risorse educative;
2. ingestion/chunking e mapping verso `ResourceDocument`;
3. nuova collection Qdrant e profilo MCP;
4. configurazione di filtri e istruzioni specifiche di WhoTeach;
5. eventuali agenti o funzionalità pedagogiche aggiuntive;
6. nuova valutazione del retrieval sul corpus didattico.

L'obiettivo è riutilizzare `RetrievalService`, MCP gateway e core applicativo, adattando le componenti specifiche del dominio.

---

### 12. Guida rapida: dove intervenire

| Se voglio... | File principale |
|---|---|
| Modificare le istruzioni del Planner per MovieLens | `configs/resource_domains_v1.json` |
| Cambiare il modello LLM | `configs/llm_profiles_v1.json` |
| Modificare il comportamento generale di un agente | `resource_llm_agents.py` |
| Modificare gli schemi degli output LLM | `resource_llm_models.py` |
| Aggiungere uno step o un ramo al workflow | `resource_llm_workflow.py` |
| Modificare un tool MCP | `resource_mcp.py` |
| Modificare come il workflow chiama MCP | `resource_mcp_gateway.py` |
| Cambiare profilo di ricerca, encoder o reranker | `configs/resource_mcp_profiles_v1.json` |
| Modificare l'inizializzazione del retrieval | `resource_mcp_profiles.py` |
| Modificare ranking/retrieval service | `resource_service.py` |
| Modificare la ricerca vettoriale | `resource_qdrant.py` |
| Modificare il CrossEncoder | `resource_rerankers.py` |
| Aggiungere il dominio WhoTeach | `resource_domain.py`, configurazioni e adapter |

**Nota finale:** questa guida descrive la baseline verificata nei notebook 10–11 e le sue estensioni previste. Le parti relative a `RequestProfile`, orchestratore dinamico e agenti aggiuntivi rappresentano sviluppi futuri, non funzionalità già completate.
