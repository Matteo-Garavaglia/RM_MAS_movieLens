# Orchestrazione iterativa v1: installazione, controlli e limiti

> **Provenienza:** `README_INSTALLAZIONE(1).md` · **Tipo:** documentazione tecnica del progetto.
> **Collocazione nel percorso:** Estensione additiva della baseline 11; smoke test offline/mocks, integrazione live da verificare.
> **Nota:** i contenuti tecnici sottostanti sono stati conservati dalla documentazione fornita;
> non costituiscono una nuova verifica dell’esecuzione del codice nel repository corrente.

## Ambito del pacchetto

Estensione **additiva** rispetto al codice identificato nel notebook 11 rifattorizzato (Library, 8 ottobre 2026). Questa versione non altera alcun file del progetto, né il notebook 11 né il flusso parallelo di benchmark collaborativo. Il codice qui consegnato è stato validato con test offline/mocks, ma **non** con la tua istanza Qdrant e con le API reali: non era disponibile una copia materializzabile dell'intero repository aggiornato.

## Installazione

Dalla root del tuo repository:

1. Copiare `src/movie_rag/resource_iterative_orchestration.py` in `src/movie_rag/`.
2. Copiare `notebooks/orchestration_iterative_v1.ipynb` in `notebooks/`. Non si usa un prefisso numerico per evitare collisioni con i notebook creati nell'altra chat.
3. Copiare `tests/test_resource_iterative_orchestration.py` in `tests/`.
4. Verificare che il notebook 11 e Qdrant funzionino come prima. Non sostituire `resource_llm_workflow.py`, `resource_llm_agents.py` o il registry MCP.
5. Da root: `python -m unittest discover -s tests -p 'test_resource_iterative_orchestration.py' -v`.
6. Avviare il notebook. Impostare `RUN_LIVE=True` solo dopo i test offline.

## Switch principali

- `WORKFLOW_MODE = "linear" | "iterative"`. La modalità lineare invoca `build_resource_assistant_workflow` esistente senza cambiarlo.
- `LLM_PROVIDER = "gemini" | "openai"`. Gemini usa il `GeminiProvider` già presente. OpenAI usa `OpenAITextProvider` con `openai.AsyncOpenAI` (`pip install openai`) e `OPENAI_API_KEY` nel `.env`. La scelta del modello OpenAI è in `OPENAI_MODEL`.
- `ENABLE_RERANK=False`: `movielens_gte_fast_v2`, senza rerank locale. `True`: `movielens_gte_quality_v2`, reranker gestito dalla configurazione MCP **server-side**. Il quality profile richiede il blocco `reranker` nella configurazione aggiornata l'8 ottobre; NON è esposto come decisione dell'LLM.
- `MAX_RETRIEVALS`, `MAX_CLARIFICATIONS`, `MAX_LLM_CALLS`, `MAX_DECISIONS`, `MAX_WALL_SECONDS`. `MAX_TOTAL_TOKENS` è **best-effort**, funziona solo quando il provider riporta usage esplicita; se manca usage `token_usage_complete=False`. I limiti sulle chiamate sono applicati dal controller.
- `USER_REPLY`: in caso di `WAITING_USER`, l'utente può riprendere il processo mantenendo il kernel aperto. Non è attivo un checkpoint/restart completo.

## Stato e tracciamento

`IterativeState` contiene `profile` (dizionario semantico del dominio), `base_filters`, `conversation`, `attempts`, `candidates` cumulativi con ID e provenance delle query, `validations`, `events`, `llm_traces`, `final_answer` e `termination_reason`. La persistenza viene eseguita atomicamente in `outputs/orchestration_runs/<run_id>/{state.json,metadata.json}`: **aggiungere `outputs/orchestration_runs/` a `.gitignore`** e considerare che il testo delle richieste può contenere dati personali. Credenziali non vengono memorizzate intenzionalmente.

## Limiti attuali e verifiche necessarie

- Il notebook 11 usa il gateway MCP tipizzato; il flusso iterativo chiama **lo stesso tool MCP** tramite `Client.call_tool`, normalizzando gli output `results`. Verificare in live che la versione installata del server restituisca proprio questo contratto; in caso contrario allineare soltanto l'adapter MCP.
- Il controller non è specifico dei film. La root del notebook ha configurazione MovieLens per la demo; WhoTeach richiede i veri schemi e metadati del suo corpus, non soltanto la sostituzione del prompt.
- La `Validation` attuale è una valutazione LLM semantica senza ground truth. Le verifiche tecniche sugli ID sono deterministiche. Non si può garantire automaticamente la correttezza fattuale dell'intero testo di risposta.
- Il nuovo percorso usa un prompt Answer leggero dedicato, mentre quello lineare mantiene l'Answer Agent originale: **prima di un confronto scientifico controllato bisognerà allineare la generazione della risposta**, oltre a modelli, input e budget. Il benchmark cold-start vs collaborative resta un filone sperimentale separato.
- `clarification_required` rilevato dal Planner del notebook 11 porta a `WAITING_USER`; il nuovo ciclo non tenta di risolvere da solo contraddizioni arbitrarie fra molte risposte. Non è prevista memory/experience library cross-user.
- La scelta dell'azione è autonoma, ma il loop è una struttura fissa: Elicitation iniziale → [Retrieval → Validation → Decision] fino a FINISH/CLARIFY/budget → Answer. Non è pianificazione libera di grafi multi-agente.

## Smoke test confermati (offline)

Test eseguiti: due retrieval cumulativi; mantenimento filtri; pause/resume chiarimento; blocco query duplicate; whitelist di ID raccomandati; risposta priva di risorse; salvataggio trace; normalizzazione MCP dict. Nessun test end-to-end API/Qdrant eseguito su questo runtime.
