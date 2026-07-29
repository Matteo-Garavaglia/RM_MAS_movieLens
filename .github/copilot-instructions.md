# Copilot Instructions 
# Sunto tesi magistrale – Daniele Raveglia

## Contesto personale e accademico
Daniele Raveglia è iscritto alla Laurea Magistrale in Informatica presso l’Università degli Studi di Milano. L’obiettivo è completare il percorso con una tesi ibrida, a metà tra compilativa e sperimentale, con un perimetro sostenibile e progressivamente aggiornabile.

## relatore e collaboratori
il relatore della tesi è Stefano Valtolina
i collaboratori lato tecnico sono: Matteo Garavaglia, Ricardo Matamoros
i collaborotari sono dipendenti di SocialThingum che ha creato anche WhoTeach

## Tema generale della tesi
La tesi riguarda l’uso di LLM, Retrieval-Augmented Generation e sistemi multi-agente per supportare i docenti nella progettazione didattica e nella selezione di risorse educative in scenari di cold-start, cioè quando il sistema non dispone ancora di storico utente, profili consolidati o dati comportamentali sufficienti.

## Posizionamento rispetto a WhoTeach
Il contributo si colloca prevalentemente a monte o nella fase iniziale di authoring di WhoTeach. WhoTeach assume che il docente definisca corso, obiettivi, topic, risorse e indicazioni pedagogiche, per poi costruire un tutor conversazionale basato su knowledge space controllato dal docente, RAG e adattamento pedagogico. La tesi di Daniele mira invece a supportare il docente quando parte da una richiesta iniziale ancora grezza o incompleta, aiutandolo a trasformarla in un piano didattico strutturato, corredato da obiettivi, prerequisiti, topic, attività e risorse consigliate.

In sintesi, il modulo proposto può essere inteso come un pre-authoring assistant integrabile con WhoTeach: input del docente → analisi del bisogno → piano didattico → recupero e selezione di risorse → spiegazione delle scelte → possibile esportazione verso WhoTeach.

## Idea centrale
L’idea principale è reinterpretare il cold-start non solo come mancanza di dati storici, ma come problema di orchestrazione decisionale. Il sistema non deve inferire preferenze inesistenti, ma decidere dinamicamente quali agenti attivare, quali fonti interrogare e quali strategie seguire a partire dai segnali semantici presenti nella richiesta iniziale del docente.

Il titolo provvisorio può essere formulato come:

**Orchestrazione Cold-Case per Sistemi Multi-Agente di Supporto alla Progettazione Didattica in Ambienti Teacher-Centred**

oppure, in forma più centrata sul posizionamento a monte:

**Un sistema multi-agente per il supporto al pre-authoring didattico in scenari cold-start**

## Architettura concettuale
La tesi prevede un’architettura composta da agenti specializzati coordinati da un orchestratore centrale:

- Agente di elicitation: interpreta la richiesta iniziale del docente ed estrae disciplina, livello, obiettivi, prerequisiti, vincoli pedagogici, tipologia di risorse e ambiguità.
- Agente di planning didattico: propone una possibile struttura del percorso/modulo didattico.
- Agente di retrieval: recupera risorse, learning objects o materiali coerenti con il bisogno didattico.
- Agente di ranking/reranking: valuta e riordina i candidati secondo criteri semantici, pedagogici e contestuali.
- Agente di spiegazione: motiva le scelte fatte e rende trasparente il processo.
- Orchestratore: decide quali agenti attivare, in quale ordine, quando chiedere chiarimenti e quando produrre l’output finale.

## Perimetro ridotto desiderato
La tesi dovrebbe evitare un’implementazione completa di WhoTeach o di un sistema multi-agente avanzato in stile HERA. Il perimetro preferibile è: progettazione del modello, prototipo leggero/offline e valutazione preliminare. Sono idealmente fuori perimetro: fine-tuning di modelli, training complesso, experience library evolutiva, prompt evolution automatica, user study esteso e integrazione completa in produzione.

## Possibile output del sistema
Dato un input come “voglio progettare un modulo sulla ricorsione per una classe quarta, con attività pratiche e materiali visuali”, il sistema dovrebbe produrre una proposta strutturata contenente:

- obiettivi formativi;
- prerequisiti;
- articolazione in lezioni o unità;
- topic principali;
- attività suggerite;
- risorse recuperate e motivate;
- eventuali domande di chiarimento se la richiesta è ambigua.

## Valutazione possibile
Una valutazione sostenibile potrebbe confrontare una pipeline fissa con un orchestratore condizionale su un insieme limitato di richieste simulate di docenti. Le metriche possono riguardare appropriatezza del workflow scelto, coerenza del piano didattico, pertinenza delle risorse, qualità delle spiegazioni, gestione di ambiguità e vincoli, e costo approssimato in step/token.

## File e fonti del progetto
- `proposta.pdf`: proposta iniziale del relatore. Definisce il paradigma di Cold-Case Orchestration, l’idea di orchestrazione dinamica e una possibile architettura multi-agente con elicitation, retrieval, ranking/reranking e spiegazione.
- `interact.pdf`: paper su WhoTeach. Descrive il sistema attuale come framework teacher-driven per creare tutor conversazionali personalizzati basati su risorse controllate dal docente, RAG, knowledge grounding e adattamento pedagogico.
- `TESI_MANZARI.pdf`: tesi precedente collegata a WhoTeach. Presenta un sistema di raccomandazione edutech ottimizzato con AI generativa, con pipeline ibrida warm-start/cold-start, dataset surrogato Amazon Books Reviews, NCF, LLM e reranking/spiegazioni.
- `paperCOPDA.pdf`: paper del relatore sul cold-start recommendation. Reinterpreta il cold-start come knowledge orchestration e propone un framework human-centred con elicitation semantica, retrieval-grounded candidate generation, LLM-based reranking e spiegazione.
- `HERA.pdf`: paper di ispirazione sui sistemi Multi-Agent RAG con orchestrazione evolutiva, experience library, prompt evolution e topologie agenti. È utile come riferimento teorico avanzato, ma non rappresenta necessariamente il perimetro implementativo della tesi.
- `info - università - carriera Daniele.rtf`: informazioni sul percorso universitario di Daniele, situazione esami, obiettivo di laurea e preferenza per una tesi sostenibile e ibrida.

## Formula sintetica da usare nelle chat future
La tesi di Daniele riguarda un sistema multi-agente leggero, integrabile a monte di WhoTeach, per supportare il docente nella fase di pre-authoring didattico. A partire da una richiesta iniziale in linguaggio naturale, il sistema estrae bisogni, vincoli e obiettivi, costruisce un possibile piano didattico, recupera risorse pertinenti e spiega le scelte. Il contributo si basa sul paradigma della Cold-Case Orchestration: in assenza di storico utente, l’orchestratore decide dinamicamente quali agenti attivare e quale workflow seguire in base alla richiesta corrente.

# implementazione
l'idea è di partire da un dataset surrogato e una volta definita e valutata l'architettura sul dataset surrogato, poi calare lo studio sui dati di whoteach sarà semplicemente un ridefinire i prompt LLM.

## dataset
per inizializzare il progetto e per validare l'idea dell'architettura è stato proposto da Matteo Garavaglia l'utilizzo del dataset alternativo MovieLens: https://grouplens.org/datasets/movielens/
in particolare la versione "recommended for education and development" il cui README di riferimento è: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html
come riportato nel README: 
- This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.

il dataset oltre a dati strutturati ha anche una descrizione piuttosto ridotta dei film, che però è abbastanza ridotta, sarebbe interessante riuscire ad integrare altre informazioni riguardo i film presenti

## progetto
voglio utilizzare python e notebooks jupiter per analizzare il dataset dei film e per arricchirlo.