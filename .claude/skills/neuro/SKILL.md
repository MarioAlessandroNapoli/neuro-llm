---
name: neuro
description: Use when working on the neuro-llm project — session start, taking or revisiting scientific decisions, updating priorities/board/task state, closing or opening research questions, recording experiment results, or when unsure how current work fits the big picture.
---

# neuro — Science & Process Coordinator

Coordinatore scientifico e di processo del progetto neuro-llm. Custodisce la big picture:
la domanda di ricerca, le decisioni prese e il loro perché, le priorità correnti.

## Fonti di verità (leggere in quest'ordine a inizio sessione)

| File | Contiene | Regola |
|---|---|---|
| `BOARD.md` | Stato e priorità (Now/Next/Later/Done) | Sempre aggiornata a fine sessione di lavoro |
| `RESEARCH_LOG.md` | Decisioni scientifiche (D*) e questioni aperte (Q*) | Solo scienza; engineering nel README/git |
| `docs/archive/` | Storico non più critico per il contesto attuale | Sola lettura; si consulta, non si ricarica |

Mai ri-derivare o ri-litigare una decisione D* chiusa: se serve cambiarla, si apre
esplicitamente una revisione citando la condizione di riconsiderazione scritta nel log.

## Procedure

**Nuova decisione scientifica** → entry D* nel RESEARCH_LOG con il template: Decisione /
Perché (con evidenze) / Scartato (con motivo) / Riconsiderare se. Una Q* che si chiude
diventa una D*; la Q si rimuove.

**Board.** Colonne: Now (max 2-3 item, ciò su cui si lavora), Next (pronto a partire),
Later (deciso ma non prossimo), Done (compresso, una riga). Ogni sessione che cambia lo
stato del progetto tocca la board prima di chiudere.

**Archiviazione (scarico del contesto).** Test per ogni blocco di BOARD/RESEARCH_LOG:
"serve per prendere una decisione oggi o nel prossimo ciclo?" Se no → si sposta in
`docs/archive/YYYY-MM-argomento.md` lasciando al suo posto una sola riga di puntatore.
Esempio già fatto: il setup infrastrutturale vive in `docs/archive/2026-08-setup-infrastruttura.md`.

**Registro esperimenti.** Ogni run con valore scientifico aggiunge una riga alla tabella
in fondo al RESEARCH_LOG (dettagli su W&B, qui solo l'essenziale). Gli smoke non contano.

**Autopsia dei modelli (step fondamentale, mai opzionale).** A chiusura di ogni ciclo di
esperimenti — e sempre prima di disegnare il ciclo successivo — si aprono i pesi dei
checkpoint e si guarda cosa il training ha *fatto*, non solo quanto ha perso: parametri
interpretabili (per gli oscillatori: r, θ, saturazione dei clamp), norme dei percorsi
(un modulo può vincere perché lavora o perché è stato silenziato), confronto init→appreso,
e i checkpoint delle run fallite/degenerate, che sono dati e non scarti (la firma del
guasto pesa le ipotesi quanto un braccio riuscito). I nostri modelli sono piccoli e
osservabili: l'autopsia è oro a costo ~zero (CPU locale, pesi su HF) e ogni suo esito va
nel RESEARCH_LOG come pre-step del design successivo. Esempio canonico: l'autopsia
spettrale della 1a in D12-design-1b (r tirato giù dal training, gerarchia di timescale
emergente, firma del guasto = perdita di smorzamento, hyb-oa morto all'init).

## Principi da far rispettare (non negoziabili senza revisione esplicita)

- **Meccanismi, non estetiche** (lezione HRM/TRM): ogni claim "neuro" deve avere
  un'ablazione che lo isola.
- **Apparato di misura congelato**: tokenizer, dati e valutazione identici tra le braccia
  del confronto (D3, D4). Mai confrontare curve prodotte da tokenizzazioni diverse.
- **Parità**: parametri (±5%) e token budget uguali tra architetture; ≥2-3 seed;
  media e range riportati; il segno del risultato si dichiara sempre, anche se negativo (D6).
- **Baseline nostra**: mai numeri di terzi come termine di confronto (D5).

**Documentazione e artefatti (skill `docs-and-artifacts`, orchestrata da qui).**
Neuro decide il momento, quella skill il contenuto. Trigger da applicare senza
eccezioni: fine sweep/braccio → tabella nel log; finding principale → pacchetto
figura+script+riproducibilità+timeline in un commit; lezione ingegneristica → gotcha
+ script-evidenza; **fine ciclo/griglia → pacchetto di chiusura** (archiviazione,
memoria di progetto, timeline, verifica artefatti) — che è anche il momento
programmato per lo scarico del contesto di sessione. `docs/TIMELINE.md` porta le date:
il progetto è giovane e la cronologia è parte del racconto scientifico.

## Comunicazione dei risultati

Findings in registro narrativo umano: prima il modello di dominio, poi il comportamento
in parole, il riferimento tecnico come prova in coda. Tabelle solo per confronti e numeri.

**Sigle sempre etichettate.** Mai citare una D*/Q* nuda: ogni menzione porta un'etichetta
breve che ne richiama il contenuto (es. «Q1-backbone», «D7-valutazione», «Q4-budget»).
L'utente non ricorda le sigle a memoria; l'etichetta è il gancio mnemonico. Etichette
canoniche: D1-due-stadi, D2-candidati, D3-tokenizer, D4-asse-token, D5-baseline-nostra,
D6-parità, D7-valutazione, Q1-backbone, Q3-granularità, Q4-budget.

## Operatività esperimenti

Per lanciare, monitorare e verificare run (Kaggle, W&B, checkpoint): usare la skill
`ml-dev` — questa skill decide *cosa e perché*, quella *come*.
