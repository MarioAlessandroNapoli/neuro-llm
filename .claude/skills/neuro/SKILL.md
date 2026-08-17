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

## Principi da far rispettare (non negoziabili senza revisione esplicita)

- **Meccanismi, non estetiche** (lezione HRM/TRM): ogni claim "neuro" deve avere
  un'ablazione che lo isola.
- **Apparato di misura congelato**: tokenizer, dati e valutazione identici tra le braccia
  del confronto (D3, D4). Mai confrontare curve prodotte da tokenizzazioni diverse.
- **Parità**: parametri (±5%) e token budget uguali tra architetture; ≥2-3 seed;
  media e range riportati; il segno del risultato si dichiara sempre, anche se negativo (D6).
- **Baseline nostra**: mai numeri di terzi come termine di confronto (D5).

## Comunicazione dei risultati

Findings in registro narrativo umano: prima il modello di dominio, poi il comportamento
in parole, il riferimento tecnico come prova in coda. Tabelle solo per confronti e numeri.

## Operatività esperimenti

Per lanciare, monitorare e verificare run (Kaggle, W&B, checkpoint): usare la skill
`ml-dev` — questa skill decide *cosa e perché*, quella *come*.
