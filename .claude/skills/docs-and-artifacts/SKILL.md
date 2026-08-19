---
name: docs-and-artifacts
description: Use when a scientific milestone needs to be captured as durable material — a finding gets a figure, a sweep/grid closes, an engineering lesson deserves evidence, or the context/memory needs offloading. Orchestrated by the neuro skill at its scheduled trigger points.
---

# docs-and-artifacts — Capitale scientifico del progetto

Trasforma i milestone in materiale durevole e riproducibile: la base per il sito
dell'esperimento (certo) e per un eventuale paper. Orchestrata da `/neuro`: è neuro a
decidere *quando* scatta (schedule sotto), questa skill dice *cosa e come* produrre.

## Dove vive cosa

| Artefatto | Posizione | Regola |
|---|---|---|
| Figure | `docs/figures/AAAA-MM-<slug>.png` (+ `-en`) | Datate; stile paper (val nel titolo, niente box sui dati); **bilingui**: lo script genera IT e EN (`[it|en|all]`, decimali "," vs "."); MAI orfane: ogni PNG è citato dal RESEARCH_LOG |
| Script di analisi/figure | `scripts/` | Ogni figura è rigenerabile: script + checkpoint HF + run W&B bastano; header con uso e dipendenze (`uv run --with ...`) |
| Timeline | `docs/TIMELINE.md` | Una riga datata (giorno + ora se nota) per milestone; si aggiorna nello stesso commit del milestone, non a posteriori |
| Riproducibilità | blocco nel RESEARCH_LOG accanto al finding | Commit hash, nomi run W&B (gruppo), checkpoint HF, ricetta completa (token, batch, lr, precisione, `NEURO_SCAN`, torch, GPU, tok/s, costo) |
| Costi | riga «Budget compute» in `BOARD.md` | Aggiornata a ogni distruzione di istanza o comunicazione saldo |
| Archivio (scarico contesto) | `docs/archive/AAAA-MM-<argomento>.md` | Procedura nella skill neuro; qui solo il promemoria che fa parte del pacchetto di fine ciclo |

## Trigger (lo schedule che neuro applica)

1. **Fine sweep o braccio** → tabella lr/val nel RESEARCH_LOG; se il verdetto è
   sorprendente, anche riga in TIMELINE.
2. **Finding principale** (cambia il quadro o motiva una direzione) → il pacchetto
   completo: figura + script in `scripts/` + blocco riproducibilità + riga TIMELINE,
   in un unico commit.
3. **Lezione ingegneristica pagata** (bug, revert, gotcha) → gotcha nella skill
   ml-dev + script-evidenza in `scripts/` se il bug merita prova (es.
   `test_bug_inductor_backward.py`) + riga TIMELINE se ha cambiato l'apparato.
4. **Fine ciclo/griglia** → il pacchetto di chiusura: (a) archiviazione dei blocchi di
   BOARD/RESEARCH_LOG non più operativi in `docs/archive/`; (b) aggiornamento della
   memoria di progetto (`project-neuro-llm-stato`); (c) TIMELINE del giorno; (d)
   verifica che ogni figura citata esista e ogni script giri. È il momento naturale
   per lo scarico del contesto della sessione.

## Principi

- **Il presente è l'unica verità** anche qui: le figure si rigenerano e sovrascrivono,
  la timeline si integra, mai duplicati versionati a mano (v2, final, ...).
- **Date sempre assolute** (il progetto corre veloce: "ieri" invecchia in ore).
- **Recupero attivo**: a ogni trigger, chiedersi «c'è materiale già prodotto ma non
  tracciato?» (script nello scratchpad di sessione, numeri solo in chat, curve solo
  su W&B) e portarlo in repo prima che la sessione lo perda.
- Lo stile delle figure: palette sobria (petrolio #0F6E6C / rosso #C0452C / grigi),
  etichette in italiano, riferimenti (baseline, ε) sempre presenti, leggibile in
  bianco e nero dove possibile.
