# Board

Stato e priorità del progetto. Regole di gestione nella skill `neuro`.
Now = in lavorazione (max 2-3) · Next = pronto a partire · Later = deciso, non prossimo · Done = compresso.

## Now

- **Design esperimento stadio 1** — chiudere Q1 (quale backbone oscillatorio, in quale
  forma, con quale ablazione) e Q2 (ipotesi falsificabile + probe oltre la loss).
  Output atteso: D7 e D8 nel RESEARCH_LOG.

## Next

- Implementare le architetture scelte in `src/models/` (registry `ARCHS`), parità
  parametri col baseline ±5%
- Aggiungere `--seed` a `train.py` (prerequisito griglia multi-seed, vedi gap in `ml-dev`)
- Griglia stadio 1: 3 arch × 2-3 seed × budget da fissare (Q4) su Kaggle

## Later

- Stadio 2: BabyLM Strict-Small (10M parole) + valutazione BLiMP per la variante migliore
- Ablazione granularità temporale (Q3: char-level o chunking appreso)
- Probe spettrale degli hidden state (se Q2 la include)

## Done

- 2026-08-17 — Setup infrastruttura completo e collaudato E2E in cloud
  (dettagli: `docs/archive/2026-08-setup-infrastruttura.md`)
- 2026-08-17 — Rassegna del campo + decisioni fondative D1-D6 (RESEARCH_LOG)
- 2026-08-17 — Pipeline dati: BPE 8k, 536M token su HF
