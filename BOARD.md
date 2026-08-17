# Board

Stato e priorità del progetto. Regole di gestione nella skill `neuro`.
Now = in lavorazione (max 2-3) · Next = pronto a partire · Later = deciso, non prossimo · Done = compresso.

## Now

- **Design stadio 1 — chiudere Q1** (quale backbone oscillatorio, in quale forma, con
  quale ablazione). Output atteso: D8 nel RESEARCH_LOG.
- **Baseline transformer 8,5M** — sweep lr pre-registrato (3 valori × ~20M token, 1 seed)
  poi 5 seed al lr congelato (σ della val loss → fissa ε di D7): non dipende da Q1; il suo
  costo/step è uno degli input di Q4 (che richiede anche quello delle varianti).

## Next

- Rigenerare i token HF e Kaggle (transitati in chiaro nei log della sessione di setup:
  HF settings/tokens; Kaggle settings/API) — pochi minuti
- Validare il giudice con una chiamata reale (serve `ANTHROPIC_API_KEY` nell'ambiente:
  la build delle richieste è già verificata, manca solo il round-trip API)
- Misurare BPB del checkpoint pubblico stories15M sul nostro val set, a contesto 256 e
  con i caveat V1/V2 dichiarati (condizioni in RESEARCH_LOG § Ancore quantitative)
- Implementare le architetture scelte in `src/models/` (registry `ARCHS`), parità
  parametri col baseline ±5%
- Sweep lr delle varianti + griglia stadio 1 su Kaggle — dopo D8/Q4 (bracci e budget);
  la parte baseline dello sweep è in Now

## Later

- Stadio 2: BabyLM Strict-Small (10M parole) + valutazione BLiMP per la variante migliore
- Ablazione granularità temporale (Q3: char-level o chunking appreso)
- Probe diagnostici esplorativi (D7: non cambiano mai il verdetto dello stadio 1):
  name-cloze a distanza, ablazione del contesto, probe spettrale degli hidden state

## Done

- 2026-08-17 — Harness D7 implementato e validato E2E (`src/eval/`): prompt set congelati
  (44 ufficiali verbatim + 50 prefissi lunghi, pool 1046 = valore atteso), generazione
  temp 1 × 10 con seed derivati e artefatti JSON, giudice Opus 5 via Batches (scoring
  batch + Elo set-vs-set cieco, dry-run ok), statistica (permutazione esatta, BT+bootstrap
  clusterizzato, BPB) verificata su casi noti; manca solo il round-trip API reale (Next)
- 2026-08-17 — Emendamento D7: giudizio pairwise set-vs-set (10 vs 10, un verdetto per
  coppia/prompt/ordine), scoring batch mono-braccio con score per-completamento, strato
  lungo fissato a 50 prefissi → 188 verdetti per coppia
- 2026-08-17 — Review avversariale del disegno (7 finding confermati) → tutti i 6 assi
  chiusi: D6 esteso (ricetta per categoria, sweep lr, parità sul backbone 6,45M con
  enforcement in train.py), D7 riscritta (tabella di verdetto, statistica, protocollo
  generazione e giudice interamente sourced, strato lungo custom), identità run con
  seed/budget/lr + resume esplicito fail-fast, assert tokenizer, registry per-arch,
  ckpt smoke archiviato su HF, igiene documentale (candidati, BPB, archivio)
- 2026-08-17 — D7: apparato valutazione stadio 1 + ipotesi falsificabile (Q2 chiusa)
- 2026-08-17 — Ancore quantitative + apparato di valutazione del campo raccolti e
  verificati con fonti (RESEARCH_LOG § Ancore quantitative)
- 2026-08-17 — Setup infrastruttura completo e collaudato E2E in cloud
  (dettagli: `docs/archive/2026-08-setup-infrastruttura.md`)
- 2026-08-17 — Rassegna del campo + decisioni fondative D1-D6 (RESEARCH_LOG)
- 2026-08-17 — Pipeline dati: BPE 8k, 536M token su HF
