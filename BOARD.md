# Board

Stato e priorità del progetto. Regole di gestione nella skill `neuro`.
Now = in lavorazione (max 2-3) · Next = pronto a partire · Later = deciso, non prossimo · Done = compresso.

**Budget compute**: 10 $ — spesi 3,42 $ (1a: 3090+4080 ~2,68 $ · 1b+asintoto:
3060+4070 ~0,74 $) · residui 6,58 $. **Zero istanze attive** (tutte distrutte,
2026-08-19 notte).

## Now

- **D16 congelata** (griglia char: backbone byte seq1024, fasi 0→sweep→A/B/C, budget
  700M/2,2B byte). In corso: implementazione fase 0-char (pipeline byte, configs,
  smoke M2) → collaudo su GPU piccola → sweep+griglia su GPU potente (budget
  ricaricabile, ok da utente 2026-08-20)

## Next

- Campagna giudice completa (D14): generazioni ≥2 seed per braccio su istanza GPU vast
  (M2 troppo lento: ~2h/run per l'ibrido), poi Elo con la regola E± pre-registrata
  (il bootstrap cluster richiede ≥2 seed) + scoring assoluto e self-agreement

## Later

- Candidate 1c su BPE parcheggiate da D15 (rassegna in docs): transizione selettiva
  r/θ disaccoppiata · curva di sostituzione attention · catena osc-osc senza MLP ·
  sonde sintetiche parity/mod-3/MQAR
- Sanity check del potere del giudice cieco (D14-riconsiderare): una coppia a loss
  distanti (es. ibrido 1,57 vs dlinoss-lp 1,95) — il giudice la distingue? Mai misurato
- Griglia 1b, fasi 1-2 (D12): dlinoss log-polare + sweep lr · omeostasi vs controllo
  log-polare · ibrido intercalato A-O · hyb-oa@3e-3 · init post-autopsia r~U[0,7;0,9] ·
  asintoto 536M per i vincitori. Fuori scope dichiarato: selettività/gating (1c)
- Stadio 2: BabyLM Strict-Small (10M parole) + valutazione BLiMP per la variante migliore
- Ablazione granularità temporale (Q3: char-level o chunking appreso)
- Probe diagnostici esplorativi (D7: non cambiano mai il verdetto dello stadio 1):
  name-cloze a distanza, ablazione del contesto, probe spettrale degli hidden state
  (per dlinoss-phi: verifica se la struttura a bande sopravvive al training)

## Done

- 2026-08-20 (notte) — **Giudizio cieco preliminare D14**: canale in-sessione (188
  giudici Opus 5 naive, costo API zero), coppia asintoto s1 → **parità qualitativa**
  (82/77/29, p=0,75), modi di fallire identici tra le architetture; conferma
  indipendente della parità vista dalla loss. Generazione batchata (~10×)
- 2026-08-19 (notte) — **STADIO 1 CHIUSO** (D13 + esiti): parità strutturale
  ibrido-transformer a 170M e 536M (1,509 vs 1,497); baseline onesta 1,558; tre
  meccanismi da autopsia; 3,42 $ totali. Archivio: docs/archive/2026-08-griglia-1a.md
  e 2026-08-griglia-1b-e-fase0.md; cronologia in docs/TIMELINE.md

- 2026-08-19 — **Griglia 1a chiusa e autopsiata** (D11+D12): due tasse (hyb-ao = solo
  addestrabilità, dlinoss = anche espressività; controllo lr-matched 1,700), oblio
  necessario-non-sufficiente, gerarchia inversa vince, gate phi FAIL; autopsia: r
  tirato giù dal training, gerarchia timescale emergente, guasto = perdita smorzamento
- 2026-08-18 — **Sei bracci implementati** (review avversariale: 3 difetti corretti);
  smoke+sweep completi su vast 3090 (~2 $); revisione D6-lr (tutte le categorie oltre il
  vecchio bordo); verdetti: linoss instabile (=risultato asse 2), wrnn negativo di
  porting; primi segnali dlinoss≈baseline e hyb-ao>hyb-oa. Dettagli: D11.

- 2026-08-18 — **Fase 1 chiusa: baseline e apparato numerico completi.** Baseline 8,5M:
  5 seed a 170M, media 1,8266, **ε = 0,017** (base-1..5); lr 1e-3, budget 170M (D8),
  config DDP b16×2+compile (176k tok/s), àncora esterna BPB 0,4407, pilot epoch piena
  1,509. Dettagli: `docs/archive/2026-08-fase1-baseline-e-apparato.md`
- 2026-08-18 — D9 (cinque assi di design da Castellanos cap. 1) + D10 (griglia 1a:
  sei bracci, equazioni verificate sui paper, deviazioni dichiarate)
- 2026-08-17 — Apparato D7 implementato e validato E2E (`src/eval/`): prompt set
  congelati, generazione, giudice Opus 5 via Batches, statistica pre-registrata;
  manca solo il round-trip API reale (Next)
- 2026-08-17 — Rassegna del campo, decisioni fondative D1-D6, ancore quantitative,
  review avversariale del disegno (RESEARCH_LOG)
- 2026-08-17 — Pipeline dati (BPE 8k, 536M token su HF) + infrastruttura E2E
  (dettagli: `docs/archive/2026-08-setup-infrastruttura.md`)
