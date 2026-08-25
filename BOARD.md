# Board

Stato e priorità del progetto. Regole di gestione nella skill `neuro`.
Now = in lavorazione (max 2-3) · Next = pronto a partire · Later = deciso, non prossimo · Done = compresso.

**Budget compute**: ricaricabile. Stadio 1: 3,42 $. Stadio char (giorni 4-5): 3090
~0,3 $ + 5090 0,40 $/h dalle 14:45 del 20/08 (~19h al mattino del 21/08 ≈ 7,5-8 $) —
la stima iniziale 1,5-2 $ è stata superata dall'estensione notturna (B2+C1+controlli),
decisa a budget dichiarato ricaricabile. **Istanza attiva: 1** (5090 — generazioni
giudice in corso, poi distruggere).

## Roadmap "Rotta Edge" (decisa 2026-08-25; ogni fase è gate della successiva)

| fase | cosa | costo | stato |
|---|---|---|---|
| 0 | data prep Nemotron-HQ + bracci mix + eval retention | 0 $ | **IN CORSO** |
| 1 | D19 curva di sostituzione (15M, 8 run) | 10-20 $ | pronta al termine di 0 |
| 2 | scala 25-40M, mix vincente, 3-10B byte | 30-60 $ | gate: ginocchio D19 |
| 3 | paper v1.0 → TMLR+arXiv (parallela a 2) | 0 $ | — |
| 4 | modello 150M su Nemotron-HQ | 500-2k $ | gate: fase 2 verde |
| 5 | edge: int8 scan, ggml, demo RAM piatta | tempo | — |

## Now

- **Fase 0**: script Nemotron→byte; bracci osc8/mix2/mix4/cb interleaved a 15M;
  ablazione del contesto nell'eval; smoke M2
- **Scrittura paper v0.4→1.0**: appendici A-E, 2 figure, bibitem; poi review finale

## Next

- **Fase 1 — D19** (design e soglie nel log § D19): 8 run su GPU, poi tabellone,
  figura, verdetto ginocchio+schermatura
- **Giudice ufficiale best-4** (cb/hard/osc0/ts @B2, Elo 6 coppie): cache
  incrementale pronta; generazioni da rifare su GPU economica; canale da decidere
  (Batches con API key o in-sessione)

- **Griglia C1 (D17-tempo-a-eventi)**: hard-appreso (fatto) vs hard-euristico (reset
  su spazi cablati) vs gerarchia-di-timescale senza reset (Harmonic-style) — loss +
  estrapolazione + probe + ℬ + autopsia. Poi C2 (secondo livello) se C1 promuove
  l'appreso e la scala frase ha segnale (misurare BPIC e lunghezze frasi prima)
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
- Stadio 2: BabyLM Strict-Small (10M parole) + BLiMP — rimandato (deciso 2026-08-21)
- Gerarchia di reset annidati (C2): NON si fa a questa scala (deciso 2026-08-21);
  trigger di riapertura: BabyLM (frasi vere) o buco emerso in scrittura
- Ablazione granularità temporale (Q3: char-level o chunking appreso)
- Probe diagnostici esplorativi (D7: non cambiano mai il verdetto dello stadio 1):
  name-cloze a distanza, ablazione del contesto, probe spettrale degli hidden state
  (per dlinoss-phi: verifica se la struttura a bande sopravvive al training)

## Done

- 2026-08-21 — **MQAR chiusa** (53 run, 2 seed, 4070S distrutta): capacità lunga
  FUNZIONALE; due leggi (selettività paga sotto carico; l'init del gate fissa
  l'orizzonte di apprendibilità, cura bias −8 verificata e gratis a corto raggio);
  figura + Results IV nel paper. Esiti in D17
- 2026-08-20/21 — **Griglie stadio char chiuse** (A+B+C1+B2): coordinata locale
  necessaria non sufficiente; confini cablati zero; spettro imposto guida e muore;
  gate appreso = unica capacità lunga + migliore estrapolazione; loss satura
  all'epoca. Esiti in D16/D17
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
