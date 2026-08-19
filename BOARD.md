# Board

Stato e priorità del progetto. Regole di gestione nella skill `neuro`.
Now = in lavorazione (max 2-3) · Next = pronto a partire · Later = deciso, non prossimo · Done = compresso.

**Budget compute**: 10 $ — spesi 2,68 $ (griglia 1a completa: 3090 smoke+sweep+baseline
~2 $, 4080 griglia+controlli ~0,68 $) · residui 7,32 $ (agg. 2026-08-19).

## Now

- **Griglia 1a chiusa (2026-08-19)** — compute finito: **distruggere l'istanza vast**
  (fattura anche da ferma). Controllo lr-matched (baseline@3e-3 = 1,700) letto: il gap
  di hyb-ao (1,68) è **tutto tetto di ottimizzazione**, quello di dlinoss (1,93) è
  anche espressività. Titolo riformulato in D11: due tasse distinte (addestrabilità ·
  espressività) → bersagli 1b: omeostasi · selettività.
- Findings registrati in D11: due-tasse; oblio necessario-non-sufficiente; gerarchia
  inversa vince; gate phi FAIL per aritmetica; tema omeostasi per la 1b.

## Next

- **Fase 0 della 1b (D12)**: parametrizzazione log-polare in OscMixer + spike
  `torch.associative_scan` per fondere lo scan (A/B di training su GPU obbligatorio)
- Validare il giudice con una chiamata reale (serve `ANTHROPIC_API_KEY` nell'ambiente:
  la build delle richieste è già verificata, manca solo il round-trip API)
- Campagna giudice post-griglia: generazioni per tutti i bracci + baseline, scoring ed
  Elo in un'unica finestra Batches (skill `eval`), poi tabella verdetti D7 per braccio

## Later

- Griglia 1b, fasi 1-2 (D12): dlinoss log-polare + sweep lr · omeostasi vs controllo
  log-polare · ibrido intercalato A-O · hyb-oa@3e-3 · init post-autopsia r~U[0,7;0,9] ·
  asintoto 536M per i vincitori. Fuori scope dichiarato: selettività/gating (1c)
- Stadio 2: BabyLM Strict-Small (10M parole) + valutazione BLiMP per la variante migliore
- Ablazione granularità temporale (Q3: char-level o chunking appreso)
- Probe diagnostici esplorativi (D7: non cambiano mai il verdetto dello stadio 1):
  name-cloze a distanza, ablazione del contesto, probe spettrale degli hidden state
  (per dlinoss-phi: verifica se la struttura a bande sopravvive al training)

## Done

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
