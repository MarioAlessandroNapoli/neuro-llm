# Board

Stato e priorità del progetto. Regole di gestione nella skill `neuro`.
Now = in lavorazione (max 2-3) · Next = pronto a partire · Later = deciso, non prossimo · Done = compresso.

**Budget compute**: 10 $ — spesi 3,42 $ (griglia 1a: 3090 ~2 $ + 4080 ~0,68 $;
**fase 0 + griglia 1b + asintoto 536M completi: 3060+4070 ~0,74 $**) · residui 6,58 $
(agg. 2026-08-19, notte — stadio 1 chiuso). 4070 distrutta; 4080 ancora stoppata a
0,007 $/h (~0,17 $/giorno): distruggere se non riparte nulla a breve.

## Now

- **Stadio 1 CHIUSO (D13, esito registrato)**: parità strutturale ibrido-transformer
  a 170M e 536M (1,509 vs 1,497 all'asintoto); baseline onesta 1,558 confermata
  (sonda b8 inverte il trend); autopsia 536M: front-end filtri = legge, gradiente
  "consumatore" non replica. **Distruggere la 4070** (compute finito) e aggiornare
  il ledger col saldo. Archiviazione contesto: a inizio prossima sessione.

## Next

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
