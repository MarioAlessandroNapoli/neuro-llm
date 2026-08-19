# Board

Stato e priorità del progetto. Regole di gestione nella skill `neuro`.
Now = in lavorazione (max 2-3) · Next = pronto a partire · Later = deciso, non prossimo · Done = compresso.

## Now

- **Griglia 1a in chiusura (oggi ~15:30)**: fatti baseline 5 seed (1,599, ε=0,007),
  hyb-ao 3 seed, verdetti NaN hyb-oa/dlinoss@bordo; in corsa dlinoss @3e-3 s2-s3,
  poi linoss-registro 20M fp32 e gate phi (atteso FAIL → phi in 1b). A fine catena:
  righe registro finali + **distruggere l'istanza vast** (fattura anche da ferma).
- Findings parziali registrati in D11: oscillatori ≤ attention a parità; oblio
  necessario-non-sufficiente; gerarchia inversa vince; tema omeostasi per la 1b.

## Next

- Validare il giudice con una chiamata reale (serve `ANTHROPIC_API_KEY` nell'ambiente:
  la build delle richieste è già verificata, manca solo il round-trip API)
- Campagna giudice post-griglia: generazioni per tutti i bracci + baseline, scoring ed
  Elo in un'unica finestra Batches (skill `eval`), poi tabella verdetti D7 per braccio

## Later

- Griglia 1b (Q5): direzioni motivate dai findings D11 — omeostasi/regolazione del
  campo oscillatorio, φ a 170M (gate fallito ma mai corso), wrnn con scarico (leak),
  hyb-oa ripescato a 3e-3; gerarchia al char-level → si fonde con Q3
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
