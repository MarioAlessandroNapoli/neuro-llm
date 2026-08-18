# Board

Stato e priorità del progetto. Regole di gestione nella skill `neuro`.
Now = in lavorazione (max 2-3) · Next = pronto a partire · Later = deciso, non prossimo · Done = compresso.

## Now

- **Design stadio 1 — chiudere Q1** (istanziazione concreta dei 5 assi D9: quali
  architetture nella griglia 1a, con quale piano di ablazione). Output atteso: D10.
  Assi congelati in D9 (2026-08-18, da lettura Castellanos cap. 1): scale temporali ·
  inibizione/oblio · scia · metastabilità · gerarchia per profondità.

## Next

- Validare il giudice con una chiamata reale (serve `ANTHROPIC_API_KEY` nell'ambiente:
  la build delle richieste è già verificata, manca solo il round-trip API)
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

- 2026-08-18 — **Baseline completa e apparato numerico chiuso**: 5 seed a 170M (registro
  base-1..5) → media 1,8266, **σ = 0,017 = banda ε di D7**; lr 1e-3, budget 170M (D8),
  config DDP b16×2+compile; checkpoint su HF. La griglia 1a ha tutto tranne D10.
- 2026-08-18 — Bench velocità (4 config × 20M): DDP 2×T4 b16×2 + compile = **176k tok/s
  (1,69×)** a batch globale e traiettoria invariati (val loss 3,669 vs 3,666 sweep) →
  adottata come standard; batch 64 (2,03×) scartato: richiederebbe ri-sweep per +20%.
  Due gotcha pagati: fused AdamW incompatibile col grad clipping AMP; sessione commit
  appesa post-training (−10h quota) → fix `os._exit(0)`
- 2026-08-18 — D8: budget stadio 1 congelato a 170M token/run (Q4 chiusa, curva dal pilot)
- 2026-08-17 — Àncora BPB misurata: stories15M sul nostro val V2 @256 → loss 1,1511
  nats/token, **BPB 0,4407** (≡ val loss 1,252 col nostro tokenizer); coerente col README
  llama2.c (1,072 su V1); script riproducibile `src/eval/anchor_stories15m.py`
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
