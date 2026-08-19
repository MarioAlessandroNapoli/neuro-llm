# Timeline del progetto

Cronologia datata dei milestone — materiale per dev-log/sito/paper. Una riga per
evento significativo; il dettaglio scientifico vive nel RESEARCH_LOG (D*), quello
operativo nei commit. Aggiornata dalla skill `docs-and-artifacts`.

## Giorno 1 — 2026-08-17 (domenica): fondazioni

- **12:09** Pipeline E2E: dati TinyStories tokenizzati (BPE 8k, 536M token su HF),
  training Lightning + W&B, checkpoint su HF Hub, runner Kaggle.
- **13:13** Decisioni fondative D1-D6 (regime from-scratch, candidati oscillatori,
  tokenizer congelato, token=tempo, baseline nostra, parità ±5%).
- **13:45** Governance: skill `/neuro` (coordinatore) e `/ml-dev` (ops), BOARD.md;
  gate pre-push anti-secret fail-closed.
- **17:42** Review avversariale del disegno + apparato di valutazione D7 (giudice
  Opus 5 via Batches, statistica pre-registrata).
- **23:42** Àncora esterna misurata: stories15M sul nostro val = BPB 0,4407.

## Giorno 2 — 2026-08-18 (lunedì): baseline e griglia 1a

- **01:23** Pilot epoch piena (536M): val 1,509 — curva di riferimento per Q4-budget.
- **11:48** D8: budget 170M token; config DDP validata su Kaggle T4×2 (176k tok/s).
- **14:00-15:00** D9 (cinque assi neuro) → D10 (griglia 1a, sei bracci).
- **14:11** Baseline chiusa: 5 seed × 170M, 1,8266 ± σ 0,017 (lr 1e-3).
- **15:55** Sei bracci implementati (review ultracode: 3 difetti corretti pre-run).
- **pomeriggio** Migrazione da Kaggle (code >1h) a vast.ai (RTX 3090, poi 4080).
- **23:44** D11: revisione lr per categoria — tutte le architetture reggevano lr ben
  oltre il vecchio bordo; baseline rifatta a lr 3e-2.

## Giorno 3 — 2026-08-19 (martedì): verdetti 1a, fase 0, fase 1

- **notte** Griglia notturna su 4080: baseline D11 = **1,599 ± ε 0,007**; hyb-ao 1,68;
  hyb-oa e dlinoss@1e-2 NaN a 170M → emendamento "un gradino sotto il bordo".
- **11:51** Griglia 1a chiusa (anticipata). Controllo lr-matched (baseline@3e-3 =
  1,700) → **finding due-tasse**: l'ibrido paga solo addestrabilità, l'oscillatore
  puro anche espressività.
- **12:36** **Autopsia spettrale 1a**: il training tira r in giù (0,95→0,74-0,90),
  gerarchia di timescale emergente con la profondità, firma del guasto = perdita di
  smorzamento, hyb-oa morto all'init. D12: design 1b.
- **13:00-14:23** **Fase 0** su RTX 3060: parametrizzazione log-polare (dlinoss-lp);
  scan fuso via `associative_scan` — scoperto e aggirato **bug PyTorch** (backward
  Inductor rotto con matmul nel combine; fix elementwise) → **9,7× a livello
  modello**; bf16 respinto dall'A/B (il GradScaler fa da omeostata primitivo);
  regola pavimento-di-rumore per gli A/B. Figura: `figures/2026-08-fase0-ab-scan.png`.
- **14:51** Sweep dlinoss-lp (RTX 4070TiS, 231k tok/s): **il tetto di lr sale al
  livello della baseline** (3e-2, bordo 1e-1) con la sola parametrizzazione.
- **15:36-15:53** **Finding principale 1b**: a lr piena l'architettura tutta-oscillatori
  degenera spontaneamente in "banco di filtri al layer 0 + pila feedforward" (r→0 nei
  layer alti), robusta a init e seed, stessa loss del classico (~1,95): la memoria LTI
  extra non compra nulla senza indirizzamento per contenuto → direzione selettività.
  Figura: `figures/2026-08-autopsia-spettrale-1b.png`.
- **sera** Ibridi log-polari: sweep hyb-oa-lp = 2,146 @3e-2 (l'ordine "biologico"
  riabilitato dalla parametrizzazione — era il braccio che moriva NaN); in corsa il
  confronto gerarchia a 170M (hyb-oa-lp vs hyb-ao-lp).

## Compute e costi

3 GPU vast.ai usa-e-getta (3090 → 4080 → 3060 → 4070TiS), budget 10 $ (ledger in
BOARD). Velocità chiave: baseline 439k tok/s (4080); oscillatori 26,7k → **231k
tok/s** (4070TiS) dopo lo scan fuso. Una run oscillatoria da 170M: da ~1,8h a ~12 min.
