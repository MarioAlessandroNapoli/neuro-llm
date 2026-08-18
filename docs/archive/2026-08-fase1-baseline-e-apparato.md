# Archivio — Fase 1: baseline 8,5M e chiusura dell'apparato (2026-08-17/18)

Registro di deployment della fase che ha prodotto baseline e apparato numerico dello
stadio 1. Ogni numero qui è già consumato nelle decisioni D5-D10 del RESEARCH_LOG;
questo file conserva il come, per consultazione.

## Campagne eseguite

| Campagna | Esito | Dettaglio |
|---|---|---|
| Sweep lr baseline (3 run × 20M, T4) | lr congelato **1e-3** | val loss: 7,05 (1e-4) / 4,47 (3e-4) / **3,67 (1e-3)**. Vincitore sul bordo della griglia, trend monotono: estendere la griglia = revisione D6 esplicita |
| Àncora BPB esterna | **0,4407** | stories15M (llama2.c, artefatti a commit pinnato) sul nostro val V2 @256: loss 1,1511 nats/token; coerente col README llama2.c (1,072 su V1). ≡ val loss nostra 1,252. Script: `src/eval/anchor_stories15m.py` |
| Pilot epoch piena (536M, 87 min) | val loss **1,509** | BPB 0,531 @512 · 0,551 @256. Curva: 100M→1,99 · 170M→1,80 · 260M→1,66 · 390M→1,56 → base evidenziale di D8 (170M). Script BPB checkpoint: `src/eval/bpb_checkpoint.py` |
| Bench velocità (4 config × 20M) | config standard **DDP b16×2+compile** | 104k (1 GPU) → 113k (+compile) → **176k (DDP b16×2, 1,69×, batch globale invariato, val loss 3,669 vs 3,666 storica)** → 211k (b32×2, scartata: batch 64 ⇒ ri-sweep per +20%) |
| 5 seed baseline (170M, D8) | **ε = σ = 0,017** | val loss: 1,8427 / 1,8064 / 1,8114 / 1,8336 / 1,8390 — media 1,8266. Tutti dentro la previsione del pilot (1,80-1,85). base-1..5 nel registro; checkpoint su HF |

## Lezioni operative pagate (dettaglio nei gotcha della skill ml-dev)

- `save_notebook` via API resetta l'acceleratore (P100/None): T4 x2 sempre dalla UI.
- Sessioni interattive dell'editor bruciano quota anche inattive; il `time_reserved`
  della quota traccia solo quelle, non i commit.
- Processo appeso allo shutdown post-training (−10h quota in una notte) → `os._exit(0)`
  in train.py; residuo sporadico nel teardown DDP: verificare la chiusura a fine sessione.
- fused AdamW incompatibile col gradient clipping sotto AMP.
- Su M2 il transiente iniziale porta la train loss a ~21 (spike a ~79 allo step 19,
  rientro entro lo step 160): è il profilo normale della ricetta, non un bug.

## Nota di metodo scoperta strada facendo (assorbita in D8)

A piccoli budget una run dedicata chiude peggio del punto intermedio di una run lunga
(20M dedicata: 3,67 vs ~3,4 della curva pilot a 20M): l'annealing precoce costa più del
rumore che toglie. I punti intermedi di una curva lunga si leggono come stima centrale.
