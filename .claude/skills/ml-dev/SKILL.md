---
name: ml-dev
description: Use when running, monitoring, or debugging neuro-llm experiments — launching training on Kaggle, querying W&B runs/metrics via MCP, managing the Kaggle notebook, verifying checkpoints on HF Hub, local smoke tests, or adding architectures to the registry.
---

# ml-dev — Experiment Ops

Operatività della sperimentazione: training, tracking, observability. Le decisioni
scientifiche (cosa testare e perché) stanno nella skill `neuro`; qui c'è il come.

## Mappa dell'infrastruttura

| Cosa | Dove |
|---|---|
| Codice | github.com/MarioAlessandroNapoli/neuro-llm (pubblico) |
| Dataset tokenizzato | HF dataset privato `MarioAlessandroNapoli/tinystories-tokenized` (536M token, BPE 8k) |
| Checkpoint | HF model privato `MarioAlessandroNapoli/neuro-llm-ckpt/<run_name>/last.ckpt` |
| Metriche | W&B, entity `marioalessandronapoli`, project `neuro-llm` (via MCP `mcp__wandb__*`) |
| Notebook Kaggle | `marionapoli/neuro-llm-notebook` (via MCP `mcp__kaggle__*`); secrets: `HF_TOKEN`, `WANDB_API_KEY` |

## Ciclo standard di un esperimento

1. Modifica codice → smoke locale su M2:
   `uv run python -m src.train --arch <a> --tokens 1000000 --seed 1 --precision 32-true --no-wandb`
2. Commit + push (il runner Kaggle clona `main`).
3. Se cambia il runner: `mcp__kaggle__save_notebook` (o UI).
4. Lancio: **Save & Run All (Commit)** — mai run lunghe in interattivo (muoiono col tab).
   Run lunghe: `--max-time 00:11:00:00`. `--seed` è obbligatorio ed entra nel nome della
   run insieme a budget e lr (`arch-dD-LN-tXM-sS-lrY`): niente collisioni tra seed, il
   run_name è anche l'id W&B. Resume solo esplicito con `--resume` (checkpoint trovato
   senza flag = errore). Contenimento registri: sweep lr senza `--hub-repo` e con
   `--group sweep-lr`; run di griglia con `--group grid-stage1`; solo le run di griglia
   hanno checkpoint su HF e riga nel registro esperimenti.
5. Monitor: W&B MCP (sotto). Stato sessione: `get_notebook_session_status` traccia solo le
   versioni committate — su run interattive risponde "No runs found"; il consumo reale si
   vede con `get_accelerator_quota`.
6. Fine run: verificare `state: finished` su W&B **e** il checkpoint su HF
   (`list_repo_files('MarioAlessandroNapoli/neuro-llm-ckpt')`).
7. Registrare: riga nel registro esperimenti di RESEARCH_LOG + board (skill `neuro`).

## W&B via MCP — query utili

MCP in sola lettura. Run e metriche finali:

```graphql
query { project(name: "neuro-llm", entityName: "marioalessandronapoli") {
  runs(first: 10) { edges { node { name state createdAt summaryMetrics } } } } }
```

via `query_wandb_tool`. Serie temporali: `get_run_history_tool`; confronti:
`compare_runs_tool`; diagnosi run fallite/anomale: `diagnose_run_tool`.
Metriche loggate: `train_loss`, `train_ppl`, `val_loss`, `val_ppl`, `tokens_per_sec`.

## Numeri di riferimento (misurati, 2026-08-17)

| | M2 (solo smoke) | Kaggle T4 |
|---|---|---|
| Throughput @8,5M params | ~5,3k tok/s | **~115k tok/s** |
| 100M token | ~5 h | **~15 min** |
| Quota | — | 30 h/settimana (`get_accelerator_quota`), sessioni max 12 h |

Griglia tipo stadio 1 (3 arch × 3 seed × 100M): ~2-2,5 h GPU.

## Gotchas (tutti già pagati una volta)

| Sintomo/rischio | Causa e fix |
|---|---|
| Crash "bfloat16 not supported" | T4/P100 pre-Ampere → `--precision 16-mixed` (su MPS: `32-true`) |
| Lightning aggancia 2 GPU su T4×2 | `devices=1` è già nel Trainer: non rimuoverlo |
| Run W&B che si "fondono" | id = run_name con `resume="allow"`: usare run-name unici per seed |
| Checkpoint stantio a fine run | `on_train_end` salva fresco e pusha su HF: non rimuoverlo |
| Run persa alla chiusura del tab | Era interattiva: usare sempre Save & Run All per run > 15 min |
| Scritture W&B da locale falliscono | Il bearer token MCP non è una API key classica: serve `wandb login` |
| `get_notebook_info` esplode il contesto | Output ~130KB (include il sorgente): salvarlo su file e parsare con python |

## Harness di valutazione (D7)

Vive nella skill `eval` (codice `src/eval/`, artefatti `eval/`): generazione, giudice
Opus 5 via Batches, analisi con i verdetti pre-registrati. A fine run di griglia il ciclo
è: checkpoint verificato su HF → `src.eval.generate` → skill `eval` per il resto.

## Aggiungere un'architettura

`src/models/<nome>.py` come `nn.Module` puro con la stessa firma del baseline
(`forward(idx) -> logits`), registrata in `ARCHS` (`src/models/__init__.py`).
Parità parametri col baseline (±5%) verificata prima della prima run. Il tokenizer e
`src/data.py` non si toccano mai (RESEARCH_LOG D3/D4).

## Gap noti

- Upload periodico su HF ogni 2000 step: per run < 30 min interviene solo a fine run.
- L'ordine dei dati non è ripristinato al resume (sampler riparte da una nuova
  permutazione): accettato per l'ordine, ma su arm molto lente il conteggio di token
  unici si accorcia — rivalutare se un'arm scende sotto ~2,5k tok/s.
