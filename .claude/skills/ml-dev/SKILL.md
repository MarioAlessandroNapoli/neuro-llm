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

| | M2 (solo smoke) | Kaggle T4 x2 (config standard) |
|---|---|---|
| Throughput @8,5M params | ~5,3k tok/s | **~176k tok/s** (DDP b16×2 + compile; 1 GPU: ~104k) |
| Run di griglia (170M, D8) | — | **~16 min** |
| Quota | — | 30 h/settimana (`get_accelerator_quota`), sessioni max 12 h |

**Config standard di lancio (bench 2026-08-18)**: `--devices 2 --batch-size 16 --compile`
— batch efficace 32 invariato (16×2), traiettoria identica alla config storica (val loss
3,669 vs 3,666 a parità di step), 1,69× di velocità. Il batch efficace è parte della
ricetta D6: non cambiarlo mai senza ri-sweep del lr.
Griglia tipo stadio 1 (3 arch × 3 seed × 170M): ~2,5 h GPU.

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
| Crash immediato `CUDA error: no kernel image` su tutte le run | `save_notebook` via API resetta l'acceleratore al default P100 (sm_60, non più supportato da PyTorch); `machineShape` viene ignorato/normalizzato a "Gpu". Fix: impostare **GPU T4 x2** dalla UI (Settings → Accelerator) e lanciare Save & Run All da lì; conferma T4 = `tokens_per_sec` ~100k |
| Quota GPU che evapora senza run (−10h in una notte, 2026-08-18) | La sessione di commit del pilot (87 min, --hub-repo) è rimasta RUNNING dopo la fine del training: processo appeso allo shutdown dell'interprete (thread residui HF/W&B/dataloader; le run brevi senza --hub-repo uscivano pulite). `time_reserved` è la prenotazione del cap 12h e si rilascia solo a processo morto. Fix cablato: `os._exit(0)` a fine `main()` in train.py. Residuo noto (visto su 1 run su 5, 2026-08-18): con DDP il teardown può appendersi sporadicamente (rank secondario esce mentre il principale è ancora nell'upload HF) — a fine sessione multi-run verificare sempre la chiusura e stoppare a mano se lingera; nel runner tenere una cella finale `rm -rf checkpoints` per alleggerire il packaging. Controllo dopo ogni lancio: la versione in kaggle.com/me/sessions deve chiudersi da sola a fine run. Anche l'editor aperto con acceleratore attivo è una sessione interattiva che brucia quota: Stop session dopo Save & Run All. ATTENZIONE (verificato 2026-08-18): `time_reserved` traccia solo le sessioni interattive, NON i commit — un commit GPU può girare con reserved=0; per sapere se un commit sta girando fa fede W&B o la version history, mai la quota |

## vast.ai — istanza a noleggio (fase griglia 1a)

GPU singola a noleggio per smoke/sweep senza coda né quota. **Le coordinate (IP, porta,
instance id) vivono SOLO in `~/.config/neuro-llm/vast.env`** (chmod 600): questo file di
skill è nel repo pubblico — mai coordinate, token o chiavi qui dentro.

```bash
source ~/.config/neuro-llm/vast.env
ssh -p $VAST_SSH_PORT root@$VAST_SSH_HOST          # chiave ~/.ssh/id_ed25519 (registrata su vast)
```

**Layout sull'istanza**: repo in `/workspace/neuro-llm`, venv `/venv/main` (da attivare:
`python` nudo non è nel PATH), secrets in `/workspace/.hf_token` e `/workspace/.wandb_key`
(chmod 600, arrivano via scp file-a-file, mai inline in comandi o log). Script di
lancio `/workspace/vast_smoke.sh`, log `/workspace/smoke.log`.

**Config di lancio su GPU singola**: `--batch-size 32` senza `--devices` — è la config
storica della ricetta D6 (batch globale 32 invariato), niente DDP.

**Gestione da remoto (senza UI vast):**

```bash
ssh -p $VAST_SSH_PORT root@$VAST_SSH_HOST 'tail -20 /workspace/smoke.log'   # progresso
ssh -p $VAST_SSH_PORT root@$VAST_SSH_HOST 'pgrep -fl src.train'             # run attive
ssh -p $VAST_SSH_PORT root@$VAST_SSH_HOST 'nvidia-smi'                      # GPU
# lancio di un job lungo: nohup + redirect + STDIN CHIUSO, altrimenti l'ssh resta appeso
ssh -p $VAST_SSH_PORT root@$VAST_SSH_HOST \
  'cd /workspace && nohup bash script.sh > job.log 2>&1 < /dev/null & echo PID $!'
```

**Fatturazione — l'unico vero rischio operativo**: l'istanza fattura **anche da ferma e
inattiva** (~0,14 $/h ≈ 3,5 $/notte); lo stop (⏸ dalla UI) ferma la GPU ma continua a
fatturare lo storage. A fine fase l'istanza si **distrugge** (Destroy dalla UI, o
`vastai destroy instance $VAST_INSTANCE_ID` se il CLI è configurato). Tutto ciò che
serve conservare sta già su GitHub/W&B/HF: l'istanza è usa-e-getta.

**Nuova istanza (l'attuale muore o si distrugge):** noleggio con template "PyTorch
(Vast)" (filtri: verified, on-demand, ≥12 GB VRAM per i bracci a scan, reliability
≥99%, banda ≥500 Mbps) → chiave SSH dalla UI dell'istanza → aggiornare `vast.env` →
setup: clone del repo, `pip install -r requirements.txt` nel venv, scp dei due secret
file, `snapshot_download` del dataset. ~5 minuti totali.

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
