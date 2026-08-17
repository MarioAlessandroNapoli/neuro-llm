# neuro-llm

Esperimenti di architetture neuro-ispirate per language model piccoli (8-33M parametri),
addestrati from scratch su TinyStories. Confronto a parità di setup: transformer baseline
vs varianti oscillatorie (LinOSS, Wave-RNN — in arrivo).

## Struttura

- `src/models/` — architetture come `nn.Module` puri, registrate in `src/models/__init__.py`
- `src/prepare_data.py` — one-shot: download TinyStories V2, BPE 8k, tokenizzazione in `.bin`
- `src/train.py` — entrypoint unico (locale e Kaggle), Lightning + W&B, resume via HF Hub
- `kaggle/runner.ipynb` — notebook da caricare su Kaggle per le run vere

## Setup locale (sviluppo e smoke test)

```bash
uv sync
uv run python -m src.prepare_data                       # scarica + tokenizza (una tantum)
uv run python -m src.train --arch transformer --tokens 1000000 --precision 32-true --no-wandb
```

Su Mac (MPS) usare `--precision 32-true`. Le run vere girano su Kaggle.

## Pipeline dati

`prepare_data.py` addestra un BPE da 8192 token su TinyStories e produce `train.bin` /
`valid.bin` (uint16). Con `--hub-repo <user>/tinystories-tokenized` li pusha su un dataset
repo HF privato: Kaggle li scarica pronti, la tokenizzazione non si ripete mai.

## Setup Kaggle (una tantum)

1. Account Kaggle con verifica telefonica (sblocca internet nei notebook)
2. Secrets (Add-ons → Secrets): `WANDB_API_KEY` ([wandb.ai/authorize](https://wandb.ai/authorize))
   e `HF_TOKEN` (token HF con scope write)
3. `python -m src.prepare_data --hub-repo <user>/tinystories-tokenized` da locale
4. Caricare `kaggle/runner.ipynb` come notebook, acceleratore **GPU T4 x1**, Internet ON

## Ciclo di iterazione

1. Modifica codice → commit + push su GitHub
2. Sul notebook Kaggle: aggiorna i flag di `src.train` se serve → **Save & Run All**
3. Metriche live su [wandb.ai](https://wandb.ai) (progetto `neuro-llm`)
4. Il training riprende da solo dall'ultimo checkpoint su HF Hub e si chiude pulito
   a `--max-time` (11h) prima del kill della sessione

I checkpoint vivono su HF Hub (`<user>/neuro-llm-ckpt`, privato): sopravvivono alle
sessioni e non consumano i 5 GB/mese del free tier W&B.

## Interrogare le run da Claude Code (MCP)

```bash
claude mcp add --scope user --transport http wandb https://mcp.withwandb.com/mcp \
  --header "Authorization: Bearer $WANDB_API_KEY"
```

Scope `user`, mai un `.mcp.json` nel repo: la API key non deve entrare nel versionamento.
Verifica: chiedere a Claude "list my W&B runs in project neuro-llm".
