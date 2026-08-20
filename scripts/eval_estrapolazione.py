"""Test di estrapolazione in lunghezza (post-B2, pre-registrato in D16).

Valuta un checkpoint char a finestre più lunghe del training (2048 → 4096/8192)
senza riaddestrare: nats/byte complessivi e per bucket di posizione (ampiezza 2048).
Predizione pre-registrata: cb (pos emb appreso) è strutturalmente incapace oltre la
tabella (il test lo dichiara e si ferma — quello È il risultato); l'ibrido hard
(coordinata = rampa dal confine, invariante per traslazione) degrada con grazia:
bucket oltre 2048 ≈ bucket sotto 2048.

Uso: python -m scripts.eval_estrapolazione --arch char-hyb-hard --run-name fB2-hard-s1 \
       --seq 2048 4096 8192
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models import build_model

BUCKET = 2048
SEED = 20260820


@torch.no_grad()
def eval_at_length(model, data, seq, n_windows, batch, device):
    rng = np.random.default_rng(SEED)
    starts = rng.choice(len(data) - seq - 1, n_windows, replace=False)
    sums = torch.zeros(seq - 1, dtype=torch.float64)
    for i in range(0, n_windows, batch):
        chunk = starts[i: i + batch]
        idx = torch.from_numpy(
            np.stack([np.asarray(data[s: s + seq]) for s in chunk]).astype(np.int64)
        ).to(device)
        logits = model(idx)
        ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            idx[:, 1:].reshape(-1),
            reduction="none",
        ).view(idx.shape[0], -1)
        sums += ce.double().sum(dim=0).cpu()
    per_pos = (sums / n_windows).numpy()
    print(f"  seq {seq}: totale {per_pos.mean():.4f} nats/byte")
    for b0 in range(0, seq - 1, BUCKET):
        seg = per_pos[b0: b0 + BUCKET]
        print(f"    posizioni {b0:>5}-{min(b0 + BUCKET, seq - 1):>5}: {seg.mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hub-repo", default="MarioAlessandroNapoli/neuro-llm-ckpt")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seq", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    model, cfg = build_model(args.arch)
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(args.hub_repo, f"{args.run_name}/last.ckpt")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()})
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    data = np.memmap(args.data_dir / "valid_bytes.bin", dtype=np.uint8, mode="r")
    print(f"{args.run_name} — estrapolazione ({args.windows} finestre, seed {SEED})")
    for seq in args.seq:
        if model.pos is not None and seq > model.pos.num_embeddings:
            print(f"  seq {seq}: STRUTTURALMENTE INCAPACE — tabella posizioni da "
                  f"{model.pos.num_embeddings}, indici oltre non esistono")
            continue
        eval_at_length(model, data, seq, args.windows, args.batch, device)


if __name__ == "__main__":
    main()
