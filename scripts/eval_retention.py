"""Ablazione del contesto (D19): la retention misurata in nats per distanza.

Stessa striscia finale di 64 byte valutata con contesti crescenti: quanto la loss
migliora grazie ai byte lontani è la memoria a lungo raggio effettivamente usata.
retention(T) = loss(contesto T) − loss(contesto pieno); una curva che continua a
scendere ai contesti lunghi = il modello sfrutta davvero la distanza.

Uso: python -m scripts.eval_retention --arch d19-mix2 --run-name d19-mix2-s1
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models import build_model

STRIPE = 64
CONTEXTS = (128, 256, 512, 1024, 2048)
SEED = 20260825


@torch.no_grad()
def stripe_loss(model, data, starts, ctx, batch, device):
    """CE media sugli ultimi STRIPE byte, con contesto di lunghezza ctx."""
    tot = 0.0
    for i in range(0, len(starts), batch):
        chunk = starts[i: i + batch]
        idx = torch.from_numpy(np.stack(
            [np.asarray(data[s - ctx: s]) for s in chunk]).astype(np.int64)).to(device)
        logits = model(idx)
        ce = F.cross_entropy(
            logits[:, -STRIPE - 1:-1].reshape(-1, logits.shape[-1]),
            idx[:, -STRIPE:].reshape(-1), reduction="sum")
        tot += ce.item()
    return tot / (len(starts) * STRIPE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hub-repo", default="MarioAlessandroNapoli/neuro-llm-ckpt")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="checkpoint locale (in alternativa a --run-name su HF)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--val-file", default="valid_bytes.bin")
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    model, cfg = build_model(args.arch)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(args.hub_repo, f"{args.run_name}/last.ckpt")
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()})
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    data = np.memmap(args.data_dir / args.val_file, dtype=np.uint8, mode="r")
    rng = np.random.default_rng(SEED)
    # start = fine della striscia: i byte valutati sono data[s-STRIPE:s], identici
    # per ogni contesto — cambia solo quanta storia il modello vede prima.
    starts = rng.choice(
        np.arange(max(CONTEXTS), len(data)), args.windows, replace=False)

    print(f"{args.run_name} — ablazione del contesto "
          f"({args.windows} strisce da {STRIPE} byte, seed {SEED})")
    losses = {}
    for ctx in CONTEXTS:
        losses[ctx] = stripe_loss(model, data, starts, ctx, args.batch, device)
    full = losses[max(CONTEXTS)]
    for ctx in CONTEXTS:
        delta = losses[ctx] - full
        print(f"  contesto {ctx:>5}: {losses[ctx]:.4f} nats/byte "
              f"(retention residua vs pieno: {delta:+.4f})")


if __name__ == "__main__":
    main()
