"""Ablazione del contesto (D19): la retention misurata in nats per distanza.

Stessa striscia finale di 64 byte valutata con contesti crescenti: quanto la loss
migliora grazie ai byte lontani è la memoria a lungo raggio effettivamente usata.
retention(T) = loss(contesto T) − loss(contesto pieno), riportata APPAIATA per
striscia (Δ ± SE appaiato); la matrice (contesti × strisce) va su disco per i
confronti braccio-vs-braccio appaiati. Post review 2026-08-25:
- --val-file OBBLIGATORIO (il default puntava ai byte TinyStories dello stadio
  char: corpus sbagliato in silenzio); nel report: file, dimensione, sha.
- strisce con EOT (0x00) dentro ESCLUSE (Δ=0 per costruzione, solo varianza);
  la frazione di contesti pieni che attraversano un EOT è misurata e dichiarata.
- modelli con pos emb (d19-cb): la Δ include lo shift di identità posizionale
  della striscia tra contesti — confound dichiarato nell'output.

Uso: python -m scripts.eval_retention --arch d19-mix2 --run-name d19-mix2-s1 \
       --data-dir data/nemotron --val-file valid_bytes.bin
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.configs import CHAR_EOT_BYTE
from src.models import build_model

STRIPE = 64
CONTEXTS = (128, 256, 512, 1024, 2048)
SEED = 20260825


@torch.no_grad()
def stripe_losses(model, data, starts, ctx, batch, device):
    """CE media per striscia (vettore (n,)) sugli ultimi STRIPE byte."""
    out = []
    for i in range(0, len(starts), batch):
        chunk = starts[i: i + batch]
        idx = torch.from_numpy(np.stack(
            [np.asarray(data[s - ctx: s]) for s in chunk]).astype(np.int64)).to(device)
        logits = model(idx)
        ce = F.cross_entropy(
            logits[:, -STRIPE - 1:-1].reshape(-1, logits.shape[-1]),
            idx[:, -STRIPE:].reshape(-1), reduction="none").view(len(chunk), STRIPE)
        out.append(ce.mean(dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hub-repo", default="MarioAlessandroNapoli/neuro-llm-ckpt")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/retention"))
    args = parser.parse_args()

    model, cfg = build_model(args.arch)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(args.hub_repo, f"{args.run_name}/last.ckpt")
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()})
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    val_path = args.data_dir / args.val_file
    data = np.memmap(val_path, dtype=np.uint8, mode="r")
    sha = hashlib.sha256(val_path.read_bytes()).hexdigest()[:16]
    rng = np.random.default_rng(SEED)
    max_ctx = max(CONTEXTS)
    # start = fine striscia; si escludono strisce che contengono EOT (Δ nulla per
    # costruzione); si misura la quota di contesti pieni che attraversano un EOT.
    cand = rng.choice(np.arange(max_ctx, len(data)), args.windows * 4, replace=False)
    arr = np.asarray(data)
    good = [s for s in cand
            if not np.any(arr[s - STRIPE: s] == CHAR_EOT_BYTE)][: args.windows]
    if len(good) < args.windows:
        raise SystemExit(f"solo {len(good)} strisce senza EOT su {args.windows} "
                         "richieste: valid troppo frammentata, allargare il file")
    starts = np.array(good)
    eot_in_ctx = np.mean([np.any(arr[s - max_ctx: s - STRIPE] == CHAR_EOT_BYTE)
                          for s in starts])

    print(f"{args.run_name} — ablazione del contesto ({len(starts)} strisce da "
          f"{STRIPE} byte, seed {SEED})")
    print(f"  val: {val_path} ({len(data):,} byte, sha {sha}) · contesti pieni "
          f"che attraversano un EOT: {eot_in_ctx:.0%}")
    if model.pos is not None:
        print("  NOTA: modello con pos emb — la Δ include lo shift di identità "
              "posizionale della striscia tra contesti (confound dichiarato)")

    mat = np.stack([stripe_losses(model, data, starts, c, args.batch, device)
                    for c in CONTEXTS])
    full = mat[-1]
    for ci, ctx in enumerate(CONTEXTS):
        d = mat[ci] - full
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  contesto {ctx:>5}: {mat[ci].mean():.4f} nats/byte · "
              f"retention residua {d.mean():+.4f} ± {se:.4f} (SE appaiato)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.run_name}.npz"
    np.savez(out, contexts=np.array(CONTEXTS), losses=mat, starts=starts,
             val_sha=sha)
    print(f"  matrice salvata: {out}")


if __name__ == "__main__":
    main()
