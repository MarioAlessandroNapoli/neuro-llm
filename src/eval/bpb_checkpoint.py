"""BPB di un checkpoint nostro sul val set completo, a contesto 256 e 512.

Metodo identico all'àncora (`anchor_stories15m.py`): finestre non sovrapposte dallo
stream di valid.bin, ogni token predetto una volta, coda scartata. Il 256 è il contesto
di confronto con l'àncora; il 512 è la finestra di training (D7).
Esecuzione: uv run python -m src.eval.bpb_checkpoint --run-name <arch-dD-LN-tXM-sS-lrY>
"""
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from ..models import build_model
from ..prepare_data import DATA_DIR, SPLITS
from .analysis import arch_of, bpb, bytes_per_token

HUB_REPO = "MarioAlessandroNapoli/neuro-llm-ckpt"
CONTEXTS = (256, 512)
BATCH = 32


@torch.no_grad()
def stream_loss(model, tokens: np.ndarray, ctx: int, device: str) -> float:
    n_blocks = (len(tokens) - 1) // ctx
    blocks = torch.from_numpy(tokens[: n_blocks * ctx + 1].astype(np.int64)).unfold(0, ctx + 1, ctx)
    total_nats, n_pred = 0.0, 0
    for i in range(0, n_blocks, BATCH):
        chunk = blocks[i : i + BATCH].to(device)
        x, y = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
        logits = model(x)
        total_nats += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        n_pred += y.numel()
    return total_nats / n_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hub-repo", default=HUB_REPO)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt_path = hf_hub_download(args.hub_repo, f"{args.run_name}/last.ckpt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model, cfg = build_model(arch_of(args.run_name))
    state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    print(f"{args.run_name}: checkpoint al global step {ckpt['global_step']}")

    tokens = np.fromfile(DATA_DIR / "valid.bin", dtype=np.uint16)
    bpt = bytes_per_token(DATA_DIR / SPLITS["valid"], DATA_DIR / "valid.bin")
    print(f"val: {len(tokens)/1e6:.2f}M token, {bpt:.4f} byte/token (EOT esclusi)")

    for ctx in CONTEXTS:
        loss = stream_loss(model, tokens, ctx, device)
        print(f"contesto {ctx}: loss {loss:.4f} nats/token · ppl {math.exp(loss):.3f} · "
              f"BPB {bpb(loss, bpt):.4f}")


if __name__ == "__main__":
    main()
