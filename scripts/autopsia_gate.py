"""Autopsia del gate di reset (fase B / D17): il gate ha ri-imparato il tokenizer?

Per ogni layer oscillatorio di un braccio con reset appreso, estrae le probabilità di
confine b_t (per i 64 gruppi) su finestre di validazione e misura:
- allineamento con i confini letterali (spazi/punteggiatura/EOT): precision/recall
  del gate massimo, e media p su confini vs non-confini;
- selettività: ai confini letterali, quanti gruppi resettano forte (p>0,5) e quanti
  portano memoria (p<0,1) — reset totale (tokenizer-like) vs parziale (selettivo);
- fuoco intra-parola: quota di posizioni non-confine col gate massimo >0,5.

Se il gate fosse un tokenizer implicito: recall~1, e ai confini TUTTI i gruppi a p~1.
Uso: python -m scripts.autopsia_gate --arch char-hyb-hard --run-name fB-hard-s2
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.configs import CHAR_BOUNDARY_BYTES
from src.models import build_model
from src.models.linoss import OscBlock

N_WINDOWS = 32
SEQ = 2048
SEED = 20260820


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hub-repo", default="MarioAlessandroNapoli/neuro-llm-ckpt")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    model, cfg = build_model(args.arch)
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(args.hub_repo, f"{args.run_name}/last.ckpt")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()})
    model.eval()

    data = np.memmap(args.data_dir / "valid_bytes.bin", dtype=np.uint8, mode="r")
    rng = np.random.default_rng(SEED)
    starts = rng.choice(len(data) - SEQ - 1, N_WINDOWS, replace=False)
    windows = np.stack([np.asarray(data[s: s + SEQ]) for s in starts])
    idx = torch.from_numpy(windows.astype(np.int64))
    is_boundary = torch.from_numpy(
        np.isin(windows, np.array(list(CHAR_BOUNDARY_BYTES), dtype=np.uint8))
    )

    print(f"{args.run_name} — autopsia gate ({N_WINDOWS} finestre; "
          f"quota byte-confine nel testo: {is_boundary.float().mean():.3f})")
    x = model.tok(idx)
    for li, blk in enumerate(model.blocks):
        if not isinstance(blk, OscBlock):
            break
        mx = blk.mixer
        u = blk.ln1(x)
        p = torch.sigmoid(mx.gate_conv(u.transpose(1, 2))[..., :SEQ]).transpose(1, 2)
        pmax = p.max(dim=-1).values  # (n, t): il gruppo più reattivo per posizione
        fire = pmax > 0.5
        tp = (fire & is_boundary).sum().item()
        prec = tp / max(fire.sum().item(), 1)
        rec = tp / is_boundary.sum().item()
        pb = p[is_boundary]      # (n_conf, 64)
        pn = p[~is_boundary]
        frac_hard = (pb > 0.5).float().mean(dim=0)   # per gruppo
        print(f"  L{li}: p medio confini {pb.mean():.3f} vs non-confini {pn.mean():.3f} | "
              f"gate-max: precision {prec:.2f} recall {rec:.2f} | "
              f"intra-parola fire {(fire & ~is_boundary).float().mean():.3f}")
        print(f"      selettività ai confini: gruppi con p>0,5: "
              f"{(frac_hard > 0.5).sum().item()}/64 · gruppi portatori (p<0,1): "
              f"{((pb < 0.1).float().mean(dim=0) > 0.5).sum().item()}/64 · "
              f"p per-gruppo min/med/max {pb.mean(dim=0).min():.2f}/"
              f"{pb.mean(dim=0).median():.2f}/{pb.mean(dim=0).max():.2f}")
        x = blk(x)


if __name__ == "__main__":
    main()
