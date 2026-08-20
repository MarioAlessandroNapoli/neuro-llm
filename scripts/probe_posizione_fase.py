"""Probe posizione-nel-chunk (D16 fase A): dove vive l'indirizzo ordinale?

Decodifica LINEARE (ridge, closed form) della variabile "byte dall'ultimo confine"
(spazio/punteggiatura/EOT, cap a MAX_POS) da diversi insiemi di feature interne:
- osc0:  fase del banco layer 0 (cos φ, sin φ per canale) · ampiezza (|s|) ·
         residual stream dopo il blocco 0
- nopos/cb: residual stream dopo il blocco 0 (la posizione della mask vive lì)

R² su finestre held-out: se la fase decodifica e il resto no, l'indirizzo è nella fase.
Uso: python scripts/probe_posizione_fase.py --arch char-osc0 --run-name fA-osc0-s1 \
       --hub-repo MarioAlessandroNapoli/neuro-llm-ckpt
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from src.models import build_model
from src.models.linoss import (
    DT_INIT, RESET_GROUPS, OscBlock, prefix_scan, prefix_scan_gated,
)

BOUNDARIES = frozenset(b" .,!?\"'\n:;") | {0}
MAX_POS = 16
N_WINDOWS = 48
SEQ = 2048
RIDGE_LAMBDA = 1e-3
SPLIT = 36  # finestre di train; il resto è test


def positions_since_boundary(window: np.ndarray) -> np.ndarray:
    pos, out = 0, np.empty(len(window), dtype=np.float32)
    for i, byte in enumerate(window):
        pos = 0 if int(byte) in BOUNDARIES else min(pos + 1, MAX_POS)
        out[i] = pos
    return out


def ridge_r2(X_tr, y_tr, X_te, y_te):
    X_tr = np.concatenate([X_tr, np.ones((len(X_tr), 1), dtype=np.float32)], axis=1)
    X_te = np.concatenate([X_te, np.ones((len(X_te), 1), dtype=np.float32)], axis=1)
    A = X_tr.T @ X_tr + RIDGE_LAMBDA * np.eye(X_tr.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, X_tr.T @ y_tr)
    resid = y_te - X_te @ w
    return 1 - (resid @ resid) / ((y_te - y_te.mean()) @ (y_te - y_te.mean()))


@torch.no_grad()
def collect(model, cfg, idx):
    """Forward parziale: residual dopo il blocco 0 e, se OscBlock, stati (b,t,m,2)."""
    x = model.tok(idx)
    if model.pos is not None:
        x = x + model.pos(torch.arange(idx.shape[1], device=idx.device))
    blk = model.blocks[0]
    states = None
    if isinstance(blk, OscBlock):
        mx = blk.mixer
        u = blk.ln1(x)
        bu = mx.B(u)
        r = torch.exp(-mx.nu_raw.exp())
        theta = math.pi * torch.sigmoid(mx.theta_raw)
        if mx.no_rotation:
            theta = torch.zeros_like(theta)
        S = r.square()
        dt = torch.full_like(r, DT_INIT)
        A = (S + 1 - 2 * r * torch.cos(theta)) / (dt.square() * S)
        row1 = torch.stack([S, -S * dt * A], dim=-1)
        row2 = torch.stack([dt * S, 1 - dt.square() * S * A], dim=-1)
        M = torch.stack([row1, row2], dim=-2)
        f = torch.stack([dt * S * bu, dt.square() * S * bu], dim=-1)
        M_t = M.unsqueeze(0).expand(idx.shape[1], -1, -1, -1)
        if mx.gate_conv is not None:
            t = u.shape[1]
            boundary = torch.sigmoid(mx.gate_conv(u.transpose(1, 2))[..., :t].transpose(1, 2))
            g = (1 - boundary).repeat_interleave(f.shape[2] // RESET_GROUPS, dim=-1)
            states = prefix_scan_gated(M_t, f, g)
        else:
            states = prefix_scan(M_t, f)
        x = x + mx.C(states[..., 1])
        x = x + blk.mlp(blk.ln2(x))
    else:
        x = blk(x)
    return x, states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hub-repo", default="MarioAlessandroNapoli/neuro-llm-ckpt")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--target", choices=["chunk", "assoluta"], default="chunk",
                        help="chunk = byte dall'ultimo confine (cap 16); assoluta = "
                             "indice nella finestra (controllo alla Haviv, D17)")
    args = parser.parse_args()

    model, cfg = build_model(args.arch)
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(args.hub_repo, f"{args.run_name}/last.ckpt")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()})
    model.eval()

    data = np.memmap(args.data_dir / "valid_bytes.bin", dtype=np.uint8, mode="r")
    rng = np.random.default_rng(20260820)
    starts = rng.choice(len(data) - SEQ - 1, N_WINDOWS, replace=False)
    windows = np.stack([np.asarray(data[s: s + SEQ]) for s in starts])
    idx = torch.from_numpy(windows.astype(np.int64))
    if args.target == "assoluta":
        y = np.tile(np.arange(SEQ, dtype=np.float32), (N_WINDOWS, 1))
    else:
        y = np.stack([positions_since_boundary(w) for w in windows])

    feats = {}
    resids, states_all = [], []
    for i in range(0, N_WINDOWS, 8):
        resid, states = collect(model, cfg, idx[i: i + 8])
        resids.append(resid.float())
        if states is not None:
            states_all.append(states.float())
    resid = torch.cat(resids)
    feats["residual-post-L0"] = resid.numpy()
    if states_all:
        s = torch.cat(states_all)  # (n, t, m, 2): [...,1]=x (letta da C), [...,0]=z
        phase = torch.atan2(s[..., 0], s[..., 1])
        feats["fase (cos,sin)"] = torch.cat([phase.cos(), phase.sin()], dim=-1).numpy()
        feats["ampiezza |s|"] = s.norm(dim=-1).numpy()

    tr, te = slice(0, SPLIT), slice(SPLIT, None)
    y_tr, y_te = y[tr].reshape(-1), y[te].reshape(-1)
    label = ("posizione ASSOLUTA (0-2047)" if args.target == "assoluta"
             else f"posizione-nel-chunk (0-{MAX_POS})")
    print(f"{args.run_name} — probe {label}, train {SPLIT}/test {N_WINDOWS - SPLIT} finestre")
    for name, X in feats.items():
        r2 = ridge_r2(X[tr].reshape(len(y_tr), -1), y_tr, X[te].reshape(len(y_te), -1), y_te)
        print(f"  R² {name:>18}: {r2:.3f}  (dim {X.shape[-1]})")


if __name__ == "__main__":
    main()
