"""Autopsia dei checkpoint MQAR-v2 (D18): cosa ha imparato ogni meccanismo.

Per ogni ckpt in ckpt/mqar-*.pt: (1) spettro appreso per layer (r → τ, quantili,
canali lunghi) vs init; (2) per i bracci col reset, probabilità di gate per
regione del task (coppie / rumore / chiavi-query / segnaposto) — il reset ha
imparato la STRUTTURA del task o spara a caso?

Uso: python -m scripts.autopsia_mqar [--ckpt-dir ckpt]
"""
import argparse
import glob
import math
import re

import torch

from scripts.mqar import RecStack, make_batch


def regioni(n_kv, seq):
    """Indici per regione: coppie, rumore, chiavi (query), segnaposto."""
    q0 = seq - 2 * n_kv
    return {
        "coppie": list(range(0, 2 * n_kv)),
        "rumore": list(range(2 * n_kv, q0)),
        "chiavi-query": list(range(q0, seq, 2)),
        "segnaposto": list(range(q0 + 1, seq, 2)),
    }


def autopsia(path, device):
    name = path.split("/")[-1]
    mm = re.match(r"mqar-([a-z]+).*?-k(\d+)-q(\d+)-s(\d+)\.pt", name)
    arm, n_kv, seq, seed = mm.group(1), *map(int, mm.groups()[1:])
    model = RecStack(arm, 128, 256)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    print(f"\n=== {name} (arm {arm}, nkv {n_kv}, seq {seq}) ===")

    for i, blk in enumerate(model.blocks):
        r = torch.exp(-blk.mixer.nu_raw.detach().exp()).flatten()
        tau = -1 / torch.log(r.clamp(max=0.999999))
        q = torch.quantile(tau, torch.tensor([0.1, 0.5, 0.9], device=tau.device))
        print(f"  layer {i}: tau p10/50/90 = {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f}, "
              f"canali tau>500: {(tau > 500).sum().item()}/{len(tau)}")

    if model.blocks[0].mixer.gate_conv is None:
        return
    probs = []

    def hook(_m, _i, out):
        probs.append(torch.sigmoid(out.detach()))

    handles = [blk.mixer.gate_conv.register_forward_hook(hook)
               for blk in model.blocks]
    rng = torch.Generator().manual_seed(99)
    with torch.no_grad():
        x, _ = make_batch(64, n_kv, seq, device, rng)
        model(x)
    for h in handles:
        h.remove()
    reg = regioni(n_kv, seq)
    for i, p in enumerate(probs):
        p = p[..., :seq].mean(dim=(0, 1))  # media su batch e gruppi → per posizione
        riga = " · ".join(f"{k} {p[v].mean():.3f}" for k, v in reg.items())
        print(f"  gate layer {i}: {riga}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="ckpt")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(glob.glob(f"{args.ckpt_dir}/mqar-*.pt"))
    if not paths:
        raise RuntimeError(f"nessun checkpoint in {args.ckpt_dir}/")
    for p in paths:
        autopsia(p, device)


if __name__ == "__main__":
    main()
