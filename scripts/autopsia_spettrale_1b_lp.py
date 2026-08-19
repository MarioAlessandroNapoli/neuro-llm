"""Autopsia spettrale dei dlinoss-lp @3e-2 170M (s1, s2): dove sono finiti gli r?

In log-polare r = exp(-exp(nu)) e theta = pi*sigmoid(theta_raw) si leggono esatti dai
parametri. Confronto: init U[0.9, 1] e il classico dlinoss@3e-3 dell'autopsia 1a
(r mediana 0.74 -> 0.90 crescente con la profondita').
"""
import math
import torch
from huggingface_hub import hf_hub_download

REPO = "MarioAlessandroNapoli/neuro-llm-ckpt"
RUNS = [
    "dlinoss-lp-d256-L8-t170M-s1-lr3e-2",
    "dlinoss-lp-d256-L8-t170M-s2-lr3e-2",
]

for run in RUNS:
    path = hf_hub_download(REPO, f"{run}/last.ckpt")
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    print(f"\n=== {run} ===")
    layers = sorted({k.rsplit(".mixer.", 1)[0] for k in sd if ".mixer.nu_raw" in k})
    for lk in layers:
        nu = sd[f"{lk}.mixer.nu_raw"].float()
        th_raw = sd[f"{lk}.mixer.theta_raw"].float()
        r = torch.exp(-nu.exp())
        theta = math.pi * torch.sigmoid(th_raw)
        period = 2 * math.pi / theta
        B_n = sd[f"{lk}.mixer.B.weight"].norm().item()
        C_n = sd[f"{lk}.mixer.C.weight"].norm().item()

        def q(t, p):
            return torch.quantile(t, p).item()

        print(
            f"  {lk}: r med {q(r,0.5):.4f} [p10 {q(r,0.1):.4f}, p90 {q(r,0.9):.4f}]"
            f"  frac r>0.99: {(r>0.99).float().mean().item():.2f}"
            f"  frac r<0.5: {(r<0.5).float().mean().item():.2f}"
            f"  | periodo med {q(period,0.5):.1f} tok [p90 {q(period,0.9):.1f}]"
            f"  | ||B||={B_n:.1f} ||C||={C_n:.1f}"
        )
