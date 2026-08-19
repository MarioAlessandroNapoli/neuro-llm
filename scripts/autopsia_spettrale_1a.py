"""Autopsia spettrale dei checkpoint della griglia 1a (D12 pre-step).

Dai pesi addestrati ricava per ogni layer oscillatorio: distribuzione di r (modulo
autovalori), periodi 2pi/theta, saturazione del clamp, norme di B/C. Confronta con
l'init (r ~ U[0.9,1], theta ~ U[0,pi] -> periodo mediano 4 token).
"""
import math
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

REPO = "MarioAlessandroNapoli/neuro-llm-ckpt"
RUNS = [
    "dlinoss-d256-L8-t170M-s1-lr3e-3",
    "dlinoss-d256-L8-t170M-s2-lr3e-3",
    "dlinoss-d256-L8-t170M-s1-lr1e-2",
    "hyb-ao-d256-L8-t170M-s1-lr3e-3",
    "hyb-ao-d256-L8-t170M-s2-lr3e-3",
    "hyb-ao-d256-L8-t170M-s3-lr3e-3",
    "hyb-oa-d256-L8-t170M-s1-lr1e-2",
    "hyb-oa-d256-L8-t170M-s2-lr1e-2",
]


def analyze(run):
    path = hf_hub_download(REPO, f"{run}/last.ckpt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    step = ckpt.get("global_step", "?")
    print(f"\n=== {run}  (global_step {step}) ===")

    nan_keys = [k for k, v in sd.items() if v.is_floating_point() and torch.isnan(v).any()]
    if nan_keys:
        print(f"  !! {len(nan_keys)} tensori con NaN (checkpoint post-esplosione), es. {nan_keys[0]}")

    layers = sorted({k.rsplit(".mixer.", 1)[0] for k in sd if ".mixer.A_raw" in k})
    for lk in layers:
        A_raw = sd[f"{lk}.mixer.A_raw"].float()
        G_raw = sd[f"{lk}.mixer.G_raw"].float()
        dt_raw = sd[f"{lk}.mixer.dt_raw"].float()
        if torch.isnan(A_raw).any() or torch.isnan(G_raw).any():
            print(f"  {lk}: parametri spettrali NaN — non analizzabile")
            continue
        dt = torch.sigmoid(dt_raw)
        S = 1 / (1 + dt * F.relu(G_raw))
        sqrt_S = S.sqrt()
        denom = dt.square() * S
        lo, hi = (1 - sqrt_S).square() / denom, (1 + sqrt_S).square() / denom
        A_pre = F.relu(A_raw)
        clamped_lo = (A_pre <= lo * 1.001).float().mean().item()
        clamped_hi = (A_pre >= hi * 0.999).float().mean().item()
        A = A_pre.clamp(lo, hi)

        M = torch.zeros(A.shape[0], 2, 2)
        M[:, 0, 0] = S
        M[:, 0, 1] = -S * dt * A
        M[:, 1, 0] = dt * S
        M[:, 1, 1] = 1 - dt.square() * S * A
        ev = torch.linalg.eigvals(M)
        r = ev.abs().amax(dim=-1)  # per coppia coniugata i moduli coincidono
        theta = ev.angle().abs().amax(dim=-1)
        period = 2 * math.pi / theta.clamp_min(1e-6)

        B_n = sd[f"{lk}.mixer.B.weight"].norm().item()
        C_n = sd[f"{lk}.mixer.C.weight"].norm().item()

        def q(t, p):
            return torch.quantile(t, p).item()

        print(
            f"  {lk}: r med {q(r,0.5):.4f} [p10 {q(r,0.1):.4f}, p90 {q(r,0.9):.4f}]"
            f"  frac r>0.99: {(r>0.99).float().mean().item():.2f}"
            f"  | periodo med {q(period,0.5):.1f} tok [p10 {q(period,0.1):.1f}, p90 {q(period,0.9):.1f}]"
            f"  | clamp lo/hi: {clamped_lo:.2f}/{clamped_hi:.2f}"
            f"  | ||B||={B_n:.1f} ||C||={C_n:.1f}"
        )


for run in RUNS:
    try:
        analyze(run)
    except Exception as e:
        print(f"\n=== {run} ===\n  ERRORE: {e}")
