"""Figura autopsia spettrale (griglia 1b): spettri appresi per layer.

Uso: `uv run --with matplotlib python scripts/fig_autopsia.py [it|en|all]`
Output: docs/figures/2026-08-autopsia-spettrale-1b.png (+ -en per l'inglese).
Dati: checkpoint HF (cache locale dopo il primo download).
"""
import sys
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

REPO = "MarioAlessandroNapoli/neuro-llm-ckpt"

MODELS = [
    ("dlinoss-d256-L8-t170M-s1-lr3e-3", {"it": "dlinoss classico @3e-3", "en": "vanilla dlinoss @3e-3"}, "classic", 1.9320),
    ("dlinoss-lp-d256-L8-t170M-s1-lr3e-2", {"it": "dlinoss-lp @3e-2 (s1)", "en": "dlinoss-lp @3e-2 (s1)"}, "lp", 1.9498),
    ("dlinoss-lp-d256-L8-t170M-s2-lr3e-2", {"it": "dlinoss-lp @3e-2 (s2)", "en": "dlinoss-lp @3e-2 (s2)"}, "lp", 1.9672),
    ("dlinoss-lp-init-d256-L8-t170M-s1-lr3e-2", {"it": "dlinoss-lp-init @3e-2 (s1)", "en": "dlinoss-lp-init @3e-2 (s1)"}, "lp", 1.9450),
    ("hyb-oa-lp-d256-L8-t170M-s1-lr3e-2", {"it": "hyb-oa-lp @3e-2 (osc 0-3, attn 4-7)", "en": "hyb-oa-lp @3e-2 (osc 0-3, attn 4-7)"}, "lp", 1.7225),
]

TEXT = {
    "it": {
        "title": "Autopsia spettrale: dove finiscono gli autovalori appresi",
        "ylab": "r = |λ| appreso  (mediana in rosso)",
        "ylab2": "norma di Frobenius",
        "xlab": "layer",
        "val": "val@170M = {v}",
        "foot": ("Riferimenti val@170M: baseline transformer 1,599±0,007 (lr 3e-2) · baseline lr-matched @3e-3: 1,700. "
                 "Init: r ~ U[0,9; 1] (lp-init: U[0,7; 0,9]). Log-polare a lr piena: layer 0 = banco di filtri "
                 "(‖B‖,‖C‖ raddoppiate), layer 1-7 potati (r→0) o silenziati (‖B‖,‖C‖→0)."),
        "dec": ",",
        "out": "docs/figures/2026-08-autopsia-spettrale-1b.png",
    },
    "en": {
        "title": "Spectral autopsy: where the learned eigenvalues end up",
        "ylab": "learned r = |λ|  (median in red)",
        "ylab2": "Frobenius norm",
        "xlab": "layer",
        "val": "val@170M = {v}",
        "foot": ("val@170M references: transformer baseline 1.599±0.007 (lr 3e-2) · lr-matched baseline @3e-3: 1.700. "
                 "Init: r ~ U[0.9, 1] (lp-init: U[0.7, 0.9]). Log-polar at full lr: layer 0 = filter bank "
                 "(‖B‖,‖C‖ doubled), layers 1-7 pruned (r→0) or silenced (‖B‖,‖C‖→0)."),
        "dec": ".",
        "out": "docs/figures/2026-08-autopsia-spettrale-1b-en.png",
    },
}


def load_layers(run, kind):
    sd = torch.load(hf_hub_download(REPO, f"{run}/last.ckpt"), map_location="cpu",
                    weights_only=False)["state_dict"]
    key = ".mixer.nu_raw" if kind == "lp" else ".mixer.G_raw"
    layers = sorted({k.rsplit(".mixer.", 1)[0] for k in sd if key in k})
    out = []
    for lk in layers:
        if kind == "lp":
            r = torch.exp(-sd[f"{lk}.mixer.nu_raw"].float().exp())
        else:
            dt = torch.sigmoid(sd[f"{lk}.mixer.dt_raw"].float())
            S = 1 / (1 + dt * F.relu(sd[f"{lk}.mixer.G_raw"].float()))
            r = S.sqrt()
        out.append((r, sd[f"{lk}.mixer.B.weight"].norm().item(),
                    sd[f"{lk}.mixer.C.weight"].norm().item()))
    return out


def render(lang, data):
    T = TEXT[lang]
    n = len(data)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.4), height_ratios=[2.1, 1],
                             sharey="row")
    fig.suptitle(T["title"], fontsize=13, y=0.98)
    for j, (run, label, kind, vloss) in enumerate(MODELS):
        layers = data[run]
        ax = axes[0, j]
        torch.manual_seed(0)
        for li, (r, _, _) in enumerate(layers):
            x = li + (torch.rand(len(r)) - 0.5) * 0.55
            ax.scatter(x, r, s=2.5, alpha=0.22, color="#0F6E6C", linewidths=0)
            ax.plot([li - 0.3, li + 0.3], [r.median()] * 2, color="#C0452C", lw=2.2,
                    zorder=3)
        ax.axhline(1.0, color="#888", lw=0.8, ls="--")
        vtxt = f"{vloss:.3f}".replace(".", T["dec"])
        ax.set_title(f"{label[lang]}\n{T['val'].format(v=vtxt)}", fontsize=10)
        ax.set_xlabel(T["xlab"])
        ax.set_xticks(range(len(layers)))
        ax.set_ylim(-0.03, 1.05)
        if j == 0:
            ax.set_ylabel(T["ylab"])
        ax2 = axes[1, j]
        idx = range(len(layers))
        ax2.bar([i - 0.18 for i in idx], [b for _, b, _ in layers], width=0.36,
                label="‖B‖", color="#0F6E6C")
        ax2.bar([i + 0.18 for i in idx], [c for _, _, c in layers], width=0.36,
                label="‖C‖", color="#C0452C")
        ax2.set_xlabel(T["xlab"])
        ax2.set_xticks(list(idx))
        if j == 0:
            ax2.set_ylabel(T["ylab2"])
            ax2.legend(frameon=False, fontsize=9)
    fig.text(0.5, 0.005, T["foot"], ha="center", fontsize=8.5, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(T["out"], dpi=180)
    print(f"salvata {T['out']}")


langs = sys.argv[1] if len(sys.argv) > 1 else "all"
langs = ["it", "en"] if langs == "all" else [langs]
data = {run: load_layers(run, kind) for run, _, kind, _ in MODELS}
for lang in langs:
    render(lang, data)
