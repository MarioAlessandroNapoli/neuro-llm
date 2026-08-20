"""Figura sweep ricetta char-baseline (D16, 2026-08-20): mappa lr e accoppiamento batch.

Dati: W&B project neuro-llm, group sweep-char (run swc-cb-*), 200M byte, val_loss in
nats/byte. Uso: python scripts/fig_sweep_char.py [it|en|all]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path("docs/figures")
DATE = "2026-08"

# (lr, val_loss) @ b32 — run swc-cb-b32-lr*
LR_MAP = [(3e-2, 2.169), (1e-2, 0.639), (3e-3, 0.831), (1e-3, 1.199)]
# batch -> [val_loss per seed] @ lr 1e-2 — run swc-cb-b{8,16,32,64}-lr1e-2[-s*]
BATCH_MAP = {8: [0.490, 0.531], 16: [0.506, 0.526, 0.541], 32: [0.639], 64: [2.041]}

TXT = {
    "it": {
        "sup": "Sweep ricetta char-baseline — 200M byte, val loss (nats/byte) · b8-s2 su RTX 5090, resto su RTX 3090",
        "lr": "Mappa lr (batch 32)",
        "batch": "Accoppiamento batch (lr 1e-2)",
        "floor": "pavimento seed σ ≈ 0,018",
        "recipe": "ricetta di fase: b16@1e-2",
        "plateau": "plateau a trigramma (~2,3)",
        "dec": ",",
    },
    "en": {
        "sup": "Char-baseline recipe sweep — 200M bytes, val loss (nats/byte) · b8-s2 on RTX 5090, rest on RTX 3090",
        "lr": "lr map (batch 32)",
        "batch": "Batch coupling (lr 1e-2)",
        "floor": "seed noise floor σ ≈ 0.018",
        "recipe": "phase recipe: b16@1e-2",
        "plateau": "trigram plateau (~2.3)",
        "dec": ".",
    },
}


def fmt(v, dec):
    return f"{v:.3f}".replace(".", dec)


def make(lang):
    t = TXT[lang]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    lrs, vals = zip(*LR_MAP)
    ax1.plot(lrs, vals, "o-", color="#2b6cb0")
    for x, y in LR_MAP:
        ax1.annotate(fmt(y, t["dec"]), (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    ax1.axhline(2.3, ls=":", color="#999")
    ax1.text(1.1e-3, 2.34, t["plateau"], fontsize=8, color="#666")
    ax1.set_xscale("log")
    ax1.set_xlabel("lr")
    ax1.set_ylabel("val loss (nats/byte)")
    ax1.set_title(t["lr"], fontsize=10)

    for b, vs in BATCH_MAP.items():
        ax2.scatter([b] * len(vs), vs, color="#2b6cb0", zorder=3)
        mean = sum(vs) / len(vs)
        ax2.scatter([b], [mean], marker="_", s=400, color="#c53030", zorder=4)
    ax2.axhspan(0.506, 0.541, alpha=0.12, color="#c53030")
    ax2.text(40, 0.56, t["floor"], fontsize=8, color="#c53030")
    ax2.annotate(t["recipe"], (16, 0.506), textcoords="offset points",
                 xytext=(10, -22), fontsize=9, color="#c53030",
                 arrowprops=dict(arrowstyle="->", color="#c53030"))
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])
    ax2.set_xlabel("batch")
    ax2.set_title(t["batch"], fontsize=10)

    fig.suptitle(t["sup"], fontsize=10)
    suffix = "" if lang == "it" else f"-{lang}"
    out = OUT / f"{DATE}-sweep-char{suffix}.png"
    fig.savefig(out, dpi=180)
    print("scritta", out)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    for lang in (["it", "en"] if arg == "all" else [arg]):
        make(lang)
