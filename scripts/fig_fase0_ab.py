"""Figura fase 0 (D12): l'A/B di training che ha smascherato il backward rotto.

Uso: WANDB_API_KEY nell'ambiente, poi
`uv run --with wandb,matplotlib python scripts/fig_fase0_ab.py [it|en|all]`
Output: docs/figures/2026-08-fase0-ab-scan.png (+ -en). Dati: W&B gruppo scan-ab.
"""
import sys
import math
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("dlinoss-scanAB-eager", {"it": "eager s1 (riferimento)", "en": "eager s1 (reference)"}, "#555555", "-"),
    ("dlinoss-scanAB-eager-s2", {"it": "eager s2 (rumore seed)", "en": "eager s2 (seed noise)"}, "#999999", "-"),
    ("dlinoss-scanAB-hoo", {"it": "hoo backward ROTTO", "en": "hoo BROKEN backward"}, "#C0452C", "--"),
    ("dlinoss-scanAB-hoo2-s1", {"it": "hoo elementwise s1", "en": "hoo elementwise s1"}, "#0F6E6C", "-"),
    ("dlinoss-scanAB-hoo2-s2", {"it": "hoo elementwise s2", "en": "hoo elementwise s2"}, "#4FA39F", "-"),
    ("dlinoss-scanAB-hoo-bf16", {"it": "bf16 (muore ~step 300)", "en": "bf16 (dies ~step 300)"}, "#D98E32", ":"),
]

TEXT = {
    "it": {
        "suptitle": "Fase 0 — validazione dell'apparato: scan fuso e precisione",
        "t1": "A/B di training: stessa run, scan diversi",
        "t2": "Throughput a livello modello",
        "xlab": "global step (5M token = 610 step)",
        "ylab": "train loss (log)",
        "xlab2": "tok/s (migliaia, RTX 3060)",
        "note": "gap hoo-rotto = 7× il rumore seed\n(0,27 vs 0,037): gradienti errati\ninvisibili ai test forward-only",
        "out": "docs/figures/2026-08-fase0-ab-scan.png",
    },
    "en": {
        "suptitle": "Phase 0 — apparatus validation: fused scan and precision",
        "t1": "Training A/B: same run, different scans",
        "t2": "Model-level throughput",
        "xlab": "global step (5M tokens = 610 steps)",
        "ylab": "train loss (log)",
        "xlab2": "tok/s (thousands, RTX 3060)",
        "note": "broken-hoo gap = 7× seed noise\n(0.27 vs 0.037): wrong gradients\ninvisible to forward-only tests",
        "out": "docs/figures/2026-08-fase0-ab-scan-en.png",
    },
}

api = wandb.Api()
curves, toks = {}, {}
for name, _, _, _ in RUNS:
    r = api.run(f"marioalessandronapoli/neuro-llm/{name}")
    h = r.history(keys=["trainer/global_step", "train_loss"], samples=400, pandas=False)
    xs, ys = [], []
    for row in h:
        v = row["train_loss"]
        if v is None or isinstance(v, str) or (isinstance(v, float) and math.isnan(v)):
            break
        xs.append(row["trainer/global_step"])
        ys.append(v)
    curves[name] = (xs, ys)
    ts = r.summary.get("tokens_per_sec")
    toks[name] = float(ts) if ts else 0.0

langs = sys.argv[1] if len(sys.argv) > 1 else "all"
for lang in (["it", "en"] if langs == "all" else [langs]):
    T = TEXT[lang]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), width_ratios=[2.1, 1])
    for name, label, color, ls in RUNS:
        xs, ys = curves[name]
        ax.plot(xs, ys, color=color, ls=ls, lw=1.6, label=label[lang])
        if "bf16" in name and xs:
            ax.scatter([xs[-1]], [ys[-1]], marker="x", s=70, color=color, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel(T["xlab"])
    ax.set_ylabel(T["ylab"])
    ax.set_title(T["t1"], fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    ax.text(0.98, 0.72, T["note"], transform=ax.transAxes, ha="right", fontsize=8,
            color="#C0452C")
    labels = [label[lang] for _, label, _, _ in RUNS]
    ax2.barh(range(len(RUNS)), [toks[name] / 1000 for name, _, _, _ in RUNS],
             color="#0F6E6C")
    ax2.set_yticks(range(len(RUNS)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel(T["xlab2"])
    ax2.set_title(T["t2"], fontsize=11)
    ax2.invert_yaxis()
    fig.suptitle(T["suptitle"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(T["out"], dpi=180)
    print(f"salvata {T['out']}")
