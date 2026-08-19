"""Figura fase 0 (D12): l'A/B di training che ha smascherato il backward rotto.

Curve train_loss @5M token (dlinoss, b16, lr 3e-3, RTX 3060) per: scan eager (2 seed,
pavimento di rumore), scan hoo con backward Inductor rotto (+0,27, bias), scan hoo
elementwise corretto (2 seed), bf16 (muore a step ~300: niente GradScaler = niente
skip degli step esplosivi). Barre: throughput a livello modello.
Dati: W&B gruppo scan-ab. Uso: WANDB_API_KEY nell'ambiente, poi
`uv run --with wandb,matplotlib python scripts/fig_fase0_ab.py`.
"""
import math
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figures/2026-08-fase0-ab-scan.png"
RUNS = [
    ("dlinoss-scanAB-eager", "eager s1 (riferimento)", "#555555", "-"),
    ("dlinoss-scanAB-eager-s2", "eager s2 (rumore seed)", "#999999", "-"),
    ("dlinoss-scanAB-hoo", "hoo backward ROTTO", "#C0452C", "--"),
    ("dlinoss-scanAB-hoo2-s1", "hoo elementwise s1", "#0F6E6C", "-"),
    ("dlinoss-scanAB-hoo2-s2", "hoo elementwise s2", "#4FA39F", "-"),
    ("dlinoss-scanAB-hoo-bf16", "bf16 (muore ~step 300)", "#D98E32", ":"),
]

api = wandb.Api()
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), width_ratios=[2.1, 1])

toks = {}
for name, label, color, ls in RUNS:
    r = api.run(f"marioalessandronapoli/neuro-llm/{name}")
    h = r.history(keys=["trainer/global_step", "train_loss"], samples=400, pandas=False)
    xs, ys = [], []
    for row in h:
        v = row["train_loss"]
        if v is None or isinstance(v, str) or (isinstance(v, float) and math.isnan(v)):
            break
        xs.append(row["trainer/global_step"])
        ys.append(v)
    ax.plot(xs, ys, color=color, ls=ls, lw=1.6, label=label)
    if "bf16" in name and xs:
        ax.scatter([xs[-1]], [ys[-1]], marker="x", s=70, color=color, zorder=5)
    ts = r.summary.get("tokens_per_sec")
    if ts:
        toks[label] = float(ts)

ax.set_yscale("log")
ax.set_xlabel("global step (5M token = 610 step)")
ax.set_ylabel("train loss (log)")
ax.set_title("A/B di training: stessa run, scan diversi", fontsize=11)
ax.legend(frameon=False, fontsize=8.5)
ax.text(0.98, 0.72, "gap hoo-rotto = 7× il rumore seed\n(0,27 vs 0,037): gradienti errati\ninvisibili ai test forward-only",
        transform=ax.transAxes, ha="right", fontsize=8, color="#C0452C")

labels = list(toks)
ax2.barh(range(len(labels)), [toks[l] / 1000 for l in labels], color="#0F6E6C")
ax2.set_yticks(range(len(labels)))
ax2.set_yticklabels(labels, fontsize=8)
ax2.set_xlabel("tok/s (migliaia, RTX 3060)")
ax2.set_title("Throughput a livello modello", fontsize=11)
ax2.invert_yaxis()

fig.suptitle("Fase 0 — validazione dell'apparato: scan fuso e precisione", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT, dpi=180)
print(f"salvata {OUT}")
