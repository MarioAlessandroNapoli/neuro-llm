"""Figura C1/D17: decomposizione del meccanismo + profili di estrapolazione.

Valori hardcoded con provenienza (registro RESEARCH_LOG, 2026-08-20/21):
- pannello A: val loss 700M byte, engine 32-true (cb 16-mixed, riferimento
  tratteggiato): lti32 0,4527 · heu 0,4523 · rel 0,7571 (transformer) ·
  ts media(0,4357;0,4277) · hard media(0,4295;0,4251) · cb 0,419.
- pannello B: nats/byte per bucket a seq 8192, checkpoint 700M
  (fB-hard-s2, fC1-ts-s2, fA-osc0-s2); cb strutturalmente incapace oltre 2048.
Output: docs/figures/2026-08-c1-decomposizione{,-en}.png
"""
import matplotlib.pyplot as plt

IT = {
    "title_a": "Cosa serve davvero (700M byte, parità)",
    "title_b": "Estrapolazione 2048→8192 (nats/byte per bucket)",
    "bars": ["niente\n(lti32)", "confini cablati\n(heu)", "solo coordinata\n(rel, transf.)",
             "spettro τ\n(ts)", "gate appreso\n(hard)"],
    "cb_line": "transformer + pos emb (cb, 0,419)",
    "wall": "cb: incapace\noltre 2048",
    "xlabel_b": "posizione nella finestra (byte)",
    "suffix": "",
}
EN = {
    "title_a": "What actually matters (700M bytes, parity)",
    "title_b": "Extrapolation 2048→8192 (nats/byte per bucket)",
    "bars": ["nothing\n(lti32)", "hardcoded bounds\n(heu)", "coordinate only\n(rel, transf.)",
             "τ spectrum\n(ts)", "learned gate\n(hard)"],
    "cb_line": "transformer + pos emb (cb, 0.419)",
    "wall": "cb: incapable\nbeyond 2048",
    "xlabel_b": "position in window (bytes)",
    "suffix": "-en",
}

LOSSES = [0.4527, 0.4523, 0.7571, 0.4317, 0.4273]
CB = 0.419
BUCKETS = [1024, 3072, 5120, 7168]
HARD = [0.4326, 0.4231, 0.4432, 0.4750]
TS = [0.4353, 0.4283, 0.4529, 0.4897]
OSC0 = [0.4189, 0.4753, 0.9078, 1.1746]

for L in (IT, EN):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = ["#9e9e9e", "#9e9e9e", "#c98a3d", "#5b8dbf", "#2f6b4f"]
    ax.bar(range(5), LOSSES, color=colors, width=0.62)
    ax.axhline(CB, color="#333", ls="--", lw=1.2)
    ax.text(4.45, CB - 0.012, L["cb_line"], ha="right", fontsize=8.5, color="#333")
    ax.set_xticks(range(5), L["bars"], fontsize=8.5)
    for i, v in enumerate(LOSSES):
        lbl = f"{v:.3f}".replace(".", ",") if L is IT else f"{v:.3f}"
        ax.text(i, v + 0.008, lbl, ha="center", fontsize=9)
    ax.set_ylim(0.38, 0.80)
    ax.set_ylabel("val loss (nats/byte)")
    ax.set_title(L["title_a"], fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    bx.plot(BUCKETS, HARD, "o-", color="#2f6b4f", label="hard")
    bx.plot(BUCKETS, TS, "s-", color="#5b8dbf", label="ts")
    bx.plot(BUCKETS, OSC0, "^-", color="#b4443c", label="osc0")
    bx.axvline(2048, color="#333", ls="--", lw=1.2)
    bx.text(2150, 1.05, L["wall"], fontsize=8.5, color="#333")
    bx.set_xlabel(L["xlabel_b"])
    bx.set_title(L["title_b"], fontsize=11)
    bx.legend(frameon=False)
    bx.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = f"docs/figures/2026-08-c1-decomposizione{L['suffix']}.png"
    fig.savefig(out, dpi=160)
    print(out)
