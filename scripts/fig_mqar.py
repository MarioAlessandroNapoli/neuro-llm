"""Figura MQAR (D17): recall associativo sui soli stack ricorrenti, 2 seed.

Valori hardcoded con provenienza (registro RESEARCH_LOG, 2026-08-21, 4070S):
accuracy sulle risposte, caso = 1/120 ≈ 0,0083; per (braccio, n_kv, seq) i due
seed; gate−8 = cura dell'orizzonte di apprendibilità (bias init −8): a 1024
nkv=8 ha 2 seed (0,162 s1; 0,2278 s2), gli altri carichi 1 seed (s2).
Output: docs/figures/2026-08-mqar{,-en}.png + paper/fig-mqar.png (EN)
"""
import matplotlib.pyplot as plt

NKV = [8, 16, 32, 64]
CHANCE = 1 / 120

S256 = {
    "lti":  [(0.2227, 0.2930), (0.1270, 0.1669), (0.0674, 0.0641), (0.0263, 0.0325)],
    "ts":   [(0.3552, 0.3616), (0.2209, 0.2123), (0.1038, 0.0822), (0.0336, 0.0400)],
    "gate": [(0.3289, 0.3430), (0.2280, 0.2329), (0.1505, 0.1533), (0.0495, 0.0452)],
}
S1024 = {
    "lti":  [(0.1968, 0.0603), (0.0952, 0.0475), (0.0216, 0.0284), (0.0081, 0.0122)],
    "ts":   [(0.3462, 0.3479), (0.1821, 0.1483), (0.0592, 0.0601), (0.0174, 0.0298)],
    "gate": [(0.0081, 0.0720), (0.0085, 0.0139), (0.0131, 0.0205), (0.0078, 0.0247)],
    "gate8": [(0.162, 0.2278), (0.1360,), (0.0900,), (0.0555,)],
}

STYLE = {
    "lti": ("#9e9e9e", "o", "lti"),
    "ts": ("#5b8dbf", "s", "ts"),
    "gate": ("#2f6b4f", "^", "gate"),
    "gate8": ("#c98a3d", "D", None),  # label per lingua
}

IT = {
    "title_a": "seq = 256 (rumore ~200 byte)",
    "title_b": "seq = 1024 (rumore ~900 byte)",
    "gate8": "gate, init −8 (cura)",
    "chance": "caso (1/120)",
    "xlabel": "coppie chiave→valore (n_kv)",
    "ylabel": "accuracy sulle risposte",
    "suffix": "",
}
EN = {
    "title_a": "seq = 256 (~200 noise bytes)",
    "title_b": "seq = 1024 (~900 noise bytes)",
    "gate8": "gate, init −8 (cure)",
    "chance": "chance (1/120)",
    "xlabel": "key→value pairs (n_kv)",
    "ylabel": "accuracy on answers",
    "suffix": "-en",
}

for L in (IT, EN):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, data, title in ((axes[0], S256, L["title_a"]), (axes[1], S1024, L["title_b"])):
        for arm, seeds in data.items():
            color, marker, label = STYLE[arm]
            if arm == "gate8":
                label = L["gate8"]
            mean = [sum(v) / len(v) for v in seeds]
            ax.plot(NKV, mean, marker + "-", color=color, label=label, zorder=3)
            for x, v in zip(NKV, seeds):
                if len(v) > 1:
                    ax.plot([x, x], [min(v), max(v)], "-", color=color, lw=1, alpha=0.55)
        ax.axhline(CHANCE, color="#333", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(NKV, [str(n) for n in NKV])
        ax.set_xlabel(L["xlabel"])
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(L["ylabel"])
    axes[1].text(60, CHANCE * 1.12, L["chance"], ha="right", fontsize=8.5, color="#333")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = f"docs/figures/2026-08-mqar{L['suffix']}.png"
    fig.savefig(out, dpi=160)
    print(out)
    if L is EN:
        fig.savefig("paper/fig-mqar.png", dpi=160)
