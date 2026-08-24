"""Figura MQAR-v2 (D18): recall associativo puro, banco senza scorciatoia, 2 seed.

Valori hardcoded con provenienza (registro RESEARCH_LOG, griglia v2 2026-08-24,
4080; protocollo 10k step, banco v2 con segnaposto): accuracy su dati freschi,
caso = 1/120 ≈ 0,0083; per (braccio, n_kv, seq) i due seed. gate−8 = cura
dell'orizzonte di apprendibilità (bias init −8), solo a seq 1024.
Output: docs/figures/2026-08-mqar-v2{,-en}.png + paper/fig-mqar.png (EN)
"""
import matplotlib.pyplot as plt

CHANCE = 1 / 120
NKV256, NKV1024 = [8, 16, 32], [8, 32]

S256 = {
    "lti":  [(0.1191, 0.1160), (0.0626, 0.0604), (0.0300, 0.0313)],
    "ts":   [(0.1472, 0.1394), (0.0748, 0.0747), (0.0436, 0.0381)],
    "gate": [(0.1453, 0.1475), (0.0981, 0.0978), (0.0648, 0.0656)],
}
S1024 = {
    "lti":  [(0.0999, 0.0469), (0.0224, 0.0167)],
    "ts":   [(0.1462, 0.1357), (0.0330, 0.0342)],
    "gate": [(0.0071, 0.0469), (0.0087, 0.0318)],
    "gate8": [(0.1143, 0.1311), (0.0513, 0.0547)],
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
    "ylabel": "accuracy su dati freschi",
    "suffix": "",
}
EN = {
    "title_a": "seq = 256 (~200 noise bytes)",
    "title_b": "seq = 1024 (~900 noise bytes)",
    "gate8": "gate, init −8 (cure)",
    "chance": "chance (1/120)",
    "xlabel": "key→value pairs (n_kv)",
    "ylabel": "accuracy on fresh data",
    "suffix": "-en",
}

for L in (IT, EN):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, data, nkv, title in ((axes[0], S256, NKV256, L["title_a"]),
                                 (axes[1], S1024, NKV1024, L["title_b"])):
        for arm, seeds in data.items():
            color, marker, label = STYLE[arm]
            if arm == "gate8":
                label = L["gate8"]
            mean = [sum(v) / len(v) for v in seeds]
            ax.plot(nkv, mean, marker + "-", color=color, label=label, zorder=3)
            for x, v in zip(nkv, seeds):
                ax.plot([x, x], [min(v), max(v)], "-", color=color, lw=1, alpha=0.55)
        ax.axhline(CHANCE, color="#333", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(nkv, [str(n) for n in nkv])
        ax.set_xlabel(L["xlabel"])
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(L["ylabel"])
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    axes[1].text(30, CHANCE * 1.15, L["chance"], ha="right", fontsize=8.5,
                 color="#333")
    fig.tight_layout()
    out = f"docs/figures/2026-08-mqar-v2{L['suffix']}.png"
    fig.savefig(out, dpi=160)
    print(out)
    if L is EN:
        fig.savefig("paper/fig-mqar.png", dpi=160)
