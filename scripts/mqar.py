"""MQAR — Multi-Query Associative Recall sui soli stack ricorrenti (D17 post-C1).

Task sintetico (linea Zoology/Based, Arora et al.): la sequenza apre con K coppie
chiave→valore (alfabeti disgiunti), prosegue con rumore di riempimento, chiude con
le query (le chiavi riproposte in ordine casuale); il modello deve emettere il
valore giusto dopo ogni query. Loss e accuratezza SOLO sulle posizioni-risposta.

Perché senza attention: l'attention risolve il recall per costruzione e maschererebbe
tutto — si confrontano gli stack oscillatori puri (4 layer + testa), come da pratica
della letteratura. Bracci: lti (ring 0,9-1) · ts (bande τ gerarchiche) · gate
(reset appreso, θ≡0 — il vincitore dello stadio char). Domanda: la capacità lunga
che il gate tiene in vita (autopsia spettrale) è FUNZIONALE? E in un task senza
confini linguistici, il gate impara a resettare sugli eventi del task (le chiavi)?

Uso: python -m scripts.mqar --arm gate --n-kv 16 --seq 512 --steps 3000
"""
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.linoss import RING_R_MAX, RING_R_MIN, OscBlock
from src.configs import ModelConfig
from src.models.hybrid import TS_TAU_FACTOR, TS_TAU_MIN

VOCAB = 256
KEY_LO, KEY_HI = 1, 120      # chiavi
VAL_LO, VAL_HI = 128, 247    # valori
NOISE_LO, NOISE_HI = 248, 256  # riempitivo (mai chiave né valore)


def make_batch(bs, n_kv, seq, device, rng):
    """(input (bs,seq), target (bs,seq) con -100 fuori dalle risposte)."""
    x = torch.randint(NOISE_LO, NOISE_HI, (bs, seq), generator=rng)
    y = torch.full((bs, seq), -100, dtype=torch.long)
    n_q = n_kv  # tutte le chiavi vengono richieste
    for b in range(bs):
        keys = torch.randperm(KEY_HI - KEY_LO, generator=rng)[:n_kv] + KEY_LO
        vals = torch.randint(VAL_LO, VAL_HI, (n_kv,), generator=rng)
        for i in range(n_kv):
            x[b, 2 * i], x[b, 2 * i + 1] = keys[i], vals[i]
        order = torch.randperm(n_kv, generator=rng)
        q0 = seq - 2 * n_q
        for j, i in enumerate(order):
            x[b, q0 + 2 * j] = keys[i]
            x[b, q0 + 2 * j + 1] = vals[i]
            y[b, q0 + 2 * j] = vals[i]  # dopo la query si predice il valore
    return x.to(device), y.to(device)


class S6Mixer(nn.Module):
    """Mamba/S6 minimale (D18 passo A): selettività di riferimento del campo.

    Recurrence diagonale per canale: h_t = exp(Δ_t·A)·h_{t-1} + Δ_t·B_t·x_t,
    y_t = C_t·h_t + D·x_t, con Δ, B, C tutti data-dependent (la selettività).
    Scan doubling SENZA divisioni (solo prodotti di a≤1: underflow benigno,
    backward limitato — la variante cumsum(b/cumprod) esplode in backward),
    a chunk checkpointati: (b,L,d,N) intero non entra mai in memoria.
    Stato = d_inner·N float.
    """

    CHUNK = 64
    DT_RANK = 8

    def __init__(self, d_model, d_state=16, expand=2, d_conv=4,
                 dt_min=0.001, dt_max=0.1):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner, self.d_state = d_inner, d_state
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner,
                              padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.DT_RANK + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.DT_RANK, d_inner)
        # Init Mamba standard: Δ ~ logU[dt_min, dt_max] (canone 0,001-0,1), A reale
        # S4D −1..−N. Orizzonte di apprendibilità (D17, seconda incarnazione): la
        # scrittura ZOH è Δ·B·x — con Δ piccolo la chiave entra nello stato ~1% delle
        # attivazioni e il gradiente del percorso di memoria muore all'init.
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1)
                                            .float()).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _chunk(self, dt, x, B, C, h):
        """Un chunk: costruisce a, b e scanna. (b,C,d,N) vive solo qui dentro."""
        A = -torch.exp(self.A_log.float())                    # (d,N)
        a = torch.exp(dt.unsqueeze(-1) * A)                   # (b,c,d,N)
        f = (dt * x).unsqueeze(-1) * B.unsqueeze(2)           # (b,c,d,N)
        f = torch.cat([(f[:, :1] + a[:, :1] * h.unsqueeze(1)), f[:, 1:]], dim=1)
        stride = 1
        while stride < a.shape[1]:
            f = torch.cat([f[:, :stride],
                           f[:, stride:] + a[:, stride:] * f[:, :-stride]], dim=1)
            a = torch.cat([a[:, :stride],
                           a[:, stride:] * a[:, :-stride]], dim=1)
            stride *= 2
        return torch.einsum("bldn,bln->bld", f, C), f[:, -1]

    def forward(self, u):
        b, L, _ = u.shape
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2)
        x = F.silu(x)
        dt_r, B, C = self.x_proj(x).split(
            [self.DT_RANK, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_r)).float()           # (b,L,d)
        h = x.new_zeros(b, self.d_inner, self.d_state, dtype=torch.float32)
        ys = []
        for s in range(0, L, self.CHUNK):
            e = s + self.CHUNK
            yc, h = torch.utils.checkpoint.checkpoint(
                self._chunk, dt[:, s:e], x[:, s:e].float(),
                B[:, s:e].float(), C[:, s:e].float(), h, use_reentrant=False)
            ys.append(yc)
        y = torch.cat(ys, dim=1).to(u.dtype) + self.D * x
        return self.out_proj(y * F.silu(z))


class MambaBlock(nn.Module):
    """Gemello di OscBlock (stessa LN/MLP): cambia solo il mixer."""

    def __init__(self, cfg, d_state, dt_min=0.001, dt_max=0.1, mlp=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.mixer = S6Mixer(cfg.d_model, d_state, dt_min=dt_min, dt_max=dt_max)
        # Variante "pure" = canone Mamba: solo mixer, niente MLP (che nel nostro
        # scheletro è un bypass per fittare la marginale senza usare lo stato)
        self.ln2 = nn.LayerNorm(cfg.d_model) if mlp else None
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        ) if mlp else None

    def forward(self, x):
        x = x + self.mixer(self.ln1(x))
        return x + self.mlp(self.ln2(x)) if self.mlp else x


class RecStack(nn.Module):
    """Stack ricorrente puro: emb + n OscBlock + head. Nessuna attention."""

    def __init__(self, arm, d_model=128, m=256, n_layer=4, gate_bias=None,
                 d_state=16, dt_min=0.001, dt_max=0.1, pure=False):
        super().__init__()
        cfg = ModelConfig(vocab_size=VOCAB, d_model=d_model, n_layer=n_layer,
                          n_head=1, seq_len=8192)
        self.tok = nn.Embedding(VOCAB, d_model)
        # Parità D18: si dichiarano parametri E stato (legge Based: recall ∝ stato)
        self.state_per_layer = (2 * d_model * d_state if arm == "mamba" else 2 * m)

        if arm == "mamba":
            # pure: 8 blocchi solo-mixer (canone Mamba), stessa parità ~1,0M
            self.blocks = nn.ModuleList(
                MambaBlock(cfg, d_state, dt_min, dt_max, mlp=not pure)
                for _ in range(8 if pure else n_layer))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, VOCAB, bias=False)
            return

        def ring(i):
            if arm != "ts":
                return (RING_R_MIN, RING_R_MAX)
            lo, hi = TS_TAU_MIN * TS_TAU_FACTOR**i, TS_TAU_MIN * TS_TAU_FACTOR**(i + 1)
            return (math.exp(-1 / lo), math.exp(-1 / hi))

        self.blocks = nn.ModuleList(
            OscBlock(cfg, m, damped=True, phi_init=False, log_polar=True,
                     ring=ring(i), reset=arm in ("gate", "gaterot"),
                     no_rotation=(arm == "gate"))
            for i in range(n_layer)
        )
        if gate_bias is not None:
            # Orizzonte di apprendibilità: all'init la memoria sopravvive a T byte di
            # rumore con fattore ~exp(-T·sigma(bias)) — il bias fissa l'orizzonte.
            for blk in self.blocks:
                if blk.mixer.gate_conv is not None:
                    nn.init.constant_(blk.mixer.gate_conv.bias, gate_bias)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)

    def forward(self, idx):
        x = self.tok(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


@torch.no_grad()
def evaluate(model, n_kv, seq, device, rng, n_batches=8, bs=64):
    hits = tot = 0
    for _ in range(n_batches):
        x, y = make_batch(bs, n_kv, seq, device, rng)
        pred = model(x).argmax(-1)
        mask = y != -100
        hits += (pred[mask] == y[mask]).sum().item()
        tot += mask.sum().item()
    return hits / tot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["lti", "ts", "gate", "gaterot", "mamba"],
                        required=True)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--dt-min", type=float, default=0.001)
    parser.add_argument("--dt-max", type=float, default=0.1)
    parser.add_argument("--pure", action="store_true")
    # Ricetta Zoology: dataset FINITO riusato per epoche (0 = on-the-fly, D17).
    # La ripetizione apre il sentiero memorizzazione→generalizzazione che il
    # flusso infinito non innesca; eval sempre su dati freschi.
    parser.add_argument("--n-train", type=int, default=0)
    parser.add_argument("--n-kv", type=int, default=16)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--gate-bias", type=float, default=None)
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    model = RecStack(args.arm, args.d_model, args.m, gate_bias=args.gate_bias,
                     d_state=args.d_state, dt_min=args.dt_min,
                     dt_max=args.dt_max, pure=args.pure).to(device)
    if args.compile:
        model = torch.compile(model)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"mqar {args.arm}: {n_par/1e6:.2f}M param, "
          f"stato {model.state_per_layer} float/layer, n_kv={args.n_kv}, "
          f"seq={args.seq}, device={device}")

    if args.n_train:
        parts = [make_batch(512, args.n_kv, args.seq, device, rng)
                 for _ in range(args.n_train // 512)]
        X = torch.cat([p[0] for p in parts])
        Y = torch.cat([p[1] for p in parts])
        print(f"dataset finito: {len(X)} esempi, "
              f"~{args.steps * args.bs / len(X):.0f} epoche")
        perm, ptr = torch.randperm(len(X), generator=rng).to(device), 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    for step in range(args.steps):
        if args.n_train:
            if ptr + args.bs > len(X):
                perm, ptr = torch.randperm(len(X), generator=rng).to(device), 0
            idx = perm[ptr:ptr + args.bs]
            ptr += args.bs
            x, y = X[idx], Y[idx]
        else:
            x, y = make_batch(args.bs, args.n_kv, args.seq, device, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            acc = evaluate(model, args.n_kv, args.seq, device, rng, n_batches=2,
                           bs=args.bs)
            extra = ""
            if args.n_train:
                with torch.no_grad():
                    mask = y != -100
                    tr = (logits.argmax(-1)[mask] == y[mask]).float().mean()
                extra = f" train_acc {tr:.3f}"
            print(f"  step {step}: loss {loss.item():.3f} acc {acc:.3f}{extra}",
                  flush=True)
    acc = evaluate(model, args.n_kv, args.seq, device, rng)
    tag = (f"mamba{'p' if args.pure else ''}-N{args.d_state}-dt{args.dt_max:g}"
           if args.arm == "mamba" else args.arm)
    if args.n_train:
        tag += f"-ft{args.n_train // 1000}k"
    print(f"FINALE {tag} n_kv={args.n_kv} seq={args.seq} seed={args.seed}: "
          f"accuracy {acc:.4f}")


if __name__ == "__main__":
    main()
