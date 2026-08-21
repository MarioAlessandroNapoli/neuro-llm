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


class RecStack(nn.Module):
    """Stack ricorrente puro: emb + n OscBlock + head. Nessuna attention."""

    def __init__(self, arm, d_model=128, m=256, n_layer=4):
        super().__init__()
        cfg = ModelConfig(vocab_size=VOCAB, d_model=d_model, n_layer=n_layer,
                          n_head=1, seq_len=8192)
        self.tok = nn.Embedding(VOCAB, d_model)

        def ring(i):
            if arm != "ts":
                return (RING_R_MIN, RING_R_MAX)
            lo, hi = TS_TAU_MIN * TS_TAU_FACTOR**i, TS_TAU_MIN * TS_TAU_FACTOR**(i + 1)
            return (math.exp(-1 / lo), math.exp(-1 / hi))

        self.blocks = nn.ModuleList(
            OscBlock(cfg, m, damped=True, phi_init=False, log_polar=True,
                     ring=ring(i), reset=(arm == "gate"), no_rotation=(arm == "gate"))
            for i in range(n_layer)
        )
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
    parser.add_argument("--arm", choices=["lti", "ts", "gate"], required=True)
    parser.add_argument("--n-kv", type=int, default=16)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    model = RecStack(args.arm, args.d_model, args.m).to(device)
    if args.compile:
        model = torch.compile(model)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"mqar {args.arm}: {n_par/1e6:.2f}M param, n_kv={args.n_kv}, seq={args.seq}, "
          f"device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    for step in range(args.steps):
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
            print(f"  step {step}: loss {loss.item():.3f} acc {acc:.3f}")
    acc = evaluate(model, args.n_kv, args.seq, device, rng)
    print(f"FINALE {args.arm} n_kv={args.n_kv} seq={args.seq} seed={args.seed}: "
          f"accuracy {acc:.4f}")


if __name__ == "__main__":
    main()
