"""Spike fase 0 (D12): fusione dello scan associativo.

Confronta su GPU, a shape reali (b32, t512, m512):
  A. scan attuale eager (_scan_fp32, loop log2(t) con cat)
  B. scan attuale sotto torch.compile
  C. torch.associative_scan (higher-order op, combine generico) sotto compile
  D. come B ma mode="reduce-overhead" (CUDA graphs)
Correttezza: allclose vs A in fp32. Tempi: mediana su 30 iter, fwd e fwd+bwd.
"""
import time
import torch

torch.manual_seed(0)
dev = "cuda"
B, T, M_ = 32, 512, 512


def scan_eager(M, f):
    t = f.shape[1]
    stride = 1
    while stride < t:
        M_hi = M[stride:]
        f_new = torch.einsum("tmij,btmj->btmi", M_hi, f[:, :-stride]) + f[:, stride:]
        M = torch.cat([M[:stride], M_hi @ M[:-stride]], dim=0)
        f = torch.cat([f[:, :stride], f_new], dim=1)
        stride *= 2
    return f


from torch._higher_order_ops.associative_scan import associative_scan


def combine(a, b):
    aM, af = a
    bM, bf = b
    return bM @ aM, torch.einsum("mij,bmj->bmi", bM, af) + bf


def scan_hoo(M, f):
    # M: (t, m, 2, 2), f: (b, t, m, 2) -> scan su dim tempo
    fT = f.movedim(1, 0).contiguous()  # (t, b, m, 2)
    _, out = associative_scan(combine, (M, fT), dim=0, combine_mode="generic")
    return out.movedim(0, 1)


def bench(fn, M, f, with_bwd, iters=30):
    for _ in range(5):
        out = fn(M, f)
        if with_bwd:
            out.sum().backward()
        M.grad = f.grad = None
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(M, f)
        if with_bwd:
            out.sum().backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        M.grad = f.grad = None
    return sorted(times)[len(times) // 2] * 1000


def make_inputs():
    r = torch.empty(M_, device=dev).uniform_(0.9, 1.0)
    th = torch.empty(M_, device=dev).uniform_(0.0, 3.14)
    S, dt = r.square(), torch.full_like(r, 0.5)
    A = (S + 1 - 2 * r * torch.cos(th)) / (dt.square() * S)
    row1 = torch.stack([S, -S * dt * A], -1)
    row2 = torch.stack([dt * S, 1 - dt.square() * S * A], -1)
    Mm = torch.stack([row1, row2], -2).unsqueeze(0).expand(T, -1, -1, -1).contiguous()
    f = torch.randn(B, T, M_, 2, device=dev) * 0.1
    return Mm.requires_grad_(), f.requires_grad_()


Mm, f = make_inputs()

with torch.no_grad():
    ref = scan_eager(Mm, f)
    out_hoo = scan_hoo(Mm, f)
    err = (ref - out_hoo).abs().max().item()
    rel = err / ref.abs().max().item()
print(f"correttezza hoo vs eager: max abs err {err:.3e} (rel {rel:.3e})")

variants = {
    "A eager": scan_eager,
    "B compile": torch.compile(scan_eager),
    "C hoo+compile": torch.compile(scan_hoo),
    "D compile+cudagraphs": torch.compile(scan_eager, mode="reduce-overhead"),
}
for name, fn in variants.items():
    try:
        fwd = bench(fn, Mm, f, with_bwd=False)
        fb = bench(fn, Mm, f, with_bwd=True)
        print(f"{name:22s}  fwd {fwd:7.2f} ms   fwd+bwd {fb:7.2f} ms")
    except Exception as e:
        print(f"{name:22s}  ERRORE: {type(e).__name__}: {str(e)[:200]}")
