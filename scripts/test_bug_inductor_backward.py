"""Gradienti dello scan sotto torch.compile su CUDA: eager vs hoo vs verita fp64."""
import torch
from torch._higher_order_ops.associative_scan import associative_scan

torch.manual_seed(0)
dev = "cuda"
B, T, M_ = 4, 512, 64


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


def combine(a, b):
    aM, af = a
    bM, bf = b
    return bM @ aM, torch.einsum("mij,bmj->bmi", bM, af) + bf


def scan_hoo(M, f):
    fT = f.movedim(1, 0).contiguous()
    _, out = associative_scan(combine, (M, fT), dim=0, combine_mode="generic")
    return out.movedim(0, 1)


r = torch.empty(M_, device=dev).uniform_(0.9, 1.0)
th = torch.empty(M_, device=dev).uniform_(0.0, 3.14)
S, dt = r.square(), torch.full_like(r, 0.5)
A = (S + 1 - 2 * r * torch.cos(th)) / (dt.square() * S)
row1 = torch.stack([S, -S * dt * A], -1)
row2 = torch.stack([dt * S, 1 - dt.square() * S * A], -1)
M0 = torch.stack([row1, row2], -2).unsqueeze(0).expand(T, -1, -1, -1).contiguous()
f0 = torch.randn(B, T, M_, 2, device=dev) * 0.1
w = torch.randn(B, T, M_, 2, device=dev)  # pesi fissi per la loss


def grads(fn, dtype, compiled):
    M = M0.to(dtype).requires_grad_()
    f = f0.to(dtype).requires_grad_()
    g = torch.compile(fn) if compiled else fn
    out = g(M, f)
    (out * w.to(dtype)).sum().backward()
    return M.grad.double(), f.grad.double()


gM_ref, gf_ref = grads(scan_eager, torch.float64, compiled=False)
sM, sf = gM_ref.abs().max().item(), gf_ref.abs().max().item()
for name, fn in [("eager fp32 compiled", scan_eager), ("hoo   fp32 compiled", scan_hoo)]:
    gM, gf = grads(fn, torch.float32, compiled=True)
    eM = (gM - gM_ref).abs().max().item() / sM
    ef = (gf - gf_ref).abs().max().item() / sf
    print(f"{name}: rel err gradM {eM:.3e}   gradF {ef:.3e}")
