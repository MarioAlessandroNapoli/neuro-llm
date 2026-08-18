"""Bracci oscillatori della griglia 1a (RESEARCH_LOG D10).

LinOSS-IMEX (arXiv:2410.03943): z[k+1]=z[k]+Δt(−A·x[k]+B·u[k+1]), x[k+1]=x[k]+Δt·z[k+1];
A=ReLU(Â) init U[0,1], Δt=1 fisso. D-LinOSS (arXiv:2505.12171): smorzamento implicito
z[k+1]=(z[k]+Δt(−A·x[k]+B·u[k+1]))/(1+Δt·G); G=ReLU(Ḡ), Δt=σ(Δt̄), A clampata nella
finestra di stabilità; init autovalori nell'anello complesso raggio [0.9, 1], angolo
uniforme. Deviazione dichiarata (D10): scheletro transformer con solo il mixer scambiato,
niente position embedding (l'ordine è nella ricorrenza).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import ModelConfig

PHI = (1 + 5**0.5) / 2
# Periodi centrali delle bande dell'init aurea (D10): Fibonacci, in token.
PHI_PERIODS = (377, 233, 144, 89, 55, 34, 21, 13, 8, 5, 3)
PHI_JITTER = 0.25  # jitter intra-banda: periodo × φ^U[−0.25, 0.25] (metà log-distanza tra bande)
RING_R_MIN, RING_R_MAX = 0.9, 1.0
DT_INIT = 0.5  # Δt̄=0 → σ=0.5; la casualità dell'init sta in (raggio, angolo)


@dataclass
class LinOSSConfig(ModelConfig):
    m: int = 512  # oscillatori per layer = 2·d_model → B,C ≈ i 4d² dell'attention (parità D6)
    damped: bool = False
    phi_init: bool = False


@dataclass
class DLinOSSConfig(LinOSSConfig):
    damped: bool = True


@dataclass
class DLinOSSPhiConfig(DLinOSSConfig):
    phi_init: bool = True


def prefix_scan(M, f):
    """Scan associativo inclusivo della ricorrenza affine s[k] = M·s[k−1] + f[k].

    M: (t, m, 2, 2) — la transizione non dipende dal batch (A, G, Δt sono parametri).
    f: (b, t, m, 2). Op binaria (a₁,a₂)•(b₁,b₂) = (b₁·a₁, b₁·a₂+b₂), log2(t) raddoppi.
    Ritorna gli stati s[k] (b, t, m, 2) con s iniziale nullo.

    Sempre in fp32 (fallback pre-registrato in D10, cablato di default): sotto
    16-mixed l'autocast casterebbe einsum/matmul a fp16 e l'errore si compone sui
    ~9 livelli — misurato ~5% sullo stato a t=512 con |λ|=1 e gradienti NaN col
    GradScaler. B, C e FFN restano in fp16.
    """
    with torch.autocast(device_type=f.device.type, enabled=False):
        return _scan_fp32(M.float(), f.float())


def _scan_fp32(M, f):
    t = f.shape[1]
    stride = 1
    while stride < t:
        M_hi = M[stride:]
        f_new = torch.einsum("tmij,btmj->btmi", M_hi, f[:, :-stride]) + f[:, stride:]
        M = torch.cat([M[:stride], M_hi @ M[:-stride]], dim=0)
        f = torch.cat([f[:, :stride], f_new], dim=1)
        stride *= 2
    return f


def phi_angles(m: int):
    band = torch.arange(m) % len(PHI_PERIODS)
    periods = torch.tensor(PHI_PERIODS, dtype=torch.float32)[band]
    periods = periods * PHI ** torch.empty(m).uniform_(-PHI_JITTER, PHI_JITTER)
    return 2 * math.pi / periods


class OscMixer(nn.Module):
    def __init__(self, d_model: int, m: int, damped: bool, phi_init: bool):
        super().__init__()
        self.damped = damped
        self.B = nn.Linear(d_model, m)
        self.C = nn.Linear(m, d_model)
        if not damped:
            self.A_raw = nn.Parameter(torch.empty(m).uniform_(0.0, 1.0))
            return
        # Inversione esatta autovaloli→(A, G) a Δt=0.5: S=r², G=(1/S−1)/Δt,
        # A=(S+1−2r·cosθ)/(Δt²S); A≥0 garantito da r²+1−2r·cosθ ≥ (r−1)².
        r = torch.empty(m).uniform_(RING_R_MIN, RING_R_MAX)
        theta = phi_angles(m) if phi_init else torch.empty(m).uniform_(0.0, math.pi)
        dt = torch.full((m,), DT_INIT)
        S = r.square()
        self.A_raw = nn.Parameter((S + 1 - 2 * r * torch.cos(theta)) / (dt.square() * S))
        self.G_raw = nn.Parameter((1 / S - 1) / dt)
        self.dt_raw = nn.Parameter(torch.zeros(m))  # σ(0) = DT_INIT

    def state_parameters(self):
        return [self.A_raw] + ([self.G_raw, self.dt_raw] if self.damped else [])

    def forward(self, u):
        t = u.shape[1]
        bu = self.B(u)  # (b, t, m)
        if self.damped:
            dt = torch.sigmoid(self.dt_raw)
            S = 1 / (1 + dt * F.relu(self.G_raw))
            # Finestra di stabilità (autovalori complessi coniugati, |λ|=√S ≤ 1):
            # discriminante < 0 ⇔ A ∈ [(1−√S)²/(Δt²S), (1+√S)²/(Δt²S)]
            sqrt_S = S.sqrt()
            denom = dt.square() * S
            A = F.relu(self.A_raw).clamp((1 - sqrt_S).square() / denom, (1 + sqrt_S).square() / denom)
        else:
            dt = torch.ones_like(self.A_raw)
            S = torch.ones_like(self.A_raw)
            A = F.relu(self.A_raw)
        row1 = torch.stack([S, -S * dt * A], dim=-1)
        row2 = torch.stack([dt * S, 1 - dt.square() * S * A], dim=-1)
        M = torch.stack([row1, row2], dim=-2)  # (m, 2, 2)
        f = torch.stack([dt * S * bu, dt.square() * S * bu], dim=-1)  # (b, t, m, 2)
        states = prefix_scan(M.unsqueeze(0).expand(t, -1, -1, -1), f)
        return self.C(states[..., 1])  # componente x dello stato


class OscBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, m: int, damped: bool, phi_init: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mixer = OscMixer(cfg.d_model, m, damped, phi_init)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x):
        x = x + self.mixer(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class OscLM(nn.Module):
    def __init__(self, cfg: LinOSSConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            OscBlock(cfg, cfg.m, cfg.damped, cfg.phi_init) for _ in range(cfg.n_layer)
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def state_parameters(self):
        return [p for blk in self.blocks for p in blk.mixer.state_parameters()]

    def forward(self, idx):
        x = self.tok(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))
