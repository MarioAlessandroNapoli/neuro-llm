"""Bracci oscillatori della griglia 1a (RESEARCH_LOG D10).

LinOSS-IMEX (arXiv:2410.03943): z[k+1]=z[k]+Δt(−A·x[k]+B·u[k+1]), x[k+1]=x[k]+Δt·z[k+1];
A=ReLU(Â) init U[0,1], Δt=1 fisso. D-LinOSS (arXiv:2505.12171): smorzamento implicito
z[k+1]=(z[k]+Δt(−A·x[k]+B·u[k+1]))/(1+Δt·G); G=ReLU(Ḡ), Δt=σ(Δt̄), A clampata nella
finestra di stabilità; init autovalori nell'anello complesso raggio [0.9, 1], angolo
uniforme. Deviazione dichiarata (D10): scheletro transformer con solo il mixer scambiato,
niente position embedding (l'ordine è nella ricorrenza).
"""

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import ModelConfig

# Implementazione dello scan: "eager" (loop log2(t) con cat, validata in griglia 1a) o
# "hoo" (torch.associative_scan, ~4,5× più veloce sotto compile — bench fase 0 su 3060:
# fwd+bwd 1084→240 ms a shape reali; accuratezza vs fp64 sequenziale identica, 2,7e-2
# relativo entrambe). Nessun fallback: valore ignoto = errore.
SCAN_IMPL = os.environ.get("NEURO_SCAN", "eager")
if SCAN_IMPL not in ("eager", "hoo"):
    raise ValueError(f"NEURO_SCAN='{SCAN_IMPL}' sconosciuto: usare 'eager' o 'hoo'")
if SCAN_IMPL == "hoo":
    from torch._higher_order_ops.associative_scan import associative_scan

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
    log_polar: bool = False
    ring_r_min: float = RING_R_MIN
    ring_r_max: float = RING_R_MAX


@dataclass
class DLinOSSConfig(LinOSSConfig):
    damped: bool = True


@dataclass
class DLinOSSPhiConfig(DLinOSSConfig):
    phi_init: bool = True


@dataclass
class DLinOSSLPConfig(DLinOSSConfig):
    log_polar: bool = True


@dataclass
class DLinOSSLPInitConfig(DLinOSSLPConfig):
    # Init post-autopsia (D12, fase 1e): partire dove il training comunque converge
    # (r mediana 0,74-0,90 nei checkpoint 1a) elimina il transito vicino al bordo.
    ring_r_min: float = 0.7
    ring_r_max: float = 0.9


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
        if SCAN_IMPL == "hoo":
            return _scan_hoo_fp32(M.float(), f.float())
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


def _hoo_combine(a, b):
    # Prodotto 2×2 in aritmetica esplicita, SOLO op elementwise: con matmul/einsum nel
    # combine il backward generato da Inductor è rotto (gradienti ~100% errati, solo
    # compile+CUDA — misurato in fase 0; il modo "pointwise" produce NaN). La forma
    # elementwise ha gradienti corretti (2e-4 vs fp64) e fonde in pochi kernel:
    # scan 1067→8 ms fwd+bwd a shape reali su 3060.
    a11, a12, a21, a22, af1, af2 = a
    b11, b12, b21, b22, bf1, bf2 = b
    return (
        b11 * a11 + b12 * a21,
        b11 * a12 + b12 * a22,
        b21 * a11 + b22 * a21,
        b21 * a12 + b22 * a22,
        b11 * af1 + b12 * af2 + bf1,
        b21 * af1 + b22 * af2 + bf2,
    )


def prefix_scan_gated(M, f, g):
    """Scan della ricorrenza col reset-su-confini (D15): s_k = g_k·M·s_{k−1} + f_k,
    g ∈ [0,1] per (batch, passo, canale) — g→0 azzera la storia al confine.

    M (t, m, 2, 2) resta condivisa sul batch; il gate scala l'intero blocco 2×2, quindi
    nel path hoo si piega nelle 6 componenti elementwise (che diventano (t, b, m)) senza
    toccare il combine. L'eager batched è l'oracolo di correttezza: doubling con la
    transizione materializzata per batch — memoria pesante, solo shape piccole/smoke."""
    with torch.autocast(device_type=f.device.type, enabled=False):
        M, f, g = M.float(), f.float(), g.float()
        if SCAN_IMPL == "hoo":
            gT = g.movedim(1, 0)  # (t, b, m)
            fT = f.movedim(1, 0)
            xs = (gT * M[:, None, :, 0, 0], gT * M[:, None, :, 0, 1],
                  gT * M[:, None, :, 1, 0], gT * M[:, None, :, 1, 1],
                  fT[..., 0].contiguous(), fT[..., 1].contiguous())
            out = associative_scan(_hoo_combine, xs, dim=0, combine_mode="generic")
            return torch.stack([out[4], out[5]], dim=-1).movedim(0, 1)
        GM = g[..., None, None] * M[None]  # (b, t, m, 2, 2)
        t = f.shape[1]
        stride = 1
        while stride < t:
            M_hi = GM[:, stride:]
            f_new = torch.einsum("btmij,btmj->btmi", M_hi, f[:, :-stride]) + f[:, stride:]
            GM = torch.cat([GM[:, :stride], M_hi @ GM[:, :-stride]], dim=1)
            f = torch.cat([f[:, :stride], f_new], dim=1)
            stride *= 2
        return f


def _scan_hoo_fp32(M, f):
    fT = f.movedim(1, 0)  # (t, b, m, 2): stessa dim di scan di M
    m11, m12 = M[:, :, 0, 0].unsqueeze(1), M[:, :, 0, 1].unsqueeze(1)
    m21, m22 = M[:, :, 1, 0].unsqueeze(1), M[:, :, 1, 1].unsqueeze(1)
    xs = (m11, m12, m21, m22, fT[..., 0].contiguous(), fT[..., 1].contiguous())
    out = associative_scan(_hoo_combine, xs, dim=0, combine_mode="generic")
    return torch.stack([out[4], out[5]], dim=-1).movedim(0, 1)


def phi_angles(m: int):
    band = torch.arange(m) % len(PHI_PERIODS)
    periods = torch.tensor(PHI_PERIODS, dtype=torch.float32)[band]
    periods = periods * PHI ** torch.empty(m).uniform_(-PHI_JITTER, PHI_JITTER)
    return 2 * math.pi / periods


RESET_KERNEL = 7
RESET_GROUPS = 64
RESET_BIAS_INIT = -4.0  # σ(−4) ≈ 0,018: a init niente reset → il braccio è quasi-LTI


class OscMixer(nn.Module):
    def __init__(
        self, d_model: int, m: int, damped: bool, phi_init: bool,
        log_polar: bool = False, ring: tuple = (RING_R_MIN, RING_R_MAX),
        reset: bool = False, no_rotation: bool = False, heuristic_reset: bool = False,
    ):
        super().__init__()
        if reset and heuristic_reset:
            raise ValueError("reset appreso e heuristic_reset sono mutuamente esclusivi")
        self.damped = damped
        self.log_polar = log_polar
        self.no_rotation = no_rotation
        self.heuristic_reset = heuristic_reset
        self.B = nn.Linear(d_model, m)
        self.C = nn.Linear(m, d_model)
        self.gate_conv = None
        if reset:
            # Confini appresi dai byte (D16 fase B): conv causale k=7 → G gruppi di
            # canali; b_t = σ(·) prob. di confine, moltiplicatore dello stato = 1−b_t.
            if m % RESET_GROUPS:
                raise ValueError(f"m={m} non divisibile nei {RESET_GROUPS} gruppi del reset")
            self.gate_conv = nn.Conv1d(d_model, RESET_GROUPS, RESET_KERNEL,
                                       padding=RESET_KERNEL - 1)
            nn.init.constant_(self.gate_conv.bias, RESET_BIAS_INIT)
        if log_polar:
            # Parametrizzazione log-polare (LRU, arXiv:2303.06349): r = exp(−exp(ν)) < 1
            # per costruzione, θ = π·σ(θ̄) ∈ (0, π) — niente clamp, update moltiplicativi
            # ben condizionati al bordo. Δt fisso a DT_INIT; (A, G) derivati nel forward
            # con la stessa inversione dell'init.
            r = torch.empty(m).uniform_(*ring)
            theta = phi_angles(m) if phi_init else torch.empty(m).uniform_(0.0, math.pi)
            self.nu_raw = nn.Parameter((-r.log()).log())
            self.theta_raw = nn.Parameter((theta / (math.pi - theta)).log())
            return
        if not damped:
            self.A_raw = nn.Parameter(torch.empty(m).uniform_(0.0, 1.0))
            return
        # Inversione esatta autovaloli→(A, G) a Δt=0.5: S=r², G=(1/S−1)/Δt,
        # A=(S+1−2r·cosθ)/(Δt²S); A≥0 garantito da r²+1−2r·cosθ ≥ (r−1)².
        r = torch.empty(m).uniform_(*ring)
        theta = phi_angles(m) if phi_init else torch.empty(m).uniform_(0.0, math.pi)
        dt = torch.full((m,), DT_INIT)
        S = r.square()
        self.A_raw = nn.Parameter((S + 1 - 2 * r * torch.cos(theta)) / (dt.square() * S))
        self.G_raw = nn.Parameter((1 / S - 1) / dt)
        self.dt_raw = nn.Parameter(torch.zeros(m))  # σ(0) = DT_INIT

    def state_parameters(self):
        if self.log_polar:
            return [self.nu_raw, self.theta_raw]
        return [self.A_raw] + ([self.G_raw, self.dt_raw] if self.damped else [])

    def forward(self, u, boundary=None):
        t = u.shape[1]
        bu = self.B(u)  # (b, t, m)
        if self.log_polar:
            r = torch.exp(-self.nu_raw.exp())
            theta = math.pi * torch.sigmoid(self.theta_raw)
            if self.no_rotation:
                # Braccio hard-reset (D16 fase B): θ≡0 — autovalore reale doppio r,
                # zero oscillazione; theta_raw resta parametro (parità) ma inerte.
                theta = torch.zeros_like(theta)
            S = r.square()
            dt = torch.full_like(r, DT_INIT)
            # λ = r·e^{±iθ} esatto: dt²SA = S+1−2r·cosθ ⇒ traccia 2r·cosθ, det S = r²
            A = (S + 1 - 2 * r * torch.cos(theta)) / (dt.square() * S)
        elif self.damped:
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
        M_t = M.unsqueeze(0).expand(t, -1, -1, -1)
        if self.heuristic_reset:
            # Braccio C1 (D17): reset cablato sui byte-confine — g=0 dove il byte
            # corrente è un confine (spazi/punteggiatura/EOT), nessun parametro.
            if boundary is None:
                raise ValueError("heuristic_reset richiede la maschera boundary dal modello")
            g = (1 - boundary)[..., None].expand(-1, -1, f.shape[2])
            states = prefix_scan_gated(M_t, f, g)
        elif self.gate_conv is not None:
            boundary = torch.sigmoid(self.gate_conv(u.transpose(1, 2))[..., :t].transpose(1, 2))
            g = (1 - boundary).repeat_interleave(f.shape[2] // RESET_GROUPS, dim=-1)
            states = prefix_scan_gated(M_t, f, g)
        else:
            states = prefix_scan(M_t, f)
        return self.C(states[..., 1])  # componente x dello stato


class OscBlock(nn.Module):
    def __init__(
        self, cfg: ModelConfig, m: int, damped: bool, phi_init: bool,
        log_polar: bool = False, ring: tuple = (RING_R_MIN, RING_R_MAX),
        reset: bool = False, no_rotation: bool = False, heuristic_reset: bool = False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mixer = OscMixer(cfg.d_model, m, damped, phi_init, log_polar, ring,
                              reset, no_rotation, heuristic_reset)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x, boundary=None):
        x = x + self.mixer(self.ln1(x), boundary)
        return x + self.mlp(self.ln2(x))


class OscLM(nn.Module):
    def __init__(self, cfg: LinOSSConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            OscBlock(cfg, cfg.m, cfg.damped, cfg.phi_init, cfg.log_polar,
                     (cfg.ring_r_min, cfg.ring_r_max))
            for _ in range(cfg.n_layer)
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
