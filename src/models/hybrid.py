"""Bracci ibridi della griglia 1a (RESEARCH_LOG D10, asse 5 — gerarchia).

4+4 layer: mixer D-LinOSS (init default: le combinazioni con φ sono materia della 1b)
+ attention della baseline. `hyb-oa` = oscillatori sotto, attention sopra; `hyb-ao`
l'inverso. Position embedding attivo (serve all'attention).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..configs import (
    CHAR_BASELINE_BACKBONE_PARAMS, CHAR_BOUNDARY_BYTES, CHAR_PARITY_TOL,
    CHAR_SEQ_LEN, CHAR_VOCAB, ModelConfig,
)
from .linoss import RING_R_MAX, RING_R_MIN, OscBlock
from .transformer import Block

# Braccio timescale (D17 C1, Harmonic-style): init dei raggi per layer oscillatorio
# su bande di τ scalate geometricamente — layer l: τ ∈ [TS_TAU_MIN·8^l, TS_TAU_MIN·8^(l+1)]
# byte, cioè (2-16, 16-128, 128-1024, 1024-8192). Solo init: il gradiente resta libero.
TS_TAU_MIN = 2.0
TS_TAU_FACTOR = 8.0


@dataclass
class HybridOAConfig(ModelConfig):
    m: int = 512
    osc_first: bool = True
    log_polar: bool = False  # 1a: parametrizzazione classica (A,G) — congelata
    n_osc: int = -1  # -1 = n_layer//2 (griglie 1a/1b); la griglia char (D16) lo fissa
    reset: bool = False
    no_rotation: bool = False
    heuristic_reset: bool = False
    ts_hierarchy: bool = False


@dataclass
class HybridAOConfig(HybridOAConfig):
    osc_first: bool = False


@dataclass
class HybridOALPConfig(HybridOAConfig):
    log_polar: bool = True  # ricetta 1b (D12): log-polare come apparato


@dataclass
class HybridAOLPConfig(HybridAOConfig):
    log_polar: bool = True


@dataclass
class CharHybConfig(HybridOALPConfig):
    # Fase B griglia char (D16): ibrido oa log-polare su byte, senza pos emb
    # (l'ordine lo danno gli oscillatori). Controllo LTI; i bracci reset sottoclassano.
    vocab_size: int = CHAR_VOCAB
    seq_len: int = CHAR_SEQ_LEN
    byte_level: bool = True
    use_pos: bool = False
    parity_ref: int = CHAR_BASELINE_BACKBONE_PARAMS
    parity_tol: float = CHAR_PARITY_TOL
    reset: bool = False
    no_rotation: bool = False


@dataclass
class CharHybHardConfig(CharHybConfig):
    reset: bool = True
    no_rotation: bool = True  # θ≡0: reset sì, oscillazione no → chunking puro


@dataclass
class CharHybPhaseConfig(CharHybConfig):
    reset: bool = True  # phase reset: la dinamica oscillatoria continua tra i confini


@dataclass
class CharHybHeuConfig(CharHybConfig):
    # C1 (D17): reset cablato sui byte-confine letterali, θ≡0 — zero parametri di gate.
    # Se hard-appreso non batte questo, il gate è un rilevatore di spazi (SOMBRERO).
    no_rotation: bool = True
    heuristic_reset: bool = True


@dataclass
class CharHybTsConfig(CharHybConfig):
    # C1 (D17): gerarchia di sole timescale senza reset (baseline Harmonic-style) —
    # lti con init dei ν per layer su bande τ scalate 8×.
    ts_hierarchy: bool = True


@dataclass
class CharOsc0Config(HybridOALPConfig):
    # Fase A griglia char (D16): banco oscillatorio al layer 0 come codice di posizione,
    # 7 layer di attention SENZA position embedding sopra.
    vocab_size: int = CHAR_VOCAB
    seq_len: int = CHAR_SEQ_LEN
    byte_level: bool = True
    use_pos: bool = False
    n_osc: int = 1
    parity_ref: int = CHAR_BASELINE_BACKBONE_PARAMS
    parity_tol: float = CHAR_PARITY_TOL


class Hybrid(nn.Module):
    def __init__(self, cfg: HybridOAConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model) if cfg.use_pos else None
        n_osc = cfg.n_layer // 2 if cfg.n_osc == -1 else cfg.n_osc

        def ring(i):
            if not cfg.ts_hierarchy:
                return (RING_R_MIN, RING_R_MAX)
            lo, hi = TS_TAU_MIN * TS_TAU_FACTOR**i, TS_TAU_MIN * TS_TAU_FACTOR**(i + 1)
            return (math.exp(-1 / lo), math.exp(-1 / hi))

        osc = [OscBlock(cfg, cfg.m, damped=True, phi_init=False, log_polar=cfg.log_polar,
                        ring=ring(i), reset=cfg.reset, no_rotation=cfg.no_rotation,
                        heuristic_reset=cfg.heuristic_reset) for i in range(n_osc)]
        attn = [Block(cfg) for _ in range(cfg.n_layer - n_osc)]
        self.blocks = nn.ModuleList(osc + attn if cfg.osc_first else attn + osc)
        if cfg.heuristic_reset:
            lut = torch.zeros(cfg.vocab_size, dtype=torch.bool)
            lut[list(CHAR_BOUNDARY_BYTES)] = True
            self.register_buffer("boundary_lut", lut, persistent=False)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def state_parameters(self):
        return [
            p for blk in self.blocks if isinstance(blk, OscBlock)
            for p in blk.mixer.state_parameters()
        ]

    def forward(self, idx):
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))
        boundary = self.boundary_lut[idx].to(x.dtype) if self.cfg.heuristic_reset else None
        for blk in self.blocks:
            x = blk(x, boundary) if isinstance(blk, OscBlock) else blk(x)
        return self.head(self.ln_f(x))
