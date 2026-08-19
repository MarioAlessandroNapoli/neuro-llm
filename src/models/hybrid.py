"""Bracci ibridi della griglia 1a (RESEARCH_LOG D10, asse 5 — gerarchia).

4+4 layer: mixer D-LinOSS (init default: le combinazioni con φ sono materia della 1b)
+ attention della baseline. `hyb-oa` = oscillatori sotto, attention sopra; `hyb-ao`
l'inverso. Position embedding attivo (serve all'attention).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..configs import ModelConfig
from .linoss import OscBlock
from .transformer import Block


@dataclass
class HybridOAConfig(ModelConfig):
    m: int = 512
    osc_first: bool = True
    log_polar: bool = False  # 1a: parametrizzazione classica (A,G) — congelata


@dataclass
class HybridAOConfig(HybridOAConfig):
    osc_first: bool = False


@dataclass
class HybridOALPConfig(HybridOAConfig):
    log_polar: bool = True  # ricetta 1b (D12): log-polare come apparato


@dataclass
class HybridAOLPConfig(HybridAOConfig):
    log_polar: bool = True


class Hybrid(nn.Module):
    def __init__(self, cfg: HybridOAConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        half = cfg.n_layer // 2
        osc = [OscBlock(cfg, cfg.m, damped=True, phi_init=False, log_polar=cfg.log_polar) for _ in range(half)]
        attn = [Block(cfg) for _ in range(cfg.n_layer - half)]
        self.blocks = nn.ModuleList(osc + attn if cfg.osc_first else attn + osc)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def state_parameters(self):
        return [
            p for blk in self.blocks if isinstance(blk, OscBlock)
            for p in blk.mixer.state_parameters()
        ]

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))
