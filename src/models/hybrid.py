"""Bracci ibridi della griglia 1a (RESEARCH_LOG D10, asse 5 — gerarchia).

4+4 layer: mixer D-LinOSS (init default: le combinazioni con φ sono materia della 1b)
+ attention della baseline. `hyb-oa` = oscillatori sotto, attention sopra; `hyb-ao`
l'inverso. Position embedding attivo (serve all'attention).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..configs import CHAR_BASELINE_BACKBONE_PARAMS, CHAR_PARITY_TOL, CHAR_SEQ_LEN, CHAR_VOCAB, ModelConfig
from .linoss import OscBlock
from .transformer import Block


@dataclass
class HybridOAConfig(ModelConfig):
    m: int = 512
    osc_first: bool = True
    log_polar: bool = False  # 1a: parametrizzazione classica (A,G) — congelata
    n_osc: int = -1  # -1 = n_layer//2 (griglie 1a/1b); la griglia char (D16) lo fissa
    reset: bool = False
    no_rotation: bool = False


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
        osc = [OscBlock(cfg, cfg.m, damped=True, phi_init=False, log_polar=cfg.log_polar,
                        reset=cfg.reset, no_rotation=cfg.no_rotation) for _ in range(n_osc)]
        attn = [Block(cfg) for _ in range(cfg.n_layer - n_osc)]
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
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))
