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
    CHAR_SEQ_LEN, CHAR_VOCAB, D19_BASELINE_BACKBONE_PARAMS, ModelConfig,
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
    # D19: posizioni dei blocchi attention nello stack (interleaved). Vuoto = layout
    # contiguo storico (osc_first/n_osc). La schermatura MQAR è la ragione
    # dell'interleaving: attention SOPRA lo stack non impara il retrieval.
    attn_positions: tuple = ()


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
class D19MixConfig(CharHybHardConfig):
    # D19 (curva di sostituzione, classe 15M): 8 blocchi d_model 384, m=2·d (parità
    # col blocco attention per costruzione), osc = gate appreso (vincitore char),
    # attention interleaved uniformemente, niente pos emb (posizione dagli osc).
    d_model: int = 384
    m: int = 768
    n_osc: int = 8
    parity_ref: int = D19_BASELINE_BACKBONE_PARAMS
    attn_positions: tuple = ()


@dataclass
class D19Mix2Config(D19MixConfig):
    attn_positions: tuple = (3, 7)


@dataclass
class D19Mix4Config(D19MixConfig):
    attn_positions: tuple = (1, 3, 5, 7)


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

        def osc_block(i):
            return OscBlock(cfg, cfg.m, damped=True, phi_init=False,
                            log_polar=cfg.log_polar, ring=ring(i), reset=cfg.reset,
                            no_rotation=cfg.no_rotation,
                            heuristic_reset=cfg.heuristic_reset)

        if cfg.attn_positions:
            self.blocks = nn.ModuleList(
                Block(cfg) if p in cfg.attn_positions else osc_block(p)
                for p in range(cfg.n_layer)
            )
        else:
            osc = [osc_block(i) for i in range(n_osc)]
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

    # --- Generazione incrementale (D17, campagna giudice): stati osc in O(1)/byte, ---
    # --- pila attention ricalcolata sulla sequenza cacheata (costo da transformer). ---
    # Contesto PIENO (nessuna finestra scorrevole: la ricorrenza non può dimenticare
    # il byte uscito; l'estrapolazione D17 giustifica — dichiarato nel log).

    @torch.no_grad()
    def prefill(self, idx):
        """→ (logits ultima posizione (b,V), cache)."""
        if not self.cfg.osc_first or self.pos is not None:
            raise NotImplementedError("prefill: solo ibridi osc-first senza pos emb")
        from .linoss import RESET_KERNEL
        x = self.tok(idx)
        boundary = self.boundary_lut[idx].to(x.dtype) if self.cfg.heuristic_reset else None
        states, bufs = [], []
        osc = [b for b in self.blocks if isinstance(b, OscBlock)]
        for blk in osc:
            u = blk.ln1(x)
            pad = torch.zeros(u.shape[0], max(RESET_KERNEL - u.shape[1], 0), u.shape[2],
                              dtype=u.dtype, device=u.device)
            bufs.append(torch.cat([pad, u[:, -(RESET_KERNEL):]], dim=1))
            out, st = blk.mixer(u, boundary, return_state=True)
            states.append(st)
            x = x + out
            x = x + blk.mlp(blk.ln2(x))
        cache = {"states": states, "bufs": bufs, "y": x}
        return self._attn_logits(cache["y"]), cache

    @torch.no_grad()
    def step(self, next_ids, cache):
        """next_ids (b,) → logits (b,V); aggiorna cache in place."""
        x = self.tok(next_ids)
        boundary_t = (self.boundary_lut[next_ids].to(x.dtype)
                      if self.cfg.heuristic_reset else None)
        osc = [b for b in self.blocks if isinstance(b, OscBlock)]
        for i, blk in enumerate(osc):
            u = blk.ln1(x)
            cache["bufs"][i] = torch.cat([cache["bufs"][i][:, 1:], u[:, None]], dim=1)
            out, cache["states"][i] = blk.mixer.step(
                u, cache["states"][i], cache["bufs"][i], boundary_t)
            x = x + out
            x = x + blk.mlp(blk.ln2(x))
        cache["y"] = torch.cat([cache["y"], x[:, None]], dim=1)
        return self._attn_logits(cache["y"])

    def _attn_logits(self, y):
        for blk in self.blocks:
            if not isinstance(blk, OscBlock):
                y = blk(y)
        return self.head(self.ln_f(y[:, -1]))
