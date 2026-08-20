from ..configs import CharTransformerConfig, CharTransformerNoPosConfig, ModelConfig
from .hybrid import (
    CharHybConfig,
    CharHybHardConfig,
    CharHybHeuConfig,
    CharHybPhaseConfig,
    CharHybTsConfig,
    CharOsc0Config,
    Hybrid,
    HybridAOConfig,
    HybridAOLPConfig,
    HybridOAConfig,
    HybridOALPConfig,
)
from .linoss import (
    DLinOSSConfig,
    DLinOSSLPConfig,
    DLinOSSLPInitConfig,
    DLinOSSPhiConfig,
    LinOSSConfig,
    OscLM,
)
from .transformer import Transformer
from .wrnn import WRNNConfig, WRNNLM

# Ogni architettura registra (classe modello, classe config): le varianti con parametri
# propri (frequenze, smorzamento, dt) sottoclassano ModelConfig senza toccare le altre.
ARCHS = {
    "transformer": (Transformer, ModelConfig),
    "linoss": (OscLM, LinOSSConfig),
    "dlinoss": (OscLM, DLinOSSConfig),
    "dlinoss-phi": (OscLM, DLinOSSPhiConfig),
    "dlinoss-lp": (OscLM, DLinOSSLPConfig),
    "dlinoss-lp-init": (OscLM, DLinOSSLPInitConfig),
    "hyb-oa": (Hybrid, HybridOAConfig),
    "hyb-ao": (Hybrid, HybridAOConfig),
    "hyb-oa-lp": (Hybrid, HybridOALPConfig),
    "hyb-ao-lp": (Hybrid, HybridAOLPConfig),
    "wrnn": (WRNNLM, WRNNConfig),
    # Griglia char, fase A (D15/D16): byte grezzi, vocab 256, seq 2048
    "char-transformer": (Transformer, CharTransformerConfig),
    "char-transformer-nopos": (Transformer, CharTransformerNoPosConfig),
    "char-osc0": (Hybrid, CharOsc0Config),
    # Griglia char, fase B (D16): reset-su-confini sull'ibrido oa log-polare
    "char-hyb": (Hybrid, CharHybConfig),
    "char-hyb-hard": (Hybrid, CharHybHardConfig),
    "char-hyb-heu": (Hybrid, CharHybHeuConfig),
    "char-hyb-phase": (Hybrid, CharHybPhaseConfig),
    "char-hyb-ts": (Hybrid, CharHybTsConfig),
}


def build_model(arch: str):
    if arch not in ARCHS:
        raise ValueError(f"arch '{arch}' sconosciuta, disponibili: {list(ARCHS)}")
    model_cls, cfg_cls = ARCHS[arch]
    cfg = cfg_cls()
    return model_cls(cfg), cfg
