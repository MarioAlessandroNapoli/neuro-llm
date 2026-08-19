from ..configs import ModelConfig
from .hybrid import Hybrid, HybridAOConfig, HybridAOLPConfig, HybridOAConfig, HybridOALPConfig
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
}


def build_model(arch: str):
    if arch not in ARCHS:
        raise ValueError(f"arch '{arch}' sconosciuta, disponibili: {list(ARCHS)}")
    model_cls, cfg_cls = ARCHS[arch]
    cfg = cfg_cls()
    return model_cls(cfg), cfg
