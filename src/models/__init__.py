from ..configs import ModelConfig
from .transformer import Transformer

# Ogni architettura registra (classe modello, classe config): le varianti con parametri
# propri (frequenze, smorzamento, dt) sottoclassano ModelConfig senza toccare le altre.
ARCHS = {
    "transformer": (Transformer, ModelConfig),
}


def build_model(arch: str):
    if arch not in ARCHS:
        raise ValueError(f"arch '{arch}' sconosciuta, disponibili: {list(ARCHS)}")
    model_cls, cfg_cls = ARCHS[arch]
    cfg = cfg_cls()
    return model_cls(cfg), cfg
