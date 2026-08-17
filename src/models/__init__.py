from ..configs import ModelConfig
from .transformer import Transformer

ARCHS = {
    "transformer": Transformer,
}


def build_model(arch: str, cfg: ModelConfig):
    if arch not in ARCHS:
        raise ValueError(f"arch '{arch}' sconosciuta, disponibili: {list(ARCHS)}")
    return ARCHS[arch](cfg)
