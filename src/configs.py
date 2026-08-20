from dataclasses import dataclass

VOCAB_SIZE = 8192
EOT_TOKEN = "<|endoftext|>"

# Parità D6: il ±5% si misura sul backbone (totale meno embedding dei token, legato alla
# testa); riferimento = baseline transformer d256-L8. Il position embedding conta nel backbone.
BASELINE_BACKBONE_PARAMS = 6_449_664
PARITY_TOL = 0.05

# Stadio char (D15/D16): byte grezzi, EOT = byte 0x00 (mai presente nel testo UTF-8),
# parità lasca ±10% sul backbone della char-baseline (pos 2048 incluso; seq_len dal
# dataset: 2048 byte contiene intero il 98,6% delle storie e il protocollo giudice D7).
CHAR_VOCAB = 256
CHAR_SEQ_LEN = 2048
CHAR_EOT_BYTE = 0
CHAR_BASELINE_BACKBONE_PARAMS = 6_842_880
CHAR_PARITY_TOL = 0.10


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    seq_len: int = 512
    d_model: int = 256
    n_layer: int = 8
    n_head: int = 8
    byte_level: bool = False
    use_pos: bool = True
    parity_ref: int = BASELINE_BACKBONE_PARAMS
    parity_tol: float = PARITY_TOL


@dataclass
class CharTransformerConfig(ModelConfig):
    vocab_size: int = CHAR_VOCAB
    seq_len: int = CHAR_SEQ_LEN
    byte_level: bool = True
    parity_ref: int = CHAR_BASELINE_BACKBONE_PARAMS
    parity_tol: float = CHAR_PARITY_TOL


@dataclass
class CharTransformerNoPosConfig(CharTransformerConfig):
    use_pos: bool = False
