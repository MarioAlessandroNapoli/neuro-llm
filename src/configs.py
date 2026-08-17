from dataclasses import dataclass

VOCAB_SIZE = 8192
EOT_TOKEN = "<|endoftext|>"

# Parità D6: il ±5% si misura sul backbone (totale meno embedding dei token, legato alla
# testa); riferimento = baseline transformer d256-L8. Il position embedding conta nel backbone.
BASELINE_BACKBONE_PARAMS = 6_449_664
PARITY_TOL = 0.05


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    seq_len: int = 512
    d_model: int = 256
    n_layer: int = 8
    n_head: int = 8
