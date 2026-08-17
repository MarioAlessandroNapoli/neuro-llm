from dataclasses import dataclass

VOCAB_SIZE = 8192
EOT_TOKEN = "<|endoftext|>"


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    seq_len: int = 512
    d_model: int = 256
    n_layer: int = 8
    n_head: int = 8
