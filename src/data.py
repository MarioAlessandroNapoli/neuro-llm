from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenWindowDataset(Dataset):
    """Finestre contigue di token da un .bin memmappato (uint16 BPE; uint8 byte, D15)."""

    def __init__(self, bin_path: Path, seq_len: int, dtype=np.uint16):
        self.tokens = np.memmap(bin_path, dtype=dtype, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, i):
        start = i * self.seq_len
        chunk = torch.from_numpy(self.tokens[start : start + self.seq_len + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]
