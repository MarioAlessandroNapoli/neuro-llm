"""Braccio wRNN della griglia 1a (RESEARCH_LOG D10, onde viaggianti vs oscillatori).

Wave-RNN (arXiv:2309.08045): campo circolare h di N celle (c canali × N/c), ricorrenza
h[t] = ReLU(u★h[t−1] + V·x[t] + b) con ★ convoluzione circolare per canale, kernel k=3
init shift-matrix (ν=1), V init a iniezione puntuale — ricette del paper. Deviazioni
dichiarate (D10): impilata nello scheletro transformer a 8 layer (il paper è mono-layer);
niente position embedding (braccio puro). Ricorrenza non lineare → niente scan, loop
sequenziale: lo smoke di velocità su T4 condiziona il calendario dei suoi seed.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import ModelConfig

KERNEL_SIZE = 3
SHIFT_NU = 1  # velocità dell'onda all'init: h_i ← h_{i−1}


@dataclass
class WRNNConfig(ModelConfig):
    field: int = 512  # N celle del campo → V, W_out ≈ i 4d² dell'attention (parità D6)
    channels: int = 16  # anelli indipendenti da field/channels celle


class WaveMixer(nn.Module):
    def __init__(self, d_model: int, field: int, channels: int):
        super().__init__()
        self.channels = channels
        ring = field // channels
        # V a iniezione puntuale: l'input entra in una sola cella per anello (paper);
        # il resto parte a zero ma resta apprendibile.
        bound = 1 / math.sqrt(d_model)
        V = torch.zeros(field, d_model)
        # Slice (view), NON advanced indexing: V[tensor] restituirebbe una copia e
        # l'init andrebbe perso — campo a zero + ReLU'(0)=0 = mixer morto per sempre.
        V[0:field:ring].uniform_(-bound, bound)
        self.V = nn.Parameter(V)
        self.bias = nn.Parameter(torch.zeros(field))
        # Kernel [1, 0, 0]: con pad circolare (1,1) dà out[i] = h[i−1], lo shift ν=1.
        kernel = torch.zeros(channels, 1, KERNEL_SIZE)
        kernel[:, 0, KERNEL_SIZE // 2 - SHIFT_NU] = 1.0
        self.kernel = nn.Parameter(kernel)
        self.out = nn.Linear(field, d_model)

    def state_parameters(self):
        # Il kernel d'onda governa la dinamica del campo: parametro di stato (D6/D10),
        # mai weight decay — come A, G, Δt degli oscillatori.
        return [self.kernel]

    # Deviazione dichiarata dalla config standard: Dynamo srotolerebbe le 512
    # iterazioni in un grafo da ~30k nodi per layer (compilazione da ore su Kaggle);
    # il mixer resta eager, LN/FFN del blocco compilano normalmente.
    @torch.compiler.disable
    def forward(self, u):
        b, t, _ = u.shape
        inject = u @ self.V.T + self.bias  # (b, t, N) — fuori dal loop
        h = torch.zeros(b, self.V.shape[0], device=u.device, dtype=inject.dtype)
        states = []
        for k in range(t):
            field = h.view(b, self.channels, -1)
            prop = F.conv1d(F.pad(field, (1, 1), mode="circular"), self.kernel, groups=self.channels)
            h = F.relu(prop.reshape(b, -1) + inject[:, k])
            states.append(h)
        return self.out(torch.stack(states, dim=1))


class WRNNBlock(nn.Module):
    def __init__(self, cfg: WRNNConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mixer = WaveMixer(cfg.d_model, cfg.field, cfg.channels)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x):
        x = x + self.mixer(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class WRNNLM(nn.Module):
    def __init__(self, cfg: WRNNConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(WRNNBlock(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def state_parameters(self):
        return [p for blk in self.blocks for p in blk.mixer.state_parameters()]

    def forward(self, idx):
        x = self.tok(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))
