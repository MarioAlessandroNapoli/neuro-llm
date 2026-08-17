"""Àncora esterna D7: BPB di stories15M (llama2.c) sul NOSTRO val set.

Condizioni di misura (RESEARCH_LOG § Ancore quantitative):
- contesto 256 (il massimo di stories15M), stream packing identico al suo training
  (llama2.c tinystories.py: BOS per storia, niente EOS, storie concatenate);
- finestre non sovrapposte, ogni token dello stream predetto una volta sola;
- BPB con la convenzione D7: byte UTF-8 del solo testo, BOS esclusi dal conteggio
  token (speculare all'esclusione degli EOT per il nostro tokenizer);
- stories15M è addestrato su TinyStories V1, il nostro val è V2-GPT4: il suo numero
  è un limite superiore (fuori distribuzione per lui) e si dichiara come tale.

Modello e tokenizer sono gli originali di llama2.c (commit pinnato), mai reimplementati.
Esecuzione: uv run --with sentencepiece python -m src.eval.anchor_stories15m
"""
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ..prepare_data import DATA_DIR, SPLITS, iter_stories
from .analysis import bpb, bytes_per_token

LLAMA2C_COMMIT = "350e04fe35433e6d2941dce5a1f53308f87058eb"
ARTIFACTS = {
    "stories15M.pt": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt",
    "llama2c_model.py": f"https://raw.githubusercontent.com/karpathy/llama2.c/{LLAMA2C_COMMIT}/model.py",
    "llama2c_tokenizer.model": f"https://raw.githubusercontent.com/karpathy/llama2.c/{LLAMA2C_COMMIT}/tokenizer.model",
}
CONTEXT = 256
BATCH = 64


def fetch_artifacts() -> dict[str, Path]:
    paths = {}
    for name, url in ARTIFACTS.items():
        path = DATA_DIR / name
        if not path.exists():
            subprocess.run(["curl", "-L", "--fail", "-o", str(path), url], check=True)
        paths[name] = path
    return paths


def load_model(model_py: Path, ckpt_path: Path):
    spec = importlib.util.spec_from_file_location("llama2c_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["llama2c_model"] = mod
    spec.loader.exec_module(mod)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = mod.Transformer(mod.ModelArgs(**ckpt["model_args"]))
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt["model_args"]


def tokenize_stream(tokenizer_model: Path, stories: list[str]) -> tuple[torch.Tensor, int]:
    """Stream come il pretok di llama2.c: encode(story, bos=True, eos=False), concatenato."""
    from sentencepiece import SentencePieceProcessor

    sp = SentencePieceProcessor(model_file=str(tokenizer_model))
    bos = sp.bos_id()
    ids = []
    for story in stories:
        ids.append(bos)
        ids.extend(sp.encode(story))
    return torch.tensor(ids, dtype=torch.long), len(stories)


@torch.no_grad()
def stream_loss(model, stream: torch.Tensor, device: str) -> tuple[float, int]:
    """Mean nats/token su finestre da CONTEXT: blocchi di CONTEXT+1 a passo CONTEXT
    (x=blocco[:-1], y=blocco[1:]), la coda incompleta si scarta e si dichiara."""
    n_blocks = (len(stream) - 1) // CONTEXT
    blocks = stream[: n_blocks * CONTEXT + 1].unfold(0, CONTEXT + 1, CONTEXT)
    total_nats, n_pred = 0.0, 0
    for i in range(0, n_blocks, BATCH):
        chunk = blocks[i : i + BATCH].to(device)
        x, y = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
        logits = model(x, y)
        total_nats += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        n_pred += y.numel()
    dropped = len(stream) - (n_blocks * CONTEXT + 1)
    print(f"finestre: {n_blocks} × {CONTEXT} (coda scartata: {dropped} token)")
    return total_nats / n_pred, n_pred


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    paths = fetch_artifacts()
    val_txt = DATA_DIR / SPLITS["valid"]

    model, model_args = load_model(paths["llama2c_model.py"], paths["stories15M.pt"])
    assert model_args["max_seq_len"] == CONTEXT, model_args
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"stories15M caricato: {n_params/1e6:.1f}M parametri, args {model_args}")

    stories = list(iter_stories(val_txt))
    stream, n_stories = tokenize_stream(paths["llama2c_tokenizer.model"], stories)
    text_bytes = sum(len(s.encode()) for s in stories)
    bpt_llama = text_bytes / (len(stream) - n_stories)
    print(f"val: {n_stories} storie, {len(stream)/1e6:.2f}M token Llama-2, "
          f"{bpt_llama:.4f} byte/token (BOS esclusi)")

    loss, n_pred = stream_loss(model, stream, device)
    print(f"\nstories15M sul nostro val V2 @ contesto {CONTEXT}:")
    print(f"  loss: {loss:.4f} nats/token ({n_pred/1e6:.2f}M token predetti)")
    print(f"  BPB:  {bpb(loss, bpt_llama):.4f}")

    bpt_ours = bytes_per_token(val_txt, DATA_DIR / "valid.bin")
    print(f"\npromemoria nostro apparato: {bpt_ours:.4f} byte/token → "
          f"BPB nostro = val_loss / ln2 / {bpt_ours:.4f}")


if __name__ == "__main__":
    main()
