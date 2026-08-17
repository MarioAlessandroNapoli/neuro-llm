"""One-shot locale: scarica TinyStories V2, addestra il BPE, tokenizza in .bin uint16.

Con --hub-repo pusha tokenizer e bin su un dataset repo HF privato, così Kaggle
li scarica pronti senza mai ritokenizzare.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from .configs import EOT_TOKEN, VOCAB_SIZE

DATA_DIR = Path(__file__).parent.parent / "data"
BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
SPLITS = {"train": "TinyStoriesV2-GPT4-train.txt", "valid": "TinyStoriesV2-GPT4-valid.txt"}
ENCODE_BATCH = 10_000


def iter_stories(txt_path: Path):
    story_lines = []
    with open(txt_path) as f:
        for line in f:
            if line.strip() == EOT_TOKEN:
                yield "".join(story_lines).strip()
                story_lines = []
            else:
                story_lines.append(line)
    if story_lines:
        yield "".join(story_lines).strip()


def train_tokenizer(train_txt: Path, out_path: Path) -> Tokenizer:
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevel()
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=[EOT_TOKEN])
    tok.train_from_iterator(iter_stories(train_txt), trainer)
    tok.save(str(out_path))
    return tok


def encode_split(tok: Tokenizer, txt_path: Path, bin_path: Path):
    eot_id = tok.token_to_id(EOT_TOKEN)
    n_tokens = 0
    batch = []
    with open(bin_path, "wb") as out:

        def flush():
            nonlocal n_tokens
            for enc in tok.encode_batch(batch):
                ids = np.array(enc.ids + [eot_id], dtype=np.uint16)
                out.write(ids.tobytes())
                n_tokens += len(ids)
            batch.clear()

        for story in iter_stories(txt_path):
            batch.append(story)
            if len(batch) >= ENCODE_BATCH:
                flush()
        flush()
    print(f"{bin_path.name}: {n_tokens/1e6:.1f}M token")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true", help="usa i txt locali così come sono")
    parser.add_argument("--hub-repo", help="dataset repo HF privato dove pushare tokenizer e bin")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    if not args.no_download:
        for fname in SPLITS.values():
            subprocess.run(
                ["curl", "-L", "-C", "-", "-o", str(DATA_DIR / fname), f"{BASE_URL}/{fname}"],
                check=True,
            )

    tokenizer_path = DATA_DIR / "tokenizer.json"
    if tokenizer_path.exists():
        tok = Tokenizer.from_file(str(tokenizer_path))
        print(f"tokenizer esistente riusato: {tokenizer_path}")
    else:
        tok = train_tokenizer(DATA_DIR / SPLITS["train"], tokenizer_path)
        print(f"tokenizer addestrato: vocab {tok.get_vocab_size()}")

    for split, fname in SPLITS.items():
        encode_split(tok, DATA_DIR / fname, DATA_DIR / f"{split}.bin")

    if args.hub_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.hub_repo, repo_type="dataset", private=True, exist_ok=True)
        for name in ["tokenizer.json", "train.bin", "valid.bin"]:
            api.upload_file(
                path_or_fileobj=DATA_DIR / name,
                path_in_repo=name,
                repo_id=args.hub_repo,
                repo_type="dataset",
            )
        print(f"pushato su hf.co/datasets/{args.hub_repo}")


if __name__ == "__main__":
    main()
