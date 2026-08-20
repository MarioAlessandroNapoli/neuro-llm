"""Pipeline dati stadio char (D15/D16): txt grezzo -> .bin uint8.

L'unico intervento sul testo: il delimitatore letterale <|endoftext|> diventa il byte
EOT 0x00 (verificato assente nel testo). Nessun tokenizer, nessuna normalizzazione.
Su un'istanza nuova i txt si scaricano dal repo pubblico roneneldan/TinyStories.
"""
import argparse
from pathlib import Path

from src.configs import CHAR_EOT_BYTE, EOT_TOKEN

CHUNK = 64 * 1024 * 1024
DELIM = EOT_TOKEN.encode()
HF_PUBLIC_REPO = "roneneldan/TinyStories"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "valid": "TinyStoriesV2-GPT4-valid.txt"}


def build(src: Path, dst: Path):
    eot = bytes([CHAR_EOT_BYTE])
    n_in = n_out = n_stories = 0
    carry = b""
    with open(src, "rb") as f, open(dst, "wb") as out:
        while True:
            chunk = carry + f.read(CHUNK)
            if not chunk:
                break
            eof = len(chunk) < len(carry) + CHUNK
            # il delimitatore può cavalcare il confine dei chunk: si trattiene una coda
            carry = b"" if eof else chunk[-(len(DELIM) - 1):]
            body = chunk if eof else chunk[: -(len(DELIM) - 1)]
            if eot in body:
                raise SystemExit(f"{src}: byte 0x00 già presente nel testo, EOT ambiguo")
            n_stories += body.count(DELIM)
            replaced = body.replace(DELIM, eot)
            n_in += len(body)
            n_out += len(replaced)
            out.write(replaced)
            if eof:
                break
    print(f"{dst}: {n_out:,} byte ({n_in:,} in ingresso, {n_stories:,} delimitatori)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true",
                        help="scarica i txt dal repo pubblico HF se assenti (istanze nuove)")
    args = parser.parse_args()

    for split, fname in FILES.items():
        src = args.data_dir / fname
        if not src.exists():
            if not args.download:
                raise SystemExit(f"{src} assente: rilancia con --download")
            from huggingface_hub import hf_hub_download

            src = Path(hf_hub_download(HF_PUBLIC_REPO, fname, repo_type="dataset"))
        build(src, args.data_dir / f"{split}_bytes.bin")


if __name__ == "__main__":
    main()
