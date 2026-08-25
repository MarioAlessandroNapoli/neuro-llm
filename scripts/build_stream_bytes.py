"""Pipeline dati D19: corpus HF in streaming -> .bin uint8 (train/valid).

Generalizza build_bytes.py (TinyStories) a qualunque corpus testuale su HF:
streama documenti, li separa col byte EOT 0x00, scrive fino al budget richiesto.
Fail-loud: se il campo testo non esiste nel primo record, si ferma stampando lo
schema reale (zero assunzioni sui dati). I byte 0x00 nel testo (possibili nel
web crawl, a differenza di TinyStories) vengono rimossi e contati.

Dataset previsti (D19): nvidia/Nemotron-CC-v2 (gated, richiede accesso HF
accettato dall'utente) con filtro qualità; fallback HuggingFaceFW/fineweb-edu.

Uso:
  python -m scripts.build_stream_bytes --dataset HuggingFaceFW/fineweb-edu \
      --config sample-10BT --train-bytes 300_000_000 --valid-bytes 5_000_000 \
      --out-prefix data/fineweb
"""
import argparse
from pathlib import Path

from src.configs import CHAR_EOT_BYTE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--quality-field", default=None,
                        help="campo di qualità da filtrare (es. per Nemotron)")
    parser.add_argument("--quality-keep", nargs="+", default=None,
                        help="valori ammessi del campo qualità")
    parser.add_argument("--train-bytes", type=int, required=True)
    parser.add_argument("--valid-bytes", type=int, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    if (args.quality_field is None) != (args.quality_keep is None):
        raise SystemExit("--quality-field e --quality-keep vanno insieme")

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config, streaming=True, split=args.split)
    it = iter(ds)
    first = next(it)
    if args.text_field not in first:
        raise SystemExit(f"campo '{args.text_field}' assente; schema reale: "
                         f"{ {k: type(v).__name__ for k, v in first.items()} }")
    if args.quality_field and args.quality_field not in first:
        raise SystemExit(f"campo qualità '{args.quality_field}' assente; schema: "
                         f"{ {k: type(v).__name__ for k, v in first.items()} }")

    eot = bytes([CHAR_EOT_BYTE])
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # valid PRIMA dal medesimo stream: nessun documento condiviso con train
    targets = [("valid", args.valid_bytes), ("train", args.train_bytes)]
    n_docs = n_skip = n_nul = 0

    def docs():
        yield first
        yield from it

    stream = docs()
    for split, budget in targets:
        out_path = Path(f"{args.out_prefix}_{split}_bytes.bin")
        written = 0
        with open(out_path, "wb") as out:
            for rec in stream:
                if args.quality_field and \
                        str(rec[args.quality_field]) not in args.quality_keep:
                    n_skip += 1
                    continue
                body = rec[args.text_field].encode("utf-8")
                if eot in body:
                    n_nul += body.count(eot)
                    body = body.replace(eot, b"")
                out.write(body + eot)
                written += len(body) + 1
                n_docs += 1
                if written >= budget:
                    break
            else:
                raise SystemExit(f"stream esaurito a {written:,} byte per {split}")
        print(f"{out_path}: {written:,} byte, cumulati {n_docs:,} documenti")
    print(f"scartati per qualità: {n_skip:,} · byte 0x00 rimossi dal testo: {n_nul:,}")


if __name__ == "__main__":
    main()
