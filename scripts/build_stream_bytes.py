"""Pipeline dati D19: corpus HF in streaming -> directory con train/valid_bytes.bin.

Generalizza build_bytes.py (TinyStories) a qualunque corpus testuale su HF:
streama documenti, li separa col byte EOT 0x00, scrive fino al budget richiesto.
Contratti (post review 2026-08-25):
- --out-dir e' una DIRECTORY per corpus (es. data/nemotron): produce i nomi che
  train.py e le eval si aspettano ({out_dir}/train_bytes.bin, valid_bytes.bin),
  senza mai collidere coi bin TinyStories dello stadio char in data/.
- --revision obbligatoria: lo stream e' pinnato a un commit del repo HF, e un
  meta.json accanto ai bin registra dataset/revision/filtri/conteggi/sha256 —
  la comparabilita' tra i gradini C/D della scala poggia su questo.
- split per HASH del documento, non per posizione: valid sparsa lungo lo stream,
  mai adiacente al train (uno stream web e' ordinato per shard/dominio).
- fail-loud sullo schema: campo assente = stop con lo schema reale stampato.
- i byte 0x00 nel testo (possibili nel web crawl) vengono rimossi e contati.

Uso:
  python -m scripts.build_stream_bytes --dataset HuggingFaceFW/fineweb-edu \
      --config sample-10BT --revision <sha-o-tag> \
      --train-bytes 300_000_000 --valid-bytes 5_000_000 --out-dir data/fineweb
"""
import argparse
import hashlib
import json
from pathlib import Path

from src.configs import CHAR_EOT_BYTE

VALID_HASH_BUCKET = 4  # primo byte sha256 < 4 → candidato valid (~1/64 dei doc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--revision", required=True,
                        help="commit/tag del repo HF: niente default a main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--quality-field", default=None)
    parser.add_argument("--quality-keep", nargs="+", default=None)
    parser.add_argument("--train-bytes", type=int, required=True)
    parser.add_argument("--valid-bytes", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.quality_field is None) != (args.quality_keep is None):
        raise SystemExit("--quality-field e --quality-keep vanno insieme")

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config, streaming=True,
                      split=args.split, revision=args.revision)
    it = iter(ds)
    first = next(it)
    for field in filter(None, (args.text_field, args.quality_field)):
        if field not in first:
            raise SystemExit(f"campo '{field}' assente; schema reale: "
                             f"{ {k: type(v).__name__ for k, v in first.items()} }")

    eot = bytes([CHAR_EOT_BYTE])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: args.out_dir / f"{s}_bytes.bin" for s in ("train", "valid")}
    budget = {"train": args.train_bytes, "valid": args.valid_bytes}
    written = {"train": 0, "valid": 0}
    n_docs = {"train": 0, "valid": 0}
    n_skip = n_nul = 0
    outs = {s: open(p, "wb") for s, p in paths.items()}

    def docs():
        yield first
        yield from it

    for rec in docs():
        if all(written[s] >= budget[s] for s in outs):
            break
        if args.quality_field and \
                str(rec[args.quality_field]) not in args.quality_keep:
            n_skip += 1
            continue
        body = rec[args.text_field].encode("utf-8")
        if eot in body:
            n_nul += body.count(eot)
            body = body.replace(eot, b"")
        split = ("valid" if hashlib.sha256(body).digest()[0] < VALID_HASH_BUCKET
                 else "train")
        if written[split] >= budget[split]:
            split = "train" if split == "valid" else "valid"
            if written[split] >= budget[split]:
                continue
        outs[split].write(body + eot)
        written[split] += len(body) + 1
        n_docs[split] += 1
    else:
        raise SystemExit(f"stream esaurito: scritti {written}")
    for f in outs.values():
        f.close()

    sha = {s: hashlib.sha256(paths[s].read_bytes()).hexdigest()[:16]
           for s in paths}
    meta = {
        "dataset": args.dataset, "config": args.config,
        "revision": args.revision, "split": args.split,
        "quality_field": args.quality_field, "quality_keep": args.quality_keep,
        "written_bytes": written, "n_docs": n_docs,
        "scartati_qualita": n_skip, "byte_nul_rimossi": n_nul,
        "sha256_16": sha,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    for s, p in paths.items():
        print(f"{p}: {written[s]:,} byte, {n_docs[s]:,} documenti, sha {sha[s]}")
    print(f"scartati per qualità: {n_skip:,} · byte 0x00 rimossi: {n_nul:,}")


if __name__ == "__main__":
    main()
