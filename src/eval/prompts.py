"""Prompt set D7: strato corto (44 ufficiali, verbatim) + strato lungo (50 prefissi dal val V2).

I file in eval/ sono apparato di misura come il tokenizer (D3): una volta generati e
committati non si rigenerano. Lo strato lungo è deterministico dato (val set, tokenizer,
SELECTION_SEED): prefisso = primi 300 token della storia troncati all'ultimo whitespace.
"""
import argparse
import random
import re
from pathlib import Path

import yaml
from tokenizers import Tokenizer

from ..prepare_data import iter_stories

EVAL_DIR = Path(__file__).parent.parent.parent / "eval"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
SHORT_PATH = EVAL_DIR / "prompts_short.yaml"
LONG_PATH = EVAL_DIR / "prompts_long.yaml"

SELECTION_SEED = 20260817
N_LONG = 50
PREFIX_TOKENS = 300
PREFIX_MIN_TOKENS = 250
MIN_STORY_TOKENS = 400  # garantisce ≥100 token di continuazione vera oltre il prefisso
N_SHORT_EXPECTED = 44


def load_prompt_set() -> list[dict]:
    """Entrambi gli strati come [{id, stratum, text}], ordine stabile."""
    short = yaml.safe_load(SHORT_PATH.read_text())
    if len(short) != N_SHORT_EXPECTED:
        raise SystemExit(f"strato corto: {len(short)} prompt, attesi {N_SHORT_EXPECTED}")
    long_doc = yaml.safe_load(LONG_PATH.read_text())
    prompts = [
        {"id": f"short-{i + 1:02d}", "stratum": "short", "text": t} for i, t in enumerate(short)
    ]
    prompts += [
        {"id": p["id"], "stratum": "long", "text": p["text"]} for p in long_doc["prompts"]
    ]
    return prompts


def build_long_set():
    if LONG_PATH.exists():
        raise SystemExit(f"{LONG_PATH} esiste già: è apparato congelato, non si rigenera")
    tok = Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))
    stories = list(iter_stories(DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"))
    encodings = tok.encode_batch(stories)
    pool = [i for i, e in enumerate(encodings) if len(e.ids) >= MIN_STORY_TOKENS]
    print(f"pool storie ≥{MIN_STORY_TOKENS} token: {len(pool)} su {len(stories)}")

    rng = random.Random(SELECTION_SEED)
    chosen = sorted(rng.sample(pool, N_LONG))

    prompts = []
    for k, idx in enumerate(chosen):
        text = tok.decode(encodings[idx].ids[:PREFIX_TOKENS])
        cut = max(m.start() for m in re.finditer(r"\s", text))
        prefix = text[:cut].rstrip()
        n_tok = len(tok.encode(prefix).ids)
        if not PREFIX_MIN_TOKENS <= n_tok <= PREFIX_TOKENS:
            raise SystemExit(f"storia {idx}: prefisso di {n_tok} token fuori da "
                             f"[{PREFIX_MIN_TOKENS}, {PREFIX_TOKENS}]")
        prompts.append({"id": f"long-{k + 1:02d}", "story_index": idx, "n_tokens": n_tok, "text": prefix})

    doc = {
        "selection_seed": SELECTION_SEED,
        "pool_size": len(pool),
        "rule": f"primi {PREFIX_TOKENS} token, troncati all'ultimo whitespace; storie ≥{MIN_STORY_TOKENS} token",
        "prompts": prompts,
    }
    LONG_PATH.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100))
    print(f"scritto {LONG_PATH}: {N_LONG} prefissi, token mediani "
          f"{sorted(p['n_tokens'] for p in prompts)[N_LONG // 2]}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="genera lo strato lungo (una tantum)").parse_args()
    build_long_set()
