"""Generazione per il giudice (D7): temp 1 × 10 completamenti per prompt, stop a EOT,
max 200 token nuovi (prefisso 300 + 200 = 500 ≤ finestra 512).

Il seed di ogni completamento deriva da (arch, seed della run, prompt_id, indice):
la riproducibilità cross-piattaforma non è garantita da PyTorch, quindi l'àncora vera
è il file JSON salvato (e pushato su HF con --hub-repo): ogni giudizio è ri-eseguibile
su testi identici.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ..configs import EOT_TOKEN
from ..models import build_model
from .prompts import EVAL_DIR, DATA_DIR, load_prompt_set

GENERATIONS_DIR = EVAL_DIR / "generations"
MAX_NEW_TOKENS = 200
N_COMPLETIONS = 10
TEMPERATURE = 1.0


def completion_seed(arch: str, run_seed: int, prompt_id: str, k: int) -> int:
    h = hashlib.sha256(f"{arch}|{run_seed}|{prompt_id}|{k}".encode()).digest()
    return int.from_bytes(h[:8], "big")


@torch.no_grad()
def generate_batch(model, prompt_ids: list[int], seeds: list[int], eot_id: int,
                   seq_len: int, device) -> list[list[int]]:
    """I K completamenti di uno stesso prompt in un solo batch: ogni riga campiona dal
    proprio Generator CPU (stessa semantica del caso batch-1); una riga che tocca EOT
    continua a scorrere ma il suo output è già chiuso."""
    gens = [torch.Generator().manual_seed(s) for s in seeds]
    ids = torch.tensor([prompt_ids] * len(seeds), device=device)
    outs = [[] for _ in seeds]
    done = [False] * len(seeds)
    for _ in range(MAX_NEW_TOKENS):
        logits = model(ids[:, -seq_len:])[:, -1]
        probs = F.softmax(logits.float() / TEMPERATURE, dim=-1).cpu()
        nxts = [torch.multinomial(probs[i], 1, generator=g).item() for i, g in enumerate(gens)]
        for i, nxt in enumerate(nxts):
            if not done[i]:
                if nxt == eot_id:
                    done[i] = True
                else:
                    outs[i].append(nxt)
        if all(done):
            break
        ids = torch.cat([ids, torch.tensor(nxts, device=device).unsqueeze(1)], dim=1)
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--seed", type=int, required=True, help="seed della run di training")
    parser.add_argument("--run-name", required=True, help="identità della run (nome ckpt e artefatto)")
    parser.add_argument("--ckpt", type=Path, help="default: checkpoints/<run-name>/last.ckpt")
    parser.add_argument("--hub-repo", help="scarica il ckpt da HF se assente e pusha le generazioni")
    parser.add_argument("--random-init", action="store_true",
                        help="SOLO SMOKE: pesi random, nessun checkpoint; mai per il giudice")
    parser.add_argument("--limit-prompts", type=int, help="solo smoke")
    parser.add_argument("--completions", type=int, default=N_COMPLETIONS)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))
    eot_id = tok.token_to_id(EOT_TOKEN)
    model, cfg = build_model(args.arch)

    ckpt_sha = None
    if args.random_init:
        print("ATTENZIONE: pesi random (--random-init), output senza valore scientifico")
        torch.manual_seed(args.seed)
    else:
        ckpt_path = args.ckpt or Path("checkpoints") / args.run_name / "last.ckpt"
        if not ckpt_path.exists() and args.hub_repo:
            from huggingface_hub import hf_hub_download

            ckpt_path = Path(hf_hub_download(args.hub_repo, f"{args.run_name}/last.ckpt"))
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()})
        ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:16]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()

    prompts = load_prompt_set()
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]

    generations = []
    for p in prompts:
        prompt_ids = tok.encode(p["text"]).ids
        seeds = [completion_seed(args.arch, args.seed, p["id"], k) for k in range(args.completions)]
        completions = [tok.decode(ids) for ids in
                       generate_batch(model, prompt_ids, seeds, eot_id, cfg.seq_len, device)]
        generations.append({"prompt_id": p["id"], "stratum": p["stratum"], "completions": completions})
        print(f"{p['id']}: {args.completions} completamenti "
              f"(mediana {sorted(len(c) for c in completions)[len(completions) // 2]} char)")

    doc = {
        "run_name": args.run_name,
        "arch": args.arch,
        "seed": args.seed,
        "ckpt_sha256": ckpt_sha,
        "random_init": args.random_init,
        "protocol": {"temperature": TEMPERATURE, "completions": args.completions,
                     "max_new_tokens": MAX_NEW_TOKENS},
        "generations": generations,
    }
    GENERATIONS_DIR.mkdir(exist_ok=True)
    out_path = GENERATIONS_DIR / f"{args.run_name}.json"
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"scritto {out_path}")

    if args.hub_repo and not args.random_init:
        from huggingface_hub import HfApi

        HfApi().upload_file(path_or_fileobj=out_path, path_in_repo=f"{args.run_name}/generations.json",
                            repo_id=args.hub_repo)
        print(f"pushato su {args.hub_repo}/{args.run_name}/generations.json")


if __name__ == "__main__":
    main()
