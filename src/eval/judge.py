"""Giudice LLM D7: claude-opus-5, effort medium, output strutturato, Batches API (−50%).

Due strumenti indipendenti:
- scoring assoluto: una chiamata per (prompt, braccio, run) → array di 10 vettori di score
  per-completamento (4 assi, rubric ancorata per fascia), ordine dei 10 mescolato con seed;
- pairwise Elo: verdetto set-vs-set — un giudizio per (coppia di run, prompt, ordine) con
  tutti i 10 completamenti per lato; i blocchi si scambiano tra i due ordini, shuffle
  intra-blocco con seed.

Il giudizio è cieco: il giudice non vede mai nomi di architetture. La mappa custom_id →
(run, prompt, permutazioni) vive nel sidecar eval/judgments/<tag>.batch.json.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from .prompts import EVAL_DIR

JUDGMENTS_DIR = EVAL_DIR / "judgments"
JUDGE_MODEL = "claude-opus-5"
EFFORT = "medium"
MAX_TOKENS = 8000  # cap su thinking adattivo + JSON di risposta
SHUFFLE_SEED = 20260817
SELF_AGREEMENT_SEED = 20260818

SCORE_1_10 = {"type": "integer", "enum": list(range(1, 11))}
SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "completion": {"type": "integer", "enum": list(range(1, 11))},
                    "grammar": SCORE_1_10,
                    "consistency": SCORE_1_10,
                    "plot": SCORE_1_10,
                    "creativity": SCORE_1_10,
                },
                "required": ["completion", "grammar", "consistency", "plot", "creativity"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}
ELO_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["A", "B", "tie"]},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "rationale"],
    "additionalProperties": False,
}

SCORE_SYSTEM = """You are evaluating story completions written by very small language models \
trained on TinyStories (children's stories with a simple vocabulary). You will see the \
beginning of a story and 10 independent completions of it. Score EACH completion \
independently on 4 axes, each on a 1-10 scale, using these anchored bands:

GRAMMAR (spelling, syntax, fluency of English):
1-2 mostly word salad, unreadable; 3-4 frequent errors that impede reading; \
5-6 understandable but with clear errors; 7-8 mostly correct, minor slips; \
9-10 fluent and correct throughout.

CONSISTENCY (with the story beginning and internally: characters, names, facts, tense):
1-2 ignores or contradicts the beginning; 3-4 major contradictions or character confusion; \
5-6 loosely follows, some inconsistencies; 7-8 consistent with minor lapses; \
9-10 fully consistent with the beginning and itself.

PLOT (does something coherent happen, with a discernible arc for a children's story):
1-2 no discernible events or pure repetition; 3-4 events without logic or connection; \
5-6 simple events, weak or missing resolution; 7-8 clear sequence with a reasonable ending; \
9-10 well-formed little arc: setup, development, resolution.

CREATIVITY (interest of the ideas, given the constraints of the genre):
1-2 degenerate repetition of the prompt; 3-4 purely formulaic continuation; \
5-6 predictable but adequate; 7-8 some fresh element that fits; \
9-10 surprising yet fitting ideas.

Return one score object per completion, in the order shown (completion = the number shown). \
Judge the text as-is; do not reward length in itself."""

ELO_SYSTEM = """You are comparing two very small language models trained on TinyStories \
(children's stories with a simple vocabulary). You will see the beginning of a story, then \
SET A: 10 independent completions sampled from model A, and SET B: 10 from model B.

Decide which MODEL is better at continuing this story, judging each set AS A WHOLE: weigh \
the typical quality across all 10 completions (grammar, consistency with the beginning, \
plot, creativity) — not the single best or worst sample. If the sets are of comparable \
overall quality, answer "tie". Do not reward length in itself. Keep the rationale to one \
or two sentences."""


def _rng(*parts) -> random.Random:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _params(system: str, user: str, schema: dict) -> dict:
    return {
        "model": JUDGE_MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
    }


def _load_generations(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if doc["random_init"]:
        raise SystemExit(f"{path}: generazioni da pesi random (--random-init), non giudicabili")
    if doc["protocol"]["completions"] != 10:
        raise SystemExit(f"{path}: {doc['protocol']['completions']} completamenti, il protocollo D7 ne richiede 10")
    return doc


def build_score_requests(doc: dict, repeat: int = 0) -> tuple[list[dict], dict]:
    """Una richiesta per prompt (10 completamenti mescolati); repeat>0 duplica un
    sottoinsieme seeded di richieste identiche per misurare la self-agreement."""
    from .prompts import load_prompt_set

    prompt_text = {p["id"]: p["text"] for p in load_prompt_set()}
    requests, meta = [], {}

    def add(gen, rep: bool):
        cid = f"r{len(requests):04d}"
        perm = list(range(10))
        _rng(SHUFFLE_SEED, doc["run_name"], gen["prompt_id"]).shuffle(perm)
        body = f"Story beginning:\n{prompt_text[gen['prompt_id']]}\n"
        for shown, orig in enumerate(perm, start=1):
            body += f"\nCompletion {shown}:\n{gen['completions'][orig]}\n"
        requests.append({"custom_id": cid, "params": _params(SCORE_SYSTEM, body, SCORE_SCHEMA)})
        meta[cid] = {"kind": "score", "run_name": doc["run_name"], "prompt_id": gen["prompt_id"],
                     "stratum": gen["stratum"], "perm": perm, "repeat": rep}

    for gen in doc["generations"]:
        add(gen, rep=False)
    if repeat:
        chosen = _rng(SELF_AGREEMENT_SEED, doc["run_name"]).sample(doc["generations"], repeat)
        for gen in chosen:
            add(gen, rep=True)
    return requests, meta


def build_elo_requests(doc_a: dict, doc_b: dict) -> tuple[list[dict], dict]:
    """188 verdetti per coppia di run: (44+50) prompt × 2 ordini, set-vs-set."""
    from .prompts import load_prompt_set

    prompt_text = {p["id"]: p["text"] for p in load_prompt_set()}
    gens_a = {g["prompt_id"]: g for g in doc_a["generations"]}
    gens_b = {g["prompt_id"]: g for g in doc_b["generations"]}
    if gens_a.keys() != gens_b.keys():
        raise SystemExit("le due run non coprono gli stessi prompt")

    pair = (doc_a["run_name"], doc_b["run_name"])
    requests, meta = [], {}
    for pid in gens_a:
        for order in (0, 1):
            cid = f"r{len(requests):04d}"
            display = [(doc_a, gens_a), (doc_b, gens_b)] if order == 0 else [(doc_b, gens_b), (doc_a, gens_a)]
            body = f"Story beginning:\n{prompt_text[pid]}\n"
            perms = {}
            for label, (doc, gens) in zip("AB", display):
                perm = list(range(10))
                _rng(SHUFFLE_SEED, *pair, pid, order, label).shuffle(perm)
                perms[label] = perm
                body += f"\nSET {label}:\n"
                for shown, orig in enumerate(perm, start=1):
                    body += f"\nCompletion {label}{shown}:\n{gens[pid]['completions'][orig]}\n"
            requests.append({"custom_id": cid, "params": _params(ELO_SYSTEM, body, ELO_SCHEMA)})
            meta[cid] = {"kind": "elo", "prompt_id": pid, "stratum": gens_a[pid]["stratum"],
                         "run_a": doc_a["run_name"], "run_b": doc_b["run_name"], "order": order,
                         "display_a": display[0][0]["run_name"], "perms": perms}
    return requests, meta


def submit(requests: list[dict], meta: dict, tag: str, kind: str, dry_run: bool):
    JUDGMENTS_DIR.mkdir(exist_ok=True)
    sidecar = JUDGMENTS_DIR / f"{tag}.batch.json"
    if sidecar.exists():
        raise SystemExit(f"{sidecar} esiste già: tag '{tag}' già sottomesso")
    if dry_run:
        (JUDGMENTS_DIR / f"{tag}.requests.json").write_text(json.dumps(requests, ensure_ascii=False, indent=1))
        print(f"dry-run: {len(requests)} richieste scritte in {tag}.requests.json, nessun invio")
        return
    import anthropic

    batch = anthropic.Anthropic().messages.batches.create(requests=requests)
    sidecar.write_text(json.dumps({"batch_id": batch.id, "kind": kind, "n_requests": len(requests),
                                   "model": JUDGE_MODEL, "effort": EFFORT, "meta": meta},
                                  ensure_ascii=False, indent=1))
    print(f"batch {batch.id} sottomesso: {len(requests)} richieste, sidecar {sidecar.name}")


def fetch(tag: str):
    import anthropic

    sidecar = json.loads((JUDGMENTS_DIR / f"{tag}.batch.json").read_text())
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(sidecar["batch_id"])
    if batch.processing_status != "ended":
        print(f"batch {batch.id}: {batch.processing_status} — {batch.request_counts}")
        return

    records, failures = [], []
    for result in client.messages.batches.results(sidecar["batch_id"]):
        m = sidecar["meta"][result.custom_id]
        if result.result.type != "succeeded":
            failures.append(f"{result.custom_id}: {result.result.type}")
            continue
        msg = result.result.message
        if msg.stop_reason not in ("end_turn", "stop_sequence"):
            failures.append(f"{result.custom_id}: stop_reason={msg.stop_reason}")
            continue
        payload = json.loads(next(b.text for b in msg.content if b.type == "text"))
        if m["kind"] == "score":
            shown_nums = sorted(s["completion"] for s in payload["scores"])
            if shown_nums != list(range(1, 11)):
                failures.append(f"{result.custom_id}: completions coperti {shown_nums}")
                continue
            for s in payload["scores"]:
                records.append({
                    "kind": "score", "run_name": m["run_name"], "prompt_id": m["prompt_id"],
                    "stratum": m["stratum"], "repeat": m["repeat"],
                    "completion_index": m["perm"][s["completion"] - 1],
                    "grammar": s["grammar"], "consistency": s["consistency"],
                    "plot": s["plot"], "creativity": s["creativity"],
                })
        else:
            v = payload["verdict"]
            winner = "tie" if v == "tie" else ("a" if (v == "A") == (m["display_a"] == m["run_a"]) else "b")
            records.append({
                "kind": "elo", "prompt_id": m["prompt_id"], "stratum": m["stratum"],
                "run_a": m["run_a"], "run_b": m["run_b"], "order": m["order"],
                "winner": winner, "rationale": payload["rationale"],
            })
    if failures:
        raise SystemExit(f"{len(failures)} richieste fallite su {sidecar['n_requests']}:\n" + "\n".join(failures))

    out = JUDGMENTS_DIR / f"{tag}.results.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    print(f"scritti {len(records)} record in {out}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("submit-score", help="scoring assoluto per una run")
    p.add_argument("--generations", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=0, help="richieste duplicate per la self-agreement")
    p.add_argument("--tag")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("submit-elo", help="pairwise set-vs-set per una coppia di run")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--tag")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("fetch", help="stato del batch; se concluso, scarica e valida i risultati")
    p.add_argument("--tag", required=True)

    args = parser.parse_args()
    if args.cmd == "submit-score":
        doc = _load_generations(args.generations)
        requests, meta = build_score_requests(doc, repeat=args.repeat)
        submit(requests, meta, args.tag or f"score-{doc['run_name']}", "score", args.dry_run)
    elif args.cmd == "submit-elo":
        doc_a, doc_b = _load_generations(args.a), _load_generations(args.b)
        requests, meta = build_elo_requests(doc_a, doc_b)
        submit(requests, meta, args.tag or f"elo-{doc_a['run_name']}--{doc_b['run_name']}", "elo", args.dry_run)
    elif args.cmd == "fetch":
        fetch(args.tag)


if __name__ == "__main__":
    main()
