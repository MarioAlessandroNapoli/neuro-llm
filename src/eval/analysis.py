"""Statistica pre-registrata D6/D7: test di permutazione esatto sulla val loss,
Bradley-Terry via MM con tie = ½ vittoria, CI bootstrap clusterizzato sulle run,
BPB (formula The Pile), report scoring con check score↔lunghezza, self-agreement.
"""
import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .prompts import DATA_DIR

BT_ITERS = 200
N_BOOT = 2000
BOOT_SEED = 20260817
AXES = ("grammar", "consistency", "plot", "creativity")


def arch_of(run_name: str) -> str:
    return run_name.split("-d")[0]


# --- asse loss -------------------------------------------------------------

def perm_pvalue(a: list[float], b: list[float]) -> float:
    """Test esatto unilaterale, H1: mean(a) < mean(b). Con 3v3: 20 permutazioni, p minimo 0.05."""
    pooled = a + b
    observed = sum(a) / len(a) - sum(b) / len(b)
    count, total = 0, 0
    for idx in itertools.combinations(range(len(pooled)), len(a)):
        xa = [pooled[i] for i in idx]
        xb = [pooled[i] for i in range(len(pooled)) if i not in idx]
        d = sum(xa) / len(xa) - sum(xb) / len(xb)
        count += d <= observed + 1e-12
        total += 1
    return count / total


def loss_verdict(arm: list[float], baseline: list[float], epsilon: float, alpha: float = 0.05) -> str:
    """Stati D7 dell'asse loss: L+ / L- / L= / indeterminato."""
    if perm_pvalue(arm, baseline) <= alpha:
        return "L+"
    if perm_pvalue(baseline, arm) <= alpha:
        return "L-"
    delta = abs(sum(arm) / len(arm) - sum(baseline) / len(baseline))
    return "L=" if delta <= epsilon else "indeterminato"


# --- BPB -------------------------------------------------------------------

def bytes_per_token(txt_path: Path, bin_path: Path) -> float:
    """Convenzione D7: byte UTF-8 del solo testo, EOT esclusi dal conteggio token."""
    from ..prepare_data import iter_stories

    stories = list(iter_stories(txt_path))
    text_bytes = sum(len(s.encode()) for s in stories)
    n_tokens = bin_path.stat().st_size // 2  # uint16
    return text_bytes / (n_tokens - len(stories))


def bpb(loss_nats: float, bpt: float) -> float:
    return loss_nats / math.log(2) / bpt


# --- Bradley-Terry + bootstrap clusterizzato -------------------------------

def bt_fit(matches: list[tuple[str, str, float]]) -> dict[str, float]:
    """matches: (i, j, esito per i: 1 vittoria, 0.5 tie, 0 sconfitta), pesabili
    ripetendo le tuple. Ritorna rating in punti Elo (400·log10), media zero."""
    wins, n = defaultdict(float), defaultdict(float)
    players = set()
    for i, j, out in matches:
        players.update((i, j))
        wins[i] += out
        wins[j] += 1 - out
        n[i, j] += 1
        n[j, i] += 1
    p = {x: 1.0 for x in players}
    for _ in range(BT_ITERS):
        new = {}
        for x in players:
            denom = sum(n[x, y] / (p[x] + p[y]) for y in players if y != x)
            new[x] = wins[x] / denom if denom else 1.0
        gm = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        p = {x: v / gm for x, v in new.items()}
    return {x: 400 * math.log10(v) for x, v in p.items()}


def elo_report(records: list[dict], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Rating BT per arch e CI 95% delle differenze, bootstrap clusterizzato sulle run
    (unità di replicazione = seed). Riportato per strato e complessivo."""
    out = {}
    for stratum in ("short", "long", "all"):
        recs = [r for r in records if stratum == "all" or r["stratum"] == stratum]
        matches = [(arch_of(r["run_a"]), arch_of(r["run_b"]),
                    {"a": 1.0, "tie": 0.5, "b": 0.0}[r["winner"]]) for r in recs]
        ratings = bt_fit(matches)
        arch_runs = defaultdict(set)
        for r in recs:
            arch_runs[arch_of(r["run_a"])].add(r["run_a"])
            arch_runs[arch_of(r["run_b"])].add(r["run_b"])

        rng = random.Random(seed)
        diffs = defaultdict(list)
        pairs = sorted({tuple(sorted((arch_of(r["run_a"]), arch_of(r["run_b"])))) for r in recs})
        for _ in range(n_boot):
            mult = {}
            for arch, runs in arch_runs.items():
                runs = sorted(runs)
                drawn = [rng.choice(runs) for _ in runs]
                for run in runs:
                    mult[run] = drawn.count(run)
            boot_matches = []
            for r in recs:
                w = mult[r["run_a"]] * mult[r["run_b"]]
                boot_matches += [(arch_of(r["run_a"]), arch_of(r["run_b"]),
                                  {"a": 1.0, "tie": 0.5, "b": 0.0}[r["winner"]])] * w
            if not boot_matches:
                continue
            br = bt_fit(boot_matches)
            for x, y in pairs:
                if x in br and y in br:
                    diffs[x, y].append(br[x] - br[y])

        ci = {}
        for pair, ds in diffs.items():
            ds.sort()
            lo, hi = ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds)) - 1]
            verdict = "E=" if lo <= 0 <= hi else ("E+" if lo > 0 else "E-")
            ci[f"{pair[0]} vs {pair[1]}"] = {"diff": ratings[pair[0]] - ratings[pair[1]],
                                             "ci95": [lo, hi], "verdict": verdict}
        out[stratum] = {"ratings": ratings, "pairs": ci, "n_matches": len(matches)}
    return out


# --- scoring ---------------------------------------------------------------

def score_report(records: list[dict], generations_docs: list[dict]) -> dict:
    """Medie per arch/strato/asse (media delle run, con range tra run) e correlazione
    diagnostica score-totale ↔ lunghezza (AlpacaEval 2 LC come razionale)."""
    lengths = {}
    for doc in generations_docs:
        for g in doc["generations"]:
            for k, c in enumerate(g["completions"]):
                lengths[doc["run_name"], g["prompt_id"], k] = len(c)

    firsts = [r for r in records if not r["repeat"]]
    out = {}
    for stratum in ("short", "long", "all"):
        recs = [r for r in firsts if stratum == "all" or r["stratum"] == stratum]
        by_arch_run = defaultdict(lambda: defaultdict(list))
        for r in recs:
            by_arch_run[arch_of(r["run_name"])][r["run_name"]].append(r)
        report = {}
        for arch, runs in by_arch_run.items():
            axes = {}
            for ax in AXES:
                run_means = [sum(r[ax] for r in rs) / len(rs) for rs in runs.values()]
                axes[ax] = {"mean": sum(run_means) / len(run_means),
                            "range": [min(run_means), max(run_means)]}
            xs = [sum(r[ax] for ax in AXES) for rs in runs.values() for r in rs]
            ys = [lengths[r["run_name"], r["prompt_id"], r["completion_index"]]
                  for rs in runs.values() for r in rs]
            report[arch] = {"axes": axes, "len_corr": _pearson(xs, ys), "n": len(xs)}
        out[stratum] = report
    return out


def self_agreement(records: list[dict]) -> dict:
    """Confronta le richieste ripetute (identiche) con la prima passata: MAD per asse
    e tasso di accordo esatto. Aggiunta nostra, dichiarata come tale in D7."""
    first, rep = {}, {}
    for r in records:
        key = (r["run_name"], r["prompt_id"], r["completion_index"])
        (rep if r["repeat"] else first)[key] = r
    common = rep.keys() & first.keys()
    if not common:
        raise SystemExit("nessuna richiesta ripetuta nei risultati (submit-score --repeat N)")
    out = {}
    for ax in AXES:
        diffs = [abs(first[k][ax] - rep[k][ax]) for k in common]
        out[ax] = {"mad": sum(diffs) / len(diffs),
                   "exact": sum(d == 0 for d in diffs) / len(diffs)}
    out["n"] = len(common)
    return out


def _pearson(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else 0.0


# --- CLI -------------------------------------------------------------------

def _load_jsonl(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for p in paths for line in p.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bpb", help="loss (nats/token) → bits-per-byte sul nostro val set")
    p.add_argument("--loss", type=float, required=True)

    p = sub.add_parser("loss-test", help="verdetto asse loss: braccio vs baseline")
    p.add_argument("--arm", type=float, nargs="+", required=True)
    p.add_argument("--baseline", type=float, nargs="+", required=True)
    p.add_argument("--epsilon", type=float, required=True, help="σ baseline dai 5 seed")

    p = sub.add_parser("elo", help="report BT + CI dai risultati pairwise")
    p.add_argument("--results", type=Path, nargs="+", required=True)

    p = sub.add_parser("scores", help="report scoring + check lunghezza")
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--generations", type=Path, nargs="+", required=True)

    p = sub.add_parser("self-agreement")
    p.add_argument("--results", type=Path, nargs="+", required=True)

    args = parser.parse_args()
    if args.cmd == "bpb":
        bpt = bytes_per_token(DATA_DIR / "TinyStoriesV2-GPT4-valid.txt", DATA_DIR / "valid.bin")
        print(f"bytes/token (val, EOT esclusi): {bpt:.4f}")
        print(f"BPB: {bpb(args.loss, bpt):.4f}")
    elif args.cmd == "loss-test":
        print(f"p (arm < baseline): {perm_pvalue(args.arm, args.baseline):.4f}")
        print(f"p (baseline < arm): {perm_pvalue(args.baseline, args.arm):.4f}")
        print(f"verdetto: {loss_verdict(args.arm, args.baseline, args.epsilon)}")
    elif args.cmd == "elo":
        print(json.dumps(elo_report(_load_jsonl(args.results)), indent=1))
    elif args.cmd == "scores":
        docs = [json.loads(p.read_text()) for p in args.generations]
        print(json.dumps(score_report(_load_jsonl(args.results), docs), indent=1))
    elif args.cmd == "self-agreement":
        print(json.dumps(self_agreement(_load_jsonl(args.results)), indent=1))


if __name__ == "__main__":
    main()
