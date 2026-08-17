---
name: eval
description: Use when evaluating trained neuro-llm checkpoints with the D7 apparatus — generating completions, submitting or fetching Opus 5 judge batches (scoring/Elo), computing pre-registered verdicts (loss, Elo, BPB), or measuring judge self-agreement.
---

# eval — Harness di valutazione D7

Operatività dell'apparato di valutazione dello stadio 1. Le decisioni (protocolli, regola
di verdetto, numerosità) stanno in RESEARCH_LOG § D7: qui c'è solo il come. Codice in
`src/eval/`, artefatti in `eval/`.

## Apparato congelato — mai toccare

| File | Cosa | Regola |
|---|---|---|
| `eval/prompts_short.yaml` | 44 prompt ufficiali TinyStories, verbatim | Mai rigenerare né riformattare |
| `eval/prompts_long.yaml` | 50 prefissi ~300 token, seed 20260817 | Idem — `src.eval.prompts` rifiuta di sovrascriverlo |

Cambiarli invalida ogni confronto già fatto (sono apparato di misura come il tokenizer, D3).

## Ciclo di valutazione

Prerequisito per il giudice: `ANTHROPIC_API_KEY` nell'ambiente (giudice = `claude-opus-5`,
effort medium, Batches API −50%).

```
# 1. Generazione (per ogni run della griglia; ~min su GPU, più lento su M2)
uv run python -m src.eval.generate --arch <a> --seed <s> --run-name <r> --hub-repo <ckpt-repo>

# 2. Scoring assoluto (una run per volta; --repeat 10 per la self-agreement)
uv run python -m src.eval.judge submit-score --generations eval/generations/<r>.json --repeat 10

# 3. Elo pairwise (una coppia di run per volta → 188 verdetti set-vs-set)
uv run python -m src.eval.judge submit-elo --a eval/generations/<rA>.json --b eval/generations/<rB>.json

# 4. Recupero (ripetere finché il batch non è "ended"; valida e scrive <tag>.results.jsonl)
uv run python -m src.eval.judge fetch --tag <tag>

# 5. Analisi
uv run python -m src.eval.analysis elo --results eval/judgments/*.results.jsonl
uv run python -m src.eval.analysis scores --results ... --generations eval/generations/*.json
uv run python -m src.eval.analysis self-agreement --results ...
uv run python -m src.eval.analysis loss-test --arm <l1 l2 l3> --baseline <l1..l5> --epsilon <σ>
uv run python -m src.eval.analysis bpb --loss <val_loss_nats>
```

I tag di default: `score-<run_name>` e `elo-<runA>--<runB>`; un tag già sottomesso non è
risottomettibile (il sidecar `eval/judgments/<tag>.batch.json` fa da lucchetto).

## Fatti operativi

- **Cecità by design**: il giudice vede solo testi anonimi ed etichette A/B; la mappa
  custom_id → (run, permutazioni, ordini) vive nel sidecar locale, mai nei prompt.
- **Artefatti = àncora**: le generazioni JSON (con sha del checkpoint) rendono ogni
  giudizio ri-eseguibile su testi identici; con `--hub-repo` finiscono anche su HF.
- **`--dry-run`** su submit-* scrive le richieste in `<tag>.requests.json` senza inviare.
- **`fetch` è fail-loud**: richieste fallite, stop_reason anomali o score incompleti
  fermano tutto elencando i custom_id — niente risultati parziali silenziosi.
- Quante coppie di seed giudicare per braccio è una scelta scientifica (skill `neuro`),
  non è cablata nel codice: `submit-elo` prende una coppia di run alla volta.

## Gotchas

| Sintomo/rischio | Causa e fix |
|---|---|
| `submit-*` rifiuta le generazioni | File da `--random-init` (solo smoke) o completamenti ≠ 10: rigenerare dal checkpoint vero |
| `fetch` dice "in_progress" | Normale: i batch chiudono in ≤1h tipicamente; ritentare |
| Auth error dal giudice | `ANTHROPIC_API_KEY` assente nell'ambiente |
| Verdetti Elo con archi invertiti | Mai leggere `verdict` grezzo dai risultati API: usare i `.results.jsonl` (già rimappati display→run) |
| Costo giudice | ~94+5 richieste per scoring di una run, 188 per coppia Elo; Batches dimezza — stimare prima di sottomettere griglie grandi |
