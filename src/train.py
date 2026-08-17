"""Entrypoint di training: locale (smoke) e Kaggle (run vere).

L'identità di una run è il suo nome: arch, forma, budget, seed e lr. Il resume è
solo esplicito (--resume): un checkpoint trovato per quel nome senza --resume è un
errore, mai una ripresa silenziosa. Con --hub-repo il checkpoint viene ricaricato
sul repo HF periodicamente e a fine run, così sopravvive alla sessione Kaggle.
Convenzione anti-proliferazione: gli sweep lr girano senza --hub-repo (niente
checkpoint remoti) e con --group sweep-lr; le run di griglia con --group grid-stage1.
"""
import argparse
import hashlib
import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from .configs import BASELINE_BACKBONE_PARAMS, EOT_TOKEN, PARITY_TOL
from .data import TokenWindowDataset
from .lit_module import LMModule
from .models import build_model

CKPT_EVERY_STEPS = 1000
HUB_UPLOAD_EVERY_STEPS = 2000
VAL_CHECK_STEPS = 1000
VAL_BATCHES = 100
WARMUP_STEPS = 200


class HubUpload(Callback):
    def __init__(self, repo_id: str, run_name: str, ckpt_dir: Path):
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.repo_id = repo_id
        self.run_name = run_name
        self.ckpt_path = ckpt_dir / "last.ckpt"
        self.api.create_repo(repo_id, private=True, exist_ok=True)

    def _push(self):
        if self.ckpt_path.exists():
            self.api.upload_file(
                path_or_fileobj=self.ckpt_path,
                path_in_repo=f"{self.run_name}/last.ckpt",
                repo_id=self.repo_id,
            )

    def on_train_batch_end(self, trainer, *_):
        if trainer.global_step > 0 and trainer.global_step % HUB_UPLOAD_EVERY_STEPS == 0:
            self._push()

    def on_train_end(self, trainer, pl_module):
        trainer.save_checkpoint(self.ckpt_path)
        self._push()


def resolve_ckpt(local_path: Path, hub_repo: str | None, run_name: str, resume: bool) -> str | None:
    found, source = None, None
    if local_path.exists():
        found, source = str(local_path), f"locale: {local_path}"
    elif hub_repo:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        try:
            found = hf_hub_download(hub_repo, f"{run_name}/last.ckpt")
            source = f"HF: {hub_repo}/{run_name}"
        except EntryNotFoundError:
            pass
    if found and not resume:
        raise SystemExit(
            f"esiste già un checkpoint per '{run_name}' ({source}) ma --resume non è stato "
            "richiesto: rilancia con --resume per riprendere, o rimuovi/rinomina il checkpoint "
            "per ripartire da zero"
        )
    if resume and not found:
        raise SystemExit(f"--resume richiesto ma nessun checkpoint per '{run_name}' (né locale né HF)")
    if found:
        print(f"resume da checkpoint {source}")
    return found


def check_ckpt_compat(ckpt_path: str, max_steps: int, lr: float):
    h = torch.load(ckpt_path, map_location="cpu", weights_only=False)["hyper_parameters"]
    if h["max_steps"] != max_steps or h["lr"] != lr:
        raise SystemExit(
            f"checkpoint incompatibile con gli argomenti: max_steps {h['max_steps']} vs {max_steps}, "
            f"lr {h['lr']} vs {lr} — riprendere con budget o lr diversi riscriverebbe lo scheduler"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--tokens", type=int, required=True, help="budget di token di training")
    parser.add_argument("--seed", type=int, required=True, help="parte dell'identità della run")
    parser.add_argument("--resume", action="store_true", help="riprendi dal checkpoint di questa run")
    parser.add_argument("--group", help="gruppo W&B: sweep-lr | grid-stage1")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--precision", default="16-mixed", choices=["16-mixed", "bf16-mixed", "32-true"])
    parser.add_argument("--hub-repo", help="repo HF privato per checkpoint (es. user/neuro-llm-ckpt)")
    parser.add_argument("--run-name")
    parser.add_argument("--max-time", help="es. 00:11:00:00 per sessioni Kaggle")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)
    model, cfg = build_model(args.arch)
    lr_tag = f"{args.lr:.0e}".replace("e-0", "e-")
    run_name = args.run_name or (
        f"{args.arch}-d{cfg.d_model}-L{cfg.n_layer}-t{args.tokens // 10**6}M-s{args.seed}-lr{lr_tag}"
    )
    max_steps = args.tokens // (args.batch_size * cfg.seq_len)

    train_ds = TokenWindowDataset(args.data_dir / "train.bin", cfg.seq_len)
    val_ds = TokenWindowDataset(args.data_dir / "valid.bin", cfg.seq_len)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    train_dl = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_dl = DataLoader(val_ds, **loader_kwargs)

    tok_path = args.data_dir / "tokenizer.json"
    vocab = json.loads(tok_path.read_text())["model"]["vocab"]
    if len(vocab) != cfg.vocab_size or vocab[EOT_TOKEN] != 0:
        raise SystemExit(
            f"tokenizer incoerente con D3: vocab {len(vocab)} vs {cfg.vocab_size}, "
            f"id di {EOT_TOKEN} = {vocab.get(EOT_TOKEN)} (atteso 0)"
        )
    tokenizer_sha = hashlib.sha256(tok_path.read_bytes()).hexdigest()[:16]

    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = n_params - cfg.vocab_size * cfg.d_model
    drift = (n_backbone - BASELINE_BACKBONE_PARAMS) / BASELINE_BACKBONE_PARAMS
    if abs(drift) > PARITY_TOL:
        raise SystemExit(
            f"parità violata (D6): backbone {n_backbone:,} vs riferimento "
            f"{BASELINE_BACKBONE_PARAMS:,} ({drift:+.1%}, tolleranza ±{PARITY_TOL:.0%})"
        )
    print(
        f"{run_name}: {n_params/1e6:.2f}M parametri ({n_backbone/1e6:.2f}M backbone, "
        f"{drift:+.1%} dal riferimento), {max_steps} step, {args.tokens/1e6:.0f}M token"
    )

    lit = LMModule(model, lr=args.lr, warmup_steps=WARMUP_STEPS, max_steps=max_steps)

    ckpt_dir = Path("checkpoints") / run_name
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, save_last=True, every_n_train_steps=CKPT_EVERY_STEPS, save_top_k=0)
    ]
    if args.hub_repo:
        callbacks.append(HubUpload(args.hub_repo, run_name, ckpt_dir))

    logger = False
    if not args.no_wandb:
        logger = WandbLogger(project="neuro-llm", name=run_name, id=run_name, group=args.group, resume="allow")
        logger.log_hyperparams({
            **vars(args), "n_params": n_params, "n_backbone": n_backbone,
            "tokenizer_sha": tokenizer_sha, **cfg.__dict__,
        })

    trainer = L.Trainer(
        devices=1,
        max_steps=max_steps,
        max_time=args.max_time,
        precision=args.precision,
        gradient_clip_val=1.0,
        val_check_interval=min(VAL_CHECK_STEPS, max_steps),
        limit_val_batches=VAL_BATCHES,
        log_every_n_steps=20,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
    )
    ckpt_path = resolve_ckpt(ckpt_dir / "last.ckpt", args.hub_repo, run_name, args.resume)
    if ckpt_path:
        check_ckpt_compat(ckpt_path, max_steps, args.lr)
    trainer.fit(lit, train_dl, val_dl, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
