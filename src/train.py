"""Entrypoint di training: locale (smoke) e Kaggle (run vere).

Resume: cerca last.ckpt in locale, poi sul repo HF (--hub-repo). Il checkpoint
viene ricaricato sul repo HF periodicamente e a fine run, così sopravvive alla
sessione Kaggle.
"""
import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from .configs import ModelConfig
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


def resolve_ckpt(local_path: Path, hub_repo: str | None, run_name: str) -> str | None:
    if local_path.exists():
        print(f"resume da checkpoint locale: {local_path}")
        return str(local_path)
    if hub_repo:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        try:
            path = hf_hub_download(hub_repo, f"{run_name}/last.ckpt")
            print(f"resume da checkpoint HF: {hub_repo}/{run_name}")
            return path
        except EntryNotFoundError:
            print(f"nessun checkpoint su {hub_repo}/{run_name}: parto da zero")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--tokens", type=int, required=True, help="budget di token di training")
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

    cfg = ModelConfig()
    run_name = args.run_name or f"{args.arch}-d{cfg.d_model}-L{cfg.n_layer}"
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

    model = build_model(args.arch, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{run_name}: {n_params/1e6:.1f}M parametri, {max_steps} step, {args.tokens/1e6:.0f}M token")

    lit = LMModule(model, lr=args.lr, warmup_steps=WARMUP_STEPS, max_steps=max_steps)

    ckpt_dir = Path("checkpoints") / run_name
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, save_last=True, every_n_train_steps=CKPT_EVERY_STEPS, save_top_k=0)
    ]
    if args.hub_repo:
        callbacks.append(HubUpload(args.hub_repo, run_name, ckpt_dir))

    logger = False
    if not args.no_wandb:
        logger = WandbLogger(project="neuro-llm", name=run_name, id=run_name, resume="allow")
        logger.log_hyperparams({**vars(args), "n_params": n_params, **cfg.__dict__})

    trainer = L.Trainer(
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
    trainer.fit(lit, train_dl, val_dl, ckpt_path=resolve_ckpt(ckpt_dir / "last.ckpt", args.hub_repo, run_name))


if __name__ == "__main__":
    main()
