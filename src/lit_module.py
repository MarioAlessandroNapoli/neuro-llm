import math
import time

import lightning as L
import torch
import torch.nn.functional as F

LR_FLOOR = 0.1


class LMModule(L.LightningModule):
    def __init__(self, model, lr: float, warmup_steps: int, max_steps: int, weight_decay: float = 0.1):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self._last_step_t = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x).transpose(1, 2), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ppl", loss.exp())
        now = time.perf_counter()
        if self._last_step_t is not None:
            self.log("tokens_per_sec", x.numel() / (now - self._last_step_t))
        self._last_step_t = now
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x).transpose(1, 2), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_ppl", loss.exp())

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )
        warmup, total = self.hparams.warmup_steps, self.hparams.max_steps

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = min((step - warmup) / max(1, total - warmup), 1.0)
            return LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
