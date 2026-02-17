from typing import Any, Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from hydra.utils import instantiate
import os

class HiFiGANTrainer:
    def __init__(
        self,
        *,
        model,
        criterion,
        metrics,
        optimizer,          # optimizer для Generator
        lr_scheduler,       # scheduler для Generator
        config,
        device: str,
        dataloaders: Dict[str, Any],
        epoch_len: Optional[int],
        logger,
        writer,
        batch_transforms=None,
        skip_oom: bool = True,
    ):
        self.model = model
        self.criterion = criterion
        self.metrics = metrics

        # Generator optimizer / scheduler
        self.optimizer_g = optimizer
        self.lr_scheduler_g = lr_scheduler

        self.config = config
        self.device = device
        self.dataloaders = dataloaders
        self.epoch_len = epoch_len
        self.logger = logger
        self.writer = writer
        self.batch_transforms = batch_transforms
        self.skip_oom = skip_oom

        # Discriminator optimizer
        self.optimizer_d = instantiate(
            config.optimizer_d,
            params=list(self.model.msd.parameters()) + list(self.model.mpd.parameters()),
        )

        # Discriminator scheduler (если есть в конфиге)
        self.lr_scheduler_d = None
        if config.get("lr_scheduler_d", None) is not None:
            self.lr_scheduler_d = instantiate(
                config.lr_scheduler_d,
                optimizer=self.optimizer_d,
            )

        # AMP
        self.scaler = GradScaler(enabled=bool(config.trainer.get("amp", True)))

        self.global_step = 0
        self.max_steps = int(config.trainer.max_steps)
        self.best_loss_g = float("inf")
        self.ckpt_dir = "checkpoints"

        self.mel = instantiate(config.mel_transform).to(self.device)

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _save_checkpoint(self, name: str):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, name)

        torch.save(
            {
                "generator": self.model.G.state_dict(),
                "msd": self.model.msd.state_dict(),
                "mpd": self.model.mpd.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict(),
                "optimizer_d": self.optimizer_d.state_dict(),
                "step": self.global_step,
            },
            path,
        )

        self.logger.info(f"Checkpoint saved: {path}")

    def train(self):
        train_loader = self.dataloaders["train"]
        it = iter(train_loader)

        log_every = int(self.config.trainer.get("log_every", 50))

        self.model.train()
        while self.global_step < self.max_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            batch = self._move_batch(batch)

            if self.batch_transforms is not None:
                batch = self.batch_transforms(batch)

            real_wav = batch["wav"]

            # Mel реального
            with torch.no_grad():
                logmel_real = self.mel(real_wav)

            # Generator forward
            with autocast(enabled=self.scaler.is_enabled()):
                fake_wav = self.model.G(logmel_real).squeeze(1)

                T = min(real_wav.size(-1), fake_wav.size(-1))
                real_wav = real_wav[..., :T]
                fake_wav = fake_wav[..., :T]

            # Mel фейкового
            with autocast(enabled=self.scaler.is_enabled()):
                logmel_fake = self.mel(fake_wav)

            # ======================
            # Train Discriminator
            # ======================
            self.optimizer_d.zero_grad(set_to_none=True)

            with autocast(enabled=self.scaler.is_enabled()):
                msd_real_y, msd_real_f = self.model.msd(real_wav.unsqueeze(1))
                msd_fake_y, msd_fake_f = self.model.msd(fake_wav.detach().unsqueeze(1))
                msd_out = (msd_real_y, msd_fake_y, msd_real_f, msd_fake_f)

                mpd_real_y, mpd_real_f = self.model.mpd(real_wav.unsqueeze(1))
                mpd_fake_y, mpd_fake_f = self.model.mpd(fake_wav.detach().unsqueeze(1))
                mpd_out = (mpd_real_y, mpd_fake_y, mpd_real_f, mpd_fake_f)

                losses_d = self.criterion(
                    logmel_real=logmel_real,
                    logmel_fake=logmel_fake.detach(),
                    msd_out=msd_out,
                    mpd_out=mpd_out,
                )

                loss_d = losses_d["loss_d"]

            self.scaler.scale(loss_d).backward()
            self.scaler.step(self.optimizer_d)

            # ======================
            # Train Generator
            # ======================
            self.optimizer_g.zero_grad(set_to_none=True)

            with autocast(enabled=self.scaler.is_enabled()):
                msd_real_y, msd_real_f = self.model.msd(real_wav.unsqueeze(1))
                msd_fake_y, msd_fake_f = self.model.msd(fake_wav.unsqueeze(1))
                msd_out = (msd_real_y, msd_fake_y, msd_real_f, msd_fake_f)

                mpd_real_y, mpd_real_f = self.model.mpd(real_wav.unsqueeze(1))
                mpd_fake_y, mpd_fake_f = self.model.mpd(fake_wav.unsqueeze(1))
                mpd_out = (mpd_real_y, mpd_fake_y, mpd_real_f, mpd_fake_f)

                losses_g = self.criterion(
                    logmel_real=logmel_real,
                    logmel_fake=logmel_fake,
                    msd_out=msd_out,
                    mpd_out=mpd_out,
                )

                loss_g = losses_g["loss_g"]

            self.scaler.scale(loss_g).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()

            if self.lr_scheduler_g is not None:
                self.lr_scheduler_g.step()

            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()

            if hasattr(self.writer, "set_step"):
                self.writer.set_step(self.global_step, mode="train")

            if (self.global_step % log_every) == 0:
                msg = (
                    f"step={self.global_step} "
                    f"loss_d={float(loss_d):.4f} "
                    f"loss_g={float(loss_g):.4f}"
                )
                self.logger.info(msg)

            if hasattr(self.writer, "add_scalar"):
                try:
                    self.writer.add_scalar("train/loss_d", float(loss_d), self.global_step)
                    self.writer.add_scalar("train/loss_g", float(loss_g), self.global_step)
                except TypeError:
                    self.writer.add_scalar("train/loss_d", float(loss_d))
                    self.writer.add_scalar("train/loss_g", float(loss_g))
            save_every = int(self.config.trainer.get("save_every", 1000))
            cur_loss_g = float(loss_g)
            if cur_loss_g < self.best_loss_g:
                self.best_loss_g = cur_loss_g
                self._save_checkpoint("best.pt")

            if self.global_step > 0 and (self.global_step % save_every) == 0:
                self._save_checkpoint("latest.pt")
                self._save_checkpoint(f"checkpoint_{self.global_step}.pt")
            self.global_step += 1
