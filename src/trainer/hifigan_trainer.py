from typing import Any, Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast


class HiFiGANTrainer:
    def __init__(
        self,
        *,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
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
        self.optimizer_g = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = device
        self.dataloaders = dataloaders
        self.epoch_len = epoch_len
        self.logger = logger
        self.writer = writer
        self.batch_transforms = batch_transforms
        self.skip_oom = skip_oom

        from hydra.utils import instantiate

        self.optimizer_d = instantiate(
            config.optimizer_d,
            params=list(self.model.msd.parameters()) + list(self.model.mpd.parameters()),
        )

        self.scaler = GradScaler(enabled=bool(config.trainer.get("amp", True)))

        self.global_step = 0
        self.max_steps = int(config.trainer.max_steps)


        self.mel = instantiate(config.mel_transform).to(device)

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

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

            with torch.no_grad():
                logmel_real = self.mel(real_wav)

            with autocast(enabled=self.scaler.is_enabled()):
                fake_wav = self.model.G(logmel_real).squeeze(1)
                T = min(real_wav.size(-1), fake_wav.size(-1))
                real_wav = real_wav[..., :T]
                fake_wav = fake_wav[..., :T]

            with autocast(enabled=self.scaler.is_enabled()):
                logmel_fake = self.mel(fake_wav)


            self.optimizer_d.zero_grad(set_to_none=True)
            with autocast(enabled=self.scaler.is_enabled()):
                msd_real_y, msd_real_f = self.model.msd(real_wav.unsqueeze(1))
                msd_fake_y, msd_fake_f = self.model.msd(fake_wav.detach().unsqueeze(1))
                msd_out = (msd_real_y, msd_fake_y, msd_real_f, msd_fake_f)

                mpd_real_y, mpd_real_f = self.model.mpd(real_wav.unsqueeze(1))
                mpd_fake_y, mpd_fake_f = self.model.mpd(fake_wav.detach().unsqueeze(1))
                mpd_out = (mpd_real_y, mpd_fake_y, mpd_real_f, mpd_fake_f)

                losses = self.criterion(
                    logmel_real=logmel_real,
                    logmel_fake=logmel_fake.detach(),
                    msd_out=msd_out,
                    mpd_out=mpd_out,
                )
                loss_d = losses["loss_d"]

            self.scaler.scale(loss_d).backward()
            self.scaler.step(self.optimizer_d)

            self.optimizer_g.zero_grad(set_to_none=True)
            with autocast(enabled=self.scaler.is_enabled()):
                msd_real_y, msd_real_f = self.model.msd(real_wav.unsqueeze(1))
                msd_fake_y, msd_fake_f = self.model.msd(fake_wav.unsqueeze(1))
                msd_out = (msd_real_y, msd_fake_y, msd_real_f, msd_fake_f)

                mpd_real_y, mpd_real_f = self.model.mpd(real_wav.unsqueeze(1))
                mpd_fake_y, mpd_fake_f = self.model.mpd(fake_wav.unsqueeze(1))
                mpd_out = (mpd_real_y, mpd_fake_y, mpd_real_f, mpd_fake_f)

                losses = self.criterion(
                    logmel_real=logmel_real,
                    logmel_fake=logmel_fake,
                    msd_out=msd_out,
                    mpd_out=mpd_out,
                )
                loss_g = losses["loss_g"]

            self.scaler.scale(loss_g).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()

            if self.lr_scheduler is not None:
                try:
                    self.lr_scheduler.step()
                except Exception:
                    pass

            if (self.global_step % log_every) == 0:
                msg = (
                    f"step={self.global_step} "
                    f"loss_d={float(losses['loss_d']):.4f} "
                    f"loss_g={float(losses['loss_g']):.4f}"
                )
                self.logger.info(msg)

            if hasattr(self.writer, "add_scalar"):
                try:
                    # tensorboard-like: (tag, scalar, step)
                    self.writer.add_scalar("train/loss_d", float(losses["loss_d"]), self.global_step)
                    self.writer.add_scalar("train/loss_g", float(losses["loss_g"]), self.global_step)
                except TypeError:
                    # cometml-like: (name, value)
                    self.writer.add_scalar("train/loss_d", float(losses["loss_d"]))
                    self.writer.add_scalar("train/loss_g", float(losses["loss_g"]))


            self.global_step += 1
