from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def lsgan_discriminator_loss(d_real: List[torch.Tensor], d_fake: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for dr, df in zip(d_real, d_fake):
        loss = loss + torch.mean((dr - 1.0) ** 2) + torch.mean(df**2)
    return loss


def lsgan_generator_loss(d_fake: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for df in d_fake:
        loss = loss + torch.mean((df - 1.0) ** 2)
    return loss


def feature_matching_loss(f_real, f_fake) -> torch.Tensor:
    """
    f_*: list over discriminators, each is list over layers.
    Feature maps can be 3D (B,C,T) for MSD or 4D (B,C,H,W) for MPD.
    We crop all dims except (B,C) to min size to avoid mismatch.
    """
    loss = 0.0
    for fr_disc, ff_disc in zip(f_real, f_fake):
        for fr_l, ff_l in zip(fr_disc, ff_disc):

            fr = fr_l
            ff = ff_l

            if fr.dim() != ff.dim():
                d = min(fr.dim(), ff.dim())
                fr = fr.view(*fr.shape[:d])
                ff = ff.view(*ff.shape[:d])

            if fr.shape != ff.shape:
                slices = [slice(None), slice(None)]
                for d in range(2, fr.dim()):
                    m = min(fr.size(d), ff.size(d))
                    slices.append(slice(0, m))
                fr = fr[tuple(slices)]
                ff = ff[tuple(slices)]

            loss = loss + F.l1_loss(ff, fr)

    return loss




class HiFiGANLoss(nn.Module):
    def __init__(self, lambda_mel: float = 45.0, lambda_fm: float = 2.0):
        super().__init__()
        self.lambda_mel = float(lambda_mel)
        self.lambda_fm = float(lambda_fm)

    def forward(
        self,
        *,
        logmel_real: torch.Tensor,
        logmel_fake: torch.Tensor,
        msd_out,
        mpd_out,
    ) -> Dict[str, torch.Tensor]:
        """
        msd_out: (d_real, d_fake, f_real, f_fake)
        mpd_out: (d_real, d_fake, f_real, f_fake)
        """
        msd_dr, msd_df, msd_fr, msd_ff = msd_out
        mpd_dr, mpd_df, mpd_fr, mpd_ff = mpd_out

        T = min(logmel_real.size(-1), logmel_fake.size(-1))
        loss_mel = F.l1_loss(logmel_fake[..., :T], logmel_real[..., :T])

        loss_d = lsgan_discriminator_loss(msd_dr, msd_df) + lsgan_discriminator_loss(mpd_dr, mpd_df)
        loss_g_adv = lsgan_generator_loss(msd_df) + lsgan_generator_loss(mpd_df)

        loss_fm = feature_matching_loss(msd_fr, msd_ff) + feature_matching_loss(mpd_fr, mpd_ff)

        loss_g = loss_g_adv + self.lambda_mel * loss_mel + self.lambda_fm * loss_fm

        return {
            "loss_d": loss_d,
            "loss_g": loss_g,
            "loss_mel": loss_mel.detach(),
            "loss_fm": loss_fm.detach(),
            "loss_g_adv": loss_g_adv.detach(),
        }
