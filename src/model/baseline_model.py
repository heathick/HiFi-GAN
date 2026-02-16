from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                padding=((kernel_size - 1) // 2) * d,
                dilation=d
            ))
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                padding=(kernel_size - 1) // 2,
                dilation=1
            ))
            for _ in dilations
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = x + xt
        return x

class Generator(nn.Module):
    def __init__(
        self,
        n_mels=80,
        upsample_rates=(8, 8, 2, 2),
        upsample_kernels=(16, 16, 4, 4),
        channels=512,
        resblock_kernels=(3, 7, 11),
        resblock_dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.pre = nn.utils.weight_norm(nn.Conv1d(n_mels, channels, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        ch = channels
        for u, k in zip(upsample_rates, upsample_kernels):
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(ch, ch // 2, k, u, padding=(k - u) // 2)
                )
            )
            ch = ch // 2

            blocks = nn.ModuleList([
                ResBlock(ch, rk, rd) for rk, rd in zip(resblock_kernels, resblock_dilations)
            ])
            self.mrfs.append(blocks)

        self.post = nn.utils.weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, mel):
        x = self.pre(mel)
        for up, blocks in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            xs = 0
            for b in blocks:
                xs = xs + b(x)
            x = xs / len(blocks)

        x = F.leaky_relu(x, 0.1)
        x = torch.tanh(self.post(x))
        return x  # (B, 1, T)

class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
            nn.utils.weight_norm(nn.Conv1d(16, 64, 41, 4, padding=20, groups=4)),
            nn.utils.weight_norm(nn.Conv1d(64, 256, 41, 4, padding=20, groups=16)),
            nn.utils.weight_norm(nn.Conv1d(256, 1024, 41, 4, padding=20, groups=64)),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, 41, 4, padding=20, groups=256)),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.post = nn.utils.weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for c in self.convs:
            x = c(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x, fmap

class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(), DiscriminatorS(), DiscriminatorS()])
        self.pooling = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=1),
            nn.AvgPool1d(4, 2, padding=1),
        ])

    def forward(self, x):
        outs = []
        fmaps = []
        for i, d in enumerate(self.discriminators):
            y, fm = d(x)
            outs.append(y)
            fmaps.append(fm)
            if i < 2:
                x = self.pooling[i](x)
        return outs, fmaps

class DiscriminatorP(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.post = nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        b, c, t = x.shape
        p = self.period
        if t % p != 0:
            pad = p - (t % p)
            x = F.pad(x, (0, pad), mode="reflect")
            t = t + pad
        x = x.view(b, c, t // p, p)

        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x, fmap

class MPD(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, x):
        outs, fmaps = [], []
        for d in self.discriminators:
            y, fm = d(x)
            outs.append(y)
            fmaps.append(fm)
        return outs, fmaps


class HiFiGAN(nn.Module):
    """
    Wrapper so Hydra can instantiate a single model object.
    """

    def __init__(
        self,
        generator: Dict,
        msd: Dict,
        mpd: Dict,
    ):
        super().__init__()
        self.G = Generator(**generator)
        self.msd = MSD(**msd)
        self.mpd = MPD(**mpd)

    @torch.no_grad()
    def generate(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [B, T] in [-1,1]
        For vocoder-like setup you probably feed mel; adjust as needed.
        """
        x = wav.unsqueeze(1)
        y = self.G(x).squeeze(1)
        return y

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        real = batch["wav"]
        fake = self.generate(real)
        return {"real_wav": real, "fake_wav": fake}
