# src/transforms/hifigan_mel.py
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn


def dynamic_range_compression_torch(x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes: torch.Tensor):
    return dynamic_range_compression_torch(magnitudes)


class HiFiGANMel(nn.Module):
    """
    Mel computation compatible with common HiFi-GAN reference code:
    STFT -> magnitude -> mel filterbank -> log compression.
    """
    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = 8000.0,
        center: bool = True,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr / 2
        self.center = center
        self.pad_mode = pad_mode

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer("mel_basis", self._build_mel_basis(), persistent=False)

    def _hz_to_mel(self, hz: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_basis(self) -> torch.Tensor:
        n_freqs = self.n_fft // 2 + 1
        fmin = torch.tensor(self.fmin, dtype=torch.float32)
        fmax = torch.tensor(self.fmax, dtype=torch.float32)

        m_min = self._hz_to_mel(fmin)
        m_max = self._hz_to_mel(fmax)
        m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)

        f_pts = self._mel_to_hz(m_pts)
        fft_freqs = torch.linspace(0, self.sr / 2, n_freqs)

        fb = torch.zeros(self.n_mels, n_freqs)
        for i in range(self.n_mels):
            f_left, f_center, f_right = f_pts[i], f_pts[i + 1], f_pts[i + 2]
            left_slope = (fft_freqs - f_left) / (f_center - f_left + 1e-8)
            right_slope = (f_right - fft_freqs) / (f_right - f_center + 1e-8)
            fb[i] = torch.clamp(torch.minimum(left_slope, right_slope), min=0.0)

        return fb

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [B, T] float in [-1,1]
        returns: log-mel [B, n_mels, frames]
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        pad = (self.n_fft - self.hop) // 2
        wav = torch.nn.functional.pad(wav, (pad, pad), mode=self.pad_mode)

        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window,
            center=self.center,
            return_complex=True,
        )
        mag = torch.abs(stft)
        mel = torch.matmul(self.mel_basis.to(mag.device), mag)
        mel = spectral_normalize_torch(mel)
        return mel

