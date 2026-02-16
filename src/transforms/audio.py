import torch
import torchaudio


class LogMelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        sr: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        f_min: float = 0.0,
        f_max: float | None = None,
        power: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = float(eps)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(sr),
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(win_length),
            n_mels=int(n_mels),
            f_min=float(f_min),
            f_max=None if f_max is None else float(f_max),
            power=float(power),
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav:
          [B, T] or [T]
        returns:
          log_mel: [B, n_mels, frames] or [n_mels, frames]
        """
        m = self.mel(wav)
        return torch.log(m.clamp(min=self.eps))
