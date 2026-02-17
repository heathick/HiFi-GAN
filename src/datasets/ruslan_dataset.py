import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset


def _list_wavs(data_dir: str | Path, exts: Tuple[str, ...] = (".wav",)) -> List[Path]:
    data_dir = Path(data_dir)
    files: List[Path] = []
    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if name.startswith("._") or name.startswith("."):
            continue
        if p.suffix.lower() in exts:
            files.append(p)

    if len(files) == 0:
        raise FileNotFoundError(f"No wav files found under: {data_dir}")
    return sorted(files)



class RuslanWavDataset(Dataset):
    """
    Returns dict:
      {
        "wav": Tensor[T] float32 in [-1, 1],
        "path": str
      }
    """

    def __init__(
        self,
        data_dir: str,
        segment_size: int,
        target_sr: int,
        mono: bool = True,
        normalize: bool = True,
        random_crop: bool = True,
        seed: Optional[int] = None,
    ):
        self.files = _list_wavs(data_dir)
        self.segment_size = int(segment_size)
        self.target_sr = int(target_sr)
        self.mono = bool(mono)
        self.normalize = bool(normalize)
        self.random_crop = bool(random_crop)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _to_mono(wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            return wav
        if wav.size(0) == 1:
            return wav[0]
        return wav.mean(dim=0)

    def _resample_if_needed(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.target_sr:
            return wav
        return torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.target_sr)

    def _crop_or_pad(self, wav: torch.Tensor) -> torch.Tensor:
        T = wav.numel()
        seg = self.segment_size
        if T == seg:
            return wav
        if T > seg:
            if self.random_crop:
                start = self.rng.randint(0, T - seg)
            else:
                start = 0
            return wav[start : start + seg]
        pad = seg - T
        return torch.nn.functional.pad(wav, (0, pad))

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.files[idx]
        wav, sr = torchaudio.load(str(path))
        if self.mono:
            wav = self._to_mono(wav)
        else:
            wav = wav[0] if wav.dim() == 2 else wav

        wav = self._resample_if_needed(wav, sr)
        wav = wav.to(torch.float32)

        if self.normalize:
            m = wav.abs().max().clamp(min=1e-8)
            wav = wav / m
            wav = wav * 0.95

        wav = wav.clamp(-1.0, 1.0)
        wav = self._crop_or_pad(wav)

        return {"wav": wav, "path": str(path)}
