import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate


def list_wavs(in_dir: Path):
    wavs = []
    for p in in_dir.rglob("*.wav"):
        name = p.name
        if name.startswith("._") or name.startswith("."):
            continue
        wavs.append(p)
    return sorted(wavs)


def load_generator_from_ckpt(cfg, ckpt_path: str, device: str):
    model = instantiate(cfg.model).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

 
    if isinstance(ckpt, dict) and "generator" in ckpt:
        state = ckpt["generator"]
    else:
        state = ckpt

    model.G.load_state_dict(state, strict=True)
    model.G.eval()
    return model.G


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoints/best.pt or latest.pt")
    parser.add_argument("--in_dir", required=True, help="input folder with wav files")
    parser.add_argument("--out_dir", required=True, help="output folder")
    parser.add_argument("--config", default="baseline", help="config name in src/configs (e.g., baseline)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir does not exist: {in_dir}")

    with initialize(config_path="src/configs", version_base=None):
        cfg = compose(config_name=args.config)

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    mel = instantiate(cfg.mel_transform).to(device)
    G = load_generator_from_ckpt(cfg, args.ckpt, device)

    target_sr = int(cfg.mel_transform.sr)

    wav_paths = list_wavs(in_dir)
    print(f"Found {len(wav_paths)} wav files")

    resamplers = {}

    for i, wav_path in enumerate(wav_paths, 1):
        rel = wav_path.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            wav, sr = sf.read(str(wav_path))
        except Exception as e:
            print(f"SKIP unreadable: {wav_path} ({e})")
            continue

        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        wav_t = torch.from_numpy(wav).float().unsqueeze(0)
        if sr != target_sr:
            key = (sr, target_sr)
            if key not in resamplers:
                resamplers[key] = torchaudio.transforms.Resample(sr, target_sr)
            wav_t = resamplers[key](wav_t)
            sr = target_sr

        wav_t = wav_t.to(device)


        logmel = mel(wav_t)
        y = G(logmel).squeeze().detach().cpu().numpy()

        sf.write(str(out_path), y, sr)

        if i % 50 == 0 or i == len(wav_paths):
            print(f"[{i}/{len(wav_paths)}] wrote {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()