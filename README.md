# HiFi-GAN Vocoder (PyTorch)

Репозиторий содержит реализацию нейросетевого вокодера HiFi-GAN для синтеза аудио из mel-спектрограмм.
Реализованы архитектура генератора и дискриминаторов, пайплайн обработки аудио, обучение с adversarial-лоссами и инференс.

## Установка

```bash
git clone https://github.com/<your_username>/HiFi-GAN.git
cd HiFi-GAN

chmod +x scripts/install.sh
TORCH_FLAVOR=cpu ./scripts/install.sh
```

Окружение создаётся в `.venv`.

Активация:

```bash
source .venv/bin/activate
```

(для GPU можно использовать `TORCH_FLAVOR=cu121` при наличии CUDA 12.1)

## Настройка логирования

Проект использует CometML. Нужно задать ключ:

```bash
export COMET_API_KEY="ваш_ключ"
```

## Обучение

```bash
python train.py -cn=baseline trainer.log_every=1
```

Все параметры находятся в `src/configs/baseline.yaml`.

## Инференс

```bash
python inference.py \
    --checkpoint path/to/checkpoint.pt \
    --input_wavs path/to/input_folder \
    --output_dir outputs
```

## Архитектура

Реализован классический HiFi-GAN:

* Generator: Conv1D + ConvTranspose1D upsampling + Multi-Receptive Field residual блоки + tanh выход.
* Multi-Scale Discriminator (MSD): анализ аудио на разных временных масштабах.
* Multi-Period Discriminator (MPD): анализ периодической структуры сигнала (гармоники).

## Функции потерь

Используются стандартные компоненты HiFi-GAN:

* Adversarial loss (LSGAN)
* Feature matching loss
* Mel reconstruction loss (L1)

Итоговый лосс генератора:

L_G = L_adv + λ_fm · L_fm + λ_mel · L_mel

## Mel-спектрограммы

Преобразование аудио:

1. STFT
2. Амплитудный спектр
3. Mel filterbank
4. Log compression

Mel вычисляется одинаково на обучении и инференсе.

## Структура

```
train.py
inference.py
scripts/install.sh

src/
  configs/
  datasets/
  transforms/
  model/
  loss/
  trainer/
  logger/
  utils/
```

## Требования

Python ≥ 3.9
PyTorch ≥ 2.x
torchaudio

