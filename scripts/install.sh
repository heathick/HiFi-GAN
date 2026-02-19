#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

TORCH_FLAVOR="${TORCH_FLAVOR:-cpu}"

TORCH_VER="${TORCH_VER:-2.2.0}"
TV_VER="${TV_VER:-0.17.0}"
TA_VER="${TA_VER:-2.2.0}"

echo "[1/5] Create venv: ${VENV_DIR}"
${PYTHON_BIN} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[2/5] Upgrade pip"
python -m pip install -U pip setuptools wheel

echo "[3/5] Install PyTorch (${TORCH_FLAVOR})"
if [[ "${TORCH_FLAVOR}" == "cpu" ]]; then
  pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" "torchaudio==${TA_VER}"
else
  pip install --index-url "https://download.pytorch.org/whl/${TORCH_FLAVOR}" \
    "torch==${TORCH_VER}" "torchvision==${TV_VER}" "torchaudio==${TA_VER}"
fi

echo "[4/5] Install project requirements"
pip install -r requirements.txt

echo "[5/5] (Optional) pre-commit install"
if command -v pre-commit >/dev/null 2>&1; then
  pre-commit install || true
fi

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
