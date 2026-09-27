#!/usr/bin/env bash
# Set up MIC in a Kaggle GPU notebook: Python 3.8 env, pinned MIC, ROSE data, MiT-B5.
# Run from the repo root: bash scripts/kaggle_mic_setup.sh
# Needs two Kaggle datasets attached to the notebook:
#   the contents of data/mic/ (folders bipbip_haricot_2019, ...) and mit_b5.pth.
# The conda env lives in /root (not /kaggle/working) so it is not saved as output.
set -e
# Kaggle sets MPLBACKEND to its notebook backend (matplotlib_inline), which the
# Python 3.8 MIC env does not have. MIC only saves figures, so use Agg.
export MPLBACKEND=Agg
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MF=/root/miniforge
PY=$MF/envs/mic/bin/python
MIC_COMMIT=2f932a9   # MIC commit the ROSE files were written against (2024-08-10)
TORCH_WHL=https://download.pytorch.org/whl/torch_stable.html
MMCV_WHL=https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

echo "== 1/6 find inputs"
DATA="$(dirname "$(find /kaggle/input -type d -name bipbip_haricot_2019 | head -1)")"
WEIGHTS="$(find /kaggle/input -name mit_b5.pth | head -1)"
[ -d "$DATA/bipbip_haricot_2019" ] || { echo "ROSE data not found under /kaggle/input"; exit 1; }
[ -f "$WEIGHTS" ] || { echo "mit_b5.pth not found under /kaggle/input"; exit 1; }
echo "data: $DATA"; echo "weights: $WEIGHTS"

echo "== 2/6 Python 3.8 env"
if [ ! -x "$MF/bin/conda" ]; then
    curl -fsSL -o /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/mf.sh -b -p "$MF" > /dev/null
fi
[ -x "$PY" ] || "$MF/bin/conda" create -y -q -n mic python=3.8 > /dev/null

echo "== 3/6 MIC (pinned to $MIC_COMMIT)"
[ -d "$ROOT/third_party/MIC" ] || git clone -q https://github.com/lhoyer/MIC.git "$ROOT/third_party/MIC"
git -C "$ROOT/third_party/MIC" checkout -q "$MIC_COMMIT"
SEG="$ROOT/third_party/MIC/seg"

echo "== 4/6 packages (torch 1.7.1+cu110, MIC requirements, mmcv-full 1.3.7 prebuilt)"
"$PY" -m pip install -q torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f "$TORCH_WHL"
"$PY" -m pip install -q -r "$SEG/requirements.txt" -f "$TORCH_WHL"
"$PY" -m pip install -q mmcv-full==1.3.7 -f "$MMCV_WHL"

echo "== 5/6 ROSE files, data link, weights"
bash "$ROOT/scripts/install_mic_rose.sh" "$DATA"
mkdir -p "$SEG/pretrained"
ln -sfn "$WEIGHTS" "$SEG/pretrained/mit_b5.pth"

echo "== 6/6 checks"
cd "$SEG"
"$PY" - <<'EOF'
import torch, mmcv
import mmcv.ops  # fails if the mmcv-full CUDA ops do not load
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU",
      "| mmcv", mmcv.__version__)
# MIC loads pretrained weights with strict=False, so check names here.
from mmseg.models.backbones.mix_transformer import mit_b5
missing, unexpected = mit_b5().load_state_dict(
    torch.load("pretrained/mit_b5.pth", map_location="cpu"), strict=False)
print("mit_b5.pth | missing", len(missing), "| unexpected", len(unexpected))
assert not missing and not unexpected, (missing[:5], unexpected[:5])
from mmcv import Config
c = Config.fromfile("configs/rose/bean_2019_to_2021_s0_smoke.py")
print("config ok | classes", c.model.decode_head.num_classes,
      "| source", c.data.train.source.data_root, "| target", c.data.train.target.data_root,
      "| iters", c.runner.max_iters)
EOF
echo "Setup done. Python for MIC: $PY"
