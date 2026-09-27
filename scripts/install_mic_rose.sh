#!/usr/bin/env bash
# Copy the ROSE dataset class and configs into the cloned MIC, and link the data.
# Usage: bash scripts/install_mic_rose.sh [path to prepared data, default data/mic]
# Run again after editing anything under mic/.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MIC="$ROOT/third_party/MIC/seg"
DATA="$(cd "${1:-$ROOT/data/mic}" && pwd)"
[ -d "$MIC" ] || { echo "MIC not found at $MIC. Run scripts/setup_third_party.sh first."; exit 1; }

cp "$ROOT/mic/mmseg/datasets/rose.py" "$MIC/mmseg/datasets/rose.py"
INIT="$MIC/mmseg/datasets/__init__.py"
if ! grep -q "ROSEDataset" "$INIT"; then
    # Register the class: import it and add it to __all__.
    sed -i "s/^from .uda_dataset import UDADataset$/from .uda_dataset import UDADataset\nfrom .rose import ROSEDataset/" "$INIT"
    sed -i "s/^    'DarkZurichDataset',$/    'DarkZurichDataset',\n    'ROSEDataset',/" "$INIT"
fi
grep -q "from .rose import ROSEDataset" "$INIT" || { echo "Failed to register ROSEDataset in $INIT"; exit 1; }

# Patch MIC debug images for 3 classes. prepare_debug_out() treats any 3-channel
# output as an RGB image, so 3-class predictions (numpy) crash in denorm() at iter 0.
# Images are torch tensors, predictions numpy: use that to tell them apart.
VIS="$MIC/mmseg/models/utils/visualization.py"
if ! grep -q "torch.is_tensor(out)" "$VIS"; then
    sed -i "s/^    if out.shape\[0\] == 3:$/    if out.shape[0] == 3 and torch.is_tensor(out):  # ROSE patch/" "$VIS"
    sed -i "s/^    elif out.shape\[0\] > 3:$/    elif out.shape[0] >= 3:  # ROSE patch: 3-class predictions/" "$VIS"
fi
[ "$(grep -c "ROSE patch" "$VIS")" = 2 ] || { echo "Failed to patch $VIS"; exit 1; }

mkdir -p "$MIC/configs/rose"
cp "$ROOT"/mic/configs/rose/*.py "$MIC/configs/rose/"

mkdir -p "$MIC/data"
ln -sfn "$DATA" "$MIC/data/rose"

echo "Installed. Dataset class: $MIC/mmseg/datasets/rose.py"
echo "Configs: $(ls "$MIC/configs/rose" | tr '\n' ' ')"
echo "Data: $MIC/data/rose -> $DATA"
