#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../third_party"
[ -d MIC ] || git clone https://github.com/lhoyer/MIC.git
git -C MIC checkout -q 2f932a9   # commit the ROSE files in mic/ were written against
[ -d sam2 ] || git clone https://github.com/facebookresearch/sam2.git
# --no-build-isolation: build against the installed torch. Without it pip downloads a
# second full CUDA torch into a temp build env (sam2 lists torch as a build requirement).
# Set SAM2_BUILD_CUDA=0 on machines without CUDA.
pip install --no-build-isolation -e sam2
echo "MIC needs its own environment (older mmcv). Follow third_party/MIC/seg/README.md."
