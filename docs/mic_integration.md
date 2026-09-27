# Integrating refinement into MIC

MIC (https://github.com/lhoyer/MIC) is built on MMSegmentation. Self-training happens in
`seg/mmseg/models/uda/dacs.py`. Inside `forward_train`, the EMA teacher produces
`ema_softmax`, then `pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)`.

Plan:
1. Phase A (offline, cheap): add a hook that saves `ema_softmax` for target images to .npy,
   run `scripts/refine_offline.py`, and measure pseudo-label IoU against target ground truth.
   No retraining needed to test whether the idea works.
2. Phase B (online): precompute SAM regions per target image once (they do not change during
   training), then in `forward_train` replace `pseudo_label` with
   `weedrefine.refine.vote.refine_pseudo_label(...)` using the cached regions.
   Crop and flip augmentations must be applied identically to the cached region maps.

Confirm file and variable names against the MIC version you clone. They may differ.
