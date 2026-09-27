# CLAUDE.md: weedrefine-uda

Project context for Claude Code. Read this fully before making changes.

## Personal context (not in repo)
@~/.claude/weedrefine-private.md

## Goal
Workshop paper (target: CVPR 2027 Vision for Agriculture workshop): foundation model
pseudo-label refinement for unsupervised domain adaptation (UDA) in crop and weed
semantic segmentation on the ROSE Challenge data.

## Paper to beat
MaskAdapt (Nadeem, Asad, Anwar, Bais), CVPR 2025 Workshops, arXiv 2505.24026.
Base method MIC with MiT-B5, RTX 3090 24 GB, about 19 hours per run, 3 seeds.
Numbers are in docs/targets.md. Verify against the PDF before citing.
Also cite and compare: Huang and Bais, greedy pseudo-labelling, CVPR 2024 Workshops;
Huang and Bais, self-training UDA, Intelligent Systems with Applications 2025.

## Method
1. MIC EMA teacher gives softmax on target image.
2. SAM 2 automatic masks give regions (small masks painted on top so weeds survive).
3. DINOv3 dense features merge adjacent SAM leaf fragments into whole plants
   (cosine similarity threshold, union-find). This is the agriculture-specific part.
4. Region majority vote over confident teacher pixels relabels each region.
5. Student trains on refined labels.

## Novelty status (checked Sep 2026)
SAM pseudo-label refinement for UDA already exists outside agriculture:
SAM4UDASS (arXiv 2401.08604), SeCo (2312.06331), CLOUDS (2312.09788),
RESAMPL-UDA (2025, biomedical), SRPL-SFDA (2506.09403), VLM pseudo-labels for
waste sorting (arXiv 2609.00898). No paper found applying it to crop/weed UDA on ROSE.
Do not claim SAM refinement itself as novel. Novelty = agriculture setting,
fragment merging for plants, beating MaskAdapt, weed IoU on cross-year pairs.

## Data
ROSE: https://data.mendeley.com/datasets/x8brgg2j28/2
Mirror: https://huggingface.co/datasets/Project-AgML/weed_segmentation_france
Pairs: BIPBIP <-> WeedElec (maize, bean, 2019) and BIPBIP 2019 <-> 2021.
Classes: background, crop, weed (four weed species merged).
Local copy: data/rose_zips/Dataset/{year}/{Team}/{Crop}/{Images,Masks}.
configs/rose.yaml layout and mask colors verified against Dataset/README.txt and
the authors' scripts/utils.py (black or orange = background, white = crop,
(216,67,82) = weed, anything else = ignore).

## Plan
- Phase A (offline, Kaggle T4): save MIC teacher softmax, run
  scripts/refine_offline.py, compare pseudo-label IoU: teacher vs SAM vote vs
  SAM + DINO vote. Start with Maize and Bean 2019 -> 2021 (weakest weed IoU, ~66).
- Kill rule: if refinement does not raise pseudo-label weed IoU offline, stop and rethink.
- Phase B (rented 24 GB GPU: g6.xlarge Spot, RunPod or Vast.ai RTX 3090):
  hook refinement into MIC (see docs/mic_integration.md), 8 pairs x 3 seeds.
- Step 1 success: reproduce MIC within about 2 mIoU of docs/targets.md.

## Current status
- Scaffold done. 12 unit tests pass (vote, merge, region map, metrics, mask decoding).
- Conda env `weedrefine` (Python 3.10, CPU torch) at ~/miniforge3. Run
  `~/miniforge3/bin/conda activate weedrefine`.
- docs/targets.md numbers verified against MaskAdapt Table 1 (2026-09-27).
- ROSE downloaded and indexed: data/rose_index.json has 1235 pairs. 125 per folder
  except Bipbip bean 2019 = 110 (15 masks missing in the release, see
  docs/experiment_log.md).
- DINOv3 access granted, HF login done. DINOv3 wrapper runs on CPU (ViT-S/16):
  aspect-preserving resize to long side 1024, returns patch grid (e.g. 48x64x384),
  merge.py maps regions onto the grid by area weights. ~1.5 s/image, <1 GB RAM.
- SAM 2 installed (CPU, tiny checkpoint), teacher-free check done: weak weed coverage,
  DINO merge hurts on bean (docs/experiment_log.md). Needs a tuning rerun.
- MIC data ready in data/mic/ (scripts/prepare_mic_data.py, 100/25 target splits per
  seed). ROSE dataset class and configs in mic/, installed into third_party/MIC/seg
  with scripts/install_mic_rose.sh. Configs: mic_daformer_rose_512x512.py plus one per
  pair and seed (scripts/make_mic_configs.py). CHOICE lines in the config are ours.
- mit_b5.pth: SegFormer OneDrive and Google Drive links are dead. Made from HF
  nvidia/mit-b5 (pinned revision) by scripts/convert_mit_b5.py -> checkpoints/mit_b5.pth
  (1052 tensors, 81.4M params, legacy torch format). MIC loads it with strict=False, so
  kaggle_mic_setup.sh checks names against MIC's mit_b5.
- NOT yet run: MIC (needs MIC env and GPU). Next: 50 iteration smoke test
  on Kaggle, then full runs on a rented RTX 3090.
- Next: get the MIC teacher softmax for the 2019 -> 2021 pairs (Phase A), and
  test the SAM 2 and DINO wrappers on 2 or 3 images on CPU.

## Environment
- Local machine: Ubuntu 24.04, i5-1135G7, 15 GB RAM, Intel Iris Xe only.
  NO NVIDIA GPU, NO CUDA. Use it for coding, tests, and CPU debugging on 2 or 3
  images with SAM 2.1 tiny and a small DINO model. Never try MIC training here.
- System Python is 3.12. Use conda env with Python 3.10 for weedrefine.
  MIC needs its own env with older mmcv (follow third_party/MIC README).
- Kaggle notebooks: code via GitHub clone, data as private Kaggle dataset under
  /kaggle/input, outputs to /kaggle/working. 12 hour sessions.
- DINOv3 on Hugging Face is gated. Fallback: a DINOv2 model name in the config.

## Rules for Claude Code
- Never invent results or numbers. Log every run in docs/experiment_log.md.
- Report mean and std over seeds. Report weed IoU separately.
- Keep pure logic in src/weedrefine and covered by tests; GPU code stays thin.
- Confirm MIC file and variable names against the cloned version before editing.
