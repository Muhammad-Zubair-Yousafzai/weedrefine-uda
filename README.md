# WeedRefine-UDA

Foundation model pseudo-label refinement for unsupervised domain adaptation (UDA) in crop and weed semantic segmentation.

Research question: can SAM region masks, merged into whole plants using DINOv3 features, clean the teacher's pseudo-labels during self-training and raise weed IoU under cross-robot and cross-year domain shift on the ROSE Challenge data?

## Status

Early research code. Nothing here is a validated result yet.

## Benchmark and targets

Dataset: ROSE Challenge (INRAE Montoldre, France), maize and bean, classes background / crop / weed.
Adaptation pairs follow MaskAdapt (Nadeem et al., CVPR 2025 Workshops) and Greedy Pseudo-Labelling (Huang and Bais, CVPR 2024 Workshops).
Target numbers to beat are in `docs/targets.md`.

## Method (planned)

1. Teacher (MIC, MiT-B5) predicts softmax probabilities on a target image.
2. SAM 2 automatic mask generator produces region masks on the same image.
3. Fragment merging: adjacent SAM fragments whose mean DINOv3 features have cosine similarity above a threshold are merged, so one plant becomes one region instead of many leaves.
4. Region voting: each merged region takes the majority class of its confident teacher pixels, if that class is pure enough. Pixels outside regions keep the teacher label.
5. The student trains on the refined pseudo-labels.

## Repository layout

```
configs/            experiment configs (one file per adaptation pair)
scripts/            data prep, pseudo-label refinement, evaluation entry points
src/weedrefine/
  data/             ROSE indexing and loading
  refine/           SAM region generation, DINO fragment merging, region voting
  eval/             per-class IoU and mIoU
  utils/            seeding, IO helpers
tests/              unit tests for the pure logic (no GPU needed)
docs/               targets, MIC integration notes, experiment log
third_party/        MIC is cloned here, not vendored
```

## Setup

```bash
conda create -n weedrefine python=3.10 -y
conda activate weedrefine
pip install -r requirements.txt
pip install -e .
bash scripts/setup_third_party.sh    # clones MIC and SAM 2
```

## Data

Download the ROSE data from Mendeley: https://data.mendeley.com/datasets/x8brgg2j28/2
Unzip into `data/rose_raw/`, then:

```bash
python scripts/prepare_rose.py --raw data/rose_raw --out data/rose_index.json
```

Check the folder structure after download. `prepare_rose.py` uses glob patterns and a mask color map set in `configs/rose.yaml`. Adjust them if the release differs.

## Workflow

1. Reproduce MIC on Maize 2019 to 2021 and Bean 2019 to 2021. See `docs/mic_integration.md`.
2. Run offline refinement on saved teacher predictions and measure pseudo-label IoU against ground truth:
   `python scripts/refine_offline.py --config configs/pairs/maize_2019_to_2021.yaml`
3. Only if offline refinement improves pseudo-label weed IoU, hook it into MIC training.

## Tests

```bash
pytest -q
```

## License

MIT
