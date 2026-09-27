# Experiment log

| Date | Pair | Method | Seed | mIoU | Crop IoU | Weed IoU | Notes |
|---|---|---|---|---|---|---|---|

## MIC setup runs (not results)

- 2026-09-27, Kaggle T4, code b8db987, MIC 2f932a9, config bean_2019_to_2021_s0_smoke
  (50 iters, seed 0). Runs end to end: 110 source / 100 target train / 25 val images,
  eval and checkpoint work. After 50 iters (lr still in warmup): IoU bg 17.01, crop
  24.95, weed 6.02, mIoU 16.0. Memory 9744 MB. Speed ~3.4 s/iter on T4, so 40k iters
  would take ~38 h on a T4. masked.decode.loss_seg = 0 (teacher below the 0.968
  pseudo-label threshold this early); check it becomes > 0 in the full run.
  Fixes needed on the way: MPLBACKEND=Agg on Kaggle; MIC debug images crash with
  3 classes (patched in scripts/install_mic_rose.sh).

## Teacher-free SAM check (scripts/sam_check.py)

2026-09-27, CPU, SAM 2.1 tiny, points_per_side 32, pred_iou 0.8, stability 0.9,
min area 50; DINOv3 ViT-S/16 long side 1024, sim_thresh 0.85, dilate 3.
5 random target images with weed per pair (seed 0). Pixel totals over the 5 images.
"Lost" = weed pixels inside a region whose GT majority is not weed (a vote erases them).

| Target | Regions | Weed covered | Weed survives vote | Weed lost |
|---|---|---|---|---|
| Bean 2021 | SAM | 34.2% | 32.7% | 1.5% |
| Bean 2021 | SAM + DINO | 34.2% | 22.1% | 12.1% |
| Maize 2021 | SAM | 27.6% | 21.8% | 5.8% |
| Maize 2021 | SAM + DINO | 27.6% | 21.3% | 6.3% |

Uncovered weed pixels keep the teacher label in the real method, so they are not lost.
Label check on all 250 2021 target masks: share of labelled pixels that look green
(excess green > 20): crop 83% (bean) / 87% (maize), weed 62% / 57%, per-image weed
median 26% / 37%. Weed labels are loose polygons, looser than crop labels.

## Data preparation

- 2026-09-27: `scripts/prepare_rose.py --raw data/rose_zips/Dataset` on the Mendeley
  Dataset.zip (dated 2023-01-05). 1235 image/mask pairs, not 1250. 125 in every
  folder except 2019/Bipbip/Haricot: 110. That folder's Masks/ holds 15 files named
  Bipbip_mais_im_*.png that are byte-identical to masks in 2019/Bipbip/Mais/Masks
  (same in the zip, not an extraction error), and 15 bean images have no mask:
  Bipbip_haricot_im_ 00211 00581 00721 00951 01341 02421 02781 02841 02901 03691
  06581 06751 07181 07331 07421. These 15 images are left out of the index.
- 2026-09-27: HF mirror Project-AgML/weed_segmentation_france (same Mendeley DOI) has
  985 images: Bipbip 485, Pead 250, Weedelec 250, no Roseau; Haricot 2019 = 360
  (110 + 125 + 125). Same 15 Bipbip bean masks missing there, so no fix from the mirror.
- 2026-09-27: `scripts/prepare_mic_data.py` for the two 2019 -> 2021 pairs. data/mic/
  has bipbip_haricot_2019 (110), bipbip_mais_2019, bipbip_haricot_2021,
  bipbip_mais_2021 (125 each). Class share bg/crop/weed: bean 2019 88.0/7.8/4.2,
  bean 2021 73.3/22.3/4.4, maize 2019 86.7/8.6/4.7, maize 2021 65.7/31.2/3.1.
  Ignore 0.000% everywhere. Target splits 100 train / 25 val for seeds 0, 1, 2.
