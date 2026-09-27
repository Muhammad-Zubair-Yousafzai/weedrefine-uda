# Numbers to beat

Source: MaskAdapt (Nadeem, Asad, Anwar, Bais), CVPR 2025 Workshops, arXiv 2505.24026, Table 1.
Averaged over 3 random data sampling seeds (paper wording). No std reported.
Base method MIC with MiT-B5. MaskAdapt is RGB-D: it adds a MiT-B3 depth encoder
fed with ViT-estimated (monocular) depth maps. MIC column is RGB only.

Verified 2026-09-27: all 32 numbers below match Table 1 (page 6) of the arXiv PDF,
checked against both extracted text and the rendered page. MIC = column "MIC [25]",
MaskAdapt = column "Ours".

| Adaptation | MIC mIoU | MIC weed IoU | MaskAdapt mIoU | MaskAdapt weed IoU |
|---|---|---|---|---|
| Bean BIPBIP to WeedElec | 83.97 | 71.64 | 85.99 | 74.62 |
| Bean WeedElec to BIPBIP | 84.60 | 69.12 | 87.15 | 76.35 |
| Maize BIPBIP to WeedElec | 85.31 | 68.79 | 88.12 | 74.97 |
| Maize WeedElec to BIPBIP | 78.98 | 76.28 | 88.58 | 80.32 |
| Bean 2019 to 2021 | 80.55 | 62.36 | 81.98 | 66.32 |
| Bean 2021 to 2019 | 83.10 | 70.23 | 84.09 | 73.07 |
| Maize 2019 to 2021 | 79.84 | 62.15 | 82.89 | 66.71 |
| Maize 2021 to 2019 | 76.26 | 64.29 | 87.12 | 74.12 |

First targets: the two 2019 to 2021 pairs (weakest weed IoU).
Step 1 success criterion: reproduce MIC within about 2 mIoU on these pairs.

Caveat: MIC is not the strongest baseline on every pair. On Maize 2021 to 2019,
column [28] (84.90) and Fourier Trans. [45] (83.70) beat MIC's 76.26 mIoU; on Maize
WeedElec to BIPBIP, CGAN L phase [59] (84.18) beats MIC's 78.98. Compare against the
best column in each row, not only MIC.

Column [28] is Huang and Bais, "Unsupervised domain adaptation for weed segmentation
using greedy pseudo-labelling", CVPR 2024 Workshops, pp. 2484-2494 (checked against
the MaskAdapt reference list, 2026-09-27). It is the strongest non-MIC baseline on
several pairs, so report it next to MIC and MaskAdapt.

Data caveat: MaskAdapt says 125 labelled images per crop per team for 2019. Our
Mendeley download has only 110 usable pairs for Bipbip bean 2019 (see
docs/experiment_log.md). Bean pairs with Bipbip 2019 may not be on identical data.
