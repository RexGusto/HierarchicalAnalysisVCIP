# Findings

One entry per `plotter.sh` stage, in run order. Each entry covers its checks and gaps, and
records results whether or not a paper prints them. Data are the committed CSVs in `data/`.
"Maximum" is the highest top-1 accuracy (`val_acc_level1`) of a backbone over its
learning-rate sweep. "ADA ratio" is 100 × standard deviation / mean accuracy over that sweep.

## 1 `backbones`: serials 23 (fine-tuned) and 24 (frozen), aircraft, CUB, cars

- `summarize_acc` rebuilds `results_all/acc/summary_*_main.csv` and the pivots for serials
  23 and 24. They equal the committed files value for value.
- `summarize_acc_metrics` is run on two inputs:
  - `data/hierarchical_all_2.csv` gives the MSC, intra/inter-class variance and diversity
    summaries in `results_all/acc_metrics/summary_*_all.csv`.
  - `data/hierarchical_all_1.csv` gives the CKA, distance and L2-norm summaries in
    `summary_features_*_all.csv`, which the regressions read.
- The weighted-AP-against-learning-rate plots (`results_all/cvgip/`, 46 figures) and the
  per-serial box plots (`results_all/new_plots/all_model_performance/`, 8 figures) are
  regenerated.
- The accuracy and ADA-ratio regressions against 14 feature metrics (672 figures in
  `results_all/new_plots/ap_w/`) are regenerated, then tiled into
  `results_all/new_plots/merged/`.

## 2 `pseudo`: serials 32, 23, 39, 40, 51–53, 64–70 on aircraft, CUB, cars

- **Summaries.** `summarize_acc` and `compute_diff` rebuild `results_all/acc/acc_*` and
  `results_all/pseudo_hyperparam/*`. 239 of the 253 regenerated CSVs equal the committed
  ones by value.
- **Stale committed summaries.** The other 14 were written before the last wandb downloads:
  - `acc_pseudo_main` and the `main_results` tables now include six serial-40 runs that are
    in `data/hierarchical_pseudo_2.csv` but were missing from the committed summary: IN21k-MIIL
    on cars, MoCo v3 and SwAV on CUB, each at ratios 50 and 70. Among the runs present in both,
    only YFCC100M on CUB at ratio 70 changed, from 85.95 to 86.21 maximum.
  - The `model_size` tables change for DeiT-3 Huge: mean accuracy 74.30 becomes 89.00, and
    standard deviation 31.39 becomes 2.19. `data/hierarchical_model_size.csv` holds runs
    added after the summary was written. The CVPRW26 model-size figure predates these runs.
- **Figures.** 60 figures: hyperparameter sweeps, clustering algorithm, extractors, baseline
  against real and pseudo-hierarchy, and SoyLocal from serials 39 and 32.

## 3 `vcip`: serials 54 (HPH) and 55 (baseline) on SoyLocal; 83–86 and 90–95 on leaf datasets

- The SoyLocal and other-leaf summaries equal the committed ones value for value.
- **Maximum per backbone.** HPH minus baseline maximum on SoyLocal ranges from −2.00
  (ResNet Gluon) to +23.67 (ResNet SparK):
  - three backbones lose accuracy: Gluon, IN21k-MIIL and MoCo v3 ResNets;
  - three are unchanged;
  - fifteen gain.
- **Mean over each sweep.** HPH beats the baseline at every cluster ratio, by +10.9 (ratio
  10) to +12.8 (ratio 70) points.
- **SoyAgeing-R1 split.** The best serial-86 SoyAgeing-R1 runs used 4,944 training images.
  At the 984-image low-resource split the maxima are 86.16 (DeiT) and 84.24 (ViT-B).
- **Figures.** 19 figures, including the three VCIP26 data figures. Their comparison with the
  paper copies is in stage 6.

## 4 `few_examples`: serials 102 (baseline) and 103 (real hierarchy)

- Runs exist for aircraft at 1, 3 and 5 images per class, CUB at 1 and 3, and cars at 1.
- `plotter.sh` derives these pairs from the data and draws a box and an ADA-ratio bar plot per
  pair and architecture: 24 figures in `results_all/plots/model_low_examples/`.

## 5 `paper`: VCIP26 build

- `drafts/VCIP26/build.sh` builds the camera-ready: 5 pages, no errors, no undefined
  references. Its text equals `drafts/VCIP26/main_v2_camera_ready.pdf` apart from one glyph
  extraction.

## 6 `verify`: VCIP26 numbers, claims and figures

`codebase.verify.verify_paper_numbers` passes 23 of 32 checks.

**Pass:**
- 16 of the 22 printed numbers;
- 4 of the 7 claims;
- all 3 figures.

**Fail:** the 9 items listed in `NEXT_STEPS.md`:
- SparK gain, twice (the SparK sentence and the abstract);
- ViT-B SoyLocal table cell;
- two gains over CSDNet;
- SoyAgeing-R1 training-image count;
- "HPH raises every backbone's maximum";
- "DeiT has the largest ViT gain";
- the SoyAgeing-R1 split.

The figure check is low-specificity; `NUMBERS.md` gives its margin.

## Checkpoints (not a plotter stage)

`codebase.verify.verify_checkpoints` covers the 932 files in
`/mnt/hdd10tb/edwin/results_hierarchical`. The results are in `docs/CHECKPOINTS.md`.
