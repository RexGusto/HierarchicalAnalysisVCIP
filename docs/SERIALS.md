# SERIALS.md: wandb run inventory

Runs live in the wandb project `nycu_pcs/Hierarchical`, which `codebase/download/` exports to
`data/*.csv`. A serial groups the runs of one experiment. The "Runs in CSV" column counts
rows in the committed CSV, not runs on wandb. Where the two differ, the CSV is a snapshot
from its download date. `results_all/serials/serial_coverage.csv`, written by
`codebase.verify.reconcile_serials`, counts every serial's wandb runs against these CSVs and
the checkpoints.

**Defaults.** All runs use 448×448 crops, batch size 8 and seed 0 unless the table says
otherwise. Training uses a cosine schedule with a 2-epoch warmup, fp16, and gradient-norm
clipping at 1.0. These are the stored configs of the checkpoints of serials 21, 32, 54 and 86.

**Learning-rate grids.** A sweep has 10 learning rates:
- Fine-tuned (23, 32, 40, 54, 55 and the other FT sweeps): SGD 0.1, 0.03, 0.01 and 0.003,
  momentum 0.9, no weight decay; AdamW 1e-3, 5e-4, 1e-4, 5e-5, 1e-5 and 5e-6, weight decay
  0.05.
- Frozen (24): SGD 0.3, 0.1, 0.03 and 0.01; AdamW 5e-3, 1e-3, 5e-4, 1e-4, 5e-5 and 1e-5.

The source slide deck is `docs/sources/serials_deck.md` (original `.pptx` beside it). Saved
checkpoints are described in `docs/CHECKPOINTS.md`.

## Terms

- **Real hierarchy**: the dataset's annotated coarse labels (aircraft, CUB and cars have
  2–3 levels).
- **Baseline**: fine labels only, no hierarchy. Dataset names end in `_baseline`.
- **Pseudo-hierarchy (HPH)**: coarse labels built by grouping class centroids with Hungarian
  matching. Dataset names end in `_pl`, or `_pt_<extractor>` in serial 39.
- **Cluster ratio**: the number of pseudo-coarse classes as a percentage of fine classes
  (`n_cluster_ratio`).
- **FT / FZ**: backbone fine-tuned / frozen (`freeze_backbone`).

## Backbone study (CVGIP25, ACMMMW25, ICME26, CVPRW26)

| Serial | What | Datasets | Models × LRs × other | Runs in CSV | CSV |
|---|---|---|---|---|---|
| 23 | Real hierarchy, FT | aircraft, cub, cars | 21 × 10 | 630 | `hierarchical_stage1.csv`, `hierarchical_all*.csv`, and 4 others |
| 24 | Real hierarchy, FZ | aircraft, cub, cars | 21 × 10 | 630 | `hierarchical_stage1.csv`, `hierarchical_all*.csv` |
| 25, 27 | CKA, Euclidean distance and L2 norm of features, for 23 / 24 | aircraft, cub, cars | 21 | not in a run CSV | `hierarchical_feature_metrics*.csv` (63 rows each) |
| 26, 28 | MSC, intra- and inter-class distance, for 23 / 24 | aircraft, cub, cars | 21 | not in a run CSV | `hierarchical_feature_metrics*.csv` |
| 30, 31 | MSC and distances on the second target | aircraft, cub, cars | 21 | not downloaded | – |
| 33, 34 | Reruns at poor LRs for feature visualisation | – | – | not downloaded | – |
| 35, 36 | CKA with intra/inter distances | aircraft, cub, cars | 21 | not downloaded | – |
| 37, 38 | As 35/36, second target | aircraft, cub, cars | 21 | not downloaded | – |
| 208, 209 | Output of `compute_features_norms.py` (code repo) for the best-2 and worst-2 models | – | 11 runs | not downloaded | – |

`data/stats_pretrainings.csv` holds per-backbone pretraining statistics (21 rows, no serial).

## Pseudo-hierarchy on aircraft, CUB, cars (ICME26, CVPRW26)

| Serial | What | Datasets | Models × LRs × other | Runs in CSV | CSV |
|---|---|---|---|---|---|
| 32 | Baseline, no hierarchy | aircraft, cub, cars (+ soylocal in older CSVs) | 21 × 10 | 620 (`_baseline` only) | `hierarchical_real.csv`, `hierarchical_pseudo*.csv` |
| 39 | Pseudo-hierarchy, 7 ratios, IG-1B or SigLIP-v2 extractor | `*_pt_ig1b`, `*_pt_siglipv2` (+ soylocal) | 1 × 10 × 7 ratios × 2 extractors | 1958 | `hierarchical_pseudo.csv` |
| 40 | Pseudo-hierarchy, ratios 25/50/70 | `*_pl` | 21 × 10 × 3 | 1790–1874 | `hierarchical_pseudo_2.csv`, `hierarchical_algo.csv`, `hierarchical_diff_extractor.csv` |
| 41 | Pseudo-hierarchy, CLIP on CUB, 20 extractors | cub | 20 × 10 | not downloaded | – |
| 51 | Pseudo-hierarchy, DeiT-3 on aircraft, 20 extractors | aircraft_pl | 1 × 10, ratio 50 | 192 | `hierarchical_diff_extractor.csv` |
| 52, 53 | Pseudo-hierarchy with k-means / agglomerative clustering | aircraft_pl | 1 × 10, ratio 50 | 10 each | `hierarchical_algo.csv` |
| 42–49 | Baseline vs pseudo, CLIP on CUB: batch size, epochs, image size, augmentations | cub | – | not downloaded | – |
| 62, 63 | Baseline vs pseudo, DeiT-3 on aircraft: batch size | aircraft | 10 batch sizes | not downloaded | – |
| 64, 65 | Baseline / pseudo: epochs | aircraft | 25–250 epochs | 10 / 13 | `hierarchical_epochs.csv` |
| 66, 67 | Baseline / pseudo: image size | aircraft | 112–560 px | 9 / 9 | `hierarchical_is.csv` |
| 68, 69 | Baseline / pseudo: augmentations | aircraft | 12 augmentations | 12 / 15 | `hierarchical_augs.csv` |
| 70 | Pseudo: model size | aircraft_pl | 4 sizes × 10 LRs | 38 | `hierarchical_model_size.csv` |

## SoyLocal and other leaf datasets (VCIP26)

| Serial | What | Datasets | Models × LRs × other | Runs in CSV | CSV |
|---|---|---|---|---|---|
| 54 | Pseudo-hierarchy (HPH), 7 ratios 10–90 | soylocal | 21 × 10 × 7 | 1470 | `hierarchical_soylocal.csv`, `label_smooth_cutmix.csv` |
| 55 | Baseline | soylocal | 21 × 10 | 210 | same |
| 56–61 | Reruns at each model's best LR: SoyLocal baseline (56), SoyLocal pseudo at 90% (57), aircraft real hierarchy (58), aircraft baseline (59), all three FZ with hierarchy (60), CUB FZ baseline (61) | – | 21 × 1 | not downloaded | – |
| 72, 73 | Baseline / pseudo 90%: batch size | soylocal | 8 each | 8 / 8 | `soylocal_bs.csv` |
| 74, 75 | Baseline / pseudo 90%: epochs | soylocal | 25–250 epochs | 10 / 10 | `soylocal_epochs.csv` |
| 76, 77 | Baseline / pseudo 90%: image size | soylocal | 112–560 px | 9 / 9 | `soylocal_is.csv` |
| 78, 79 | Baseline / pseudo 90%: augmentations | soylocal | 12 each | 12 / 12 | `soylocal_augs.csv` |
| 81 | Baseline, LR search, train on 80% of train, evaluate on 20% | 8 leaf datasets | 2 × 10 | 158 | `other_leaves.csv` |
| 82 | Pseudo 90%, as 81 | 7 leaf datasets | 2 × 10 | 140 | `other_leaves.csv` |
| 83 | Baseline, train on train+val, evaluate on test | cotton, soyageing, soyageing_r1, soygene, soyglobal | 2 × 10 | 99 | `other_soy_stuff.csv`, `label_smooth_cutmix.csv` |
| 84 | Pseudo 70/90, best LR, seeds 1/10/100 | 8 leaf datasets | 2 × 8 LRs | 53 | `cotton_seeds.csv` |
| 85 | Baseline, best LR, seeds 1/10/100 | 8 leaf datasets | 2 × 8 LRs | 48 | `cotton_seeds.csv` |
| 86 | Pseudo 70/90, as 83 | as 83 | 2 × 10 × 2 | 185 | `other_soy_stuff.csv`, `label_smooth_cutmix.csv` |
| 90–95 | Same-class mix and hierarchical label smoothing at 90% | cotton, soylocal | 2 × 10 | 40–60 each | `label_smooth_cutmix.csv` |

The two ViT-B models in 81–86 are DeiT-B (`hideit_base_patch16_224.fb_in1k`) and ViT-B
(`hivit_base_patch16_224.orig_in21k`). The slide deck defines 91–94 as follows:
- 91: pseudo-hierarchy with same-class mix on target 1
- 92: pseudo-hierarchy with hierarchical label smoothing
- 93: both
- 94: baseline with label smoothing

Serials 90 and 95 are in the CSV but not in the deck.

## Few-example study (after VCIP26)

| Serial | What | Datasets | Models × LRs × other | Runs in CSV | CSV |
|---|---|---|---|---|---|
| 102 | Baseline, 1/3/5 images per class | aircraft, cub, cars (`_baseline`) | 21 × 10 × 3 | 1060 | `model_low_examples.csv` |
| 103 | Real hierarchy, 1/3/5 images per class | aircraft, cub, cars | 21 × 10 × 3 | 1087 | `model_low_examples.csv` |

## Other serials on wandb

No paper uses these serials, and `data/` does not export them. Descriptions come from the run
configs in `data/wandb_runs.csv`.

| Serial | What | Datasets | Runs on wandb | Checkpoints |
|---|---|---|---|---|
| 21 | Real hierarchy, FT, at each backbone's best serial-23 setting, seed 1 | aircraft, cub, cars | 63 | 63 |
| 29 | Pseudo-hierarchy with the `amla` head, SGD 0.01 and 0.03, 5 backbones | cars | 11 | 5 (deprecated) |
| 87 | Pseudo-hierarchy at 90%, 50 epochs, DeiT and ViT-B with pseudo-hierarchies from several feature extractors (`soylocal_pseudo_<extractor>_90.yaml`) | soylocal | 130 | 0 |
| 99 | Pseudo-hierarchy at 90%, 50 epochs, DeiT and ViT-B, 6 AdamW learning rates | soylocal | 12 | 0 |
| 100 | DeiT baseline and pseudo-hierarchy at 90%, SGD 0.01, 50 epochs | soylocal | none (ed520 only) | 2 |
| 101 | Baseline, an earlier sweep of the 21 backbones | aircraft, cars, cub (`_baseline`) | 380 | 0 |
| 104 | Baseline, 3 images per class, one run | aircraft | 1 | 0 |
| 105, 106 | No hierarchy, SGD 0.01, 50 epochs, seed 42; run names end in `_Metric_n16` and similar. 106 was still logging on 2026-09-28 | aircraft, cars, cub | 62, 46 | 0 |
| 107, 108 | Frozen ViT-B with hierarchical adapters from code not committed to the code repo, 50 epochs | aircraft | 2, 11 | ed520 only |
| 110, 200, 210, 300, 310 | HiFA-head experiments (`hifa*` models), frozen (110, 210, 310) or fine-tuned (200, 300) | aircraft, cars, cub | 157, 9, 32, 58, 100 | 19 (deprecated) |
| 120, 121 | Single test runs: DeiT-3 baseline at AdamW 5e-6, and pseudo-hierarchy at 50% at AdamW 5e-5 | aircraft | 1, 1 | 0 |

Serials 52 and 53 gained 11 and 10 SoyLocal runs on 2026-09-23 (k-means and agglomerative
grouping at 70%). `data/hierarchical_algo.csv` predates them.

## Where the deck and the CSVs disagree

- **Serial 40.** The deck gives 1890 runs (3 datasets × 21 models × 3 ratios × 10 LRs); the CSVs hold 1790–1874.
- **Serial 39.** The deck says the SoyLocal runs moved from 32 and 39 to serial 54. `hierarchical_pseudo.csv` predates the move and still holds 39 and 32 SoyLocal rows.
- **Serials 102 and 103.** The deck lists aircraft only; the CSV holds aircraft, CUB and cars.
- **Serial 21.** The checkpoint store's README names serial 21 as the real-hierarchy run; no CSV contains serial 21.
