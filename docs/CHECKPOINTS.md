# Checkpoints

The paper numbers come from the wandb CSVs in `data/`, not from checkpoints. This document
records which trained models exist and whether they match those CSVs.

**Where the checkpoints are.** The only full copy is on server-3090 at
`/mnt/hdd10tb/edwin/results_hierarchical/`: 932 `.pth` files. That folder's `README.md` gives
its serial list. No Hierarchical checkpoint is on the ERISLab Hub or in `fgir_zoo`. The
FGIR-Backbones Model Zoo session is preparing two Hub repositories: one for the backbone
study, one for the pseudo-hierarchy study.

**What a file holds.** Each file holds `{model, config, epoch, accuracy}`:
- `config` is the training argparse namespace;
- `accuracy` is the final-epoch top-1 on the evaluation split;
- each file is one run.

A learning-rate sweep kept only its last run, AdamW 5e-6. A max-over-learning-rate number
therefore has a checkpoint only when its best run was that last one.

**The check.** `python -m codebase.verify.verify_checkpoints` (fgir_timm912 env) writes
`results_all/checkpoints/checkpoint_inventory.csv`, one row per file, and
`checkpoint_summary.csv`, one row per folder.
- **Matching.** It matches each file to its CSV row on serial, dataset, model, feature
  extractor, cluster ratio, optimiser, learning rate, seed and training-set size.
- **Loading.** It strict-loads two files per folder into the code repo's `TIMM` model,
  filling options missing from old configs with the code repo's defaults.

The FGIR-Backbones Model Zoo session ran two further checks. Frozen copies of its outputs
are in `results_all/checkpoints/`.
- **Load test** (`loadtest_all.csv`): all 932 files, strict. 922 load. The 10 failures are
  `deprecated/hifa_head/*_fz_110`, whose weights hold head layers their stored config does not
  build. The 41 `from_tgda_20260922` files are plain timm models and load into stock timm 0.9.12
  after dropping the `model.` prefix and renaming the head.
- **wandb match** (`wandb_match.csv`): each file matched to its run in `nycu_pcs/Hierarchical`.
  819 files match a run on configuration and final accuracy, 11 on name and serial only, and
  102 have no run.

| Folder under `results_hierarchical/` | Serial in file | Files | Match a CSV row | Accuracy equal | Notes |
|---|---|---|---|---|---|
| `ckpts/real_hierarchy_ckpts` | 21 | 63 | 0 | – | No CSV holds serial 21. It reruns serial 23 at each backbone's best setting (seed 1): on average 0.13 points below the serial-23 maximum. |
| `ckpts/baseline_ckpts` | 32 | 84 | 82 | 45 | 37 differ from the CSV row, by up to 54.4 points (CLIP on cars: 27.47 against 81.88). The serial-32 CSV rows mix two run generations. The 21 SoyLocal files are the serial-55 SoyLocal baseline, and all 21 equal their serial-55 rows. |
| `ckpts/pseudo_hierarchy_ckpts` | 40 | 184 | 182 | 178 | Largest difference 0.41. |
| `ckpts/serial39_soylocal_ckpts` | 39 | 147 | 147 | 142 | These are the last runs (AdamW 5e-6) of SoyLocal serial 54; 140 match the serial-54 rows exactly. Includes the VCIP26 headline DeiT, ratio 90. Five SigLIP-v2 files match none of their two CSV rows. |
| `ckpts/serial56_ckpts` … `serial61_ckpts` | 56–61 | 153 | 0 | – | Best-learning-rate reruns; no CSV holds these serials. Per the Model Zoo session's wandb match, serial 60 reruns serial 24 (63 files). |
| `ckpts/serial81_ckpts` … `serial86_ckpts` | 81–86 | 93 | 85 | 83 | 81, 82, 83 and 86 match exactly; 84 and 85 differ by at most 0.42. |
| `ckpts/serial102_ckpts`, `serial103_ckpts` | 102, 103 | 42 | 42 | 42 | Exact. |
| `deprecated/*` | 29, 33, 34, 39–41, 51, 70–73, 91–94, 110–310 | 125 | 65 | 58 | Retired experiments. |
| `from_tgda_20260922/*` | 32 | 41 | 41 | 40 | Plain timm models from TGDA; the code repo's `TIMM` cannot build them. |

No checkpoints exist anywhere for the serial 23 and 24 sweeps (the backbone results of
CVGIP25, ACMMMW25 and ICME26). Their best settings were rerun as serials 21 and 60.

Among the VCIP26 headline numbers:
- DeiT with HPH on SoyLocal (64.33) has an exact checkpoint.
- Its baseline (51.83), the ViT-B SoyLocal maximum and the Cotton and SoyAgeing-R1 maxima have
  none. Those maxima were reached at learning rates other than the saved last one.

The non-timm pretrained weights are:
- SwAV and MoCo v3 ResNet-50: cached in `/mnt/hdd10tb/edwin/torch/hub/checkpoints/`.
- SparK and SupCon ResNet-50: on the HuggingFace repos `RexGusto/resnet50_spark` and
  `RexGusto/resnet50_supcon`, not in the local HF cache.
