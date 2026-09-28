# Checkpoints

The paper numbers come from the wandb CSVs in `data/`, not from checkpoints. This document
records which trained models exist and whether they match those CSVs.

**Where the checkpoints are.** The only full copy is on server-3090 at
`/mnt/hdd10tb/edwin/results_hierarchical/`: 932 `.pth` files, 749 in `ckpts/` and 183 in `deprecated/`. That folder's `README.md` gives
its serial list. No Hierarchical checkpoint is on the ERISLab Hub or in `fgir_zoo`. The
FGIR-Backbones Model Zoo session is preparing two Hub repositories: one for the backbone
study, one for the pseudo-hierarchy study.

A second set comes from ed520's training folders, for runs that exist nowhere else: serials
100, 102, 107 and 108. It is deprecated and will not be released. It sits in
`/mnt/hdd10tb/edwin/deprecated/ed520_cleanup_20260928/`, split between
`hierarchical_ed520_only/results_train/` and `serial100_ckpts/`: 103 `.pth` files, 35
distinct models. Per run, the best, last and epoch-200 files hold identical weights. The
FGIR-Backbones Model Zoo session checked them:
- Serial 102 (31 runs): each file matches one wandb run by final accuracy, and all load
  strictly.
- Serial 100 (2 runs): no wandb run; both load.
- Serials 107 and 108: their weights hold `model.hier_adapters.*`, from code not committed to
  the code repo, so they fail a strict load.

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
  build. The 41 `deprecated/from_tgda_20260922` files are plain timm models and load into stock timm 0.9.12
  after dropping the `model.` prefix and renaming the head.
- **wandb match** (`wandb_match.csv`): each file matched to its run in `nycu_pcs/Hierarchical`.
  819 files match a run on configuration and final accuracy, 11 on name and serial only, and
  102 have no run.

| Folder under `results_hierarchical/` | Serial in file | Files | Match a CSV row | Accuracy equal | Notes |
|---|---|---|---|---|---|
| `ckpts/real_hierarchy_ckpts` | 21 | 63 | 0 | – | No CSV holds serial 21. It reruns serial 23 at each backbone's best setting (seed 1): on average 0.13 points below the serial-23 maximum. |
| `ckpts/baseline_ckpts` | 32 | 63 | 61 | 24 | 37 differ from the CSV row, by up to 54.4 points (CLIP on cars: 27.47 against 81.88). The serial-32 CSV rows mix two run generations. |
| `ckpts/pseudo_hierarchy_ckpts` | 40 | 184 | 182 | 178 | Largest difference 0.41. |
| `ckpts/serial54_ckpts` | 39 | 140 | 140 | 140 | The last runs (AdamW 5e-6) of SoyLocal HPH serial 54, renamed from `serial39_soylocal_ckpts`; the config inside still says 39. Includes the VCIP26 headline DeiT at ratio 90. |
| `ckpts/serial55_ckpts` | 32 | 21 | 21 | 21 | The last runs of the SoyLocal baseline serial 55, renamed from `baseline_ckpts`; the config inside still says 32. |
| `ckpts/serial56_ckpts` … `serial61_ckpts` | 56–61 | 151 | 0 | – | Best-learning-rate reruns; no CSV holds these serials. Per the Model Zoo session's wandb match, serial 60 reruns serial 24 (63 files). |
| `ckpts/serial81_ckpts` … `serial86_ckpts` | 81–86 | 85 | 83 | 83 | Exact. |
| `ckpts/serial102_ckpts`, `serial103_ckpts` | 102, 103 | 42 | 42 | 42 | Exact. |
| `deprecated/*` | 29, 32–34, 39–41, 51, 57, 70–73, 84, 85, 91–94, 110–310 | 183 | 115 | 100 | See the next section. The 41 TGDA files load only into stock timm. |

## Deprecated checkpoints

`results_hierarchical/deprecated/` holds the files retired so far. Nothing has been deleted.
The folder's `README.md` gives the reason for each group and how to restore it. File renames
are listed in `results_hierarchical/RENAMES_20260929.tsv`. The release keeps serials 21, 58,
59 and 60 (backbone study) and 56, 57, 84 and 85 (pseudo-hierarchy study): 236 files. Other
files in `ckpts/` may still move to `deprecated/`; that release is decided in the
FGIR-Backbones Model Zoo session.

| Folder under `deprecated/` | Files | Why |
|---|---|---|
| `hifa_head` | 29 | HiFA-head experiments, unused. 10 of them (`*_fz_110`) do not load: their weights hold `model.head.add_pre_classifiers.3.*`, which the stored config does not build. |
| `pseudo_hierarchy_serial39` | 45 | First pseudo-hierarchy runs (ICME26's BPH section), replaced by serial 40. 43 match no wandb run; 2 match on name and serial only. |
| `from_tgda_20260922` | 41 | The older serial-32 run generation, plain timm models. Their wandb runs no longer exist, and `ckpts/baseline_ckpts` holds the same configurations. |
| `serial57_ratio70_no_run` | 2 | SoyLocal DeiT-3 at ratio 70 with no wandb run; serial 57 is documented as ratio 90. |
| `serial39_siglip_no_run` | 7 | SoyLocal SigLIP-v2 pseudo-hierarchy files at ratios 10–90 with no wandb run; their accuracy matches no serial-54 run. |
| `serial84_85_no_run` | 8 | Serial 84 (5) and 85 (3) files whose seed and accuracy match no wandb run. |
| `feature_visualization_serial34`, `serial29_ckpts`, `serial33_ckpts` | 26 | Feature-visualisation and early runs. |
| `other_hyperparameters`, `pseudo_other_extractor` | 9 | Hyperparameter and extractor studies with negative or duplicate results. |
| `serial91_ckpts` … `serial94_ckpts` | 16 | Same-class mix and hierarchical label smoothing. |

## Files with no wandb run

The Model Zoo session's match (`results_all/checkpoints/wandb_match.csv`, written before the
moves above) finds no wandb run for 102 files:
- 41 in `from_tgda_20260922` and 43 in `pseudo_hierarchy_serial39`: older runs whose wandb
  entries were replaced; both folders are deprecated.
- 7 SigLIP-v2 SoyLocal files: older runs of a serial-54 configuration, deprecated above.
- 2 in serial 57 and 6 in serials 84/85, deprecated above.
- 3 in other `deprecated/` folders.

Another 11 files match a run on name and serial only. They are 6 CUB ResNets at SGD 0.1 in
`baseline_ckpts`, 1 in `pseudo_hierarchy_ckpts`, 2 in `pseudo_hierarchy_serial39`, and 1 each
in serials 84 and 85; those last two are deprecated above.

## Coverage of the papers

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
